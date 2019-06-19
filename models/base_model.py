import tensorflow as tf
import abc
import model_helper as helper
from dataset import iterator_wrapper
# from tensorlib import layers
from tensorflow.contrib import slim
from . import rpn_model
from . import rcnn_model
from utils import misc_util as misc
from utils import proposal_util, anchor_util
import collections

# Put anything that you want to print into 'debug_pool', then it will be printed when session runs
debug_pool = dict()


def inject_to_debug_pool(debug=True, **kwargs):
    if not debug:
        return
    global debug_pool
    debug_pool.update(kwargs)


class DetectResult(collections.namedtuple("DetectResult",
                                          ("images", "bboxes",
                                           "categories", "im_info",
                                           "scores"))):
    pass


class TrainOutputTuple(collections.namedtuple(
    "TrainOutputTuple", ("loss", "global_step",
                         "summary", "batch_size",
                         "grad_norm", "learning_rate"))):
    pass


class EvalOutputTuple(collections.namedtuple("EvalOutputTuple",
                                             ("loss", "accuracy",
                                              "batch_size", "summary"))):
    """To allow for flexibily in returning different outputs."""
    pass


class InferOutputTuple(collections.namedtuple(
    "InferOutputTuple", ("images", "bboxes", "categories",
                         "scores", "summary"))):
    """To allow for flexibility in returning different outputs."""
    pass


class BaseModel(object):
    @abc.abstractmethod
    def _image_to_head(self, inputs, reuse=None):
        """
        SubClass must implement this method
        According to different kind of CNN, this method will get different conv-k feature
        :param inputs
        :param reuse:
        :return: conv-k feature
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _head_to_tail(self, inputs, reuse=None):
        """
        SubClass must implement this method
        According to different kind of CNN, this method will have different output
        :param inputs:
        :param reuse:
        :return:
        """
        raise NotImplementedError

    def __init__(self, hparams, reverse_cate_table, data_wrapper=None, scope=None):
        self._set_params(hparams, data_wrapper, reverse_cate_table, scope)
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                             slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                            weights_regularizer=self.weights_regularizer,
                            biases_regularizer=self.biases_regularizer,
                            biases_initializer=tf.constant_initializer(0.0)):
            losses, info = self._build_graph()
        self._deploy_exe_info(losses, info)
        # Saver
        with tf.device("/cpu:0"):
            self.saver = tf.train.Saver(
                tf.global_variables(), hparams.ckpt_storage)

    def _setup_gloabal_step(self):
        """Sets up the global step Tensor."""
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP,
                         tf.GraphKeys.GLOBAL_VARIABLES])

    def _set_params(self, hparams, data_wrapper, reverse_cate_table, scope=None):
        # Store tensors that will be feed in histogram summary
        self.histogram = dict()
        # Store summaries
        self.summaries = list()
        # Store the specific layer's activation for visual utility
        self.activations = list()
        self.restore_op = None
        self.reverse_cate_table = reverse_cate_table
        self.scope = scope
        self.feat_stride = [16]
        self.hparams = hparams
        self.ori_count = len(hparams.anchor_ratios) * len(hparams.anchor_scales)
        self.trainable = hparams.mode is "train"
        self.tunable = hparams.tunable
        self.predicable = hparams.mode is "infer"
        # Set data
        if not self.predicable:
            assert isinstance(data_wrapper, iterator_wrapper.DataWrapper)
            # !!!Make sure dataset batch size is 1
            self.im_info = data_wrapper.images_size
            self.images_data = data_wrapper.images_data
            self.gt_bboxes = data_wrapper.bbox_locations
            self.categories_id = data_wrapper.categories_id
        else:
            self.im_info = tf.placeholder(shape=[3], dtype=tf.int32, name="size_feed")
            self.images_data = tf.placeholder(
                shape=[1, None, None, 3], dtype=tf.float32, name="image_feed")
            self.gt_bboxes = None
            self.categories_id = None
        debug_pool.update(images_info=self.im_info)
        # Initializer
        self.initializer = helper.get_initializer(
            hparams.init_op, hparams.ran_seed, hparams.init_weight)
        self.bbox_initializer = helper.get_initializer(
            hparams.bbox_init_op, hparams.bbox_ran_seed, hparams.bbox_init_weight)
        # Regularization
        # weights_decay = hparams.weight_decay_factor
        # if hparams.bias_decay:
        #     biases_decay = weights_decay
        # else:
        #     biases_decay = None
        # layers.fill_arg_scope(weights_decay=weights_decay, biases_decay=biases_decay)
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(hparams.weight_decay_factor)
        if hparams.bias_decay:
            self.biases_regularizer = self.weights_regularizer
        else:
            self.biases_regularizer = None
        # tf.get_variable_scope().set_initializer(self.initializer)
        # Set up global step
        self._setup_gloabal_step()

    def _deploy_exe_info(self, losses, info):
        with tf.name_scope("deploy_exe_info"):
            hp = self.hparams
            if self.trainable:  # Train
                self.train_loss = losses
                params = tf.trainable_variables()
                if hp.tunable:
                    learning_rate = hp.tune_rate
                else:
                    learning_rate = hp.learning_rate
                self.learning_rate = tf.constant(learning_rate, dtype=tf.float32)
                # Warm-up
                self.learning_rate = self._get_learning_rate_warmup()
                # Decay
                self.learning_rate = self._get_learning_rate_decay()
                # Optimizer
                opt = tf.train.MomentumOptimizer(self.learning_rate, hp.momentum_factor)
                # Gradient
                gradients = tf.gradients(self.train_loss, params)
                # Gradient clip
                clipped_grads, grad_norm_summaries, grad_norm = helper.gradient_clip(
                    gradients, max_gradient_norm=hp.max_grad_norm)
                # Gradient norm
                for summary in grad_norm_summaries:
                    self._add_to_summaries(summary)
                self.grad_norm = grad_norm
                # Apply update to params
                self.update = opt.apply_gradients(
                    zip(clipped_grads, params), global_step=self.global_step)
                # Trainable params summary
                print("# Trainable variables")
                print("Format: <name>, <shape>, <(soft) device placement>")
                for param in params:
                    self.histogram.update({param.name: param})
                    print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                            param.op.device))
                self.categories = self.reverse_cate_table.lookup(self.categories_id)[0]
                self.bbox_scores = info["roi_scores"]
                self.train_summary = self._config_train_summary()
            elif self.predicable:  # Infer
                # stddevs = tf.tile(tf.constant(hp.bbox_norm_stddevs), multiples=[hp.num_class])
                # means = tf.tile(tf.constant(hp.bbox_norm_means), multiples=[hp.num_class])
                deltas = info["roi_deltas"]
                # Restore bbox predicts

                # deltas = tf.add(tf.multiply(deltas, stddevs), means)
                # info["bbox_predicts"] = deltas
                rois = info["rois"]
                self.bbox_scores = info["roi_scores"]
                if hp.forward_rcnn:
                    self.categories = self.reverse_cate_table.lookup(
                        tf.to_int64(info["class_predicts"]))
                else:
                    self.categories = info["class_predicts"]
                # Get ground-truth bounding boxes
                self.pgt_bboxes = anchor_util.get_coco_anchors(
                    proposal_util.bboxes_regression(rois[:, 1:], deltas))
                debug_pool.update(rois=rois, gt_bboxes=self.pgt_bboxes)
                # Get predicted ground-truth bbox
                self.infer_summary = self._config_infer_summary()
            else:  # Eval
                rois = info["rois"]
                deltas = info["roi_deltas"]
                self.eval_loss = losses
                # Get predicted ground-truth bounding boxes
                self.pgt_bboxes = anchor_util.get_coco_anchors(
                    proposal_util.bboxes_regression(rois[:, 1:], deltas))
                self.accuracy = misc.mean_avg_overlap(
                    self.pgt_bboxes, self.gt_bboxes[:, 0:4])
                self.eval_summary = self._config_eval_summary()

    def _build_graph(self):
        rcnn_info = None
        # Get conv-k feature map
        conv_feat = self._image_to_head(self.images_data)
        # RPN
        with slim.arg_scope(rpn_model.rpn_arg_scope(weights_initializer=self.initializer)):
            rois, rpn_info, rpn_activation = rpn_model.rpn_base(
                conv_feat, self.hparams, self.im_info, self.gt_bboxes,
                feat_stride=self.feat_stride[0], anchor_count=self.ori_count,
                trainable=self.trainable, predictable=self.predicable)
            self.activations.append(rpn_activation)
            self.histogram.update(rpn_dict=rpn_info)
        if self.hparams.forward_rcnn:
            # Fast-RCNN
            rcnn_info, rcnn_activation = rcnn_model.rcnn_base(
                conv_feat, self.hparams, rois,
                roi_scores=rpn_info["bbox_scores"],
                bbox_labels=self.gt_bboxes,
                anchor_labels=rpn_info["bbox_labels"],
                trainable=self.trainable,
                predictable=self.predicable,
                roi_pool_layer=self._roi_pool_layer,
                head_to_tail=self._head_to_tail,
                cls_weights_initializer=self.initializer,
                reg_weights_initializer=self.bbox_initializer)
            self.activations.append(rcnn_activation)
            self.histogram.update(rcnn_dict=rpn_info)
        # Calculate loss
        losses = self._loss_func(rpn_info, rcnn_info)
        info = rcnn_info if rcnn_info else rpn_info
        return losses, info

    def _roi_pool_layer(self, inputs, rois):
        hp = self.hparams
        pooling_mode = hp.pooling_mode
        with tf.variable_scope(pooling_mode+"_pool_layer"):
            if pooling_mode == "pyramid":
                outputs = helper.spatial_pyramid_pooling(hp.bin_size_list,
                                                         inputs, rois,
                                                         self.feat_stride[0])
            elif pooling_mode == "crop":
                outputs = helper.crop_resize_pooling(inputs, rois,
                                                     self.feat_stride[0],
                                                     hp.unify_size,
                                                     max_pool=hp.max_pool,
                                                     flatten=hp.flatten)
            else:
                raise NotImplementedError("The pooling mode you specified '%s' is not implemented yet" % pooling_mode)
        return outputs

    @staticmethod
    def _smooth_l1_loss(pre_deltas, tgt_deltas, in_weights,
                        out_weights, sigma=3., axis=None):
        sigma = sigma ** 2
        differs = tf.subtract(pre_deltas, tgt_deltas)
        posi_differs = tf.abs(tf.multiply(differs, in_weights))
        smooth_signal = tf.stop_gradient(tf.to_float(tf.less(posi_differs, tf.divide(1., sigma))))
        res = tf.multiply(tf.multiply(tf.pow(posi_differs, 2), tf.multiply(0.5, sigma)), smooth_signal)
        ults = tf.multiply(tf.subtract(posi_differs, tf.multiply(0.5, sigma)), tf.subtract(1., smooth_signal))
        results = tf.add(res, ults)
        loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(results, out_weights), axis=axis))
        return loss

    def _loss_func(self, rpn_info=None, rcnn_info=None):
        if self.predicable:
            return tf.constant(0.0)
        with tf.name_scope("cal_loss"):
            with tf.device("/cpu:0"):
                rpn_loss = rpn_model.rpn_loss(rpn_info, self._smooth_l1_loss)
                rcnn_loss = rcnn_model.rcnn_loss(rcnn_info, self._smooth_l1_loss)
                losses = tf.add(rpn_loss, rcnn_loss)
                losses = tf.add(losses, tf.add_n(tf.losses.get_regularization_losses()))
        return losses

# ------------------------------------------->gather message & executing------------------->

    def _add_to_summaries(self, summary):
        self.summaries.append(summary)

    def _config_train_summary(self):
        with tf.name_scope("infer_summary"):
            with tf.device("/cpu:0"):
                # Visualize activations
                for item in self.activations:
                    self._add_to_summaries(tf.summary.histogram(
                        'TRAIN/activation/' + item.op.name, item))
                    self._add_to_summaries(tf.summary.scalar(
                        'TRAIN/zero_fraction/' + item.op.name,
                        tf.nn.zero_fraction(item)))
                # Histogram summaries
                for key, item in self.histogram.items():
                    if item is not None and type(item) == dict:
                        for k, v in item.items():
                            self._add_to_summaries(tf.summary.histogram(
                                'TRAIN/' + key + '/' + k, v))
                    else:
                        self._add_to_summaries(tf.summary.histogram('TRAIN/' + key, item))
                # Detected image with ground-truth box
                self._add_to_summaries(tf.summary.image(
                    'INFER/image', self.images_data))
                # loss & lr
                self._add_to_summaries(tf.summary.scalar(
                    'TRAIN/train_loss', self.train_loss))
                self._add_to_summaries(tf.summary.scalar(
                    'TRAIN/learning_rate', self.learning_rate))
        return tf.summary.merge(self.summaries)

    def _config_eval_summary(self):
        with tf.name_scope("eval_summary"):
            with tf.device("/cpu:0"):
                self._add_to_summaries(tf.summary.scalar(
                    'EVAL/eval_loss', self.eval_loss))
                self._add_to_summaries(tf.summary.scalar(
                    'EVAL/accuracy', self.accuracy))
        return tf.summary.merge(self.summaries)

    def _config_infer_summary(self):
        with tf.name_scope("infer_summary"):
            with tf.device("/cpu:0"):
                self._add_to_summaries(tf.summary.image(
                    'INFER/image', self.images_data))
        return tf.summary.merge(self.summaries)

    def train(self, sess):
        output_tuple = TrainOutputTuple(loss=self.train_loss,
                                        global_step=self.global_step,
                                        batch_size=tf.shape(self.images_data)[0],
                                        summary=self.train_summary,
                                        learning_rate=self.learning_rate,
                                        grad_norm=self.grad_norm)
        detect_result = DetectResult(images=self.images_data[0],
                                     bboxes=anchor_util.get_anchors_info(self.gt_bboxes),
                                     categories=self.categories,
                                     im_info=self.im_info,
                                     scores=self.bbox_scores)
        if debug_pool:
            print("DEBUG--->", sess.run(debug_pool))
        return sess.run([self.update, output_tuple, detect_result])

    def eval(self, sess):
        output_tuple = EvalOutputTuple(loss=self.eval_loss,
                                       batch_size=tf.shape(self.images_data)[0],
                                       summary=self.eval_summary,
                                       accuracy=self.accuracy)
        if debug_pool:
            print("DEBUG--->", sess.run(debug_pool))
        return sess.run(output_tuple)

    def infer(self, sess, image_feed, size_feed):
        # bboxes: [ctr_x, ctr_y, width, height]
        output_tuple = InferOutputTuple(bboxes=self.pgt_bboxes,
                                        images=self.images_data[0],
                                        categories=self.categories,
                                        scores=self.bbox_scores,
                                        summary=self.infer_summary)
        # gt_bboxes = proposal_util.clip_bboxes(anchor_util.get_coco_anchors(debug_pool["anchors"]), self.im_info)
        detect_result = DetectResult(images=self.images_data[0],
                                     bboxes=self.pgt_bboxes,
                                     categories=self.categories,
                                     im_info=self.im_info,
                                     scores=self.bbox_scores)
        if debug_pool:
            print("DEBUG--->", sess.run(debug_pool, feed_dict={
                'image_feed:0': image_feed, 'size_feed:0': size_feed}))
        return sess.run([output_tuple, detect_result], feed_dict={
            'image_feed:0': image_feed, 'size_feed:0': size_feed})
# ------------------------------------->Learning rate---------------------------->

    def _get_learning_rate_warmup(self):
        """Get learninng rate warmup..
        Returns:
            learning_rate
        Raises:
            ValueError of unknown value
        """
        warmup_steps = self.hparams.warmup_steps
        warmup_scheme = self.hparams.warmup_scheme
        print("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
              (self.hparams.learning_rate, warmup_steps, warmup_scheme))
        # Decaying learning_rate until global_step >= warmup_steps
        # When global_step < warmup_steps
        # learning_rate *= warmup_factor ** (warmup_steps - global_step)
        if warmup_scheme == "t2t":
            warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
            inv_decay = warmup_factor ** tf.to_float((warmup_steps - self.global_step))
        else:
            raise ValueError("unknown warmup scheme %s" % warmup_scheme)

        return tf.cond(
            self.global_step < self.hparams.warmup_steps,
            lambda: inv_decay * self.learning_rate,
            lambda: self.learning_rate,
            name="learning_rate_warmup_cond")

    def _get_decay_info(self):
        """Return decay info based on decay_scheme."""
        decay_scheme = self.hparams.decay_scheme
        num_train_steps = self.hparams.num_train_steps
        if decay_scheme in ["luong5", "luong10", "luong234"]:
            decay_factor = 0.5
            if decay_scheme == "luong5":
                start_decay_step = int(num_train_steps / 2)
                decay_times = 5
            elif decay_scheme == "luong10":
                start_decay_step = int(num_train_steps / 2)
                decay_times = 10
            else:
                start_decay_step = int(num_train_steps * 2 / 3)
                decay_times = 4
            remain_steps = num_train_steps - start_decay_step
            decay_steps = int(remain_steps / decay_times)  # decay per decay steps after start_decay_step
        elif not decay_scheme:  # no decay
            start_decay_step = num_train_steps
            decay_steps = 0
            decay_factor = 1.0
        else:
            raise ValueError("Unknown decay scheme %s" % decay_scheme)
        return start_decay_step, decay_steps, decay_factor

    def _get_learning_rate_decay(self):
        """Get learning rate decay.
        Returns:
            learning_rate
        Raises:
        """
        start_decay_step, decay_steps, decay_factor = self._get_decay_info()
        print("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
              "decay_factor %g" % (self.hparams.decay_scheme, start_decay_step,
                                   decay_steps, decay_factor))
        # Decay per decay steps after start_decay_step[with discrete interval]
        return tf.cond(
            tf.less(self.global_step, start_decay_step),
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                tf.subtract(self.global_step, start_decay_step),
                decay_steps,
                decay_factor,
                staircase=True,
                name="learning_rate_decay_cond"))

    # ------------------------------------->Learning rate---------------------------->
