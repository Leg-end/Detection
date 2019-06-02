import tensorflow as tf
import abc
from tensorflow.contrib import layers
import hparams
from utils import proposal_util
import model_helper as helper


class BaseModel(object):
    @abc.abstractmethod
    def _image_to_head(self, reuse=None):
        """
        SubClass must implement this method
        According to different kind of CNN, this method will get different conv-k feature
        :param reuse:
        :return: conv-k feature
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _head_to_tail(self, inputs):
        """
        SubClass must implement this method
        According to different kind of CNN, this method will have different output
        :param inputs:
        :return:
        """
        raise NotImplementedError

    def __init__(self, hparams):
        self._set_params(hparams)
        loss = self._build_graph()
        self._deploy_exe_info(loss)

    def _set_params(self, hparams):
        self.hparams = hparams
        self.trainable = hparams.mode is "train"
        self.initializer = helper.get_initializer(
            hparams.init_op, hparams.ran_seed, hparams.init_weight)
        self.bbox_initializer = helper.get_initializer(
            hparams.bbox_init_op, hparams.bbox_ran_seed, hparams.bbox_init_weight)
        tf.get_variable_scope().set_initializer(self.initializer)
        pass

    def _deploy_exe_info(self, info):
        pass

    def _softmax_layer(self, inputs, name="softmax", keep_dim=False):
        if keep_dim:
            shape = tf.shape(inputs)
            reshaped = tf.reshape(inputs, shape=[-1, shape[-1]])
            outputs = tf.nn.softmax(reshaped, name=name)
            return tf.reshape(outputs, shape=shape)
        return tf.nn.softmax(inputs, name=name)
# ------------------------------------>RPN--------------------------------------> #

    def _rpn_reg_layer(self, inputs, num_outputs, name="rpn_reg_layer"):
        reg_conv = layers.conv2d(inputs, num_outputs, [1, 1], trainable=self.trainable,
                                 weights_initializer=self.initializer, padding='valid',
                                 activation_fn=None, scope=name)
        return reg_conv

    def _rpn_cls_layer(self, inputs, num_outputs, name="rpn_cls_layer"):
        cls_conv = layers.conv2d(inputs, num_outputs, [1, 1], trainable=self.trainable,
                                 weights_initializer=self.initializer, padding='valid',
                                 activation_fn=None, scope=name)
        # Fix channel as 2
        cls_conv = helper.reshape_for_forward_pairs(cls_conv, 2, name="cls_conv_reshape")
        cls_scores = self._softmax_layer(cls_conv, keep_dim=True, name="rpn_cls_prob_softmax")
        # ?
        cls_preds = tf.argmax(tf.reshape(cls_conv, shape=[-1, 2]), axis=1, name="cls_pred_argmax")
        # Switch to original channel, after softmax, here we get prob for positive and negative
        # Form as [n, n,....,n (count=num_outputs/2), p, p, ......,p (count=num_outputs/2)]
        cls_scores = helper.reshape_for_forward_pairs(cls_scores, num_outputs, name="cls_prob_reshape")
        return cls_preds, cls_scores

    def _rpn(self, inputs):
        """
        A Region Proposal Network (RPN) takes an image (of any size) as input and outputs a set of
        rectangular object proposals, each with an objectness score[quoted from Faster-RCNN]
        :param inputs: image
        :return: rect object proposals, scores
        """
        hp = self.hparams
        with tf.variable_scope("rpn"):
            # Build the anchors for image
            anchors, count = helper.generate_img_anchors(hp.im_info, hp.feat_stride)
            rpn_conv = layers.conv2d(inputs, hp.rpn_channel, [3, 3], trainable=self.trainable,
                                     weights_initializer=self.initializer, scope="rpn_conv/3x3")
            cls_preds, cls_scores = self._rpn_cls_layer(rpn_conv, count*2)
            bbox_deltas = self._rpn_reg_layer(rpn_conv, count*4)
            # Generate roi
            if self.trainable:
                rois, scores = helper.sample_rois_from_anchors(cls_scores, bbox_deltas,
                                                               hp.im_info, anchors, count)
                # todo label rois, target anchors, target class
            else:
                rois, _ = helper.sample_rois_from_anchors(cls_scores, bbox_deltas, hp.im_info,
                                                          anchors, count, strategy=hp.infer_mode)
        return rois

# ------------------------------------>RPN-------------------------------------->
    def _roi_pool_layer(self, conv_feat, rois):
        # todo adjust outputs dimension
        hp = self.hparams
        pooling_mode = hp.pooling_mode
        with tf.variable_scope(pooling_mode+"_pool_layer"):
            if pooling_mode == "pyramid":
                outputs = helper.spatial_pyramid_pooling(hp.bin_size_list,
                                                         conv_feat, rois,
                                                         hp.feat_stride)
            elif pooling_mode == "crop":
                outputs = helper.crop_resize_pooling(conv_feat, rois,
                                                     hp.feat_stride,
                                                     hp.unify_size)
            else:
                raise NotImplementedError("The pooling mode you specified '%s' is not implement yet" % pooling_mode)
        return outputs

    def _frcnn_cls_layer(self, inputs, num_outputs, name="frcnn_cls_layer"):
        with tf.variable_scope(name):
            scores = layers.fully_connected(inputs, num_outputs,
                                            activation_fn=None,
                                            trainable=self.trainable,
                                            weights_initializer=self.initializer)
            predicts = self._softmax_layer(scores, name="frcnn_cls_prob_softmax")
        return predicts, scores

    def _frcnn_reg_layer(self, inputs, num_outputs, name="frcnn_reg_layer"):
        with tf.variable_scope(name):
            logits = layers.fully_connected(inputs, num_outputs,
                                            activation_fn=None,
                                            trainable=self.trainable,
                                            weights_initializer=self.bbox_initializer)
        return logits

    def _fast_rcnn(self, inputs, rois):
        hp = self.hparams
        with tf.variable_scope("fast_rcnn"):
            pool = self._roi_pool_layer(inputs, rois)
            fc = self._head_to_tail(pool)
            cls_preds, cls_scores = self._frcnn_cls_layer(fc, hp.cate_num)
            bbox_deltas = self._frcnn_reg_layer(fc, 4*hp.cate_num)
        return cls_scores, bbox_deltas

    def _smooth_l1_loss(self, predicts, targets):
        # todo
        with tf.name_scope("smooth_l1_loss"):
            differs = tf.abs(tf.subtract(predicts, targets))
            results = tf.cond(tf.greater(1, differs),
                              lambda: tf.multiply(0.5, tf.pow(differs, 2)),
                              lambda: tf.subtract(differs, 0.5))
            results = tf.reduce_mean(tf.reduce_sum(results, axis=1))
        return results

    def _loss_func(self, rois, final_cls_scores, final_cls_preds, final_bbox_deltas, targets):
        predicts = proposal_util.bboxes_regression(rois[:, 1:], final_bbox_deltas)
        bbox_losses = self._smooth_l1_loss(predicts, targets)
        cls_losses = tf.nn.softmax_cross_entropy_with_logits()
        # todo

    def _build_graph(self, scope=None):
        # Get conv-k feature map
        conv_feat = self._image_to_head()
        # Get ROI
        rois = self._rpn(conv_feat)
        # Region classification and regression by fast-RCNN
        final_cls_scores, final_bbox_deltas = self._fast_rcnn(conv_feat, rois)
        # Calculate loss
        loss = self._loss_func(final_cls_scores, final_bbox_deltas)
        return loss

# ------------------------------------------->gather message & executing------------------->

    def _get_train_summary(self):
        pass

    def _get_eval_summary(self):
        pass

    def _get_infer_summary(self):
        pass

    def train(self, sess):
        pass

    def eval(self, sess):
        pass

    def infer(self, sess):
        pass
