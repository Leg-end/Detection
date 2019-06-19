import tensorflow as tf
from tensorflow.contrib import slim
import model_helper as helper
from utils import misc_util as misc


def rcnn_arg_scope(trainable=True,
                   batch_norm_decay=0.997,
                   batch_norm_epsilon=1e-5,
                   batch_norm_scale=True,
                   activation_fn=tf.nn.relu,
                   weights_initializer=slim.variance_scaling_initializer()):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=weights_initializer,
            trainable=trainable,
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def rcnn_base(inputs,
              hp,
              rois,
              roi_scores,
              bbox_labels,
              roi_pool_layer,
              head_to_tail,
              trainable=True,
              predictable=False,
              anchor_labels=None,
              cls_weights_initializer=None,
              reg_weights_initializer=None):
    """
    A Region Proposal Network (RPN) takes an image (of any size) as input and outputs a set of
    rectangular object proposals, each with an objectness score[quoted from Faster-RCNN]
    :param inputs: image
    :param hp: hyper parameters
    :param rois: regions of interest
    :param roi_scores: scores of all rois
    :param roi_pool_layer: rois' pooling function(layer)
    :param head_to_tail: fully connect network
    :param trainable: whether to train this network
    :param predictable: is in infer mode
    :param bbox_labels: target bounding box
    :param anchor_labels: anchor labels when using rpn to generate roi
    :param cls_weights_initializer: weights initializer for classification layer
    :param reg_weights_initializer: weights initializer for regression layer
    :return: rect object proposals, scores, print_pool(dict) for debugging, activation(dict) for visualization
    """
    with tf.variable_scope("rcnn"):
        with tf.device(helper.get_device_str(device_id=0, num_gpus=hp.num_gpus)):
            # Fill rcnn's bbox_label, class_label, in_weights, out_weights, rois
            rcnn_info = dict()
            if not predictable:
                rcnn_info, rois, _ = helper.pack_proposal_info(
                    anchor_labels, rois, bbox_scores=roi_scores,
                    bbox_targets=bbox_labels,
                    num_class=hp.num_class)
            pool = roi_pool_layer(inputs, rois)
            fc = head_to_tail(pool)
            probs, predicts, scores = _rcnn_cls_layer(fc, hp.num_class,
                                                      trainable=trainable,
                                                      weights_initializer=cls_weights_initializer)
            deltas = _rcnn_reg_layer(fc, 4 * hp.num_class,
                                     trainable=trainable,
                                     weights_initializer=reg_weights_initializer)
            # Pack rcnn info into dict for calculating loss, only for training
            misc.append_params(rcnn_info,
                               class_scores=scores, class_predicts=predicts,
                               class_probs=probs, bbox_predicts=deltas)
    return rcnn_info, pool


def rcnn_loss(rcnn_info, smooth_l1_loss):
    if not rcnn_info:
        return tf.constant(0.)
    with tf.name_scope("rcnn_loss"):
        # RCNN class loss
        class_scores = rcnn_info["class_scores"]
        class_labels = tf.reshape(rcnn_info["class_labels"], [-1])
        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=class_labels, logits=class_scores)

        # RCNN box loss
        bbox_predicts = rcnn_info["bbox_predicts"]
        bbox_labels = rcnn_info["bbox_labels"]
        in_weights = rcnn_info["in_weights"]
        out_weights = rcnn_info["out_weights"]
        bbox_loss = smooth_l1_loss(bbox_predicts, bbox_labels,
                                   in_weights, out_weights, axis=[1])
        return tf.add(class_loss, bbox_loss)


def _rcnn_cls_layer(inputs,
                    num_outputs,
                    trainable=True,
                    weights_initializer=None,
                    scope="rcnn_cls_layer"):
    with tf.variable_scope(scope):
        # scores = layers.fully_connected(inputs, num_outputs,
        #                                 activation_fn=None,
        #                                 trainable=self.trainable,
        #                                 weights_initializer=self.initializer)
        scores = slim.fully_connected(inputs, num_outputs,
                                      activation_fn=None,
                                      trainable=trainable,
                                      weights_initializer=weights_initializer)
        probs = tf.nn.softmax(scores, name="rcnn_cls_prob_softmax")
        predicts = tf.argmax(scores, axis=1, name="rcnn_cls_pred_argmax")
    return probs, predicts, scores


def _rcnn_reg_layer(inputs,
                    num_outputs,
                    trainable=True,
                    weights_initializer=None,
                    scope="rcnn_reg_layer"):
    with tf.variable_scope(scope):
        # deltas = layers.fully_connected(inputs, num_outputs,
        #                                 activation_fn=None,
        #                                 trainable=self.trainable,
        #                                 weights_initializer=self.bbox_initializer)
        deltas = slim.fully_connected(inputs, num_outputs,
                                      activation_fn=None,
                                      trainable=trainable,
                                      weights_initializer=weights_initializer)
    return deltas
