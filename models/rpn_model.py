import tensorflow as tf
from tensorflow.contrib import slim
import model_helper as helper
from utils import misc_util as misc
from . import base_model


def rpn_arg_scope(trainable=True,
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


def rpn_base(inputs,
             hp,
             im_info,
             gt_bboxes,
             feat_stride=16,
             anchor_count=9,
             trainable=True,
             predictable=False):
    """
    A Region Proposal Network (RPN) takes an image (of any size) as input and outputs a set of
    rectangular object proposals, each with an objectness score[quoted from Faster-RCNN]
    :param inputs: image
    :param hp: hyper parameters
    :param im_info: image size [height, width, channel]
    :param feat_stride: num of division
    :param anchor_count: original generate anchors' count
    :param trainable: whether to train this network
    :param gt_bboxes: ground-truth bounding boxes
    :param predictable: is in infer mode
    :return: rect object proposals, scores, print_pool(dict) for debugging, activation(dict) for visualization
    """
    with tf.variable_scope("rpn"):
        with tf.device(helper.get_device_str(device_id=0, num_gpus=hp.num_gpus)):
            # Build the anchors for image
            anchors, all_count = helper.generate_img_anchors(im_info, feat_stride,
                                                             ratios=hp.anchor_ratios,
                                                             scales=hp.anchor_scales)
            # rpn_conv = layers.conv2d(inputs, hp.rpn_channel, [3, 3], trainable=self.trainable,
            #                          weights_initializer=self.initializer, scope="rpn_conv_3x3")
            rpn_conv = slim.conv2d(inputs, hp.rpn_channel, [3, 3], trainable=trainable, scope="rpn_conv_3x3")
            probs, predicts, scores, reshaped_scores = _rpn_cls_layer(
                rpn_conv, anchor_count * 2)
            deltas = _rpn_reg_layer(rpn_conv, anchor_count * 4)
            rpn_info = dict()
            if not predictable:
                # Generate rois, roi scores on image
                rois, roi_scores, roi_deltas = helper.sample_rois_from_anchors(
                    probs, deltas, im_info, anchors, anchor_count)
                # Gather info for calculating rpn's loss
                # Fill rpn's bbox_label, class_label, in_weights, out_weights
                rpn_info = helper.pack_anchor_info(
                    im_info, anchors, ori_anchor_count=anchor_count,
                    bbox_targets=gt_bboxes,
                    anchor_scores=scores)
                # base_model.inject_to_debug_pool(class_labels=tf.where(
                #     tf.logical_and(tf.not_equal(rpn_info["class_labels"], -1),
                #                    tf.not_equal(rpn_info["class_labels"], 0))),
                #     bbox_labels=rpn_info["bbox_labels"])
                # base_model.inject_to_debug_pool(gt_boxes=gt_bboxes, bbox_labels=rpn_info["bbox_labels"])
            else:
                # Why use probs as scores?
                rois, roi_scores, roi_deltas = helper.sample_rois_from_anchors(
                    probs, deltas, im_info, anchors, anchor_count)
            # Fill rest info of rpn
            misc.append_params(rpn_info, rois=rois,
                               class_probs=probs, class_predicts=predicts,
                               class_reshaped_scores=reshaped_scores, sigma=hp.rpn_sigma,
                               # Using the full score instead of roi_score so that gradient
                               # can back passing all params in rpn_cls_layer, same as deltas
                               bbox_predicts=deltas, class_scores=scores,
                               roi_deltas=roi_deltas, roi_scores=roi_scores)
    return rois, rpn_info, rpn_conv


def rpn_loss(rpn_info, smooth_l1_loss):
    if not rpn_info:
        return tf.constant(0.)
    with tf.name_scope("rpn_loss"):
        # RPN class loss
        class_scores = tf.reshape(rpn_info["class_reshaped_scores"], [-1, 2])
        class_labels = tf.reshape(rpn_info["class_labels"], [-1])
        """
        select_inds = tf.where(tf.not_equal(class_labels, -1))
        """
        cls_scores_masks = tf.where(tf.not_equal(class_scores, -1), class_scores, tf.zeros_like(class_scores))
        cls_labels_masks = tf.where(tf.not_equal(class_labels, -1), class_labels, tf.zeros_like(class_labels))
        # base_model.inject_to_debug_pool(class_scores=class_scores, class_labels=class_labels, select_inds=select_inds)
        """
        class_scores = tf.reshape(tf.gather(class_scores, select_inds), [-1, 2])
        class_labels = tf.reshape(tf.gather(class_labels, select_inds), [-1])
        """
        class_scores = tf.reshape(cls_scores_masks, [-1, 2])
        class_labels = tf.reshape(cls_labels_masks, [-1])
        class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=class_scores, labels=class_labels))

        # RPN box loss
        bbox_predicts = rpn_info["bbox_predicts"]
        bbox_labels = rpn_info["bbox_labels"]
        in_weights = rpn_info["in_weights"]
        out_weights = rpn_info["out_weights"]
        sigma = rpn_info["sigma"]
        bbox_loss = smooth_l1_loss(bbox_predicts, bbox_labels,
                                   in_weights, out_weights,
                                   sigma=sigma, axis=[1, 2, 3])
        return tf.add(class_loss, bbox_loss)


def _rpn_reg_layer(inputs, num_output, scope="rpn_reg_layer", trainable=True):
    # deltas = layers.conv2d(inputs, num_output, [1, 1], trainable=self.trainable,
    #                        weights_initializer=self.initializer, padding='VALID',
    #                        activation_fn=None, scope=name)
    deltas = slim.conv2d(inputs, num_output, [1, 1], trainable=trainable, padding='VALID',
                         activation_fn=None, scope=scope)
    return deltas


def _rpn_cls_layer(inputs, num_output, scope="rpn_cls_layer", trainable=True):
    # scores = layers.conv2d(inputs, num_output, [1, 1], trainable=self.trainable,
    #                        weights_initializer=self.initializer, padding='VALID',
    #                        activation_fn=None, scope=name)
    scores = slim.conv2d(inputs, num_output, [1, 1], trainable=trainable, padding='VALID',
                         activation_fn=None, scope=scope)
    # Fix channel as 2
    reshaped_scores = helper.reshape_for_forward_pairs(scores, 2, name="rpn_cls_score_reshape")
    shape = tf.shape(reshaped_scores)
    reshaped = tf.reshape(reshaped_scores, shape=[-1, shape[-1]])
    outputs = tf.nn.softmax(reshaped, name="rpn_cls_score_softmax")
    reshaped_probs = tf.reshape(outputs, shape=shape)
    # Get predict class id
    predicts = tf.argmax(tf.reshape(reshaped_scores, shape=[-1, 2]), axis=1, name="rpn_cls_predict_argmax")
    # Switch to original channel, after softmax, here we get prob for positive and negative
    # Form as [n, n,....,n (count=num_outputs/2), p, p, ......,p (count=num_outputs/2)]
    probs = helper.reshape_for_forward_pairs(reshaped_probs, num_output, name="rpn_cls_prob_reshape")
    return probs, predicts, scores, reshaped_scores
