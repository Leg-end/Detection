# import os
# import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import model_helper as helper
import math
import main
import argparse
from utils import anchor_util
import os


def resnet_arg_scope(trainable=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     weight_decay=0.0001):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=trainable,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


class ResNetV1(object):
    def __init__(self, num_layers=50, scope=None):
        if not scope:
            self.scope = 'resnet_v1_%d' % num_layers
        self.feat_stride = [16]
        self.num_layers = num_layers
        self.tunable = False
        self.trainable = False
        self._decide_block()

    def _build_base(self, inputs):
        with tf.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(inputs, 64, 7, stride=2, scope="conv1")
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
        return net

    def image_to_head(self, inputs, reuse=None):
        fixed_blocks = 1
        with slim.arg_scope(resnet_arg_scope(trainable=self.tunable)):
            net_conv = self._build_base(inputs)
        blocks = self.blocks
        if fixed_blocks > 0:
            blocks = blocks[0:fixed_blocks]
        if fixed_blocks < 3:
            blocks = blocks[fixed_blocks:-1]
        with slim.arg_scope(resnet_arg_scope(trainable=self.tunable)):
            net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                              blocks,
                                              global_pool=False,
                                              include_root_block=False,
                                              reuse=reuse,
                                              scope=self.scope)
        return net_conv

    def head_to_tail(self, inputs, reuse=None):
        with slim.arg_scope(resnet_arg_scope(trainable=self.trainable)):
            fc, _ = resnet_v1.resnet_v1(inputs,
                                        self.blocks[-1:],
                                        global_pool=False,
                                        include_root_block=False,
                                        reuse=reuse,
                                        scope=self.scope)
            # average pooling done by reduce_mean
            fc = tf.reduce_mean(fc, axis=[1, 2])
        return fc

    def _decide_block(self):
        # choose different blocks for different number of layers
        base_depths = list()
        strides = list()
        self.blocks = list()
        scopes = ["block1", "block2", "block3", "block4"]
        for i in range(2):
            base_depths.append(int(math.pow(2, i+6)))
            strides.append(2)
        for i in range(2):
            base_depths.append(int(math.pow(2, i + 8)))
            strides.append(1)
        if self.num_layers == 50:
            num_units = [3, 4, 6, 3]
        elif self.num_layers == 101:
            num_units = [3, 4, 23, 3]
        elif self.num_layers == 152:
            num_units = [3, 8, 36, 3]
        else:
            # other numbers are not supported
            raise NotImplementedError
        for i in range(4):
            self.blocks.append(resnet_v1_block(
                scope=scopes[i], base_depth=base_depths[i],
                num_units=num_units[i], stride=strides[i]))


class VGG16B(object):
    def __init__(self, scope=None):
        if not scope:
            self.scope = "vgg_16"
        self.feat_stride = [16]
        self.tunable = False
        self.trainable = False

    def image_to_head(self, inputs, reuse=None):
        with tf.variable_scope(self.scope, self.scope, reuse=reuse):
            kernel_size = [3, 3]
            pool_size = [2, 2]
            net = slim.repeat(inputs, 2, slim.conv2d, 64, kernel_size,
                              trainable=self.tunable, scope='conv1')
            net = slim.max_pool2d(net, pool_size, padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, kernel_size,
                              trainable=self.tunable, scope='conv2')
            net = slim.max_pool2d(net, pool_size, padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, kernel_size,
                              trainable=self.tunable, scope='conv3')
            net = slim.max_pool2d(net, pool_size, padding='SAME',
                                  scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, kernel_size,
                              trainable=self.tunable, scope='conv4')
            net = slim.max_pool2d(net, pool_size, padding='SAME',
                                  scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, kernel_size,
                              trainable=self.tunable, scope='conv5')
            # prepare restore op
            # if self.restore_op is None:
            #     self.restore_op = helper.restore_pre_model(
            #         self.scope, os.path.join(self.hparams.pre_ckpt_dir, self.scope + ".ckpt"))
        return net

    def head_to_tail(self, inputs, reuse=None):
        with tf.variable_scope(self.scope, self.scope, reuse=reuse):
            net = slim.fully_connected(inputs, 4096, scope="fc6",
                                       trainable=self.trainable)
            if self.trainable:
                net = slim.dropout(net, keep_prob=0.5, scope="dropout6",
                                   is_training=self.trainable)
            net = slim.fully_connected(net, 4096, scope="fc7",
                                       trainable=self.trainable)
            if self.trainable:
                net = slim.dropout(net, keep_prob=0.5, scope="dropout7",
                                   is_training=self.trainable)
        return net


def _rpn_cls_layer(inputs, num_output, scope="rpn_cls_layer", trainable=False):
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


def rpn():
    param_parser = argparse.ArgumentParser()
    main.add_arguments(param_parser)
    flags = param_parser.parse_args()
    flags = vars(flags)
    def_hparams = main.create_hparams(flags)
    main.pre_fill_params_into_utils(def_hparams)
    image = tf.ones(shape=[1, 640, 480, 3])
    bbox_target = tf.constant([[80.54, 25.89, 250.25, 606.92, 25.],
                               [125.55, 367.46, 349.09, 263.35, 46.],
                               [480.99, 177.05, 23.44, 29.71, 71.]])
    net = VGG16B()
    net_conv = net.image_to_head(image)
    anchors, _ = helper.generate_img_anchors([640, 480, 3])
    rpn = slim.conv2d(net_conv, 128, [3, 3], trainable=False,  scope="rpn_conv/3x3")
    rpn_bbox_pred = slim.conv2d(rpn, 36, [1, 1], trainable=False,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    probs, predicts, scores, reshaped_scores = _rpn_cls_layer(
        rpn, 18)
    rois, roi_scores = helper.sample_rois_from_anchors(
        probs, rpn_bbox_pred, [640, 480, 3], anchors, 9)
    rpn_labels, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights = anchor_util.generate_anchor_targets(
        scores, anchors, bbox_target, [640, 80, 3], 9)
    rpn_labels = tf.to_int32(rpn_labels)
    with tf.Session() as sess:
        print(sess.run([rpn_outside_weights]))


if __name__ == "__main__":
    """a = tf.range(2)*2
    b = tf.range(2)*2
    _a, _b = tf.meshgrid(a, b)
    _a = tf.reshape(_a, shape=(-1, ))
    _b = tf.reshape(_b, shape=(-1, ))
    c = tf.transpose(tf.stack([_a, _b, _a, _b]))
    c = tf.transpose(tf.reshape(c, shape=[1, 4, 4]), perm=(1, 0, 2))
    with tf.Session() as sess:
        print(sess.run(c))"""
    rpn()
