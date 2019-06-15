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


def rpn():
    image = tf.ones(shape=[1, 640, 480, 3])
    net = ResNetV1()
    net_conv = net.image_to_head(image)
    rpn = slim.conv2d(net_conv, 128, [3, 3], trainable=False,  scope="rpn_conv/3x3")
    rpn_bbox_pred = slim.conv2d(rpn, 9 * 4, [1, 1], trainable=False,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    with tf.Session() as sess:
        print(sess.run(tf.shape(rpn_bbox_pred)))


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
