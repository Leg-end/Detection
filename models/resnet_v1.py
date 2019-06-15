import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
from .base_model import BaseModel
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


class ResNetV1(BaseModel):
    def __init__(self, hparams, data_wrapper, reverse_cate_table, num_layers=50, scope=None):
        if not scope:
            scope = 'resnet_v1_%d' % num_layers
        self.feat_stride = [16]
        self.num_layers = num_layers
        self._decide_block()
        super(ResNetV1, self).__init__(hparams, data_wrapper, reverse_cate_table, scope)

    def _build_base(self, inputs):
        with tf.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(inputs, 64, 7, stride=2, scope="conv1")
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
        return net

    def _image_to_head(self, inputs, reuse=None):
        hp = self.hparams
        fixed_blocks = hp.resnet_fixed_blocks
        with slim.arg_scope(resnet_arg_scope(trainable=self.tunable,
                                             weight_decay=hp.weight_decay_factor)):
            net_conv = self._build_base(inputs)
        blocks = self.blocks
        if fixed_blocks > 0:
            blocks = blocks[0:fixed_blocks]
        if fixed_blocks < 3:
            blocks = blocks[fixed_blocks:-1]
        with slim.arg_scope(resnet_arg_scope(trainable=self.tunable,
                                             weight_decay=hp.weight_decay_factor)):
            net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                              blocks,
                                              global_pool=False,
                                              include_root_block=False,
                                              reuse=reuse,
                                              scope=self.scope)
        self.activations.append(net_conv)
        self.restore_op = helper.restore_pre_model(
            self.scope, os.path.join(hp.pre_ckpt_dir, self.scope+".ckpt"))
        return net_conv

    def _head_to_tail(self, inputs, reuse=None):
        with slim.arg_scope(resnet_arg_scope(trainable=self.trainable,
                                             weight_decay=self.hparams.weight_decay_factor)):
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
