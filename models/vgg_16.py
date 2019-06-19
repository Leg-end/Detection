from .base_model import BaseModel
from tensorflow.contrib import slim
import tensorflow as tf
import model_helper as helper
import os


class VGG16B(BaseModel):
    def __init__(self, hparams, reverse_cate_table, data_wrapper, scope=None):
        if not scope:
            scope = "vgg_16"
        self.feat_stride = [16]
        super(VGG16B, self).__init__(hparams, reverse_cate_table, data_wrapper, scope)

    def _image_to_head(self, inputs, reuse=None):
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
            if self.restore_op is None:
                self.restore_op = helper.restore_pre_model(
                    self.scope, os.path.join(self.hparams.pre_ckpt_dir, self.scope + ".ckpt"))
        return net

    def _head_to_tail(self, inputs, reuse=None):
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
