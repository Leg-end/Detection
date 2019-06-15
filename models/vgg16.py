from .base_model import BaseModel
from tensorlib import layers
import tensorflow as tf
import model_helper as helper
import os


class VGG16(BaseModel):
    def __init__(self, hparams, data_wrapper, reverse_cate_table, scope=None):
        if not scope:
            scope = "vgg_16"
        self.feat_stride = [16]
        super(VGG16, self).__init__(hparams, data_wrapper, reverse_cate_table, scope)

    def _image_to_head(self, inputs, reuse=None):
        with tf.variable_scope(self.scope, self.scope, reuse=reuse):
            kernel_size = [3, 3]
            pool_size = [2, 2]
            net = layers.repeat_conv2d(inputs, 2, 64, kernel_size,
                                       trainable=self.tunable, scope='conv1')
            net = layers.max_pool2d(net, pool_size, padding='SAME', scope='pool1')
            net = layers.repeat_conv2d(net, 2, 128, kernel_size,
                                       trainable=self.tunable, scope='conv2')
            net = layers.max_pool2d(net, pool_size, padding='SAME', scope='pool2')
            net = layers.repeat_conv2d(net, 3, 256, kernel_size,
                                       trainable=self.tunable, scope='conv3')
            net = layers.max_pool2d(net, pool_size, padding='SAME',
                                    scope='pool3')
            net = layers.repeat_conv2d(net, 3, 512, kernel_size,
                                       trainable=self.tunable, scope='conv4')
            net = layers.max_pool2d(net, pool_size, padding='SAME',
                                    scope='pool4')
            net = layers.repeat_conv2d(net, 3, 512, kernel_size,
                                       trainable=self.tunable, scope='conv5')
            # prepare restore op
            self.restore_op = helper.restore_pre_model(
                self.scope, os.path.join(self.hparams.pre_ckpt_dir, self.scope + ".ckpt"))
        return net

    def _head_to_tail(self, inputs, reuse=None):
        with tf.variable_scope(self.scope, self.scope, reuse=reuse):
            net = layers.fully_connected(inputs, 4096, scope="fc6")
            net = layers.drop_out(net, rate=0.5, scope="dropout6",
                                  trainable=self.tunable)
            net = layers.fully_connected(net, 4096, scope="fc7")
            net = layers.drop_out(net, rate=0.5, scope="dropout7",
                                  trainable=self.tunable)
        return net
