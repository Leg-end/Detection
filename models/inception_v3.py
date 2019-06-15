from .base_model import BaseModel
import tensorflow as tf
import model_helper as helper
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
from tensorflow.contrib import slim
import os


class InceptionV3(BaseModel):
    def __init__(self, hparams, data_wrapper, reverse_cate_table, scope=None):
        if not scope:
            scope = "InceptionV3"
        self.feat_stride = [16]
        super(InceptionV3, self).__init__(hparams, data_wrapper, reverse_cate_table, scope)

    def _image_to_head(self, inputs, reuse=None):
        inception_output = self.inception_v3(
            inputs,
            trainable=self.tunable,
            is_training=self.trainable)
        # prepare restore op
        self.restore_op = helper.restore_pre_model(
            self.scope, os.path.join(self.hparams.pre_ckpt_dir, self.scope + ".ckpt"))
        return inception_output

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

    def inception_v3(self,
                     images,
                     trainable=True,
                     is_training=True,
                     weight_decay=0.00004,
                     stddev=0.1,
                     dropout_keep_prob=0.8,
                     use_batch_norm=True,
                     batch_norm_params=None,
                     add_summaries=True,
                     scope="InceptionV3"):
        """Builds an Inception V3 subgraph for image embeddings.
        Args:
            images: A float32 Tensor of shape [batch, height, width, channels].
            trainable: Whether the inception submodel should be trainable or not.
            is_training: Boolean indicating training mode or not.
            weight_decay: Coefficient for weight regularization.
            stddev: The standard deviation of the trunctated normal weight initializer.
            dropout_keep_prob: Dropout keep probability.
            use_batch_norm: Whether to use batch normalization.
            batch_norm_params: Parameters for batch normalization. See
            tf.contrib.layers.batch_norm for details.
            add_summaries: Whether to add activation summaries.
            scope: Optional Variable scope.
        Returns:
            end_points: A dictionary of activations from inception_v3 layers.
        """
        # Only consider the inception model to be in training mode if it's trainable.
        is_inception_model_training = trainable and is_training

        if use_batch_norm:
            # Default parameters for batch normalization.
            if not batch_norm_params:
                batch_norm_params = {
                    "is_training": is_inception_model_training,
                    "trainable": trainable,
                    # Decay for the moving averages.
                    "decay": 0.9997,
                    # Epsilon to prevent 0s in variance.
                    "epsilon": 0.001,
                    # Collection containing the moving mean and moving variance.
                    "variables_collections": {
                        "beta": None,
                        "gamma": None,
                        "moving_mean": ["moving_vars"],
                        "moving_variance": ["moving_vars"],
                    }
                }
            else:
                batch_norm_params = None

            if trainable:
                weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
            else:
                weights_regularizer = None
            with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
                with slim.arg_scope(
                        [slim.conv2d, slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        trainable=trainable):
                    with slim.arg_scope(
                            [slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
                        net, end_points = inception_v3_base(images, scope=scope)
                        with tf.variable_scope("logits"):
                            shape = net.get_shape().as_list()
                            net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                            net = slim.dropout(
                                net,
                                keep_prob=dropout_keep_prob,
                                is_training=is_inception_model_training,
                                scope="dropout")
                            net = slim.flatten(net, scope="flatten")

        # Add summaries.
        if add_summaries:
            for v in end_points.values():
                tf.contrib.layers.summaries.summarize_activation(v)
        return net
