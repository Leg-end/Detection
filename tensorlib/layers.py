"""Lib for transfer and construct pre-trained model, similar to slim"""
import tensorflow as tf
# from tensorflow.contrib import slim
import tensorlib.intializers as initializers
import tensorlib.core as core

# def func():
#    slim.fully_connected


arg_scope = dict()


def fill_arg_scope(**kwargs):
    global arg_scope
    arg_scope.update(**kwargs)


def _variable_with_weight_decay(name,
                                shape,
                                initializer,
                                trainable=True,
                                dtype=tf.float32,
                                wd=None):
    var = tf.get_variable(name, shape, trainable=trainable,
                          initializer=initializer, dtype=dtype)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),
                                   wd, name="weight_loss")
        tf.add_to_collection('regularization_losses', weight_decay)
    return var


def conv2d(inputs,
           num_output,
           kernel_size,
           strides=(1, 1),
           padding='SAME',
           trainable=True,
           activation_fn=tf.nn.relu,
           weights_initializer=tf.truncated_normal(0.0, 1e-1, dtype=tf.float32),
           weights_decay_factor=None,
           use_bias=True,
           biases_initializer=tf.constant_initializer(0.0),
           biases_decay_factor=None,
           data_format="NHWC",
           reuse=None,
           dtype=tf.float32,
           scope=None):
    global arg_scope
    if arg_scope["weights_decay"]:
        weights_decay_factor = arg_scope["weights_decay"]
    if arg_scope["biases_decay"]:
        biases_decay_factor = arg_scope["biases_decay"]
    with tf.variable_scope(scope or "conv2d", reuse=tf.AUTO_REUSE):
        if data_format == "NHWC":
            kernel_shape = [inputs.get_shape().as_list()[0], kernel_size[0], kernel_size[1], num_output]
        else:  # "NCHW"
            kernel_shape = [inputs.get_shape().as_list()[0], num_output, kernel_size[0], kernel_size[1]]
        # if hasattr(weights_initializer, '__call__'):
        #     weights_initializer = weights_initializer(shape=kernel_shape)
        # if not weights_initializer:
        #     weights_initializer = tf.glorot_normal_initializer
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             initializer=weights_initializer,
                                             trainable=trainable,
                                             wd=weights_decay_factor,
                                             dtype=dtype)
        strides = [1, strides[0], strides[1], 1]
        conv = tf.nn.conv2d(inputs, kernel, strides, padding=padding, name=scope)
        if use_bias:
            biases = _variable_with_weight_decay('biases',
                                                 shape=[num_output],
                                                 initializer=biases_initializer,
                                                 trainable=trainable,
                                                 wd=biases_decay_factor)
            conv = tf.nn.bias_add(conv, biases)
        if activation_fn:
            conv = activation_fn(conv, name=scope)

        return conv


def max_pool2d(inputs,
               kernel_size,
               strides=(1, 1),
               padding='SAME',
               data_format="NHWC",
               scope=None):
    with tf.name_scope(scope or "max_pool2d"):
        if data_format == "NHWC":
            kernel_shape = [1, kernel_size[0], kernel_size[1], inputs.get_shape().as_list()[3]]
        else:  # "NCHW"
            kernel_shape = [1, inputs.get_shape().as_list()[3], kernel_size[0], kernel_size[1]]
        strides = [1, strides[0], strides[1], 1]
        pool = tf.nn.max_pool(inputs, ksize=kernel_shape, strides=strides,
                              padding=padding, name=scope)
    return pool


def repeat_conv2d(inputs,
                  repetitions,
                  num_outputs,
                  kernel_size,
                  **kwargs):
    scope = kwargs["scope"]
    with tf.variable_scope(scope or "conv2d"):
        net = inputs
        for i in range(repetitions):
            kwargs["scope"] = scope + "_" + str(i + 1)
            net = conv2d(net, num_outputs, kernel_size, **kwargs)
    return net


def fully_connected(inputs,
                    num_units,
                    use_bias=True,
                    trainable=True,
                    activation_fn=tf.nn.sigmoid,
                    weights_initializer=None,
                    weights_decay_factor=None,
                    biases_initializer=tf.constant_initializer(0.0),
                    biases_decay_factor=None,
                    scope=None):
    with tf.variable_scope(scope or "fc"):
        global arg_scope
        if arg_scope["weights_decay"]:
            weights_decay_factor = arg_scope["weights_decay"]
        if arg_scope["biases_decay"]:
            biases_decay_factor = arg_scope["biases_decay"]
        kernel_shape = [inputs.get_shape().as_list()[1], num_units]
        # if hasattr(weights_initializer, '__call__'):
        #     weights_initializer = weights_initializer(shape=kernel_shape)
        if not weights_initializer:
            weights_initializer = tf.glorot_normal_initializer
        weights = _variable_with_weight_decay('weights', kernel_shape,
                                              initializer=weights_initializer,
                                              trainable=trainable,
                                              wd=weights_decay_factor)
        outputs = tf.matmul(inputs, weights)
        if use_bias:
            biases = _variable_with_weight_decay('biases', [num_units],
                                                 initializer=biases_initializer,
                                                 trainable=trainable,
                                                 wd=biases_decay_factor)
            outputs = tf.nn.bias_add(outputs, biases)
        if activation_fn:
            outputs = activation_fn(outputs)
    return outputs


def drop_out(inputs,
             rate=0.5,
             trainable=True,
             noise_shape=None,
             seed=None,
             scope=None):
    with tf.variable_scope(scope or "dropout") as sc:
        if trainable:
            return tf.nn.dropout(inputs, rate=rate,
                                 noise_shape=noise_shape,
                                 seed=seed, name=sc.name)
    return inputs


class Conv2D(core.Layer):

    def __init__(self,
                 out_channels,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 in_channels=None,
                 padding='VALID',
                 data_format='NHWC',
                 activation_fn=tf.nn.relu,
                 weights_initializer=tf.initializers.zeros(),
                 weights_regularizer=None,
                 biases_initializer=tf.initializers.zeros(),
                 biases_regularizer=None,
                 name=None):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.strides = strides
        self.weights_init = weights_initializer
        self.biases_init = biases_initializer
        self.in_channels = in_channels
        self.activation_fn = activation_fn
        self.weights_regular = weights_regularizer
        self.biases_regular = biases_regularizer
        self.data_format = data_format
        self.padding = padding

    def build(self, input_shape):
        if self.data_format == 'NHWC':
            if self.in_channels is None:
                self.strides = [1, self.strides[0], self.strides[1], 1]
                self.in_channels = input_shape[-1]
        elif self.data_format == 'NCHW':
            if self.in_channels is None:
                self.strides = [1, 1, self.strides[0], self.strides[1]]
                self.in_channels = input_shape[1]
        else:
            raise ValueError("Data_format must be 'NHWC' or 'NCHW'.")

        self.kernel_shape = [self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels]
        self.biases_shape = [self.out_channels]
        self.weights = self._get_weights(name="kernels",
                                         shape=self.kernel_shape,
                                         initializer=self.weights_init,
                                         regularizer=self.weights_regular)

        if self.biases_init:
            self.biases = self._get_weights(name="biases",
                                            shape=self.biases_shape,
                                            initializer=self.biases_init,
                                            regularizer=self.weights_regular)

    def forward(self, inputs):
        outputs = tf.nn.conv2d(
                inputs,
                filter=self.weights,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilations=[1, 1, 1, 1],
                name=self._name)

        if self.biases_init is True:
            outputs = tf.nn.bias_add(
                value=outputs,
                bias=self.biases,
                data_format=self.data_format,
                name="bias_add")

        return outputs