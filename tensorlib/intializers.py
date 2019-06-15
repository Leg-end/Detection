import math

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import tensorflow as tf


def xavier_initializer(uniform=True,
                       seed=None,
                       dtype=dtypes.float32):
    """
    This function implements the weight initialization from:
    Xavier Glorot and Yoshua Bengio (2010):
        [Understanding the difficulty of training deep feedforward neural
        networks. International conference on artificial intelligence and
        statistics.](
        http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    This initializer is designed to keep the scale of the gradients roughly the
    same in all layers. In uniform distribution this ends up being the range:
        `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution a standard
        deviation of `sqrt(2. / (in + out))` is used.
    :param uniform:
    :param seed:
    :param dtype:
    :return:
    """
    return variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                        uniform=uniform, seed=seed, dtype=dtype)


def variance_scaling_initializer(factor=1.0,
                                 mode='FAN_AVG',
                                 uniform=False,
                                 seed=None,
                                 dtype=dtypes.float32):
    if not dtype.is_floating:
        raise TypeError('Cannot create initializer for non-floating point type.')
    if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
        raise TypeError('Unknown mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

    # pylint: disable=unused-argument
    def _initializer(shape, dtype=dtype, partition_info=None):
        """Initializer function."""
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating point type.')
        # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
        # This is the right thing for matrix multiply and convolutions.
        if shape:
            f1 = lambda: tf.cast(shape[-2], dtype)
            f2 = lambda: tf.cast(shape[-1], dtype)
            fan_in = tf.case([(tf.greater(len(shape), 1), f1)], default=f2)
            fan_out = tf.cast(shape[-1], dtype)
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in = tf.multiply(dim, fan_in)
            fan_out = tf.multiply(dim, fan_out)
        if mode == 'FAN_IN':
            # Count only number of input connections.
            n = fan_in
        elif mode == 'FAN_OUT':
            # Count only number of output connections.
            n = fan_out
        elif mode == 'FAN_AVG':
            # Average number of inputs and output connections.
            n = tf.divide(tf.add(fan_in, fan_out), 2.0)
        if uniform:
            # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            limit = tf.sqrt(tf.divide(tf.multiply(3.0, factor), n))
            return random_ops.random_uniform(shape, -limit, limit,
                                             dtype, seed=seed)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = tf.sqrt(tf.divide(tf.multiply(1.3, factor), n))
            return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                               seed=seed)

    # pylint: enable=unused-argument
    return _initializer
