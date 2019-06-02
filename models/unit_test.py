import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    a = tf.range(2)*2
    b = tf.range(2)*2
    _a, _b = tf.meshgrid(a, b)
    _a = tf.reshape(_a, shape=(-1, ))
    _b = tf.reshape(_b, shape=(-1, ))
    c = tf.transpose(tf.stack([_a, _b, _a, _b]))
    c = tf.transpose(tf.reshape(c, shape=[1, 4, 4]), perm=(1, 0, 2))
    with tf.Session() as sess:
        print(sess.run(c))
