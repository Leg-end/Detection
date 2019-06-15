import os
import tensorflow as tf
import time
import numpy as np
from utils import anchor_util
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_anchors():
    anchors = anchor_util.generate_anchors()
    with tf.Session() as sess:
        print(sess.run(anchors))


def test_image_anchors():
    height = tf.to_int32(tf.ceil(tf.divide(tf.to_float(480), tf.to_float(16))))
    width = tf.to_int32(tf.ceil(tf.divide(tf.to_float(640), tf.to_float(16))))
    # anchors, num = anchor_util.generate_image_anchors(height, width)
    anchors, num = generate_anchors_pre_tf(height, width)
    with tf.Session() as sess:
        print(sess.run([anchors, num]))


def test_scale():
    feat_stride = 16
    height = tf.to_int32(tf.ceil(tf.divide(tf.to_float(480), tf.to_float(feat_stride))))
    width = tf.to_int32(tf.ceil(tf.divide(tf.to_float(640), tf.to_float(feat_stride))))
    shift_x = tf.multiply(tf.range(width), feat_stride)
    shift_y = tf.multiply(tf.range(height), feat_stride)
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    with tf.Session() as sess:
        print(sess.run([shift_x, shift_y]))


def test_target_anchors():
    overlap = tf.constant([[[.1], [.2], [.3], [.5], [.7], [.2]],
                           [[.9], [.4], [.6], [.1], [.2], [.1]],
                           [[.3], [.6], [.6], [.1], [.6], [.2]]])
    p = anchor_util.generate_positives_negatives(overlap)
    print(p)


def generate_anchors_pre_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    shift_x = tf.range(width) * feat_stride  # width
    shift_y = tf.range(height) * feat_stride  # height
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    sx = tf.reshape(shift_x, shape=(-1,))
    sy = tf.reshape(shift_y, shape=(-1,))
    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
    K = tf.multiply(width, height)
    shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))
    anchors = anchor_util.generate_anchors()
    A = anchors.shape[0]
    anchor_constant = tf.to_int32(tf.reshape(anchors, shape=[1, A, 4]))
    length = K * A
    anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
    return tf.cast(anchors_tf, dtype=tf.float32), length


if __name__ == "__main__":
    t = time.time()
    test_image_anchors()
    # test_scale()
    # test_scale()
    # test_target_anchors()
    print(time.time()-t)
    # a = tf.expand_dims(tf.reshape(tf.range(20), shape=[5, 4]), dim=0)
    # a = tf.tile(a, multiples=[2, 1, 1])
    # with tf.Session() as sess:
    #     print(sess.run(tf.shape(a)))
