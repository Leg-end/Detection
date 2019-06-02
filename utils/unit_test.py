import os
import tensorflow as tf
import time
from utils import anchor_util
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_anchors():
    anchors = anchor_util.generate_anchors()
    with tf.Session() as sess:
        print(sess.run(anchors))


def test_image_anchors():
    anchors, num = anchor_util.generate_image_anchors(224, 224)
    with tf.Session() as sess:
        print(sess.run([anchors, num]))


def test_target_anchors():
    overlap = tf.constant([[[.1], [.2], [.3], [.5], [.7], [.2]],
                           [[.9], [.4], [.6], [.1], [.2], [.1]],
                           [[.3], [.6], [.6], [.1], [.6], [.2]]])
    p = anchor_util.generate_positives_negatives(overlap)
    print(p)


if __name__ == "__main__":
    t = time.time()
    test_target_anchors()
    print(time.time()-t)
