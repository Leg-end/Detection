import os
import tensorflow as tf
import time
import model_helper as helper
import eval
import main
import argparse
from utils import misc_util as misc
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_pyramid_pool():
    inputs = tf.reshape(tf.range(32), shape=[2, 2, 2, 4])
    bins_size_set = [[2, 2], [1, 1]]
    outputs = helper.spatial_pyramid_pooling(bins_size_set, inputs)
    with tf.Session() as sess:
        print(sess.run(outputs))


def test_generate_img_anchors():
    images_info = tf.constant([[600, 400, 3], [600, 400, 3]], dtype=tf.int32)
    anchors, all_count = helper.generate_img_anchors(images_info)
    with tf.Session() as sess:
        print(sess.run(tf.shape(anchors)))


def test_eval_mAP():
    bbox_labels = tf.reshape(
        tf.range(16), shape=[4, 4])
    bbox_infers = tf.reshape(
        tf.range(16), shape=[4, 4])
    results = eval.mean_avg_overlap(bbox_labels, bbox_infers)
    with tf.Session() as sess:
        print(sess.run(results))


def test_hparams():
    # param_parser = argparse.ArgumentParser()
    # main.add_arguments(param_parser)
    # flags, un_parsed = param_parser.parse_known_args()
    # flags = param_parser.parse_args()
    # flags = vars(flags)
    # a = {'a': 1, 'b': 2}
    # def_hparams = main.create_hparams(flags)
    def_hparams = misc.load_hparams("D:/Detection/hparams/")
    # print(def_hparams.repr__())
    misc.print_hparams(def_hparams)
    # misc.save_hparams(os.path.join(def_hparams.base_dir, "hparams"), def_hparams)


def test():
    helper.print_variables_in_ckpt("D:/Detection/pre_ckpt/vgg_16.ckpt")


def test_swap():
    a = tf.reshape(tf.range(16), shape=[4, 4])
    data = tf.data.Dataset()
    data = data.from_tensor_slices(a)
    iterator = data.make_one_shot_iterator()
    item = iterator.get_next()
    shape = tf.shape(item)
    dim1 = 1
    dim2 = 3
    indices = tf.range(shape)

    indices = tf.constant([[2], [1], [0], [3]])
    shape = tf.constant([4])
    scatter = tf.scatter_nd(indices=indices, shape=shape, updates=item)
    with tf.Session() as sess:
        print(sess.run(scatter))


if __name__ == "__main__":
    t = time.time()
    # test_eval_mAP()
    # test_hparams()
    # test_generate_img_anchors()
    test_swap()
    print(time.time() - t)
