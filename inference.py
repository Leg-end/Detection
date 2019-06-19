import tensorflow as tf
import model_helper as helper
from utils import misc_util as misc
import os
import time
import numpy as np
from PIL import Image


tf.logging.set_verbosity(tf.logging.INFO)


def get_model_creator(model_type):
    # from models.vgg16 import VGG16
    from models.inception_v3 import InceptionV3
    from models.vgg_16 import VGG16B
    from models.resnet_v1 import ResNetV1
    if model_type == "VGG_16":
        return VGG16B
    elif model_type == "InceptionV3":
        return InceptionV3
    elif model_type == "ResNetV1":
        return ResNetV1


def start_sess_and_load_model(infer_model, ckpt_dir):
    sess = tf.Session(
        graph=infer_model.graph, config=misc.get_config_proto())
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = helper.create_or_load_model(
          infer_model.model, ckpt_dir, sess)
    return sess, loaded_infer_model, global_step


def infer(hparams, ckpt_dir, scope=None, target_session=""):
    output_dir = os.path.join(hparams.base_dir, "infer")
    if not misc.check_file_existence(output_dir):
        tf.gfile.MakeDirs(output_dir)
    model_creator = get_model_creator(hparams.model_type)
    infer_model = helper.create_infer_model(model_creator, hparams, scope)
    infer_sess, loaded_infer_model, global_step = start_sess_and_load_model(
        infer_model, ckpt_dir)
    tf.logging.info("Restore model from global step %d" % global_step)
    # Summary
    summary_name = "infer_summary"
    summary_path = os.path.join(output_dir, summary_name)
    if not tf.gfile.Exists(summary_path):
        tf.gfile.MakeDirs(summary_path)
    summary_writer = tf.summary.FileWriter(
        summary_path, infer_model.graph)
    infer_results = []
    tf.logging.info("Ready to infer")
    start_time = time.time()
    step = 0
    detect_results = list()
    dataset_dir = os.path.join(hparams.data_dir, "test")
    filenames = tf.gfile.ListDirectory(dataset_dir)
    for filename in filenames:
        im = Image.open(os.path.join(dataset_dir, filename))
        image_feed = np.expand_dims(np.asarray(im, dtype='float32'), axis=0)
        size_feed = np.asarray([im.size[0], im.size[1], 3], dtype='int32')
        tf.logging.info("Start infer step:%d" % step)
        step_result, detect_result = loaded_infer_model.infer(infer_sess, image_feed=image_feed, size_feed=size_feed)
        summary_writer.add_summary(step_result.summary, global_step)
        infer_results.append(step_result.bboxes)
        detect_results.append(detect_result)
        step += 1
    tf.logging.info("Finish infer <spend time:%d>" % (time.time() - start_time))
    misc.draw_detect_results(detect_results)
