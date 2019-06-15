import tensorflow as tf
import model_helper as helper
from utils import misc_util as misc
import os
import time
from matplotlib import pyplot as plt


tf.logging.set_verbosity(tf.logging.INFO)


def get_model_creator(model_type):
    from models.vgg16 import VGG16
    from models.inception_v3 import InceptionV3
    from models.vgg_16 import VGG16B
    if model_type == "VGG16":
        return VGG16
    elif model_type == "VGG_16":
        return VGG16B
    elif model_type == "InceptionV3":
        return InceptionV3


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
    while True:
        try:
            tf.logging.info("Start infer step:%d" % step)
            results = loaded_infer_model.infer(infer_sess)
            summary_writer.add_summary(results.summary, global_step)
            infer_results.append(results.detected_images)
            step += 1
        except tf.errors.OutOfRangeError:
            tf.logging.info("Finish infer <time:%d>" % (start_time-time.time()))
            break
    for result in infer_results:
        plt.show(result)
