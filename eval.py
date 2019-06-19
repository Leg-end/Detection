import tensorflow as tf
import time
import model_helper as helper
from utils import misc_util as misc
import os


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


def evaluate(hparams, scope=None, target_session="", ckpt_path=None,
             summary_writer=None, global_step_=0, alternative=False):
    if alternative:
        out_dir = os.path.join(hparams.base_dir, "train")
    else:
        out_dir = os.path.join(hparams.base_dir, "eval")
        ckpt_path = os.path.join(hparams.base_dir, "ckpt")
    if not misc.check_file_existence(out_dir):
        tf.gfile.MakeDirs(out_dir)
    tf.logging.info("All eval relevant results will be put in %s" % out_dir)

    # Create model
    model_creator = get_model_creator(hparams.model_type)
    eval_model = helper.create_eval_model(model_creator,
                                          hparams, scope)
    config_proto = misc.get_config_proto(
        log_device_placement=hparams.log_device_placement,
        num_intra_threads=hparams.num_intra_threads,
        num_inter_threads=hparams.num_inter_threads)
    eval_sess = tf.Session(
        target=target_session, config=config_proto,
        graph=eval_model.graph)
    tf.logging.info("Create model successfully")
    with eval_model.graph.as_default():
        loaded_eval_model, global_step = helper.create_or_load_model(
            eval_model.model,
            ckpt_path,
            eval_sess)
    if global_step > 0:
        if global_step_ > 0:
            assert global_step_ == global_step
        tf.logging.info("Loading model from global step %d to evaluate" % global_step)
    else:
        tf.logging.info("With global step is 0, can not execute evaluation")
        return
    # Summary writer
    if summary_writer is None:
        summary_name = "eval_summary"
        summary_path = os.path.join(out_dir, summary_name)
        if not tf.gfile.Exists(summary_path):
            tf.gfile.MakeDirs(summary_path)
        summary_writer = tf.summary.FileWriter(
            os.path.join(out_dir, summary_name), eval_model.graph)
    eval_sess.run(eval_model.data_wrapper.initializer)
    tf.logging.info("Ready to eval")
    step = 0
    accuracies = 0.
    while True:
        start_time = time.time()
        try:
            tf.logging.info("Start eval step:%d" % step)
            results = loaded_eval_model.eval(eval_sess)
            summary_writer.add_summary(results.summary, global_step)
            tf.logging.info("Evaluation step %d, accuracy is %f, %s"
                            % (step, results.accuracy, time.ctime()))
            accuracies += results.accuracy
            step += 1
        except tf.errors.OutOfRangeError:
            avg_accuracy = accuracies / step
            tf.logging.info("After %d steps of evaluation, accuracy is %f <time:%f>"
                            % (step, avg_accuracy, time.time()-start_time))
            tf.logging.info("Finish evaluating")
            summary_writer.close()
            break
