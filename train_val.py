import tensorflow as tf
import model_helper as helper
import eval
from models import base_model
from utils import misc_util as misc
import os
import time


# tf.logging.set_verbosity(tf.logging.INFO)


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


def init_stats():
    return {"step_time": 0.0, "train_loss": 0.0,
            "grad_norm": 0.0}


def before_train(loaded_train_model,
                 train_model,
                 train_sess,
                 global_step):
    """
    Prepare the info, states that will be printed during training
    Initialize iterators of dataset
    :return: train states, print info, start train time
    """
    stats = init_stats()
    info = {"speed": 0.0, "avg_train_loss": 0.0, "avg_step_time": 0.0,
            "avg_grad_norm": 0.0, "learning_rate":
                loaded_train_model.learning_rate.eval(session=train_sess)}
    start_train_time = time.time()
    tf.logging.info("# Start step %d, lr %g, %s" %
                    (global_step, info["learning_rate"], time.ctime()))
    # Initializer all of the iterators
    train_sess.run(train_model.data_wrapper.initializer)
    return stats, info, start_train_time


def process_stats(stats, info, steps_per_stats, batch_size):
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
    info["avg_train_loss"] = stats["train_loss"] / batch_size
    # todo calculate speed fps/s


def update_stats(stats, start_time, step_result):
    output_tuple = step_result
    assert isinstance(output_tuple, base_model.TrainOutputTuple)
    batch_size = output_tuple.batch_size
    stats["step_time"] += time.time() - start_time
    stats["train_loss"] += output_tuple.loss * batch_size
    stats["grad_norm"] += output_tuple.grad_norm
    return (output_tuple.global_step, output_tuple.learning_rate,
            output_tuple.summary)


def print_step_info(global_step, info, stats):
    """Print all info at the current global step."""
    tf.logging.info(
        "In step %d, total_train_loss %.2f, avg-train-loss %.2f, learning rate %g,"
        " avg-step-time %.2fs, avg-grad-norm %.2f, speed %.2f, %s" %
        (global_step, stats["train_loss"], info["avg_train_loss"], info["learning_rate"],
         info["avg_step_time"], info["avg_grad_norm"], info["speed"], time.ctime()))


def train(hparams, ckpt_dir, scope=None, target_session="", alternative=False):
    out_dir = os.path.join(hparams.base_dir, "train")
    if not misc.check_file_existence(out_dir):
        tf.gfile.MakeDirs(out_dir)
    tf.logging.info("All train relevant results will be put in %s" % out_dir)
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats
    ckpt_path = os.path.join(ckpt_dir, "model.ckpt")

    # Create model
    model_creator = get_model_creator(hparams.model_type)
    train_model = helper.create_train_model(model_creator,
                                            hparams, scope)
    config_proto = misc.get_config_proto(
        log_device_placement=False,
        allow_soft_placement=True,
        num_intra_threads=hparams.num_intra_threads,
        num_inter_threads=hparams.num_inter_threads)
    train_sess = tf.Session(
        target=target_session, config=config_proto, graph=train_model.graph)
    tf.logging.info("Create model successfully")
    with train_model.graph.as_default():
        loaded_train_model, global_step = helper.create_or_load_model(
            train_model.model, ckpt_dir, train_sess, train_model.model.restore_op)
    # Summary writer
    summary_name = "train_summary"
    summary_path = os.path.join(out_dir, summary_name)
    if not tf.gfile.Exists(summary_path):
        tf.gfile.MakeDirs(summary_path)
    summary_writer = tf.summary.FileWriter(
        summary_path, train_model.graph)
    last_stats_step = global_step
    # Training iteration
    stats, info, start_train_time = before_train(
        loaded_train_model, train_model, train_sess, global_step)
    tf.logging.info("Ready to train")
    epoch_step = 0
    num_train_steps += global_step
    detect_results = []
    while global_step < num_train_steps:
        start_time = time.time()
        try:
            tf.logging.info("Start train epoch:%d" % epoch_step)
            _, step_result, detect_result = loaded_train_model.train(train_sess)
            detect_results.append(detect_result)
            epoch_step += 1
        except tf.errors.OutOfRangeError:
            tf.logging.info("Saving epoch step %d model into checkpoint" % epoch_step)
            loaded_train_model.saver.save(
                train_sess,
                ckpt_path,
                global_step=global_step)
            # Training while evaluating alternately
            if alternative:
                eval.evaluate(hparams, scope=scope,
                              target_session=target_session,
                              global_step_=global_step,
                              ckpt_path=ckpt_path,
                              alternative=alternative)
            # Finished going through the training dataset.  Go to next epoch.
            epoch_step = 0
            tf.logging.info("# Finished an epoch, step %d."
                            % global_step)
            train_sess.run(train_model.data_wrapper.initializer)
            continue

        global_step, info["learning_rate"], step_summary = update_stats(
            stats, start_time, step_result)
        summary_writer.add_summary(step_summary, global_step)
        # Once in a while, print statistics.
        if global_step - last_stats_step >= steps_per_stats:
            tf.logging.info("In global step:%d, time to print train statistics"
                            % global_step)
            last_stats_step = global_step
            # Update info
            process_stats(stats, info, steps_per_stats, hparams.img_batch_size)
            print_step_info(global_step, info, stats)
            # Reset statistic
            stats = init_stats()
    # Done training
    tf.logging.info("Finish training, saving model into checkpoint")
    loaded_train_model.saver.save(
        train_sess,
        ckpt_path,
        global_step=global_step)
    summary_writer.close()
    misc.draw_detect_results(detect_results)
