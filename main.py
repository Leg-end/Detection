import argparse
import utils.misc_util as misc
import train_val
import inference
import eval
import tensorflow as tf
import random
import os

tf.logging.set_verbosity(tf.logging.INFO)

__all__ = ["add_arguments", "create_hparams",
           "create_or_load_hparams",
           "ensure_compatible_hparams"]


def add_arguments(parser):
    # Set new type bool
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data
    parser.add_argument("--data_dir", default="D:/Detection/dataset/COCO_tfrecord/",
                        help="Data set dir")
    parser.add_argument("--category_file", default="D:/Detection/dataset/COCO/id_to_category.txt",
                        help="File that store category name order by category id")
    parser.add_argument("--base_dir", default="D:/Detection/",
                        help="Base directory")
    # Checkpoint config
    parser.add_argument("--pre_ckpt_dir", default="D:/Detection/pre_ckpt/",
                        help="Pre-trained model's checkpoint file path")
    parser.add_argument("--ckpt_storage", type=int, default=8,
                        help="Max number of storing checkpoint")

    # Constrain & Threshold & Fraction
    parser.add_argument("--pixel_mean", type=float, default=[[[102.9801, 115.9465, 122.7717]]],
                        help="""Pixel mean values (BGR order) as a (1, 1, 3) array
                                 Use the same pixel mean for all networks even though it's not exactly what
                                 they were trained with""")
    parser.add_argument("--top_limitation", type=int, default=10,
                        help="The top n selected when strategy is top")
    parser.add_argument("--pre_nms_limitation", type=int, default=12000,
                        help="Number of top scoring boxes to keep before apply NMS to RPN proposals")
    parser.add_argument("--post_nms_limitation", type=int, default=2000,
                        help="Number of top scoring boxes to keep after applying NMS to RPN proposals")
    parser.add_argument("--nms_thresh", type=float, default=0.7,
                        help="NMS threshold used on RPN proposals")
    parser.add_argument("--unify_size", type=int, default=7,
                        help="Unify size for resizing when specify sample_mode as top")
    parser.add_argument("--anchor_ratios", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                        help="Ratio list for changing anchor's ratio")
    parser.add_argument("--anchor_scales", type=float, nargs="+", default=[6., 8., 10.],
                        help="Scale list for changing anchor's scale")
    parser.add_argument("--rpn_fg_fraction", type=float, default=0.5,
                        help="Max number of foreground examples")
    parser.add_argument("--rpn_batch_size", type=int, default=64,  # 256
                        help="Total number of examples")
    parser.add_argument("--fg_thresh", type=float, default=0.5,
                        help="Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)")
    parser.add_argument("--bg_thresh_range", type=float, default=[0.1, 0.5], nargs=2,
                        help="Overlap threshold for a ROI to be considered background"
                             " (class = 0 if overlap in [LO, HI)")
    parser.add_argument("--roi_batch_size", type=int, default=64,  # 128
                        help="Mini batch size (number of regions of interest [ROIs])")
    parser.add_argument("--img_batch_size", type=int, default=1,
                        help="Mini batch size (number of image in each batch)")
    parser.add_argument("--bbox_targets_pre_norm", type="bool", nargs="?", const=True, default=True,
                        help="Normalize the targets using 'precomputed' (or made up) means and stddevs")
    parser.add_argument("--bbox_norm_means", type=float, default=[0., 0., 0., 0.], nargs=4,
                        help="Bounding box locations' normalization means")
    parser.add_argument("--bbox_norm_stddevs", type=float, default=[0.1, 0.1, 0.2, 0.2], nargs=4,
                        help="Bounding box locations' normalization standard deviations")
    parser.add_argument("--bbox_in_weights", type=float, default=[1., 1., 1., 1.], nargs=4,
                        help="Bounding box inside weights")

    # Init config
    parser.add_argument("--init_op", default="uniform", choices=["uniform", "glorot_normal", "glorot_uniform"],
                        help="Specify initializer of weights")
    parser.add_argument("--ran_seed", default=12345, type=int,
                        help="Random seed for initialize that can be redo")
    parser.add_argument("--init_weight", type=float, default=0.001,
                        help="Initial weight for trainable variables")
    parser.add_argument("--bbox_init_op", default="glorot_normal", choices=[
        "uniform", "glorot_normal", "glorot_uniform"],
                        help="Specify initializer of regression weights")
    parser.add_argument("--bbox_ran_seed", default=12345, type=int,
                        help="Random seed for initialize that can be redo")
    parser.add_argument("--bbox_init_weight", type=float, default=0.001,
                        help="Initial weight for regression trainable variables")

    # Training config
    parser.add_argument("--num_train_steps", type=int, default=400,
                        help="Total train steps")
    parser.add_argument("--momentum_factor", type=float, default=0.9,
                        help="Factor use for momentum optimizer")
    parser.add_argument("--tunable", type="bool", nargs="?", const=False, default=False,
                        help="Whether to fine tune")
    parser.add_argument("--max_grad_norm", type=int, default=5,
                        help="Limitation value of max gradient")
    parser.add_argument("--steps_per_stats", type=int, default=20,
                        help="Count of step for each states printing when training")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Number of Step for warm up strategy in learning_rate")
    parser.add_argument("--warmup_scheme", choices=['t2t'], default='t2t',
                        help="Scheme of warm up strategy")
    parser.add_argument("--tune_rate", type=float, default=0.0001,
                        help="Learning rate of fine tune")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate without fine tune")
    parser.add_argument("--decay_scheme", choices=["luong5", "luong10", "luong234", ""],
                        help="Scheme of learning rate decay")
    parser.add_argument("--rpn_in_weights", type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0],
                        help="Mask-like factor that only keep positive ones to regression loss calculation")
    parser.add_argument("--rpn_positive_weight", type=float, default=-1,
                        help="""Give the positive RPN examples weight of p * 1 / {num positives} and give 
                                    negatives a weight of (1 - p) Set to -1.0 to use uniform example weighting""")
    parser.add_argument("--rpn_sigma", type=float, default=3.0,
                        help="Multiply factor of calculating regression loss in rpn")
    parser.add_argument("--weight_decay_factor", type=float, default=0.0001,
                        help="Weight decay factor configuration for weight regularizer")
    parser.add_argument("--bias_decay", type="bool", default=False, const=False, nargs="?",
                        help="Whether to do bias decay using weight decay factor")

    # Net Structure
    parser.add_argument("--rpn_channel", type=int, default=64,   # 512
                        help="Number of kernel for first convolution layer in rpn")
    parser.add_argument("--pooling_mode", default="crop", choices=["pyramid", "crop"],
                        help="The strategy for pooling rois")
    parser.add_argument("--sample_mode", default="nms", choices=["top", "nms"],
                        help="The strategy for sampling and filtering rois from anchors")
    parser.add_argument("--bin_size_list", type=int, nargs="+", default=[1, 2, 4],
                        help="List of bin side value when specifying pooling mode as pyramid")
    parser.add_argument("--model_type", default="VGG_16", choices=["InceptionV3", "VGG16", "VGG_16", "ResNetV1"],
                        help="Pre-trained model type")
    parser.add_argument("--max_pool", type="bool", default=True, const=True, nargs="?",
                        help="Whether to do max pooling, only use when model_type is ResNetV1")
    parser.add_argument("--resnet_fixed_blocks", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Number of fixed blocks during training, by default the first "
                             "of all 4 blocks is fixed Range: 0 (none) to 3 (all)")
    parser.add_argument("--flatten", type="bool", default=True, const=True, nargs="?",
                        help="Whether to do flatten, when model type is ResNetV1"
                             "flatten is false")
    parser.add_argument("--forward_rcnn", type="bool", default=False, const=False, nargs="?",
                        help="Whether to train with rcnn")

    # Misc
    parser.add_argument("--mode", choices=["train", "infer", "eval"], default="train",
                        help="Specify execution mode")
    parser.add_argument("--use_tgt", type="bool", nargs="?", const=False, default=False,
                        help="Whether to add ground truth boxes to the pool when sampling regions")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPU that is available")
    parser.add_argument("--num_class", type=int, default=80,
                        help="Number of class for classification")
    parser.add_argument("--num_intra_threads", type=int, default=0,
                        help="Thread number configuration for session")
    parser.add_argument("--num_inter_threads", type=int, default=0,
                        help="Thread number configuration for session")
    parser.add_argument("--log_device_placement", type="bool", nargs="?", const=False, default=False,
                        help="Whether to log device placement")
    parser.add_argument("--feat_stride", type=int, default=16,
                        help="The length of side of region for sampling roi in image")


"""def create_hparams(flags):
    # import hyper_params
    # assert isinstance(flags, hyper_params.StandardHParams)
    return tf.contrib.training.HParams(
        # Data
        data_dir=flags.data_dir,
        category_file=flags.category_file,
        base_dir=flags.base_dir,

        # Checkpoint config
        pre_ckpt=flags.pre_ckpt,
        ckpt_storage=flags.ckpt_storage,

        # Constrain & Threshold & Fraction
        im_info=flags.im_info,
        pixel_mean=flags.pixel_mean,
        top_limitation=flags.top_limitation,
        pre_nms_limitation=flags.pre_nms_limitation,
        post_nms_limitation=flags.post_nms_limitation,
        nms_thresh=flags.nms_thresh,
        unify_size=flags.unify_size,
        anchor_ratios=flags.anchor_ratios,
        anchor_scales=flags.anchor_scales,
        rpn_fg_fraction=flags.rpn_fg_fraction,
        rpn_batch_size=flags.rpn_batch_size,
        fg_thresh=flags.fg_thresh,
        bg_thresh_range=flags.bg_thresh_range,
        roi_batch_size=flags.roi_batch_size,
        img_batch_size=flags.img_batch_size,
        bbox_targets_pre_norm=flags.bbox_targets_pre_norm,
        bbox_norm_means=flags.bbox_norm_means,
        bbox_norm_stddevs=flags.bbox_norm_stddevs,
        bbox_in_weights=flags.bbox_in_weights,

        # Init config
        init_op=flags.init_op,
        ran_seed=flags.ran_seed,
        init_weight=flags.init_weight,
        bbox_init_op=flags.bbox_init_op,
        bbox_ran_seed=flags.bbox_ran_seed,
        bbox_init_weight=flags.bbox_init_weight,

        # Training config
        num_train_steps=flags.num_train_steps,
        optimizer=flags.optimizer,
        tunable=flags.tunable,
        max_grad_num=flags.max_grad_num,
        steps_per_stats=flags.steps_per_stats,
        warmup_steps=flags.warmup_steps,
        warmup_scheme=flags.warmup_scheme,
        tune_rate=flags.tune_rate,
        learning_rate=flags.learning_rate,
        decay_scheme=flags.decay_scheme,
        rpn_in_weights=flags.rpn_in_weights,
        rpn_positive_weight=flags.rpn_positive_weight,
        rpn_sigma=flags.rpn_sigma,

        # Net Structure
        rpn_channel=flags.rpn_channel,
        pooling_mode=flags.pooling_mode,
        sample_mode=flags.sample_mode,
        bin_size_list=flags.bin_size_list,

        # Misc
        mode=flags.mode,
        use_tgt=flags.use_tgt,
        num_gpus=flags.num_gpus,
        num_class=flags.num_class,
        num_intra_threads=flags.num_intra_threads,
        num_inter_threads=flags.num_inter_threads,
        log_device_placement=flags.log_device_placement,
        feat_stride=flags.feat_stride
    )"""


def create_hparams(flag_dict):
    from tensorlib import training
    return training.HParams(**flag_dict)


def _add_argument(hparams, key, value, update=True):
    if hasattr(hparams, key):
        if update:
            setattr(hparams, key, value)
    else:
        hparams.add_hparam(key, value)


def ensure_compatible_hparams(hparams, default_hparams, path=""):
    """
    Make sure the loaded hparams is compatible with new changes
    :param hparams: loaded hparams
    :param default_hparams: standard hparams
    :param path: hparams' store path
    :return: a compatible hparams
    """
    default_hparams = misc.parse_standard_hparams(
        default_hparams, path)
    default_config = default_hparams.values()
    config = hparams.values()
    for key in default_config:
        if key not in config:
            hparams.add_hparam(key, default_config[key])
    return hparams


def create_or_load_hparams(default_hparams,
                           path,
                           default_path=None,
                           save_hparams=True):
    extra_hparams = misc.load_hparams(path)
    if not extra_hparams:
        hparams = default_hparams
    else:
        hparams = ensure_compatible_hparams(extra_hparams,
                                            default_hparams,
                                            path=default_path)
    if save_hparams:
        misc.save_hparams(path, extra_hparams)
    misc.print_hparams(hparams)
    return hparams


def pre_fill_params_into_utils(hparams):
    from utils import anchor_util, proposal_util
    anchor_util.fill_params(rpn_batch_size=hparams.rpn_batch_size,
                            rpn_fg_fraction=hparams.rpn_fg_fraction,
                            rpn_in_weights=hparams.rpn_in_weights,
                            rpn_positive_weight=hparams.rpn_positive_weight)
    proposal_util.fill_params(fg_thresh=hparams.fg_thresh,
                              bg_thresh_range=hparams.bg_thresh_range,
                              roi_batch_size=hparams.roi_batch_size,
                              image_batch_size=hparams.img_batch_size,
                              bbox_in_weights=hparams.bbox_in_weights,
                              bbox_norm_means=hparams.bbox_norm_means,
                              bbox_norm_stddevs=hparams.bbox_norm_stddevs,
                              bbox_pre_norm=hparams.bbox_targets_pre_norm,
                              post_nms_limit=hparams.post_nms_limitation,
                              nms_thresh=hparams.nms_thresh,
                              top_limit=hparams.top_limitation,
                              sample_mode=hparams.sample_mode,
                              use_tgt=hparams.use_tgt)


def run_main(hparams, train_func, infer_func, eval_func, target_session=""):
    # GPU device
    print("Devices visible to Tensorflow: %s" % repr(tf.Session().list_devices()))

    # Random
    ran_seed = hparams.ran_seed
    if ran_seed is not None and ran_seed > 0:
        print("Set random seed to %d" % ran_seed)
        random.seed(ran_seed)

    # Check directory
    ckpt_dir = os.path.join(hparams.base_dir, "ckpt")
    hp_dir = os.path.join(hparams.base_dir, "hparams")
    if hparams.base_dir and not misc.check_file_existence(hparams.base_dir):
        print("Creating output directory %s ..." % hparams.base_dir)
        tf.gfile.MakeDirs(hparams.base_dir)
    if ckpt_dir and not misc.check_file_existence(ckpt_dir):
        print("Creating checkpoint directory %s ..." % ckpt_dir)
        tf.gfile.MakeDirs(ckpt_dir)
    if hp_dir and not misc.check_file_existence(hp_dir):
        print("Creating hyper parameter directory %s ..." % hp_dir)
        tf.gfile.MakeDirs(hp_dir)
    # Load hparams
    """hparams_file = os.path.join(hp_dir, "hparams.json")
    def_hparams_file = os.path.join(hp_dir, "default_hparams.json")
    if misc.check_file_existence(hparams_file):
        hparams = create_or_load_hparams(
            default_hparams, hparams_file,
            def_hparams_file, save_hparams=False)
    else:
        hparams = default_hparams"""
    pre_fill_params_into_utils(hparams)
    if hparams.model_type != "ResNetV1":
        hparams.max_pool = True
    else:
        hparams.flatten = False
    if hparams.mode == "train":
        tf.logging.info("Execute training")
        train_func(hparams, ckpt_dir, target_session=target_session)
    elif hparams.mode == "infer":
        tf.logging.info("Execute inference")
        infer_func(hparams, ckpt_dir, target_session=target_session)
    elif hparams.mode == "eval":
        tf.logging.info("Execute evaluating")
        eval_func(hparams, ckpt_path=ckpt_dir, target_session=target_session)


if __name__ == "__main__":
    param_parser = argparse.ArgumentParser()
    add_arguments(param_parser)
    flags = param_parser.parse_args()
    flags = vars(flags)
    def_hparams = create_hparams(flags)
    train_fn = train_val.train
    infer_fn = inference.infer
    eval_fn = eval.evaluate
    run_main(def_hparams, train_fn, infer_fn, eval_fn)
