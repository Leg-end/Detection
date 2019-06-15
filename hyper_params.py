from collections import namedtuple


class StandardHParams(namedtuple("StandardHParams",
                                 ("data_dir", "category_file", "base_dir", "pre_ckpt",
                                  "ckpt_storage", "im_info", "pixel_mean", "top_limitation",
                                  "pre_nms_limitation", "post_nms_limitation", "nms_thresh",
                                  "unify_size", "anchor_ratios", "anchor_scales", "rpn_fg_fraction",
                                  "rpn_batch_size", "fg_thresh", "bg_thresh_range", "roi_batch_size",
                                  "img_batch_size", "bbox_targets_pre_norm", "bbox_norm_means", "bbox_norm_stddevs",
                                  "bbox_in_weights", "init_op", "ran_seed", "init_weight", "bbox_init_op",
                                  "bbox_ran_seed", "bbox_init_weight", "num_train_steps", "optimizer",
                                  "tunable", "max_grad_num", "steps_per_stats", "warmup_steps", "warmup_scheme",
                                  "tune_rate", "learning_rate", "decay_scheme", "rpn_in_weights",
                                  "rpn_positive_weight", "rpn_sigma", "rpn_channel", "pooling_mode",
                                  "sample_mode", "bin_size_list", "mode", "use_tgt", "num_gpus", "num_class",
                                  "num_intra_threads", "num_inter_threads", "log_device_placement", "feat_stride"))):
    """Store necessary params"""
    pass
