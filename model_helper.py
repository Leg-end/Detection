"""Utility functions for building models(concern to model's construction)"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
from utils import proposal_util
from utils import anchor_util
from utils import misc_util as misc
from dataset import iterator_wrapper


__all__ = ["get_initializer", "get_device_str", "generate_img_anchors",
           "restore_pre_model", "reshape_for_forward_pairs",
           "create_or_load_model", "load_model", "crop_resize_pooling",
           "spatial_pyramid_pooling", "sample_rois_from_anchors",
           "create_train_model", "create_infer_model", "create_eval_model"]


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(
            seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


def generate_img_anchors(images_info, feat_stride=16,
                         ratios=tf.constant([0.5, 1.0, 2.0]),
                         scales=tf.multiply(tf.range(3, 6), 2)):
    with tf.name_scope("generate_img_anchors"):
        height = tf.to_int32(tf.ceil(tf.divide(tf.to_float(images_info[0]), tf.to_float(feat_stride))))
        width = tf.to_int32(tf.ceil(tf.divide(tf.to_float(images_info[1]), tf.to_float(feat_stride))))
        # When using padded_batch so that can we use height[0], width[0] directly,
        # or will cause different num of anchors in each batch
        anchors, all_count = anchor_util.generate_image_anchors(height, width, ratios=ratios, scales=scales)
        # todo Tile up to [batch size, all_count, 4] when set batch_size greater than 1
        # anchors = tf.tile(tf.expand_dims(anchors, dim=0), multiples=[tf.shape(images_info)[0], 1, 1])
    return anchors, all_count


def reshape_for_forward_pairs(inputs, num_dim, name="reshape_forward_pairs"):
    """
    First, separating inputs by its channel, the result that can be imaged as a
    cubic whose depth is inputs' channel, then divide the cubic into [num_dim]
    parts in its channel dimension, then next step, we keep the batch-dim and
    changed channel-dim still, maintain one of the size-dim while changing the
    other one(e.g. maintain weight, change height by pile them bottom-up to each other in height-dim),
    and aligns each element from each part in channel-dim while merging them,
    that will force its channel fixed in [num_dim]. Finally, switch it back to its original shape form.
    We will end up with elements in changed channel used to next to each other after num_dim channels.
    -------------------------------------------------->
    Following is a example of all procedures:
    Original inputs: shape=[1, 2, 2, 4] imaged as 4 slices in channel-dim of a cubic
    [[1, 2],     [[5, 6],     [[9, 10],      [[13, 14],
     [3, 4]] C0   [7, 8]] C1   [11, 12]] C2   [15, 16]] C3
    Real elements in inputs in channel axis
    C-axis:[[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]
    Set num_dim as 2, so we merge C0, C1 as a channel, so as C2, C3, and we need reshape size and
    check from channel(transpose) axis, so we get:
    [[1, 2],               [[9, 10],
     [3, 4], old C0         [11, 12], old C2
     [5, 6], new C = C0     [13, 14], new C = C1
     [7, 8]] old C1         [15, 16]] old C3
    C-axis:[[1, 9],...., [5, 13],...]
    That validate that 'Elements in changed channel used to next to each other after num_dim channels.'
    PS: When ignoring the batch dimension, we can treat 'transpose' as a action to see a cubic from specific side
    e.g. When inputs shape formed as [H, W, C], it's likes see a cubic from [H, W] surface,
    the rest can be done in the same manner.
    :param inputs: 4-D Tensor has dim as [batch, height, width, channel]
    :param num_dim: The integer value to fix channel
    :param name: op name
    :return: A reshaped Tensor which has a fixed channel specified by [num_dim]
    """
    with tf.name_scope(name):
        shape = tf.shape(inputs)
        outputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        # Force it to have 2 channels
        outputs = tf.reshape(outputs, shape=tf.concat(
            values=[[1, num_dim, -1], [shape[2]]], axis=0))
        # Switch it back
        outputs = tf.transpose(outputs, perm=[0, 2, 3, 1])
    return outputs


def sample_rois_from_anchors(cls_scores,
                             bbox_deltas,
                             im_info,
                             anchors,
                             count,
                             name="sample_rois"):
    with tf.name_scope(name):
        rois, scores = proposal_util.sample_regions(cls_scores, bbox_deltas, im_info,
                                                    anchors, count)
    return rois, scores


def pack_anchor_info(im_info,
                     anchors,
                     ori_anchor_count,
                     bbox_targets,
                     anchor_scores):
    """
    Fill bbox label and class_label(positive and negative, self-supervise) into anchor_info
    :param im_info: image property:[height, width, channel]
    :param anchors: all anchors on image
    :param ori_anchor_count: original anchor's count
    :param bbox_targets: target bbox
    :param anchor_scores: calculated by rpn's classification layer, scores of (anchors)bboxes
    :return: a dict with execution info inside and anchor_labels
    """
    with tf.name_scope("pack_anchor_info"):
        class_labels, bbox_targets, in_weights, out_weights = anchor_util.generate_anchor_targets(
            anchor_scores, anchors, bbox_targets, im_info, ori_anchor_count)
        anchor_info = dict()
        misc.append_params(anchor_info,
                           bbox_labels=bbox_targets, class_labels=tf.to_int32(class_labels),
                           out_weights=out_weights, in_weights=in_weights)
    return anchor_info


def pack_proposal_info(rpn_labels,
                       rois,
                       bbox_scores,
                       bbox_targets,
                       num_class):
    with tf.name_scope("pack_proposal_info"):
        with tf.control_dependencies([rpn_labels]):
            labels, rois, roi_scores, bbox_targets, in_weights, out_weights = proposal_util.generate_proposal_target(
                rois, bbox_scores, bbox_targets, num_class)
        proposal_info = dict()
        misc.append_params(proposal_info, class_labels=tf.to_int32(labels),
                           bbox_labels=bbox_targets, rois=rois,
                           in_weights=in_weights, out_weights=out_weights)
    return proposal_info, rois, roi_scores


def spatial_pyramid_pooling(bin_size_list, inputs, rois, feat_stride, padding="VALID"):
    """
    Spatial pyramid pooling layer from SPP-Net
    :param bin_size_list: list of int
    Specify a pyramid level of multi-size bins, each number in list specifies a nxn bins which is
    a part of fixed output size requirement after pooling. For example [1,2,4] would be 3
    regions with 1x1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    :param inputs: output from final convolution-layer
    :param rois: batch indices, regions of interest on images
    :param feat_stride: convolution stride on last convolutional layer
    :param padding: default same
    :return: 2-D tensor of shape [batch, vector_size] which has same type as inputs and whose
    length fit the input size of next fc
    """
    # Map images' rois to feature maps
    bboxes = tf.stop_gradient(proposal_util.map_rois_to_feature(
        tf.shape(inputs), rois[:, 1:], feat_stride))
    outputs = []
    # Crop all the rois in feature maps
    crops = tf.image.crop_to_bounding_box(inputs, bboxes)
    for crop in crops:
        shape = tf.shape(crop)
        roi_pools = []
        for bin_size in bin_size_list:
            win_h = shape[1] / bin_size
            win_w = shape[2] / bin_size
            pool_size = [1, tf.to_int32(tf.ceil(win_h)), tf.to_int32(tf.ceil(win_w)), 1]
            pool_stride = [1, tf.to_int32(tf.floor(win_h)), tf.to_int32(tf.floor(win_w)), 1]
            # Original max_pooling function does not support tensor-like pool_size and pool_stride
            # One solution is that max_pooling can be replaced by reduce_max after some transformation
            # on inputs(https://github.com/Sarthak-02/keras-spp/blob/master/SpatialPyramidPooling.py),
            # another solution is reconstructing max_pooling, already done by
            # [https://github.com/yongtang, https://github.com/tensorflow/tensorflow/pull/11875]
            results = gen_nn_ops.max_pool_v2(inputs, pool_size, pool_stride, padding=padding)
            roi_pools = tf.concat(roi_pools.append(tf.layers.flatten(results)), axis=1)
        outputs.append(roi_pools)
    return tf.concat(outputs, axis=1)


def crop_resize_pooling(inputs, rois, feat_stride, pool_size,
                        max_pool=True, padding='SAME', flatten=True):
    """
    Crop regions from feature map according to rois, then resize them to
    the same size, this will get same size of pooling feature after pooling
    :param inputs: output from final convolution-layer
    :param rois: batch indices, regions of interest on images
    :param feat_stride: convolution stride on last convolutional layer
    :param pool_size: unify size after pooling
    :param padding: default same
    :param max_pool: If true ,do max pooling, specify in resnet
    :param flatten: If true, do flatten, specify in resnet
    :return: 2-D tensor of shape [batch, vector_size] which has same type as inputs and whose
    length fit the input size of next fc
    """
    shape = tf.shape(inputs)
    batch_inds = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1]), axis=1)
    bboxes = tf.stop_gradient(proposal_util.map_rois_to_feature(shape, rois[:, 1:], feat_stride))
    if max_pool:
        pool_size = pool_size*2
        crops = tf.image.crop_and_resize(inputs, bboxes, tf.to_int32(batch_inds),
                                        [pool_size, pool_size], name="crops")
        crops = tf.nn.max_pool(crops, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                               padding=padding, name="max_pool")
    else:
        crops = tf.image.crop_and_resize(inputs, bboxes, tf.to_int32(batch_inds),
                                         [pool_size, pool_size], name="crops")
    if flatten:
        return tf.layers.flatten(crops)
    else:
        return crops


class TrainModel(collections.namedtuple("TrainModel",
                                        ("graph", "model", "data_wrapper"))):
    pass


def create_train_model(model_creator,
                       hparams,
                       scope=None):
    dataset_dir = os.path.join(hparams.data_dir, "eval")  # test
    filenames = tf.gfile.ListDirectory(dataset_dir)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(dataset_dir, filenames[i])
    graph = tf.Graph()
    with graph.as_default(), tf.container(scope or "train"):
        with tf.name_scope("Data"):
            with tf.device("/cpu:0"):
                reverse_cate_table = misc.create_reverse_category_table(hparams.category_file)
                raw_dataset = tf.data.TFRecordDataset(filenames)
                data_wrapper = iterator_wrapper.get_iterator_wrapper(
                    raw_dataset, hparams.img_batch_size)
        model = model_creator(
            hparams=hparams,
            reverse_cate_table=reverse_cate_table,
            data_wrapper=data_wrapper,
            scope=scope)
    return TrainModel(
        graph=graph,
        model=model,
        data_wrapper=data_wrapper)


class InferModel(collections.namedtuple("InferModel",
                                        ("graph", "model", "data_wrapper"))):
    pass


def create_infer_model(model_creator,
                       hparams,
                       scope=None):
    dataset_dir = os.path.join(hparams.data_dir, "eval")
    filenames = tf.gfile.ListDirectory(dataset_dir)
    images = []
    for i in range(len(filenames)):
        filenames[i] = os.path.join(dataset_dir, filenames[i])
        with tf.gfile.GFile(filenames[i], "rb") as f:
            images.append(f.read())
    graph = tf.Graph()
    with graph.as_default(), tf.container(scope or "eval"):
        with tf.name_scope("Data"):
            with tf.device("/cpu:0"):
                reverse_cate_table = misc.create_reverse_category_table(hparams.category_file)
                raw_dataset = tf.data.Dataset()
                raw_dataset = raw_dataset.from_tensor_slices(images)
                data_wrapper = iterator_wrapper.get_infer_iterator_wrapper(
                    raw_dataset, hparams.img_batch_size)
        model = model_creator(
            hparams=hparams,
            reverse_cate_table=reverse_cate_table,
            data_wrapper=data_wrapper,
            scope=scope)
        return InferModel(
            graph=graph,
            model=model,
            data_wrapper=data_wrapper)


class EvalModel(collections.namedtuple("EvalModel",
                                       ("graph", "model", "data_wrapper"))):
    pass


def create_eval_model(model_creator,
                      hparams,
                      scope=None):
    dataset_dir = os.path.join(hparams.data_dir, "eval")
    filenames = tf.gfile.ListDirectory(dataset_dir)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(dataset_dir, filenames[i])
    graph = tf.Graph()
    with graph.as_default(), tf.container(scope or "eval"):
        with tf.name_scope("Data"):
            with tf.device("/cpu:0"):
                reverse_cate_table = misc.create_reverse_category_table(hparams.category_file)
                raw_dataset = tf.data.TFRecordDataset(filenames)
                data_wrapper = iterator_wrapper.get_iterator_wrapper(
                    raw_dataset, hparams.img_batch_size)
        model = model_creator(
            hparams=hparams,
            reverse_cate_table=reverse_cate_table,
            data_wrapper=data_wrapper,
            scope=scope)
        return EvalModel(
            graph=graph,
            model=model,
            data_wrapper=data_wrapper)


def gradient_clip(gradients, max_gradient_norm):
    with tf.device("/CPU:0"):
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(
            gradients, max_gradient_norm)
        gradient_norm_summaries = list()
        gradient_norm_summaries.append(tf.summary.scalar("grad_norm", gradient_norm))
        gradient_norm_summaries.append(
            tf.summary.scalar("TRAIN/clipped_gradient", tf.global_norm(clipped_gradients)))
    return clipped_gradients, gradient_norm_summaries, gradient_norm


def restore_pre_model(model_scope, ckpt_file):
    """
    Create an op that restore pre-trained model from
    specified checkpoint file
    :param model_scope: the variable scope where variables were stored
    :param ckpt_file: checkpoint file
    :return: restore variables operation
    """
    print("Set up pre-trained model's restore op")
    variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)
    saver = tf.train.Saver(variables)
    restore_fn = tf.no_op()
    if ckpt_file and misc.check_file_existence(ckpt_file):
        def restore_fn(sess):
            tf.logging.info(
                "Restore variables form '%s'" % ckpt_file)
            saver.restore(sess, ckpt_file)
    return restore_fn


def print_variables_in_ckpt(ckpt_path):
    """Print a list of variables in a checkpoint together with their shapes."""
    print("# Variables in ckpt %s" % ckpt_path)
    reader = tf.train.NewCheckpointReader(ckpt_path)
    variable_map = reader.get_variable_to_shape_map()
    for key in sorted(variable_map.keys()):
        print("  %s: %s" % (key, variable_map[key]))


def load_model(model,
               ckpt_path,
               session,
               init_op=None,
               name="restore"):
    start_time = time.time()
    ckpt = tf.train.get_checkpoint_state(
        os.path.dirname(ckpt_path))
    if ckpt and ckpt.model_checkpoint_path:
        if init_op:
            print("Restore a pre-trained model")
            init_op(session)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Can't load checkpoint")
    print(
        "  loaded %s model parameters from %s, time %.2fs" %
        (name, ckpt_path, time.time() - start_time))
    return model


def create_or_load_model(model,
                         model_dir,
                         session,
                         init_op=None):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, init_op, "restore")
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("  created %s model with fresh parameters, time %.2fs" %
              (model.scope, time.time() - start_time))
    global_step = model.global_step.eval(session=session)
    print("global_step=%d" % global_step)
    return model, global_step
