"""Utility functions for building models(concern to model's construction)"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
from utils import proposal_util
from utils import anchor_util
from dataset import build_tfrecord
from dataset import iterator_wrapper


__all__ = ["get_initializer", "get_device_str", "generate_img_anchors"]


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


def generate_img_anchors(im_info, feat_stride=16,
                         ratios=tf.constant([0.5, 1.0, 2.0]),
                         scales=tf.multiply(tf.range(3, 6), 2)):
    height = tf.to_int32(tf.ceil(tf.divide(im_info[0], tf.to_float(feat_stride))))
    width = tf.to_int32(tf.ceil(tf.divide(im_info[1], tf.to_float(feat_stride))))
    anchors, count = anchor_util.generate_image_anchors(height, width, ratios=ratios, scales=scales)
    return anchors, count


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
    shape = tf.shape(inputs)
    with tf.name_scope(name):
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
                             strategy="nms",
                             name="rois"):
    with tf.name_scope(name):
        rois, scores = proposal_util.sample_regions(cls_scores, bbox_deltas, im_info,
                                                    anchors, count, strategy=strategy)
    return rois, scores


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
    :return: 3-D tensor of shape [batch, num_roi, out_size] which has same type as inputs and whose
    length fit the input size of next fc
    """
    # Map images' rois to feature maps

    bboxes = proposal_util.map_rois_to_feature(tf.shape(inputs), rois[:, 1:], feat_stride)
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
            # result = tf.layers.max_pooling2d(inputs, pool_size, pool_stride, padding=padding)
            # Original max_pooling function does not support tensor-like pool_size and pool_stride
            # One solution is that max_pooling can be replaced by reduce_max after some transformation
            # for inputs(https://github.com/Sarthak-02/keras-spp/blob/master/SpatialPyramidPooling.py),
            # another solution is reconstructing max_pooling, already done by
            # [https://github.com/yongtang, https://github.com/tensorflow/tensorflow/pull/11875]
            results = gen_nn_ops.max_pool_v2(inputs, pool_size, pool_stride, padding=padding)
            roi_pools = tf.expand_dims(tf.concat(roi_pools.append(tf.reshape(
                results, shape=[shape[0], -1])), axis=1), axis=1)
        outputs.append(roi_pools)
    return tf.concat(outputs, axis=1)


def crop_resize_pooling(inputs, rois, feat_stride, unify_size, padding='SAME'):
    """
    Crop regions from feature map according to rois, then resize them to
    the same size, this will get same size of pooling feature after pooling
    :param inputs: output from final convolution-layer
    :param rois: batch indices, regions of interest on images
    :param feat_stride: convolution stride on last convolutional layer
    :param unify_size: unify size for resizing
    :param padding: default same
    :return: 3-D tensor of shape [batch, num_roi, out_size] which has same type as inputs and whose
    length fit the input size of next fc
    """
    shape = tf.shape(inputs)
    batch_inds = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1]), axis=1)
    bboxes = tf.stop_gradient(proposal_util.map_rois_to_feature(shape, rois[:, 1:], feat_stride))
    crops = tf.image.crop_and_resize(inputs, bboxes, tf.to_int32(batch_inds), unify_size)
    return tf.reshape(tf.nn.max_pool(crops, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                                     padding=padding, name="max_pool"),
                      shape=[shape[0], tf.shape(rois)[0], -1])


class TrainModel(collections.namedtuple("TrainModel",
                                        ("graph", "model", "data_wrapper"))):
    pass


def create_train_model(model_creator,
                       hparams,
                       scope=None):
    dataset_dir = os.path.join(build_tfrecord.FLAGS.output_dir, "train")
    filenames = tf.gfile.ListDirectory(dataset_dir)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(dataset_dir, filenames[i])
    graph = tf.Graph()
    with graph.as_default(), tf.container(scope or "train"):
        with tf.name_scope("DataSet"):
            raw_dataset = tf.data.TFRecordDataset(filenames)
            data_wrapper = iterator_wrapper.get_iterator_wrapper(
                raw_dataset, hparams.batch_size)
        model = model_creator(
            haparms=hparams,
            wrapper=data_wrapper,
            scope=scope)
    return TrainModel(
        graph=graph,
        model=model,
        data_wrapper=data_wrapper)


class InferModel(collections.namedtuple("InferModel",
                                        ("graph", "model", "iterator"))):
    pass


def create_infer_model():
    pass


class EvalModel(collections.namedtuple("EvalModel",
                                       ("graph", "model", "iterator"))):
    pass


def create_eval_model():
    pass
