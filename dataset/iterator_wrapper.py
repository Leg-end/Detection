"""For loading data into Faster-RCNN models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import tensorflow as tf

__all__ = ["get_infer_iterator_wrapper", "get_iterator_wrapper",
           "process_image"]


class DataWrapper(
    collections.namedtuple("DataWrapper",
                           ("initializer",
                            "images_data",
                            "images_size",
                            "bbox_locations"))):
    pass


def process_image(image, image_format="jpeg"):
    if image_format == "jpeg":
        image = tf.image.decode_jpeg(image)
    elif image_format == "png":
        image = tf.image.decode_png(image)
    return tf.image.convert_image_dtype(image, dtype=tf.float32)


def get_iterator_wrapper(src_dataset,
                         batch_size,
                         image_format="jpeg",
                         num_parallel_calls=4,
                         output_buffer_size=10,
                         image_feature="image/data",
                         size_feature="image/size",
                         coord_xl_feature="bbox/locations/x_l",
                         coord_yt_feature="bbox/locations/y_t",
                         coord_xr_feature="bbox/locations/x_r",
                         coord_yb_feature="bbox/locations/y_b",
                         categories_feature="bbox/categories"):
    def get_feature_description(image_feature_, size_feature_,
                                coord_xl_feature_, coord_yt_feature_,
                                coord_xr_feature_, coord_yb_feature_,
                                categories_feature_):
        context_features_proto = {
            image_feature_: tf.FixedLenFeature([], dtype=tf.string)
        }
        sequence_features_proto = {
            coord_xl_feature_: tf.FixedLenSequenceFeature([], dtype=tf.float32),
            coord_yt_feature_: tf.FixedLenSequenceFeature([], dtype=tf.float32),
            coord_xr_feature_: tf.FixedLenSequenceFeature([], dtype=tf.float32),
            coord_yb_feature_: tf.FixedLenSequenceFeature([], dtype=tf.float32),
            categories_feature_: tf.FixedLenSequenceFeature([], dtype=tf.int64),
            size_feature_: tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }
        return context_features_proto, sequence_features_proto
    context_features, sequence_features = get_feature_description(
        image_feature, size_feature,
        coord_xl_feature, coord_yt_feature,
        coord_xr_feature, coord_yb_feature, categories_feature)
    # fetch context and sequence from record
    src_dataset = src_dataset.map(
        lambda x: (tf.parse_single_sequence_example(
            x, context_features, sequence_features)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Data argument or normalize?
    # fetch image data, height, width, bbox category, overlap, coordinate from record
    src_dataset = src_dataset.map(
        lambda context, sequence: (
            process_image(context[image_feature], image_format),
            tf.to_int32(sequence[size_feature]),
            tf.transpose([sequence[coord_xl_feature], sequence[coord_yt_feature],
                          sequence[coord_xr_feature], sequence[coord_yb_feature],
                          tf.to_float(sequence[categories_feature])])),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if batch_size > 1:
        print("Model only train and test in batch_size is 1")
    batch_size = 1

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
               [-1, -1, 3],
               [3],
               [-1, 5]),
            padding_values=(
                -1., 0, -1.))
    src_dataset = batching_func(src_dataset)
    # src_dataset = src_dataset.batch(batch_size)
    batched_iter = src_dataset.make_initializable_iterator()
    (img_data, img_size, bbox_locations) = batched_iter.get_next()
    # bbox_locations = tf.transpose(bbox_locations)
    return DataWrapper(
        initializer=batched_iter.initializer,
        images_data=img_data,
        images_size=tf.squeeze(img_size, axis=0),
        # Contain bbox categories in last dim
        bbox_locations=bbox_locations)


def get_infer_iterator_wrapper(src_dataset,
                               batch_size,
                               image_format="jpeg"):
    # Data argument or normalize?
    # batch
    if batch_size > 1:
        print("Model only train and test in batch_size is 1")
    batch_size = 1
    batched_dataset = src_dataset.batch(batch_size)
    batched_iter = batched_dataset.make_initializable_iterator()
    img_data, img_size = batched_iter.get_next()
    img_data = process_image(img_data, image_format)
    return DataWrapper(
        initializer=batched_iter.initializer,
        images_data=img_data,
        images_size=tf.squeeze(img_size, axis=0),
        bbox_locations=None)
