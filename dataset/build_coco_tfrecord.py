"""This is a base library for translate MCOCO-dataset into TFRecord"""
import tensorflow as tf
import json
import os
import numpy as np
from collections import namedtuple
import random
import threading
from datetime import datetime
import sys

__all__ = [""]


tf.flags.DEFINE_string("train_image_dir", "D:/dataset/Images/COCO/train2014/",
                       "Training image directory.")
tf.flags.DEFINE_string("val_image_dir", "D:/dataset/Images/COCO/val2014/",
                       "Validation image directory.")
tf.flags.DEFINE_string("category_file", "D:/Detection/dataset/COCO/id_to_category.txt", "Category file")
tf.flags.DEFINE_string("train_instance_file", "D:/Detection/dataset/COCO/instances_train2014.json",
                       "Training boxs JSON file.")
tf.flags.DEFINE_string("val_boxs_file", "D:/Detection/dataset/COCO/instances_val2014.json",
                       "Validation boxs JSON file.")

tf.flags.DEFINE_string("output_dir", "D:/Detection/dataset/COCO_tfrecord/", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS
ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "height", "width", "path", "bboxes"])


class ImageDecoder(object):

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()
        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[value.encode('utf-8') if type(value) == str else value]))


def _float_feature(value):
    """Wrapper for inserting a float Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _float_feature_list(values):
    """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _load_cls_bboxes(annotations, im_infos):
    id_to_cls_bboxes = {}
    for entity in annotations:
        image_id = entity["image_id"]
        _, height, width = im_infos[image_id]
        cls_id = entity["category_id"]
        bbox = _transform_bbox(entity["bbox"], cls_id, height, width)
        # [[x_left], [y_bottom], [x_right], [y_top], [category_id]]
        id_to_cls_bboxes.setdefault(image_id, [[], [], [], [], []])
        for i, item in enumerate(id_to_cls_bboxes[image_id]):
            item.append(bbox[i])
    return id_to_cls_bboxes


def _transform_bbox(bbox, cls_id, height, width):
    """
    Validate that whether the box is out of the range of image
    :param bbox: bounding box [x_left, y_right, width, height]
    :param cls_id: bounding box's category id
    :return: [x_left, y_bottom, x_right, y_top, category_id]
    """
    x_left = max(0, bbox[0])
    y_top = max(0, bbox[1])
    x_right = min(width-1, x_left + max(0, bbox[2]-1))
    y_bottom = min(height-1, y_top + max(0, bbox[3]-1))
    return [x_left, y_bottom, x_right, y_top, cls_id]


def _to_sequence_example(image, decoder):
    """
    Builds a SequenceExample proto for an image-detection pair.
    :param image: An ImageMetadata object.
    :param decoder: An ImageDecoder object.
    :return: a SequenceExample proto for an image-detection pair.
    """
    with tf.gfile.FastGFile(image.path, "rb") as f:
        encoded_image = f.read()
    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.path)
    bboxes = image.bboxes
    context = tf.train.Features(feature={
        "image/image_id": _int64_feature(image.image_id),
        "image/data": _bytes_feature(encoded_image)
    })
    feature_lists = tf.train.FeatureLists(feature_list={
        "bbox/locations/x_l": _float_feature_list(bboxes[0]),
        "bbox/locations/y_b": _float_feature_list(bboxes[1]),
        "bbox/locations/x_r": _float_feature_list(bboxes[2]),
        "bbox/locations/y_t": _float_feature_list(bboxes[3]),
        "bbox/categories": _int64_feature_list(bboxes[4]),
        "image/size": _int64_feature_list([image.height, image.width, 3])
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)
    return sequence_example


def _process_image_files(thread_index, ranges, name, images,
                         decoder, num_shards, out_dir):
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    indices = ranges[thread_index]
    shard_ranges = np.linspace(indices[0], indices[1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = indices[1] - indices[0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(out_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]
            sequence_example = _to_sequence_example(image, decoder)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-box pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
    print("%s [thread %d]: Wrote %d image-box pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def process_data(name, images, num_shards):
    """
    Processes a complete data set and saves it as a TFRecord.
    :param name: Unique identifier specifying the dataset.
    :param images: List of ImageMetadata.
    :param num_shards: Integer number of shards for the output files.
    :return: TFRecord dataset
    """
    # No need to break up each image into a separate entity for each bbox.
    # Cause each bbox in the same image can share computation of convolution
    # images = [ImageMetadata(image.image_id, image.height, image.width, image.path, label)
    # for image in images for label in image.labels]
    count = len(images)
    if count < num_shards:
        num_shards = count
    # Shuffle the ordering of images. Make the randomization repeatable
    random.seed(12345)
    random.shuffle(images)
    out_dir = os.path.join(FLAGS.output_dir, name)
    if not tf.gfile.Exists(out_dir):
        tf.gfile.MakeDirs(out_dir)
    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launch %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, num_shards, out_dir)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)
        # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-box pairs in data set '%s'." %
          (datetime.now(), len(images), name))


def debug(id_to_img_info, id_to_bboxes):
    arg1 = [entity[0] for entity in id_to_img_info]
    arg2 = list(id_to_bboxes.keys())
    for v in arg2:
        arg1.remove(v)
    print(arg1)


def load_and_process_metadata(desc_file, img_dir):
    with tf.gfile.FastGFile(desc_file, "r") as f:
        desc_data = json.load(f)
    # Extract filename of image
    id_to_img_info = {entity["id"]: (entity["file_name"], entity["height"], entity["width"])
                      for entity in desc_data["images"]}
    # Extract the target label, each image_id may associated with multiple labels
    annotations = desc_data["annotations"]
    id_to_bboxes = _load_cls_bboxes(annotations, id_to_img_info)
    # debug(id_to_img_info, id_to_bboxes)
    print("Loaded detection metadata for %d images from %s"
          % (len(id_to_img_info), desc_file))
    # Process the labels and combine the data into a list of ImageMetadata
    print("Processing bboxes")
    image_metadata = []
    num_bbox = 0
    skip_num = 0
    for image_id, (base_filename, height, width) in id_to_img_info.items():
        if image_id not in id_to_bboxes:
            skip_num += 1
            continue
        bboxes = id_to_bboxes[image_id]
        path = os.path.join(img_dir, base_filename)
        image_metadata.append(ImageMetadata(image_id, height, width, path, bboxes))
        num_bbox += len(bboxes)
    assert (len(id_to_img_info) - skip_num) == len(id_to_bboxes)
    print("Finished processing %d boxs for %d images in %s,"
          " skip %d images which have no box contained" %
          (num_bbox, len(id_to_img_info), desc_file, skip_num))
    return image_metadata
