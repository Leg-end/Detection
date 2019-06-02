"""This is a base library for translate MCOCO-dataset-2014 into TFRecord"""
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

tf.flags.DEFINE_string("train_instance_file", "D:/dataset/Detection/COCO/instances_train2014.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_captions_file", "D:/dataset/Detection/COCO/instances_val2014.json",
                       "Validation captions JSON file.")

tf.flags.DEFINE_string("output_dir", "D:/dataset/Detection/COCO_tfrecord/", "Output data directory.")

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
                           ["image_id", "height", "width", "path", "labels"])


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


def _load_cls_bboxes(annotations):
    id_to_cls_bboxes = {}
    for entity in annotations:
        image_id = entity["image_id"]
        cls_id = entity["category_id"]
        bbox = entity["bbox"]
        overlap = 1.0
        # If crowd is 1, means this bbox will be excluded during training
        if entity["iscrowd"]:
            overlap = -1.0
        id_to_cls_bboxes.setdefault(image_id, [])
        id_to_cls_bboxes[image_id].append([cls_id, overlap, bbox])

    return id_to_cls_bboxes


def _valid_bbox(label, height, width):
    """
    Validate that whether the box is out of the range of image
    :param label: [category_id, overlap, bbox]
    :param height: image height
    :param width: image width
    :return: [category_id, overlap, x_left, y_top, x_right, y_bottom]
    """
    bbox = label[2]
    x_l = max(0, bbox[0])
    y_t = max(0, bbox[1])
    x_r = min(width-1, max(0, bbox[2]))
    y_b = min(height-1, max(0, bbox[3]))
    return [label[0], label[1], x_l, y_t, x_r, y_b]


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
    assert len(image.labels) == 6
    labels = image.labels
    context = tf.train.Features(feature={
        "image/image_id": _int64_feature(image.image_id),
        "image/height": _int64_feature(image.height),
        "image/width": _int64_feature(image.width),
        "image/data": _bytes_feature(encoded_image),
        "bbox/cate_id": _int64_feature(labels[0]),
        "bbox/overlap": _float_feature(labels[1]),
    })

    feature_lists = tf.train.FeatureLists(feature_list={
        "bbox/coord": _float_feature_list(labels[2:]),
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)
    return sequence_example


def _process_image_files(thread_index, ranges, name, images,
                         decoder, num_shards):
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
        output_file = os.path.join(FLAGS.output_dir, name, output_filename)
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
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
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
    # Break up each image into a separate entity for each caption.
    images = [ImageMetadata(image.image_id, image.height, image.width, image.path, label)
              for image in images for label in image.labels]
    count = len(images)
    if count < num_shards:
        num_shards = count
    # Shuffle the ordering of images. Make the randomization repeatable
    random.seed(12345)
    random.shuffle(images)

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
        args = (thread_index, ranges, name, images, decoder, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)
        # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
          (datetime.now(), len(images), name))


def load_and_process_metadata(desc_file, img_dir):
    with tf.gfile.FastGFile(desc_file, "r") as f:
        desc_data = json.load(f)
    # Extract filename of image
    id_to_img_info = [(entity["id"], entity["file_name"], entity["height"], entity["width"])
                      for entity in desc_data["images"]]
    # Extract the target label, each image_id may associated with multiple labels
    annotations = desc_data["annotations"]
    id_to_labels = _load_cls_bboxes(annotations)
    assert len(id_to_img_info) == len(id_to_labels)
    assert set([entity[0] for entity in id_to_img_info]) == set(id_to_labels.keys())
    print("Loaded detection metadata for %d images from %s"
          % (len(id_to_img_info), desc_file))

    # Process the labels and combine the data into a list of ImageMetadata
    print("Processing labels")
    image_metadata = []
    num_labels = 0
    for image_id, base_filename, height, width in id_to_img_info:
        path = os.path.join(img_dir, base_filename)
        labels = id_to_labels[image_id]
        labels = [_valid_bbox(label, height, width) for label in labels]
        image_metadata.append(ImageMetadata(image_id, height, width, path, labels))
        num_labels += len(labels)
    print("Finished processing %d captions for %d images in %s" %
          (num_labels, len(id_to_img_info), desc_file))
    return image_metadata

