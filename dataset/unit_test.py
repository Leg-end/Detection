from dataset import build_coco_tfrecord
from dataset import iterator_wrapper
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_convert_to_tfrecord():
    file = "D:/dataset/Detection/COCO/instance_unit_test.json"
    mscoco_train_dataset = build_coco_tfrecord.load_and_process_metadata(file, build_coco_tfrecord.FLAGS.train_image_dir)
    """for data in mscoco_train_dataset:
        print(data.labels[0])"""
    build_coco_tfrecord.process_data("train", mscoco_train_dataset, build_coco_tfrecord.FLAGS.train_shards)


def test_tfrecord_dataset():
    dataset_dir = os.path.join(build_coco_tfrecord.FLAGS.output_dir, "train")
    filenames = tf.gfile.ListDirectory(dataset_dir)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(dataset_dir, filenames[i])
    print(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    batch_input = iterator_wrapper.get_iterator_wrapper(dataset, 4)
    with tf.Session() as sess:
        sess.run(batch_input.initializer)
        print(sess.run(batch_input.bboxes_info))


if __name__ == "__main__":
    # test_convert_to_tfrecord()
    test_tfrecord_dataset()
