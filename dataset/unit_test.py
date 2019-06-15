from dataset import build_coco_tfrecord
from dataset import iterator_wrapper
import os
import tensorflow as tf
import json
import codecs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_convert_to_tfrecord():
    # file = "D:/Detection/dataset/COCO/instance_unit_test.json"
    train_file = "D:/Detection/dataset/COCO/instances_train2014.json"
    eval_file = "D:/Detection/dataset/COCO/instances_val2014.json"
    mscoco_train_dataset = build_coco_tfrecord.load_and_process_metadata(
        train_file, build_coco_tfrecord.FLAGS.train_image_dir)
    mscoco_eval_dataset = build_coco_tfrecord.load_and_process_metadata(
        eval_file, build_coco_tfrecord.FLAGS.val_image_dir)
    # Redistribute the MSCOCO data as follows:
    #   train_dataset = 100% of mscoco_train_dataset + 85% of mscoco_val_dataset.
    #   val_dataset = 5% of mscoco_val_dataset (for validation during training).
    #   test_dataset = 10% of mscoco_val_dataset (for final evaluation).
    train_cutoff = int(0.85 * len(mscoco_eval_dataset))
    val_cutoff = int(0.90 * len(mscoco_eval_dataset))
    train_dataset = mscoco_train_dataset + mscoco_eval_dataset[0:train_cutoff]
    val_dataset = mscoco_eval_dataset[train_cutoff:val_cutoff]
    test_dataset = mscoco_eval_dataset[val_cutoff:]
    build_coco_tfrecord.process_data("train", train_dataset, build_coco_tfrecord.FLAGS.train_shards)
    build_coco_tfrecord.process_data("eval", val_dataset, build_coco_tfrecord.FLAGS.val_shards)
    build_coco_tfrecord.process_data("test", test_dataset, build_coco_tfrecord.FLAGS.test_shards)


def test_tfrecord_dataset():
    dataset_dir = os.path.join(build_coco_tfrecord.FLAGS.output_dir, "infer")
    filenames = tf.gfile.ListDirectory(dataset_dir)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(dataset_dir, filenames[i])
    print(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    batch_input = iterator_wrapper.get_iterator_wrapper(dataset, 1)
    with tf.Session() as sess:
        sess.run(batch_input.initializer)
        print("1:", sess.run([tf.shape(batch_input.images_data),
                              tf.shape(batch_input.images_size),
                              tf.shape(batch_input.bbox_locations),
                              tf.shape(batch_input.bbox_categories)]))
        """print("2:", sess.run([tf.shape(batch_input.images_data),
                              tf.shape(batch_input.images_size),
                              tf.shape(batch_input.bbox_locations)]))
        print("3:", sess.run([tf.shape(batch_input.images_data),
                              tf.shape(batch_input.images_size),
                              tf.shape(batch_input.bbox_locations)]))"""


def create_category_file():
    def takeFirst(elem):
        return elem[0]
    with tf.gfile.FastGFile("D:/dataset/Detection/COCO/instance_unit_test.json", "r") as f:
        desc_data = json.load(f)
    # Extract category
    id_to_categories = [(entity["id"], entity["name"])for entity in desc_data["categories"]]
    id_to_categories.sort(key=takeFirst)
    with codecs.getwriter("utf-8")(
            tf.gfile.GFile("D:/dataset/Detection/COCO/id_to_category.txt", "wb")) as f:
        for i, word in id_to_categories:
            print(i)
            f.write("%s\n" % word)


if __name__ == "__main__":
    # test_convert_to_tfrecord()
    test_tfrecord_dataset()
    # create_category_file()
