from dataset import build_coco_tfrecord
from dataset import iterator_wrapper
import os
import tensorflow as tf
import json
import codecs
from matplotlib import pyplot as plt
from matplotlib import patches
from utils import anchor_util
from PIL import Image
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
    build_coco_tfrecord.process_data("infer", test_dataset, build_coco_tfrecord.FLAGS.test_shards)


def test_small_convertion():
    # file = "D:/Detection/dataset/COCO/instance_unit_test.json"
    # mscoco_train_dataset = build_coco_tfrecord.load_and_process_metadata(
    #     file, build_coco_tfrecord.FLAGS.train_image_dir)
    # build_coco_tfrecord.process_data("unit_test", mscoco_train_dataset, build_coco_tfrecord.FLAGS.train_shards)
    test_tfrecord_dataset("unit_test")


def test_tfrecord_dataset(name='infer'):
    dataset_dir = os.path.join(build_coco_tfrecord.FLAGS.output_dir, name)
    filenames = tf.gfile.ListDirectory(dataset_dir)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(dataset_dir, filenames[i])
    print(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    batch_input = iterator_wrapper.get_iterator_wrapper(dataset, 1)
    with tf.Session() as sess:
        sess.run(batch_input.initializer)
        print("1:", sess.run([batch_input.images_id, tf.shape(batch_input.images_data),
                              batch_input.images_size,
                              batch_input.bbox_locations,
                              anchor_util.get_coco_anchors(batch_input.bbox_locations)]))
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


def test_box():
    box = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]]
    box = build_coco_tfrecord._valid_bbox(box, img_width=10, img_height=10)
    print(box)


def test_draw_box():
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(Image.open("D:/dataset/Images/COCO/val2014/COCO_val2014_000000439410.jpg"))
    # Create a Rectangle patch
    rect = patches.Rectangle((417.11, 145.4), 9.67, 11.38,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    # test_convert_to_tfrecord()
    test_tfrecord_dataset()
    # test_small_convertion()
    # test_box()
    # create_category_file()
