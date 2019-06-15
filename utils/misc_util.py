import tensorflow as tf
import codecs
import json
import os
from . import proposal_util
from tensorflow.python.ops import lookup_ops
import numpy as np
from six.moves import range
from tensorlib.training import HParams
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen']


NUM_COLORS = len(STANDARD_COLORS)


try:
    FONT = ImageFont.truetype('arial.ttf', 24)
except IOError:
    FONT = ImageFont.load_default()


def create_reverse_category_table(src_cate_file):
    reverse_cate_table = lookup_ops.index_to_string_table_from_file(
        src_cate_file, default_value="unk")
    return reverse_cate_table


def create_category_tables(src_cate_file):
    """Create vocabulary table from file"""
    src_cate_table = lookup_ops.index_table_from_file(
        src_cate_file, default_value=-1)
    return src_cate_table


def append_params(_dict, **params):
    # _dict.update({sys._getframe().f_back.f_locals[param]:param})
    _dict.update(params)


def append_param(_dict, **param):
    # _dict.update({sys._getframe().f_back.f_locals[param]:param})
    _dict.update(param)


def check_file_existence(filename):
    if not tf.gfile.Exists(filename):
        tf.logging.fatal("File '%s' not found." % filename)
        return False
    else:
        return True


def print_hparams(hparams, skip_patterns=None):
    values = hparams.values()
    for key in sorted(values.keys()):
        if not skip_patterns or all(
                [skip_pattern not in key
                 for skip_pattern in skip_patterns]):
            print(" %s=%s" % (key, str(values[key])))


def load_hparams(hp_dir):
    path = os.path.join(hp_dir, "hparams.json")
    if check_file_existence(path):
        print("# Loading hparams from '%s'" % path)
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(path, "rb")) as f:
            try:
                hparams_values = json.load(f)
                hparams = HParams(**hparams_values)
            except ValueError:
                print(" Can't load hparams file '%s'" % path)
                return None
        return hparams
    else:
        return None


def parse_standard_hparams(hparams, path):
    if path and check_file_existence(path):
        print("# Loading hparams from '%s'" % path)
        with codecs.getwriter("utf-8")(
                tf.gfile.GFile(path, "rb")) as f:
            hparams.parse_json(f.read())


def save_hparams(out_dir, hparams):
    path = os.path.join(out_dir, "hparams.json")
    print(" Saving hparams to '%s'" % path)
    with codecs.getwriter("utf-8")(
            tf.gfile.GFile(path, "wb")) as f:
        f.write(hparams.to_json(indent=4, sort_keys=True))


def add_summary(summary_writer, global_step, tag, value):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def debug_tensor(s, msg=None, summarize=10):
    """Print the shape and value of a tensor at test time. Return a new tensor."""
    if not msg:
        msg = s.name
    return tf.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)


def back_to_rgb(image, pixel_means, im_info):
    image += pixel_means
    image = tf.image.resize_bilinear(image, tf.to_int32(im_info[:2] / im_info[2]))
    return tf.reverse(image, axis=[-1])


def _draw_box_on_image(image, xmin, ymin, xmax, ymax, display_str, font, color='black', thickness=4):
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    text_bottom = bottom
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin),
         (left + text_width, text_bottom)], fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str, fill='black', font=font)
    return image


def draw_boxes_on_image(image, bboxes, prob, category, im_info,
                        bgr=False, pixel_means=None):
    if bgr:
        image = back_to_rgb(image, pixel_means, im_info)
    num_boxes = bboxes.shape[0]
    gt_boxes_new = bboxes.copy()
    gt_boxes_new[:, :4] = np.round(gt_boxes_new[:, :4].copy() / im_info[2])
    disp_image = Image.fromarray(np.uint8(image[0]))
    for i in range(num_boxes):
        this_class = int(gt_boxes_new[i, 4])
        disp_image = _draw_box_on_image(disp_image,
                                        gt_boxes_new[i, 0],
                                        gt_boxes_new[i, 1],
                                        gt_boxes_new[i, 2],
                                        gt_boxes_new[i, 3],
                                        'N%02d-C%s-P%02d' % (i, category, prob),
                                        FONT,
                                        color=STANDARD_COLORS[this_class % NUM_COLORS])
        image[0, :] = np.array(disp_image)
    return image


def get_config_proto(log_device_placement=False,
                     allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.5

    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
        if num_inter_threads:
            config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto


def mean_avg_overlap(bbox_labels, bbox_infer):
    bbox_overlaps = proposal_util.boxes_iou(bbox_labels, bbox_infer)
    bbox_overlaps = tf.squeeze(bbox_overlaps, axis=2)
    max_indices = tf.argmax(bbox_overlaps, axis=1)
    bbox_overlaps = tf.gather(
        bbox_overlaps, max_indices, axis=0)
    return tf.divide(tf.reduce_sum(bbox_overlaps),
                     tf.to_float(tf.shape(bbox_labels)[0]))
