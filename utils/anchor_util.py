import tensorflow as tf
from . import proposal_util

__all__ = ["generate_image_anchors", "generate_anchors", "repeat_tf",
           "generate_positives_negatives", "generate_target_anchors",
           "get_mat_columns", "get_anchors_info", "make_anchors"]


def generate_image_anchors(height, width, feat_stride=16,
                           ratios=tf.constant([0.5, 1.0, 2.0]),
                           scales=tf.multiply(tf.range(3, 6), 2)):
    """
    Generate anchors around all image
    :param width: image width
    :param height: image height
    :param feat_stride: region stride
    :param ratios:
    :param scales:
    :return: anchors around all image and anchors' count
    """
    shift_x = tf.multiply(tf.range(width), feat_stride)
    shift_y = tf.multiply(tf.range(height), feat_stride)
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    sx = tf.reshape(shift_x, shape=(-1,))
    sy = tf.reshape(shift_y, shape=(-1,))
    # Calculate all kinds of shift within image
    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
    k = tf.multiply(width, height)
    anchors = generate_anchors(feat_stride, ratios=ratios, scales=scales)
    n = tf.shape(anchors)[0]
    # Each shift will apply 9 times to anchors
    shifts = tf.cast(repeat_tf(shifts, n, axis=0), dtype=tf.float32)
    anchors = tf.tile(anchors, [k, 1])
    im_anchors = tf.add(anchors, shifts)
    return im_anchors, tf.shape(im_anchors)[0]


def generate_anchors(base_size=16, ratios=tf.constant([0.5, 1.0, 2.0]),
                     scales=tf.multiply(tf.range(3, 6), 2)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    box:[x_left, y_top, x_right, y_bottom]
    :param base_size: stride of convolutional kernel
    :param ratios: ratios between width and height, default vary in [0.5, 1, 2]
    :param scales: scales for entire region, default vary in [3, 4, 5]
    :return: 9 kinds of box with different anchors but with same center
    """
    scales = tf.cast(scales, dtype=tf.float32)
    base_anchor = tf.subtract(tf.constant([1.0, 1.0, base_size, base_size]), 1.0)
    ratio_anchors = _ratio_enum(tf.expand_dims(base_anchor, axis=0), ratios)
    anchors = tf.stack(_scale_enum(ratio_anchors, scales), axis=0)
    return anchors


def _ratio_enum(anchors, ratios):
    # Each ratios corresponds to each row in anchors that copy three times
    # So we need record anchors' total count before repeat it
    n = tf.shape(anchors)[0]
    m = tf.shape(ratios)[0]
    # Tile each row in anchors m time
    anchors = repeat_tf(anchors, m, axis=0)
    w, h, ctr_x, ctr_y = get_anchors_info(anchors)
    size = tf.multiply(w, h)
    # Tile ratios shape(anchors)[0](denoted as n) time
    ratios = tf.tile(tf.expand_dims(ratios, axis=1), [n, 1])
    # Then we have size as m*n X 1, ratios as m*n X 1, they can divide between each row directly
    size_ratios = tf.divide(size, ratios)
    w = tf.round(tf.sqrt(size_ratios))
    h = tf.round(tf.multiply(w, ratios))
    anchors = make_anchors(w, h, ctr_x, ctr_y)
    return anchors


def _scale_enum(anchors, scales):
    n = tf.shape(anchors)[0]
    m = tf.shape(scales)[0]
    anchors = repeat_tf(anchors, m, axis=0)
    w, h, ctr_x, ctr_y = get_anchors_info(anchors)
    scales = tf.tile(tf.expand_dims(scales, axis=1), [n, 1])
    w = tf.multiply(w, scales)
    h = tf.multiply(h, scales)
    return make_anchors(w, h, ctr_x, ctr_y)


def get_mat_columns(matrix):
    x_l = tf.slice(matrix, [0, 0], [-1, 1])
    x_r = tf.slice(matrix, [0, 2], [-1, 1])
    y_t = tf.slice(matrix, [0, 1], [-1, 1])
    y_b = tf.slice(matrix, [0, 3], [-1, 1])
    return x_l, y_t, x_r, y_b


def get_anchors_info(anchors):
    x_l, y_t, x_r, y_b = get_mat_columns(anchors)
    w = tf.subtract(x_r, x_l)
    h = tf.subtract(y_b, y_t)
    ctr_x = tf.add(x_l, tf.divide(w, 2))
    ctr_y = tf.add(y_t, tf.divide(h, 2))
    return tf.add(w, 1), tf.add(h, 1), ctr_x, ctr_y


def make_anchors(w, h, ctr_x, ctr_y):
    half_w = tf.divide(tf.subtract(w, 1), 2)
    half_h = tf.divide(tf.subtract(h, 1), 2)
    x_l = tf.subtract(ctr_x, half_w)
    x_r = tf.add(ctr_x, half_w)
    y_t = tf.subtract(ctr_y, half_h)
    y_b = tf.add(ctr_y, half_h)
    anchors = tf.squeeze(tf.stack([x_l, y_t, x_r, y_b], axis=1))
    return anchors


def repeat_tf(matrix, num, axis=0):
    """
    This method is tensorflow-version of numpy.repeat
    :param matrix: 2-D tensor
    :param num: time of repeat
    :param axis: repeat axis of matrix, 0 is row, 1 is column
    :return: if axis=0, a row(matrix)*num X column(matrix) tensor with same type as matrix
                else transpose of that tensor
    """
    shape = tf.shape(matrix)
    matrix = tf.tile(matrix, [1, num])
    matrix = tf.reshape(matrix, [shape[0]*num, shape[1]])
    if axis == 1:
        return tf.transpose(matrix)
    return matrix

# ---------------------------------->target_anchors---------------------------------------


def subsample_labels(tensor, size, scope=None):
    with tf.name_scope(scope, 'subsample_labels'):
        indices = tf.slice(tf.where(tf.equal(tensor, 1)), [0, 0], [-1, 1])
        indices = tf.slice(tf.random_shuffle(indices), [0, 0], [1, -1])
        updates = tf.zeros([indices.shape[0], 1], tf.int32)

        return tf.tensor_scatter_update(tensor, indices, updates=updates)


def generate_positives_negatives(boxes_overlaps, scope=None):
    with tf.name_scope(scope, 'generate_score'):
        max_ind = tf.argmax(boxes_overlaps, axis=0, output_type=tf.int32)
        max_overlaps = tf.gather_nd(tf.transpose(boxes_overlaps, perm=[1, 0, 2]), tf.concat(
            [tf.reshape(tf.range(boxes_overlaps.shape[1]), [boxes_overlaps.shape[1], -1]), max_ind], 1))

        max_ind = tf.argmax(boxes_overlaps, axis=1, output_type=tf.int32)
        arg_max_overlaps = tf.gather_nd(boxes_overlaps, tf.concat([tf.reshape(tf.range(
            boxes_overlaps.shape[0]), [boxes_overlaps.shape[0], -1]), max_ind], 1))
        get_argmax = tf.equal(tf.reshape(arg_max_overlaps, [arg_max_overlaps.shape[0],
                                                            arg_max_overlaps.shape[1], 1]), boxes_overlaps)
        arg_max_overlaps = tf.cast(tf.cast(tf.reduce_sum(tf.cast(get_argmax, tf.int32), 0), tf.bool), tf.int32)

        negatives = tf.cast(
            tf.less(max_overlaps, 0.3), tf.int32)  # TRAIN.RPN_NEGATIVE_THRESHOLID
        positives = tf.maximum(arg_max_overlaps,
                               tf.cast(tf.greater_equal(max_overlaps, 0.8), tf.int32))  # TRAIN.PRN_POSITIVE_THRESHOLID

        allow_num_fg = tf.constant([3], tf.int32)  # TRAIN.PRN_FG * TRAIN.PRN_BATCHSIZE
        num_fg = tf.reduce_sum(positives)
        if num_fg > allow_num_fg:
            positives = subsample_labels(positives, num_fg - allow_num_fg)

        allow_num_bg = tf.constant([2], tf.int32)  # TRAIN.PRN_FG * TRAIN.PRN_BATCHSIZE
        num_bg = tf.reduce_sum(negatives)
        if num_bg > allow_num_bg:
            negatives = subsample_labels(negatives, num_bg - allow_num_bg)

        label_negatives = tf.subtract(negatives, 1)
        label_positives = tf.subtract(tf.multiply(positives, 2), 1)
        labels = tf.maximum(label_positives, label_negatives)

        return labels


def generate_target_anchors(all_anchors, targt_boxes, image_info):
    allow_border = 0
    inner_indices = tf.slice(tf.where(
        tf.greater_equal(tf.slice(all_anchors, [0, 0], [-1, 1]), allow_border) &
        tf.greater_equal(tf.slice(all_anchors, [0, 1], [-1, 1]), allow_border) &
        tf.less(tf.slice(all_anchors, [0, 2], [-1, 1]), image_info[1] + allow_border) &
        tf.less(tf.slice(all_anchors, [0, 3], [-1, 1]), image_info[0] + allow_border))[0, 0], [-1, 1])

    anchors = tf.gather_nd(all_anchors, inner_indices)

    # return shape[N, M, 1],where N is number of tgt , M is number of anchor
    boxes_overlaps = proposal_util.boxes_iou(anchors, targt_boxes)
    labels = generate_positives_negatives(boxes_overlaps)
    return labels
