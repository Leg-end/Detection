import tensorflow as tf
from . import proposal_util
from collections import namedtuple

__all__ = ["generate_image_anchors", "generate_anchors", "repeat_tf",
           "generate_anchor_targets",
           "get_mat_columns", "get_anchors_info", "make_anchors"]


class Params(namedtuple("Params",
                        ("rpn_fg_fraction", "rpn_batch_size",
                         "rpn_in_weights", "rpn_positive_weight"))):
    """Store necessary params"""
    pass


params = None
is_invoke = False


def fill_params(rpn_fg_fraction, rpn_batch_size,
                rpn_in_weights, rpn_positive_weight):
    """
    Avoiding passing too may hyper parameters,
    store some necessary params in a dict
    This method must be invoked when hyper parameters have constructed
    :param rpn_in_weights: Mask-like factor that only keep positive ones to regression loss calculation
    :param rpn_positive_weight: Give the positive RPN examples weight of p * 1 / {num positives} and give
                            negatives a weight of (1 - p) Set to -1.0 to use uniform example weighting
    :param rpn_batch_size: Total number of examples
    :param rpn_fg_fraction: Max number of foreground examples
    """
    global is_invoke
    global params
    params = Params(rpn_fg_fraction=rpn_fg_fraction, rpn_batch_size=rpn_batch_size,
                    rpn_positive_weight=rpn_positive_weight, rpn_in_weights=rpn_in_weights)
    is_invoke = True


def check_invoke():
    global is_invoke
    if not is_invoke:
        raise EnvironmentError("You must prepare execution environment by invoking method 'fill_params' firstly!")


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
    :return: anchors around all image, all anchors' count
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


def random_choice(population, n_samples, axis=0, replace=False, scope=None):

    """
    The parameter population can be a tensor of any shape, parameter n_samples is the number of samples,
    the axis represent that sampling is carried out along the dimension of the population, By default it is 0th.
    the parameter replace indicates whether the sampling process puts back elements, by default is False.

    Note that when the population is empty, the return parameter is the population itself, which is empty
    when replace is False, The number of n_samples is larger than the population, the sample is the population.
    This method return value of same type as population
    """
    with tf.name_scope(scope, "random_choice"):
        def _choice_replace(tensor, size):
            out_shape = tf.convert_to_tensor([size], tf.int32)
            indices = tf.random_uniform(
                shape=out_shape,
                minval=0,
                maxval=tf.shape(tensor)[0],
                dtype=tf.int32)
            outputs = tf.gather(
                tensor,
                indices=indices)
            return outputs

        def _choice_no_replace(tensor, size):
            outputs = tf.random_shuffle(tensor)[:size]
            return outputs

        def _swap_dims(tensor, dim_1, dim_2):
            updates = tf.gather(tensor, indices=[dim_1, dim_2])
            return tf.tensor_scatter_update(tensor, indices=[[dim_2], [dim_1]], updates=updates)

        def _fn2(inputs):
            dim_indices = tf.range(
                tf.reduce_sum(tf.ones_like(tf.shape(inputs), dtype=tf.int32)))

            inputs = tf.transpose(
                inputs,
                perm=_swap_dims(dim_indices, dim_1=0, dim_2=axis))
            outputs = tf.case({
                tf.cast(replace, tf.bool): lambda: _choice_replace(inputs, n_samples)},
                default=lambda: _choice_no_replace(inputs, n_samples),
                exclusive=True)
            outputs = tf.transpose(
                outputs,
                perm=_swap_dims(dim_indices, dim_1=0, dim_2=axis))
            return outputs

        def _fn1(inputs):
            outputs = inputs
            return outputs

        samples = tf.cond(tf.equal(tf.shape(population)[axis], 0),
                          fn1=lambda: _fn1(population),
                          fn2=lambda: _fn2(population))
    return samples


def _subsample_labels(tensor, size):

    indices = tf.slice(tf.where(tf.equal(tensor, 1)), [0, 0], [-1, 1])
    indices = random_choice(indices, size, replace=False)
    updates = tf.zeros([size, 1], tf.int32)
    return tf.tensor_scatter_update(tensor, indices, updates)


def _generate_positives_negatives(boxes_overlaps, index_masks,
                                  rpn_fg_fraction, rpn_batch_size, scope=None):
    """
        :param boxes_overlaps: is shape of [N, M, 1]
        :param index_masks: is shape of [m, 1]
        :param scope: name scope
        :return:
    """
    with tf.name_scope(scope, 'generate_score'):
        max_indices = tf.argmax(boxes_overlaps, axis=0, output_type=tf.int32)
        max_overlaps = tf.gather_nd(tf.transpose(boxes_overlaps, perm=[1, 0, 2]), tf.concat(
            [tf.expand_dims(tf.range(tf.shape(max_indices)[0]), 1), max_indices], 1))

        argmax_indices = tf.argmax(boxes_overlaps, axis=1, output_type=tf.int32)
        arg_max_overlaps = tf.gather_nd(boxes_overlaps, tf.concat(
            [tf.expand_dims(tf.range(tf.shape(argmax_indices)[0]), 1), argmax_indices], 1))

        mat_mask = tf.cast(tf.equal(tf.expand_dims(arg_max_overlaps, axis=2), boxes_overlaps), tf.int32)

        max_mask_positives = tf.cast(tf.cast(
                        tf.multiply(tf.reduce_sum(mat_mask, 0), index_masks), tf.bool), tf.int32)

        negatives = tf.multiply(tf.cast(tf.less(max_overlaps, 0.6), tf.int32), index_masks)
        valid_max_mask = tf.cast(tf.greater(tf.subtract(max_mask_positives, negatives), 0), tf.int32)

        positives = tf.maximum(valid_max_mask,
                               tf.cast(tf.greater_equal(max_overlaps, 0.8), tf.int32))

        allow_num_fg = tf.cast(rpn_fg_fraction * rpn_batch_size, tf.int32)
        num_fg = tf.reduce_sum(positives)
        positives = tf.cond(tf.greater(num_fg, allow_num_fg),
                            fn1=lambda: _subsample_labels(positives, num_fg-allow_num_fg), fn2=lambda: positives)

        allow_num_bg = tf.cast(rpn_batch_size, tf.int32)
        num_bg = tf.reduce_sum(negatives)
        negatives = tf.cond(tf.greater(num_bg, allow_num_bg),
                            fn1=lambda: _subsample_labels(negatives, num_bg-allow_num_bg), fn2=lambda: negatives)

        label_negatives = tf.subtract(negatives, 1)
        label_positives = tf.subtract(tf.multiply(positives, 2), 1)
        labels = tf.cast(tf.maximum(label_positives, label_negatives), tf.float32)

        return labels, tf.to_float(positives), tf.to_float(negatives), max_indices


def _compute_bbox_targets(anchor_boxes, target_boxes):
    return proposal_util.bboxes_regression(anchor_boxes, target_boxes)


def generate_anchor_targets(rpn_cls_score, all_anchors, target_boxes,
                            image_info, num_anchors, scope=None):
    """
        :param all_anchors: is shape of [M, 4]
        :param target_boxes: is shape of [N, 4]
        :param image_info:  is shape of [3,]
        :param num_anchors: is
        :param scope: name scope
        :param rpn_cls_score: scores of rois
        :return: target anchors
    """
    check_invoke()
    global params
    image_info = tf.cast(image_info, tf.float32)
    with tf.name_scope(scope, "anchor_targets"):
        allow_border = 0.0
        index_masks = (
                    tf.greater_equal(tf.slice(all_anchors, [0, 0], [-1, 1]), allow_border) &
                    tf.greater_equal(tf.slice(all_anchors, [0, 1], [-1, 1]), allow_border) &
                    tf.less(tf.slice(all_anchors, [0, 2], [-1, 1]), tf.add(image_info[1], allow_border)) &
                    tf.less(tf.slice(all_anchors, [0, 3], [-1, 1]), tf.add(image_info[0], allow_border)))

        mask_anchors = tf.multiply(all_anchors, tf.cast(index_masks, tf.float32))
        boxes_overlaps = proposal_util.boxes_iou(mask_anchors, target_boxes[:, :-1])

        labels, mask_positive_anchors, mask_negative_anchors, max_0 = _generate_positives_negatives(
            boxes_overlaps, tf.cast(index_masks, tf.int32),
            params.rpn_fg_fraction, params.rpn_batch_size, scope=scope)

        bbox_targets = tf.where(tf.equal(tf.squeeze(mask_positive_anchors), 0.0),
                                tf.zeros_like(mask_anchors),
                                _compute_bbox_targets(mask_anchors, tf.gather_nd(target_boxes, max_0)[:, :-1]))

        bbox_inside_weights = tf.multiply(mask_positive_anchors,
                                          tf.constant(params.rpn_in_weights, tf.float32))
        rpn_positive_weight = params.rpn_positive_weight
        if rpn_positive_weight < 0.0:
            num_examples = tf.reduce_sum(tf.cast(tf.greater_equal(labels, 0), tf.float32))
            positive_weights = tf.multiply(
                                mask_positive_anchors * tf.ones([4], tf.float32),
                                tf.divide(1.0, num_examples))

            negative_weights = tf.multiply(
                                mask_positive_anchors * tf.ones([4], tf.float32),
                                tf.divide(1.0, num_examples))
        else:
            assert ((rpn_positive_weight > 0.0) &
                    (rpn_positive_weight < 1.0))
            positive_weights = tf.multiply(
                        mask_positive_anchors * tf.ones([4], tf.float32),
                        tf.divide(rpn_positive_weight, tf.reduce_sum(mask_positive_anchors)))

            negative_weights = tf.multiply(
                        mask_negative_anchors * tf.ones([4], tf.float32),
                        tf.divide(1.0 - rpn_positive_weight, tf.reduce_sum(mask_negative_anchors)))

        bbox_outside_weights = tf.maximum(positive_weights, negative_weights)

        height = tf.shape(rpn_cls_score)[1]
        width = tf.shape(rpn_cls_score)[2]
        labels = tf.reshape(labels, [1, height, width, num_anchors])
        rpn_labels = tf.reshape(tf.transpose(labels, perm=[0, 3, 1, 2]),
                                [1, 1, num_anchors * height, width])
        rpn_bbox_targets = tf.reshape(bbox_targets, [1, height, width, num_anchors * 4])
        rpn_inside_weights = tf.reshape(bbox_inside_weights, [1, height, width, num_anchors * 4])
        rpn_outside_weights = tf.reshape(bbox_outside_weights, [1, height, width, num_anchors * 4])

        return rpn_labels, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights
