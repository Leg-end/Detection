import tensorflow as tf
from . import anchor_util
from collections import namedtuple

__all__ = ["map_rois_to_feature", "sample_regions", "_compute_area",
           "_merge_ppl_tgt_boxes", "boxes_iou",
           "generate_proposal_target"]


class Params(namedtuple("Params",
                        ("fg_thresh", "bg_thresh_range",
                         "roi_batch_size", "image_batch_size",
                         "bbox_in_weights", "bbox_norm_means", "bbox_norm_stddevs",
                         "use_tgt", "bbox_pre_norm", "post_nms_limit", "nms_thresh",
                         "top_limit", "sample_mode"))):
    """Store necessary params"""
    pass


params = None
is_invoke = False


def fill_params(fg_thresh, bg_thresh_range,
                roi_batch_size, image_batch_size,
                bbox_in_weights, bbox_norm_means, bbox_norm_stddevs,
                post_nms_limit, nms_thresh, top_limit, sample_mode="top",
                use_tgt=True, bbox_pre_norm=True):
    """Avoiding passing too may hyper parameters,
    store some necessary params in a dict
    This method must be invoked when hyper parameters have constructed
    :param fg_thresh: IOU thresh for foreground(positive)
    :param bg_thresh_range: IOU thresh range for background(negative)
    :param roi_batch_size: Number of roi in one batch
    :param image_batch_size: Number of image in one batch
    :param bbox_in_weights: Bounding box inside weights
    :param bbox_norm_means: Bounding box normalization means
    :param bbox_norm_stddevs: Bounding box normalization standard deviation
    :param use_tgt: Whether to use target bounding boxes
    :param bbox_pre_norm: Whether to do normalize pre
    :param post_nms_limit: Number of top scoring boxes to keep after applying NMS to RPN proposals
    :param nms_thresh: NMS threshold used on RPN proposals
    :param top_limit: The top n selected when strategy is top
    :param sample_mode: The strategy for sampling and filtering rois from anchors, nms or top
    """
    global is_invoke
    global params
    params = Params(fg_thresh=fg_thresh, bg_thresh_range=bg_thresh_range,
                    use_tgt=use_tgt, roi_batch_size=roi_batch_size, image_batch_size=image_batch_size,
                    bbox_in_weights=bbox_in_weights, bbox_pre_norm=bbox_pre_norm, sample_mode=sample_mode,
                    bbox_norm_means=bbox_norm_means, bbox_norm_stddevs=bbox_norm_stddevs,
                    post_nms_limit=post_nms_limit, top_limit=top_limit, nms_thresh=nms_thresh)
    is_invoke = True


def check_invoke():
    global is_invoke
    if not is_invoke:
        raise EnvironmentError("You must prepare execution environment by invoking method 'fill_params' firstly!")


def map_rois_to_feature(shape, rois, feat_stride):
    height = tf.multiply(tf.to_float(shape[1]) - 1., feat_stride)
    width = tf.multiply(tf.to_float(shape[2]) - 1., feat_stride)
    x_l, y_t, x_r, y_b = anchor_util.get_mat_columns(rois)
    x_l = tf.divide(x_l, width)
    x_r = tf.divide(x_r, width)
    y_t = tf.divide(y_t, height)
    y_b = tf.divide(y_b, height)
    return tf.concat([y_t, x_l, y_b, x_r], axis=1)


def sample_regions(bbox_scores, bbox_deltas,
                   im_info, anchors, count):
    """
    Sampling a set of regions from anchors as RoIs
    1.Take the top K regions according to RPN score
    2.Non-maximal suppression with overlapping ratio of 0.7 is applied to perform de-duplication
    3.Top k regions are selected as RoIs
    (quoted from arXiv:1702.02138v2)
    :param bbox_scores: RPN scores for all anchors in image
    :param bbox_deltas: RPN delta for all anchors after regression
    :param im_info: Image's size info, height and width
    :param anchors: Anchors in image
    :param count: Count of anchors
    :return: The most possible k anchors with object inside and corresponding scores
             The rois are formed as [batch_inds, proposal]
    """
    check_invoke()
    global params
    # Get the scores and bounding boxes
    scores = bbox_scores[:, :, :, count:]
    scores = tf.reshape(scores, shape=(-1,))
    bbox_deltas = tf.reshape(bbox_deltas, shape=[-1, 4])
    # Pick n top score anchors, then do regression
    if params.sample_mode == "top":
        scores, indices = tf.nn.top_k(scores, k=params.top_limit)
        scores = tf.expand_dims(scores, axis=1)
        anchors = tf.gather(anchors, indices)
        bbox_deltas = tf.gather(bbox_deltas, indices)
        proposals = bboxes_regression(anchors, bbox_deltas)
        proposals = _clip_bboxes(proposals, im_info)
    # Do regression to all anchors, the do nms
    elif params.sample_mode == "nms":
        proposals = bboxes_regression(anchors, bbox_deltas)
        proposals = _clip_bboxes(proposals, im_info)
        indices = tf.image.non_max_suppression(proposals, scores,
                                               max_output_size=params.post_nms_limit,
                                               iou_threshold=params.nms_thresh)
        proposals = tf.gather(proposals, indices=indices)
        scores = tf.expand_dims(tf.gather(scores, indices=indices), axis=1)
    else:
        raise NotImplementedError("The infer mode you specific '%s', is not implemented yet" % params.sample_mode)
    batch_inds = tf.zeros(shape=(tf.shape(proposals)[0], 1), dtype=tf.float32)
    rois = tf.concat([batch_inds, proposals], axis=1)
    return rois, scores


def bboxes_target_deltas(ex_rois, tg_rois):
    assert tf.shape(ex_rois)[0] == tf.shape(tg_rois)[0]
    ex_w, ex_h, ex_ctr_x, ex_ctr_y = anchor_util.get_anchors_info(ex_rois)
    tg_w, tg_h, tg_ctr_x, tg_ctr_y = anchor_util.get_anchors_info(tg_rois)
    tg_dx = tf.divide(tf.subtract(tg_ctr_x, ex_ctr_x), ex_w)
    tg_dy = tf.divide(tf.subtract(tg_ctr_y, ex_ctr_y), ex_h)
    tg_dw = tf.log(tf.divide(tg_w, ex_w))
    tg_dh = tf.log(tf.divide(tg_h, ex_h))
    return tf.stack([tg_dx, tg_dy, tg_dw, tg_dh], axis=1)


def bboxes_regression(anchors, deltas):
    assert type(anchors) == type(deltas)
    widths, heights, ctr_x, ctr_y = anchor_util.get_anchors_info(anchors)
    dx, dy, dw, dh = anchor_util.get_mat_columns(deltas)
    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)
    return anchor_util.make_anchors(pred_w, pred_h, pred_ctr_x, pred_ctr_y)


def _clip_bboxes(bboxes, im_info):
    height = tf.cast(im_info[0] - 1, tf.float32)
    width = tf.cast(im_info[1] - 1, tf.float32)
    x_l, y_t, x_r, y_b = anchor_util.get_mat_columns(bboxes)
    x_l = tf.maximum(tf.minimum(x_l, width), 0.0)
    y_t = tf.maximum(tf.minimum(y_t, height), 0.0)
    x_r = tf.maximum(tf.minimum(x_l, width), 0.0)
    y_b = tf.maximum(tf.minimum(y_t, height), 0.0)
    return tf.squeeze(tf.stack([x_l, y_t, x_r, y_b], axis=1))


def _compute_area(area_width_height, scope=None):
    """
        :param area_width_height: shape of [N, M, 2], where 3th-D is width and hight of area
        :param scope: name scope
        :return: shape of [N, M, 1], where 3th-D is area
    """
    with tf.name_scope(scope, 'compute_area'):
        return tf.cast(tf.multiply(tf.slice(area_width_height, [0, 0, 0], [-1, -1, 1]),
                                   tf.slice(area_width_height, [0, 0, 1], [-1, -1, 1])), tf.float32)


def _merge_ppl_tgt_boxes(proposal_boxes, tgt_boxes, scope=None):

    """
        :param proposal_boxes: shape of [M, 4]
        :param tgt_boxes: shape of [N, 4]
        :param scope: name scope
        :return: four tensor with same shape of [N, M, 2]
    """

    with tf.name_scope(scope, 'merge'):
        proposal_left_bottom = tf.slice(proposal_boxes, [0, 0], [-1, 2])
        proposal_right_up = tf.slice(proposal_boxes, [0, 2], [-1, 2])
        tgt_left_bottom = tf.slice(tgt_boxes, [0, 0], [-1, 2])
        tgt_right_up = tf.slice(tgt_boxes, [0, 2], [-1, 2])
        merge = tf.expand_dims(tf.ones_like(
            tf.matmul(tf.zeros_like(tgt_boxes), tf.zeros_like(proposal_boxes), transpose_b=True)), axis=2)
        proposal_left_bottom = tf.multiply(merge, proposal_left_bottom)

        proposal_right_up = tf.multiply(merge, proposal_right_up)

        tgt_left_bottom = tf.expand_dims(tgt_left_bottom, axis=1)
        tgt_right_up = tf.expand_dims(tgt_right_up, axis=1)

        return proposal_left_bottom, proposal_right_up, tgt_left_bottom, tgt_right_up


def boxes_iou(proposal_boxes, tgt_boxes, scope=None):
    """
        :param proposal_boxes: shape of [M, 4]
        :param tgt_boxes:  shape of [N, 4]
        :param scope: name scope
        :return: shape of [N, M, 1], is the ratio of the intersection to the union
    """
    with tf.name_scope(scope, 'IoU'):
        proposal_l_b, proposal_r_u, tgt_l_b, tgt_r_u = _merge_ppl_tgt_boxes(proposal_boxes, tgt_boxes)

        i_left_bottom = tf.maximum(proposal_l_b, tgt_l_b)
        i_right_up = tf.minimum(proposal_r_u, tgt_r_u)

        d_value = tf.subtract(i_right_up, i_left_bottom)
        i_width_height = tf.maximum(tf.zeros_like(d_value), d_value)
        overlap_areas = _compute_area(i_width_height)

        proposal_width_height = tf.subtract(proposal_r_u, proposal_l_b)
        tgt_width_height = tf.subtract(tgt_r_u, tgt_l_b)

        proposal_areas = _compute_area(proposal_width_height)
        tgt_areas = _compute_area(tgt_width_height)

        union_areas = tf.subtract(tf.add(proposal_areas, tgt_areas), overlap_areas)

        return tf.where(tf.equal(overlap_areas, 0.0), tf.zeros_like(overlap_areas),
                        tf.truediv(overlap_areas, union_areas))

# ---------------------target proposal-------------------------------->


def _sample_indices(tensor, size, replace):
    indices = tf.where(tf.equal(tensor, 1))[:, 0]
    indices = anchor_util.random_choice(indices, size, replace=replace)

    return indices


def _compute_target(ex_rois, tgt_rois, labels, means, stddevs, pre_norm=True):
    targets = bboxes_regression(ex_rois, tgt_rois)
    if pre_norm:
        targets = tf.divide(tf.subtract(targets, tf.constant(means, tf.float32)),
                            tf.constant(stddevs, tf.float32))

    return tf.concat([tf.expand_dims(tf.cast(labels, tf.float32), 1), targets], 1)


def _get_bbox_regression_labels(bbox_target_data, num_classes, bbox_in_weights):
    num_targets = tf.expand_dims(tf.range(tf.shape(bbox_target_data)[0]), 1)
    clss = tf.concat([num_targets, tf.to_int32(tf.expand_dims(bbox_target_data[:, 0], 1))], 1)
    tgt_updates = bbox_target_data[:, 1:]
    bbox_targets = tf.tensor_scatter_update(
        tensor=tf.ones([tf.shape(bbox_target_data)[0], num_classes, 4]),
        indices=clss,
        updates=tgt_updates)
    """
    bbox_targets = tf.scatter_nd(indices=clss,
                                 updates=tgt_updates,
                                 shape=[tf.shape(bbox_target_data)[0], num_classes, 4])
    """
    weight_updates = tf.multiply(tf.ones_like(tgt_updates), bbox_in_weights)
    bbox_inside_weights = tf.tensor_scatter_update(
        tensor=tf.ones([tf.shape(bbox_target_data)[0], num_classes, 4]),
        indices=clss,
        updates=weight_updates)
    """
    bbox_inside_weights = tf.scatter_nd(indices=clss,
                                        updates=weight_updates,
                                        shape=[tf.shape(bbox_target_data)[0], num_classes, 4])
    """

    return bbox_targets, bbox_inside_weights


def generate_proposal_target(rpn_rois, rpn_scores,
                             tgt_boxes, num_classes, scope=None):
    """
    Generate target proposal
    :param rpn_rois: Region of Interest calculated by rpn
    :param rpn_scores: Scores of rois
    :param tgt_boxes: Target bounding box
    :param num_classes: Number of class for classification
    :param scope
    :return: labels, rois, roi_scores, bbox_targets, bbox_inside_weights, bbox_outside_weights
    """
    check_invoke()
    global params
    with tf.name_scope(scope, "gen_proposal_tgt"):
        all_rois = rpn_rois
        all_scores = rpn_scores

        if params.use_tgt:
            tf.concat([all_rois, tf.concat(
                [tf.zeros([tf.shape(tgt_boxes)[0], 1]), tgt_boxes[:, :-1]], axis=1)], axis=0)

        overlaps = boxes_iou(tf.slice(all_rois, [0, 1], [-1, -1]), tgt_boxes[:, :-1])
        max_indices = tf.squeeze(tf.argmax(overlaps, axis=0, output_type=tf.int32))
        max_overlaps = tf.gather_nd(
                params=tf.transpose(overlaps, perm=[1, 0, 2]),
                indices=tf.concat([tf.expand_dims(tf.range(tf.shape(max_indices)[0], dtype=tf.int32), 1),
                                   tf.expand_dims(max_indices, 1)], 1))

        labels = tf.gather(tgt_boxes, max_indices)[:, 4]
        fg_thresh = params.fg_thresh
        bg_thresh_range = params.bg_thresh_range
        fg_masks = tf.cast(tf.greater_equal(max_overlaps, fg_thresh), tf.int32)
        bg_masks = tf.cast(tf.greater_equal(max_overlaps, bg_thresh_range[0]) &
                           tf.less(max_overlaps, bg_thresh_range[1]), tf.int32)

        num_fg = tf.reduce_sum(fg_masks)
        num_bg = tf.reduce_sum(bg_masks)

        rois_per_image = params.roi_batch_size / params.image_batch_size
        fg_rois_per_image = fg_thresh * rois_per_image
        rois_per_image = tf.cast(rois_per_image, tf.int32)
        fg_rois_per_image = tf.cast(fg_rois_per_image, tf.int32)

        fg_rois_per_image = tf.minimum(num_fg, fg_rois_per_image)
        bg_rois_per_image = num_bg

        to_replace, fg_rois_per_image, bg_rois_per_image = tf.case({
            tf.greater(fg_rois_per_image, 0) & tf.equal(bg_rois_per_image, 0):
                lambda: [tf.logical_not(tf.greater(rois_per_image, fg_rois_per_image)),
                         rois_per_image,
                         0],
            tf.equal(fg_rois_per_image, 0) & tf.greater(bg_rois_per_image, 0):
                lambda: [tf.greater(rois_per_image, bg_rois_per_image),
                         0,
                         rois_per_image],
            tf.greater(fg_rois_per_image, 0) & tf.greater(bg_rois_per_image, 0):
                lambda: [tf.greater(rois_per_image-fg_rois_per_image, num_bg),
                         fg_rois_per_image,
                         rois_per_image-fg_rois_per_image]},
            default=lambda: [tf.cast(False, tf.bool), 0, 0],
            exclusive=True)

        fg_indices = _sample_indices(fg_masks, fg_rois_per_image, tf.logical_not(to_replace))
        bg_indices = _sample_indices(bg_masks, bg_rois_per_image, to_replace)

        keep_indices = tf.concat([fg_indices, bg_indices], axis=0)
        fg_labels = tf.to_int32(tf.gather(labels, indices=fg_indices))
        labels = tf.tensor_scatter_update(
            tensor=tf.zeros([rois_per_image], dtype=tf.int32),
            indices=tf.expand_dims(tf.range(fg_rois_per_image), 1),
            updates=fg_labels)
        """
        labels = tf.scatter_nd(indices=tf.expand_dims(tf.range(fg_rois_per_image), 1),
                               updates=fg_labels,
                               shape=[rois_per_image])
        """
        rois = tf.gather(all_rois, indices=keep_indices)
        roi_scores = tf.gather(all_scores, keep_indices)
        bbox_target_data = _compute_target(
            ex_rois=rois[:, 1:5],
            tgt_rois=tf.gather(tgt_boxes, tf.gather(max_indices, keep_indices))[:, :4],
            means=params.bbox_norm_means, stddevs=params.bbox_norm_stddevs,
            pre_norm=params.bbox_pre_norm, labels=labels)

        bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            bbox_target_data, num_classes, params.bbox_in_weights)

        bbox_inside_weights = tf.reshape(bbox_inside_weights, [-1, 4*num_classes])
        bbox_outside_weights = tf.cast(tf.greater(bbox_inside_weights, 0), tf.float32)
        bbox_targets = tf.reshape(bbox_targets, [-1, 4*num_classes])
        rois = tf.reshape(rois, [-1, 5])
        roi_scores = tf.reshape(roi_scores, [-1])

    return tf.to_int64(labels), rois, roi_scores, bbox_targets, bbox_inside_weights, bbox_outside_weights
