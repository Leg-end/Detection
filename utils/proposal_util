import tensorflow as tf
from . import anchor_util


__all__ = ["map_rois_to_feature", "sample_regions", "compute_area",
           "merge_ppl_tgt_boxes", "boxes_iou"]


def map_rois_to_feature(shape, rois, feat_stride):
    height = tf.multiply(tf.to_float(shape[1]) - 1., feat_stride)
    width = tf.multiply(tf.to_float(shape[2]) - 1., feat_stride)
    x_l, y_t, x_r, y_b = anchor_util.get_mat_columns(rois)
    x_l = tf.divide(x_l, width)
    x_r = tf.divide(x_r, width)
    y_t = tf.divide(y_t, height)
    y_b = tf.divide(y_b, height)
    return tf.concat([y_t, x_l, y_b, x_r], axis=1)


def sample_regions(cls_scores, bbox_deltas, im_info, anchors,
                   count, strategy="nms", top_n=10, nms_thresh=0.7):
    """
    Sampling a set of regions from anchors as RoIs
    1.Take the top K regions according to RPN score
    2.Non-maximal suppression with overlapping ratio of 0.7 is applied to perform de-duplication
    3.Top k regions are selected as RoIs
    (quoted from arXiv:1702.02138v2)
    :param cls_scores: RPN scores for all anchors in image
    :param bbox_deltas: RPN delta for all anchors after regression
    :param im_info: Image's size info, height and width
    :param anchors: Anchors in image
    :param count: Count of anchors
    :param strategy: 'nms' or 'top' Strategy to select regions
    :param top_n: nms_top_n or top_n
    :param nms_thresh: thresh when specify nms
    :return: The most possible k anchors with object inside and corresponding scores
    """
    # Get the scores and bounding boxes
    scores = cls_scores[:, :, :, count:]
    scores = tf.reshape(scores, shape=(-1,))
    bbox_deltas = tf.reshape(bbox_deltas, shape=[-1, 4])
    # Pick n top score anchors, then do regression
    if strategy == "top":
        scores, indices = tf.nn.top_k(scores, k=top_n)
        scores = tf.expand_dims(scores, axis=1)
        anchors = tf.gather(anchors, indices)
        bbox_deltas = tf.gather(bbox_deltas, indices)
        proposals = bboxes_regression(anchors, bbox_deltas)
        proposals = _clip_bboxes(proposals, im_info[:2])
    # Do regression to all anchors, the do nms
    elif strategy == "nms":
        proposals = bboxes_regression(anchors, bbox_deltas)
        proposals = _clip_bboxes(proposals, im_info[:2])
        indices = tf.image.non_max_suppression(proposals, scores,
                                               max_output_size=top_n,
                                               iou_threshold=nms_thresh)
        proposals = tf.gather(proposals, indices=indices)
        scores = tf.expand_dims(tf.gather(scores, indices=indices), axis=1)
    else:
        raise NotImplementedError("The infer mode you specific '%s', is not implemented yet" % strategy)
    batch_inds = tf.zeros(shape=(tf.shape(proposals)[0], 1), dtype=tf.float32)
    rois = tf.concat([batch_inds, proposals], axis=1)
    return rois, scores


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
    height = im_info[0] - 1
    width = im_info[1] - 1
    x_l, y_t, x_r, y_b = anchor_util.get_mat_columns(bboxes)
    x_l = tf.maximum(tf.minimum(x_l, width), 0)
    y_t = tf.maximum(tf.minimum(y_t, height), 0)
    x_r = tf.maximum(tf.minimum(x_l, width), 0)
    y_b = tf.maximum(tf.minimum(y_t, height), 0)
    return tf.squeeze(tf.stack([x_l, y_t, x_r, y_b], axis=1))


def boxes_iou(proposal_boxes, tgt_boxes, scope=None):
    """
        :param proposal_boxes: shape of [M, 4]
        :param tgt_boxes:  shape of [N, 4]
        :param scope: name scope
        :return: shape of [N, M, 1], is the ratio of the intersection to the union
    """
    with tf.name_scope(scope, 'IoU'):
        proposal_l_b, proposal_r_u, tgt_l_b, tgt_r_u = merge_ppl_tgt_boxes(proposal_boxes, tgt_boxes)

        i_left_bottom = tf.maximum(proposal_l_b, tgt_l_b)
        i_right_up = tf.minimum(proposal_r_u, tgt_r_u)

        d_value = tf.subtract(i_right_up, i_left_bottom)
        i_width_height = tf.maximum(tf.zeros_like(d_value), d_value)
        overlap_areas = compute_area(i_width_height)

        proposal_width_height = tf.subtract(proposal_r_u, proposal_l_b)
        tgt_width_height = tf.subtract(tgt_r_u, tgt_l_b)

        proposal_areas = compute_area(proposal_width_height)
        tgt_areas = compute_area(tgt_width_height)

        union_areas = tf.subtract(tf.add(proposal_areas, tgt_areas), overlap_areas)

        return tf.where(tf.equal(overlap_areas, 0.0),
                        tf.zeros_like(overlap_areas),
                        tf.truediv(overlap_areas, union_areas))


def compute_area(area_width_height, scope=None):
    """
        :param area_width_height: shape of [N, M, 2], where 3th-D is width and hight of area
        :param scope: name scope
        :return: shape of [N, M, 1], where 3th-D is area
    """
    with tf.name_scope(scope, 'compute_area'):
        return tf.cast(tf.multiply(tf.slice(
            area_width_height, [0, 0, 0], [-1, -1, 1]), tf.slice(
            area_width_height, [0, 0, 1], [-1, -1, 1])), tf.float32)


def merge_ppl_tgt_boxes(proposal_boxes, tgt_boxes, scope=None):
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
        proposal_left_bottom = tf.multiply(tf.ones(
            [tgt_left_bottom.shape[0], proposal_left_bottom.shape[0], 1], tf.int32), proposal_left_bottom)
        proposal_right_up = tf.multiply(tf.ones(
            [tgt_right_up.shape[0], proposal_right_up.shape[0], 1], tf.int32), proposal_right_up)
        tgt_left_bottom = tf.reshape(tgt_left_bottom, [tgt_left_bottom.shape[0], -1, tgt_left_bottom.shape[1]])
        tgt_right_up = tf.reshape(tgt_right_up, [tgt_right_up.shape[0], -1, tgt_right_up.shape[1]])
        return proposal_left_bottom, proposal_right_up, tgt_left_bottom, tgt_right_up


