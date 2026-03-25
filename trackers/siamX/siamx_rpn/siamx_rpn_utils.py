"""Shared utility functions for the siamX RPN/FC trackers.

These functions are adapted from deep_mdp's siamx_rpn_utils.py and provide
sub-window extraction and bounding-box conversion helpers used by SiamFCTracker.
"""

import numpy as np
import cv2


def get_subwindow_tracking(im: np.ndarray, pos: tuple, model_sz: int, original_sz: int,
                           avg_chans: np.ndarray) -> np.ndarray:
    """Extract a square sub-window from *im* centered at *pos*, padded with *avg_chans*.

    Args:
        im:          Input image (H, W, C) as uint8 or float.
        pos:         (y, x) center of the crop in image coordinates.
        model_sz:    Output size (side length in pixels) after resizing.
        original_sz: Actual crop size in the source image before resizing.
        avg_chans:   Per-channel mean used as padding colour when the crop extends
                     beyond image boundaries.

    Returns:
        Cropped and resized image of shape (model_sz, model_sz, C).
    """
    im_h, im_w = im.shape[:2]
    c = (original_sz + 1) / 2
    context_xmin = round(pos[1] - c)
    context_xmax = context_xmin + original_sz - 1
    context_ymin = round(pos[0] - c)
    context_ymax = context_ymin + original_sz - 1

    left_pad   = max(0, -context_xmin)
    top_pad    = max(0, -context_ymin)
    right_pad  = max(0, context_xmax - im_w + 1)
    bottom_pad = max(0, context_ymax - im_h + 1)

    context_xmin += left_pad
    context_xmax += left_pad
    context_ymin += top_pad
    context_ymax += top_pad

    if any(p > 0 for p in (left_pad, top_pad, right_pad, bottom_pad)):
        te_im = np.zeros(
            (im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, im.shape[2]),
            dtype=np.uint8,
        )
        te_im[top_pad:top_pad + im_h, left_pad:left_pad + im_w] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + im_w] = avg_chans
        if bottom_pad:
            te_im[top_pad + im_h:, left_pad:left_pad + im_w] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad] = avg_chans
        if right_pad:
            te_im[:, left_pad + im_w:] = avg_chans
        im_patch_original = te_im[
            int(context_ymin):int(context_ymax + 1),
            int(context_xmin):int(context_xmax + 1),
        ]
    else:
        im_patch_original = im[
            int(context_ymin):int(context_ymax + 1),
            int(context_xmin):int(context_xmax + 1),
        ]

    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    return im_patch


def cxy_wh_2_rect(pos: np.ndarray, sz: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) to top-left (x, y, w, h)."""
    return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])


def rect_2_cxy_wh(rect: np.ndarray) -> tuple:
    """Convert top-left (x, y, w, h) to centre (cx, cy) and size (w, h)."""
    return (
        np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]),
        np.array([rect[2], rect[3]]),
    )


def get_axis_aligned_bbox(region: np.ndarray) -> tuple:
    """Return axis-aligned bounding box of an arbitrary polygon.

    Args:
        region: array of 4 or 8 values (either (x,y,w,h) or the 4 corner
                points in row-major order).

    Returns:
        (cx, cy, w, h) centre and size.
    """
    if len(region) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = region
        cx = (x1 + x2 + x3 + x4) / 4
        cy = (y1 + y2 + y3 + y4) / 4
        w = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4))
        h = (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
    else:
        # Already (x, y, w, h) top-left
        x, y, w, h = region
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h
