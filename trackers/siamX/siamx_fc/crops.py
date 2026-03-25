"""Image cropping utilities for SiamFC.

Provides functions to extract template (z) and search (x) crops from a frame
for the SiamFC tracker.  Adapted from deep_mdp's crops.py.
"""

from __future__ import annotations

import numpy as np
import cv2
import torch


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def pad_frame(im: np.ndarray, pos_y: float, pos_x: float, patch_sz: int,
              avg_chans: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Pad *im* so that a square crop of side *patch_sz* centred at
    (pos_y, pos_x) never reaches outside the image.

    Returns:
        (padded_image, new_pos_y, new_pos_x)
    """
    h, w = im.shape[:2]
    half = patch_sz // 2

    top    = max(0, half - int(round(pos_y)))
    bottom = max(0, int(round(pos_y)) + half - h + 1)
    left   = max(0, half - int(round(pos_x)))
    right  = max(0, int(round(pos_x)) + half - w + 1)

    if top > 0 or bottom > 0 or left > 0 or right > 0:
        te_im = np.zeros(
            (h + top + bottom, w + left + right, im.shape[2]),
            dtype=np.uint8,
        )
        te_im[top:top + h, left:left + w] = im
        if top:
            te_im[:top, left:left + w] = avg_chans
        if bottom:
            te_im[top + h:, left:left + w] = avg_chans
        if left:
            te_im[:, :left] = avg_chans
        if right:
            te_im[:, left + w:] = avg_chans
        im = te_im
        pos_y += top
        pos_x += left

    return im, pos_y, pos_x


def _extract_crop(im: np.ndarray, pos_y: float, pos_x: float,
                  patch_sz: int, out_sz: int,
                  avg_chans: np.ndarray) -> np.ndarray:
    """Extract a square crop and resize to *out_sz* × *out_sz*."""
    im_pad, py, px = pad_frame(im, pos_y, pos_x, patch_sz, avg_chans)
    py, px = int(round(py)), int(round(px))
    half = patch_sz // 2
    crop = im_pad[py - half: py - half + patch_sz,
                  px - half: px - half + patch_sz]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        crop = np.tile(avg_chans, (out_sz, out_sz, 1)).astype(np.uint8)
    else:
        crop = cv2.resize(crop, (out_sz, out_sz), interpolation=cv2.INTER_LINEAR)
    return crop


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_crops_z(im: np.ndarray, pos_y: float, pos_x: float,
                    z_sz: float, exemplar_sz: int = 127) -> np.ndarray:
    """Return a single template crop of size *exemplar_sz* × *exemplar_sz*.

    Args:
        im:          Source image (H, W, C) uint8.
        pos_y:       Vertical centre of the target in *im*.
        pos_x:       Horizontal centre of the target in *im*.
        z_sz:        Side length of the region to crop (before resizing).
        exemplar_sz: Output side length (default 127).

    Returns:
        Crop of shape (*exemplar_sz*, *exemplar_sz*, C) as uint8 ndarray.
    """
    avg_chans = np.mean(im, axis=(0, 1)).astype(np.uint8)
    patch_sz = int(round(z_sz))
    return _extract_crop(im, pos_y, pos_x, patch_sz, exemplar_sz, avg_chans)


def extract_crops_x(im: np.ndarray, pos_y: float, pos_x: float,
                    x_sz: float, scale_factors: list[float],
                    search_sz: int = 255) -> list[np.ndarray]:
    """Return one search crop per scale factor.

    Args:
        im:            Source image (H, W, C) uint8.
        pos_y:         Vertical centre of the target.
        pos_x:         Horizontal centre of the target.
        x_sz:          Base search crop size at scale 1.0 (pixels).
        scale_factors: List of scale multipliers (e.g. [0.96, 1.0, 1.04]).
        search_sz:     Output side length for each crop (default 255).

    Returns:
        List of crops, each (search_sz, search_sz, C) uint8 ndarray.
    """
    avg_chans = np.mean(im, axis=(0, 1)).astype(np.uint8)
    crops = []
    for sf in scale_factors:
        patch_sz = int(round(x_sz * sf))
        crops.append(_extract_crop(im, pos_y, pos_x, patch_sz, search_sz, avg_chans))
    return crops


def image_to_tensor(img: np.ndarray, avg_image: list[float] | None = None,
                    device: torch.device | None = None) -> torch.Tensor:
    """Convert an HxWxC uint8 BGR NumPy array to a normalised float tensor.

    Args:
        img:       Crop in BGR format (HWC uint8).
        avg_image: Per-channel mean to subtract [R, G, B].  If ``None``, no
                   mean subtraction is performed.
        device:    Target torch device.

    Returns:
        Float tensor of shape (1, C, H, W).
    """
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if avg_image is not None:
        img -= np.array(avg_image, dtype=np.float32)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def gen_xz(im: np.ndarray, pos_y: float, pos_x: float,
           z_sz: float, x_sz: float, scale_factors: list[float],
           exemplar_sz: int = 127, search_sz: int = 255,
           avg_image: list[float] | None = None,
           device: torch.device | None = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate template and search tensors ready for the siamese network.

    Returns:
        (z_tensor, x_tensor) where z has shape (1, 3, exemplar_sz, exemplar_sz)
        and x has shape (n_scales, 3, search_sz, search_sz).
    """
    z_crop = extract_crops_z(im, pos_y, pos_x, z_sz, exemplar_sz)
    x_crops = extract_crops_x(im, pos_y, pos_x, x_sz, scale_factors, search_sz)

    z_t = image_to_tensor(z_crop, avg_image, device)
    x_t = torch.cat(
        [image_to_tensor(c, avg_image, device) for c in x_crops], dim=0
    )
    return z_t, x_t
