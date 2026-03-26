"""Image cropping utilities for SiamFC.

Provides functions to extract template (z) and search (x) crops from a frame
for the SiamFC tracker.  Adapted from zllrunning/SiameseX.PyTorch
``demo_utils/crops.py``.

Input images are accepted as numpy BGR uint8 ndarrays (OpenCV format); PIL
Image is used internally for cropping to match the SiameseX.PyTorch pipeline.
``image_to_tensor`` applies the SiameseX.PyTorch normalisation
(mean=0.5, std=0.25 per channel).

A faster OpenCV-based path is also provided via :func:`_extract_crop_cv2`
and :func:`image_to_tensor_batch`.  These are used when ``fast=True`` is
passed to :func:`extract_crops_z` / :func:`extract_crops_x`.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import torch


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def pad_frame(im: Image.Image, pos_x: float, pos_y: float,
              patch_sz: int) -> tuple[Image.Image, float, float]:
    """Pad a PIL Image so that a square of side *patch_sz* centred at
    (pos_x, pos_y) lies entirely within the padded canvas.

    Returns:
        (padded_image, new_pos_x, new_pos_y)
    """
    w, h = im.size
    half = patch_sz // 2

    top    = max(0, half - int(round(pos_y)))
    bottom = max(0, int(round(pos_y)) + half - h + 1)
    left   = max(0, half - int(round(pos_x)))
    right  = max(0, int(round(pos_x)) + half - w + 1)

    if top > 0 or bottom > 0 or left > 0 or right > 0:
        # Fill padding with the image mean colour
        arr = np.array(im)
        avg = tuple(int(v) for v in arr.mean(axis=(0, 1)))
        new_w = w + left + right
        new_h = h + top + bottom
        canvas = Image.new(im.mode, (new_w, new_h), avg)
        canvas.paste(im, (left, top))
        im = canvas
        pos_x += left
        pos_y += top

    return im, pos_x, pos_y


def _extract_crop(im: np.ndarray, pos_y: float, pos_x: float,
                  patch_sz: int, out_sz: int) -> Image.Image:
    """Extract a square crop from a BGR uint8 ndarray and return a PIL RGB
    Image of size *out_sz* × *out_sz*."""
    # Convert BGR numpy → PIL RGB
    pil_im = Image.fromarray(im[:, :, ::-1])  # BGR→RGB

    pil_im, px, py = pad_frame(pil_im, pos_x, pos_y, patch_sz)
    px, py = int(round(px)), int(round(py))
    half = patch_sz // 2

    box = (px - half, py - half, px - half + patch_sz, py - half + patch_sz)
    crop = pil_im.crop(box)
    crop = crop.resize((out_sz, out_sz), Image.BILINEAR)
    return crop


def _extract_crop_cv2(im: np.ndarray, pos_y: float, pos_x: float,
                      patch_sz: int, out_sz: int) -> np.ndarray:
    """Faster OpenCV-based crop+resize, returning an RGB uint8 ndarray.

    Replaces the PIL pipeline (``_extract_crop``) to avoid multiple
    format-conversion round-trips.  Padding with the image mean colour is
    handled via ``cv2.copyMakeBorder``.

    Args:
        im:       Source image as BGR uint8 ndarray (H, W, 3).
        pos_y:    Vertical centre of the crop in *im*.
        pos_x:    Horizontal centre of the crop in *im*.
        patch_sz: Side length of the square region to extract.
        out_sz:   Output side length after resizing.

    Returns:
        RGB uint8 ndarray of shape ``(out_sz, out_sz, 3)``.
    """
    h, w = im.shape[:2]
    half = patch_sz // 2

    # Compute the crop window in the original image
    x1 = int(round(pos_x)) - half
    y1 = int(round(pos_y)) - half
    x2 = x1 + patch_sz
    y2 = y1 + patch_sz

    # Padding amounts needed to keep the window in-bounds
    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - w)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        mean_bgr = im.mean(axis=(0, 1)).astype(np.uint8).tolist()
        im = cv2.copyMakeBorder(
            im,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=mean_bgr,
        )
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    crop = im[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_sz, out_sz), interpolation=cv2.INTER_LINEAR)
    # BGR → RGB (make contiguous to avoid downstream strides issues)
    crop = np.ascontiguousarray(crop[:, :, ::-1])
    return crop


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_crops_z(im: np.ndarray, pos_y: float, pos_x: float,
                    z_sz: float, exemplar_sz: int = 127,
                    fast: bool = True) -> Image.Image | np.ndarray:
    """Return a single template crop of size *exemplar_sz* × *exemplar_sz*.

    Args:
        im:          Source image (H, W, C) BGR uint8 ndarray.
        pos_y:       Vertical centre of the target in *im*.
        pos_x:       Horizontal centre of the target in *im*.
        z_sz:        Side length of the region to crop (before resizing).
        exemplar_sz: Output side length (default 127).
        fast:        When ``True`` (default) use the OpenCV path and return a
                     numpy RGB ndarray.  When ``False`` use PIL and return a
                     PIL Image.

    Returns:
        RGB crop as a PIL Image (``fast=False``) or numpy ndarray (``fast=True``).
    """
    patch_sz = int(round(z_sz))
    if fast:
        return _extract_crop_cv2(im, pos_y, pos_x, patch_sz, exemplar_sz)
    return _extract_crop(im, pos_y, pos_x, patch_sz, exemplar_sz)


def extract_crops_x(im: np.ndarray, pos_y: float, pos_x: float,
                    x_sz: float, scale_factors: list[float],
                    search_sz: int = 255,
                    fast: bool = True) -> list[Image.Image] | list[np.ndarray]:
    """Return one search crop per scale factor.

    Args:
        im:            Source image (H, W, C) BGR uint8 ndarray.
        pos_y:         Vertical centre of the target.
        pos_x:         Horizontal centre of the target.
        x_sz:          Base search crop size at scale 1.0 (pixels).
        scale_factors: List of scale multipliers.
        search_sz:     Output side length for each crop (default 255).
        fast:          When ``True`` (default) use the OpenCV path and return
                       numpy RGB ndarrays.  When ``False`` use PIL.

    Returns:
        List of RGB crops as PIL Images (``fast=False``) or numpy ndarrays
        (``fast=True``).
    """
    crops = []
    for sf in scale_factors:
        patch_sz = int(round(x_sz * sf))
        if fast:
            crops.append(_extract_crop_cv2(im, pos_y, pos_x, patch_sz, search_sz))
        else:
            crops.append(_extract_crop(im, pos_y, pos_x, patch_sz, search_sz))
    return crops


def image_to_tensor(img: Image.Image | np.ndarray,
                    device: torch.device | None = None) -> torch.Tensor:
    """Convert a PIL RGB Image *or* an RGB uint8 numpy ndarray to a normalised
    float tensor.

    Applies the SiameseX.PyTorch normalisation:
        out = (pixel / 255 − 0.5) / 0.25

    Args:
        img:    PIL RGB Image (H × W × 3) or RGB uint8 numpy ndarray.
        device: Target torch device.

    Returns:
        Float tensor of shape (1, 3, H, W).
    """
    if isinstance(img, Image.Image):
        arr = np.array(img).astype(np.float32) / 255.0
    else:
        arr = img.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.25                          # [−2, 2]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def image_to_tensor_batch(imgs: list[np.ndarray],
                          device: torch.device | None = None) -> torch.Tensor:
    """Batch-convert a list of RGB uint8 numpy ndarrays to a single tensor.

    More efficient than calling :func:`image_to_tensor` in a loop because it
    does a single ``np.stack`` followed by a single ``.to(device)`` transfer.

    Args:
        imgs:   List of RGB uint8 ndarrays, each of shape (H, W, 3).
        device: Target torch device.

    Returns:
        Float tensor of shape (N, 3, H, W).
    """
    arr = np.stack(imgs, axis=0).astype(np.float32) / 255.0  # (N, H, W, 3)
    arr = (arr - 0.5) / 0.25
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def gen_xz(im: np.ndarray, pos_y: float, pos_x: float,
           z_sz: float, x_sz: float, scale_factors: list[float],
           exemplar_sz: int = 127, search_sz: int = 255,
           device: torch.device | None = None,
           fast: bool = True,
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate template and search tensors ready for the siamese network.

    Returns:
        (z_tensor, x_tensor) where z has shape (1, 3, exemplar_sz, exemplar_sz)
        and x has shape (n_scales, 3, search_sz, search_sz).
    """
    z_crop = extract_crops_z(im, pos_y, pos_x, z_sz, exemplar_sz, fast=fast)
    x_crops = extract_crops_x(im, pos_y, pos_x, x_sz, scale_factors,
                               search_sz, fast=fast)

    if fast:
        z_t = image_to_tensor(z_crop, device)
        x_t = image_to_tensor_batch(x_crops, device)
    else:
        z_t = image_to_tensor(z_crop, device)
        x_t = torch.cat(
            [image_to_tensor(c, device) for c in x_crops], dim=0
        )
    return z_t, x_t
