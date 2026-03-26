"""test_video_siamx.py — Interactive SiamX-360 tracker demo on an MP4 video.

Reads an MP4 video, lets the user draw a bounding box on the first frame
(using ``cv2.selectROI``), then runs the SiamFC-360 omnidirectional tracker
in real-time with live visualisation.  The annotated output is saved to a
new MP4 file and per-frame / average inference speed is reported.

Usage::

    python scripts/test_video_siamx.py --video /path/to/video.mp4 \
        --model_path trackers/siamX/pretrained_weights/SiamFC.pth

    # Use GPU
    python scripts/test_video_siamx.py --video /path/to/video.mp4 \
        --model_path trackers/siamX/pretrained_weights/SiamFC.pth \
        --device cuda

    # Specify output path
    python scripts/test_video_siamx.py --video /path/to/video.mp4 \
        --output results/demo_out.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'lib')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.omni import OmniImage
from lib.utils import Bbox, Bfov
from trackers.siamfc_omni import SiamFCOmniTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pixel_bbox_to_bfov(omni: OmniImage, x: int, y: int, w: int, h: int) -> Bfov:
    """Convert a pixel-space bounding box on an equirectangular image to BFoV.

    Args:
        omni: OmniImage instance matching the image dimensions.
        x, y: Top-left corner of the bounding box (pixels).
        w, h: Width and height of the bounding box (pixels).

    Returns:
        Bfov object with (clon, clat, fov_h, fov_v, rotation) in degrees.
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    bbox = Bbox(cx, cy, w, h, rotation=0)
    bfov, _u, _v = omni.bbox2Bfov(bbox, need_rotation=False)
    return bfov


def overlay_info(img: np.ndarray, frame_idx: int, fps: float,
                 avg_fps: float) -> np.ndarray:
    """Draw frame index and FPS information on the visualisation frame."""
    text_fps = f"Frame {frame_idx}  FPS: {fps:.1f}  Avg: {avg_fps:.1f}"
    cv2.putText(img, text_fps, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Interactive SiamX-360 tracker test on an MP4 video.'
    )
    parser.add_argument(
        '-v', '--video', required=True,
        help='Path to the input MP4 video file.',
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Path for the output MP4 video.  Defaults to '
             '<video_name>_track.mp4 next to the input.',
    )
    parser.add_argument(
        '-m', '--model_path', default=None,
        help='Path to SiamFC.pth pretrained weights.',
    )
    parser.add_argument(
        '--search_scale', type=float, default=2.0,
        help='BFoV scale factor for the search region (default: 2.0).',
    )
    parser.add_argument(
        '--crop_size', type=int, default=500,
        help='Horizontal pixel resolution of perspective crops (default: 500).',
    )
    parser.add_argument(
        '--device', default='cpu', choices=['cpu', 'cuda'],
        help='Compute device for the SiamFC model.',
    )
    parser.add_argument(
        '--no_show', action='store_true',
        help='Disable live visualisation window (useful on headless servers).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Open video --------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video : {args.video}")
    print(f"Size  : {img_w}x{img_h}  FPS: {fps_video:.1f}  Frames: {total_frames}")

    # --- Read the first frame and let user draw bbox -----------------------
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read the first frame.")
        sys.exit(1)

    print("\n>>> Draw a bounding box on the first frame and press ENTER / SPACE to confirm.")
    print(">>> Press 'c' to cancel selection.\n")
    roi = cv2.selectROI("Select Target", first_frame, fromCenter=False,
                        showCrosshair=True)
    cv2.destroyWindow("Select Target")
    x, y, w, h = roi

    if w == 0 or h == 0:
        print("[ERROR] No bounding box selected — exiting.")
        sys.exit(1)

    print(f"Selected ROI (x, y, w, h): {x}, {y}, {w}, {h}")

    # --- Convert pixel bbox → BFoV ----------------------------------------
    omni = OmniImage(img_w=img_w, img_h=img_h)
    init_bfov = pixel_bbox_to_bfov(omni, x, y, w, h)
    print(f"Initial BFoV: clon={init_bfov.clon:.2f}°  clat={init_bfov.clat:.2f}°  "
          f"fov_h={init_bfov.fov_h:.2f}°  fov_v={init_bfov.fov_v:.2f}°")

    # --- Create tracker and initialise ------------------------------------
    tracker = SiamFCOmniTracker(
        model_path=args.model_path,
        img_w=img_w,
        img_h=img_h,
        search_scale=args.search_scale,
        crop_size=args.crop_size,
        device=args.device,
    )
    tracker.initialize(first_frame, init_bfov)
    print("Tracker initialised.\n")

    # --- Prepare video writer ---------------------------------------------
    if args.output is None:
        base = os.path.splitext(args.video)[0]
        output_path = base + '_track.mp4'
    else:
        output_path = args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps_video, (img_w, img_h))

    # Write first frame with the initial bbox overlay
    vis = omni.plot_bfov(first_frame.copy(), init_bfov, color=(0, 255, 0))
    vis = overlay_info(vis, 0, 0.0, 0.0)
    writer.write(vis)
    if not args.no_show:
        cv2.imshow('SiamX-360 Tracking', vis)
        cv2.waitKey(1)

    # --- Tracking loop -----------------------------------------------------
    frame_idx = 1
    infer_times: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()
        pred_bfov = tracker.track(frame)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        infer_times.append(elapsed)
        cur_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        avg_fps = len(infer_times) / sum(infer_times)

        # Visualise
        vis = omni.plot_bfov(frame.copy(), pred_bfov, color=(0, 255, 0))
        vis = overlay_info(vis, frame_idx, cur_fps, avg_fps)
        writer.write(vis)

        if not args.no_show:
            cv2.imshow('SiamX-360 Tracking', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q' — stopping early.")
                break

        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            print(f"  Frame {frame_idx}/{total_frames}  "
                  f"FPS: {cur_fps:.1f}  Avg: {avg_fps:.1f}")

        frame_idx += 1

    # --- Cleanup -----------------------------------------------------------
    cap.release()
    writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    # --- Summary -----------------------------------------------------------
    if infer_times:
        total_time = sum(infer_times)
        avg_fps = len(infer_times) / total_time
        print(f"\n{'=' * 50}")
        print(f"Tracking finished.")
        print(f"  Frames tracked : {len(infer_times)}")
        print(f"  Total infer    : {total_time:.3f} s")
        print(f"  Average FPS    : {avg_fps:.2f}")
        print(f"  Output saved   : {output_path}")
        print(f"{'=' * 50}")
    else:
        print("\nNo frames were tracked.")


if __name__ == '__main__':
    main()
