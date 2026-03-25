"""run_siamfc_360.py — Run SiamFC on the 360VOT benchmark.

This script applies the SiamFC omnidirectional tracker to one or all
sequences in the 360VOT dataset and writes per-sequence result files in the
format expected by ``scripts/eval_360VOT.py``.

Output format (one line per frame)::

    clon,clat,fov_h,fov_v,rotation

Example usage::

    # Track a single sequence
    python scripts/run_siamfc_360.py \\
        --dataset_dir /data/360VOT \\
        --sequence    0001 \\
        --output_dir  results/SiamFC \\
        --model_path  trackers/siamX/pretrained_weights/SiamFC.pth

    # Track all sequences
    python scripts/run_siamfc_360.py \\
        --dataset_dir /data/360VOT \\
        --output_dir  results/SiamFC \\
        --model_path  trackers/siamX/pretrained_weights/SiamFC.pth

    # Visualise tracking (first sequence only)
    python scripts/run_siamfc_360.py \\
        --dataset_dir /data/360VOT \\
        --sequence    0001 \\
        --output_dir  results/SiamFC \\
        --model_path  trackers/siamX/pretrained_weights/SiamFC.pth \\
        --vis

Dataset directory structure assumed::

    <dataset_dir>/
      <sequence>/
        img/                      ← frame images (*.jpg or *.png)
        label.json                ← annotations with keys "bfov" per frame
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import tqdm

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root
# ---------------------------------------------------------------------------
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'lib')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.utils import dict2Bfov, Bfov
from trackers.siamfc_omni import SiamFCOmniTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sequence(seq_dir: str) -> tuple[list[str], list[Bfov]]:
    """Load frame paths and BFoV annotations for a single sequence.

    Returns:
        (frame_paths, bfov_list) — both lists have the same length.
    """
    label_path = os.path.join(seq_dir, 'label.json')
    with open(label_path, 'r') as f:
        annos = json.load(f)

    img_dir = os.path.join(seq_dir, 'img')
    frame_names = sorted(os.listdir(img_dir))

    frame_paths = [os.path.join(img_dir, n) for n in frame_names]

    # Build ordered list of BFoV annotations matching frame order
    # label.json keys are string-formatted frame indices
    bfovs: list[Bfov] = []
    for key in sorted(annos.keys(), key=lambda k: int(k)):
        annotation = annos[key]
        bfovs.append(dict2Bfov(annotation['bfov']))

    # Trim to shortest list length in case of mismatch
    n = min(len(frame_paths), len(bfovs))
    return frame_paths[:n], bfovs[:n]


def bfov_to_line(bfov: Bfov) -> str:
    """Format a Bfov as a comma-separated result line."""
    return f"{bfov.clon:.6f},{bfov.clat:.6f},{bfov.fov_h:.6f},{bfov.fov_v:.6f},{bfov.rotation:.6f}"


# ---------------------------------------------------------------------------
# Per-sequence tracking
# ---------------------------------------------------------------------------

def track_sequence(
    seq_dir:      str,
    output_file:  str,
    model_path:   str | None,
    search_scale: float,
    crop_size:    int,
    device:       str,
    visualise:    bool,
) -> None:
    """Track a single sequence and write results to *output_file*."""

    frame_paths, bfovs = load_sequence(seq_dir)
    if len(frame_paths) == 0:
        print(f"  [WARN] No frames found in {seq_dir}, skipping.")
        return

    # Infer image dimensions from first frame
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print(f"  [WARN] Cannot read {frame_paths[0]}, skipping.")
        return
    img_h, img_w = first_frame.shape[:2]

    tracker = SiamFCOmniTracker(
        model_path=model_path,
        img_w=img_w,
        img_h=img_h,
        search_scale=search_scale,
        crop_size=crop_size,
        device=device,
    )

    results: list[str] = []

    for idx, (fpath, gt_bfov) in enumerate(
        tqdm.tqdm(zip(frame_paths, bfovs), total=len(frame_paths),
                  desc=os.path.basename(seq_dir), leave=False)
    ):
        frame = cv2.imread(fpath)
        if frame is None:
            # Write the last result again to keep frame count consistent
            results.append(results[-1] if results else "0,0,1,1,0")
            continue

        if idx == 0:
            tracker.initialize(frame, gt_bfov)
            pred_bfov = gt_bfov
        else:
            pred_bfov = tracker.track(frame)

        results.append(bfov_to_line(pred_bfov))

        if visualise:
            vis = tracker.omni.plot_bfov(frame.copy(), pred_bfov)
            cv2.imshow('SiamFC 360', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if visualise:
        cv2.destroyAllWindows()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(results) + '\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run SiamFC on 360VOT sequences.'
    )
    parser.add_argument(
        '-d', '--dataset_dir', required=True,
        help='Root directory of the 360VOT dataset.',
    )
    parser.add_argument(
        '-s', '--sequence', default=None,
        help='Name of a single sequence to track.  Omit to track all.',
    )
    parser.add_argument(
        '-o', '--output_dir', default='results/SiamFC',
        help='Directory where result .txt files are saved.',
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
        '--vis', action='store_true',
        help='Show live visualisation (press Q to quit).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir
    output_dir  = os.path.join(args.output_dir, 'SiamFC')  # sub-dir per tracker name

    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = sorted(os.listdir(dataset_dir))

    print(f"Tracking {len(sequences)} sequence(s) → {output_dir}")

    for seq in tqdm.tqdm(sequences, desc='sequences'):
        seq_dir     = os.path.join(dataset_dir, seq)
        output_file = os.path.join(output_dir, seq + '.txt')

        if not os.path.isdir(seq_dir):
            continue

        if os.path.exists(output_file):
            print(f"  [SKIP] {seq} (result already exists)")
            continue

        try:
            track_sequence(
                seq_dir=seq_dir,
                output_file=output_file,
                model_path=args.model_path,
                search_scale=args.search_scale,
                crop_size=args.crop_size,
                device=args.device,
                visualise=args.vis,
            )
        except Exception as exc:
            print(f"  [ERROR] {seq}: {exc}")


if __name__ == '__main__':
    main()
