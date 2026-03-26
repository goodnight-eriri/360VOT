"""export_siamfc_onnx.py — Standalone ONNX export and TensorRT engine build script.

Run this script *once* before using ``--use_trt`` in ``test_video_siamx.py``
to pre-export the SiamFC backbone to ONNX and (optionally) build the TensorRT
engine.  This avoids the one-time compilation delay on the first tracking run.

Usage::

    # Export ONNX only
    python scripts/export_siamfc_onnx.py \
        --model_path trackers/siamX/pretrained_weights/SiamFC.pth

    # Export ONNX and build TensorRT engines (requires TensorRT + PyCUDA)
    python scripts/export_siamfc_onnx.py \
        --model_path trackers/siamX/pretrained_weights/SiamFC.pth \
        --build_trt

    # Use FP32 instead of FP16 for TRT
    python scripts/export_siamfc_onnx.py \
        --model_path trackers/siamX/pretrained_weights/SiamFC.pth \
        --build_trt --no_fp16

    # Save outputs to a custom directory
    python scripts/export_siamfc_onnx.py \
        --model_path trackers/siamX/pretrained_weights/SiamFC.pth \
        --output_dir /tmp/siamfc_engines --build_trt
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'lib')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(message)s')

from trackers.siamX.siamx_models.builder import build_siamfc
from trackers.siamX.siamx_fc.trt_backend import export_backbone_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export SiamFC backbone to ONNX and optionally build TRT engines.'
    )
    parser.add_argument(
        '-m', '--model_path', required=True,
        help='Path to SiamFC.pth pretrained weights.',
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Directory to store ONNX / TRT files.  Defaults to the same '
             'directory as model_path.',
    )
    parser.add_argument(
        '--build_trt', action='store_true',
        help='Also build TensorRT engines from the exported ONNX files.',
    )
    parser.add_argument(
        '--no_fp16', action='store_true',
        help='Build TRT engines with FP32 instead of FP16.',
    )
    parser.add_argument(
        '--device', default='cpu',
        help='Torch device used for ONNX tracing (default: cpu).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"[ERROR] model_path not found: {args.model_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.model_path))
    os.makedirs(output_dir, exist_ok=True)

    onnx_z = os.path.join(output_dir, 'siamfc_backbone_z.onnx')
    onnx_x = onnx_z + '.search.onnx'
    engine_z = os.path.join(output_dir, 'siamfc_backbone_z.trt')
    engine_x = os.path.join(output_dir, 'siamfc_backbone_x.trt')

    # ------------------------------------------------------------------
    # Step 1: Load model and export ONNX
    # ------------------------------------------------------------------
    print(f"Loading SiamFC model from: {args.model_path}")
    import torch
    model = build_siamfc(args.model_path)
    model.to(torch.device(args.device))
    model.eval()

    print(f"Exporting backbone ONNX to: {output_dir}")
    export_backbone_onnx(
        model,
        onnx_z,
        template_shape=(1, 3, 127, 127),
        search_shape=(3, 3, 255, 255),
    )
    print(f"  Template ONNX: {onnx_z}")
    print(f"  Search   ONNX: {onnx_x}")

    # ------------------------------------------------------------------
    # Step 2: (Optional) Build TensorRT engines
    # ------------------------------------------------------------------
    if args.build_trt:
        try:
            from trackers.siamX.siamx_fc.trt_backend import _build_engine
        except ImportError as exc:
            print(f"[ERROR] Cannot import trt_backend: {exc}")
            sys.exit(1)

        fp16 = not args.no_fp16
        print(f"\nBuilding TRT engines (fp16={fp16}) …")

        print(f"  Building template engine → {engine_z}")
        _build_engine(onnx_z, engine_z, fp16=fp16)

        print(f"  Building search engine   → {engine_x}")
        _build_engine(onnx_x, engine_x, fp16=fp16)

        print("\nTRT engines built successfully.")
        print(f"  Template engine: {engine_z}")
        print(f"  Search   engine: {engine_x}")
    else:
        print("\nSkipping TRT build (pass --build_trt to also build engines).")

    print("\nDone.")


if __name__ == '__main__':
    main()
