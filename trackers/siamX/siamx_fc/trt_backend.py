"""TensorRT inference backend for SiamFC backbone.

Provides :class:`TRTBackbone` which wraps the AlexNet feature extractor in a
TensorRT engine for accelerated inference on NVIDIA Jetson (and other TRT-
capable devices).  When TensorRT or PyCUDA are not available the module
degrades gracefully: importing it will succeed and ``TRTBackbone`` can still
be instantiated, but a :class:`RuntimeError` is raised at runtime so callers
can fall back to the PyTorch path.

Typical usage::

    from trackers.siamX.siamx_fc.trt_backend import export_backbone_onnx, TRTBackbone

    # One-time export (skip if files already exist)
    export_backbone_onnx(pytorch_model, onnx_path)

    # Build / load engine (cached after first run)
    backbone_trt = TRTBackbone(onnx_path, engine_cache_path, fp16=True)

    # Drop-in replacement for model.features(tensor)
    out_np = backbone_trt(input_np)          # numpy in, numpy out
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — fail gracefully
# ---------------------------------------------------------------------------
try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  — initializes CUDA context
    _PYCUDA_AVAILABLE = True
except ImportError:
    _PYCUDA_AVAILABLE = False


def _trt_available() -> bool:
    return _TRT_AVAILABLE and _PYCUDA_AVAILABLE


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------

def export_backbone_onnx(model, onnx_path: str,
                          template_shape: tuple = (1, 3, 127, 127),
                          search_shape: tuple = (3, 3, 255, 255)) -> None:
    """Export the SiamFC backbone (``model.features``) to ONNX.

    Two ONNX files are produced:
    * ``<onnx_path>`` — for the template (static batch=1).
    * ``<onnx_path>.search.onnx`` — for the search crops (batch=num_scales).

    Args:
        model:          :class:`SiamFC` PyTorch model (already on a device).
        onnx_path:      Destination path for the *template* ONNX file.
        template_shape: Input shape for the template branch ``(N,C,H,W)``.
        search_shape:   Input shape for the search branch  ``(N,C,H,W)``.
    """
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    search_onnx_path = onnx_path + ".search.onnx"

    features = model.features
    features.eval()

    # --- Template branch ---------------------------------------------------
    if not os.path.exists(onnx_path):
        dummy_z = torch.zeros(template_shape,
                              dtype=torch.float32,
                              device=next(features.parameters()).device)
        torch.onnx.export(
            features,
            dummy_z,
            onnx_path,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None,
            do_constant_folding=True,
        )
        logger.info("Exported template backbone ONNX → %s", onnx_path)

    # --- Search branch (larger batch) --------------------------------------
    if not os.path.exists(search_onnx_path):
        dummy_x = torch.zeros(search_shape,
                              dtype=torch.float32,
                              device=next(features.parameters()).device)
        torch.onnx.export(
            features,
            dummy_x,
            search_onnx_path,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None,
            do_constant_folding=True,
        )
        logger.info("Exported search backbone ONNX → %s", search_onnx_path)


# ---------------------------------------------------------------------------
# TensorRT engine builder
# ---------------------------------------------------------------------------

_TRT_WORKSPACE_BYTES = 1 << 30  # 1 GiB workspace for TRT engine build


def _build_engine(onnx_path: str, engine_path: str, fp16: bool = True) -> None:
    """Build a TensorRT engine from *onnx_path* and serialise it to *engine_path*.

    Args:
        onnx_path:   Path to the ONNX model.
        engine_path: Destination for the serialised engine file.
        fp16:        Enable FP16 precision (requires hardware support).
    """
    if not _TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not installed.")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _TRT_WORKSPACE_BYTES)

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("TRT engine will use FP16 precision.")
        else:
            logger.info("TRT engine will use FP32 precision.")

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i))
                          for i in range(parser.num_errors)]
                raise RuntimeError(
                    "ONNX parsing failed:\n" + "\n".join(errors)
                )

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build failed.")

        os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized)
        logger.info("TRT engine saved → %s", engine_path)


# ---------------------------------------------------------------------------
# TRTBackbone
# ---------------------------------------------------------------------------

class TRTBackbone:
    """TensorRT-accelerated backbone that mimics ``model.features(tensor)``.

    Pre-allocates pinned host and device buffers for zero-copy transfers.

    Args:
        onnx_path:         Path to the ONNX model file.
        engine_cache_path: Path where the serialised TRT engine is cached.
        fp16:              Build the engine with FP16 precision when ``True``.
        max_batch:         Maximum batch size supported by the engine.
    """

    def __init__(self, onnx_path: str, engine_cache_path: str,
                 fp16: bool = True, max_batch: int = 4):
        if not _trt_available():
            raise RuntimeError(
                "TensorRT and/or PyCUDA are not installed. "
                "Install them or use use_trt=False."
            )

        self._engine_path = engine_cache_path
        self._onnx_path = onnx_path
        self.fp16 = fp16
        self.max_batch = max_batch

        self._engine = None
        self._context = None
        self._stream = cuda.Stream()

        # Lazy buffers — allocated on first call
        self._h_input: Optional[np.ndarray] = None
        self._h_output: Optional[np.ndarray] = None
        self._d_input = None
        self._d_output = None
        self._input_shape: Optional[tuple] = None
        self._output_shape: Optional[tuple] = None

        self._load_or_build_engine()

    # ------------------------------------------------------------------
    # Engine management
    # ------------------------------------------------------------------

    def _load_or_build_engine(self) -> None:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        if not os.path.exists(self._engine_path):
            logger.info("No cached TRT engine found — building from ONNX …")
            _build_engine(self._onnx_path, self._engine_path, fp16=self.fp16)

        with open(self._engine_path, 'rb') as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialise TRT engine from {self._engine_path}."
            )
        self._context = self._engine.create_execution_context()
        logger.info("TRT engine loaded from %s", self._engine_path)

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _allocate_buffers(self, input_shape: tuple) -> None:
        """Allocate (or reallocate) pinned host + device buffers."""
        if self._engine is None:
            raise RuntimeError("Engine not loaded.")

        # Determine output shape by running a dummy inference query
        # We rely on TRT binding names 'input' and 'output'
        in_idx  = self._engine.get_binding_index('input')
        out_idx = self._engine.get_binding_index('output')

        # Set dynamic batch if needed (static shapes for our use-case)
        self._context.set_binding_shape(in_idx, input_shape)
        output_shape = tuple(self._context.get_binding_shape(out_idx))

        nbytes_in  = int(np.prod(input_shape))  * np.dtype(np.float32).itemsize
        nbytes_out = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize

        # Free old allocations
        if self._d_input is not None:
            self._d_input.free()
        if self._d_output is not None:
            self._d_output.free()

        self._h_input  = cuda.pagelocked_empty(int(np.prod(input_shape)),
                                                dtype=np.float32)
        self._h_output = cuda.pagelocked_empty(int(np.prod(output_shape)),
                                                dtype=np.float32)
        self._d_input  = cuda.mem_alloc(nbytes_in)
        self._d_output = cuda.mem_alloc(nbytes_out)

        self._input_shape  = input_shape
        self._output_shape = output_shape

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run TRT inference.

        Args:
            x: Float32 numpy array of shape ``(N, 3, H, W)``.

        Returns:
            Float32 numpy array of the backbone output, shape
            ``(N, C_out, H_out, W_out)``.
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        x = np.ascontiguousarray(x, dtype=np.float32)

        in_shape = x.shape
        if self._input_shape != in_shape:
            self._allocate_buffers(in_shape)

        # H2D
        np.copyto(self._h_input, x.ravel())
        cuda.memcpy_htod_async(self._d_input, self._h_input, self._stream)

        # Execute
        in_idx  = self._engine.get_binding_index('input')
        out_idx = self._engine.get_binding_index('output')
        bindings = [None] * self._engine.num_bindings
        bindings[in_idx]  = int(self._d_input)
        bindings[out_idx] = int(self._d_output)
        self._context.execute_async_v2(bindings, self._stream.handle)

        # D2H
        cuda.memcpy_dtoh_async(self._h_output, self._d_output, self._stream)
        self._stream.synchronize()

        return self._h_output.reshape(self._output_shape).copy()
