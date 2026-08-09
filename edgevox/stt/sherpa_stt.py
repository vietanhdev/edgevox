"""Sherpa-ONNX Zipformer STT backend for Vietnamese.

Native ONNX transducer (encoder + decoder + joiner), no torch at runtime.

Provenance and licence, verified 2026-08-09:

* Weights: ``csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20``, whose card states
  "Models in this directory are from https://huggingface.co/zzasdf/viet_iter3_pseudo_label".
* That upstream (VietASR, 68M) declares ``license: apache-2.0``.

The sherpa-onnx mirror itself declares no licence, so the upstream is the
authority; keep this note in sync if the pin moves.

Until 2026-08-09 this module pinned ``hynt/Zipformer-30M-RNNT-6000h`` (via the
``…-vi-30M-int8-2026-02-09`` mirror) and this docstring claimed "Apache 2.0".
That upstream is in fact ``cc-by-nc-nd-4.0``: NonCommercial *and* NoDerivatives,
which an MIT project cannot ship, and under which the int8 conversion and our
re-host were themselves derivatives. The replacement is measurably weaker on
VLSP2020-T1 (14.45 vs 12.29 WER, publisher-reported, not measured here); that
regression is the price of a licence we can actually distribute.
"""

from __future__ import annotations

import ctypes
import logging
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from edgevox.stt import BaseSTT


def _preload_onnxruntime():
    """Ensure libonnxruntime.so is discoverable for sherpa-onnx.

    sherpa-onnx links against the unversioned libonnxruntime.so, but the
    onnxruntime pip package ships only a versioned .so (e.g. .so.1.23.2).
    Create an unversioned symlink and add the directory to the search path.
    """
    try:
        import onnxruntime as _ort

        capi_dir = Path(_ort.__file__).parent / "capi"
        unversioned = capi_dir / "libonnxruntime.so"
        if not unversioned.exists():
            versioned = sorted(capi_dir.glob("libonnxruntime.so.*"))
            if versioned:
                import os

                os.symlink(versioned[0].name, str(unversioned))
        if unversioned.exists():
            ctypes.CDLL(str(unversioned), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


_preload_onnxruntime()

log = logging.getLogger(__name__)

# Apache-2.0 upstream (see module docstring). The previous primary,
# ``nrl-ai/edgevox-models`` at ``stt/sherpa-zipformer-vi-30M-int8``, re-hosts
# CC-BY-NC-ND weights and is deliberately NOT consulted any more; those files
# should be removed from that repo separately.
_MODELS_REPO = "csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20"
_ENCODER = "encoder-epoch-12-avg-8.onnx"
_DECODER = "decoder-epoch-12-avg-8.onnx"
_JOINER = "joiner-epoch-12-avg-8.onnx"
_TOKENS = "tokens.txt"
_MODEL_FILES = [_ENCODER, _DECODER, _JOINER, _TOKENS]


def _ensure_model() -> Path:
    """Download the Vietnamese zipformer if not already cached."""
    first = hf_hub_download(_MODELS_REPO, _MODEL_FILES[0])
    model_dir = Path(first).parent
    for f in _MODEL_FILES[1:]:
        hf_hub_download(_MODELS_REPO, f)
    return model_dir


def _pick_provider() -> str:
    from edgevox.core.gpu import has_cuda

    return "cuda" if has_cuda() else "cpu"


class SherpaSTT(BaseSTT):
    """Sherpa-ONNX Zipformer transducer for Vietnamese."""

    def __init__(self, device: str | None = None):
        import sherpa_onnx

        provider = device or _pick_provider()
        log.info(f"Loading Sherpa-ONNX Zipformer-vi on {provider}...")

        model_dir = _ensure_model()

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / _ENCODER),
            decoder=str(model_dir / _DECODER),
            joiner=str(model_dir / _JOINER),
            tokens=str(model_dir / _TOKENS),
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
        )
        self._backend_name = "Sherpa"
        self._model_size = "zipformer-vi-68M"
        self._device = provider
        self._warmed_up = False
        log.info("Sherpa-ONNX loaded (VietASR zipformer, Apache-2.0).")

    def transcribe(self, audio: np.ndarray, language: str = "vi") -> str:
        t0 = time.perf_counter()

        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, audio.astype(np.float32))
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip().capitalize()

        elapsed = time.perf_counter() - t0
        if not self._warmed_up:
            self._warmed_up = True
            log.info(f'STT Sherpa-ONNX (warmup): {elapsed:.2f}s -> "{text}"')
        else:
            log.info(f'STT Sherpa-ONNX: {elapsed:.2f}s -> "{text}"')
        return text
