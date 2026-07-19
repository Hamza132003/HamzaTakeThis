"""Shared model cache with a 6 GB VRAM budget.

Policy: ONE heavy model on the GPU at a time within a file; keep loaded
models cached (on CPU RAM where the framework allows moving) across files in
a batch so the second file skips the multi-minute cold loads.

CTranslate2 models (faster-whisper) cannot be moved between devices, so for
those "release" means dropping the object entirely; reloading from the local
HF cache takes ~15 s, which is acceptable (quality-first).
"""
from __future__ import annotations

import gc

from .utils import log, free_cuda


class ModelManager:
    def __init__(self):
        self._models: dict[str, object] = {}
        self._on_gpu: set[str] = set()

    def get(self, key: str, loader, device: str = "cpu"):
        """Return the cached model for `key`, calling `loader()` on a miss.

        `loader` receives no arguments and must return the ready model
        (already on the right device).
        """
        if key not in self._models:
            log(f"[models] loading '{key}' ...")
            self._models[key] = loader()
        else:
            log(f"[models] cache hit '{key}'")
        if device == "cuda":
            self._on_gpu.add(key)
        return self._models[key]

    def to_cpu(self, key: str) -> None:
        """Move a cached torch model off the GPU without dropping it."""
        m = self._models.get(key)
        if m is None:
            return
        try:
            m.to("cpu")
            self._on_gpu.discard(key)
            free_cuda()
        except Exception:
            # Not movable (e.g. CTranslate2 / pipeline objects): evict instead.
            self.evict(key)

    def to_gpu(self, key: str) -> None:
        m = self._models.get(key)
        if m is None:
            return
        try:
            m.to("cuda")
            self._on_gpu.add(key)
        except Exception:
            pass

    def evict(self, key: str) -> None:
        if key in self._models:
            log(f"[models] evicting '{key}'")
            del self._models[key]
            self._on_gpu.discard(key)
            gc.collect()
            free_cuda()

    def release_gpu(self, keep: str | None = None) -> None:
        """Push every cached model off the GPU except `keep`."""
        for key in list(self._on_gpu):
            if key != keep:
                self.to_cpu(key)

    def gpu_report(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 2**30
                reserved = torch.cuda.memory_reserved() / 2**30
                return f"cuda alloc {alloc:.2f} GiB / reserved {reserved:.2f} GiB"
        except Exception:
            pass
        return "cuda n/a"


# Singleton shared by the CLI and the Flask worker process.
MANAGER = ModelManager()
