# SPDX-License-Identifier: Apache-2.0
"""Repo-root conftest — injects MLX stubs on platforms where MLX is unavailable.

On macOS with Apple Silicon the real ``mlx`` package is present and this file
is a no-op.  On Linux CI runners ``mlx`` cannot be installed, so we inject
lightweight stub modules into ``sys.modules`` *before* pytest collects tests.
This lets ``import omlx`` succeed so the pure-Python test suite can run.
MLX-dependent tests still self-skip via their own ``HAS_MLX`` guards.
"""

import sys
import types

try:
    import mlx.core  # noqa: F401
except ImportError:

    def _make_stub(name, attrs=None):
        """Register a stub module in ``sys.modules``."""
        mod = types.ModuleType(name)
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    class _DType:
        """Sentinel standing in for ``mlx.core`` dtype constants."""
        pass

    # ── mlx stubs ──────────────────────────────────────────────────────
    _make_stub("mlx", {"__path__": []})
    _make_stub("mlx.core", {
        "Stream": type("Stream", (), {}),
        "array": type("array", (), {}),
        "float32": _DType(), "float16": _DType(), "bfloat16": _DType(),
        "int8": _DType(), "int16": _DType(), "int32": _DType(), "int64": _DType(),
        "uint8": _DType(), "uint16": _DType(), "uint32": _DType(), "uint64": _DType(),
        "bool_": _DType(),
    })
    _make_stub("mlx.core.fast", {})
    _make_stub("mlx.core.linalg", {})
    _make_stub("mlx.core.random", {})
    _make_stub("mlx.core.metal", {})
    _make_stub("mlx.nn", {"Module": type("Module", (), {})})
    _make_stub("mlx.utils", {
        "tree_flatten": lambda *a, **k: [],
        "tree_unflatten": lambda *a, **k: None,
    })

    # ── mlx-lm stubs ──────────────────────────────────────────────────
    _make_stub("mlx_lm", {"__path__": []})
    _make_stub("mlx_lm.generate", {
        "Batch": type("Batch", (), {}),
        "BatchGenerator": type("BatchGenerator", (), {}),
        "generation_stream": object(),
        "_left_pad_prompts": lambda *a, **k: None,
        "_make_cache": lambda *a, **k: None,
        "_merge_caches": lambda *a, **k: None,
        "_right_pad_prompts": lambda *a, **k: None,
    })
    _make_stub("mlx_lm.sample_utils", {
        "make_sampler": lambda *a, **k: None,
        "make_logits_processors": lambda *a, **k: None,
        "make_presence_penalty": lambda *a, **k: None,
    })
    _make_stub("mlx_lm.models", {"__path__": []})
    _make_stub("mlx_lm.models.cache", {
        "KVCache": type("KVCache", (), {}),
        "RotatingKVCache": type("RotatingKVCache", (), {}),
        "BatchRotatingKVCache": type("BatchRotatingKVCache", (), {}),
        "CacheList": type("CacheList", (), {}),
        "make_prompt_cache": lambda *a, **k: None,
    })
    _make_stub("mlx_lm.tokenizer_utils", {
        "NaiveStreamingDetokenizer": type("NaiveStreamingDetokenizer", (), {}),
    })

    # ── mlx-vlm stubs ─────────────────────────────────────────────────
    _make_stub("mlx_vlm", {"__path__": []})
    _make_stub("mlx_vlm.utils", {})
