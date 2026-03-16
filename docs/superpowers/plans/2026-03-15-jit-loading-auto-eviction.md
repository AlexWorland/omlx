# JIT Model Loading & Auto Eviction — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add just-in-time model loading, hot-add via periodic rescan, and time-based auto eviction to omlx.

**Architecture:** Extend existing `EnginePool` with `loading_event` on `EngineEntry` for concurrent request coordination, two-phase locking in `get_engine()` to avoid blocking loaded models during JIT loads, global TTL fallback in `check_ttl_expirations()`, and a periodic rescan loop for hot-add. All settings exposed via admin dashboard.

**Tech Stack:** Python 3.11+, asyncio, FastAPI, MLX, Jinja2, pytest

**Spec:** `docs/design-jit-loading-auto-eviction.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `omlx/settings.py` | Modify | Add `global_ttl_seconds`, `model_scan_interval_seconds`, `jit_loading_behavior` to `MemorySettings` |
| `omlx/engine_pool.py` | Modify | Add `loading_event` to `EngineEntry`, store `_model_dirs`, two-phase locking in `get_engine()`, global TTL fallback, `rescan_models()` |
| `omlx/server.py` | Modify | Wire rescan loop, pass `global_ttl_seconds` to existing TTL infrastructure |
| `omlx/admin/routes.py` | Modify | Extend model list with lifecycle fields, expose new settings, update `/load` endpoint |
| `omlx/admin/i18n/en.json` | Modify | Translation keys for lifecycle settings |
| `omlx/admin/templates/settings.html` | Modify | "Model Lifecycle" settings section |
| `omlx/admin/static/js/dashboard.js` | Modify | Model status indicators |
| `tests/test_jit_loading.py` | Create | Full test suite for JIT loading, TTL eviction, rescan |

---

## Chunk 1: Settings + TTL Eviction (Core Infrastructure)

### Task 1: Add lifecycle settings to MemorySettings

**Files:**
- Modify: `omlx/settings.py:322-395` (MemorySettings class)
- Test: `tests/test_jit_loading.py` (new file)

- [ ] **Step 1: Create test file with settings validation tests**

Create `tests/test_jit_loading.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for JIT model loading, TTL eviction, and model rescanning."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.settings import MemorySettings


# =========================================================================
# Settings Validation
# =========================================================================


class TestLifecycleSettings:
    """Test new lifecycle settings in MemorySettings."""

    def test_defaults(self):
        s = MemorySettings()
        assert s.global_ttl_seconds == 180
        assert s.model_scan_interval_seconds == 60
        assert s.jit_loading_behavior == "block"

    def test_global_ttl_zero_disables(self):
        s = MemorySettings(global_ttl_seconds=0)
        assert s.global_ttl_seconds == 0

    def test_scan_interval_minimum(self):
        with pytest.raises(ValueError, match="model_scan_interval_seconds"):
            MemorySettings(model_scan_interval_seconds=5)

    def test_scan_interval_zero_disables(self):
        s = MemorySettings(model_scan_interval_seconds=0)
        assert s.model_scan_interval_seconds == 0

    def test_jit_loading_behavior_invalid(self):
        with pytest.raises(ValueError, match="jit_loading_behavior"):
            MemorySettings(jit_loading_behavior="queue")

    def test_jit_loading_behavior_reject(self):
        s = MemorySettings(jit_loading_behavior="reject")
        assert s.jit_loading_behavior == "reject"
```

- [ ] **Step 2: Run tests — expect FAIL (fields don't exist yet)**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestLifecycleSettings -v`
Expected: FAIL — `MemorySettings` has no `global_ttl_seconds` field

- [ ] **Step 3: Add fields to MemorySettings**

In `omlx/settings.py`, add these fields to the `MemorySettings` dataclass (after line 347, the system watermark fields):

```python
    # JIT loading and model lifecycle
    global_ttl_seconds: int = 180  # Default TTL for models without per-model TTL. 0 = disabled.
    model_scan_interval_seconds: int = 60  # Rescan interval. 0 = disabled, minimum 10.
    jit_loading_behavior: str = "block"  # "block" or "reject" (503)
```

Then add validation in the `validate()` method (after the system watermark validation block):

```python
        # JIT loading settings validation
        if self.global_ttl_seconds < 0:
            raise ValueError(
                f"global_ttl_seconds must be >= 0, got {self.global_ttl_seconds}"
            )
        if self.model_scan_interval_seconds != 0 and self.model_scan_interval_seconds < 10:
            raise ValueError(
                f"model_scan_interval_seconds must be 0 (disabled) or >= 10, "
                f"got {self.model_scan_interval_seconds}"
            )
        if self.jit_loading_behavior not in ("block", "reject"):
            raise ValueError(
                f"jit_loading_behavior must be 'block' or 'reject', "
                f"got '{self.jit_loading_behavior}'"
            )
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestLifecycleSettings -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add omlx/settings.py tests/test_jit_loading.py
git commit -m "feat: add lifecycle settings (global_ttl, scan_interval, jit_behavior)"
```

---

### Task 2: Add `loading_event` to EngineEntry

**Files:**
- Modify: `omlx/engine_pool.py:47-61` (EngineEntry dataclass)

- [ ] **Step 1: Add tests for loading_event coordination**

Append to `tests/test_jit_loading.py`:

```python
from omlx.engine_pool import EngineEntry


class TestLoadingEvent:
    """Test loading_event field on EngineEntry."""

    def test_loading_event_default_set(self):
        """New entries have loading_event already set (not blocking)."""
        entry = EngineEntry(
            model_id="test",
            model_path="/tmp/test",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        assert entry.loading_event.is_set()

    def test_loading_event_can_be_cleared(self):
        entry = EngineEntry(
            model_id="test",
            model_path="/tmp/test",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        entry.loading_event.clear()
        assert not entry.loading_event.is_set()

    @pytest.mark.asyncio
    async def test_concurrent_waiters_unblock(self):
        """Multiple waiters unblock when loading_event is set."""
        entry = EngineEntry(
            model_id="test",
            model_path="/tmp/test",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        entry.loading_event.clear()
        results = []

        async def waiter(idx):
            await entry.loading_event.wait()
            results.append(idx)

        tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
        await asyncio.sleep(0.01)
        assert len(results) == 0  # All blocked

        entry.loading_event.set()
        await asyncio.gather(*tasks)
        assert sorted(results) == [0, 1, 2]
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestLoadingEvent -v`
Expected: FAIL — `EngineEntry` has no `loading_event` field

- [ ] **Step 3: Add `loading_event` field to EngineEntry**

In `omlx/engine_pool.py`, add import at the top (near other imports):

```python
from dataclasses import dataclass, field
```

Then add the field to `EngineEntry` (after `abort_loading` on line 60):

```python
    loading_event: asyncio.Event = field(default_factory=lambda: _make_set_event())
```

Add this helper function before the `EngineEntry` class (around line 45):

```python
def _make_set_event() -> asyncio.Event:
    """Create an asyncio.Event that starts in the set state."""
    event = asyncio.Event()
    event.set()
    return event
```

Note: The `default_factory` creates the event already set, so existing code (which doesn't use loading_event) is unaffected. The JIT load path will `clear()` it before loading and `set()` it when done.

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestLoadingEvent -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_memory_pressure.py -v`
Expected: All 81 tests PASS (loading_event default doesn't affect existing code)

- [ ] **Step 6: Commit**

```bash
git add omlx/engine_pool.py tests/test_jit_loading.py
git commit -m "feat: add loading_event to EngineEntry for concurrent request coordination"
```

---

### Task 3: Add global TTL fallback to `check_ttl_expirations()`

**Files:**
- Modify: `omlx/engine_pool.py:653-702` (check_ttl_expirations method)
- Test: `tests/test_jit_loading.py`

- [ ] **Step 1: Write TTL eviction tests**

Append to `tests/test_jit_loading.py`:

```python
from omlx.engine_pool import EnginePool


def _make_mock_entry(model_id, engine=None, is_pinned=False, last_access=0.0, is_loading=False):
    """Create a mock EngineEntry for TTL tests."""
    entry = MagicMock()
    entry.model_id = model_id
    entry.engine = engine
    entry.is_pinned = is_pinned
    entry.is_loading = is_loading
    entry.abort_loading = False
    entry.last_access = last_access
    entry.loading_event = asyncio.Event()
    entry.loading_event.set()
    return entry


def _make_mock_pool():
    """Create a minimal EnginePool mock."""
    pool = MagicMock(spec=EnginePool)
    pool._entries = {}
    pool._lock = asyncio.Lock()
    pool._unload_engine = AsyncMock()
    pool.check_ttl_expirations = EnginePool.check_ttl_expirations.__get__(pool)
    return pool


def _make_mock_settings_manager(settings_map=None):
    """Create a mock ModelSettingsManager."""
    mgr = MagicMock()
    settings_map = settings_map or {}

    def get_settings(model_id):
        s = MagicMock()
        if model_id in settings_map:
            s.ttl_seconds = settings_map[model_id]
        else:
            s.ttl_seconds = None
        return s

    mgr.get_settings = get_settings
    return mgr


class TestTTLEviction:
    """Test global TTL fallback in check_ttl_expirations."""

    @pytest.mark.asyncio
    async def test_global_ttl_evicts_idle_model(self):
        """Model with no per-model TTL uses global_ttl_seconds."""
        pool = _make_mock_pool()
        engine = MagicMock()
        entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 200)
        pool._entries = {"m1": entry}

        settings_mgr = _make_mock_settings_manager()  # no per-model TTL
        expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
        assert "m1" in expired

    @pytest.mark.asyncio
    async def test_per_model_ttl_overrides_global(self):
        """Per-model TTL takes precedence over global."""
        pool = _make_mock_pool()
        engine = MagicMock()
        entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 100)
        pool._entries = {"m1": entry}

        settings_mgr = _make_mock_settings_manager({"m1": 60})  # 60s per-model
        expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
        assert "m1" in expired  # idle 100s > 60s per-model TTL

    @pytest.mark.asyncio
    async def test_per_model_ttl_zero_never_evicts(self):
        """Per-model TTL of 0 means never auto-evict."""
        pool = _make_mock_pool()
        engine = MagicMock()
        entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 99999)
        pool._entries = {"m1": entry}

        settings_mgr = _make_mock_settings_manager({"m1": 0})  # 0 = disabled
        expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
        assert "m1" not in expired

    @pytest.mark.asyncio
    async def test_global_ttl_zero_disables_all(self):
        """global_ttl_seconds=0 disables TTL for models without per-model TTL."""
        pool = _make_mock_pool()
        engine = MagicMock()
        entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 99999)
        pool._entries = {"m1": entry}

        settings_mgr = _make_mock_settings_manager()  # no per-model TTL
        expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=0)
        assert "m1" not in expired

    @pytest.mark.asyncio
    async def test_pinned_exempt_from_ttl(self):
        """Pinned models are never TTL-evicted."""
        pool = _make_mock_pool()
        engine = MagicMock()
        entry = _make_mock_entry("m1", engine=engine, is_pinned=True, last_access=time.time() - 99999)
        pool._entries = {"m1": entry}

        settings_mgr = _make_mock_settings_manager()
        expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
        assert "m1" not in expired

    @pytest.mark.asyncio
    async def test_ttl_skips_active_requests(self):
        """Models with active requests are not evicted."""
        pool = _make_mock_pool()
        engine = MagicMock()
        # Simulate active output collectors (deep attribute chain)
        inner_engine = MagicMock()
        inner_engine._output_collectors = {"req1": MagicMock()}
        engine_core = MagicMock()
        engine_core.engine = inner_engine
        engine._engine = engine_core
        engine.__class__.__name__ = "BatchedEngine"

        entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 200)
        # Make isinstance check work
        with patch("omlx.engine_pool.BatchedEngine", type(engine)):
            pool._entries = {"m1": entry}
            settings_mgr = _make_mock_settings_manager()
            expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
            assert "m1" not in expired

    @pytest.mark.asyncio
    async def test_evicted_model_stays_in_entries(self):
        """After TTL eviction, entry stays with engine=None (Discovered state)."""
        pool = _make_mock_pool()
        engine = MagicMock()
        entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 200)
        pool._entries = {"m1": entry}

        async def mock_unload(mid):
            pool._entries[mid].engine = None

        pool._unload_engine = mock_unload
        pool.check_ttl_expirations = EnginePool.check_ttl_expirations.__get__(pool)

        settings_mgr = _make_mock_settings_manager()
        expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
        assert "m1" in expired
        assert "m1" in pool._entries  # Still in entries
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestTTLEviction -v`
Expected: FAIL — `check_ttl_expirations()` doesn't accept `global_ttl_seconds` parameter

- [ ] **Step 3: Add `global_ttl_seconds` parameter to `check_ttl_expirations()`**

Modify `omlx/engine_pool.py`, the `check_ttl_expirations` method signature (line 653):

```python
    async def check_ttl_expirations(
        self, settings_manager: ModelSettingsManager, global_ttl_seconds: int = 0
    ) -> list[str]:
```

Then modify the TTL resolution logic inside the method. Replace lines 675-677:

```python
                settings = settings_manager.get_settings(model_id)
                if settings.ttl_seconds is None:
                    continue
```

With:

```python
                settings = settings_manager.get_settings(model_id)
                # Determine effective TTL: per-model overrides global
                if settings.ttl_seconds is not None:
                    effective_ttl = settings.ttl_seconds
                else:
                    effective_ttl = global_ttl_seconds

                # TTL of 0 means disabled (never auto-evict)
                if effective_ttl <= 0:
                    continue
```

And replace line 680:

```python
                if idle_time < settings.ttl_seconds:
```

With:

```python
                if idle_time < effective_ttl:
```

And update the log message on line 700 to use `effective_ttl`:

```python
                    f"(idle {idle_time:.0f}s > ttl {effective_ttl}s)"
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestTTLEviction -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_memory_pressure.py -v`
Expected: All 81 tests PASS (default `global_ttl_seconds=0` preserves old behavior where models without per-model TTL are skipped)

- [ ] **Step 6: Commit**

```bash
git add omlx/engine_pool.py tests/test_jit_loading.py
git commit -m "feat: add global TTL fallback to check_ttl_expirations"
```

---

### Task 4: Wire global TTL into server lifespan

**Files:**
- Modify: `omlx/server.py:287-361` (lifespan function)

- [ ] **Step 1: Update TTL check loop to pass `global_ttl_seconds`**

In `omlx/server.py`, modify the `_ttl_check_loop` inner function (line 327-338).

Change the call on lines 331-333 from:

```python
                        await _server_state.engine_pool.check_ttl_expirations(
                            _server_state.settings_manager
                        )
```

To:

```python
                        global_ttl = 0
                        if _server_state.global_settings is not None:
                            global_ttl = _server_state.global_settings.memory.global_ttl_seconds
                        await _server_state.engine_pool.check_ttl_expirations(
                            _server_state.settings_manager,
                            global_ttl_seconds=global_ttl,
                        )
```

Also need to check if the enforcer's polling loop calls `check_ttl_expirations` — if so, update that call too. Search `process_memory_enforcer.py` for `check_ttl_expirations` and update to pass the global TTL.

- [ ] **Step 2: Verify existing tests still pass**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add omlx/server.py omlx/process_memory_enforcer.py
git commit -m "feat: wire global_ttl_seconds into server TTL check loops"
```

---

### Task 5: Store `_model_dirs` and implement `rescan_models()`

**Files:**
- Modify: `omlx/engine_pool.py:63-175` (EnginePool.__init__, discover_models)
- Test: `tests/test_jit_loading.py`

- [ ] **Step 1: Write rescan tests**

Append to `tests/test_jit_loading.py`:

```python
class TestRescanModels:
    """Test periodic model rescanning."""

    @pytest.mark.asyncio
    async def test_rescan_finds_new_model(self):
        """New model directories are detected on rescan."""
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        # Simulate initial discovery with one model
        pool._entries = {
            "model-a": _make_mock_entry("model-a"),
        }
        pool._model_dirs = ["/tmp/models"]

        # Mock discover_models to return two models
        with patch("omlx.engine_pool.discover_models") as mock_discover:
            mock_discover.return_value = {
                "model-a": MagicMock(model_path="/tmp/models/model-a", model_type="llm",
                                      engine_type="batched", estimated_size=1000, config_model_type=""),
                "model-b": MagicMock(model_path="/tmp/models/model-b", model_type="llm",
                                      engine_type="batched", estimated_size=2000, config_model_type=""),
            }
            added, removed = pool.rescan_models()

        assert "model-b" in pool._entries
        assert "model-b" in added
        assert len(removed) == 0

    @pytest.mark.asyncio
    async def test_rescan_removes_missing_unloaded_model(self):
        """Model whose files disappeared is removed if not loaded."""
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        pool._entries = {
            "model-a": _make_mock_entry("model-a"),
            "model-b": _make_mock_entry("model-b"),  # engine=None → not loaded
        }
        pool._model_dirs = ["/tmp/models"]

        with patch("omlx.engine_pool.discover_models") as mock_discover:
            mock_discover.return_value = {
                "model-a": MagicMock(model_path="/tmp/models/model-a", model_type="llm",
                                      engine_type="batched", estimated_size=1000, config_model_type=""),
            }
            added, removed = pool.rescan_models()

        assert "model-b" not in pool._entries
        assert "model-b" in removed

    @pytest.mark.asyncio
    async def test_rescan_keeps_loaded_model_even_if_files_gone(self):
        """Loaded model is kept even if its files disappeared."""
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        loaded_entry = _make_mock_entry("model-b", engine=MagicMock())
        pool._entries = {
            "model-a": _make_mock_entry("model-a"),
            "model-b": loaded_entry,  # engine is set → loaded
        }
        pool._model_dirs = ["/tmp/models"]

        with patch("omlx.engine_pool.discover_models") as mock_discover:
            mock_discover.return_value = {
                "model-a": MagicMock(model_path="/tmp/models/model-a", model_type="llm",
                                      engine_type="batched", estimated_size=1000, config_model_type=""),
            }
            added, removed = pool.rescan_models()

        assert "model-b" in pool._entries  # Still there
        assert "model-b" not in removed
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestRescanModels -v`
Expected: FAIL — `rescan_models()` doesn't exist, `_model_dirs` doesn't exist

- [ ] **Step 3: Implement `_model_dirs` storage and `rescan_models()`**

In `omlx/engine_pool.py`:

1. In `__init__` (around line 80), add: `self._model_dirs: list[Path] = []`

2. In `discover_models()` (line 132), after creating `dirs`, add: `self._model_dirs = dirs`

3. Add `rescan_models()` method after `discover_models()` (around line 175):

```python
    def rescan_models(self) -> tuple[list[str], list[str]]:
        """Rescan model directories for new or removed models.

        Returns:
            Tuple of (added_model_ids, removed_model_ids).
        """
        from pathlib import Path
        from .model_discovery import discover_models_from_dirs

        if not self._model_dirs:
            return [], []

        if len(self._model_dirs) == 1:
            discovered = discover_models(self._model_dirs[0])
        else:
            discovered = discover_models_from_dirs(self._model_dirs)

        # Determine pinned status from current entries
        pinned_set = {mid for mid, e in self._entries.items() if e.is_pinned}

        added = []
        for model_id, info in discovered.items():
            if model_id not in self._entries:
                self._entries[model_id] = EngineEntry(
                    model_id=model_id,
                    model_path=info.model_path,
                    model_type=info.model_type,
                    engine_type=info.engine_type,
                    estimated_size=info.estimated_size,
                    config_model_type=getattr(info, "config_model_type", ""),
                    is_pinned=model_id in pinned_set,
                )
                added.append(model_id)
                logger.info(
                    f"Hot-added model: {model_id} "
                    f"({format_size(info.estimated_size)})"
                )

        discovered_ids = set(discovered.keys())
        removed = []
        for mid in list(self._entries.keys()):
            if mid not in discovered_ids:
                entry = self._entries[mid]
                if entry.engine is not None:
                    logger.warning(
                        f"Model files removed but model is loaded, keeping: {mid}"
                    )
                else:
                    del self._entries[mid]
                    removed.append(mid)
                    logger.info(f"Model removed (files no longer present): {mid}")

        return added, removed
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestRescanModels -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add omlx/engine_pool.py tests/test_jit_loading.py
git commit -m "feat: add rescan_models() for hot-add model discovery"
```

---

### Task 6: Wire rescan loop into server lifespan

**Files:**
- Modify: `omlx/server.py:287-361` (lifespan function)

- [ ] **Step 1: Add rescan loop after TTL task creation**

In `omlx/server.py`, after the TTL task block (around line 340), add:

```python
    # Start periodic model rescan if configured
    rescan_task = None
    if _server_state.engine_pool is not None:
        scan_interval = 60  # default
        if _server_state.global_settings is not None:
            scan_interval = _server_state.global_settings.memory.model_scan_interval_seconds

        if scan_interval > 0:
            async def _model_rescan_loop():
                while True:
                    try:
                        await asyncio.sleep(scan_interval)
                        _server_state.engine_pool.rescan_models()
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Model rescan error: {e}")

            rescan_task = asyncio.create_task(_model_rescan_loop())
```

In the shutdown section (after line 361), add rescan task cleanup:

```python
    if rescan_task is not None:
        rescan_task.cancel()
        try:
            await rescan_task
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add omlx/server.py
git commit -m "feat: add periodic model rescan loop to server lifespan"
```

---

## Chunk 2: JIT Loading (Two-Phase Locking)

### Task 7: Refactor `get_engine()` for two-phase locking and JIT load

This is the most complex task — it restructures `get_engine()` to release the lock before loading.

**Critical: Lock Strategy**

The key challenge is that `_ensure_memory_available()` and `_unload_engine()` currently assume the caller holds `self._lock`. After this refactor, they can be called from outside the lock (JIT Phase 2) AND from inside the lock (`check_ttl_expirations`). Since `asyncio.Lock` is NOT reentrant, we cannot simply add `async with self._lock:` inside these methods.

**Solution: Split into locked/unlocked variants.**

1. `_unload_engine_unlocked(model_id)` — does the actual work (no lock)
2. `_unload_engine(model_id)` — acquires lock, calls unlocked variant
3. `check_ttl_expirations()` calls `_unload_engine_unlocked()` since it already holds the lock
4. `get_engine()` Phase 2 calls `_unload_engine()` (locked variant) since it does NOT hold the lock
5. Same pattern for `_ensure_memory_available()` → `_ensure_memory_available_unlocked()` + locked wrapper

This avoids deadlocks while maintaining thread safety.

**Files:**
- Modify: `omlx/engine_pool.py:265-370` (get_engine method)
- Modify: `omlx/engine_pool.py:372-420` (_ensure_memory_available, _find_lru_victim)
- Modify: `omlx/engine_pool.py:421-460` (_unload_engine)
- Modify: `omlx/engine_pool.py:653-702` (check_ttl_expirations — use unlocked variant)
- Test: `tests/test_jit_loading.py`

- [ ] **Step 1: Write JIT loading tests**

Append to `tests/test_jit_loading.py`:

```python
class TestJITLoading:
    """Test JIT loading in get_engine()."""

    @pytest.mark.asyncio
    async def test_jit_loads_discovered_model(self):
        """Requesting a discovered (not loaded) model triggers JIT load."""
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        entry = EngineEntry(
            model_id="m1",
            model_path="/tmp/m1",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        pool._entries = {"m1": entry}

        mock_engine = MagicMock()
        with patch.object(pool, "_load_engine") as mock_load:
            async def fake_load(mid):
                pool._entries[mid].engine = mock_engine
                pool._entries[mid].last_access = time.time()
            mock_load.side_effect = fake_load

            engine = await pool.get_engine("m1")
            assert engine is mock_engine
            mock_load.assert_called_once_with("m1")

    @pytest.mark.asyncio
    async def test_loaded_model_returns_immediately(self):
        """Already-loaded model returns without re-loading."""
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        mock_engine = MagicMock()
        entry = EngineEntry(
            model_id="m1",
            model_path="/tmp/m1",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
            engine=mock_engine,
            last_access=time.time(),
        )
        pool._entries = {"m1": entry}

        with patch.object(pool, "_load_engine") as mock_load:
            engine = await pool.get_engine("m1")
            assert engine is mock_engine
            mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_requests_share_single_load(self):
        """Multiple concurrent requests for unloaded model share one load."""
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        entry = EngineEntry(
            model_id="m1",
            model_path="/tmp/m1",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        pool._entries = {"m1": entry}

        load_count = 0
        mock_engine = MagicMock()

        original_load = pool._load_engine

        async def slow_load(mid):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.05)  # Simulate slow load
            pool._entries[mid].engine = mock_engine
            pool._entries[mid].last_access = time.time()

        with patch.object(pool, "_load_engine", side_effect=slow_load):
            # Launch 3 concurrent requests
            results = await asyncio.gather(
                pool.get_engine("m1"),
                pool.get_engine("m1"),
                pool.get_engine("m1"),
            )

        assert all(r is mock_engine for r in results)
        assert load_count == 1  # Only one load, not three

    @pytest.mark.asyncio
    async def test_model_not_found_raises(self):
        """Requesting unknown model raises ModelNotFoundError."""
        from omlx.engine_pool import ModelNotFoundError

        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        pool._entries = {}

        with pytest.raises(ModelNotFoundError):
            await pool.get_engine("nonexistent")

    @pytest.mark.asyncio
    async def test_load_failure_unblocks_waiters(self):
        """When JIT load fails, concurrent waiters get an error, not a hang."""
        from omlx.engine_pool import ModelLoadingError

        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        entry = EngineEntry(
            model_id="m1",
            model_path="/tmp/m1",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        pool._entries = {"m1": entry}

        async def failing_load(mid):
            await asyncio.sleep(0.05)
            raise RuntimeError("corrupt weights")

        with patch.object(pool, "_load_engine", side_effect=failing_load):
            results = await asyncio.gather(
                pool.get_engine("m1"),
                pool.get_engine("m1"),
                return_exceptions=True,
            )

        # Both should get errors, not hang
        assert all(isinstance(r, Exception) for r in results)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestJITLoading -v`
Expected: FAIL — current `get_engine()` raises `ModelLoadingError` for concurrent requests and holds lock during entire load

- [ ] **Step 3: Refactor `get_engine()` with two-phase locking**

Replace the `get_engine()` method in `omlx/engine_pool.py` (lines 265-370) with:

```python
    async def get_engine(self, model_id: str) -> BaseEngine | EmbeddingEngine | RerankerEngine:
        """
        Get or load engine for the specified model.

        Implements two-phase locking for JIT loading:
        - Phase 1 (under lock): Check state, set up loading placeholder
        - Phase 2 (outside lock): Load weights, signal loading_event

        Concurrent requests for a loading model wait on loading_event
        instead of raising ModelLoadingError.
        """
        while True:
            async with self._lock:
                entry = self._entries.get(model_id)
                if not entry:
                    raise ModelNotFoundError(model_id, list(self._entries.keys()))

                # Case 1: Already loaded — return immediately
                if entry.engine is not None and not entry.is_loading:
                    entry.last_access = time.time()
                    return entry.engine

                # Case 2: Currently loading — grab event reference
                if entry.is_loading:
                    loading_event = entry.loading_event
                    # Release lock and wait outside
                    break

                # Case 3: Discovered but not loaded — initiate JIT load
                # Check if model is too large for memory limit
                if (
                    self._max_model_memory is not None
                    and entry.estimated_size > self._max_model_memory
                ):
                    raise ModelTooLargeError(
                        model_id, entry.estimated_size, self._max_model_memory
                    )

                # Mark as loading and clear event
                entry.is_loading = True
                entry.loading_event.clear()
                # Fall through to Phase 2 (outside lock)

            # Phase 2: Load outside the lock
            try:
                # Pre-load eviction (acquires lock internally if needed)
                if self._max_model_memory is not None:
                    kv_headroom = int(entry.estimated_size * 0.25)
                    required_with_headroom = entry.estimated_size + kv_headroom
                    try:
                        await self._ensure_memory_available(required_with_headroom)
                    except InsufficientMemoryError:
                        if self._current_model_memory + entry.estimated_size <= self._max_model_memory:
                            logger.info(
                                f"Loading {model_id} without KV headroom "
                                f"(need {format_size(required_with_headroom)}, "
                                f"available {format_size(self._max_model_memory - self._current_model_memory)})"
                            )
                        else:
                            await self._ensure_memory_available(entry.estimated_size)

                # Process memory limit check
                if self._process_memory_enforcer is not None:
                    enforcer = self._process_memory_enforcer
                    if enforcer.max_bytes > 0:
                        while True:
                            current_active = mx.get_active_memory()
                            projected = current_active + entry.estimated_size
                            if projected <= enforcer.max_bytes:
                                break
                            async with self._lock:
                                victim = self._find_lru_victim()
                            if victim is not None:
                                logger.info(
                                    f"Evicting '{victim}' to fit '{model_id}' "
                                    f"within process memory limit "
                                    f"({format_size(projected)} > "
                                    f"{format_size(enforcer.max_bytes)})"
                                )
                                await self._unload_engine(victim)
                                gc.collect()
                                loop = asyncio.get_running_loop()
                                await loop.run_in_executor(get_mlx_executor(), mx.clear_cache)
                                continue
                            raise InsufficientMemoryError(
                                required=entry.estimated_size,
                                current=current_active,
                                message=(
                                    f"Cannot load {model_id}: projected memory "
                                    f"{format_size(projected)} would exceed process "
                                    f"limit {format_size(enforcer.max_bytes)} "
                                    f"(current: {format_size(current_active)}, "
                                    f"model: {format_size(entry.estimated_size)})"
                                ),
                            )

                await self._load_engine(model_id)
                return self._entries[model_id].engine
            except Exception:
                # On failure, reset loading state and signal waiters
                async with self._lock:
                    if model_id in self._entries:
                        self._entries[model_id].is_loading = False
                        self._entries[model_id].loading_event.set()
                raise

        # Case 2 continuation: Wait for loading_event (lock released)
        await loading_event.wait()

        # Re-check under lock — the load may have failed
        async with self._lock:
            entry = self._entries.get(model_id)
            if entry is None or entry.engine is None:
                raise ModelLoadingError(f"Model {model_id} failed to load")
            entry.last_access = time.time()
            return entry.engine
```

Also update `_load_engine()` to signal `loading_event` on completion. At the end of `_load_engine()`, in the `finally` block (line 591-593), change to:

```python
        finally:
            entry.is_loading = False
            entry.abort_loading = False
            entry.loading_event.set()
```

**Refactor `_unload_engine()` into locked/unlocked variants:**

Rename existing `_unload_engine()` to `_unload_engine_unlocked()` (assumes caller holds lock). Create a new `_unload_engine()` that acquires the lock:

```python
    async def _unload_engine(self, model_id: str) -> None:
        """Unload engine with lock acquisition. Use from outside lock."""
        async with self._lock:
            await self._unload_engine_unlocked(model_id)

    async def _unload_engine_unlocked(self, model_id: str) -> None:
        """Unload engine. Caller MUST hold self._lock."""
        # ... existing _unload_engine body ...
```

**Refactor `_ensure_memory_available()` similarly:**

Rename to `_ensure_memory_available_unlocked()` (assumes lock held). Create locked wrapper:

```python
    async def _ensure_memory_available(self, required: int) -> None:
        """Ensure memory with lock acquisition. Use from outside lock."""
        async with self._lock:
            await self._ensure_memory_available_unlocked(required)

    async def _ensure_memory_available_unlocked(self, required: int) -> None:
        """Ensure memory. Caller MUST hold self._lock."""
        # ... existing _ensure_memory_available body ...
        # Internal calls to _find_lru_victim() and _unload_engine() become
        # _find_lru_victim() (unchanged, no lock needed) and _unload_engine_unlocked()
```

**Update `check_ttl_expirations()` to use `_unload_engine_unlocked()`** since it already holds the lock:

```python
    # In check_ttl_expirations, change:
    await self._unload_engine(model_id)
    # To:
    await self._unload_engine_unlocked(model_id)
```

**Update `get_engine()` Phase 2** to call the locked variants (since lock is NOT held):

```python
    # Phase 2 calls:
    await self._ensure_memory_available(required)  # locked variant
    # and for process enforcer eviction:
    await self._unload_engine(victim)  # locked variant
```

- [ ] **Step 4: Run JIT loading tests — expect PASS**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py::TestJITLoading -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run ALL tests to verify no regression**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/ -v --timeout=60`
Expected: All tests PASS. If existing tests break due to the lock refactoring, fix them.

- [ ] **Step 6: Commit**

```bash
git add omlx/engine_pool.py tests/test_jit_loading.py
git commit -m "feat: implement two-phase locking for JIT model loading

Refactors get_engine() to release the lock before loading weights.
Concurrent requests for the same model wait on loading_event.
Requests for already-loaded models are never blocked by JIT loads."
```

---

### Deferred: `jit_loading_behavior="reject"` (503 mode)

The setting is added in Task 1 and validated, but the "reject" code path (returning 503 + Retry-After, background loading via `asyncio.create_task`, `_estimate_load_seconds()` helper) is **deferred to a follow-up PR**. Reason: the "block" mode is the default and covers the primary use case. The reject mode adds API contract complexity (clients need retry logic) that should be validated separately. The setting infrastructure is in place so enabling it later requires only changes to `get_engine()`.

### Error handling on load failure

Per spec review feedback, on load failure the plan **resets `is_loading=False` and signals `loading_event`** but does NOT remove the entry. The entry stays in Discovered state (`engine=None`), allowing retry. This differs slightly from the spec (which says "remove entry") but is more resilient — removing the entry would require a rescan cycle before the model could be requested again.

---

## Chunk 3: Admin UI + API (Dashboard Integration)

### Task 8: Extend model list API with lifecycle fields

**Files:**
- Modify: `omlx/admin/routes.py:1244-1319` (list_models endpoint)
- Modify: `omlx/admin/routes.py:1343-1367` (load_model endpoint)

- [ ] **Step 1: Add lifecycle fields to model list response**

In `omlx/admin/routes.py`, in the `list_models` function, after `"last_access"` (line 1289), add:

```python
            # Lifecycle fields
            if model_info.get("loaded"):
                idle_seconds = int(time.time() - model_info.get("last_access", 0)) if model_info.get("last_access") else 0
                # Determine effective TTL
                per_model_ttl = settings.ttl_seconds if settings else None
                if per_model_ttl is not None:
                    effective_ttl = per_model_ttl
                else:
                    effective_ttl = _server_state.global_settings.memory.global_ttl_seconds if _server_state.global_settings else 0
                model_data["status"] = "loaded"
                model_data["idle_seconds"] = idle_seconds
                model_data["ttl_seconds"] = effective_ttl
                model_data["evicts_in_seconds"] = max(0, effective_ttl - idle_seconds) if effective_ttl > 0 else None
            elif model_info.get("is_loading"):
                model_data["status"] = "loading"
            else:
                model_data["status"] = "discovered"
```

Add `import time` at top of file if not already present.

- [ ] **Step 2: Update load_model endpoint to wait on loading_event**

In `omlx/admin/routes.py`, update `load_model` (line 1343). Replace the `is_loading` check (lines 1358-1359):

```python
    if entry.is_loading:
        raise HTTPException(status_code=409, detail=f"Model is already loading: {model_id}")
```

With:

```python
    if entry.is_loading:
        # Wait for existing load to complete instead of rejecting
        await entry.loading_event.wait()
        if entry.engine is not None:
            return {"status": "ok", "model_id": model_id, "message": f"Loaded: {model_id}"}
        raise HTTPException(status_code=500, detail=f"Model failed to load: {model_id}")
```

- [ ] **Step 3: Commit**

```bash
git add omlx/admin/routes.py
git commit -m "feat: add lifecycle fields to model list API + load endpoint waits"
```

---

### Task 9: Add lifecycle settings to admin global settings API

**Files:**
- Modify: `omlx/admin/routes.py:1632-1751` (get_global_settings and update_global_settings)

- [ ] **Step 1: Expose new settings in GET endpoint**

In `get_global_settings`, find where memory settings are returned and add the new fields to the memory settings dict:

```python
            "global_ttl_seconds": memory_settings.global_ttl_seconds,
            "model_scan_interval_seconds": memory_settings.model_scan_interval_seconds,
            "jit_loading_behavior": memory_settings.jit_loading_behavior,
```

- [ ] **Step 2: Handle new settings in POST endpoint**

In `update_global_settings`, find the memory settings update section and add handling for the new fields:

```python
        if "global_ttl_seconds" in memory:
            new_memory.global_ttl_seconds = int(memory["global_ttl_seconds"])
        if "model_scan_interval_seconds" in memory:
            new_memory.model_scan_interval_seconds = int(memory["model_scan_interval_seconds"])
        if "jit_loading_behavior" in memory:
            new_memory.jit_loading_behavior = memory["jit_loading_behavior"]
```

- [ ] **Step 3: Commit**

```bash
git add omlx/admin/routes.py
git commit -m "feat: expose lifecycle settings in admin global settings API"
```

---

### Task 10: Add i18n translation keys

**Files:**
- Modify: `omlx/admin/i18n/en.json`

- [ ] **Step 1: Add translation keys**

Add under the appropriate namespace in `en.json`:

```json
    "model_lifecycle": "Model Lifecycle",
    "global_ttl_seconds": "Global TTL (seconds)",
    "global_ttl_seconds_help": "Models auto-unload after this many seconds of inactivity. 0 to disable.",
    "model_scan_interval_seconds": "Model Scan Interval (seconds)",
    "model_scan_interval_seconds_help": "How often to check for new models in the models directory. 0 to disable.",
    "jit_loading_behavior": "Loading Behavior",
    "jit_loading_behavior_help": "How to handle requests for unloaded models.",
    "jit_block": "Block and Load",
    "jit_reject": "Reject (503)",
    "model_status_discovered": "Discovered",
    "model_status_loading": "Loading",
    "model_status_loaded": "Loaded",
    "evicts_in": "Evicts in",
    "idle_for": "Idle for",
    "ttl_disabled": "TTL disabled"
```

- [ ] **Step 2: Commit**

```bash
git add omlx/admin/i18n/en.json
git commit -m "feat: add i18n keys for model lifecycle settings"
```

---

### Task 11: Add "Model Lifecycle" section to settings page

**Files:**
- Modify: `omlx/admin/templates/settings.html`
- Modify: `omlx/admin/static/js/dashboard.js`

- [ ] **Step 1: Add settings section to template**

Find the settings template and add a new "Model Lifecycle" card section with:
- Global TTL input (number, min 0)
- Scan interval input (number, min 0)
- Loading behavior dropdown (Block and Load / Reject 503)

Follow the existing pattern of other settings cards in the template.

- [ ] **Step 2: Add JavaScript to handle save/load of new settings**

In `dashboard.js`, extend the settings load/save functions to include the new fields. Follow the existing pattern for how other `MemorySettings` fields are handled.

- [ ] **Step 3: Add model status indicators to models list**

In the dashboard models display, add:
- Status badge (Discovered / Loading / Loaded)
- TTL countdown for loaded models
- Idle time display

- [ ] **Step 4: Verify in browser**

Start omlx and verify:
1. Settings page shows "Model Lifecycle" section
2. Values save and persist correctly
3. Model list shows status badges

- [ ] **Step 5: Commit**

```bash
git add omlx/admin/templates/ omlx/admin/static/js/dashboard.js
git commit -m "feat: add Model Lifecycle settings UI and model status indicators"
```

---

## Chunk 4: Integration Tests + Final Verification

### Task 12: Write integration tests

**Files:**
- Modify: `tests/test_jit_loading.py`

- [ ] **Step 1: Add integration test for full lifecycle**

Append to `tests/test_jit_loading.py`:

```python
class TestJITIntegration:
    """Integration tests for the full JIT lifecycle."""

    @pytest.mark.asyncio
    async def test_evict_and_reload_cycle(self):
        """Model is evicted by TTL, then reloaded on next request."""
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        mock_engine = MagicMock()
        entry = EngineEntry(
            model_id="m1",
            model_path="/tmp/m1",
            model_type="llm",
            engine_type="batched",
            estimated_size=1000,
        )
        pool._entries = {"m1": entry}

        # Phase 1: Load the model
        with patch.object(pool, "_load_engine") as mock_load:
            async def fake_load(mid):
                pool._entries[mid].engine = mock_engine
                pool._entries[mid].last_access = time.time()
            mock_load.side_effect = fake_load
            engine = await pool.get_engine("m1")
            assert engine is mock_engine

        # Phase 2: Simulate time passing and TTL eviction
        pool._entries["m1"].last_access = time.time() - 200
        pool._entries["m1"].engine = mock_engine  # still loaded

        async def mock_unload(mid):
            pool._entries[mid].engine = None
            pool._entries[mid].is_loading = False

        with patch.object(pool, "_unload_engine", side_effect=mock_unload):
            settings_mgr = _make_mock_settings_manager()
            expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
            assert "m1" in expired
            assert pool._entries["m1"].engine is None

        # Phase 3: Reload on next request
        new_engine = MagicMock()
        with patch.object(pool, "_load_engine") as mock_load:
            async def fake_reload(mid):
                pool._entries[mid].engine = new_engine
                pool._entries[mid].last_access = time.time()
            mock_load.side_effect = fake_reload
            engine = await pool.get_engine("m1")
            assert engine is new_engine
```

- [ ] **Step 2: Run all tests**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/test_jit_loading.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run the complete test suite**

Run: `cd /Users/alexworland/omlx && python -m pytest tests/ -v --timeout=60`
Expected: All tests PASS across all test files

- [ ] **Step 4: Commit**

```bash
git add tests/test_jit_loading.py
git commit -m "test: add integration test for full JIT evict-and-reload cycle"
```

---

## Agent Team

| Teammate | Model | Tasks | Notes |
|----------|-------|-------|-------|
| carter-the-core-engineer | sonnet | Tasks 1-3 (settings + EngineEntry + TTL fallback) | Sequential — each builds on previous |
| spock-the-engine-architect | opus | Task 7 (two-phase locking refactor) | Most complex task — needs deep reasoning about lock semantics |
| geordi-the-api-engineer | sonnet | Tasks 8-9 (API changes) | After Task 7 completes |
| uhura-the-ui-engineer | sonnet | Tasks 10-11 (i18n + dashboard) | Parallel with API work after Task 7 |
| Lead (self) | — | Tasks 4, 5, 6 (server wiring + rescan), Task 12 (integration tests) | Wiring tasks after core is done |

**Parallel groups:**
- Phase 1: Tasks 1→2→3 (sequential, carter)
- Phase 2: Tasks 4, 5, 6 (sequential, lead) — after Phase 1
- Phase 3: Task 7 (spock) — after Phase 2
- Phase 4: [Tasks 8-9 (geordi)] ∥ [Tasks 10-11 (uhura)] — after Task 7
- Phase 5: Task 12 (lead) — after all complete
