# SPDX-License-Identifier: Apache-2.0
"""Tests for JIT model loading, TTL eviction, and model rescanning."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.settings import MemorySettings


# =========================================================================
# Settings Validation
# =========================================================================


class TestLifecycleSettings:
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


# =========================================================================
# Loading Event
# =========================================================================

from omlx.engine_pool import EngineEntry


class TestLoadingEvent:
    def test_loading_event_default_set(self):
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

    def test_concurrent_waiters_unblock(self):
        async def _run():
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
            assert len(results) == 0

            entry.loading_event.set()
            await asyncio.gather(*tasks)
            assert sorted(results) == [0, 1, 2]

        asyncio.run(_run())


# =========================================================================
# TTL Eviction with Global Fallback
# =========================================================================

from omlx.engine_pool import EnginePool


def _make_mock_entry(model_id, engine=None, is_pinned=False, last_access=0.0, is_loading=False):
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
    pool = MagicMock(spec=EnginePool)
    pool._entries = {}
    pool._lock = asyncio.Lock()
    pool._unload_engine_unlocked = AsyncMock()
    pool.check_ttl_expirations = EnginePool.check_ttl_expirations.__get__(pool)
    return pool


def _make_mock_settings_manager(settings_map=None):
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
    def test_global_ttl_evicts_idle_model(self):
        async def _run():
            pool = _make_mock_pool()
            engine = MagicMock()
            entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 200)
            pool._entries = {"m1": entry}
            settings_mgr = _make_mock_settings_manager()
            expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
            assert "m1" in expired

        asyncio.run(_run())

    def test_per_model_ttl_overrides_global(self):
        async def _run():
            pool = _make_mock_pool()
            engine = MagicMock()
            entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 100)
            pool._entries = {"m1": entry}
            settings_mgr = _make_mock_settings_manager({"m1": 60})
            expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
            assert "m1" in expired

        asyncio.run(_run())

    def test_per_model_ttl_zero_never_evicts(self):
        async def _run():
            pool = _make_mock_pool()
            engine = MagicMock()
            entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 99999)
            pool._entries = {"m1": entry}
            settings_mgr = _make_mock_settings_manager({"m1": 0})
            expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
            assert "m1" not in expired

        asyncio.run(_run())

    def test_global_ttl_zero_disables_all(self):
        async def _run():
            pool = _make_mock_pool()
            engine = MagicMock()
            entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 99999)
            pool._entries = {"m1": entry}
            settings_mgr = _make_mock_settings_manager()
            expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=0)
            assert "m1" not in expired

        asyncio.run(_run())

    def test_pinned_exempt_from_ttl(self):
        async def _run():
            pool = _make_mock_pool()
            engine = MagicMock()
            entry = _make_mock_entry("m1", engine=engine, is_pinned=True, last_access=time.time() - 99999)
            pool._entries = {"m1": entry}
            settings_mgr = _make_mock_settings_manager()
            expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
            assert "m1" not in expired

        asyncio.run(_run())

    def test_evicted_model_stays_in_entries(self):
        async def _run():
            pool = _make_mock_pool()
            engine = MagicMock()
            entry = _make_mock_entry("m1", engine=engine, last_access=time.time() - 200)
            pool._entries = {"m1": entry}

            async def mock_unload(mid):
                pool._entries[mid].engine = None
            pool._unload_engine_unlocked = mock_unload

            settings_mgr = _make_mock_settings_manager()
            expired = await pool.check_ttl_expirations(settings_mgr, global_ttl_seconds=180)
            assert "m1" in expired
            assert "m1" in pool._entries

        asyncio.run(_run())


# =========================================================================
# Rescan Models
# =========================================================================

from unittest.mock import patch


class TestRescanModels:
    def test_rescan_finds_new_model(self):
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        pool._entries = {"model-a": _make_mock_entry("model-a")}
        pool._model_dirs = ["/tmp/models"]

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

    def test_rescan_removes_missing_unloaded_model(self):
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        pool._entries = {
            "model-a": _make_mock_entry("model-a"),
            "model-b": _make_mock_entry("model-b"),
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

    def test_rescan_keeps_loaded_model_even_if_files_gone(self):
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        loaded_entry = _make_mock_entry("model-b", engine=MagicMock())
        pool._entries = {
            "model-a": _make_mock_entry("model-a"),
            "model-b": loaded_entry,
        }
        pool._model_dirs = ["/tmp/models"]

        with patch("omlx.engine_pool.discover_models") as mock_discover:
            mock_discover.return_value = {
                "model-a": MagicMock(model_path="/tmp/models/model-a", model_type="llm",
                                      engine_type="batched", estimated_size=1000, config_model_type=""),
            }
            added, removed = pool.rescan_models()

        assert "model-b" in pool._entries
        assert "model-b" not in removed


# =========================================================================
# JIT Loading (Two-Phase Locking)
# =========================================================================

from omlx.exceptions import ModelNotFoundError, ModelLoadingError


class TestJITLoading:
    def test_jit_loads_discovered_model(self):
        """Requesting a discovered model triggers JIT load."""
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

        async def fake_load(mid):
            pool._entries[mid].engine = mock_engine
            pool._entries[mid].last_access = time.time()

        with patch.object(pool, "_load_engine", side_effect=fake_load):
            engine = asyncio.run(pool.get_engine("m1"))
            assert engine is mock_engine

    def test_loaded_model_returns_immediately(self):
        """Already-loaded model returns without calling _load_engine."""
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
            engine = asyncio.run(pool.get_engine("m1"))
            assert engine is mock_engine
            mock_load.assert_not_called()

    def test_concurrent_requests_share_single_load(self):
        """Multiple concurrent requests share one load via loading_event."""

        async def _run():
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

            async def slow_load(mid):
                nonlocal load_count
                load_count += 1
                await asyncio.sleep(0.05)
                pool._entries[mid].engine = mock_engine
                pool._entries[mid].last_access = time.time()
                # Must replicate what real _load_engine's finally block does
                pool._entries[mid].is_loading = False
                pool._entries[mid].loading_event.set()

            with patch.object(pool, "_load_engine", side_effect=slow_load):
                results = await asyncio.gather(
                    pool.get_engine("m1"),
                    pool.get_engine("m1"),
                    pool.get_engine("m1"),
                )
            assert all(r is mock_engine for r in results)
            assert load_count == 1

        asyncio.run(_run())

    def test_model_not_found_raises(self):
        """Requesting a nonexistent model raises ModelNotFoundError."""
        pool = EnginePool(max_model_memory=None, scheduler_config=MagicMock())
        pool._entries = {}
        with pytest.raises(ModelNotFoundError):
            asyncio.run(pool.get_engine("nonexistent"))

    def test_load_failure_unblocks_waiters(self):
        """When JIT load fails, concurrent waiters get an error."""

        async def _run():
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
            assert all(isinstance(r, Exception) for r in results)

        asyncio.run(_run())
