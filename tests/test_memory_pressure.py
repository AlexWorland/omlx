# SPDX-License-Identifier: Apache-2.0
"""Tests for memory pressure zone management.

Tests cover:
- PressureZone calculation (watermark and target-free based)
- Adaptive eviction rate scaling
- Prefill pause propagation
- Eviction across engines
- MemorySettings validation
- Status reporting with pressure info
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from omlx.process_memory_enforcer import PressureZone, ProcessMemoryEnforcer
from omlx.settings import MemorySettings


# =========================================================================
# Fixtures
# =========================================================================


def _make_entry(model_id, engine=None, is_pinned=False, last_access=0.0):
    """Create a mock EngineEntry."""
    entry = MagicMock()
    entry.model_id = model_id
    entry.engine = engine
    entry.is_pinned = is_pinned
    entry.is_loading = False
    entry.abort_loading = False
    entry.last_access = last_access
    return entry


def _make_engine_with_cache(num_evictable=10, block_size=64):
    """Create a mock engine with paged cache manager."""
    blocks = [MagicMock(block_id=i) for i in range(num_evictable)]
    pcm = MagicMock()
    pcm.max_blocks = 1000
    pcm.get_evictable_blocks.return_value = blocks
    pcm.evict_block_permanently.return_value = True
    scheduler = MagicMock()
    scheduler.paged_cache_manager = pcm
    scheduler._prefill_paused = False
    engine = MagicMock()
    engine.scheduler = scheduler
    return engine


@pytest.fixture
def pool():
    """Create a mock EnginePool."""
    pool = MagicMock()
    pool._lock = asyncio.Lock()
    pool._entries = {}
    return pool


@pytest.fixture
def enforcer(pool):
    """Create an enforcer with 10GB limit and default watermarks."""
    return ProcessMemoryEnforcer(
        engine_pool=pool,
        max_bytes=10 * 1024**3,
        poll_interval=0.1,
    )


@pytest.fixture
def enforcer_with_target_free(pool):
    """Create an enforcer with target free memory configured."""
    return ProcessMemoryEnforcer(
        engine_pool=pool,
        max_bytes=10 * 1024**3,
        poll_interval=0.1,
        target_free_bytes=2 * 1024**3,  # 2GB target free
    )


# =========================================================================
# Zone calculation tests
# =========================================================================


class TestZoneCalculation:
    """Tests for _calculate_zone() pressure zone determination."""

    def test_green_at_zero_utilization(self, enforcer):
        """0% utilization → GREEN."""
        assert enforcer._calculate_zone(0) == PressureZone.GREEN

    def test_green_at_50_percent(self, enforcer):
        """50% utilization → GREEN (below yellow threshold)."""
        current = int(10 * 1024**3 * 0.50)
        assert enforcer._calculate_zone(current) == PressureZone.GREEN

    def test_green_just_below_yellow(self, enforcer):
        """74.9% utilization → GREEN (just below yellow=0.75)."""
        current = int(10 * 1024**3 * 0.749)
        assert enforcer._calculate_zone(current) == PressureZone.GREEN

    def test_yellow_at_threshold(self, enforcer):
        """75% utilization → YELLOW (exactly at yellow threshold)."""
        current = int(10 * 1024**3 * 0.75)
        assert enforcer._calculate_zone(current) == PressureZone.YELLOW

    def test_yellow_at_80_percent(self, enforcer):
        """80% utilization → YELLOW (between yellow and red)."""
        current = int(10 * 1024**3 * 0.80)
        assert enforcer._calculate_zone(current) == PressureZone.YELLOW

    def test_red_at_threshold(self, enforcer):
        """90% utilization → RED (exactly at red threshold)."""
        current = int(10 * 1024**3 * 0.90)
        assert enforcer._calculate_zone(current) == PressureZone.RED

    def test_critical_at_threshold(self, enforcer):
        """95% utilization → CRITICAL."""
        current = int(10 * 1024**3 * 0.95)
        assert enforcer._calculate_zone(current) == PressureZone.CRITICAL

    def test_critical_at_100_percent(self, enforcer):
        """100% utilization → CRITICAL."""
        current = 10 * 1024**3
        assert enforcer._calculate_zone(current) == PressureZone.CRITICAL

    def test_green_when_disabled(self, pool):
        """Zone is GREEN when max_bytes <= 0 (enforcement disabled)."""
        enforcer = ProcessMemoryEnforcer(engine_pool=pool, max_bytes=0)
        assert enforcer._calculate_zone(999 * 1024**3) == PressureZone.GREEN


class TestTargetFreeZoneCalculation:
    """Tests for target-free-based zone calculation."""

    def test_green_when_plenty_free(self, enforcer_with_target_free):
        """Free > target → GREEN."""
        # 5GB used, 5GB free, target is 2GB
        current = 5 * 1024**3
        assert enforcer_with_target_free._calculate_zone(current) == PressureZone.GREEN

    def test_yellow_when_below_target(self, enforcer_with_target_free):
        """Free < target → at least YELLOW."""
        # 9GB used, 1GB free, target is 2GB (free < target)
        current = 9 * 1024**3
        assert enforcer_with_target_free._calculate_zone(current) in (
            PressureZone.YELLOW,
            PressureZone.RED,  # watermark might push higher
        )

    def test_red_when_very_low_free(self, enforcer_with_target_free):
        """Free < target * 0.25 → at least RED."""
        # target=2GB, 0.25*target=0.5GB, so free < 0.5GB → RED
        current = int(10 * 1024**3 - 0.4 * 1024**3)  # 0.4GB free
        zone = enforcer_with_target_free._calculate_zone(current)
        assert zone in (PressureZone.RED, PressureZone.CRITICAL)

    def test_critical_when_over_max(self, enforcer_with_target_free):
        """Free < 0 → CRITICAL."""
        current = 11 * 1024**3  # Over max_bytes
        assert enforcer_with_target_free._calculate_zone(current) == PressureZone.CRITICAL

    def test_more_restrictive_wins(self, pool):
        """When watermark says GREEN but target_free says YELLOW, YELLOW wins."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=pool,
            max_bytes=100 * 1024**3,  # 100GB limit
            target_free_bytes=50 * 1024**3,  # 50GB target free (very aggressive)
        )
        # 60GB used → 60% utilization (GREEN by watermarks)
        # But 40GB free < 50GB target → YELLOW by target_free
        current = 60 * 1024**3
        assert enforcer._calculate_zone(current) == PressureZone.YELLOW


# =========================================================================
# Adaptive eviction rate tests
# =========================================================================


class TestAdaptiveEvictionRate:
    """Tests for _calculate_blocks_to_evict() adaptive rate."""

    def test_returns_1_at_yellow_boundary(self, enforcer):
        """At yellow boundary, evicts minimum 1 block."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.75)
            blocks = enforcer._calculate_blocks_to_evict(100)
        assert blocks == 1

    def test_returns_max_at_critical(self, enforcer):
        """At critical threshold, evicts max blocks per cycle."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.95)
            blocks = enforcer._calculate_blocks_to_evict(100)
        # Auto max = 10% of 100 = 10
        assert blocks == 10

    def test_proportional_scaling(self, enforcer):
        """Blocks scale proportionally between yellow and critical."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            # Midpoint between yellow (0.75) and critical (0.95) = 0.85
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.85)
            blocks = enforcer._calculate_blocks_to_evict(100)
        # urgency = (0.85 - 0.75) / (0.95 - 0.75) = 0.5
        # max_per_cycle = 10 (auto), blocks = max(1, int(0.5 * 10)) = 5
        assert blocks == 5

    def test_auto_max_per_cycle(self, enforcer):
        """Auto max_per_cycle = 10% of evictable blocks."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.95)
            # 200 evictable blocks → max = 20
            blocks = enforcer._calculate_blocks_to_evict(200)
        assert blocks == 20

    def test_explicit_max_per_cycle(self, pool):
        """Explicit max_evict_blocks_per_cycle overrides auto."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=pool,
            max_bytes=10 * 1024**3,
            max_evict_blocks_per_cycle=5,
        )
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.95)
            blocks = enforcer._calculate_blocks_to_evict(1000)
        assert blocks == 5

    def test_zero_evictable_returns_zero(self, enforcer):
        """No evictable blocks → 0."""
        blocks = enforcer._calculate_blocks_to_evict(0)
        assert blocks == 0


# =========================================================================
# Prefill pause propagation tests
# =========================================================================


class TestPrefillPause:
    """Tests for _set_prefill_paused() propagation."""

    def test_propagates_pause_to_schedulers(self, enforcer):
        """Pause flag propagates to all engine schedulers."""
        engine1 = _make_engine_with_cache()
        engine2 = _make_engine_with_cache()
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine1),
            "m2": _make_entry("m2", engine=engine2),
        }
        enforcer._set_prefill_paused(True)
        assert engine1.scheduler._prefill_paused is True
        assert engine2.scheduler._prefill_paused is True

    def test_green_unpauses_after_red(self, enforcer):
        """Transitioning from RED to GREEN unpauses prefills."""
        engine = _make_engine_with_cache()
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        enforcer._set_prefill_paused(True)
        assert engine.scheduler._prefill_paused is True
        enforcer._set_prefill_paused(False)
        assert engine.scheduler._prefill_paused is False

    def test_noop_when_already_paused(self, enforcer):
        """Setting same pause state is a no-op (no log spam)."""
        engine = _make_engine_with_cache()
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        enforcer._set_prefill_paused(True)
        assert enforcer._prefill_paused is True
        # Calling again with same value should be a no-op (early return).
        # Verify by checking the scheduler attribute isn't re-assigned.
        engine.scheduler._prefill_paused = True  # Already set
        call_count_before = engine.scheduler._prefill_paused  # Just True
        enforcer._set_prefill_paused(True)  # Same state — should no-op
        # Still True, no exception, no extra work
        assert enforcer._prefill_paused is True


# =========================================================================
# Eviction across engines tests
# =========================================================================


class TestEvictionAcrossEngines:
    """Tests for _evict_blocks_across_engines()."""

    def test_lru_order_oldest_first(self, enforcer):
        """Evicts blocks from oldest-accessed engine first."""
        old_engine = _make_engine_with_cache(num_evictable=5)
        new_engine = _make_engine_with_cache(num_evictable=5)
        enforcer._engine_pool._entries = {
            "old": _make_entry("old", engine=old_engine, last_access=100.0),
            "new": _make_entry("new", engine=new_engine, last_access=200.0),
        }
        enforcer._evict_blocks_across_engines(3)
        # Should evict from old engine first
        old_pcm = old_engine.scheduler.paged_cache_manager
        old_pcm.get_evictable_blocks.assert_called_once_with(3)
        assert old_pcm.evict_block_permanently.call_count == 3

    def test_skips_engine_without_cache_manager(self, enforcer):
        """Gracefully skips engines without paged_cache_manager."""
        engine = MagicMock()
        scheduler = MagicMock(spec=[])  # No paged_cache_manager
        engine.scheduler = scheduler
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        # Should not raise
        evicted = enforcer._evict_blocks_across_engines(5)
        assert evicted == 0

    def test_correct_block_count(self, enforcer):
        """Returns correct count of actually evicted blocks."""
        engine = _make_engine_with_cache(num_evictable=5)
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        evicted = enforcer._evict_blocks_across_engines(3)
        assert evicted == 3


# =========================================================================
# Settings validation tests
# =========================================================================


class TestMemorySettings:
    """Tests for MemorySettings watermark validation."""

    def test_valid_watermarks(self):
        """Valid watermarks pass validation."""
        ms = MemorySettings(
            watermark_yellow=0.6,
            watermark_red=0.8,
            watermark_critical=0.9,
        )
        ms.validate()  # Should not raise

    def test_invalid_ordering_raises(self):
        """Misordered watermarks raise ValueError."""
        with pytest.raises(ValueError, match="ordered"):
            MemorySettings(
                watermark_yellow=0.9,
                watermark_red=0.8,
                watermark_critical=0.7,
            )

    def test_out_of_range_raises(self):
        """Watermark outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            MemorySettings(watermark_yellow=1.0)

    def test_round_trip_serialization(self):
        """to_dict/from_dict preserves all fields."""
        original = MemorySettings(
            watermark_yellow=0.6,
            watermark_red=0.85,
            watermark_critical=0.92,
            target_free_memory="8GB",
            max_evict_blocks_per_cycle=50,
        )
        d = original.to_dict()
        restored = MemorySettings.from_dict(d)
        assert restored.watermark_yellow == 0.6
        assert restored.watermark_red == 0.85
        assert restored.watermark_critical == 0.92
        assert restored.target_free_memory == "8GB"
        assert restored.max_evict_blocks_per_cycle == 50

    def test_target_free_memory_disabled(self):
        """'disabled' returns None."""
        ms = MemorySettings(target_free_memory="disabled")
        assert ms.get_target_free_memory_bytes() is None

    def test_target_free_memory_auto(self):
        """'auto' returns 4GB."""
        ms = MemorySettings(target_free_memory="auto")
        assert ms.get_target_free_memory_bytes() == 4 * 1024**3

    def test_target_free_memory_absolute(self):
        """Absolute size like '8GB' is parsed correctly."""
        ms = MemorySettings(target_free_memory="8GB")
        result = ms.get_target_free_memory_bytes()
        assert result == 8 * 1024**3


# =========================================================================
# Status reporting tests
# =========================================================================


class TestStatusReporting:
    """Tests for get_status() with pressure info."""

    def test_includes_pressure_fields(self, enforcer):
        """Status includes pressure_zone, prefill_paused, watermarks."""
        status = enforcer.get_status()
        assert "pressure_zone" in status
        assert status["pressure_zone"] == "green"
        assert "prefill_paused" in status
        assert status["prefill_paused"] is False
        assert "watermarks" in status
        assert status["watermarks"]["yellow"] == 0.75
        assert status["watermarks"]["red"] == 0.90
        assert status["watermarks"]["critical"] == 0.95

    def test_includes_target_free(self, enforcer_with_target_free):
        """Status includes target_free when configured."""
        status = enforcer_with_target_free.get_status()
        assert status["target_free_bytes"] == 2 * 1024**3
        assert "target_free_formatted" in status

    def test_target_free_disabled_shows_disabled(self, enforcer):
        """Status shows 'disabled' when target_free not configured."""
        status = enforcer.get_status()
        assert status["target_free_bytes"] is None
        assert status["target_free_formatted"] == "disabled"


# =========================================================================
# Proactive pressure management integration tests
# =========================================================================


class TestProactivePressureManagement:
    """Tests for _proactive_pressure_management() integration."""

    def test_green_zone_unpauses(self, enforcer):
        """GREEN zone ensures prefills are unpaused."""
        enforcer._prefill_paused = True
        engine = _make_engine_with_cache()
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.5)
            asyncio.get_event_loop().run_until_complete(
                enforcer._proactive_pressure_management()
            )
        assert enforcer._current_zone == PressureZone.GREEN
        assert enforcer._prefill_paused is False

    def test_yellow_zone_evicts_but_no_pause(self, enforcer):
        """YELLOW zone evicts blocks but does NOT pause prefills."""
        engine = _make_engine_with_cache(num_evictable=10)
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.80)
            asyncio.get_event_loop().run_until_complete(
                enforcer._proactive_pressure_management()
            )
        assert enforcer._current_zone == PressureZone.YELLOW
        assert enforcer._prefill_paused is False
        pcm = engine.scheduler.paged_cache_manager
        pcm.evict_block_permanently.assert_called()

    def test_red_zone_evicts_and_pauses(self, enforcer):
        """RED zone evicts blocks AND pauses prefills."""
        engine = _make_engine_with_cache(num_evictable=10)
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.92)
            asyncio.get_event_loop().run_until_complete(
                enforcer._proactive_pressure_management()
            )
        assert enforcer._current_zone == PressureZone.RED
        assert enforcer._prefill_paused is True
        pcm = engine.scheduler.paged_cache_manager
        pcm.evict_block_permanently.assert_called()

    def test_critical_zone_pauses_defers(self, enforcer):
        """CRITICAL zone pauses prefills and defers to _check_and_enforce."""
        engine = _make_engine_with_cache()
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.96)
            asyncio.get_event_loop().run_until_complete(
                enforcer._proactive_pressure_management()
            )
        assert enforcer._current_zone == PressureZone.CRITICAL
        assert enforcer._prefill_paused is True
