# SPDX-License-Identifier: Apache-2.0
"""Tests for memory pressure zone management.

Tests cover:
- PressureZone calculation (watermark and target-free based)
- System RAM zone calculation with hysteresis
- Combined dual-signal zone calculation
- Adaptive eviction rate scaling
- Prefill pause propagation
- Eviction across engines
- Hot cache shrink behavior
- MemorySettings validation (including system watermarks)
- Status reporting with dual-signal pressure info
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from omlx.process_memory_enforcer import PressureZone, ProcessMemoryEnforcer, ZoneResult
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


def _make_engine_with_ssd_cache(hot_cache_max_bytes=1024 * 1024):
    """Create a mock engine with SSD cache manager for hot cache testing."""
    ssd_mgr = MagicMock()
    ssd_mgr._hot_cache_max_bytes = hot_cache_max_bytes
    ssd_mgr.shrink_hot_cache.return_value = 5  # 5 entries evicted
    pcm = MagicMock()
    pcm.max_blocks = 1000
    pcm.get_evictable_blocks.return_value = []
    scheduler = MagicMock()
    scheduler.paged_cache_manager = pcm
    scheduler.paged_ssd_cache_manager = ssd_mgr
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


@pytest.fixture
def enforcer_system_only(pool):
    """Create an enforcer where only system monitoring matters (metal stays GREEN)."""
    return ProcessMemoryEnforcer(
        engine_pool=pool,
        max_bytes=10 * 1024**3,
        poll_interval=0.1,
        system_memory_monitoring=True,
        system_watermark_yellow=0.80,
        system_watermark_red=0.90,
        system_watermark_critical=0.95,
    )


# =========================================================================
# Metal zone calculation tests
# =========================================================================


class TestMetalZoneCalculation:
    """Tests for _calculate_metal_zone() pressure zone determination."""

    def test_green_at_zero_utilization(self, enforcer):
        """0% utilization → GREEN."""
        assert enforcer._calculate_metal_zone(0, PressureZone.GREEN) == PressureZone.GREEN

    def test_green_at_50_percent(self, enforcer):
        """50% utilization → GREEN (below yellow threshold)."""
        current = int(10 * 1024**3 * 0.50)
        assert enforcer._calculate_metal_zone(current, PressureZone.GREEN) == PressureZone.GREEN

    def test_green_just_below_yellow(self, enforcer):
        """74.9% utilization → GREEN (just below yellow=0.75)."""
        current = int(10 * 1024**3 * 0.749)
        assert enforcer._calculate_metal_zone(current, PressureZone.GREEN) == PressureZone.GREEN

    def test_yellow_at_threshold(self, enforcer):
        """75% utilization → YELLOW (exactly at yellow threshold)."""
        current = int(10 * 1024**3 * 0.75)
        assert enforcer._calculate_metal_zone(current, PressureZone.GREEN) == PressureZone.YELLOW

    def test_yellow_at_80_percent(self, enforcer):
        """80% utilization → YELLOW (between yellow and red)."""
        current = int(10 * 1024**3 * 0.80)
        assert enforcer._calculate_metal_zone(current, PressureZone.GREEN) == PressureZone.YELLOW

    def test_red_at_threshold(self, enforcer):
        """90% utilization → RED (exactly at red threshold)."""
        current = int(10 * 1024**3 * 0.90)
        assert enforcer._calculate_metal_zone(current, PressureZone.GREEN) == PressureZone.RED

    def test_critical_at_threshold(self, enforcer):
        """95% utilization → CRITICAL."""
        current = int(10 * 1024**3 * 0.95)
        assert enforcer._calculate_metal_zone(current, PressureZone.GREEN) == PressureZone.CRITICAL

    def test_critical_at_100_percent(self, enforcer):
        """100% utilization → CRITICAL."""
        current = 10 * 1024**3
        assert enforcer._calculate_metal_zone(current, PressureZone.GREEN) == PressureZone.CRITICAL

    def test_green_when_disabled(self, pool):
        """Zone is GREEN when max_bytes <= 0 (enforcement disabled)."""
        enforcer = ProcessMemoryEnforcer(engine_pool=pool, max_bytes=0)
        assert enforcer._calculate_metal_zone(999 * 1024**3, PressureZone.GREEN) == PressureZone.GREEN


class TestZoneCalculation:
    """Tests for combined _calculate_zone() — backward compatibility."""

    def test_green_at_zero_utilization(self, enforcer):
        """0% utilization → GREEN."""
        result = enforcer._calculate_zone(0, 0.0, PressureZone.GREEN)
        assert result.zone == PressureZone.GREEN

    def test_green_at_50_percent(self, enforcer):
        """50% utilization → GREEN (below yellow threshold)."""
        current = int(10 * 1024**3 * 0.50)
        result = enforcer._calculate_zone(current, 0.0, PressureZone.GREEN)
        assert result.zone == PressureZone.GREEN

    def test_yellow_at_threshold(self, enforcer):
        """75% utilization → YELLOW."""
        current = int(10 * 1024**3 * 0.75)
        result = enforcer._calculate_zone(current, 0.0, PressureZone.GREEN)
        assert result.zone == PressureZone.YELLOW

    def test_red_at_threshold(self, enforcer):
        """90% utilization → RED."""
        current = int(10 * 1024**3 * 0.90)
        result = enforcer._calculate_zone(current, 0.0, PressureZone.GREEN)
        assert result.zone == PressureZone.RED

    def test_critical_at_threshold(self, enforcer):
        """95% utilization → CRITICAL."""
        current = int(10 * 1024**3 * 0.95)
        result = enforcer._calculate_zone(current, 0.0, PressureZone.GREEN)
        assert result.zone == PressureZone.CRITICAL

    def test_green_when_disabled(self, pool):
        """Zone is GREEN when max_bytes <= 0."""
        enforcer = ProcessMemoryEnforcer(engine_pool=pool, max_bytes=0)
        result = enforcer._calculate_zone(999 * 1024**3, 0.5, PressureZone.GREEN)
        assert result.zone == PressureZone.GREEN


class TestTargetFreeZoneCalculation:
    """Tests for target-free-based zone calculation."""

    def test_green_when_plenty_free(self, enforcer_with_target_free):
        """Free > target → GREEN."""
        current = 5 * 1024**3
        result = enforcer_with_target_free._calculate_zone(current, 0.0, PressureZone.GREEN)
        assert result.zone == PressureZone.GREEN

    def test_yellow_when_below_target(self, enforcer_with_target_free):
        """Free < target → at least YELLOW."""
        current = 9 * 1024**3
        result = enforcer_with_target_free._calculate_zone(current, 0.0, PressureZone.GREEN)
        assert result.zone in (PressureZone.YELLOW, PressureZone.RED)

    def test_red_when_very_low_free(self, enforcer_with_target_free):
        """Free < target * 0.25 → at least RED."""
        current = int(10 * 1024**3 - 0.4 * 1024**3)
        result = enforcer_with_target_free._calculate_zone(current, 0.0, PressureZone.GREEN)
        assert result.zone in (PressureZone.RED, PressureZone.CRITICAL)

    def test_critical_when_over_max(self, enforcer_with_target_free):
        """Free < 0 → CRITICAL."""
        current = 11 * 1024**3
        result = enforcer_with_target_free._calculate_zone(current, 0.0, PressureZone.GREEN)
        assert result.zone == PressureZone.CRITICAL

    def test_more_restrictive_wins(self, pool):
        """When watermark says GREEN but target_free says YELLOW, YELLOW wins."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=pool,
            max_bytes=100 * 1024**3,
            target_free_bytes=50 * 1024**3,
        )
        current = 60 * 1024**3
        result = enforcer._calculate_zone(current, 0.0, PressureZone.GREEN)
        assert result.zone == PressureZone.YELLOW


# =========================================================================
# System zone calculation tests
# =========================================================================


class TestSystemZoneCalculation:
    """Tests for _calculate_system_zone() with system RAM utilization."""

    def test_green_at_low_system_util(self, enforcer_system_only):
        """50% system util → GREEN."""
        assert enforcer_system_only._calculate_system_zone(0.50, PressureZone.GREEN) == PressureZone.GREEN

    def test_green_just_below_yellow(self, enforcer_system_only):
        """79% system util → GREEN (just below 80% threshold)."""
        assert enforcer_system_only._calculate_system_zone(0.79, PressureZone.GREEN) == PressureZone.GREEN

    def test_yellow_at_threshold(self, enforcer_system_only):
        """80% system util → YELLOW."""
        assert enforcer_system_only._calculate_system_zone(0.80, PressureZone.GREEN) == PressureZone.YELLOW

    def test_yellow_at_85_percent(self, enforcer_system_only):
        """85% system util → YELLOW (between yellow and red)."""
        assert enforcer_system_only._calculate_system_zone(0.85, PressureZone.GREEN) == PressureZone.YELLOW

    def test_red_at_threshold(self, enforcer_system_only):
        """90% system util → RED."""
        assert enforcer_system_only._calculate_system_zone(0.90, PressureZone.GREEN) == PressureZone.RED

    def test_critical_at_threshold(self, enforcer_system_only):
        """95% system util → CRITICAL."""
        assert enforcer_system_only._calculate_system_zone(0.95, PressureZone.GREEN) == PressureZone.CRITICAL

    def test_hysteresis_prevents_flapping(self, enforcer_system_only):
        """When in YELLOW, dropping to 0.79 stays YELLOW (within hysteresis band)."""
        # hysteresis_band = 0.02, so effective yellow = 0.80 - 0.02 = 0.78
        # At 0.79 with prev=YELLOW, still YELLOW
        assert enforcer_system_only._calculate_system_zone(0.79, PressureZone.YELLOW) == PressureZone.YELLOW

    def test_hysteresis_allows_drop(self, enforcer_system_only):
        """When in YELLOW, dropping below 0.78 returns to GREEN."""
        # hysteresis_band = 0.02, so effective yellow = 0.80 - 0.02 = 0.78
        # At 0.77 with prev=YELLOW, drops to GREEN
        assert enforcer_system_only._calculate_system_zone(0.77, PressureZone.YELLOW) == PressureZone.GREEN

    def test_hysteresis_red_to_yellow(self, enforcer_system_only):
        """When in RED, dropping to 0.89 stays RED (within band)."""
        # effective red = 0.90 - 0.02 = 0.88
        assert enforcer_system_only._calculate_system_zone(0.89, PressureZone.RED) == PressureZone.RED

    def test_hysteresis_red_drops(self, enforcer_system_only):
        """When in RED, dropping to 0.87 returns to YELLOW."""
        # effective red = 0.90 - 0.02 = 0.88
        # 0.87 < 0.88 but >= effective yellow (0.80 - 0.02 = 0.78) → YELLOW
        assert enforcer_system_only._calculate_system_zone(0.87, PressureZone.RED) == PressureZone.YELLOW


# =========================================================================
# Combined zone calculation tests
# =========================================================================


class TestCombinedZoneCalculation:
    """Tests for combined _calculate_zone() with both signals."""

    def test_metal_green_system_red_gives_red(self, enforcer_system_only):
        """Metal GREEN + system RED = RED (system wins)."""
        # 50% metal = GREEN, 91% system = RED
        metal_bytes = int(10 * 1024**3 * 0.50)
        result = enforcer_system_only._calculate_zone(metal_bytes, 0.91, PressureZone.GREEN)
        assert result.zone == PressureZone.RED
        assert result.driver == "system"

    def test_metal_red_system_green_gives_red(self, enforcer_system_only):
        """Metal RED + system GREEN = RED (metal wins)."""
        metal_bytes = int(10 * 1024**3 * 0.92)
        result = enforcer_system_only._calculate_zone(metal_bytes, 0.50, PressureZone.GREEN)
        assert result.zone == PressureZone.RED
        assert result.driver == "metal"

    def test_both_yellow_gives_yellow(self, enforcer_system_only):
        """Both signals YELLOW = YELLOW (driver=both)."""
        metal_bytes = int(10 * 1024**3 * 0.80)  # YELLOW
        result = enforcer_system_only._calculate_zone(metal_bytes, 0.85, PressureZone.GREEN)
        assert result.zone == PressureZone.YELLOW
        assert result.driver == "both"

    def test_system_disabled_uses_metal_only(self, pool):
        """System monitoring off → metal zone only."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=pool,
            max_bytes=10 * 1024**3,
            system_memory_monitoring=False,
        )
        metal_bytes = int(10 * 1024**3 * 0.92)
        result = enforcer._calculate_zone(metal_bytes, 0.99, PressureZone.GREEN)
        assert result.zone == PressureZone.RED
        assert result.system_zone == PressureZone.GREEN  # disabled = always GREEN
        assert result.driver == "metal"

    def test_zone_result_has_correct_fields(self, enforcer_system_only):
        """ZoneResult contains all attribution fields."""
        metal_bytes = int(10 * 1024**3 * 0.50)
        result = enforcer_system_only._calculate_zone(metal_bytes, 0.50, PressureZone.GREEN)
        assert isinstance(result, ZoneResult)
        assert result.zone == PressureZone.GREEN
        assert result.metal_zone == PressureZone.GREEN
        assert result.system_zone == PressureZone.GREEN
        assert result.driver == "metal"  # Both GREEN, defaults to metal

    def test_system_critical_overrides_metal_green(self, enforcer_system_only):
        """System CRITICAL overrides metal GREEN → CRITICAL."""
        metal_bytes = int(10 * 1024**3 * 0.10)  # GREEN
        result = enforcer_system_only._calculate_zone(metal_bytes, 0.96, PressureZone.GREEN)
        assert result.zone == PressureZone.CRITICAL
        assert result.driver == "system"


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
        assert blocks == 10

    def test_proportional_scaling(self, enforcer):
        """Blocks scale proportionally between yellow and critical."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.85)
            blocks = enforcer._calculate_blocks_to_evict(100)
        assert blocks == 5

    def test_auto_max_per_cycle(self, enforcer):
        """Auto max_per_cycle = 10% of evictable blocks."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.95)
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
        engine.scheduler._prefill_paused = True
        call_count_before = engine.scheduler._prefill_paused
        enforcer._set_prefill_paused(True)
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
        old_pcm = old_engine.scheduler.paged_cache_manager
        old_pcm.get_evictable_blocks.assert_called_once_with(3)
        assert old_pcm.evict_block_permanently.call_count == 3

    def test_skips_engine_without_cache_manager(self, enforcer):
        """Gracefully skips engines without paged_cache_manager."""
        engine = MagicMock()
        scheduler = MagicMock(spec=[])
        engine.scheduler = scheduler
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
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
# Hot cache shrink tests
# =========================================================================


class TestHotCacheShrink:
    """Tests for _shrink_hot_caches() enforcer method."""

    def test_shrink_to_half(self, enforcer):
        """Shrinking to 50% calls shrink_hot_cache with half the max."""
        engine = _make_engine_with_ssd_cache(hot_cache_max_bytes=2 * 1024**2)
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        enforcer._shrink_hot_caches(0.5)
        ssd_mgr = engine.scheduler.paged_ssd_cache_manager
        ssd_mgr.shrink_hot_cache.assert_called_once_with(1 * 1024**2)

    def test_shrink_to_zero(self, enforcer):
        """Shrinking to 0% calls shrink_hot_cache(0)."""
        engine = _make_engine_with_ssd_cache(hot_cache_max_bytes=2 * 1024**2)
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        enforcer._shrink_hot_caches(0.0)
        ssd_mgr = engine.scheduler.paged_ssd_cache_manager
        ssd_mgr.shrink_hot_cache.assert_called_once_with(0)

    def test_increments_shrink_counter(self, enforcer):
        """Hot cache shrink increments the observability counter."""
        engine = _make_engine_with_ssd_cache()
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        assert enforcer._hot_cache_shrink_count == 0
        enforcer._shrink_hot_caches(0.5)
        assert enforcer._hot_cache_shrink_count == 1

    def test_skips_engine_without_ssd_manager(self, enforcer):
        """Gracefully skips engines without paged_ssd_cache_manager."""
        engine = _make_engine_with_cache()
        # No paged_ssd_cache_manager attribute
        enforcer._engine_pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        # Should not raise
        enforcer._shrink_hot_caches(0.5)
        assert enforcer._hot_cache_shrink_count == 0


# =========================================================================
# MLX cache clear tests
# =========================================================================


class TestMLXCacheClear:
    """Tests for _clear_mlx_cache() and _restore_caches()."""

    def test_clears_when_cache_has_memory(self, enforcer):
        """Clears MLX cache when cache memory > 0."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_cache_memory.return_value = 100 * 1024**2
            enforcer._clear_mlx_cache()
        mock_mx.clear_cache.assert_called_once()
        assert enforcer._mlx_cache_clear_count == 1

    def test_noop_when_cache_empty(self, enforcer):
        """Does nothing when MLX cache is empty."""
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_cache_memory.return_value = 0
            enforcer._clear_mlx_cache()
        mock_mx.clear_cache.assert_not_called()
        assert enforcer._mlx_cache_clear_count == 0

    def test_restore_caches_is_safe(self, enforcer):
        """_restore_caches() runs without error."""
        enforcer._restore_caches()  # Should be a no-op


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

    def test_system_watermark_valid(self):
        """Valid system watermarks pass validation."""
        ms = MemorySettings(
            system_watermark_yellow=0.75,
            system_watermark_red=0.85,
            system_watermark_critical=0.92,
        )
        ms.validate()

    def test_system_watermark_invalid_ordering_raises(self):
        """Misordered system watermarks raise ValueError."""
        with pytest.raises(ValueError, match="System watermarks must be ordered"):
            MemorySettings(
                system_watermark_yellow=0.95,
                system_watermark_red=0.85,
                system_watermark_critical=0.80,
            )

    def test_system_watermark_out_of_range_raises(self):
        """System watermark outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="system_watermark_yellow must be in"):
            MemorySettings(system_watermark_yellow=1.0)

    def test_system_settings_round_trip(self):
        """System watermark settings survive to_dict/from_dict."""
        original = MemorySettings(
            system_memory_monitoring=False,
            system_watermark_yellow=0.70,
            system_watermark_red=0.85,
            system_watermark_critical=0.92,
        )
        d = original.to_dict()
        restored = MemorySettings.from_dict(d)
        assert restored.system_memory_monitoring is False
        assert restored.system_watermark_yellow == 0.70
        assert restored.system_watermark_red == 0.85
        assert restored.system_watermark_critical == 0.92


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

    def test_includes_system_memory_fields(self, enforcer):
        """Status includes system_memory dict with all fields."""
        status = enforcer.get_status()
        assert "system_memory" in status
        sm = status["system_memory"]
        assert "enabled" in sm
        assert "total_bytes" in sm
        assert "available_bytes" in sm
        assert "utilization" in sm
        assert "pressure_zone" in sm
        assert "watermarks" in sm
        assert sm["watermarks"]["yellow"] == 0.80
        assert sm["watermarks"]["red"] == 0.90
        assert sm["watermarks"]["critical"] == 0.95

    def test_includes_zone_driver(self, enforcer):
        """Status includes zone_driver field."""
        status = enforcer.get_status()
        assert "zone_driver" in status
        assert status["zone_driver"] == "metal"

    def test_includes_cache_action_counters(self, enforcer):
        """Status includes hot_cache_shrink_count and mlx_cache_clear_count."""
        status = enforcer.get_status()
        assert status["hot_cache_shrink_count"] == 0
        assert status["mlx_cache_clear_count"] == 0


# =========================================================================
# Proactive pressure management integration tests
# =========================================================================


class TestProactivePressureManagement:
    """Tests for _proactive_pressure_management() integration."""

    def test_green_zone_unpauses(self, enforcer):
        """GREEN zone ensures prefills are unpaused."""
        enforcer._prefill_paused = True
        enforcer._system_monitoring_enabled = False  # Isolate metal signal
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
        enforcer._system_monitoring_enabled = False
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
        enforcer._system_monitoring_enabled = False
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
        enforcer._system_monitoring_enabled = False
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

    def test_system_pressure_triggers_hot_cache_shrink(self, pool):
        """System YELLOW pressure triggers hot cache shrink."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=pool,
            max_bytes=10 * 1024**3,
            system_memory_monitoring=True,
            system_watermark_yellow=0.80,
        )
        engine = _make_engine_with_ssd_cache()
        pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.80)  # YELLOW
            # Mock psutil to return high system utilization
            with patch.object(enforcer, "_sample_system_memory", return_value=0.85):
                asyncio.get_event_loop().run_until_complete(
                    enforcer._proactive_pressure_management()
                )
        assert enforcer._current_zone == PressureZone.YELLOW
        ssd_mgr = engine.scheduler.paged_ssd_cache_manager
        ssd_mgr.shrink_hot_cache.assert_called()

    def test_green_zone_restores_caches(self, pool):
        """GREEN zone calls _restore_caches after pressure subsides."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=pool,
            max_bytes=10 * 1024**3,
            system_memory_monitoring=False,
        )
        enforcer._current_zone = PressureZone.RED  # Was in RED

        engine = _make_engine_with_cache()
        pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.5)  # GREEN
            asyncio.get_event_loop().run_until_complete(
                enforcer._proactive_pressure_management()
            )
        assert enforcer._current_zone == PressureZone.GREEN
        assert enforcer._prefill_paused is False

    def test_zone_driver_tracked(self, pool):
        """Zone driver is tracked after pressure management runs."""
        enforcer = ProcessMemoryEnforcer(
            engine_pool=pool,
            max_bytes=10 * 1024**3,
            system_memory_monitoring=True,
            system_watermark_yellow=0.80,
        )
        engine = _make_engine_with_cache()
        pool._entries = {
            "m1": _make_entry("m1", engine=engine),
        }
        with patch("omlx.process_memory_enforcer.mx") as mock_mx:
            # Metal is GREEN (50%), system will be RED
            mock_mx.get_active_memory.return_value = int(10 * 1024**3 * 0.50)
            mock_mx.get_cache_memory.return_value = 0  # No MLX cache to clear
            with patch.object(enforcer, "_sample_system_memory", return_value=0.92):
                asyncio.get_event_loop().run_until_complete(
                    enforcer._proactive_pressure_management()
                )
        assert enforcer._zone_driver == "system"
