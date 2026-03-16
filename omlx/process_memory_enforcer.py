# SPDX-License-Identifier: Apache-2.0
"""
Process-level memory enforcer for oMLX.

Monitors two signals:
  1. Metal VRAM usage via mx.get_active_memory()
  2. System-wide RAM availability via psutil.virtual_memory()

The more restrictive signal drives the pressure zone. On Apple Silicon's
unified memory, freeing Metal allocations also relieves system RAM pressure.

Pressure zones (GREEN → YELLOW → RED → CRITICAL) enable graduated response:
- GREEN: normal operation
- YELLOW: proactive KV cache block eviction + shrink hot cache to 50%
- RED: aggressive eviction + pause prefills + clear MLX cache + shrink hot cache to 0%
- CRITICAL: abort requests + unload models
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from .engine_pool import EnginePool
    from .model_settings import ModelSettingsManager

logger = logging.getLogger(__name__)


class PressureZone(IntEnum):
    """Memory pressure zones for graduated response.

    IntEnum so zones support ordering comparisons (GREEN < YELLOW < RED < CRITICAL).
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        """Human-readable label for JSON/logging."""
        return self.name.lower()


@dataclass
class ZoneResult:
    """Result of combined zone calculation with attribution."""

    zone: PressureZone
    metal_zone: PressureZone
    system_zone: PressureZone
    driver: str  # "metal" | "system" | "both"


def _format_gb(b: int) -> str:
    """Format bytes as GB string."""
    return f"{b / 1024**3:.1f}GB"


class ProcessMemoryEnforcer:
    """
    Background task that enforces process-level memory limits.

    Monitors both Metal VRAM and system RAM, using the more restrictive
    signal to drive the pressure zone. Graduated response actions include
    KV cache eviction, hot cache shrinking, MLX cache clearing, prefill
    pausing, and model unloading.
    """

    def __init__(
        self,
        engine_pool: EnginePool,
        max_bytes: int,
        poll_interval: float = 1.0,
        settings_manager: ModelSettingsManager | None = None,
        pressure_management_enabled: bool = True,
        watermark_yellow: float = 0.75,
        watermark_red: float = 0.90,
        watermark_critical: float = 0.95,
        target_free_bytes: int | None = None,
        max_evict_blocks_per_cycle: int = 0,
        hysteresis_band: float = 0.02,
        # Dual-signal: system RAM monitoring
        system_memory_monitoring: bool = True,
        system_watermark_yellow: float = 0.80,
        system_watermark_red: float = 0.90,
        system_watermark_critical: float = 0.95,
    ):
        self._engine_pool = engine_pool
        self._max_bytes = max_bytes
        self._poll_interval = poll_interval
        self._settings_manager = settings_manager
        self._pressure_management_enabled = pressure_management_enabled
        self._watermark_yellow = watermark_yellow
        self._watermark_red = watermark_red
        self._watermark_critical = watermark_critical
        self._target_free_bytes = target_free_bytes
        self._max_evict_blocks_per_cycle = max_evict_blocks_per_cycle
        self._hysteresis_band = hysteresis_band
        self._task: asyncio.Task | None = None
        self._running = False
        self._current_zone: PressureZone = PressureZone.GREEN
        self._prefill_paused: bool = False

        # Dual-signal: system RAM settings
        self._system_monitoring_enabled = system_memory_monitoring
        self._sys_watermark_yellow = system_watermark_yellow
        self._sys_watermark_red = system_watermark_red
        self._sys_watermark_critical = system_watermark_critical

        # Global TTL (seconds); 0 = disabled
        self._global_ttl_seconds: int = 0

        # Zone attribution tracking
        self._current_metal_zone: PressureZone = PressureZone.GREEN
        self._current_system_zone: PressureZone = PressureZone.GREEN
        self._zone_driver: str = "metal"

        # Last sampled system memory values (for status reporting)
        self._last_system_total: int = 0
        self._last_system_available: int = 0
        self._last_system_util: float = 0.0

        # Observability counters
        self._transition_count: int = 0
        self._zone_entry_counts: dict = {zone: 0 for zone in PressureZone}
        self._total_blocks_evicted: int = 0
        self._last_transition_at: float = 0.0
        self._prefill_pause_count: int = 0
        self._zone_entry_time: float = time.monotonic()
        self._hot_cache_shrink_count: int = 0
        self._mlx_cache_clear_count: int = 0

    @property
    def max_bytes(self) -> int:
        """Maximum allowed Metal memory in bytes."""
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int) -> None:
        old = self._max_bytes
        self._max_bytes = value
        if self._running:
            self._propagate_memory_limit()
        logger.info(
            f"Process memory limit changed: "
            f"{_format_gb(old)} -> {_format_gb(value)}"
        )

    @property
    def is_running(self) -> bool:
        """Whether the enforcement loop is active."""
        return self._running

    def start(self) -> None:
        """Start the background enforcement loop."""
        if self._running:
            return
        self._running = True
        self._propagate_memory_limit()
        self._task = asyncio.create_task(self._enforcement_loop())
        logger.info(
            f"Process memory enforcer started "
            f"(limit: {_format_gb(self._max_bytes)}, "
            f"interval: {self._poll_interval}s, "
            f"system_ram_monitoring: {self._system_monitoring_enabled})"
        )
        logger.info(
            f"Prefill state: {'paused' if self._prefill_paused else 'unpaused'} "
            f"(enforcer initialized)"
        )

    def _get_hard_limit_bytes(self) -> int:
        """Hard limit for inline prefill check: system_ram - 4GB.

        Returns 0 if enforcement is disabled (max_bytes <= 0).
        Always >= max_bytes so prefill gets headroom above the soft limit.
        """
        if self._max_bytes <= 0:
            return 0
        from .settings import get_system_memory

        return max(get_system_memory() - 4 * 1024**3, self._max_bytes)

    def _propagate_memory_limit(self) -> None:
        """Propagate soft/hard memory limits to schedulers for inline prefill checking."""
        hard_limit = self._get_hard_limit_bytes()
        for entry in self._engine_pool._entries.values():
            if entry.engine is not None:
                scheduler = getattr(entry.engine, "scheduler", None)
                if scheduler is not None:
                    scheduler._memory_limit_bytes = self._max_bytes
                    scheduler._memory_hard_limit_bytes = hard_limit
                    bg = getattr(scheduler, "batch_generator", None)
                    if bg is not None and hasattr(bg, "_memory_limit_bytes"):
                        bg._memory_limit_bytes = self._max_bytes
                        bg._memory_hard_limit_bytes = hard_limit

    async def stop(self) -> None:
        """Stop the background enforcement loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Process memory enforcer stopped")

    # =========================================================================
    # System memory sampling
    # =========================================================================

    def _sample_system_memory(self) -> float:
        """Sample system-wide RAM utilization.

        Returns:
            System utilization as a float in [0.0, 1.0].
            Returns 0.0 if psutil is unavailable.
        """
        try:
            import psutil
            vm = psutil.virtual_memory()
            self._last_system_total = vm.total
            self._last_system_available = vm.available
            self._last_system_util = 1.0 - (vm.available / vm.total)
            return self._last_system_util
        except Exception:
            return 0.0

    # =========================================================================
    # Pressure zone calculation (dual-signal)
    # =========================================================================

    def _calculate_metal_zone(
        self, metal_bytes: int, prev_zone: PressureZone
    ) -> PressureZone:
        """Calculate pressure zone from Metal VRAM usage.

        Uses the MORE RESTRICTIVE of watermark-based and target-free-based
        calculations. Applies direction-aware hysteresis.
        """
        if self._max_bytes <= 0:
            return PressureZone.GREEN

        utilization = metal_bytes / self._max_bytes
        hb = self._hysteresis_band

        eff_yellow = self._watermark_yellow - (hb if prev_zone >= PressureZone.YELLOW else 0)
        eff_red = self._watermark_red - (hb if prev_zone >= PressureZone.RED else 0)
        eff_crit = self._watermark_critical - (hb if prev_zone >= PressureZone.CRITICAL else 0)

        if utilization >= eff_crit:
            wm_zone = PressureZone.CRITICAL
        elif utilization >= eff_red:
            wm_zone = PressureZone.RED
        elif utilization >= eff_yellow:
            wm_zone = PressureZone.YELLOW
        else:
            wm_zone = PressureZone.GREEN

        # Target-free-based zone with hysteresis (if configured)
        if self._target_free_bytes is not None:
            free = self._max_bytes - metal_bytes
            tf_hb = self._target_free_bytes * hb
            if free < 0:
                tf_zone = PressureZone.CRITICAL
            elif free < self._target_free_bytes * 0.25 + (tf_hb if prev_zone < PressureZone.RED else 0):
                tf_zone = PressureZone.RED
            elif free < self._target_free_bytes + (tf_hb if prev_zone < PressureZone.YELLOW else 0):
                tf_zone = PressureZone.YELLOW
            else:
                tf_zone = PressureZone.GREEN

            result_zone = max(wm_zone, tf_zone)
            logger.debug(
                f"Metal zone: current={_format_gb(metal_bytes)}, "
                f"utilization={utilization:.1%}, "
                f"wm={wm_zone.label}, tf={tf_zone.label}, "
                f"result={result_zone.label}"
            )
            return result_zone

        if wm_zone != PressureZone.GREEN:
            logger.debug(
                f"Metal zone: current={_format_gb(metal_bytes)}, "
                f"utilization={utilization:.1%}, result={wm_zone.label}"
            )
        return wm_zone

    def _calculate_system_zone(
        self, system_util: float, prev_zone: PressureZone
    ) -> PressureZone:
        """Calculate pressure zone from system RAM utilization.

        Uses direction-aware hysteresis with system-specific watermarks.

        Args:
            system_util: System RAM utilization as a float in [0.0, 1.0].
            prev_zone: The previous combined zone (for hysteresis).
        """
        hb = self._hysteresis_band

        eff_yellow = self._sys_watermark_yellow - (hb if prev_zone >= PressureZone.YELLOW else 0)
        eff_red = self._sys_watermark_red - (hb if prev_zone >= PressureZone.RED else 0)
        eff_crit = self._sys_watermark_critical - (hb if prev_zone >= PressureZone.CRITICAL else 0)

        if system_util >= eff_crit:
            return PressureZone.CRITICAL
        elif system_util >= eff_red:
            return PressureZone.RED
        elif system_util >= eff_yellow:
            return PressureZone.YELLOW
        return PressureZone.GREEN

    def _calculate_zone(
        self,
        metal_bytes: int,
        system_util: float,
        prev_zone: PressureZone,
    ) -> ZoneResult:
        """Calculate combined pressure zone from both signals.

        The more restrictive signal wins. Returns a ZoneResult with
        attribution for monitoring/debugging.
        """
        metal_zone = self._calculate_metal_zone(metal_bytes, prev_zone)
        system_zone = (
            self._calculate_system_zone(system_util, prev_zone)
            if self._system_monitoring_enabled
            else PressureZone.GREEN
        )

        combined = max(metal_zone, system_zone)

        if metal_zone > system_zone:
            driver = "metal"
        elif system_zone > metal_zone:
            driver = "system"
        elif combined == PressureZone.GREEN:
            driver = "metal"  # Both GREEN, default to metal
        else:
            driver = "both"

        return ZoneResult(
            zone=combined,
            metal_zone=metal_zone,
            system_zone=system_zone,
            driver=driver,
        )

    # =========================================================================
    # Adaptive eviction rate
    # =========================================================================

    def _calculate_blocks_to_evict(
        self, total_evictable_blocks: int
    ) -> int:
        """Calculate how many blocks to evict based on urgency.

        Linear scaling from yellow threshold to critical threshold.
        """
        if self._max_bytes <= 0 or total_evictable_blocks == 0:
            return 0

        current = mx.get_active_memory()
        utilization = current / self._max_bytes

        if self._max_evict_blocks_per_cycle > 0:
            max_per_cycle = self._max_evict_blocks_per_cycle
        else:
            max_per_cycle = max(1, total_evictable_blocks // 10)

        span = self._watermark_critical - self._watermark_yellow
        if span <= 0:
            return max_per_cycle

        urgency = (utilization - self._watermark_yellow) / span
        urgency = max(0.0, min(1.0, urgency))

        result = max(1, int(urgency * max_per_cycle))
        logger.debug(
            f"Eviction rate: urgency={urgency:.2f}, max_per_cycle={max_per_cycle}, "
            f"blocks_to_evict={result}, total_evictable={total_evictable_blocks}"
        )
        return result

    # =========================================================================
    # KV cache block eviction across engines
    # =========================================================================

    def _evict_blocks_across_engines(self, count: int) -> int:
        """Evict KV cache blocks across all engines, oldest first."""
        if count <= 0:
            return 0

        total_evicted = 0
        remaining = count

        sorted_entries = sorted(
            self._engine_pool._entries.values(),
            key=lambda e: e.last_access,
        )

        for entry in sorted_entries:
            if remaining <= 0:
                break
            if entry.engine is None:
                continue
            scheduler = getattr(entry.engine, "scheduler", None)
            if scheduler is None:
                continue
            pcm = getattr(scheduler, "paged_cache_manager", None)
            if pcm is None:
                continue

            blocks = pcm.get_evictable_blocks(remaining)
            evicted_from_engine = 0
            for block in blocks:
                if pcm.evict_block_permanently(block.block_id):
                    total_evicted += 1
                    evicted_from_engine += 1
                    remaining -= 1
                    if remaining <= 0:
                        break
            if evicted_from_engine > 0:
                logger.debug(
                    f"Evicted {evicted_from_engine} blocks from engine '{entry.model_id}' "
                    f"(requested {evicted_from_engine + remaining})"
                )

        if total_evicted > 0:
            logger.info(
                f"Evicted {total_evicted} KV cache blocks "
                f"(zone: {self._current_zone.label})"
            )

        return total_evicted

    def _get_total_evictable_blocks(self) -> int:
        """Count total evictable blocks across all engines."""
        total = 0
        for entry in self._engine_pool._entries.values():
            if entry.engine is None:
                continue
            scheduler = getattr(entry.engine, "scheduler", None)
            if scheduler is None:
                continue
            pcm = getattr(scheduler, "paged_cache_manager", None)
            if pcm is None:
                continue
            total += len(pcm.get_evictable_blocks(pcm.max_blocks))
        return total

    # =========================================================================
    # Prefill pause propagation
    # =========================================================================

    def _set_prefill_paused(self, paused: bool) -> None:
        """Propagate prefill pause flag to all schedulers."""
        if self._prefill_paused == paused:
            return

        self._prefill_paused = paused
        if paused:
            self._prefill_pause_count += 1
        for entry in self._engine_pool._entries.values():
            if entry.engine is not None:
                scheduler = getattr(entry.engine, "scheduler", None)
                if scheduler is not None:
                    scheduler._prefill_paused = paused

        logger.info(
            f"Prefill {'paused' if paused else 'resumed'} "
            f"(zone: {self._current_zone.label})"
        )

    # =========================================================================
    # System RAM response actions
    # =========================================================================

    def _shrink_hot_caches(self, target_ratio: float) -> None:
        """Evict hot cache entries to reduce Python heap footprint.

        Args:
            target_ratio: Fraction of current hot_cache_max_bytes to keep.
                          0.5 = shrink to half. 0.0 = evict everything.
        """
        for entry in self._engine_pool._entries.values():
            if entry.engine is None:
                continue
            scheduler = getattr(entry.engine, "scheduler", None)
            if scheduler is None:
                continue
            ssd_mgr = getattr(scheduler, "paged_ssd_cache_manager", None)
            if ssd_mgr is None:
                continue
            if not hasattr(ssd_mgr, "shrink_hot_cache") or not hasattr(ssd_mgr, "_hot_cache_max_bytes"):
                continue
            target_bytes = int(ssd_mgr._hot_cache_max_bytes * target_ratio)
            evicted = ssd_mgr.shrink_hot_cache(target_bytes)
            if isinstance(evicted, int) and evicted > 0:
                self._hot_cache_shrink_count += 1
                logger.debug(
                    f"Shrunk hot cache: evicted {evicted} entries "
                    f"(target_ratio={target_ratio})"
                )

    def _clear_mlx_cache(self) -> None:
        """Clear MLX internal allocator cache to free unified memory.

        Uses mx.clear_cache() which releases cached allocations back to the
        OS without changing the cache limit. The cache rebuilds naturally
        during subsequent operations.
        """
        cache_bytes = mx.get_cache_memory()
        if cache_bytes > 0:
            mx.clear_cache()
            self._mlx_cache_clear_count += 1
            logger.info(
                f"Cleared MLX cache: freed {_format_gb(cache_bytes)} "
                f"(zone: {self._current_zone.label})"
            )

    def _restore_caches(self) -> None:
        """Restore cache limits when returning to GREEN zone.

        Currently a no-op since clear_cache() doesn't alter the limit.
        Kept as a hook for future cache management actions.
        """
        pass

    # =========================================================================
    # Proactive pressure management
    # =========================================================================

    async def _proactive_pressure_management(self) -> None:
        """Proactive memory management based on dual-signal pressure zones.

        Samples both Metal VRAM and system RAM each tick. The more
        restrictive signal drives the zone. Actions are graduated:
        - GREEN: ensure prefills unpaused, restore caches
        - YELLOW: evict KV blocks + shrink hot cache to 50%
        - RED: aggressive evict + pause prefills + clear MLX cache + shrink hot cache to 0%
        - CRITICAL: ensure paused, defer to _check_and_enforce()
        """
        if not self._pressure_management_enabled:
            return
        if self._max_bytes <= 0:
            return

        # Sample both signals
        metal_bytes = mx.get_active_memory()
        system_util = (
            self._sample_system_memory()
            if self._system_monitoring_enabled
            else 0.0
        )

        result = self._calculate_zone(metal_bytes, system_util, self._current_zone)

        # Log zone transitions with driver attribution
        zone_changed = result.zone != self._current_zone
        if zone_changed:
            metal_util = metal_bytes / self._max_bytes if self._max_bytes > 0 else 0.0
            logger.info(
                f"Pressure zone: {self._current_zone.label} -> "
                f"{result.zone.label} "
                f"(driver: {result.driver}, "
                f"system_util={system_util:.2f}, metal_util={metal_util:.2f})"
            )
            self._transition_count += 1
            self._zone_entry_counts[result.zone] += 1
            self._last_transition_at = time.monotonic()
            self._zone_entry_time = time.monotonic()
            self._current_zone = result.zone

        # Update attribution state
        self._current_metal_zone = result.metal_zone
        self._current_system_zone = result.system_zone
        self._zone_driver = result.driver

        if self._current_zone == PressureZone.GREEN:
            self._set_prefill_paused(False)
            if zone_changed:
                self._restore_caches()
                logger.debug("GREEN: restored caches, no action needed")
            return

        if self._current_zone == PressureZone.YELLOW:
            self._set_prefill_paused(False)
            total_evictable = self._get_total_evictable_blocks()
            blocks_to_evict = self._calculate_blocks_to_evict(total_evictable)
            evicted = self._evict_blocks_across_engines(blocks_to_evict)
            self._total_blocks_evicted += evicted
            # System RAM action: shrink hot cache to 50%
            if result.system_zone >= PressureZone.YELLOW:
                self._shrink_hot_caches(0.5)
            logger.debug(f"YELLOW action: evicted {evicted} blocks, prefill_paused=False")
            return

        if self._current_zone == PressureZone.RED:
            self._set_prefill_paused(True)
            total_evictable = self._get_total_evictable_blocks()
            blocks_to_evict = self._calculate_blocks_to_evict(total_evictable)
            evicted = self._evict_blocks_across_engines(blocks_to_evict)
            self._total_blocks_evicted += evicted
            # System RAM actions: shrink hot cache to 0% + clear MLX cache
            if result.system_zone >= PressureZone.RED:
                self._shrink_hot_caches(0.0)
                self._clear_mlx_cache()
            logger.debug(f"RED action: evicted {evicted} blocks, prefill_paused=True")
            return

        # CRITICAL: ensure paused, defer to _check_and_enforce()
        self._set_prefill_paused(True)
        logger.debug("CRITICAL action: prefill paused, deferring to _check_and_enforce")

    # =========================================================================
    # Enforcement loop
    # =========================================================================

    async def _enforcement_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._proactive_pressure_management()
                await self._check_and_enforce()
                await self._check_ttl()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process memory enforcer error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _check_ttl(self) -> None:
        """Check and unload models that exceeded their TTL."""
        if self._settings_manager is None:
            return
        await self._engine_pool.check_ttl_expirations(
            self._settings_manager,
            global_ttl_seconds=self._global_ttl_seconds,
        )

    async def _check_and_enforce(self) -> None:
        """Check current memory and enforce limit if exceeded.

        Only runs when in CRITICAL zone. YELLOW and RED are handled by
        _proactive_pressure_management().
        """
        if self._max_bytes <= 0:
            return

        if self._current_zone != PressureZone.CRITICAL:
            return

        current = mx.get_active_memory()
        if current <= self._max_bytes:
            return

        overage = current - self._max_bytes
        logger.warning(
            f"Process memory limit exceeded: "
            f"{_format_gb(current)} / {_format_gb(self._max_bytes)} "
            f"(over by {_format_gb(overage)})"
        )

        async with self._engine_pool._lock:
            while mx.get_active_memory() > self._max_bytes:
                victim = self._engine_pool._find_lru_victim()
                if victim is not None:
                    loaded_non_pinned = [
                        mid
                        for mid, e in self._engine_pool._entries.items()
                        if e.engine is not None and not e.is_pinned
                    ]
                    if len(loaded_non_pinned) > 1:
                        entry = self._engine_pool._entries.get(victim)
                        if entry and entry.engine is not None:
                            if hasattr(entry.engine, "abort_all_requests"):
                                aborted = await entry.engine.abort_all_requests()
                                if aborted > 0:
                                    logger.warning(
                                        f"Aborted {aborted} requests on "
                                        f"'{victim}' before eviction"
                                    )
                        logger.warning(
                            f"Evicting model '{victim}' to enforce "
                            f"process memory limit"
                        )
                        await self._engine_pool._unload_engine(victim)
                        continue
                    else:
                        entry = self._engine_pool._entries.get(victim)
                        if entry and entry.engine is not None:
                            if hasattr(entry.engine, "abort_all_requests"):
                                aborted = await entry.engine.abort_all_requests()
                                if aborted > 0:
                                    logger.warning(
                                        f"Aborted {aborted} requests on "
                                        f"'{victim}' due to memory pressure "
                                        f"(model kept loaded)"
                                    )
                        break

                aborted_any = False
                for entry in self._engine_pool._entries.values():
                    if entry.is_loading and not entry.abort_loading:
                        logger.warning(
                            f"Requesting abort of loading model "
                            f"'{entry.model_id}' — process memory "
                            f"limit exceeded"
                        )
                        entry.abort_loading = True
                        aborted_any = True

                if not aborted_any:
                    has_loaded = any(
                        e.engine is not None
                        for e in self._engine_pool._entries.values()
                    )
                    if has_loaded:
                        logger.warning(
                            "Process memory limit exceeded but all "
                            "loaded models are pinned — cannot evict."
                        )
                    else:
                        logger.warning(
                            "Process memory limit exceeded but no "
                            "models are loaded to evict."
                        )
                break

    def get_status(self) -> dict:
        """Get enforcer status for monitoring endpoints."""
        current = mx.get_active_memory() if self._running else 0
        now = time.monotonic()
        return {
            "enabled": self._running,
            "pressure_management_enabled": self._pressure_management_enabled,
            "max_bytes": self._max_bytes,
            "max_formatted": _format_gb(self._max_bytes),
            "current_bytes": current,
            "current_formatted": _format_gb(current),
            "utilization": (
                current / self._max_bytes if self._max_bytes > 0 else 0.0
            ),
            "pressure_zone": self._current_zone.label,
            "prefill_paused": self._prefill_paused,
            "watermarks": {
                "yellow": self._watermark_yellow,
                "red": self._watermark_red,
                "critical": self._watermark_critical,
            },
            "target_free_bytes": self._target_free_bytes,
            "target_free_formatted": (
                _format_gb(self._target_free_bytes)
                if self._target_free_bytes is not None
                else "disabled"
            ),
            "transition_count": self._transition_count,
            "zone_entry_counts": {
                zone.label: count
                for zone, count in self._zone_entry_counts.items()
            },
            "total_blocks_evicted": self._total_blocks_evicted,
            "last_transition_seconds_ago": (
                round(now - self._last_transition_at, 1)
                if self._last_transition_at > 0
                else None
            ),
            "prefill_pause_count": self._prefill_pause_count,
            "time_in_current_zone_seconds": round(
                now - self._zone_entry_time, 1
            ),
            # Dual-signal: system memory metrics
            "system_memory": {
                "enabled": self._system_monitoring_enabled,
                "total_bytes": self._last_system_total,
                "total_formatted": _format_gb(self._last_system_total) if self._last_system_total > 0 else "N/A",
                "available_bytes": self._last_system_available,
                "available_formatted": _format_gb(self._last_system_available) if self._last_system_available > 0 else "N/A",
                "utilization": self._last_system_util,
                "pressure_zone": self._current_system_zone.label,
                "watermarks": {
                    "yellow": self._sys_watermark_yellow,
                    "red": self._sys_watermark_red,
                    "critical": self._sys_watermark_critical,
                },
            },
            "zone_driver": self._zone_driver,
            "hot_cache_shrink_count": self._hot_cache_shrink_count,
            "mlx_cache_clear_count": self._mlx_cache_clear_count,
        }
