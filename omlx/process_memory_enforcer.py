# SPDX-License-Identifier: Apache-2.0
"""
Process-level memory enforcer for oMLX.

Monitors total Metal memory usage via mx.get_active_memory() and enforces
the max_process_memory limit by unloading LRU models from EnginePool.

The enforcer runs as a background asyncio task that polls memory usage at
a configurable interval (default: 1 second). When usage exceeds the limit,
it immediately unloads the least-recently-used non-pinned model. If the
model is mid-inference, the inference is aborted as part of engine shutdown.

Pressure zones (GREEN → YELLOW → RED → CRITICAL) enable graduated response:
- GREEN: normal operation
- YELLOW: proactive KV cache block eviction (gentle rate)
- RED: aggressive eviction + pause new prefills
- CRITICAL: existing nuclear behavior (abort requests + unload models)
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from .engine_pool import EnginePool
    from .model_settings import ModelSettingsManager

logger = logging.getLogger(__name__)


class PressureZone(Enum):
    """Memory pressure zones for graduated response."""

    GREEN = "green"  # normal
    YELLOW = "yellow"  # proactive offload
    RED = "red"  # aggressive offload + pause prefills
    CRITICAL = "critical"  # abort + unload (existing)


def _format_gb(b: int) -> str:
    """Format bytes as GB string."""
    return f"{b / 1024**3:.1f}GB"


class ProcessMemoryEnforcer:
    """
    Background task that enforces process-level memory limits.

    Polls mx.get_active_memory() every poll_interval seconds and unloads
    LRU models from EnginePool when the limit is exceeded.
    """

    def __init__(
        self,
        engine_pool: EnginePool,
        max_bytes: int,
        poll_interval: float = 1.0,
        settings_manager: ModelSettingsManager | None = None,
        watermark_yellow: float = 0.75,
        watermark_red: float = 0.90,
        watermark_critical: float = 0.95,
        target_free_bytes: int | None = None,
        max_evict_blocks_per_cycle: int = 0,
    ):
        """
        Initialize the process memory enforcer.

        Args:
            engine_pool: The engine pool to evict models from.
            max_bytes: Maximum allowed Metal memory in bytes.
            poll_interval: Seconds between memory checks.
            settings_manager: Optional settings manager for TTL checks.
            watermark_yellow: Utilization fraction to begin proactive offload.
            watermark_red: Utilization fraction for aggressive offload + pause.
            watermark_critical: Utilization fraction for nuclear behavior.
            target_free_bytes: Target free memory floor in bytes, or None.
            max_evict_blocks_per_cycle: Max blocks to evict per cycle (0=auto).
        """
        self._engine_pool = engine_pool
        self._max_bytes = max_bytes
        self._poll_interval = poll_interval
        self._settings_manager = settings_manager
        self._watermark_yellow = watermark_yellow
        self._watermark_red = watermark_red
        self._watermark_critical = watermark_critical
        self._target_free_bytes = target_free_bytes
        self._max_evict_blocks_per_cycle = max_evict_blocks_per_cycle
        self._task: asyncio.Task | None = None
        self._running = False
        self._current_zone: PressureZone = PressureZone.GREEN
        self._prefill_paused: bool = False

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
            f"interval: {self._poll_interval}s)"
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
    # Pressure zone calculation
    # =========================================================================

    def _calculate_zone(self, current_bytes: int) -> PressureZone:
        """Calculate the current pressure zone from memory usage.

        Uses the MORE RESTRICTIVE of watermark-based and target-free-based
        calculations.

        Args:
            current_bytes: Current memory usage in bytes.

        Returns:
            The pressure zone.
        """
        if self._max_bytes <= 0:
            return PressureZone.GREEN

        # Watermark-based zone
        utilization = current_bytes / self._max_bytes
        if utilization >= self._watermark_critical:
            wm_zone = PressureZone.CRITICAL
        elif utilization >= self._watermark_red:
            wm_zone = PressureZone.RED
        elif utilization >= self._watermark_yellow:
            wm_zone = PressureZone.YELLOW
        else:
            wm_zone = PressureZone.GREEN

        # Target-free-based zone (if configured)
        if self._target_free_bytes is not None:
            free = self._max_bytes - current_bytes
            if free < 0:
                tf_zone = PressureZone.CRITICAL
            elif free < self._target_free_bytes * 0.25:
                tf_zone = PressureZone.RED
            elif free < self._target_free_bytes:
                tf_zone = PressureZone.YELLOW
            else:
                tf_zone = PressureZone.GREEN

            # Use the more restrictive zone
            zone_order = [
                PressureZone.GREEN,
                PressureZone.YELLOW,
                PressureZone.RED,
                PressureZone.CRITICAL,
            ]
            wm_idx = zone_order.index(wm_zone)
            tf_idx = zone_order.index(tf_zone)
            return zone_order[max(wm_idx, tf_idx)]

        return wm_zone

    # =========================================================================
    # Adaptive eviction rate
    # =========================================================================

    def _calculate_blocks_to_evict(
        self, total_evictable_blocks: int
    ) -> int:
        """Calculate how many blocks to evict based on urgency.

        Linear scaling from yellow threshold to critical threshold.

        Args:
            total_evictable_blocks: Total number of evictable blocks across
                all engines.

        Returns:
            Number of blocks to evict this cycle.
        """
        if self._max_bytes <= 0 or total_evictable_blocks == 0:
            return 0

        current = mx.get_active_memory()
        utilization = current / self._max_bytes

        # Calculate max blocks per cycle
        if self._max_evict_blocks_per_cycle > 0:
            max_per_cycle = self._max_evict_blocks_per_cycle
        else:
            # Auto: 10% of evictable blocks, minimum 1
            max_per_cycle = max(1, total_evictable_blocks // 10)

        # Linear urgency scaling
        span = self._watermark_critical - self._watermark_yellow
        if span <= 0:
            return max_per_cycle

        urgency = (utilization - self._watermark_yellow) / span
        urgency = max(0.0, min(1.0, urgency))

        return max(1, int(urgency * max_per_cycle))

    # =========================================================================
    # KV cache block eviction across engines
    # =========================================================================

    def _evict_blocks_across_engines(self, count: int) -> int:
        """Evict KV cache blocks across all engines, oldest first.

        Iterates engine pool entries sorted by last_access (oldest first).
        For each engine, gets evictable blocks from paged_cache_manager and
        evicts them permanently.

        Args:
            count: Number of blocks to evict.

        Returns:
            Total blocks actually evicted.
        """
        if count <= 0:
            return 0

        total_evicted = 0
        remaining = count

        # Sort entries by last_access (oldest first)
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
            for block in blocks:
                if pcm.evict_block_permanently(block.block_id):
                    total_evicted += 1
                    remaining -= 1
                    if remaining <= 0:
                        break

        if total_evicted > 0:
            logger.info(
                f"Evicted {total_evicted} KV cache blocks "
                f"(zone: {self._current_zone.value})"
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
        """Propagate prefill pause flag to all schedulers.

        Args:
            paused: Whether new prefills should be paused.
        """
        if self._prefill_paused == paused:
            return

        self._prefill_paused = paused
        for entry in self._engine_pool._entries.values():
            if entry.engine is not None:
                scheduler = getattr(entry.engine, "scheduler", None)
                if scheduler is not None:
                    scheduler._prefill_paused = paused

        logger.info(
            f"Prefill {'paused' if paused else 'resumed'} "
            f"(zone: {self._current_zone.value})"
        )

    # =========================================================================
    # Proactive pressure management
    # =========================================================================

    async def _proactive_pressure_management(self) -> None:
        """Proactive memory management based on pressure zones.

        Runs BEFORE _check_and_enforce() in the enforcement loop.
        - GREEN: ensure prefills unpaused, return
        - YELLOW: evict blocks (gentle rate), prefills stay unpaused
        - RED: evict blocks (aggressive rate), pause new prefills
        - CRITICAL: ensure paused, defer to existing _check_and_enforce()
        """
        if self._max_bytes <= 0:
            return

        current = mx.get_active_memory()
        new_zone = self._calculate_zone(current)

        # Log zone transitions
        if new_zone != self._current_zone:
            logger.info(
                f"Pressure zone: {self._current_zone.value} -> "
                f"{new_zone.value} "
                f"({_format_gb(current)} / {_format_gb(self._max_bytes)}, "
                f"{current / self._max_bytes:.1%})"
            )
            self._current_zone = new_zone

        if self._current_zone == PressureZone.GREEN:
            self._set_prefill_paused(False)
            return

        if self._current_zone == PressureZone.YELLOW:
            self._set_prefill_paused(False)
            total_evictable = self._get_total_evictable_blocks()
            blocks_to_evict = self._calculate_blocks_to_evict(
                total_evictable
            )
            self._evict_blocks_across_engines(blocks_to_evict)
            return

        if self._current_zone == PressureZone.RED:
            self._set_prefill_paused(True)
            total_evictable = self._get_total_evictable_blocks()
            blocks_to_evict = self._calculate_blocks_to_evict(
                total_evictable
            )
            self._evict_blocks_across_engines(blocks_to_evict)
            return

        # CRITICAL: ensure paused, defer to _check_and_enforce()
        self._set_prefill_paused(True)

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
        await self._engine_pool.check_ttl_expirations(self._settings_manager)

    async def _check_and_enforce(self) -> None:
        """Check current memory and enforce limit if exceeded.

        Only runs when in CRITICAL zone. YELLOW and RED are handled by
        _proactive_pressure_management().

        Handles three scenarios via the while loop:
        1. Multiple models, one inferring: evict LRU (idle) model,
           inference on the other continues.
        2. Single model: abort all requests, keep model loaded.
           Short-context requests can be served afterward.
        3. Multiple models, both inferring: first iteration evicts LRU
           (aborting its requests), second iteration aborts remaining
           single model's requests.
        """
        if self._max_bytes <= 0:
            return

        # Only run nuclear enforcement in CRITICAL zone
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

        # Acquire EnginePool lock and unload LRU models until under limit.
        # Note: prefill loops self-check via _memory_limit_bytes (same thread,
        # no GIL issue), so they will abort independently of this enforcer.
        async with self._engine_pool._lock:
            while mx.get_active_memory() > self._max_bytes:
                victim = self._engine_pool._find_lru_victim()
                if victim is not None:
                    # Count loaded non-pinned models
                    loaded_non_pinned = [
                        mid
                        for mid, e in self._engine_pool._entries.items()
                        if e.engine is not None and not e.is_pinned
                    ]
                    if len(loaded_non_pinned) > 1:
                        # Multiple models: evict LRU victim.
                        # First abort active requests so clients receive
                        # error messages — EngineCore.stop() only cancels
                        # the engine loop silently without notifying collectors.
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
                        # Single model: abort all requests, keep model
                        # loaded. This frees KV cache blocks internally
                        # so short-context requests can be served without
                        # new Metal allocation.
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

                # No loaded non-pinned model to evict.
                # Check if any model is currently loading — request abort.
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
                    # Nothing we can do — all models are either pinned
                    # or there are no loaded/loading models
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
        return {
            "enabled": self._running,
            "max_bytes": self._max_bytes,
            "max_formatted": _format_gb(self._max_bytes),
            "current_bytes": current,
            "current_formatted": _format_gb(current),
            "utilization": (
                current / self._max_bytes if self._max_bytes > 0 else 0.0
            ),
            "pressure_zone": self._current_zone.value,
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
        }
