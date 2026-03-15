"""SSD block prefetcher for lazy cache restore.

Pre-reads cache blocks from SSD in background threads to reduce
cache-hit restore latency. Worker threads read raw bytes only;
mx.load() conversions happen on the main thread (Metal GPU constraint).
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PrefetchResult:
    """Result of a single block prefetch."""
    block_hash: bytes  # Raw bytes hash identifying the block
    raw_bytes: bytes   # Raw file bytes, NOT mx.array
    read_time: float   # Time to read from SSD


@dataclass
class PrefetchJob:
    """Tracks an active prefetch job for a request."""
    request_id: str
    block_hashes: List[bytes]
    futures: List[Future] = field(default_factory=list)
    results: List[PrefetchResult] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.time)
    completed: bool = False
    cancelled: bool = False


@dataclass
class PrefetchStats:
    """Observability counters for prefetch operations."""
    jobs_submitted: int = 0
    jobs_completed: int = 0
    jobs_cancelled: int = 0
    blocks_prefetched: int = 0
    blocks_used: int = 0
    blocks_wasted: int = 0   # Prefetched but never consumed
    total_read_time: float = 0.0
    fallback_count: int = 0  # Times we fell back to sync read


MAX_ACTIVE_JOBS = 8
MAX_BLOCKS_PER_JOB = 256
NUM_WORKERS = 2


class SSDBlockPrefetcher:
    """Async SSD block prefetcher using a thread pool.

    IMPORTANT: Worker threads read raw bytes from SSD only (pure file I/O).
    mx.load() from worker threads deadlocks Metal GPU. Conversion from
    raw bytes to mx.arrays must happen on the main thread.

    SSD cache layout mirrors PagedSSDCacheManager:
      <ssd_cache_dir>/<hash_hex[0]>/<hash_hex>.safetensors
    """

    def __init__(self, ssd_cache_dir: Optional[str] = None):
        """Initialize prefetcher.

        Args:
            ssd_cache_dir: Path to SSD cache directory. If None, prefetcher is disabled.
        """
        self._ssd_cache_dir = ssd_cache_dir
        self._executor: Optional[ThreadPoolExecutor] = None
        self._active_jobs: Dict[str, PrefetchJob] = {}
        self._stats = PrefetchStats()
        self._lock = threading.Lock()
        self._enabled = ssd_cache_dir is not None

        if self._enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=NUM_WORKERS,
                thread_name_prefix="ssd-prefetch"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def submit_prefetch(self, request_id: str, block_hashes: List[bytes]) -> bool:
        """Submit a prefetch job for the given blocks.

        Args:
            request_id: ID of the request needing these blocks.
            block_hashes: Hashes identifying blocks to pre-read from SSD.

        Returns:
            True if job was submitted, False if rejected (at capacity or disabled).
        """
        if not self._enabled or not block_hashes:
            return False

        with self._lock:
            if len(self._active_jobs) >= MAX_ACTIVE_JOBS:
                logger.debug(
                    f"Prefetch rejected for {request_id}: at capacity ({MAX_ACTIVE_JOBS} active)"
                )
                return False

            if request_id in self._active_jobs:
                logger.debug(f"Prefetch already active for {request_id}")
                return False

            # Limit blocks per job
            hashes = block_hashes[:MAX_BLOCKS_PER_JOB]

            job = PrefetchJob(request_id=request_id, block_hashes=hashes)

            # Submit read tasks to thread pool
            for block_hash in hashes:
                future = self._executor.submit(self._read_block, block_hash)
                job.futures.append(future)

            self._active_jobs[request_id] = job
            self._stats.jobs_submitted += 1

        logger.debug(f"Prefetch submitted for {request_id}: {len(hashes)} blocks")
        return True

    def _read_block(self, block_hash: bytes) -> Optional[PrefetchResult]:
        """Read a single block from SSD. Runs in worker thread.

        IMPORTANT: Only reads raw bytes. Does NOT call mx.load().

        SSD layout: <ssd_cache_dir>/<hash_hex[0]>/<hash_hex>.safetensors
        """
        import os

        if self._ssd_cache_dir is None:
            return None

        # Construct path matching PagedSSDCacheManager._get_file_path()
        hash_hex = block_hash.hex()
        subdir = hash_hex[0]
        filename = f"{hash_hex}.safetensors"
        block_path = os.path.join(self._ssd_cache_dir, subdir, filename)

        start = time.monotonic()
        try:
            with open(block_path, "rb") as f:
                raw_bytes = f.read()
            elapsed = time.monotonic() - start
            logger.debug(f"Prefetch read block {hash_hex[:16]}: {len(raw_bytes)} bytes in {elapsed*1000:.1f}ms")
            return PrefetchResult(
                block_hash=block_hash,
                raw_bytes=raw_bytes,
                read_time=elapsed,
            )
        except FileNotFoundError:
            logger.debug(f"Prefetch block not found on SSD: {hash_hex}")
            return None
        except OSError as e:
            logger.warning(f"Prefetch read error for block {hash_hex}: {e}")
            return None

    def get_prefetched_data(self, request_id: str) -> Optional[List[PrefetchResult]]:
        """Get prefetched data for a request (destructive read).

        Returns results only if ALL futures have completed. Removes the job.

        Args:
            request_id: The request to get data for.

        Returns:
            List of PrefetchResults if all done, None if still in progress or not found.
        """
        with self._lock:
            job = self._active_jobs.get(request_id)
            if job is None or job.cancelled:
                return None

            # Check if all futures are done
            if not all(f.done() for f in job.futures):
                return None

            # Collect results
            results = []
            for future in job.futures:
                try:
                    result = future.result(timeout=0)
                    if result is not None:
                        results.append(result)
                        self._stats.blocks_prefetched += 1
                        self._stats.total_read_time += result.read_time
                except Exception as e:
                    logger.warning(f"Prefetch future error: {e}")

            # Mark completed and remove
            job.completed = True
            job.results = results
            del self._active_jobs[request_id]
            self._stats.jobs_completed += 1

            if results:
                logger.debug(f"Prefetch data ready for {request_id}: {len(results)} blocks")
            return results if results else None

    def cancel_prefetch(self, request_id: str) -> None:
        """Cancel a pending prefetch job.

        Args:
            request_id: The request to cancel prefetch for.
        """
        with self._lock:
            job = self._active_jobs.get(request_id)
            if job is None:
                return

            job.cancelled = True
            for future in job.futures:
                future.cancel()

            del self._active_jobs[request_id]
            self._stats.jobs_cancelled += 1

        logger.debug(f"Prefetch cancelled for {request_id}")

    def has_pending_prefetch(self, request_id: str) -> bool:
        """Check if a request has a pending prefetch job."""
        with self._lock:
            job = self._active_jobs.get(request_id)
            return job is not None and not job.cancelled

    def mark_blocks_used(self, count: int) -> None:
        """Record that prefetched blocks were actually used."""
        with self._lock:
            self._stats.blocks_used += count

    def mark_blocks_wasted(self, count: int) -> None:
        """Record that prefetched blocks were not used."""
        with self._lock:
            self._stats.blocks_wasted += count

    def record_fallback(self) -> None:
        """Record a fallback to synchronous read."""
        with self._lock:
            self._stats.fallback_count += 1

    def get_stats(self) -> dict:
        """Get prefetch statistics."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "active_jobs": len(self._active_jobs),
                "jobs_submitted": self._stats.jobs_submitted,
                "jobs_completed": self._stats.jobs_completed,
                "jobs_cancelled": self._stats.jobs_cancelled,
                "blocks_prefetched": self._stats.blocks_prefetched,
                "blocks_used": self._stats.blocks_used,
                "blocks_wasted": self._stats.blocks_wasted,
                "total_read_time_ms": round(self._stats.total_read_time * 1000, 2),
                "fallback_count": self._stats.fallback_count,
            }

    def shutdown(self) -> None:
        """Shutdown the prefetcher and clean up resources."""
        with self._lock:
            # Cancel all active jobs
            for job in self._active_jobs.values():
                job.cancelled = True
                for future in job.futures:
                    future.cancel()
            self._active_jobs.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        self._enabled = False
        logger.info("SSD block prefetcher shut down")
