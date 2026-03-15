# SPDX-License-Identifier: Apache-2.0
"""Tests for SSDBlockPrefetcher.

Tests cover:
- Submit and retrieve prefetch results
- Cancel pending prefetch
- Capacity limits (MAX_ACTIVE_JOBS)
- Missing SSD files
- Stats tracking
- Shutdown cleanup
"""

import os
import tempfile
import time

import pytest

from omlx.cache.prefetch import (
    MAX_ACTIVE_JOBS,
    SSDBlockPrefetcher,
)


@pytest.fixture
def ssd_dir(tmp_path):
    """Create a temp SSD cache directory with some block files."""
    cache_dir = tmp_path / "ssd_cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def prefetcher(ssd_dir):
    """Create an enabled prefetcher."""
    pf = SSDBlockPrefetcher(ssd_cache_dir=ssd_dir)
    yield pf
    pf.shutdown()


@pytest.fixture
def disabled_prefetcher():
    """Create a disabled prefetcher (no cache dir)."""
    return SSDBlockPrefetcher(ssd_cache_dir=None)


def _write_block_file(ssd_dir: str, block_hash: bytes, content: bytes = b"test data"):
    """Write a fake block file matching SSD cache layout."""
    hex_hash = block_hash.hex()
    subdir = os.path.join(ssd_dir, hex_hash[0])
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"{hex_hash}.safetensors")
    with open(path, "wb") as f:
        f.write(content)
    return path


# =========================================================================
# Basic functionality
# =========================================================================


class TestSubmitAndRetrieve:
    """Tests for basic submit/retrieve flow."""

    def test_submit_and_get_results(self, prefetcher, ssd_dir):
        """Submit prefetch, wait, get results."""
        h1 = b"\x01" * 16
        _write_block_file(ssd_dir, h1, b"block1data")

        assert prefetcher.submit_prefetch("req-1", [h1]) is True

        # Wait for completion
        for _ in range(100):
            results = prefetcher.get_prefetched_data("req-1")
            if results is not None:
                break
            time.sleep(0.01)

        assert results is not None
        assert len(results) == 1
        assert results[0].block_hash == h1
        assert results[0].raw_bytes == b"block1data"

    def test_disabled_prefetcher_rejects(self, disabled_prefetcher):
        """Disabled prefetcher rejects submissions."""
        assert disabled_prefetcher.submit_prefetch("req-1", [b"\x01"]) is False
        assert disabled_prefetcher.enabled is False

    def test_empty_hashes_rejected(self, prefetcher):
        """Empty block hash list is rejected."""
        assert prefetcher.submit_prefetch("req-1", []) is False

    def test_duplicate_request_rejected(self, prefetcher, ssd_dir):
        """Second submission for same request_id is rejected."""
        h1 = b"\x02" * 16
        _write_block_file(ssd_dir, h1)
        assert prefetcher.submit_prefetch("req-1", [h1]) is True
        assert prefetcher.submit_prefetch("req-1", [h1]) is False
        # Cleanup
        prefetcher.cancel_prefetch("req-1")


class TestCancelPrefetch:
    """Tests for cancellation."""

    def test_cancel_removes_job(self, prefetcher, ssd_dir):
        """Cancelling removes the job."""
        h1 = b"\x03" * 16
        _write_block_file(ssd_dir, h1)
        prefetcher.submit_prefetch("req-1", [h1])
        assert prefetcher.has_pending_prefetch("req-1") is True

        prefetcher.cancel_prefetch("req-1")
        assert prefetcher.has_pending_prefetch("req-1") is False
        assert prefetcher.get_prefetched_data("req-1") is None

    def test_cancel_nonexistent_is_noop(self, prefetcher):
        """Cancelling a non-existent request doesn't raise."""
        prefetcher.cancel_prefetch("doesnt-exist")  # Should not raise


class TestCapacityLimits:
    """Tests for MAX_ACTIVE_JOBS enforcement."""

    def test_rejects_at_capacity(self, prefetcher, ssd_dir):
        """Rejects new jobs when at MAX_ACTIVE_JOBS."""
        # Fill up to capacity with jobs that won't complete (missing files)
        for i in range(MAX_ACTIVE_JOBS):
            h = bytes([i]) * 16
            # Don't create file — job will have futures that return None
            _write_block_file(ssd_dir, h)
            assert prefetcher.submit_prefetch(f"req-{i}", [h]) is True

        # Next one should be rejected
        h_extra = b"\xff" * 16
        _write_block_file(ssd_dir, h_extra)
        assert prefetcher.submit_prefetch("req-overflow", [h_extra]) is False

        # Clean up
        for i in range(MAX_ACTIVE_JOBS):
            prefetcher.cancel_prefetch(f"req-{i}")


class TestMissingSSDFiles:
    """Tests for graceful handling of missing files."""

    def test_missing_file_returns_none_result(self, prefetcher):
        """Missing SSD block file produces no result for that block."""
        h1 = b"\x04" * 16  # No file written

        prefetcher.submit_prefetch("req-1", [h1])

        # Wait for completion
        for _ in range(100):
            results = prefetcher.get_prefetched_data("req-1")
            if results is not None:
                break
            # Check if job completed (all futures done but no results)
            if not prefetcher.has_pending_prefetch("req-1"):
                results = prefetcher.get_prefetched_data("req-1")
                break
            time.sleep(0.01)

        # All blocks missing → returns None (empty results)
        # The job completes but with 0 successful results
        assert results is None


class TestStats:
    """Tests for statistics tracking."""

    def test_stats_initial(self, prefetcher):
        """Initial stats are all zero."""
        stats = prefetcher.get_stats()
        assert stats["enabled"] is True
        assert stats["jobs_submitted"] == 0
        assert stats["jobs_completed"] == 0
        assert stats["active_jobs"] == 0

    def test_stats_after_submit(self, prefetcher, ssd_dir):
        """Stats track submissions."""
        h1 = b"\x05" * 16
        _write_block_file(ssd_dir, h1)
        prefetcher.submit_prefetch("req-1", [h1])

        stats = prefetcher.get_stats()
        assert stats["jobs_submitted"] == 1
        assert stats["active_jobs"] >= 0  # May have completed already

        prefetcher.cancel_prefetch("req-1")

    def test_stats_track_cancellations(self, prefetcher, ssd_dir):
        """Stats track cancelled jobs."""
        h1 = b"\x06" * 16
        _write_block_file(ssd_dir, h1)
        prefetcher.submit_prefetch("req-1", [h1])
        prefetcher.cancel_prefetch("req-1")

        stats = prefetcher.get_stats()
        assert stats["jobs_cancelled"] == 1

    def test_mark_blocks_used(self, prefetcher):
        """mark_blocks_used increments counter."""
        prefetcher.mark_blocks_used(5)
        assert prefetcher.get_stats()["blocks_used"] == 5

    def test_record_fallback(self, prefetcher):
        """record_fallback increments counter."""
        prefetcher.record_fallback()
        prefetcher.record_fallback()
        assert prefetcher.get_stats()["fallback_count"] == 2


class TestShutdown:
    """Tests for clean shutdown."""

    def test_shutdown_cancels_active_jobs(self, ssd_dir):
        """Shutdown cancels all active jobs."""
        pf = SSDBlockPrefetcher(ssd_cache_dir=ssd_dir)
        h1 = b"\x07" * 16
        _write_block_file(ssd_dir, h1)
        pf.submit_prefetch("req-1", [h1])

        pf.shutdown()
        assert pf.enabled is False
        assert pf.get_stats()["active_jobs"] == 0

    def test_double_shutdown_safe(self, ssd_dir):
        """Calling shutdown twice doesn't raise."""
        pf = SSDBlockPrefetcher(ssd_cache_dir=ssd_dir)
        pf.shutdown()
        pf.shutdown()  # Should not raise
