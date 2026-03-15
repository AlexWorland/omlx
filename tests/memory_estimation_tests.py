import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omlx.cache.type_handlers import (
    KVCacheHandler,
    RotatingKVCacheHandler,
    ArraysCacheHandler,
)
from omlx.cache.scheduler_memory_budget import SchedulerMemoryBudget


class TestKVCacheMemoryEstimation:
    """Tests for KVCacheHandler.estimate_memory_per_token()."""

    def test_kv_cache_default_dtype(self):
        """Default 16-bit estimate should return 4 bytes per token."""
        kv = KVCacheHandler()
        result = kv.estimate_memory_per_token(16)
        assert result == 4

    def test_kv_cache_8bit_dtype(self):
        """8-bit estimate should return 2 bytes per token."""
        kv = KVCacheHandler()
        result = kv.estimate_memory_per_token(8)
        assert result == 2

    def test_kv_cache_32bit_dtype(self):
        """32-bit estimate should return 8 bytes per token."""
        kv = KVCacheHandler()
        result = kv.estimate_memory_per_token(32)
        assert result == 8


class TestRotatingKVCacheMemoryEstimation:
    """Tests for RotatingKVCacheHandler.estimate_memory_per_token()."""

    def test_rotating_cache_default_dtype(self):
        """Default 16-bit estimate should return 4 bytes per token."""
        rot = RotatingKVCacheHandler()
        result = rot.estimate_memory_per_token(16)
        assert result == 4

    def test_rotating_cache_8bit_dtype(self):
        """8-bit estimate should return 2 bytes per token."""
        rot = RotatingKVCacheHandler()
        result = rot.estimate_memory_per_token(8)
        assert result == 2


class TestArraysCacheMemoryEstimation:
    """Tests for ArraysCacheHandler.estimate_memory_per_token()."""

    def test_arrays_cache_default_dtype(self):
        """Default 16-bit estimate should return 4 bytes per token."""
        arr = ArraysCacheHandler()
        result = arr.estimate_memory_per_token(16)
        assert result == 4

    def test_arrays_cache_8bit_dtype(self):
        """8-bit estimate should return 2 bytes per token."""
        arr = ArraysCacheHandler()
        result = arr.estimate_memory_per_token(8)
        assert result == 2


class TestMemoryEstimationConsistency:
    """Tests to verify all cache type handlers provide consistent estimates."""

    def test_all_handlers_implement_method(self):
        """All cache handlers should implement estimate_memory_per_token()."""
        kv = KVCacheHandler()
        rot = RotatingKVCacheHandler()
        arr = ArraysCacheHandler()

        for handler in [kv, rot, arr]:
            result = handler.estimate_memory_per_token(16)
            assert isinstance(result, int)
            assert result > 0

    def test_all_handlers_consistent_with_dtype_bits(self):
        """All handlers should use same formula: 2 * (bits // 8)."""
        kv = KVCacheHandler()
        rot = RotatingKVCacheHandler()
        arr = ArraysCacheHandler()

        test_cases = [(8, 2), (16, 4), (32, 8)]

        for bits, expected in test_cases:
            assert kv.estimate_memory_per_token(bits) == expected
            assert rot.estimate_memory_per_token(bits) == expected
            assert arr.estimate_memory_per_token(bits) == expected


class TestSchedulerMemoryBudgetEvictable:
    """Tests for SchedulerMemoryBudget with evictable_bytes_fn."""

    def test_over_hard_limit_not_too_tight_with_eviction(self):
        """Over raw hard limit but within effective hard → not 'too tight'."""
        budget = SchedulerMemoryBudget(
            hard_limit_bytes=1000,
            soft_limit_bytes=800,
            evictable_bytes_fn=lambda: 500,  # 500 bytes evictable
        )
        budget.byte_count = 1200  # Over hard (1000) but under effective (1500)
        budget.max_tokens = 10
        remaining_soft, remaining_hard, fits, msg = budget.budget_check()
        # fits=False because over soft limit, but message should NOT say "too tight"
        assert "too tight" not in msg
        # remaining_hard is positive (within effective hard)
        assert remaining_hard > 0

    def test_fits_when_under_both_limits_with_eviction(self):
        """Under both soft and effective hard limits → fits=True."""
        budget = SchedulerMemoryBudget(
            hard_limit_bytes=1000,
            soft_limit_bytes=800,
            evictable_bytes_fn=lambda: 500,
        )
        budget.byte_count = 700  # Under soft (800) and effective hard (1500)
        budget.max_tokens = 10
        _, _, fits, msg = budget.budget_check()
        assert fits is True

    def test_over_hard_limit_even_with_eviction(self):
        """Over hard limit even with evictable headroom → doesn't fit."""
        budget = SchedulerMemoryBudget(
            hard_limit_bytes=1000,
            soft_limit_bytes=800,
            evictable_bytes_fn=lambda: 100,
        )
        budget.byte_count = 1200  # Over effective (1100)
        budget.max_tokens = 10
        remaining_soft, remaining_hard, fits, msg = budget.budget_check()
        assert fits is False

    def test_no_evictable_provider_original_behavior(self):
        """No evictable_bytes_fn → original hard limit behavior."""
        budget = SchedulerMemoryBudget(
            hard_limit_bytes=1000,
            soft_limit_bytes=800,
        )
        # Under both soft and hard → fits
        budget.byte_count = 700
        budget.max_tokens = 10
        _, _, fits, _ = budget.budget_check()
        assert fits is True

        # Over hard → doesn't fit (too tight)
        budget.byte_count = 1100
        _, _, fits, msg = budget.budget_check()
        assert fits is False
        assert "too tight" in msg

    def test_soft_limit_not_adjusted_by_evictable(self):
        """Soft limit is NOT widened by evictable bytes."""
        budget = SchedulerMemoryBudget(
            hard_limit_bytes=1000,
            soft_limit_bytes=800,
            evictable_bytes_fn=lambda: 500,
        )
        budget.byte_count = 900  # Over soft (800) but under hard (1000)
        budget.max_tokens = 10
        remaining_soft, remaining_hard, fits, msg = budget.budget_check()
        # Over soft limit → fits_soft is False, but fits_hard is True
        # The method returns fits=False when over soft but under hard
        # (because the existing code returns fits based on fits_soft in this branch)
        assert remaining_soft < 0  # Over soft limit
