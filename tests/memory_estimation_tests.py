import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omlx.cache.type_handlers import (
    KVCacheHandler,
    RotatingKVCacheHandler,
    ArraysCacheHandler,
)


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
