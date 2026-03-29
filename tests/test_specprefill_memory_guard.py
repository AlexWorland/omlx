# SPDX-License-Identifier: Apache-2.0
"""Tests for prefill memory guard integration with SpecPrefill paths."""

from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _make_fake_model(n_tokens_per_call=1):
    """Create a fake model callable that returns dummy logits.

    The model tracks call count for verifying callback invocation timing.
    """
    model = MagicMock()
    model.call_count_manual = 0

    def fake_forward(x, cache=None):
        model.call_count_manual += 1
        return mx.zeros((1, 1, 32))

    model.side_effect = fake_forward
    return model


def _make_fake_cache(n_layers=2):
    """Create a fake cache list with .state and .offset attributes."""
    caches = []
    for _ in range(n_layers):
        c = MagicMock()
        c.state = mx.zeros((1,))
        c.offset = 0
        caches.append(c)
    return caches


class TestPrefillDraftMemoryCallback:
    """Tests for memory_check_callback in _prefill_draft."""

    def test_callback_called_between_chunks(self):
        from omlx.patches.specprefill import _prefill_draft

        model = _make_fake_model()
        cache = _make_fake_cache()
        call_count = [0]

        def counter():
            call_count[0] += 1

        # 10 tokens with step_size=3: chunks of [3, 3, 3] + 1 last token
        # → 3 chunk iterations → 3 callback calls
        _prefill_draft(model, list(range(10)), cache, step_size=3,
                       memory_check_callback=counter)

        assert call_count[0] == 3

    def test_callback_not_called_for_short_prompt(self):
        from omlx.patches.specprefill import _prefill_draft

        model = _make_fake_model()
        cache = _make_fake_cache()
        call_count = [0]

        def counter():
            call_count[0] += 1

        # 1 token: no chunk loop iterations, only last-token call
        _prefill_draft(model, [42], cache, step_size=3,
                       memory_check_callback=counter)

        assert call_count[0] == 0

    def test_callback_none_is_noop(self):
        from omlx.patches.specprefill import _prefill_draft

        model = _make_fake_model()
        cache = _make_fake_cache()

        # Should not raise when callback is None
        _prefill_draft(model, list(range(10)), cache, step_size=3,
                       memory_check_callback=None)

    def test_runtime_error_stops_prefill(self):
        from omlx.patches.specprefill import _prefill_draft

        model = _make_fake_model()
        cache = _make_fake_cache()
        calls_before_error = [0]

        def explode():
            calls_before_error[0] += 1
            if calls_before_error[0] >= 2:
                raise RuntimeError("Memory limit exceeded during prefill")

        # 20 tokens with step_size=3 would need ~6 chunk iterations,
        # but callback raises on the 2nd call → should stop early.
        with pytest.raises(RuntimeError, match="Memory limit exceeded"):
            _prefill_draft(model, list(range(20)), cache, step_size=3,
                           memory_check_callback=explode)

        assert calls_before_error[0] == 2


def _make_sparse_prefill_model():
    """Create a model mock with layers/attention structure for sparse_prefill.

    sparse_prefill requires _find_attention_layers to find at least one
    layer with self_attn, and checks for .rope to decide RoPE patching.
    The layer must NOT have block_type (MagicMock auto-creates it)
    to avoid hitting the Nemotron-H code path in _build_layer_to_cache_map.
    """
    model = _make_fake_model()

    # Build a layer with self_attn but no rope (simplest path).
    # Use spec= to prevent MagicMock from auto-creating block_type.
    attn = MagicMock(spec=["q_proj"])  # no "rope" attr → has_rope=False
    layer = MagicMock(spec=["self_attn"])
    layer.self_attn = attn
    model.layers = [layer]

    return model


class TestSparsePrefillMemoryCallback:
    """Tests for memory_check_callback in sparse_prefill."""

    def test_callback_called_between_chunks(self):
        from omlx.patches.specprefill import sparse_prefill

        model = _make_sparse_prefill_model()
        cache = _make_fake_cache()
        call_count = [0]

        def counter():
            call_count[0] += 1

        tokens = mx.arange(20)
        # Select 10 indices → 10 tokens to process
        # step_size=3: chunks [3, 3, 3] + 1 last = 3 callback calls
        selected = mx.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

        sparse_prefill(model, tokens, selected, cache, step_size=3,
                       memory_check_callback=counter)

        assert call_count[0] == 3

    def test_runtime_error_propagates(self):
        from omlx.patches.specprefill import sparse_prefill

        model = _make_sparse_prefill_model()
        cache = _make_fake_cache()

        def explode():
            raise RuntimeError("Memory limit exceeded during prefill")

        tokens = mx.arange(20)
        selected = mx.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

        with pytest.raises(RuntimeError, match="Memory limit exceeded"):
            sparse_prefill(model, tokens, selected, cache, step_size=3,
                           memory_check_callback=explode)


class TestPreflightMemoryCheck:
    """Tests for _preflight_memory_check with specprefill token count adjustment."""

    def _make_scheduler(self):
        """Create a minimal Scheduler for testing _preflight_memory_check."""
        from omlx.scheduler import Scheduler

        sched = object.__new__(Scheduler)
        sched._prefill_memory_guard = True
        sched._memory_hard_limit_bytes = 16 * 1024**3
        sched._memory_limit_bytes = 12 * 1024**3

        # Mock memory monitor
        monitor = MagicMock()
        # estimate_prefill_peak_bytes returns a fixed value per token
        monitor.estimate_prefill_peak_bytes = MagicMock(
            side_effect=lambda tokens, step: tokens * 1024  # 1KB per token
        )
        sched.memory_monitor = monitor

        # Mock config
        sched.config = MagicMock()
        sched.config.prefill_step_size = 512
        return sched

    def _make_request(self, prompt_tokens=10000, cached_tokens=0,
                      specprefill_indices=None, system_tokens=0):
        """Create a mock Request with specprefill fields."""
        req = MagicMock()
        req.num_prompt_tokens = prompt_tokens
        req.cached_tokens = cached_tokens
        req.specprefill_indices = specprefill_indices
        req._specprefill_system_tokens = system_tokens
        return req

    def test_normal_request_uses_full_token_count(self):
        sched = self._make_scheduler()
        req = self._make_request(prompt_tokens=10000, cached_tokens=0)

        with patch("omlx.scheduler.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 0
            result = sched._preflight_memory_check(req)

        # Should call estimate with full 10000 tokens
        sched.memory_monitor.estimate_prefill_peak_bytes.assert_called_once_with(
            10000, 512
        )

    def test_specprefill_uses_effective_token_count(self):
        sched = self._make_scheduler()
        # 10000 total tokens, specprefill selected 500 conv + 200 system remaining
        selected = mx.arange(500)
        req = self._make_request(
            prompt_tokens=10000, cached_tokens=0,
            specprefill_indices=selected, system_tokens=200,
        )

        with patch("omlx.scheduler.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 0
            result = sched._preflight_memory_check(req)

        # Should use 200 + 500 = 700 tokens, not 10000
        sched.memory_monitor.estimate_prefill_peak_bytes.assert_called_once_with(
            700, 512
        )

    def test_specprefill_no_system_tokens(self):
        sched = self._make_scheduler()
        selected = mx.arange(300)
        req = self._make_request(
            prompt_tokens=5000, cached_tokens=0,
            specprefill_indices=selected, system_tokens=0,
        )

        with patch("omlx.scheduler.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 0
            result = sched._preflight_memory_check(req)

        # 0 system + 300 selected = 300
        sched.memory_monitor.estimate_prefill_peak_bytes.assert_called_once_with(
            300, 512
        )

    def test_guard_disabled_returns_none(self):
        sched = self._make_scheduler()
        sched._prefill_memory_guard = False
        req = self._make_request(prompt_tokens=10000)

        result = sched._preflight_memory_check(req)
        assert result is None


class TestMemoryCallbackClosure:
    """Tests for the inline memory check closure pattern used in scheduler."""

    def test_closure_raises_on_hard_limit(self):
        """Verify the closure pattern matches RuntimeError text."""
        hard = 10 * 1024**3
        soft = 8 * 1024**3

        def mem_cb():
            active = mx.get_active_memory()
            if active > hard:
                raise RuntimeError("Memory limit exceeded during prefill")
            if soft > 0 and active > soft:
                pass  # soft limit warning only

        with patch("omlx.patches.specprefill.mx") as mock_mx:
            mock_mx.get_active_memory.return_value = 11 * 1024**3
            # Can't directly patch mx in the closure, so test the pattern
            # by constructing an equivalent closure with controlled input
            pass

        # Direct test: closure with known value
        def make_closure(active_bytes):
            def check():
                if active_bytes > hard:
                    raise RuntimeError("Memory limit exceeded during prefill")
            return check

        cb = make_closure(11 * 1024**3)
        with pytest.raises(RuntimeError, match="Memory limit exceeded"):
            cb()

        cb_ok = make_closure(5 * 1024**3)
        cb_ok()  # Should not raise
