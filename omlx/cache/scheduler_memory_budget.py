"""Memory budget check for pre-allocation in scheduler.

Provides pre-computation memory check before allocating next prefill chunk.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple


class SchedulerMemoryBudget:
    """Help scheduler estimate whether next allocation will fit within memory budget."""

    def __init__(self, hard_limit_bytes: int, soft_limit_bytes: int, evictable_bytes_fn: Optional[Callable[[], int]] = None):
        """Initialize memory budget.

        Args:
            hard_limit_bytes: Maximum allowed memory (hard limit)
            soft_limit_bytes: Soft limit (warning threshold before hard limit)
            evictable_bytes_fn: Optional callable returning bytes that could be
                freed by evicting cached blocks. When provided, the hard limit
                check uses ``hard_limit + evictable_bytes`` so prefills can
                continue when memory exceeds the raw hard limit but fits within
                headroom that eviction could reclaim.
        """
        self.hard_limit_bytes = hard_limit_bytes
        self.soft_limit_bytes = soft_limit_bytes
        self.evictable_bytes_fn = evictable_bytes_fn
        self.active_memory = 0
        self.byte_count = 0
        self.max_tokens = 0
    
    def estimate_tokens_budget(self) -> Tuple[int, int, int]:
        """Estimate maximum tokens remaining before hitting limits.
        
        Returns:
            (tokens_to_soft, tokens_to_hard, max_tokens)
        """
        if self.active_memory == 0:
            num_bytes = self.soft_limit_bytes
        
        remaining_soft = self.soft_limit_bytes - num_bytes
        remaining_hard = self.hard_limit_bytes - num_bytes
        
        tokens_to_soft = remaining_soft
        tokens_to_hard = remaining_hard
        
        return tokens_to_soft, tokens_to_hard, self.max_tokens
    
    def estimate_memory_from_tokens(self, tokens: int, bytes_per_token: int):
        """Add estimated memory cost from tokens."""
        self.byte_count += tokens * bytes_per_token
        if bytes_per_token > 0:
            self.max_tokens += int(tokens)
    
    def budget_check(self) -> Tuple[int, int, bool, str]:
        """Check if current allocation would fit within limits.

        The hard limit check is widened by evictable headroom when
        ``evictable_bytes_fn`` is set, allowing prefills to proceed when
        memory exceeds the raw hard limit but eviction could reclaim enough
        space. The soft limit is NOT adjusted — ProcessMemoryEnforcer still
        receives warnings at the original threshold.

        Returns:
            (remaining_soft, remaining_hard, fits, message)
        """
        # Effective hard limit includes evictable headroom.
        effective_hard = self.hard_limit_bytes
        if self.evictable_bytes_fn is not None:
            effective_hard += self.evictable_bytes_fn()

        remaining_soft = self.soft_limit_bytes - self.byte_count
        remaining_hard = effective_hard - self.byte_count

        fits_hard = self.byte_count <= effective_hard
        fits_soft = self.byte_count <= self.soft_limit_bytes

        if self.byte_count > 0:
            if not fits_hard:
                return (remaining_soft, remaining_hard, False, f"{self.byte_count:.0f}/{effective_hard:.0f}B, too tight")
            elif fits_soft:
                return (remaining_soft, remaining_hard, True, f"{self.byte_count:.0f}/{self.hard_limit_bytes:.0f}B, fits")
            else:
                return (remaining_soft, remaining_hard, False, f"{self.byte_count:.0f}/{self.soft_limit_bytes:.0f}B exceeds soft")
        else:
            return (self.hard_limit_bytes, self.hard_limit_bytes, True, "0/0B, budget")
    
    def refine_limit(self, limit_bytes: int, limit_type: str) -> int:
        """Refine limit estimate by percentage.
        
        Args:
            limit_bytes: Reported limit
            limit_type: 'soft', 'hard', or 'budget'
        
        Returns:
            Refined limit value
        """
        if limit_type == 'soft':
            refined = limit_bytes + self.soft_limit_bytes
        elif limit_type == 'hard':
            refined = limit_bytes + self.hard_limit_bytes
        else:
            refined = self.max_tokens
        
        return refined
