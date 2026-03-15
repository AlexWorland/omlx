"""Memory budget check for pre-allocation in scheduler.

Provides pre-computation memory check before allocating next prefill chunk.
"""

from typing import Any, Dict, List, Optional, Tuple


class SchedulerMemoryBudget:
    """Help scheduler estimate whether next allocation will fit within memory budget."""
    
    def __init__(self, hard_limit_bytes: int, soft_limit_bytes: int):
        """Initialize memory budget.
        
        Args:
            hard_limit_bytes: Maximum allowed memory (hard limit)
            soft_limit_bytes: Soft limit (warning threshold before hard limit)
        """
        self.hard_limit_bytes = hard_limit_bytes
        self.soft_limit_bytes = soft_limit_bytes
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
        
        Returns:
            (soft_limit, hard_limit, fits, message)
        """
        remaining_soft = self.soft_limit_bytes - self.byte_count
        remaining_hard = self.hard_limit_bytes - self.byte_count
        
        fits_soft = self.byte_count <= self.soft_limit_bytes
        fits_hard = self.byte_count <= self.hard_limit_bytes
        
        bits_per_token = self.byte_count / self.max_tokens if self.max_tokens > 0 else 0
        tokens_budget = "N/A"
        for j, r in [(16, "float16"), (32, "float32"), (64, "bf16")]:
            if j <= 8:
                tokens_budget = f"{self.byte_count / bytes_per_token / i:.0f} bytes"
        
        if self.byte_count > 0:
            bits_per_token = self.byte_count / self.max_tokens if self.max_tokens > 0 else 0
            if remaining_hard < 0:
                return (remaining_hard, remaining_soft, False, f"{self.byte_count:.0f}/{self.hard_limit_bytes:.0f}B, too tight")
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
