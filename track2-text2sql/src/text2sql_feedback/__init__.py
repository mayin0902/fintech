"""Text-to-SQL execution feedback and post-training building blocks."""

from .contracts import ExecutionOutcome, ExpectedResult, QueryResult
from .evaluation import Comparison, compare_results
from .reward import ExecutionReward, RewardBreakdown

__all__ = [
    "Comparison",
    "ExecutionOutcome",
    "ExecutionReward",
    "ExpectedResult",
    "QueryResult",
    "RewardBreakdown",
    "compare_results",
]
