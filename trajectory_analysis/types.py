"""Data structures for trajectory analysis."""

from dataclasses import dataclass, field


@dataclass
class LoopInfo:
    kind: str  # "consecutive_repeat", "consecutive_group", "non_consecutive_group"
    actions: list[str]
    start_step: int
    repeat_count: int
    group_size: int
    # For non-consecutive: list of start positions
    occurrences: list[int] = field(default_factory=list)


@dataclass
class TaskAnalysis:
    task_name: str
    task_dir: str
    pipeline: str  # "oracle" or "agent"
    n_steps: int
    reward: float
    outcome: str  # "success", "false_positive", "max_steps_exhausted"
    loops: list[dict] = field(default_factory=list)
    has_loops: bool = False
    # Number of steps consumed by non-overlapping loops (largest-first greedy)
    looped_steps: int = 0
    looped_step_ratio: float = 0.0
    # No-effect steps: actions where the selected candidate's effect indicates nothing happened
    no_effect_steps: list[dict] = field(default_factory=list)
    no_effect_count: int = 0
