"""
Custom WorkArena benchmark configurations.

This module provides custom benchmark configurations for WorkArena L3
that keep only 1 seed per unique task type instead of multiple perturbations.
"""

import numpy as np
from browsergym.experiments.benchmark.base import Benchmark, HighLevelActionSetArgs
from browsergym.experiments.benchmark.metadata.utils import task_metadata
from browsergym.experiments.loop import EnvArgs
from collections import defaultdict


def make_env_args_list_from_workarena_curriculum_single_seed(
    level: str,
    task_category_filter: str,
    meta_seed: int,
    max_steps: int,
    curriculum_type: str = "agent",
    seeds_l1: int = 10,
):
    """
    Returns a WorkArena predefined task curriculum with only 1 seed per unique task type.

    This is a modified version of browsergym's make_env_args_list_from_workarena_curriculum
    that filters out duplicate task types, keeping only the first seed for each unique task.

    Args:
        level: Task level ("l1", "l2", or "l3")
        task_category_filter: Filter for specific task categories
        meta_seed: Random seed for reproducibility
        max_steps: Maximum steps per task
        curriculum_type: "human" or "agent" curriculum
        seeds_l1: Number of seeds for L1 tasks

    Returns:
        List of EnvArgs with 1 entry per unique task type
    """
    assert level in ("l1", "l2", "l3")
    assert curriculum_type in ("human", "agent")

    from browsergym.workarena import get_all_tasks_agents

    all_task_tuples = get_all_tasks_agents(
        filter=f"{level}.{task_category_filter}" if task_category_filter else level,
        meta_seed=meta_seed,
        is_agent_curriculum=(curriculum_type == "agent"),
        n_seed_l1=seeds_l1,
    )

    # Group by task name and keep only the first seed for each unique task
    seen_tasks = {}
    for task, seed in all_task_tuples:
        task_name = task.get_task_id()
        if task_name not in seen_tasks:
            seen_tasks[task_name] = seed

    # Create env_args_list with unique tasks only
    env_args_list = []
    for task_name, seed in seen_tasks.items():
        env_args_list.append(EnvArgs(task_name=task_name, task_seed=seed, max_steps=max_steps))

    return env_args_list


# Default high-level action set for WorkArena
WORKARENA_ACTION_SET = HighLevelActionSetArgs(
    subsets=["workarena++"],
    multiaction=False,
    strict=False,
    retry_with_force=True,
    demo_mode="off",
)


def workarena_l3_single_seed(n_repeats=1):
    """
    WorkArena L3 benchmark with only 1 seed per unique task type.

    Args:
        n_repeats: Ignored (kept for API compatibility)

    Returns:
        Benchmark object with unique tasks only
    """
    return Benchmark(
        name="workarena_l3_single_seed",
        high_level_action_set_args=WORKARENA_ACTION_SET,
        is_multi_tab=True,
        supports_parallel_seeds=True,
        backends=["workarena"],
        env_args_list=make_env_args_list_from_workarena_curriculum_single_seed(
            level="l3",
            task_category_filter=None,
            meta_seed=42,
            max_steps=50,
            curriculum_type="agent",
        ),
        task_metadata=task_metadata("workarena"),
    )


def workarena_l2_single_seed(n_repeats=1):
    """
    WorkArena L2 benchmark with only 1 seed per unique task type.

    Args:
        n_repeats: Ignored (kept for API compatibility)

    Returns:
        Benchmark object with unique tasks only
    """
    return Benchmark(
        name="workarena_l2_single_seed",
        high_level_action_set_args=WORKARENA_ACTION_SET,
        is_multi_tab=True,
        supports_parallel_seeds=True,
        backends=["workarena"],
        env_args_list=make_env_args_list_from_workarena_curriculum_single_seed(
            level="l2",
            task_category_filter=None,
            meta_seed=42,
            max_steps=50,
            curriculum_type="agent",
        ),
        task_metadata=task_metadata("workarena"),
    )


def workarena_l1_single_seed(n_repeats=1):
    """
    WorkArena L1 benchmark with only 1 seed per unique task type.

    Args:
        n_repeats: Ignored (kept for API compatibility)

    Returns:
        Benchmark object with unique tasks only
    """
    return Benchmark(
        name="workarena_l1_single_seed",
        high_level_action_set_args=HighLevelActionSetArgs(
            subsets=["workarena"],
            multiaction=False,
            strict=False,
            retry_with_force=True,
            demo_mode="off",
        ),
        is_multi_tab=False,
        supports_parallel_seeds=True,
        backends=["workarena"],
        env_args_list=make_env_args_list_from_workarena_curriculum_single_seed(
            level="l1",
            task_category_filter=None,
            meta_seed=42,
            max_steps=15,
            curriculum_type="agent",
            seeds_l1=1,  # Only 1 seed for L1 tasks
        ),
        task_metadata=task_metadata("workarena"),
    )
