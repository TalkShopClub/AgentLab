"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging

from agentlab.agents.generic_agent import (
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
    AGENT_o3_MINI,
    AGENT_37_SONNET,
    AGENT_CLAUDE_SONNET_35,
    AGENT_GPT5_MINI,
    AGENT_GEMINI3,
    AGENT_GPT5,
    GenericAgentArgs
)
from agentlab.agents.visual_agent import (
    VISUAL_AGENT_GPT5,
    VISUAL_AGENT_QWEN3_VL_30B_A3B_INSTRUCT,
)
from agentlab.agents.wm_visual_agent.agent_configs import (
    WM_VISUAL_AGENT_GPT5_IMAGE,
    WM_VISUAL_AGENT_GPT5_TEXT,
)
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents import dynamic_prompting as dp
from agentlab.benchmarks.custom_workarena import workarena_l3_single_seed, workarena_l2_single_seed 

logging.getLogger().setLevel(logging.INFO)

# choose your agent or provide a new agent
# agent_args = [AGENT_GEMINI3]
# agent_args = [VISUAL_AGENT_GPT5]
agent_args = [AGENT_GPT5]
# agent_args = [WM_VISUAL_AGENT_GPT5_TEXT]   # World model augmented (text predictions)
# agent_args = [WM_VISUAL_AGENT_GPT5_IMAGE]  # World model augmented (image predictions)


# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
# benchmark = "workarena_l1"
# benchmark = "workarena_l2_agent_curriculum_eval"
# benchmark = "workarena_l3_agent_curriculum_eval"  # Default (with all perturbations)
# benchmark = "webarena"

# Run on 5 random tasks
# benchmark = workarena_l3_single_seed()  # Custom: only 1 seed per task type
benchmark = workarena_l3_single_seed()  # Custom: only 1 seed per task type

# Deterministic task selection and execution.
# TASK_SEED controls both which task is picked and the task content (hashtags, users, etc.).
# Set to None to use the curriculum-assigned seeds without overriding.
TASK_SEED = 20000

# ── Skip already-finished tasks ──
# Set to a study directory path to skip tasks that already have summary_info.json
# SKIP_COMPLETED_FROM = "results_emu_fixed/2026-03-10_22-42-12_genericagent-gpt-5-2025-08-07-on-workarena-l3-single-seed"
SKIP_COMPLETED_FROM = None

# ── Task source ──
# Option A: Load ordered task list from sampled_tasks.txt (written by run_parallel.py)
TASK_FILE = "oracle_wm/parallel_logs/sampled_tasks.txt"
# TASK_FILE = None

import random
from pathlib import Path

if TASK_FILE and Path(TASK_FILE).is_file():
    # Parse ordered task names and seeds from the file
    ordered_tasks = []  # list of (task_name, seed)
    for line in Path(TASK_FILE).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        task_name = parts[0]
        seed = TASK_SEED
        ordered_tasks.append((task_name, seed))
    # Build env_args_list directly from file, creating EnvArgs for each task
    from agentlab.experiments.loop import EnvArgs
    benchmark.env_args_list = []
    for task_name, seed in ordered_tasks:
        benchmark.env_args_list.append(EnvArgs(task_name=task_name, task_seed=seed, max_steps=40))
    print(f"Loaded {len(benchmark.env_args_list)} tasks from {TASK_FILE} (ordered, with per-task seeds)")
else:
    # Option B: Random sampling (or specific-task override)
    SPECIFIC_TASKS = [
        "workarena.servicenow.filter-single-item-expenses-and-delete-wrong-investments-medium-l2",
        "workarena.servicenow.dashboard-retrieve-incident-and-min-filter-asset-list-l2",
    ]
    # SPECIFIC_TASKS = None
    if SPECIFIC_TASKS:
        task_set = set(SPECIFIC_TASKS)
        benchmark.env_args_list = [e for e in benchmark.env_args_list if e.task_name in task_set]
        for env_args in benchmark.env_args_list:
            env_args.task_seed = TASK_SEED
    else:
        rng = random.Random(TASK_SEED)
        benchmark.env_args_list = rng.sample(benchmark.env_args_list, 3)
        if TASK_SEED is not None:
            for env_args in benchmark.env_args_list:
                env_args.task_seed = TASK_SEED

# ── Filter out completed tasks ──
if SKIP_COMPLETED_FROM:
    skip_dir = Path(SKIP_COMPLETED_FROM)
    completed_tasks = set()
    if skip_dir.is_dir():
        for d in skip_dir.iterdir():
            if d.is_dir() and (d / "summary_info.json").exists():
                # Extract task name: ..._on_<task_name>_<seed>
                parts = d.name.split("_on_")
                if len(parts) >= 2:
                    task_and_seed = parts[-1]
                    task_name = "_".join(task_and_seed.rsplit("_", 1)[:-1])  # drop trailing seed
                    completed_tasks.add(task_name)
    before = len(benchmark.env_args_list)
    benchmark.env_args_list = [e for e in benchmark.env_args_list if e.task_name not in completed_tasks]
    print(f"Skipping {before - len(benchmark.env_args_list)} completed tasks, {len(benchmark.env_args_list)} remaining")

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
n_jobs = 2  # Sequential execution for testing


if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.WARNING, save_som=False)

    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        strict_reproducibility=reproducibility_mode,
        n_relaunch=1,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
