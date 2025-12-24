#!/usr/bin/env python3
"""
Quick verification script to compare task counts between default and custom benchmarks.
"""

from agentlab.benchmarks.custom_workarena import workarena_l3_single_seed
import browsergym.experiments.benchmark as bgym_bench

# Load the default benchmark
default_benchmark = bgym_bench.configs.DEFAULT_BENCHMARKS["workarena_l3_agent_curriculum_eval"]()

# Load the custom benchmark
custom_benchmark = workarena_l3_single_seed()

print(f"Default WorkArena L3 benchmark:")
print(f"  - Total tasks: {len(default_benchmark.env_args_list)}")

print(f"\nCustom WorkArena L3 (single seed) benchmark:")
print(f"  - Total tasks: {len(custom_benchmark.env_args_list)}")

print(f"\nReduction: {len(default_benchmark.env_args_list) - len(custom_benchmark.env_args_list)} tasks removed")
print(f"Reduction percentage: {(1 - len(custom_benchmark.env_args_list) / len(default_benchmark.env_args_list)) * 100:.1f}%")

# Show sample tasks
print(f"\nSample custom benchmark tasks (first 10):")
for i, env_args in enumerate(custom_benchmark.env_args_list[:10]):
    print(f"  {i+1}. {env_args.task_name} (seed: {env_args.task_seed})")
