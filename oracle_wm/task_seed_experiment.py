#!/usr/bin/env python3
"""
Experiment: verify that task randomness is stable across multiple initializations
given the same seed. WorkArena's source has been patched so that all randomness
(uuid4, Faker, random module, id(self)-based hashtags) is seeded from task.seed.

Each run creates a FRESH env (not reset on the same env) to avoid the double-teardown
issue: env.reset() tears down the old task (including deleting its ServiceNow resources),
then if env.close() is also called on the same task it gets 404 on already-deleted records.

Usage:
    python oracle_wm/task_seed_experiment.py \
        --task workarena.servicenow.dashboard-retrieve-catalog-and-max-order-apple-watch-l2 \
        --seed 42 --n 3
"""

import argparse
import re
from pathlib import Path

from browsergym.experiments.loop import EnvArgs
from browsergym.workarena.api.utils import table_api_call

import browsergym.workarena  # noqa: F401

from oracle_wm._bid_utils import _ACTION_SET, get_valid_snow_instance


def _verify_snow_data(instance, task, hashtag: str) -> dict:
    """
    Query ServiceNow to confirm that the expected records were actually created.
    Returns a dict of {table: found_count} for each relevant table.
    """
    results = {}

    # sys_report: dashboard catalog/incident tasks create a report with the hashtag as title
    if hasattr(task, "report_sys_id"):
        resp = table_api_call(
            instance=instance,
            table="sys_report",
            params={
                "sysparm_query": f"sys_id={task.report_sys_id}",
                "sysparm_fields": "sys_id,title",
                "sysparm_limit": "1",
            },
        )
        records = resp.get("result", [])
        results["sys_report(by_sys_id)"] = len(records)
        if records:
            results["sys_report_title"] = records[0].get("title", "")

    # Also search by hashtag title directly (independent of stored sys_id)
    if hashtag and hashtag != "(not found)":
        resp = table_api_call(
            instance=instance,
            table="sys_report",
            params={
                "sysparm_query": f"titleCONTAINS{hashtag}",
                "sysparm_fields": "sys_id,title",
                "sysparm_limit": "5",
            },
        )
        results["sys_report(by_title)"] = len(resp.get("result", []))

        resp = table_api_call(
            instance=instance,
            table="problem",
            params={
                "sysparm_query": f"short_descriptionCONTAINS{hashtag}",
                "sysparm_fields": "sys_id,number,short_description",
                "sysparm_limit": "10",
            },
        )
        results["problem(by_hashtag)"] = len(resp.get("result", []))

    return results


def _print_task_data(task) -> None:
    """Print all seeded/random attributes of a task for stability verification."""
    attrs = [
        "_base_user_name", "_base_user_sysid", "unique_id",
        "catalog_hashtag", "incident_hashtag", "report_sys_id",
        "private_task_id",
    ]
    for attr in attrs:
        val = getattr(task, attr, None)
        if val is not None:
            print(f"    {attr}: {val}")

    # Subtask agent sysids (incident/catalog tasks assign agents dynamically)
    for subtask_attr in ("subtasks", "_subtasks", "tasks"):
        subtasks = getattr(task, subtask_attr, None)
        if subtasks:
            for i, st in enumerate(subtasks):
                for agent_attr in ("agent_sysid", "_agent_sysid", "assigned_to_sysid"):
                    agent = getattr(st, agent_attr, None)
                    if agent:
                        print(f"    subtask[{i}].{agent_attr}: {agent}")

    # Report config (contains agent assignments and incident numbers)
    cfg = getattr(task, "service_catalog_report_config", None) or getattr(task, "report_config", None)
    if cfg:
        print(f"    report_config: {cfg}")


def run_experiment(task_name: str, seed: int, n_runs: int) -> None:
    instance = get_valid_snow_instance()

    env_args = EnvArgs(
        task_name=task_name,
        task_seed=seed,
        max_steps=1,
    )

    results = []
    for run in range(n_runs):
        # Fresh env each run — avoids double-teardown 404 that happens when
        # env.reset() tears down the old task then env.close() tries again.
        env = env_args.make_env(
            action_mapping=_ACTION_SET.to_python_code,
            exp_dir=Path("."),
            exp_task_kwargs={"instance": instance},
            use_raw_page_output=False,
        )
        task = None
        hashtag = "(not found)"
        try:
            obs, _ = env.reset(seed=seed)
            goal = obs.get("goal", "")

            hashtag_match = re.search(r"#(?:SERIES-[a-f0-9-]+|[A-Z]+\d+)", goal)
            hashtag = hashtag_match.group(0) if hashtag_match else "(not found)"

            task = env.unwrapped.task
            snow_check = _verify_snow_data(instance, task, hashtag)

            print(f"\n--- Run {run + 1}/{n_runs} ---")
            print(f"  hashtag    : {hashtag}")
            _print_task_data(task)
            print(f"  snow_check : {snow_check}")
            print(f"  Goal: {goal}")
            task_data = {
                attr: getattr(task, attr, None)
                for attr in ("_base_user_name", "_base_user_sysid", "unique_id",
                             "catalog_hashtag", "incident_hashtag", "report_sys_id")
                if getattr(task, attr, None) is not None
            }
            results.append({"hashtag": hashtag, "goal": goal, "snow_check": snow_check, "task_data": task_data})
        finally:
            try:
                env.close()
            except Exception as e:
                print(f"  [env.close() error: {e}]")
            # Explicit base-user cleanup: DashboardRetrieveCatalogAndDoTask.teardown() deletes
            # the report first, then calls super().teardown() which iterates subtasks. The
            # catalog subtask's _wait_for_ready() times out, raising before
            # AbstractServiceNowTask.teardown() (which deletes _base_user_sysid) can run.
            # Without this, a seeded run leaves the user behind and the next run with the
            # same Faker seed conflicts on the identical username.
            _env_task = env.unwrapped.task if hasattr(env, "unwrapped") else None
            if _env_task is not None and getattr(_env_task, "_base_user_sysid", None):
                try:
                    table_api_call(
                        instance=instance,
                        table=f"sys_user/{_env_task._base_user_sysid}",
                        method="DELETE",
                    )
                except Exception:
                    pass  # already deleted by teardown
            if task is not None:
                post_check = _verify_snow_data(instance, task, hashtag)
                print(f"  post-close : {post_check}")

    print(f"\n{'='*60}")
    hashtags = [r["hashtag"] for r in results]
    print(f"Hashtags stable : {len(set(hashtags)) == 1}  {hashtags}")
    print(f"Goals stable    : {len({r['goal'] for r in results}) == 1}")
    all_task_data_keys = {k for r in results for k in r["task_data"]}
    for k in sorted(all_task_data_keys):
        vals = [r["task_data"].get(k) for r in results]
        stable = len(set(str(v) for v in vals)) == 1
        print(f"  {k} stable={stable}: {vals[0]}")
    for r in results:
        print(f"  snow_check: {r['snow_check']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()
    run_experiment(args.task, args.seed, args.n)


if __name__ == "__main__":
    main()
