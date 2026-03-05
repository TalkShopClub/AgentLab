"""Run oracle tasks in parallel — one per ServiceNow instance in the pool.

Uses a worker-pool model: tasks are queued and each grabs an instance as soon
as one becomes free. No batch boundaries — fast tasks free up instances for the
next pending task immediately.

Each subprocess gets its instance pinned via SNOW_INSTANCE_URL/UNAME/PWD env vars,
so SNowInstance() inside the oracle pipeline deterministically uses that instance.

Task sources:
  --task-dir : Read unique task names from folder names in the given directory.
               Filters out infeasible tasks. Runs ALL matching tasks.
               Seeds are enumerated (task 0 -> seed 0, etc.)
               unless overridden by --task-seed.
  (default)  : Random sampling from feasible tasks with family deduplication.
"""

import argparse
import gzip
import json
import os
import pickle
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _task_family(task_id: str) -> str:
    """Reduce a task ID to its family key for deduplication."""
    name = task_id.split(".")[-1]
    name = re.sub(r"-l[23]$", "", name)
    name = re.sub(r"-(small|medium|large)$", "", name)
    name = re.sub(r"-\w+-list$", "-list", name)
    return name


def _get_feasible_tasks(level: str = "l2") -> list[str]:
    from browsergym.workarena import ALL_WORKARENA_TASKS

    suffix = f"-{level}"
    return [
        t.get_task_id()
        for t in ALL_WORKARENA_TASKS
        if suffix in t.get_task_id().lower() and "infeasible" not in t.get_task_id().lower()
    ]


def _fetch_pool() -> list[dict]:
    from browsergym.workarena.instance import fetch_instances

    return fetch_instances()


def _sample_tasks(n: int, seed: int, level: str = "l2") -> list[str]:
    """Return task names via family-deduplicated random sampling."""
    all_tasks = _get_feasible_tasks(level)
    rng = random.Random(seed)

    families: dict[str, list[str]] = defaultdict(list)
    for t in all_tasks:
        families[_task_family(t)].append(t)

    family_keys = list(families.keys())
    rng.shuffle(family_keys)

    selected = []
    for key in family_keys:
        if len(selected) >= n:
            break
        selected.append(rng.choice(families[key]))

    return selected


def _load_tasks_from_dir(task_dir: str) -> list[str]:
    """Parse unique task names from folder names like ..._on_workarena.servicenow.TASK_SEED."""
    seen = set()
    tasks = []
    for entry in sorted(Path(task_dir).iterdir()):
        if not entry.is_dir():
            continue
        m = re.search(r"_on_(workarena\.servicenow\..+)_\d+$", entry.name)
        if not m:
            continue
        task_name = m.group(1)
        if "infeasible" in task_name:
            continue
        if task_name not in seen:
            seen.add(task_name)
            tasks.append(task_name)
    return tasks


def _check_completed(result_dir: Path) -> tuple[bool, float | None]:
    """Check if a task completed by reading the last step pkl. Returns (completed, reward)."""
    if not result_dir.is_dir():
        return False, None
    # Also check summary.json first (written by newer runs)
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        s = json.loads(summary_path.read_text())
        return True, s.get("reward")
    # Fallback: find the highest step pkl and check terminated/truncated
    step_files = sorted(result_dir.glob("step_*.pkl.gz"))
    if not step_files:
        return False, None
    try:
        with gzip.open(step_files[-1], "rb") as f:
            info = pickle.load(f)
        if info.terminated or info.truncated:
            return True, info.reward
    except Exception:
        pass
    return False, None


def _find_resume_point(result_dir: Path, run_dir: Path) -> int:
    """Find the step to resume from for an incomplete task.

    A step N is fully committed only when ALL of these exist:
      - result_dir/step_N.pkl.gz   (StepInfo saved after execution)
      - run_dir/step_N/committed_step.json  (resume data written before execution)
      - run_dir/step_N/selection.json       (selection metadata)

    Returns the first step that is NOT complete (0 means start fresh).
    """
    if not run_dir.is_dir():
        return 0
    step = 0
    while True:
        pkl_ok = (result_dir / f"step_{step}.pkl.gz").exists() if result_dir.is_dir() else False
        committed_ok = (run_dir / f"step_{step}" / "committed_step.json").exists()
        selection_ok = (run_dir / f"step_{step}" / "selection.json").exists()
        if pkl_ok and committed_ok and selection_ok:
            step += 1
        else:
            break
    return step


def _stream_output(proc, label: str, log_path: Path):
    with open(log_path, "w") as f:
        for line in proc.stdout:
            text = f"[{label}] {line}"
            sys.stdout.write(text)
            sys.stdout.flush()
            f.write(text)


def main():
    parser = argparse.ArgumentParser(description="Run oracle tasks in parallel (one per SNOW instance)")
    parser.add_argument("--task-dir", help="Directory with failed task folders to re-run (reads task names from folder names)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random sampling mode (default: 0)")
    parser.add_argument("--task-seed-offset", type=int, default=0, help="Task seeds are enumerated as offset+0, offset+1, ... (default: 0)")
    parser.add_argument("--level", choices=["l2", "l3"], default="l2",
                        help="Task level to run (default: l2)")
    parser.add_argument("--model", default="openai/gpt-5-2025-08-07")
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per task (default: 50 for L2, 25 for L3)")
    parser.add_argument("--save-dir", default="oracle_results")
    parser.add_argument("--run-dir", default="oracle_wm/runs")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--agent-mode", default="text", choices=["vision", "text"])
    parser.add_argument("--no-sel-effects", action="store_true")
    parser.add_argument("--no-sel-images", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Max parallel tasks per batch (default: use all instances)")
    parser.add_argument("--continue", dest="cont", action="store_true",
                        help="Resume from sampled_tasks.txt: skip completed tasks, resume incomplete ones from their last committed step")
    args = parser.parse_args()

    if args.max_steps is None:
        args.max_steps = 25 if args.level == "l3" else 50

    def _result_dir(task_name, seed):
        return Path(args.save_dir) / f"{task_name.replace('/', '_')}_seed{seed}"

    def _run_artifact_dir(task_name, seed):
        return Path(args.run_dir) / f"{task_name.replace('/', '_')}_seed{seed}"

    pool = _fetch_pool()
    if args.max_parallel and args.max_parallel < len(pool):
        pool = pool[:args.max_parallel]
    batch_size = len(pool)

    print(f"Instance pool has {batch_size} instances.\n")
    for i, entry in enumerate(pool):
        print(f"  Instance {i}: {entry['url']}")
    print()

    log_dir = Path("oracle_wm/parallel_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    task_list_path = log_dir / "sampled_tasks.txt"

    if args.cont:
        # ── Continue mode: read sampled_tasks.txt, skip completed, clean & re-run the rest ──
        if not task_list_path.is_file():
            print(f"ERROR: --continue requires {task_list_path} to exist (from a previous run)")
            sys.exit(1)

        all_entries = []
        for line in task_list_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            task_name = parts[0]
            seed = int(parts[1].split("=")[1]) if len(parts) > 1 and "seed=" in parts[1] else 0
            all_entries.append((task_name, seed))

        completed, pending = [], []
        for task_name, seed in all_entries:
            done, reward = _check_completed(_result_dir(task_name, seed))
            if done:
                completed.append((task_name, seed, reward))
            else:
                resume_step = _find_resume_point(_result_dir(task_name, seed), _run_artifact_dir(task_name, seed))
                # Clean up the partial step directory (the one that was in progress)
                partial_step = _run_artifact_dir(task_name, seed) / f"step_{resume_step}"
                if partial_step.exists():
                    shutil.rmtree(partial_step)
                partial_pkl = _result_dir(task_name, seed) / f"step_{resume_step}.pkl.gz"
                if partial_pkl.exists():
                    partial_pkl.unlink()
                pending.append((task_name, seed, resume_step))

        print(f"Continue: {len(completed)} done, {len(pending)} remaining out of {len(all_entries)}")
        for t, s, r in completed:
            r_str = f"{r:.3f}" if r is not None else "?"
            print(f"  DONE  {t}  seed={s}  reward={r_str}")
        for t, s, rs in pending:
            print(f"  RESUME  {t}  seed={s}  from_step={rs}")

        all_tasks = [t for t, _, _ in pending]
        task_seeds = [s for _, s, _ in pending]
        resume_steps = [rs for _, _, rs in pending]
    else:
        # ── Normal mode: load or sample tasks ──
        if args.task_dir:
            all_tasks = _load_tasks_from_dir(args.task_dir)
            random.Random(args.seed).shuffle(all_tasks)
            print(f"Loaded {len(all_tasks)} unique feasible tasks from {args.task_dir}/ (shuffled with seed={args.seed})")
        else:
            all_tasks = _sample_tasks(batch_size, args.seed, args.level)
            print(f"Sampled {len(all_tasks)} {args.level.upper()} tasks (sampling seed={args.seed})")

        task_seeds = [args.task_seed_offset + i for i in range(len(all_tasks))]
        resume_steps = [0] * len(all_tasks)

        with open(task_list_path, "w") as f:
            for t, s in zip(all_tasks, task_seeds):
                f.write(f"{t}  seed={s}\n")
        print(f"Task list written to: {task_list_path}\n")

    if not all_tasks:
        print("Nothing to run — all tasks completed.")
        sys.exit(0)

    print(f"Total: {len(all_tasks)} tasks  |  {batch_size} parallel workers  |  seeds {task_seeds[0]}..{task_seeds[-1]}\n")

    extra_args = [
        "--model", args.model,
        "--n-candidates", str(args.n_candidates),
        "--max-steps", str(args.max_steps),
        "--save-dir", args.save_dir,
        "--run-dir", args.run_dir,
    ]
    if args.headless:
        extra_args.append("--headless")
    if args.agent_mode != "vision":
        extra_args += ["--agent-mode", args.agent_mode]
    if args.no_sel_effects:
        extra_args.append("--no-sel-effects")
    if args.no_sel_images:
        extra_args.append("--no-sel-images")
    if args.cleanup:
        extra_args.append("--cleanup")

    python = sys.executable
    active_procs: list[subprocess.Popen] = []
    procs_lock = threading.Lock()
    interrupted = threading.Event()

    def _terminate_all():
        with procs_lock:
            for p in active_procs:
                if p.poll() is None:
                    p.terminate()

    def _sigint_handler(sig, frame):
        print("\nInterrupted — terminating all subprocesses...", flush=True)
        interrupted.set()
        _terminate_all()

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    def _run_task_tracked(global_idx: int, task: str, task_seed: int, instance_entry: dict, resume_from: int = 0) -> tuple[int, str, int]:
        label = f"{global_idx}:{task.split('.')[-1][:45]}"
        log_path = log_dir / f"task_{global_idx}.log"

        cmd = [python, "-m", "oracle_wm.oracle_pipeline.run_oracle", "--task", task, "--seed", str(task_seed)] + extra_args
        if resume_from > 0:
            cmd += ["--resume-from", str(resume_from), "--cleanup"]

        env = os.environ.copy()
        env["SNOW_INSTANCE_URL"] = instance_entry["url"]
        env["SNOW_INSTANCE_UNAME"] = "admin"
        env["SNOW_INSTANCE_PWD"] = instance_entry["password"]

        resume_msg = f"  resume_from={resume_from}" if resume_from > 0 else ""
        print(f"[{label}] Starting  seed={task_seed}  instance={instance_entry['url']}{resume_msg}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        with procs_lock:
            active_procs.append(proc)

        stream_thread = threading.Thread(target=_stream_output, args=(proc, label, log_path), daemon=True)
        stream_thread.start()
        proc.wait()
        stream_thread.join()

        with procs_lock:
            active_procs.remove(proc)

        return global_idx, task, proc.returncode

    # ── Worker pool: instances are queued, each task grabs one when free ──
    instance_queue = queue.Queue()
    for entry in pool:
        instance_queue.put(entry)

    def _run_with_pool(global_idx, task, seed, resume_from):
        instance = instance_queue.get()
        try:
            return _run_task_tracked(global_idx, task, seed, instance, resume_from)
        finally:
            instance_queue.put(instance)

    all_results = []
    t_total_start = time.time()

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(_run_with_pool, i, task, seed, rs): (i, task)
            for i, (task, seed, rs) in enumerate(zip(all_tasks, task_seeds, resume_steps))
        }
        for future in as_completed(futures):
            if interrupted.is_set():
                break
            idx, task, rc = future.result()
            all_results.append((idx, task, rc))
            reward_str = ""
            summary_path = _result_dir(task, task_seeds[idx]) / "summary.json"
            if summary_path.is_file():
                r = json.loads(summary_path.read_text()).get("reward")
                if r is not None:
                    reward_str = f"  reward={r:.3f}"
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  DONE {idx}: {task}  ->  {status}{reward_str}")

    elapsed = time.time() - t_total_start
    ok = sum(1 for _, _, rc in all_results if rc == 0)
    fail = len(all_results) - ok

    print(f"\n{'='*60}")
    print(f"Done. {len(all_results)}/{len(all_tasks)} tasks in {elapsed:.0f}s  |  {ok} OK  {fail} FAILED")
    print(f"{'='*60}")
    print(f"Logs: {log_dir}/")


if __name__ == "__main__":
    main()
