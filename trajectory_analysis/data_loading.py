"""Pipeline detection, task enumeration, and step/reward loading."""

import gzip
import json
import pickle
import re
from pathlib import Path


def detect_pipeline(results_dir: Path) -> str:
    """Detect whether results_dir contains oracle or agent pipeline results."""
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "summary.json").exists():
            return "oracle"
        if (child / "summary_info.json").exists():
            return "agent"
        # Agent pipeline has two-level nesting: study_dir/task_dir/summary_info.json
        for grandchild in child.iterdir():
            if grandchild.is_dir() and (grandchild / "summary_info.json").exists():
                return "agent"
    raise RuntimeError(f"Cannot detect pipeline type in {results_dir}")


def enumerate_task_dirs(results_dir: Path, pipeline: str) -> list[tuple[str, Path]]:
    """Return (task_name, task_dir) pairs."""
    tasks = []
    if pipeline == "oracle":
        for child in sorted(results_dir.iterdir()):
            if child.is_dir() and (child / "summary.json").exists():
                tasks.append((child.name, child))
    else:
        # Agent: results_dir is either the study dir or the parent of study dirs
        has_summary = any(
            (c / "summary_info.json").exists()
            for c in results_dir.iterdir()
            if c.is_dir()
        )
        if has_summary:
            for child in sorted(results_dir.iterdir()):
                if child.is_dir() and (child / "summary_info.json").exists():
                    tasks.append((child.name, child))
        else:
            for study in sorted(results_dir.iterdir()):
                if not study.is_dir():
                    continue
                for child in sorted(study.iterdir()):
                    if child.is_dir() and (child / "summary_info.json").exists():
                        tasks.append((child.name, child))
    return tasks


def load_reward(task_dir: Path, pipeline: str) -> float:
    """Load the final reward from the summary file."""
    if pipeline == "oracle":
        summary = json.loads((task_dir / "summary.json").read_text())
        return summary.get("reward", 0.0)
    else:
        summary = json.loads((task_dir / "summary_info.json").read_text())
        return summary.get("cum_reward", 0.0)


def load_steps(task_dir: Path) -> list:
    """Load all StepInfo objects from step_N.pkl.gz files, sorted by step number."""
    step_files = sorted(
        task_dir.glob("step_*.pkl.gz"),
        key=lambda p: int(re.search(r"step_(\d+)", p.stem).group(1)),
    )
    steps = []
    for sf in step_files:
        with gzip.open(sf, "rb") as f:
            steps.append(pickle.load(f))
    return steps
