"""CLI entry point for the oracle pipeline."""

import argparse
import logging

from .oracle_loop import run_oracle_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Oracle pipeline: real-env action selection")
    parser.add_argument("--task", required=True, help="BrowserGym task name (e.g. workarena.servicenow.xxx)")
    parser.add_argument("--seed", type=int, required=True, help="Task seed")
    parser.add_argument("--model", default="openai/gpt-5-2025-08-07", help="LLM model key from CHAT_MODEL_ARGS_DICT")
    parser.add_argument("--n-candidates", type=int, default=5, help="Number of candidate actions per step")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode")
    parser.add_argument("--save-dir", default="oracle_results", help="Directory to save results")
    parser.add_argument("--debug-dir", default="oracle_wm/debug", help="Directory to save debug artifacts")
    parser.add_argument("--cleanup", action="store_true", help="Delete orphaned @workarena.com users before starting (use after interrupted runs)")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--resume-from", type=int, default=0, metavar="N", help="Resume from step N (reads committed history from debug dir)")
    args = parser.parse_args()

    save_path = run_oracle_pipeline(
        task_name=args.task,
        task_seed=args.seed,
        model=args.model,
        n_candidates=args.n_candidates,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        debug_dir=args.debug_dir,
        cleanup=args.cleanup,
        headless=args.headless,
        resume_from=args.resume_from,
    )
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
