"""
Multi-seed stress test — runs the LLM agent across multiple seeds per task
to get statistical confidence in scores and detect edge cases.

Usage:
  HF_TOKEN=... python stress_test.py
  OPENAI_API_KEY=... API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini python stress_test.py
"""

import os
import sys
import statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import run_task, EnvClient, TASKS
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:7860"
SEEDS = [42, 1, 7, 99, 123]


def main():
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or OPENAI_API_KEY or API_KEY")
        sys.exit(1)

    llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env_client = EnvClient(ENV_BASE_URL)

    try:
        env_client.health()
    except Exception as e:
        print(f"ERROR: Cannot connect to environment at {ENV_BASE_URL}: {e}")
        sys.exit(1)

    all_results = {}
    for task in TASKS:
        task_scores = []
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Task: {task} | Seed: {seed}")
            print(f"{'='*60}")
            score = run_task(env_client, llm_client, task, seed=seed)
            task_scores.append(score)

        all_results[task] = task_scores

    # Summary
    print(f"\n\n{'='*70}")
    print("MULTI-SEED STRESS TEST RESULTS")
    print(f"{'='*70}")
    print(f"{'Task':<25} {'Mean':>6} {'Min':>6} {'Max':>6} {'StdDev':>7} {'Seeds':>8}")
    print("-" * 70)

    overall_scores = []
    for task, scores in all_results.items():
        mean = statistics.mean(scores)
        mn = min(scores)
        mx = max(scores)
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        overall_scores.append(mean)
        print(f"{task:<25} {mean:>6.2f} {mn:>6.2f} {mx:>6.2f} {sd:>7.3f} {len(scores):>8}")

    avg = statistics.mean(overall_scores) if overall_scores else 0.0
    print("-" * 70)
    print(f"{'OVERALL AVERAGE':<25} {avg:>6.2f}")
    print(f"{'='*70}")

    # Flag potential issues
    print("\n--- DIAGNOSTICS ---")
    for task, scores in all_results.items():
        if min(scores) < 0.3:
            print(f"  WARNING: {task} has a score below 0.3 (seed may have extreme complications)")
        if max(scores) - min(scores) > 0.5:
            print(f"  WARNING: {task} has high variance ({max(scores) - min(scores):.2f})")
    if all(statistics.mean(s) > 0.6 for s in all_results.values()):
        print("  ✓ All tasks averaging above 0.6 — healthy")
    print()

    env_client.close()


if __name__ == "__main__":
    main()
