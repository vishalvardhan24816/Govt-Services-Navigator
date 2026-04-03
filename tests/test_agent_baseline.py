"""
Baseline agent — validates environment quality without needing an API key.
Runs optimal flows for all 4 tasks, prints [START]/[STEP]/[END] logs,
and verifies scores are in expected ranges.
"""

import json
import httpx
import sys

ENV_BASE_URL = "http://localhost:7860"
BENCHMARK = "govt-services-navigator"


class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def reset(self, task: str, seed: int = 42) -> dict:
        r = self.client.post(f"{self.base_url}/reset", json={"task": task, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, parameters: dict = None) -> dict:
        r = self.client.post(f"{self.base_url}/step", json={"action_type": action_type, "parameters": parameters or {}})
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


# ── Optimal strategies per task ──

def pan_aadhaar_strategy(obs: dict) -> list:
    """Optimal PAN-Aadhaar flow."""
    steps = [("check_prerequisites", {})]
    issues = obs.get("pending_issues", [])
    # After first step, we get issues from the observation
    return steps + [
        ("compare_documents", {}),
        ("evaluate_options", {}),
        ("fix_document", {"target": "aadhaar_name"}),
        ("fix_document", {"target": "aadhaar_dob"}),
        ("pay_fee", {"amount": 1000}),
        ("submit_application", {}),
        ("check_status", {}),
    ]


def passport_strategy(obs: dict) -> list:
    return [
        ("check_prerequisites", {}),
        ("compare_documents", {}),
        ("evaluate_options", {}),
        ("gather_document", {"target": "all"}),
        ("fix_document", {"target": "aadhaar_address"}),
        ("fill_form", {"applicant_name": "Test", "dob": "1990-01-01"}),
        ("pay_fee", {"amount": 1500}),
        ("book_appointment", {}),
        ("submit_application", {}),
        ("wait", {"days": 30}),
        ("check_status", {}),
        ("check_status", {}),
    ]


def driving_licence_strategy(obs: dict) -> list:
    return [
        ("check_prerequisites", {}),
        ("compare_documents", {}),
        ("evaluate_options", {}),
        ("check_eligibility", {}),
        ("gather_document", {"target": "all"}),
        ("fill_form", {"applicant_name": "Test"}),
        ("pay_fee", {"amount": 500}),
        ("book_appointment", {}),
        ("take_test", {"test_type": "written"}),
        ("submit_application", {}),
        ("wait", {"days": 30}),
        ("check_status", {}),
        ("take_test", {"test_type": "practical"}),
        ("check_status", {}),
        ("check_status", {}),
    ]


def vehicle_registration_strategy(obs: dict) -> list:
    return [
        ("check_prerequisites", {}),
        ("compare_documents", {}),
        ("evaluate_options", {}),
        ("fix_document", {"target": "insurance"}),
        ("fix_document", {"target": "address"}),
        ("fix_document", {"target": "invoice"}),
        ("gather_document", {"target": "puc"}),
        ("gather_document", {"target": "bank_noc"}),
        ("fill_form", {"owner_name": "Test"}),
        ("pay_fee", {}),
        ("book_appointment", {}),
        ("submit_application", {}),
        ("wait", {"days": 5}),
        ("check_status", {}),
        ("fix_document", {"target": "chassis"}),
        ("check_status", {}),
        ("wait", {"days": 10}),
        ("check_status", {}),
    ]


STRATEGIES = {
    "pan_aadhaar_link": pan_aadhaar_strategy,
    "passport_fresh": passport_strategy,
    "driving_licence": driving_licence_strategy,
    "vehicle_registration": vehicle_registration_strategy,
}


def run_task(env: EnvClient, task: str, seed: int = 42) -> float:
    print(f"[START] task={task} env={BENCHMARK} model=baseline-agent")

    reset_result = env.reset(task, seed=seed)
    obs = reset_result["observation"]
    strategy = STRATEGIES[task](obs)

    rewards = []
    final_score = 0.0
    done = False
    step_num = 0

    for action_type, params in strategy:
        step_num += 1
        try:
            result = env.step(action_type, params)
            obs = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error = obs.get("last_action_error") or "null"
        except Exception as e:
            reward = 0.0
            done = False
            error = str(e)

        rewards.append(reward)
        action_str = f"{action_type}({json.dumps(params) if params else ''})"
        print(f"[STEP]  step={step_num} action={action_str} reward={reward:.2f} done={'true' if done else 'false'} error={error}")

        if done:
            final_score = reward
            info = result.get("info", {})
            fg = info.get("final_grade", {})
            if fg:
                print(f"        GRADE: {fg.get('score', 0):.2f}")
                for d in fg.get("breakdown", []):
                    print(f"          {d['name']}: {d['score']:.2f} (w={d['weight']:.2f})")
            break

    if not done:
        final_score = rewards[-1] if rewards else 0.0

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success = done and final_score > 0.3
    print(f"[END]   success={'true' if success else 'false'} steps={step_num} score={final_score:.2f} rewards={rewards_str}\n")
    return final_score


def main():
    env = EnvClient(ENV_BASE_URL)

    # Verify health
    try:
        r = httpx.get(f"{ENV_BASE_URL}/health", timeout=5)
        print(f"Server healthy: {r.json()}\n")
    except Exception as e:
        print(f"ERROR: Server not reachable at {ENV_BASE_URL}: {e}")
        sys.exit(1)

    scores = {}
    for task in STRATEGIES:
        scores[task] = run_task(env, task, seed=42)

    print("=" * 60)
    print("BASELINE AGENT RESULTS")
    print("=" * 60)
    for task, score in scores.items():
        status = "PASS" if score >= 0.3 else "FAIL"
        print(f"  [{status}] {task}: {score:.2f}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average: {avg:.2f}")
    print("=" * 60)

    # Assertions
    for task, score in scores.items():
        assert score > 0.0, f"{task} scored 0 — environment may be broken"
    assert avg > 0.3, f"Average score {avg:.2f} too low — environment may have issues"
    print("\nAll baseline checks PASSED!")

    env.close()


if __name__ == "__main__":
    main()
