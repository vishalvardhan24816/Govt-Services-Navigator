"""Inference Script — Government Services Navigator
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

Uses OpenAI client pointed at HuggingFace router.
"""

import json
import os
import re
import sys
from typing import List, Optional, Tuple

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("HG_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:7860"
BENCHMARK = "govt-services-navigator"
TEMPERATURE = 0.6

TASKS = [
    "pan_aadhaar_link",
    "passport_fresh",
    "driving_licence",
    "vehicle_registration",
]

# All valid action types (for parse_action fallback)
VALID_ACTIONS = frozenset({
    "check_prerequisites", "compare_documents", "evaluate_options",
    "check_eligibility", "check_status",
    "fill_form", "pay_fee", "book_appointment", "submit_application",
    "gather_document", "fix_document", "take_test", "wait",
    "appeal_rejection",
})

# ──────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent navigating Indian government services.

Each step you receive an observation describing the current state: task, phase, available actions,
completed steps, pending issues, services status, citizen profile, and the result of your last action.

Respond with exactly ONE JSON action:
{"action_type": "<action_name>", "parameters": {...}}

Use the observation to reason about what to do next. Learn from errors and rewards."""

# ──────────────────────────────────────────────────────────────────────
# ENV CLIENT
# ──────────────────────────────────────────────────────────────────────

class EnvClient:
    """HTTP client for the environment server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def health(self) -> dict:
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def reset(self, task: str, seed: Optional[int] = None) -> dict:
        payload = {"task": task}
        if seed is not None:
            payload["seed"] = seed
        r = self.client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, parameters: dict = None) -> dict:
        payload = {"action_type": action_type, "parameters": parameters or {}}
        r = self.client.post(f"{self.base_url}/step", json=payload)
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def parse_action(model_output: str) -> Tuple[str, dict]:
    """Parse model output into (action_type, parameters)."""
    text = model_output.strip()

    # Extract JSON from markdown code blocks or raw text
    m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            text = m.group(0)

    try:
        data = json.loads(text)
        return data.get("action_type", "check_prerequisites"), data.get("parameters", {})
    except (json.JSONDecodeError, KeyError):
        for kw in VALID_ACTIONS:
            if kw in text.lower():
                return kw, {}
        return "check_prerequisites", {}


ACTION_SEMANTICS = {
    "check_prerequisites": "verify documents and eligibility",
    "check_eligibility": "check age/identity eligibility",
    "compare_documents": "cross-check fields across documents",
    "evaluate_options": "review available service options",
    "fix_document": "correct a document issue — params: {\"target\": \"<issue_name>\"}",
    "gather_document": "collect required documents — params: {\"target\": \"all\"}",
    "fill_form": "complete the application form — include citizen data fields as params",
    "pay_fee": "pay the required fee",
    "book_appointment": "schedule an appointment or test slot",
    "submit_application": "submit the completed application",
    "take_test": "attempt an examination — params: {\"test_type\": \"written\"} or {\"test_type\": \"practical\"}",
    "wait": "advance time for processing — params: {\"days\": <number>}",
    "check_status": "check the current processing status",
    "appeal_rejection": "appeal a rejected application",
}


_ACTION_TYPES_BY_LEN = sorted(VALID_ACTIONS, key=len, reverse=True)


def _normalize_step(step_name: str) -> str:
    """Map any internal step name to its closest action_type via substring matching.
    General-purpose — works for any task without hardcoded mappings."""
    # Direct match: "compare_documents", "check_eligibility", etc.
    for act in _ACTION_TYPES_BY_LEN:
        if act in step_name:
            return act
    # Common verb/noun patterns for steps that don't embed an action name
    if "prereq" in step_name:
        return "check_prerequisites"
    if "slot" in step_name or "appointment" in step_name:
        return "book_appointment"
    if "test" in step_name or "exam" in step_name:
        return "take_test"
    if any(w in step_name for w in ("fix", "correct", "obtain", "replace")):
        return "fix_document"
    if "received" in step_name or "issued" in step_name:
        return "check_status"
    return step_name


def build_prompt(observation: dict, step_num: int, last_reward: float = 0.0) -> str:
    """Build a prompt from observation. Presents all state information
    for the LLM to reason about — no filtering, no strategy hints."""
    obs = observation
    parts = [f"=== Step {step_num} of {obs.get('max_steps', 25)} ==="]

    if obs.get('last_action_error'):
        parts.append(f"\n*** ERROR: {obs['last_action_error']} ***")

    if step_num > 1:
        parts.append(f"Last reward: {last_reward:.2f}")

    parts.append(f"\nTask: {obs.get('task_description', '')}")
    parts.append(f"Phase: {obs.get('current_phase', '')} | Progress: {obs.get('progress_pct', 0):.0%} | Day: {obs.get('simulated_day', 0)}")

    if obs.get('status_summary'):
        parts.append(f"Status: {obs['status_summary']}")

    svc = obs.get('services_status', {})
    if svc:
        parts.append(f"Services: {', '.join(k + '=' + v for k, v in svc.items())}")

    if obs.get('pending_issues'):
        parts.append(f"Pending issues: {', '.join(obs['pending_issues'])}")

    completed = obs.get('completed_steps', [])
    if completed:
        translated = list(dict.fromkeys(_normalize_step(s) for s in completed))
        parts.append(f"Completed: {', '.join(translated[-10:])}")

    avail = obs.get('available_actions', [])
    action_lines = []
    for a in avail:
        desc = ACTION_SEMANTICS.get(a, "")
        action_lines.append(f"  {a}: {desc}" if desc else f"  {a}")
    parts.append(f"\nAvailable actions:\n" + "\n".join(action_lines))

    parts.append(f"\n{obs.get('citizen_summary', '')}")

    # Show citizen documents (needed for form filling)
    docs = obs.get('citizen_documents', {})
    if docs:
        doc_lines = []
        for dname, dinfo in docs.items():
            if isinstance(dinfo, dict):
                fields = dinfo.get('fields', dinfo)
                doc_lines.append(f"  {dname}: {fields}")
            else:
                doc_lines.append(f"  {dname}: {dinfo}")
        parts.append("Documents:\n" + "\n".join(doc_lines[:6]))

    if obs.get('last_action_result') and not obs.get('last_action_error'):
        parts.append(f"\nResult: {obs['last_action_result'][:500]}")

    parts.append("\nRespond with ONE JSON action: {\"action_type\": \"...\", \"parameters\": {...}}")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# MAIN INFERENCE LOOP
# ──────────────────────────────────────────────────────────────────────

def run_task(env_client: EnvClient, llm_client: OpenAI, task_name: str, seed: int = 42) -> float:
    """Run one task episode. Returns final score.

    Pure LLM baseline — no guardrails, no hardcoded strategy.
    The LLM reasons from observations and learns from errors/rewards.
    """
    rewards: List[float] = []
    steps = 0
    success = False
    final_score = 0.0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        reset_result = env_client.reset(task_name, seed=seed)
        observation = reset_result["observation"]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        max_steps = observation.get("max_steps", 30)
        for step_num in range(1, max_steps + 1):
            steps = step_num

            # Build prompt and call LLM
            last_reward = rewards[-1] if rewards else 0.0
            user_prompt = build_prompt(observation, step_num, last_reward=last_reward)

            # Inject reward trend into prompt so LLM sees stagnation
            if len(rewards) >= 2:
                prev = rewards[-2]
                delta = last_reward - prev
                if delta > 0.01:
                    trend = f"(reward increased +{delta:.2f})"
                elif delta < -0.01:
                    trend = f"(reward DECREASED {delta:.2f} — try a different action)"
                else:
                    trend = "(reward unchanged — try a different action)"
                user_prompt = user_prompt.replace(
                    f"Last reward: {last_reward:.2f}",
                    f"Last reward: {last_reward:.2f} {trend}",
                )
            messages.append({"role": "user", "content": user_prompt})
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

            try:
                response = llm_client.chat.completions.create(
                    model=MODEL_NAME, messages=messages,
                    temperature=TEMPERATURE, max_tokens=500,
                )
                model_output = response.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": model_output})
            except Exception as api_err:
                print(f"  [LLM ERROR] {api_err}", file=sys.stderr)
                model_output = '{"action_type": "check_prerequisites"}'

            # Parse LLM output and execute — no overrides, no guards
            action_type, parameters = parse_action(model_output)

            action_str = f"{action_type}({json.dumps(parameters) if parameters else ''})"
            try:
                step_result = env_client.step(action_type, parameters)
                observation = step_result["observation"]
                reward = step_result.get("reward", 0.0)
                done = step_result.get("done", False)
                error = observation.get("last_action_error")
            except Exception as e:
                reward, done, error = 0.0, False, str(e)

            rewards.append(reward)
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={'true' if done else 'false'} error={error or 'null'}", flush=True)

            if done:
                final_score = reward
                info = step_result.get("info", {})
                success = info.get("final_grade", {}).get("task_completed", False)
                break

        if not done:
            final_score = rewards[-1] if rewards else 0.0

    except Exception as e:
        print(f"[STEP] step={steps} action=error reward=0.00 done=true error={str(e)}", flush=True)
        rewards.append(0.0)
        final_score = 0.0

    final_score = min(max(final_score, 0.0), 1.0)  # clamp to [0, 1]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={'true' if success else 'false'} steps={steps} score={final_score:.3f} rewards={rewards_str}", flush=True)
    return final_score


def main():
    """Run inference on all tasks."""
    llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env_client = EnvClient(ENV_BASE_URL)

    try:
        health = env_client.health()
        print(f"Environment healthy: {health.get('environment', 'unknown')}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Cannot connect to environment at {ENV_BASE_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    scores = {}
    for task in TASKS:
        scores[task] = run_task(env_client, llm_client, task, seed=42)

    env_client.close()


if __name__ == "__main__":
    main()
