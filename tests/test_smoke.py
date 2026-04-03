"""
Smoke test for inference.py components and full API flow.
Run with server already running on port 7860.
"""

import sys
sys.path.insert(0, ".")

from inference import EnvClient, parse_action, build_prompt


def test_parse_json_action():
    a, p = parse_action('{"action_type": "check_prerequisites", "parameters": {}}')
    assert a == "check_prerequisites"
    assert p == {}


def test_parse_json_with_params():
    a, p = parse_action('{"action_type": "pay_fee", "parameters": {"amount": 1000}}')
    assert a == "pay_fee"
    assert p == {"amount": 1000}


def test_parse_markdown_block():
    text = '```json\n{"action_type": "pay_fee", "parameters": {"amount": 1000}}\n```'
    a, p = parse_action(text)
    assert a == "pay_fee"
    assert p == {"amount": 1000}


def test_parse_text_fallback():
    a, p = parse_action("I think we should check_prerequisites first")
    assert a == "check_prerequisites"


def test_parse_garbage_defaults():
    a, p = parse_action("asdfghjkl random gibberish")
    assert a == "check_prerequisites"  # safe default (avoids check_status loops)


def test_build_prompt():
    obs = {
        "task_description": "Test task",
        "difficulty": "easy",
        "citizen_summary": "Test citizen",
        "current_phase": "test",
        "progress_pct": 0.5,
        "steps_taken": 3,
        "max_steps": 15,
        "simulated_day": 5,
        "pending_issues": ["issue1"],
        "completed_steps": ["step1"],
        "last_action_result": "Result text",
        "last_action_error": None,
        "available_actions": ["check_status", "pay_fee"],
        "status_summary": "Do something",
    }
    prompt = build_prompt(obs, 4)
    assert len(prompt) > 50
    assert "Step 4" in prompt
    assert "check_status" in prompt
    assert "issue1" in prompt


def test_full_api_flow():
    """Full end-to-end episode via HTTP API."""
    client = EnvClient("http://localhost:7860")

    # Health
    health = client.health()
    assert health["status"] == "healthy"
    assert health["status"] in ("healthy", "ok")

    # Reset
    r = client.reset("pan_aadhaar_link", seed=10)
    obs = r["observation"]
    assert obs["task_id"] == "pan_aadhaar_link"
    assert obs["steps_taken"] == 0

    # Run perfect flow
    actions = [
        ("check_prerequisites", {}),
        ("compare_documents", {}),
        ("pay_fee", {"amount": 1000}),
        ("submit_application", {}),
        ("check_status", {}),
    ]

    prev_reward = 0.0
    for action_type, params in actions:
        r = client.step(action_type, params)
        reward = r["reward"]
        done = r["done"]
        # Reward should be monotonically increasing
        assert reward >= prev_reward, f"{action_type}: {reward} < {prev_reward}"
        prev_reward = reward
        if done:
            break

    assert r["done"] is True
    assert r["reward"] > 0.7
    assert "final_grade" in r["info"]
    assert r["info"]["final_grade"]["task_completed"] is True

    # Verify breakdown exists
    breakdown = r["info"]["final_grade"]["breakdown"]
    assert len(breakdown) == 7
    dim_names = {d["name"] for d in breakdown}
    assert dim_names == {"diagnosis", "planning", "verification", "execution", "recovery", "efficiency", "safety"}

    print(f"Final score: {r['reward']}")
    for d in breakdown:
        print(f"  {d['name']}: {d['score']:.2f} (w={d['weight']:.2f})")

    client.close()
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    test_parse_json_action()
    test_parse_json_with_params()
    test_parse_markdown_block()
    test_parse_text_fallback()
    test_parse_garbage_defaults()
    test_build_prompt()
    test_full_api_flow()
