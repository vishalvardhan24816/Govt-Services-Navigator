"""Quick verification of the Vehicle Registration (4th expert task)."""

from server.env import GovtServicesEnv
from server.models import Action, ActionType


def test_vehicle_registration_full_flow():
    env = GovtServicesEnv(seed=42)
    obs = env.reset_for_http("vehicle_registration")
    print("Task:", obs.task_id, "| Difficulty:", obs.difficulty)
    print("Citizen:", obs.citizen_summary.split("\n")[0])

    steps = [
        ("check_prerequisites", {}),
        ("compare_documents", {}),
        ("evaluate_options", {}),
        ("fill_form", {"owner_name": "Test"}),
        ("pay_fee", {}),
        ("book_appointment", {}),
        ("submit_application", {}),
        ("wait", {"days": 5}),
        ("check_status", {}),
        ("wait", {"days": 10}),
        ("check_status", {}),
    ]

    for action_name, params in steps:
        r = env.step_for_http(Action(action_type=ActionType(action_name), parameters=params))
        status = "OK" if r.observation.last_action_success else "FAIL: " + str(r.observation.last_action_error)
        print(f"  {action_name}: reward={r.reward:.2f} done={r.done} | {status}")
        if r.done:
            if "final_grade" in r.info:
                fg = r.info["final_grade"]
                print(f"\n  FINAL SCORE: {fg['score']:.2f}")
                for d in fg["breakdown"]:
                    print(f"    {d['name']}: {d['score']:.2f}")
            break

    assert r.done, "Episode should be complete"
    assert r.reward > 0.3, f"Score too low: {r.reward}"
    print("\nVehicle registration flow: PASSED")


def test_vehicle_registration_with_complications():
    """Find a seed with complications and verify they're handled."""
    for seed in range(50):
        env = GovtServicesEnv(seed=seed)
        obs = env.reset_for_http("vehicle_registration")
        r = env.step_for_http(Action(action_type=ActionType.CHECK_PREREQUISITES, parameters={}))
        if r.observation.pending_issues:
            print(f"Seed {seed}: issues={r.observation.pending_issues}")
            # Fix all issues
            for issue in list(r.observation.pending_issues):
                if "insurance" in issue:
                    env.step_for_http(Action(action_type=ActionType.FIX_DOCUMENT, parameters={"target": "insurance"}))
                elif "address" in issue:
                    env.step_for_http(Action(action_type=ActionType.FIX_DOCUMENT, parameters={"target": "address"}))
                elif "invoice" in issue:
                    env.step_for_http(Action(action_type=ActionType.FIX_DOCUMENT, parameters={"target": "invoice"}))
                elif "puc" in issue:
                    env.step_for_http(Action(action_type=ActionType.GATHER_DOCUMENT, parameters={"target": "puc"}))
                elif "hypothecation" in issue:
                    env.step_for_http(Action(action_type=ActionType.GATHER_DOCUMENT, parameters={"target": "bank_noc"}))

            # Continue flow
            env.step_for_http(Action(action_type=ActionType.COMPARE_DOCUMENTS, parameters={}))
            env.step_for_http(Action(action_type=ActionType.FILL_FORM, parameters={"owner_name": "Test"}))
            env.step_for_http(Action(action_type=ActionType.PAY_FEE, parameters={}))
            env.step_for_http(Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={}))
            r = env.step_for_http(Action(action_type=ActionType.SUBMIT_APPLICATION, parameters={}))
            assert r.observation.last_action_success, f"Submit failed: {r.observation.last_action_error}"
            print(f"  Submit OK after fixing issues")
            return

    print("No complicated citizen found (unlikely)")


def test_all_tasks_still_work():
    """Verify all 4 tasks reset and step without errors."""
    for task in ["pan_aadhaar_link", "passport_fresh", "driving_licence", "vehicle_registration"]:
        env = GovtServicesEnv(seed=10)
        obs = env.reset_for_http(task)
        r = env.step_for_http(Action(action_type=ActionType.CHECK_PREREQUISITES, parameters={}))
        assert r.observation.last_action_success
        print(f"  {task}: OK (issues={r.observation.pending_issues})")


if __name__ == "__main__":
    print("=== Test 1: Full flow ===")
    test_vehicle_registration_full_flow()
    print("\n=== Test 2: Complications ===")
    test_vehicle_registration_with_complications()
    print("\n=== Test 3: All 4 tasks ===")
    test_all_tasks_still_work()
    print("\nAll vehicle registration tests PASSED!")
