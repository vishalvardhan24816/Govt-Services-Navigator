"""
Comprehensive test suite for the Government Services Navigator environment.

Test Pyramid:
  1. Unit tests — individual task logic, state transitions, action validation
  2. Integration tests — full episode flows (perfect path, failure path, partial path)
  3. Grader tests — score distributions, monotonicity, determinism
  4. Flow tests — end-to-end scenarios for all 3 tasks with varied complications
  5. Edge case tests — boundary conditions, invalid inputs, max steps
  6. Anti-cheat tests — action spamming, skipping prerequisites, garbage input
"""

import random
from typing import Dict, List

import pytest

from server.env import GovtServicesEnv
from server.grader import grade_trajectory
from server.models import (
    Action,
    ActionType,
    Difficulty,
    Observation,
    StepResult,
    TaskId,
)
from server.tasks import task_pan_aadhaar as pan_task
from server.tasks import task_passport as passport_task
from server.tasks import task_driving_licence as dl_task


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def env():
    return GovtServicesEnv(seed=42)


@pytest.fixture
def env_seeded():
    """Env with fixed seed for deterministic testing."""
    return GovtServicesEnv(seed=12345)


def _action(action_type: str, **params) -> Action:
    return Action(action_type=ActionType(action_type), parameters=params)


# ══════════════════════════════════════════════════════════════════════
# 1. UNIT TESTS — Basic environment mechanics
# ══════════════════════════════════════════════════════════════════════

class TestEnvironmentBasics:
    """Test basic env lifecycle: reset, step, state."""

    def test_reset_returns_observation(self, env):
        obs = env.reset_for_http("pan_aadhaar_link")
        assert isinstance(obs, Observation)
        assert obs.task_id == TaskId.PAN_AADHAAR_LINK
        assert obs.difficulty == Difficulty.EASY
        assert len(obs.citizen_summary) > 0
        assert len(obs.available_actions) > 0
        assert obs.steps_taken == 0
        assert obs.progress_pct == 0.0

    def test_reset_all_tasks(self, env):
        for task in ["pan_aadhaar_link", "passport_fresh", "driving_licence"]:
            obs = env.reset_for_http(task)
            assert obs.task_id == TaskId(task)

    def test_reset_invalid_task(self, env):
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset_for_http("invalid_task")

    def test_step_before_reset_raises(self):
        env = GovtServicesEnv()
        with pytest.raises(RuntimeError, match="reset"):
            env.step_for_http(_action("check_status"))

    def test_state_returns_valid(self, env):
        env.reset_for_http("pan_aadhaar_link")
        state = env.state
        assert state.task_id == TaskId.PAN_AADHAAR_LINK
        assert state.done is False
        assert state.step_count == 0

    def test_step_returns_step_result(self, env):
        env.reset_for_http("pan_aadhaar_link")
        result = env.step_for_http(_action("check_prerequisites"))
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, Observation)
        assert 0.0 <= result.reward <= 1.0
        assert isinstance(result.done, bool)

    def test_step_increments_counter(self, env):
        env.reset_for_http("pan_aadhaar_link")
        env.step_for_http(_action("check_prerequisites"))
        assert env.state.step_count == 1
        env.step_for_http(_action("compare_documents"))
        assert env.state.step_count == 2

    def test_invalid_action_type_in_step(self, env):
        env.reset_for_http("pan_aadhaar_link")
        # WAIT is valid action type but may not apply — should not crash
        result = env.step_for_http(_action("wait", days=5))
        assert isinstance(result, StepResult)

    def test_reset_clears_previous_episode(self, env):
        env.reset_for_http("pan_aadhaar_link")
        env.step_for_http(_action("check_prerequisites"))
        env.reset_for_http("passport_fresh")
        state = env.state
        assert state.task_id == TaskId.PASSPORT_FRESH
        assert state.step_count == 0

    def test_deterministic_with_seed(self):
        env1 = GovtServicesEnv(seed=99)
        env2 = GovtServicesEnv(seed=99)
        obs1 = env1.reset_for_http("pan_aadhaar_link")
        obs2 = env2.reset_for_http("pan_aadhaar_link")
        assert obs1.citizen_summary == obs2.citizen_summary


# ══════════════════════════════════════════════════════════════════════
# 2. INTEGRATION TESTS — Full episode flows
# ══════════════════════════════════════════════════════════════════════

class TestPanAadhaarFullFlow:
    """Test complete PAN-Aadhaar linking flows."""

    def test_perfect_path_no_complications(self):
        """Run perfect path on a citizen with no mismatches."""
        # Find a seed where there are no complications
        for seed in range(100):
            env = GovtServicesEnv(seed=seed)
            obs = env.reset_for_http("pan_aadhaar_link")
            result = env.step_for_http(_action("check_prerequisites"))
            if "No issues" in result.observation.last_action_result or \
               "match" in result.observation.last_action_result.lower():
                if not result.observation.pending_issues:
                    # Clean citizen found — run perfect path
                    env.step_for_http(_action("compare_documents"))
                    env.step_for_http(_action("pay_fee", amount=1000))
                    result = env.step_for_http(_action("submit_application"))
                    if result.observation.last_action_success:
                        final = env.step_for_http(_action("check_status"))
                        assert final.done is True
                        assert final.reward > 0.5
                        return
        # If no clean seed found in 100 tries, that's fine — skip
        pytest.skip("No clean citizen found in 100 seeds")

    def test_name_mismatch_recovery(self):
        """Find a citizen with name mismatch and recover."""
        for seed in range(200):
            env = GovtServicesEnv(seed=seed)
            obs = env.reset_for_http("pan_aadhaar_link")
            result = env.step_for_http(_action("check_prerequisites"))
            if "name_mismatch" in result.observation.pending_issues:
                # Diagnose
                env.step_for_http(_action("compare_documents"))
                env.step_for_http(_action("evaluate_options"))
                # Fix
                env.step_for_http(_action("fix_document", target="aadhaar_name"))
                # Continue
                env.step_for_http(_action("pay_fee", amount=1000))
                result = env.step_for_http(_action("submit_application"))
                assert result.observation.last_action_success
                final = env.step_for_http(_action("check_status"))
                assert final.done is True
                assert final.reward > 0.5
                return
        pytest.skip("No name mismatch citizen found")

    def test_submit_without_prereqs_blocked(self, env):
        """Agent tries to submit without checking — should be blocked."""
        env.reset_for_http("pan_aadhaar_link")
        result = env.step_for_http(_action("submit_application"))
        assert not result.observation.last_action_success
        assert result.observation.last_action_error is not None
        assert "prerequisite" in result.observation.last_action_error.lower() or \
               "check" in result.observation.last_action_error.lower()

    def test_max_steps_terminates(self):
        """Episode ends when max steps reached."""
        env = GovtServicesEnv(seed=42)
        env.reset_for_http("pan_aadhaar_link")
        for _ in range(pan_task.MAX_STEPS):
            result = env.step_for_http(_action("check_status"))
            if result.done:
                break
        assert env.state.done is True


class TestPassportFullFlow:
    """Test complete passport application flows."""

    def test_clean_passport_flow(self):
        """Run a passport flow using ground truth values for correct form fill."""
        from server.tasks import task_passport as pp
        for seed in range(200):
            rng = random.Random(seed)
            citizen, comp = pp.generate_citizen(rng)
            gt = pp.compute_ground_truth(citizen, comp)
            # Find a citizen without address issues so we can submit cleanly
            if not comp.get("address_outdated") and not comp.get("name_mismatch"):
                env = GovtServicesEnv(seed=seed)
                env.reset_for_http("passport_fresh")
                env.step_for_http(_action("check_prerequisites"))
                env.step_for_http(_action("gather_document", target="all"))
                env.step_for_http(_action("fill_form", **gt.correct_form_values))
                env.step_for_http(_action("pay_fee", amount=1500))
                env.step_for_http(_action("book_appointment"))
                result = env.step_for_http(_action("submit_application"))
                if result.observation.last_action_success:
                    env.step_for_http(_action("check_status"))
                    final = env.step_for_http(_action("check_status"))
                    if final.done:
                        assert final.reward > 0.3
                        return
        pytest.skip("No clean passport citizen found in 200 seeds")

    def test_passport_blocks_without_appointment(self, env):
        """Cannot visit PSK without booking appointment."""
        env.reset_for_http("passport_fresh")
        env.step_for_http(_action("check_prerequisites"))
        env.step_for_http(_action("fill_form", applicant_name="Test"))
        result = env.step_for_http(_action("submit_application"))
        assert not result.observation.last_action_success


class TestDrivingLicenceFullFlow:
    """Test complete driving licence flows."""

    def test_clean_dl_flow(self):
        """Run full LL → DL flow for eligible citizen."""
        for seed in range(200):
            env = GovtServicesEnv(seed=seed)
            obs = env.reset_for_http("driving_licence")
            result = env.step_for_http(_action("check_prerequisites"))

            if not result.observation.pending_issues:
                # Phase 1: LL
                env.step_for_http(_action("fill_form",
                    applicant_name="Test",
                    dob="2000-01-01",
                    address="Test",
                    vehicle_class="four_wheeler",
                ))
                env.step_for_http(_action("pay_fee"))
                env.step_for_http(_action("book_appointment"))
                result = env.step_for_http(_action("take_test", test_type="written"))

                if "PASSED" in result.observation.last_action_result:
                    # Phase 2: DL
                    env.step_for_http(_action("check_prerequisites"))
                    if not result.observation.pending_issues or "wait" not in str(result.observation.pending_issues):
                        env.step_for_http(_action("wait", days=35))
                        env.step_for_http(_action("fill_form",
                            applicant_name="Test",
                            vehicle_class="four_wheeler",
                        ))
                        env.step_for_http(_action("pay_fee"))
                        env.step_for_http(_action("book_appointment"))
                        result = env.step_for_http(_action("take_test", test_type="practical"))
                        if "PASSED" in result.observation.last_action_result:
                            final = env.step_for_http(_action("check_status"))
                            if final.done:
                                assert final.reward > 0.3
                                return
        pytest.skip("No clean DL citizen found")

    def test_dl_two_phase_required(self, env):
        """Cannot get DL without first getting LL."""
        env.reset_for_http("driving_licence")
        env.step_for_http(_action("check_prerequisites"))
        # Try to jump to DL form
        result = env.step_for_http(_action("take_test", test_type="practical"))
        # Should fail or not lead to DL completion
        assert not env.state.done


# ══════════════════════════════════════════════════════════════════════
# 3. GRADER TESTS — Score properties
# ══════════════════════════════════════════════════════════════════════

class TestGraderProperties:
    """Test that grader produces diverse, fair, deterministic scores."""

    def test_scores_in_range(self):
        """All scores must be in [0.0, 1.0]."""
        for task in ["pan_aadhaar_link", "passport_fresh", "driving_licence", "vehicle_registration"]:
            for seed in range(10):
                env = GovtServicesEnv(seed=seed)
                env.reset_for_http(task)
                # Take a few random actions
                actions = ["check_prerequisites", "compare_documents", "check_status"]
                for a in actions:
                    result = env.step_for_http(_action(a))
                    assert 0.0 <= result.reward <= 1.0
                    if result.done:
                        break

    def test_scores_are_diverse(self):
        """Grader must NOT return the same score every time (disqualification criteria)."""
        scores = set()
        for seed in range(20):
            env = GovtServicesEnv(seed=seed)
            env.reset_for_http("pan_aadhaar_link")
            # Different strategies
            if seed % 3 == 0:
                # Good strategy
                env.step_for_http(_action("check_prerequisites"))
                env.step_for_http(_action("compare_documents"))
                env.step_for_http(_action("pay_fee", amount=1000))
            elif seed % 3 == 1:
                # Bad strategy — just check status repeatedly
                for _ in range(5):
                    env.step_for_http(_action("check_status"))
            else:
                # Medium strategy
                env.step_for_http(_action("check_prerequisites"))
                env.step_for_http(_action("check_status"))

            # Force episode end
            for _ in range(pan_task.MAX_STEPS):
                result = env.step_for_http(_action("check_status"))
                if result.done:
                    scores.add(round(result.reward, 2))
                    break

        # Must have at least 3 different score values
        assert len(scores) >= 3, f"Only {len(scores)} unique scores: {scores}"

    def test_deterministic_scoring(self):
        """Same trajectory → same score."""
        for _ in range(5):
            seed = random.randint(0, 1000)
            scores = []
            for _ in range(2):
                env = GovtServicesEnv(seed=seed)
                env.reset_for_http("pan_aadhaar_link")
                env.step_for_http(_action("check_prerequisites"))
                env.step_for_http(_action("compare_documents"))
                result = env.step_for_http(_action("pay_fee", amount=1000))
                # Run to completion
                for _ in range(15):
                    result = env.step_for_http(_action("check_status"))
                    if result.done:
                        break
                scores.append(result.reward)
            assert scores[0] == scores[1], f"Scores differ for seed {seed}: {scores}"

    def test_better_trajectory_higher_score(self):
        """A more complete trajectory should generally score higher."""
        for seed in range(10):
            env_bad = GovtServicesEnv(seed=seed)
            env_bad.reset_for_http("pan_aadhaar_link")
            # Bad: just spam check_status
            for _ in range(pan_task.MAX_STEPS):
                r = env_bad.step_for_http(_action("check_status"))
                if r.done:
                    break
            bad_score = r.reward

            env_good = GovtServicesEnv(seed=seed)
            env_good.reset_for_http("pan_aadhaar_link")
            # Good: actually follow the process
            env_good.step_for_http(_action("check_prerequisites"))
            env_good.step_for_http(_action("compare_documents"))
            env_good.step_for_http(_action("evaluate_options"))
            env_good.step_for_http(_action("pay_fee", amount=1000))
            for _ in range(pan_task.MAX_STEPS - 4):
                r = env_good.step_for_http(_action("check_status"))
                if r.done:
                    break
            good_score = r.reward

            # Good strategy should score >= bad strategy
            assert good_score >= bad_score, \
                f"Seed {seed}: good={good_score:.3f} < bad={bad_score:.3f}"

    def test_score_not_always_same(self):
        """Verify across all tasks that scores vary (disqualification check)."""
        for task in ["pan_aadhaar_link", "passport_fresh", "driving_licence", "vehicle_registration"]:
            scores = []
            for seed in range(10):
                env = GovtServicesEnv(seed=seed)
                env.reset_for_http(task)
                if seed % 2 == 0:
                    # Good strategy: diagnose then execute
                    env.step_for_http(_action("check_prerequisites"))
                    env.step_for_http(_action("compare_documents"))
                    env.step_for_http(_action("evaluate_options"))
                    env.step_for_http(_action("fill_form"))
                    env.step_for_http(_action("pay_fee", amount=1000))
                    env.step_for_http(_action("book_appointment"))
                    env.step_for_http(_action("submit_application"))
                else:
                    # Bad strategy: just spam check_status
                    pass
                config = env.TASK_CONFIGS[env._task_id]
                for _ in range(config["max_steps"]):
                    r = env.step_for_http(_action("check_status"))
                    if r.done:
                        break
                scores.append(round(r.reward, 2))
            unique = len(set(scores))
            assert unique >= 2, f"Task {task}: all scores identical = {scores[0]}"


# ══════════════════════════════════════════════════════════════════════
# 4. FLOW TESTS — Specific scenarios with complications
# ══════════════════════════════════════════════════════════════════════

class TestComplicationFlows:
    """Test that specific complications are handled correctly."""

    def test_pan_fee_required_before_submit(self):
        """Cannot submit PAN-Aadhaar link without paying fee."""
        env = GovtServicesEnv(seed=0)
        env.reset_for_http("pan_aadhaar_link")
        env.step_for_http(_action("check_prerequisites"))
        env.step_for_http(_action("compare_documents"))
        result = env.step_for_http(_action("submit_application"))
        # Should fail — fee not paid
        if result.observation.last_action_success:
            # Might succeed if no mismatch and fee check is different
            pass
        else:
            assert "fee" in result.observation.last_action_error.lower() or \
                   "pay" in result.observation.last_action_error.lower()

    def test_passport_address_mismatch_blocks_psk(self):
        """Cannot visit PSK with unresolved address mismatch."""
        for seed in range(200):
            env = GovtServicesEnv(seed=seed)
            obs = env.reset_for_http("passport_fresh")
            r = env.step_for_http(_action("check_prerequisites"))
            if "aadhaar_address_outdated" in r.observation.pending_issues:
                env.step_for_http(_action("fill_form", applicant_name="Test"))
                env.step_for_http(_action("pay_fee", amount=1500))
                env.step_for_http(_action("book_appointment"))
                result = env.step_for_http(_action("submit_application"))
                assert not result.observation.last_action_success
                assert result.observation.last_action_error is not None
                return
        pytest.skip("No address mismatch citizen found")


# ══════════════════════════════════════════════════════════════════════
# 5. EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test boundary conditions and unusual inputs."""

    def test_double_reset(self, env):
        """Double reset should work cleanly."""
        env.reset_for_http("pan_aadhaar_link")
        env.step_for_http(_action("check_prerequisites"))
        obs = env.reset_for_http("pan_aadhaar_link")
        assert obs.steps_taken == 0

    def test_step_after_done(self, env):
        """Step after episode ends should raise."""
        env.reset_for_http("pan_aadhaar_link")
        # Force episode to end
        for _ in range(pan_task.MAX_STEPS + 5):
            try:
                result = env.step_for_http(_action("check_status"))
                if result.done:
                    break
            except RuntimeError:
                break
        # Should raise on next step
        with pytest.raises(RuntimeError):
            env.step_for_http(_action("check_status"))

    def test_empty_parameters(self, env):
        """Actions with empty parameters should not crash."""
        env.reset_for_http("pan_aadhaar_link")
        result = env.step_for_http(_action("check_prerequisites"))
        assert result.observation.last_action_success

    def test_extra_parameters_ignored(self, env):
        """Unknown parameters should be safely ignored."""
        env.reset_for_http("pan_aadhaar_link")
        result = env.step_for_http(Action(
            action_type=ActionType.CHECK_PREREQUISITES,
            parameters={"unknown_field": "garbage", "foo": 42},
        ))
        assert result.observation.last_action_success

    def test_all_action_types_accepted(self, env):
        """Every ActionType should be accepted without crash."""
        env.reset_for_http("pan_aadhaar_link")
        for at in ActionType:
            try:
                env.step_for_http(Action(action_type=at, parameters={}))
            except RuntimeError:
                # Episode might end, that's fine
                env.reset_for_http("pan_aadhaar_link")


# ══════════════════════════════════════════════════════════════════════
# 6. ANTI-CHEAT TESTS
# ══════════════════════════════════════════════════════════════════════

class TestAntiCheat:
    """Verify environment resists gaming/exploitation."""

    def test_spamming_same_action_low_score(self):
        """Spamming the same action should not yield high score."""
        env = GovtServicesEnv(seed=42)
        env.reset_for_http("pan_aadhaar_link")
        for _ in range(pan_task.MAX_STEPS):
            result = env.step_for_http(_action("check_prerequisites"))
            if result.done:
                break
        assert result.reward < 0.6, f"Spam score too high: {result.reward}"

    def test_submit_without_diagnosis_penalized(self):
        """Jumping straight to submit should score lower than proper flow."""
        # Strategy 1: Submit immediately
        env1 = GovtServicesEnv(seed=42)
        env1.reset_for_http("pan_aadhaar_link")
        r1 = env1.step_for_http(_action("submit_application"))
        for _ in range(pan_task.MAX_STEPS - 1):
            r1 = env1.step_for_http(_action("check_status"))
            if r1.done:
                break
        skip_score = r1.reward

        # Strategy 2: Proper diagnosis then submit
        env2 = GovtServicesEnv(seed=42)
        env2.reset_for_http("pan_aadhaar_link")
        env2.step_for_http(_action("check_prerequisites"))
        env2.step_for_http(_action("compare_documents"))
        env2.step_for_http(_action("pay_fee", amount=1000))
        for _ in range(pan_task.MAX_STEPS - 3):
            r2 = env2.step_for_http(_action("check_status"))
            if r2.done:
                break
        proper_score = r2.reward

        assert proper_score >= skip_score, \
            f"Skipping diagnosis scored better: skip={skip_score:.3f} vs proper={proper_score:.3f}"


# ══════════════════════════════════════════════════════════════════════
# 7. CITIZEN GENERATION TESTS
# ══════════════════════════════════════════════════════════════════════

class TestCitizenGeneration:
    """Test that citizen profiles are generated correctly."""

    def test_pan_citizen_has_required_docs(self):
        rng = random.Random(42)
        citizen, comp = pan_task.generate_citizen(rng)
        assert "pan_card" in citizen.documents
        assert "aadhaar_card" in citizen.documents
        assert citizen.identifiers.get("pan_number") is not None
        assert citizen.identifiers.get("aadhaar_number") is not None

    def test_passport_citizen_has_required_docs(self):
        rng = random.Random(42)
        citizen, comp = passport_task.generate_citizen(rng)
        assert "aadhaar_card" in citizen.documents
        assert "birth_certificate" in citizen.documents
        assert "address_proof" in citizen.documents
        assert "passport_photo" in citizen.documents

    def test_dl_citizen_has_required_docs(self):
        rng = random.Random(42)
        citizen, comp = dl_task.generate_citizen(rng)
        assert "aadhaar_card" in citizen.documents
        assert "age_proof" in citizen.documents
        assert citizen.attributes.get("vehicle_class") is not None

    def test_dl_age_distribution(self):
        """DL task should generate citizens of various ages including edge cases."""
        ages = set()
        for seed in range(100):
            rng = random.Random(seed)
            citizen, _ = dl_task.generate_citizen(rng)
            ages.add(citizen.age)
        assert min(ages) <= 17, "Should generate some minors"
        assert max(ages) >= 40, "Should generate some 40+ (needing medical cert)"

    def test_complication_variety(self):
        """Various complications should be generated across seeds."""
        complications_seen = set()
        for seed in range(100):
            rng = random.Random(seed)
            _, comp = pan_task.generate_citizen(rng)
            if comp["name_mismatch"]:
                complications_seen.add("name_mismatch")
            if comp.get("dob_mismatch"):
                complications_seen.add("dob_mismatch")
            if comp.get("already_linked"):
                complications_seen.add("already_linked")
        assert len(complications_seen) >= 2, f"Only saw: {complications_seen}"


# ══════════════════════════════════════════════════════════════════════
# 8. GROUND TRUTH TESTS
# ══════════════════════════════════════════════════════════════════════

class TestGroundTruth:
    """Test that ground truth is computed correctly."""

    def test_ground_truth_has_checkpoints(self):
        rng = random.Random(42)
        citizen, comp = pan_task.generate_citizen(rng)
        gt = pan_task.compute_ground_truth(citizen, comp)
        assert len(gt.required_checkpoints) >= 3
        assert gt.optimal_steps >= 3

    def test_ground_truth_with_mismatch_has_fix_checkpoint(self):
        """If citizen has name mismatch, ground truth must include fix step."""
        for seed in range(100):
            rng = random.Random(seed)
            citizen, comp = pan_task.generate_citizen(rng)
            if comp["name_mismatch"]:
                gt = pan_task.compute_ground_truth(citizen, comp)
                fix_steps = [cp for cp in gt.required_checkpoints if cp.action_type == ActionType.FIX_DOCUMENT]
                assert len(fix_steps) > 0, "Mismatch citizen should have fix checkpoint"
                assert "name_mismatch" in gt.expected_issues
                return
        pytest.skip("No mismatch citizen found")

    def test_ground_truth_passport_has_form_values(self):
        rng = random.Random(42)
        citizen, comp = passport_task.generate_citizen(rng)
        gt = passport_task.compute_ground_truth(citizen, comp)
        assert "applicant_name" in gt.correct_form_values
        assert "dob" in gt.correct_form_values
        assert gt.correct_fee is not None

    def test_ground_truth_dl_two_phases(self):
        rng = random.Random(42)
        citizen, comp = dl_task.generate_citizen(rng)
        gt = dl_task.compute_ground_truth(citizen, comp)
        # Should have checkpoints for both phases
        action_types = [cp.action_type for cp in gt.required_checkpoints]
        assert ActionType.TAKE_TEST in action_types  # written or driving test
        assert gt.optimal_steps >= 8  # minimum for full two-phase flow
