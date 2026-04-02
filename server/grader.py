"""
Government Services Navigator — Grader

7-Dimension adaptive scoring with per-task weight adjustment.

Dimensions:
  1. Diagnosis    — Did agent check prerequisites / compare documents BEFORE acting?
  2. Planning     — Did agent evaluate options and choose a valid resolution path?
  3. Verification — Did agent verify documents and detect issues correctly?
  4. Execution    — Were forms filled correctly, right fees paid, right docs submitted?
  5. Recovery     — Did agent handle failures and resolve them?
  6. Efficiency   — How many steps vs optimal? Wasted actions?
  7. Safety       — Did agent avoid harmful/forbidden actions?

Each dimension scored 0.0-1.0 independently, then combined with task-specific weights.
Final score capped at 0.95 to leave room for edge cases.
"""

from __future__ import annotations

from typing import Any, Dict, List

from server.models import (
    ActionType,
    DimensionScore,
    Reward,
    TaskId,
    Trajectory,
)

# ──────────────────────────────────────────────────────────────────────
# TASK-SPECIFIC WEIGHTS
# ──────────────────────────────────────────────────────────────────────

TASK_WEIGHTS = {
    TaskId.PAN_AADHAAR_LINK: {
        "diagnosis":    0.30,
        "planning":     0.10,
        "verification": 0.05,
        "execution":    0.15,
        "recovery":     0.25,
        "efficiency":   0.10,
        "safety":       0.05,
    },
    TaskId.PASSPORT_FRESH: {
        "diagnosis":    0.15,
        "planning":     0.10,
        "verification": 0.15,
        "execution":    0.25,
        "recovery":     0.15,
        "efficiency":   0.10,
        "safety":       0.10,
    },
    TaskId.DRIVING_LICENCE: {
        "diagnosis":    0.15,
        "planning":     0.20,
        "verification": 0.10,
        "execution":    0.15,
        "recovery":     0.15,
        "efficiency":   0.15,
        "safety":       0.10,
    },
    TaskId.VEHICLE_REGISTRATION: {
        "diagnosis":    0.10,
        "planning":     0.15,
        "verification": 0.15,
        "execution":    0.20,
        "recovery":     0.20,
        "efficiency":   0.10,
        "safety":       0.10,
    },
}

# Diagnostic action types (show thinking)
DIAGNOSTIC_ACTIONS = {
    ActionType.CHECK_PREREQUISITES,
    ActionType.COMPARE_DOCUMENTS,
    ActionType.EVALUATE_OPTIONS,
    ActionType.CHECK_ELIGIBILITY,
    ActionType.CHECK_STATUS,
}

# Execution action types (do things)
EXECUTION_ACTIONS = {
    ActionType.GATHER_DOCUMENT,
    ActionType.FILL_FORM,
    ActionType.PAY_FEE,
    ActionType.BOOK_APPOINTMENT,
    ActionType.SUBMIT_APPLICATION,
    ActionType.FIX_DOCUMENT,
    ActionType.TAKE_TEST,
    ActionType.APPEAL_REJECTION,
}


# ──────────────────────────────────────────────────────────────────────
# DIMENSION SCORERS
# ──────────────────────────────────────────────────────────────────────

def _score_diagnosis(task_state: Any, trajectory: Trajectory) -> DimensionScore:
    """Did agent check prerequisites and compare documents BEFORE taking execution actions?"""
    score = 0.0
    feedback_parts = []

    # Check if diagnostic actions came before execution actions
    actions = task_state.action_history
    first_diagnostic_step = None
    first_execution_step = None

    for a in actions:
        try:
            at = ActionType(a["action"])
        except ValueError:
            continue
        if at in DIAGNOSTIC_ACTIONS and first_diagnostic_step is None:
            first_diagnostic_step = a["step"]
        if at in EXECUTION_ACTIONS and first_execution_step is None:
            first_execution_step = a["step"]

    # Did agent diagnose before executing?
    if first_diagnostic_step is not None:
        if first_execution_step is None or first_diagnostic_step < first_execution_step:
            score += 0.5
            feedback_parts.append("Diagnosed before executing")
        else:
            score += 0.1
            feedback_parts.append("Executed before diagnosing — risky approach")
    else:
        feedback_parts.append("No diagnostic actions taken")

    # Did agent check prerequisites?
    prereqs_checked = getattr(task_state, "prerequisites_checked", False) or \
                      getattr(task_state, "ll_prereqs_checked", False)
    if prereqs_checked:
        score += 0.3
        feedback_parts.append("Prerequisites checked")
    else:
        feedback_parts.append("Prerequisites NOT checked")

    # Did agent compare documents?
    docs_compared = getattr(task_state, "documents_compared", False) or \
                    getattr(task_state, "ll_documents_compared", False)
    if docs_compared:
        score += 0.2
        feedback_parts.append("Documents compared")

    return DimensionScore(
        name="diagnosis",
        score=min(score, 1.0),
        weight=0.0,  # set by caller
        feedback="; ".join(feedback_parts),
    )


def _score_planning(task_state: Any, trajectory: Trajectory) -> DimensionScore:
    """Did agent evaluate options and choose a valid resolution path?"""
    score = 0.0
    feedback_parts = []

    gt = trajectory.ground_truth
    expected_issues = gt.expected_issues if gt else []

    # Did agent evaluate options when there were issues?
    options_evaluated = getattr(task_state, "options_evaluated", False) or \
                        getattr(task_state, "ll_options_evaluated", False)

    if expected_issues:
        if options_evaluated:
            score += 0.5
            feedback_parts.append("Evaluated resolution options")
        else:
            score += 0.1
            feedback_parts.append("Did not evaluate options before fixing")

        # Did agent detect the expected issues?
        detected = 0
        for issue in expected_issues:
            # Check various detection flags
            if issue == "name_mismatch" and getattr(task_state, "name_mismatch_detected", False):
                detected += 1
            elif issue == "name_mismatch_docs" and getattr(task_state, "name_mismatch_detected", False):
                detected += 1
            elif issue == "dob_mismatch" and getattr(task_state, "dob_mismatch_detected", False):
                detected += 1
            elif issue == "aadhaar_address_outdated" and getattr(task_state, "address_outdated_detected", False):
                detected += 1
            elif issue == "address_proof_invalid" and getattr(task_state, "address_proof_issue_detected", False):
                detected += 1
            elif issue == "photo_rejected" and getattr(task_state, "photo_issue_detected", False):
                detected += 1
            elif issue == "underage_for_class" and getattr(task_state, "underage_detected", False):
                detected += 1
            elif issue == "address_mismatch" and getattr(task_state, "address_mismatch_detected", False):
                detected += 1
            elif issue == "medical_cert_missing" and getattr(task_state, "medical_missing_detected", False):
                detected += 1
            elif issue == "already_linked" and getattr(task_state, "already_linked_detected", False):
                detected += 1
            elif issue in ("written_test_fail", "driving_test_fail", "dl_applied_too_early", "ll_expired"):
                detected += 1  # these are runtime events, not pre-detectable
            elif issue in ("inspection_failure",):
                detected += 1  # runtime event discovered via check_status
            elif issue in ("insurance_expired", "missing_puc", "hypothecation_required",
                           "invoice_discrepancy") and getattr(task_state, "prerequisites_checked", False):
                detected += 1  # VR issues detected via check_prerequisites

        if expected_issues:
            detection_rate = detected / len(expected_issues)
            score += 0.5 * detection_rate
            feedback_parts.append(f"Detected {detected}/{len(expected_issues)} expected issues")
    else:
        # No issues — planning is straightforward, give credit for checking
        score = 0.8 if getattr(task_state, "diagnostic_before_execution", False) else 0.5
        feedback_parts.append("No issues to plan around — straightforward case")

    return DimensionScore(
        name="planning",
        score=min(score, 1.0),
        weight=0.0,
        feedback="; ".join(feedback_parts),
    )


def _score_verification(task_state: Any, trajectory: Trajectory) -> DimensionScore:
    """Did agent verify documents and detect issues correctly?"""
    score = 0.0
    feedback_parts = []

    gt = trajectory.ground_truth
    if not gt:
        return DimensionScore(name="verification", score=0.5, weight=0.0, feedback="No ground truth")

    # Check if required checkpoints of type COMPARE_DOCUMENTS or CHECK_PREREQUISITES were hit
    required_diagnostic_checkpoints = [
        cp for cp in gt.required_checkpoints
        if cp.action_type in DIAGNOSTIC_ACTIONS
    ]

    if required_diagnostic_checkpoints:
        hit = 0
        actions_taken = {a["action"] for a in task_state.action_history if a.get("success", False)}

        for cp in required_diagnostic_checkpoints:
            if cp.action_type.value in actions_taken:
                hit += 1

        rate = hit / len(required_diagnostic_checkpoints)
        score = rate
        feedback_parts.append(f"Hit {hit}/{len(required_diagnostic_checkpoints)} required verification checkpoints")
    else:
        score = 0.8
        feedback_parts.append("No specific verification checkpoints required")

    return DimensionScore(
        name="verification",
        score=min(score, 1.0),
        weight=0.0,
        feedback="; ".join(feedback_parts),
    )


def _score_execution(task_state: Any, trajectory: Trajectory) -> DimensionScore:
    """Were forms filled correctly, right fees paid, correct documents submitted?"""
    score = 0.0
    feedback_parts = []
    sub_scores = []

    gt = trajectory.ground_truth
    if not gt:
        return DimensionScore(name="execution", score=0.5, weight=0.0, feedback="No ground truth")

    # Check form accuracy
    form_data = getattr(task_state, "form_data", None) or getattr(task_state, "ll_form_data", {})
    form_errors = getattr(task_state, "form_errors", [])

    if gt.correct_form_values:
        if form_data:
            correct_fields = 0
            total_fields = len(gt.correct_form_values)
            for field, expected in gt.correct_form_values.items():
                submitted = form_data.get(field, "")
                if str(submitted).strip().lower() == str(expected).strip().lower():
                    correct_fields += 1

            if total_fields > 0:
                form_score = correct_fields / total_fields
                sub_scores.append(form_score)
                feedback_parts.append(f"Form: {correct_fields}/{total_fields} fields correct")

            if form_errors:
                penalty = min(len(form_errors) * 0.15, 0.5)
                sub_scores.append(max(0, 1.0 - penalty))
                feedback_parts.append(f"Form errors: {len(form_errors)}")
            else:
                sub_scores.append(1.0)
        else:
            sub_scores.append(0.0)
            feedback_parts.append("Form not filled")
    else:
        sub_scores.append(0.8)
        feedback_parts.append("No form required")

    # Check fee accuracy
    if gt.correct_fee is not None:
        # DL task has two-phase fees (LL + DL)
        ll_fee = getattr(task_state, "ll_fee_amount", None)
        dl_fee = getattr(task_state, "dl_fee_amount", None)
        single_fee = getattr(task_state, "fee_amount", None)

        if ll_fee is not None or dl_fee is not None:
            # Two-phase fee task (driving licence)
            total_paid = (ll_fee or 0) + (dl_fee or 0)
            if total_paid == gt.correct_fee:
                sub_scores.append(1.0)
                feedback_parts.append(f"Correct total fee paid (LL={ll_fee}, DL={dl_fee})")
            elif total_paid > 0:
                sub_scores.append(0.5)
                feedback_parts.append(f"Fee mismatch: paid {total_paid}, expected {gt.correct_fee}")
            else:
                sub_scores.append(0.0)
                feedback_parts.append("Fee not paid")
        elif single_fee is not None:
            if single_fee == gt.correct_fee:
                sub_scores.append(1.0)
                feedback_parts.append("Correct fee paid")
            else:
                sub_scores.append(0.5)
                feedback_parts.append(f"Wrong fee: paid {single_fee}, expected {gt.correct_fee}")
        else:
            fee_paid = getattr(task_state, "fee_paid", False)
            if fee_paid:
                # Fee was paid but amount not tracked — partial credit
                sub_scores.append(0.7)
                feedback_parts.append("Fee paid but amount not verified")
            else:
                sub_scores.append(0.0)
                feedback_parts.append("Fee not paid")

    # Check task completion
    task_completed = getattr(task_state, "done", False)
    if task_completed:
        sub_scores.append(1.0)
        feedback_parts.append("Task completed")
    else:
        sub_scores.append(0.0)
        feedback_parts.append("Task NOT completed")

    if sub_scores:
        score = sum(sub_scores) / len(sub_scores)

    return DimensionScore(
        name="execution",
        score=min(score, 1.0),
        weight=0.0,
        feedback="; ".join(feedback_parts),
    )


def _score_recovery(task_state: Any, trajectory: Trajectory) -> DimensionScore:
    """Did agent handle failures and resolve them?"""
    score = 0.0
    feedback_parts = []

    gt = trajectory.ground_truth
    if not gt:
        return DimensionScore(name="recovery", score=0.5, weight=0.0, feedback="No ground truth")

    expected_issues = gt.expected_issues
    if not expected_issues:
        # No issues to recover from — full credit if task completed
        if getattr(task_state, "done", False):
            score = 1.0
            feedback_parts.append("No recovery needed — clean run")
        else:
            score = 0.5
            feedback_parts.append("No issues but task not completed")
    else:
        resolved = 0
        total = 0

        for issue in expected_issues:
            # Runtime events (test failures) — check if retake happened
            if issue in ("written_test_fail", "driving_test_fail"):
                total += 1
                test_passed = False
                if issue == "written_test_fail":
                    test_passed = getattr(task_state, "written_test_passed", False)
                else:
                    test_passed = getattr(task_state, "driving_test_passed", False)
                if test_passed:
                    resolved += 1
                continue

            if issue == "dl_applied_too_early":
                total += 1
                if getattr(task_state, "waited_for_practice", False):
                    resolved += 1
                continue

            if issue == "ll_expired":
                total += 1
                # Hard to recover from — agent gets partial credit for detecting it
                continue

            # Pre-detectable issues — check if fixed
            total += 1
            fixed = False
            if issue in ("name_mismatch", "name_mismatch_docs"):
                fixed = getattr(task_state, "name_fixed", False)
            elif issue in ("dob_mismatch",):
                fixed = getattr(task_state, "dob_fixed", False)
            elif issue in ("aadhaar_address_outdated", "address_mismatch"):
                fixed = getattr(task_state, "aadhaar_address_fixed", False) or \
                        getattr(task_state, "address_fixed", False)
            elif issue == "address_proof_invalid":
                fixed = getattr(task_state, "address_proof_fixed", False)
            elif issue == "photo_rejected":
                fixed = getattr(task_state, "photo_fixed", False)
            elif issue == "underage_for_class":
                fixed = getattr(task_state, "vehicle_class_corrected", False)
            elif issue == "medical_cert_missing":
                fixed = getattr(task_state, "medical_obtained", False)
            elif issue == "already_linked":
                fixed = getattr(task_state, "already_linked_detected", False)
            elif issue == "fee_not_paid":
                fixed = getattr(task_state, "fee_paid", False)
            # Vehicle Registration issues
            elif issue == "insurance_expired":
                fixed = getattr(task_state, "insurance_fixed", False)
            elif issue == "missing_puc":
                fixed = getattr(task_state, "puc_obtained", False)
            elif issue == "invoice_discrepancy":
                fixed = getattr(task_state, "invoice_fixed", False)
            elif issue == "hypothecation_required":
                fixed = getattr(task_state, "bank_noc_obtained", False)
            elif issue in ("chassis_mismatch", "inspection_failure"):
                fixed = getattr(task_state, "inspection_issue_fixed", False)
            elif issue == "address_mismatch_vr":
                fixed = getattr(task_state, "address_fixed", False)
            elif issue == "photo_invalid":
                fixed = getattr(task_state, "photo_fixed", False)

            if fixed:
                resolved += 1

        if total > 0:
            score = resolved / total
            feedback_parts.append(f"Resolved {resolved}/{total} issues")
        else:
            score = 0.8

    return DimensionScore(
        name="recovery",
        score=min(score, 1.0),
        weight=0.0,
        feedback="; ".join(feedback_parts) if feedback_parts else "N/A",
    )


def _score_efficiency(task_state: Any, trajectory: Trajectory) -> DimensionScore:
    """How many steps vs optimal? Penalize wasted actions."""
    gt = trajectory.ground_truth
    if not gt:
        return DimensionScore(name="efficiency", score=0.5, weight=0.0, feedback="No ground truth")

    steps_taken = task_state.steps_taken
    optimal = gt.optimal_steps
    invalid_actions = getattr(task_state, "invalid_action_count", 0)

    if steps_taken == 0:
        return DimensionScore(name="efficiency", score=0.0, weight=0.0, feedback="No steps taken")

    # Ratio-based scoring: optimal/actual, capped at 1.0
    if steps_taken <= optimal:
        ratio_score = 1.0
    else:
        ratio_score = max(0.0, optimal / steps_taken)

    # Penalty for invalid actions
    invalid_penalty = min(invalid_actions * 0.1, 0.4)

    score = max(0.0, ratio_score - invalid_penalty)

    feedback = f"Steps: {steps_taken} (optimal: {optimal}), Invalid actions: {invalid_actions}"

    return DimensionScore(
        name="efficiency",
        score=min(score, 1.0),
        weight=0.0,
        feedback=feedback,
    )


def _score_safety(task_state: Any, trajectory: Trajectory) -> DimensionScore:
    """Did agent avoid harmful/forbidden actions?"""
    gt = trajectory.ground_truth
    violations = getattr(task_state, "forbidden_violations", [])

    if not violations:
        return DimensionScore(
            name="safety",
            score=1.0,
            weight=0.0,
            feedback="No safety violations",
        )

    # Each violation costs 0.3, minimum score 0.0
    unique_violations = list(set(violations))
    penalty = len(unique_violations) * 0.3
    score = max(0.0, 1.0 - penalty)

    return DimensionScore(
        name="safety",
        score=score,
        weight=0.0,
        feedback=f"Violations: {', '.join(unique_violations)}",
    )


# ──────────────────────────────────────────────────────────────────────
# MAIN GRADING FUNCTION
# ──────────────────────────────────────────────────────────────────────

def grade_trajectory(task_id: TaskId, task_state: Any, trajectory: Trajectory) -> Reward:
    """
    Grade an agent's full trajectory.

    Computes 7 dimension scores with task-specific weights.
    Returns Reward with aggregate score (0.0-0.95) and breakdown.
    """
    weights = TASK_WEIGHTS.get(task_id, TASK_WEIGHTS[TaskId.PAN_AADHAAR_LINK])

    # Compute each dimension
    dims = {
        "diagnosis": _score_diagnosis(task_state, trajectory),
        "planning": _score_planning(task_state, trajectory),
        "verification": _score_verification(task_state, trajectory),
        "execution": _score_execution(task_state, trajectory),
        "recovery": _score_recovery(task_state, trajectory),
        "efficiency": _score_efficiency(task_state, trajectory),
        "safety": _score_safety(task_state, trajectory),
    }

    # Apply weights
    weighted_score = 0.0
    breakdown = []
    for dim_name, dim_score in dims.items():
        w = weights.get(dim_name, 0.1)
        weighted_dim = DimensionScore(
            name=dim_score.name,
            score=dim_score.score,
            weight=w,
            feedback=dim_score.feedback,
        )
        breakdown.append(weighted_dim)
        weighted_score += dim_score.score * w

    # Cap at 0.95
    final_score = min(round(weighted_score, 4), 0.95)

    # Check task completion
    task_completed = getattr(task_state, "done", False)

    # Build feedback summary
    feedback_parts = [f"Task: {task_id.value}", f"Completed: {task_completed}"]
    for d in breakdown:
        feedback_parts.append(f"  {d.name}: {d.score:.2f} (weight: {d.weight:.2f}) — {d.feedback}")

    return Reward(
        score=final_score,
        breakdown=breakdown,
        feedback="\n".join(feedback_parts),
        trajectory_length=task_state.steps_taken,
        task_completed=task_completed,
    )
