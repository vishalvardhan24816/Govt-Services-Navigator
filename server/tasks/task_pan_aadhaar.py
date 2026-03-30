"""
Task 1 (EASY): PAN-Aadhaar Linking

Source: https://incometax.gov.in/iec/foportal/help/how-to-link-aadhaar
Verified: April 2026

Flow: Check prerequisites → detect mismatches → fix if needed → pay fee → submit link → verify
Complications: name mismatch, DOB mismatch, already linked, fee not paid
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from server.models import (
    Action,
    ActionType,
    CitizenProfile,
    Difficulty,
    DocumentInfo,
    DocumentStatus,
    EnvironmentState,
    GroundTruth,
    GroundTruthCheckpoint,
    Observation,
    ServiceStatus,
    TaskId,
)

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────

TASK_ID = TaskId.PAN_AADHAAR_LINK
DIFFICULTY = Difficulty.EASY
MAX_STEPS = 15
LINKING_FEE = 1000

FIRST_NAMES = ["Rajesh", "Priya", "Amit", "Sunita", "Vikram", "Neha", "Suresh", "Kavita", "Arun", "Deepa"]
LAST_NAMES = ["Kumar", "Sharma", "Singh", "Patel", "Verma", "Gupta", "Reddy", "Nair", "Joshi", "Rao"]
MIDDLE_NAMES = ["", "K", "S", "R", "M", "V"]

ADDRESSES = [
    "42, MG Road, Bangalore 560001",
    "15, Park Street, Kolkata 700016",
    "88, Nehru Place, New Delhi 110019",
    "33, Anna Salai, Chennai 600002",
    "7, FC Road, Pune 411004",
]


# ──────────────────────────────────────────────────────────────────────
# CITIZEN PROFILE GENERATION
# ──────────────────────────────────────────────────────────────────────

def _generate_pan_number(rng: random.Random) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(rng.choices(letters, k=3))
        + "P"
        + rng.choice(letters)
        + "".join(str(rng.randint(0, 9)) for _ in range(4))
        + rng.choice(letters)
    )


def _generate_aadhaar_number(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(12))


def generate_citizen(rng: random.Random) -> Tuple[CitizenProfile, Dict[str, Any]]:
    """
    Generate a citizen profile with potential complications.
    Returns (citizen, complications_dict).
    """
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    middle = rng.choice(MIDDLE_NAMES)

    full_name = f"{first} {middle} {last}".replace("  ", " ").strip()
    age = rng.randint(21, 60)
    gender = rng.choice(["Male", "Female"])
    address = rng.choice(ADDRESSES)
    pan_number = _generate_pan_number(rng)
    aadhaar_number = _generate_aadhaar_number(rng)
    dob = f"{rng.randint(1965, 2004)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"

    # Decide complications
    complications = {}
    roll = rng.random()

    if roll < 0.30:
        # Name mismatch: PAN has abbreviated/different name
        pan_name_variants = [
            f"{first} {last}",           # missing middle
            f"{first} {middle[0] if middle else ''} {last}".strip(),  # abbreviated middle
            f"{first.upper()} {last.upper()}",  # different casing with missing middle
            f"{first} {middle} {last}A",   # typo in last name
        ]
        pan_name = rng.choice(pan_name_variants)
        if pan_name == full_name:
            pan_name = f"{first.upper()} {last}"  # fallback: case difference ensures mismatch
        complications["name_mismatch"] = True
        complications["pan_name"] = pan_name
        complications["aadhaar_name"] = full_name
    else:
        complications["name_mismatch"] = False
        complications["pan_name"] = full_name
        complications["aadhaar_name"] = full_name

    if roll > 0.85:
        # DOB mismatch
        wrong_day = str(rng.randint(1, 28)).zfill(2)
        pan_dob = dob[:-2] + wrong_day
        if pan_dob == dob:
            pan_dob = dob[:-2] + ("01" if dob[-2:] != "01" else "15")
        complications["dob_mismatch"] = True
        complications["pan_dob"] = pan_dob
        complications["aadhaar_dob"] = dob
    else:
        complications["dob_mismatch"] = False
        complications["pan_dob"] = dob
        complications["aadhaar_dob"] = dob

    if roll > 0.92:
        complications["already_linked"] = True
    else:
        complications["already_linked"] = False

    # Build citizen
    citizen = CitizenProfile(
        name=full_name,
        age=age,
        gender=gender,
        current_address=address,
        identifiers={"pan_number": pan_number, "aadhaar_number": aadhaar_number},
        complication_flags={
            "name_mismatch": complications["name_mismatch"],
            "dob_mismatch": complications.get("dob_mismatch", False),
        },
        documents={
            "pan_card": DocumentInfo(
                doc_type="pan_card",
                status=DocumentStatus.PRESENT,
                fields={
                    "pan_number": pan_number,
                    "name": complications["pan_name"],
                    "dob": complications["pan_dob"],
                },
            ),
            "aadhaar_card": DocumentInfo(
                doc_type="aadhaar_card",
                status=DocumentStatus.PRESENT,
                fields={
                    "aadhaar_number": aadhaar_number,
                    "name": complications["aadhaar_name"],
                    "dob": complications["aadhaar_dob"],
                    "address": address,
                },
            ),
        },
    )

    return citizen, complications


# ──────────────────────────────────────────────────────────────────────
# GROUND TRUTH COMPUTATION
# ──────────────────────────────────────────────────────────────────────

def compute_ground_truth(citizen: CitizenProfile, complications: Dict[str, Any]) -> GroundTruth:
    """Precompute the correct solution path based on the citizen's situation."""
    citizen_id = str(uuid.uuid4())[:8]
    checkpoints: List[GroundTruthCheckpoint] = []
    forbidden: List[str] = []
    expected_issues: List[str] = []
    optimal_steps = 0

    # Step 1: Always check prerequisites first
    checkpoints.append(GroundTruthCheckpoint(
        step_id="check_prereqs",
        action_type=ActionType.CHECK_PREREQUISITES,
        description="Check if PAN and Aadhaar details match",
        required=True,
        order_matters=True,
    ))
    optimal_steps += 1

    # Step 2: If mismatch exists, must detect it
    if complications["name_mismatch"]:
        expected_issues.append("name_mismatch")
        checkpoints.append(GroundTruthCheckpoint(
            step_id="detect_name_mismatch",
            action_type=ActionType.COMPARE_DOCUMENTS,
            description="Compare PAN name vs Aadhaar name to detect mismatch",
            required=True,
            order_matters=True,
        ))
        checkpoints.append(GroundTruthCheckpoint(
            step_id="fix_name",
            action_type=ActionType.FIX_DOCUMENT,
            description="Fix name mismatch on either PAN or Aadhaar",
            required=True,
            order_matters=True,
        ))
        forbidden.append("ignore_mismatch")
        optimal_steps += 2

    if complications.get("dob_mismatch", False):
        expected_issues.append("dob_mismatch")
        checkpoints.append(GroundTruthCheckpoint(
            step_id="detect_dob_mismatch",
            action_type=ActionType.COMPARE_DOCUMENTS,
            description="Compare PAN DOB vs Aadhaar DOB",
            required=True,
        ))
        checkpoints.append(GroundTruthCheckpoint(
            step_id="fix_dob",
            action_type=ActionType.FIX_DOCUMENT,
            description="Fix DOB mismatch",
            required=True,
        ))
        optimal_steps += 2

    # Step 3: Pay fee
    checkpoints.append(GroundTruthCheckpoint(
        step_id="pay_fee",
        action_type=ActionType.PAY_FEE,
        description="Pay Rs.1000 linking fee",
        required=True,
    ))
    forbidden.append("skip_fee")
    optimal_steps += 1

    # Step 4: Submit link request
    checkpoints.append(GroundTruthCheckpoint(
        step_id="submit_link",
        action_type=ActionType.SUBMIT_APPLICATION,
        description="Submit PAN-Aadhaar link request",
        required=True,
        order_matters=True,
    ))
    forbidden.append("submit_without_check")
    optimal_steps += 1

    # Step 5: Verify
    checkpoints.append(GroundTruthCheckpoint(
        step_id="verify_link",
        action_type=ActionType.CHECK_STATUS,
        description="Verify linking status",
        required=True,
    ))
    optimal_steps += 1

    if complications.get("already_linked", False):
        expected_issues.append("already_linked")

    return GroundTruth(
        task_id=TASK_ID,
        citizen_id=citizen_id,
        required_checkpoints=checkpoints,
        forbidden_actions=forbidden,
        valid_completions=["pan_aadhaar_linked"],
        expected_issues=expected_issues,
        optimal_steps=optimal_steps,
        correct_form_values={
            "pan_number": citizen.identifiers["pan_number"],
            "aadhaar_number": citizen.identifiers["aadhaar_number"],
        },
        correct_fee=LINKING_FEE,
    )


# ──────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ──────────────────────────────────────────────────────────────────────

class PanAadhaarState:
    """Mutable internal state for a PAN-Aadhaar linking episode."""

    def __init__(self, citizen: CitizenProfile, complications: Dict[str, Any], ground_truth: GroundTruth):
        self.citizen = citizen
        self.complications = complications
        self.ground_truth = ground_truth
        self.citizen_id = ground_truth.citizen_id

        # State tracking
        self.prerequisites_checked = False
        self.documents_compared = False
        self.name_mismatch_detected = False
        self.dob_mismatch_detected = False
        self.name_fixed = False
        self.dob_fixed = False
        self.fee_paid = False
        self.fee_amount: Optional[float] = None
        self.link_submitted = False
        self.link_verified = False
        self.already_linked_detected = False
        self.options_evaluated = False

        # Tracking
        self.completed_steps: List[str] = []
        self.pending_issues: List[str] = []
        self.steps_taken = 0
        self.simulated_day = 0
        self.done = False
        self.diagnostic_before_execution = False

        # Per-step tracking for grader
        self.action_history: List[Dict[str, Any]] = []
        self.forbidden_violations: List[str] = []
        self.invalid_action_count = 0

    def get_services_status(self) -> Dict[str, str]:
        if self.link_verified:
            return {"pan_aadhaar_link": ServiceStatus.COMPLETED.value}
        elif self.link_submitted:
            return {"pan_aadhaar_link": ServiceStatus.IN_PROGRESS.value}
        elif self.pending_issues:
            return {"pan_aadhaar_link": ServiceStatus.BLOCKED.value}
        else:
            return {"pan_aadhaar_link": ServiceStatus.NOT_STARTED.value}

    def get_progress(self) -> float:
        """Calculate cumulative progress 0.0 to 1.0."""
        total_weight = 0.0
        if self.prerequisites_checked:
            total_weight += 0.15
        if self.documents_compared:
            total_weight += 0.10
        if self.complications["name_mismatch"] and self.name_mismatch_detected:
            total_weight += 0.10
        if self.complications["name_mismatch"] and self.name_fixed:
            total_weight += 0.15
        if self.complications.get("dob_mismatch") and self.dob_mismatch_detected:
            total_weight += 0.05
        if self.complications.get("dob_mismatch") and self.dob_fixed:
            total_weight += 0.10
        if self.fee_paid:
            total_weight += 0.15
        if self.link_submitted:
            total_weight += 0.15
        if self.link_verified:
            total_weight += 0.05

        # If no mismatch complications, redistribute weight
        if not self.complications["name_mismatch"] and not self.complications.get("dob_mismatch"):
            if self.prerequisites_checked:
                total_weight = 0.20
            if self.documents_compared:
                total_weight = 0.35
            if self.fee_paid:
                total_weight = 0.55
            if self.link_submitted:
                total_weight = 0.80
            if self.link_verified:
                total_weight = 1.00

        return min(total_weight, 1.0)


# ──────────────────────────────────────────────────────────────────────
# ACTION HANDLERS
# ──────────────────────────────────────────────────────────────────────

def _get_available_actions(state: PanAadhaarState) -> List[str]:
    """Return list of currently valid action types."""
    actions = []

    if not state.prerequisites_checked:
        actions.append(ActionType.CHECK_PREREQUISITES.value)
    if not state.documents_compared:
        actions.append(ActionType.COMPARE_DOCUMENTS.value)
    if not state.options_evaluated and (state.name_mismatch_detected or state.dob_mismatch_detected):
        actions.append(ActionType.EVALUATE_OPTIONS.value)

    if state.name_mismatch_detected and not state.name_fixed:
        actions.append(ActionType.FIX_DOCUMENT.value)
    if state.dob_mismatch_detected and not state.dob_fixed:
        actions.append(ActionType.FIX_DOCUMENT.value)

    if not state.fee_paid:
        actions.append(ActionType.PAY_FEE.value)
    if not state.link_submitted:
        actions.append(ActionType.SUBMIT_APPLICATION.value)

    actions.append(ActionType.CHECK_STATUS.value)
    actions.append(ActionType.CHECK_ELIGIBILITY.value)

    return list(set(actions))


def _build_citizen_summary(citizen: CitizenProfile, state: PanAadhaarState) -> str:
    """Build a clear text summary for the agent."""
    pan_fields = citizen.documents["pan_card"].fields
    aad_fields = citizen.documents["aadhaar_card"].fields

    lines = [
        f"Citizen: {citizen.name}, Age: {citizen.age}, Gender: {citizen.gender}",
        f"PAN Number: {pan_fields['pan_number']}",
        f"PAN Name: {pan_fields['name']}",
        f"PAN DOB: {pan_fields['dob']}",
        f"Aadhaar Number: {aad_fields['aadhaar_number']}",
        f"Aadhaar Name: {aad_fields['name']}",
        f"Aadhaar DOB: {aad_fields['dob']}",
        f"Address: {citizen.current_address}",
    ]

    if state.prerequisites_checked:
        lines.append(f"\n[Prerequisites checked]")
        if state.pending_issues:
            lines.append(f"Issues found: {', '.join(state.pending_issues)}")
        else:
            lines.append("No issues found.")

    if state.name_fixed:
        lines.append("[Name mismatch resolved]")
    if state.dob_fixed:
        lines.append("[DOB mismatch resolved]")
    if state.fee_paid:
        lines.append(f"[Fee of Rs.{LINKING_FEE} paid]")
    if state.link_submitted:
        lines.append("[Link request submitted]")
    if state.link_verified:
        lines.append("[Link verified successfully]")

    return "\n".join(lines)


def _build_action_hints(state: PanAadhaarState) -> str:
    """Phase-level status summary — describes WHERE in the process the citizen is.
    Does NOT prescribe specific actions. Agent reasons from available_actions,
    completed_steps, pending_issues, and services_status."""
    if not state.prerequisites_checked:
        return "Document verification phase."
    if state.pending_issues:
        return "Issue resolution phase."
    if not state.fee_paid or not state.link_submitted:
        return "Application submission phase."
    if not state.link_verified:
        return "Processing and verification phase."
    return "Process complete."


def handle_action(state: PanAadhaarState, action: Action) -> Tuple[str, bool, Optional[str]]:
    """
    Process an action and update state.
    Returns: (result_message, success, error_message)
    """
    state.steps_taken += 1
    action_record = {"step": state.steps_taken, "action": action.action_type.value, "params": action.parameters}

    at = action.action_type

    # ── CHECK_PREREQUISITES ──
    if at == ActionType.CHECK_PREREQUISITES:
        state.prerequisites_checked = True
        state.diagnostic_before_execution = True
        issues = []

        if state.complications["name_mismatch"] and not getattr(state, 'name_fixed', False):
            issues.append("NAME MISMATCH: PAN name and Aadhaar name do not match")
            if "name_mismatch" not in state.pending_issues:
                state.pending_issues.append("name_mismatch")

        if state.complications.get("dob_mismatch") and not getattr(state, 'dob_fixed', False):
            issues.append("DOB MISMATCH: PAN date of birth and Aadhaar date of birth do not match")
            if "dob_mismatch" not in state.pending_issues:
                state.pending_issues.append("dob_mismatch")

        if state.complications.get("already_linked"):
            issues.append("ALREADY LINKED: PAN may already be linked to a different Aadhaar")
            if "already_linked" not in state.pending_issues:
                state.pending_issues.append("already_linked")
            state.already_linked_detected = True

        state.completed_steps.append("check_prerequisites")
        action_record["success"] = True

        if issues:
            msg = "Prerequisites check complete. Issues found:\n" + "\n".join(f"  - {i}" for i in issues)
        else:
            msg = "Prerequisites check complete. PAN and Aadhaar details match. Ready to proceed with linking."

        state.action_history.append(action_record)
        return msg, True, None

    # ── COMPARE_DOCUMENTS ──
    if at == ActionType.COMPARE_DOCUMENTS:
        state.documents_compared = True
        state.diagnostic_before_execution = True
        pan = state.citizen.documents["pan_card"].fields
        aad = state.citizen.documents["aadhaar_card"].fields

        comparisons = [
            f"PAN Name: '{pan['name']}' vs Aadhaar Name: '{aad['name']}' → {'MATCH' if pan['name'] == aad['name'] else 'MISMATCH'}",
            f"PAN DOB: '{pan['dob']}' vs Aadhaar DOB: '{aad['dob']}' → {'MATCH' if pan['dob'] == aad['dob'] else 'MISMATCH'}",
        ]

        if pan["name"] != aad["name"]:
            state.name_mismatch_detected = True
        if pan["dob"] != aad["dob"]:
            state.dob_mismatch_detected = True

        state.completed_steps.append("compare_documents")
        action_record["success"] = True
        state.action_history.append(action_record)
        return "Document comparison:\n" + "\n".join(f"  {c}" for c in comparisons), True, None

    # ── EVALUATE_OPTIONS ──
    if at == ActionType.EVALUATE_OPTIONS:
        state.options_evaluated = True
        options = []
        if state.name_mismatch_detected and not state.name_fixed:
            options.append(
                "Option A: Update Aadhaar name (cost: Rs.50, time: ~7 days) via UIDAI\n"
                "Option B: Update PAN name (cost: Rs.107, time: ~15 days) via NSDL/UTIITSL"
            )
        if state.dob_mismatch_detected and not state.dob_fixed:
            options.append(
                "Option C: Update Aadhaar DOB (cost: Rs.50, time: ~10 days, needs birth certificate)\n"
                "Option D: Update PAN DOB (cost: Rs.107, time: ~15 days)"
            )

        state.completed_steps.append("evaluate_options")
        action_record["success"] = True
        state.action_history.append(action_record)

        if options:
            return "Available resolution options:\n" + "\n".join(options), True, None
        return "No issues to resolve. Proceed with linking.", True, None

    # ── CHECK_ELIGIBILITY ──
    if at == ActionType.CHECK_ELIGIBILITY:
        state.diagnostic_before_execution = True
        msg = (
            "Eligibility check:\n"
            f"  - Has valid PAN: Yes ({state.citizen.identifiers['pan_number']})\n"
            f"  - Has valid Aadhaar: Yes ({state.citizen.identifiers['aadhaar_number']})\n"
            f"  - PAN-Aadhaar linking is mandatory under Section 139AA\n"
            f"  - Deadline: PAN becomes inoperative if not linked\n"
            f"  - Fee: Rs.{LINKING_FEE}"
        )
        state.completed_steps.append("check_eligibility")
        action_record["success"] = True
        state.action_history.append(action_record)
        return msg, True, None

    # ── FIX_DOCUMENT ──
    if at == ActionType.FIX_DOCUMENT:
        target = action.parameters.get("target", "")

        if target in ("aadhaar_name", "pan_name"):
            if not state.name_mismatch_detected and not state.prerequisites_checked:
                state.forbidden_violations.append("fix_without_diagnosis")
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "Prerequisites have not been verified and documents have not been compared."

            correct_name = state.citizen.name
            if target == "aadhaar_name":
                state.citizen.documents["aadhaar_card"].fields["name"] = correct_name
                cost, days = 50, 7
            else:
                state.citizen.documents["pan_card"].fields["name"] = correct_name
                cost, days = 107, 15

            state.name_fixed = True
            state.simulated_day += days
            if "name_mismatch" in state.pending_issues:
                state.pending_issues.remove("name_mismatch")
            state.completed_steps.append(f"fix_{target}")
            action_record["success"] = True
            state.action_history.append(action_record)
            return (
                f"Name updated on {'Aadhaar' if 'aadhaar' in target else 'PAN'} to '{correct_name}'. "
                f"Cost: Rs.{cost}. Time: {days} days elapsed. Names now match."
            ), True, None

        if target in ("aadhaar_dob", "pan_dob"):
            if not state.dob_mismatch_detected and not state.prerequisites_checked:
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "No date-of-birth discrepancy has been detected."

            correct_dob = state.complications["aadhaar_dob"]
            if target == "aadhaar_dob":
                state.citizen.documents["aadhaar_card"].fields["dob"] = correct_dob
                cost, days = 50, 10
            else:
                state.citizen.documents["pan_card"].fields["dob"] = correct_dob
                cost, days = 107, 15

            state.dob_fixed = True
            state.simulated_day += days
            if "dob_mismatch" in state.pending_issues:
                state.pending_issues.remove("dob_mismatch")
            state.completed_steps.append(f"fix_{target}")
            action_record["success"] = True
            state.action_history.append(action_record)
            return (
                f"DOB updated on {'Aadhaar' if 'aadhaar' in target else 'PAN'} to '{correct_dob}'. "
                f"Cost: Rs.{cost}. Time: {days} days elapsed."
            ), True, None

        action_record["success"] = False
        state.invalid_action_count += 1
        state.action_history.append(action_record)
        return "", False, f"Unknown fix target: '{target}'. Valid targets: aadhaar_name, pan_name, aadhaar_dob, pan_dob"

    # ── PAY_FEE ──
    if at == ActionType.PAY_FEE:
        if state.fee_paid:
            action_record["success"] = False
            state.invalid_action_count += 1
            state.action_history.append(action_record)
            return "", False, "Fee already paid."

        amount = action.parameters.get("amount", LINKING_FEE)
        state.fee_paid = True
        state.fee_amount = amount
        state.completed_steps.append("pay_fee")
        action_record["success"] = True
        state.action_history.append(action_record)
        return f"Payment of Rs.{amount} for PAN-Aadhaar linking fee processed successfully via Challan 280.", True, None

    # ── SUBMIT_APPLICATION ──
    if at == ActionType.SUBMIT_APPLICATION:
        # Check preconditions
        if not state.prerequisites_checked:
            state.forbidden_violations.append("submit_without_check")
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Prerequisites have not been verified."

        if state.complications["name_mismatch"] and not state.name_fixed:
            state.forbidden_violations.append("ignore_mismatch")
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Submission blocked: name discrepancy between PAN and Aadhaar is unresolved."

        if state.complications.get("dob_mismatch") and not state.dob_fixed:
            state.forbidden_violations.append("ignore_mismatch")
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Submission blocked: date-of-birth discrepancy between PAN and Aadhaar is unresolved."

        if not state.fee_paid:
            state.forbidden_violations.append("skip_fee")
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Submission blocked: linking fee has not been paid."

        if state.already_linked_detected:
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Submission blocked: PAN is already linked to another Aadhaar."

        state.link_submitted = True
        state.simulated_day += 1
        state.completed_steps.append("submit_link")
        action_record["success"] = True
        state.action_history.append(action_record)
        return (
            f"PAN-Aadhaar link request submitted successfully.\n"
            f"PAN: {state.citizen.identifiers['pan_number']}, Aadhaar: {state.citizen.identifiers['aadhaar_number']}\n"
            f"Processing time: 3-7 business days."
        ), True, None

    # ── CHECK_STATUS ──
    if at == ActionType.CHECK_STATUS:
        if state.link_submitted and not state.link_verified:
            state.link_verified = True
            state.simulated_day += 5
            state.done = True
            state.completed_steps.append("verify_link")
            action_record["success"] = True
            state.action_history.append(action_record)
            return (
                f"PAN-Aadhaar linking status: SUCCESSFUL\n"
                f"PAN {state.citizen.identifiers['pan_number']} is now linked to Aadhaar {state.citizen.identifiers['aadhaar_number']}.\n"
                f"PAN is active and operative. Task complete."
            ), True, None

        if state.link_verified:
            action_record["success"] = True
            state.action_history.append(action_record)
            return "PAN-Aadhaar already linked and verified. No further action needed.", True, None

        status_msg = "Current status:\n"
        status_msg += f"  - Prerequisites checked: {'Yes' if state.prerequisites_checked else 'No'}\n"
        status_msg += f"  - Documents compared: {'Yes' if state.documents_compared else 'No'}\n"
        status_msg += f"  - Fee paid: {'Yes' if state.fee_paid else 'No'}\n"
        status_msg += f"  - Link submitted: {'Yes' if state.link_submitted else 'No'}\n"
        if state.pending_issues:
            status_msg += f"  - Pending issues: {', '.join(state.pending_issues)}\n"
        action_record["success"] = True
        state.action_history.append(action_record)
        return status_msg, True, None

    # ── UNKNOWN / INVALID ACTION ──
    state.invalid_action_count += 1
    action_record["success"] = False
    state.action_history.append(action_record)
    return "", False, f"Action '{at.value}' is not applicable to PAN-Aadhaar linking task."


# ──────────────────────────────────────────────────────────────────────
# OBSERVATION BUILDER
# ──────────────────────────────────────────────────────────────────────

def build_observation(state: PanAadhaarState, result_msg: str, success: bool, error: Optional[str]) -> Observation:
    """Build an Observation from current state."""
    doc_summary = {}
    for doc_id, doc in state.citizen.documents.items():
        doc_summary[doc_id] = f"{doc.status.value} — {doc.fields}"

    return Observation(
        task_id=TASK_ID,
        task_description=(
            "Help the citizen link their PAN card with Aadhaar card on the Income Tax portal. "
            "This is mandatory under Section 139AA — PAN becomes inoperative if not linked. "
            "Identity documents may contain discrepancies that must be resolved for successful linking."
        ),
        difficulty=DIFFICULTY,
        citizen_summary=_build_citizen_summary(state.citizen, state),
        citizen_documents=doc_summary,
        current_phase="pan_aadhaar_linking",
        services_status=state.get_services_status(),
        completed_steps=list(state.completed_steps),
        pending_issues=list(state.pending_issues),
        last_action=state.action_history[-1]["action"] if state.action_history else None,
        last_action_result=result_msg,
        last_action_success=success,
        last_action_error=error,
        available_actions=_get_available_actions(state),
        status_summary=_build_action_hints(state),
        progress_pct=state.get_progress(),
        steps_taken=state.steps_taken,
        max_steps=MAX_STEPS,
        simulated_day=state.simulated_day,
    )


def build_initial_observation(state: PanAadhaarState) -> Observation:
    """Build the initial observation returned by reset()."""
    return build_observation(
        state,
        result_msg="Episode started. You are helping a citizen link their PAN with Aadhaar.",
        success=True,
        error=None,
    )
