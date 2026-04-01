"""
Task 3 (HARD): Driving Licence (Learner's Licence → Permanent DL)

Source: https://parivahan.gov.in/parivahan/
Verified: April 2026

Two-phase flow:
  Phase 1: LL — check age/docs → fill Form 2 → pay → book RTO → take written test → get LL
  Phase 2: DL — wait 30+ days → verify LL valid (< 6 months) → fill Form 4 → pay → book → driving test → get DL

Complications: underage for class, written test fail, driving test fail, LL expired,
               applied too early (< 30 days), medical cert missing (40+), address mismatch
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
    GroundTruth,
    GroundTruthCheckpoint,
    Observation,
    ServiceStatus,
    TaskId,
)

TASK_ID = TaskId.DRIVING_LICENCE
DIFFICULTY = Difficulty.HARD
MAX_STEPS = 30
LL_FEE = 200
DL_FEE = 300
LL_VALIDITY_DAYS = 180
MANDATORY_WAIT_DAYS = 30

FIRST_NAMES = ["Aditya", "Pooja", "Rohit", "Shruti", "Nikhil", "Swati", "Gaurav", "Anita", "Sanjay", "Rekha"]
LAST_NAMES = ["Mishra", "Tiwari", "Pandey", "Agarwal", "Saxena", "Kapoor", "Malhotra", "Mehra", "Chopra", "Arora"]

ADDRESSES = [
    "24, Sector 18, Noida 201301",
    "78, MG Road, Gurgaon 122001",
    "15, Civil Lines, Lucknow 226001",
    "33, Rajouri Garden, Delhi 110027",
    "52, Aundh, Pune 411007",
]

OLD_ADDRESSES = [
    "11, Laxmi Nagar, Delhi 110092",
    "45, Vaishali, Ghaziabad 201010",
    "29, Gomti Nagar, Lucknow 226010",
    "88, Dwarka, Delhi 110075",
    "62, Kothrud, Pune 411038",
]

VEHICLE_CLASSES = ["two_wheeler_gearless", "two_wheeler_geared", "four_wheeler"]


def _generate_aadhaar_number(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(12))


def generate_citizen(rng: random.Random) -> Tuple[CitizenProfile, Dict[str, Any]]:
    """Generate a citizen profile with DL-specific complications."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    full_name = f"{first} {last}"

    # Age distribution: include some edge cases
    age_roll = rng.random()
    if age_roll < 0.15:
        age = rng.randint(16, 17)  # minor — only gearless two-wheeler
    elif age_roll < 0.85:
        age = rng.randint(18, 35)  # standard
    else:
        age = rng.randint(40, 55)  # needs medical cert

    gender = rng.choice(["Male", "Female"])
    address = rng.choice(ADDRESSES)
    aadhaar_number = _generate_aadhaar_number(rng)
    dob = f"{2026 - age}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"

    # Vehicle class selection
    if age < 18:
        desired_class = rng.choice(["two_wheeler_gearless", "two_wheeler_geared"])  # may pick wrong one
    else:
        desired_class = rng.choice(VEHICLE_CLASSES)

    complications: Dict[str, Any] = {
        "underage_for_class": age < 18 and desired_class != "two_wheeler_gearless",
        "needs_guardian_consent": age < 18,
        "needs_medical_cert": age >= 40,
        "has_medical_cert": age >= 40 and rng.random() > 0.4,  # 40% chance missing
        "address_mismatch": rng.random() < 0.20,
        "will_fail_written_test": rng.random() < 0.20,
        "will_fail_driving_test": rng.random() < 0.25,
        "ll_timing": "normal",  # can be "too_early" or "expired"
        "desired_class": desired_class,
        "dob": dob,
    }

    # LL timing complication
    timing_roll = rng.random()
    if timing_roll < 0.15:
        complications["ll_timing"] = "too_early"  # agent tries DL before 30 days
    elif timing_roll < 0.25:
        complications["ll_timing"] = "expired"  # LL will be > 180 days old

    aadhaar_address = rng.choice(OLD_ADDRESSES) if complications["address_mismatch"] else address

    citizen = CitizenProfile(
        name=full_name,
        age=age,
        gender=gender,
        current_address=address,
        identifiers={"aadhaar_number": aadhaar_number},
        attributes={"vehicle_class": desired_class},
        complication_flags={
            "address_mismatch": complications["address_mismatch"],
            "missing_document": complications["needs_medical_cert"] and not complications["has_medical_cert"],
        },
        documents={
            "aadhaar_card": DocumentInfo(
                doc_type="aadhaar_card",
                status=DocumentStatus.PRESENT,
                fields={
                    "aadhaar_number": aadhaar_number,
                    "name": full_name,
                    "address": aadhaar_address,
                    "dob": dob,
                },
            ),
            "age_proof": DocumentInfo(
                doc_type="age_proof",
                status=DocumentStatus.PRESENT,
                fields={"dob": dob, "type": "10th_marksheet"},
            ),
            "passport_photos": DocumentInfo(
                doc_type="passport_photos",
                status=DocumentStatus.PRESENT,
                fields={"count": 4},
            ),
            "medical_certificate": DocumentInfo(
                doc_type="medical_certificate",
                status=DocumentStatus.PRESENT if complications.get("has_medical_cert") else DocumentStatus.MISSING,
                fields={"required": age >= 40},
            ),
            "guardian_consent": DocumentInfo(
                doc_type="guardian_consent",
                status=DocumentStatus.PRESENT if age < 18 else DocumentStatus.MISSING,
                fields={"required": age < 18},
            ),
        },
    )

    return citizen, complications


def compute_ground_truth(citizen: CitizenProfile, complications: Dict[str, Any]) -> GroundTruth:
    """Precompute correct solution for driving licence."""
    citizen_id = str(uuid.uuid4())[:8]
    checkpoints: List[GroundTruthCheckpoint] = []
    forbidden: List[str] = []
    expected_issues: List[str] = []
    optimal_steps = 0

    # Phase 1: Learner's Licence
    checkpoints.append(GroundTruthCheckpoint(
        step_id="check_prereqs_ll", action_type=ActionType.CHECK_PREREQUISITES,
        description="Check age, documents, vehicle class eligibility", required=True, order_matters=True,
    ))
    optimal_steps += 1

    if complications["underage_for_class"]:
        expected_issues.append("underage_for_class")
        forbidden.append("wrong_vehicle_class_underage")

    if complications["address_mismatch"]:
        expected_issues.append("address_mismatch")
        checkpoints.append(GroundTruthCheckpoint(
            step_id="fix_address", action_type=ActionType.FIX_DOCUMENT,
            description="Fix Aadhaar address", required=True,
        ))
        optimal_steps += 1

    if complications["needs_medical_cert"] and not complications["has_medical_cert"]:
        expected_issues.append("medical_cert_missing")
        checkpoints.append(GroundTruthCheckpoint(
            step_id="get_medical", action_type=ActionType.GATHER_DOCUMENT,
            description="Get medical certificate", required=True,
        ))
        forbidden.append("submit_without_medical_40plus")
        optimal_steps += 1

    checkpoints.append(GroundTruthCheckpoint(
        step_id="fill_form_ll", action_type=ActionType.FILL_FORM,
        description="Fill LL application (Form 2)", required=True, order_matters=True,
    ))
    optimal_steps += 1

    checkpoints.append(GroundTruthCheckpoint(
        step_id="pay_fee_ll", action_type=ActionType.PAY_FEE,
        description="Pay LL fee Rs.200", required=True,
    ))
    optimal_steps += 1

    checkpoints.append(GroundTruthCheckpoint(
        step_id="book_slot_ll", action_type=ActionType.BOOK_APPOINTMENT,
        description="Book RTO slot for written test", required=True,
    ))
    optimal_steps += 1

    checkpoints.append(GroundTruthCheckpoint(
        step_id="take_written", action_type=ActionType.TAKE_TEST,
        description="Take written test at RTO", required=True, order_matters=True,
    ))
    forbidden.append("skip_written_test")
    optimal_steps += 1

    if complications["will_fail_written_test"]:
        expected_issues.append("written_test_fail")
        optimal_steps += 2  # retake

    # Phase 2: Permanent DL
    checkpoints.append(GroundTruthCheckpoint(
        step_id="check_ll_validity", action_type=ActionType.CHECK_PREREQUISITES,
        description="Verify LL is 30+ days old and not expired", required=True, order_matters=True,
    ))
    optimal_steps += 1

    if complications["ll_timing"] == "too_early":
        expected_issues.append("dl_applied_too_early")
        forbidden.append("apply_dl_before_30_days")

    if complications["ll_timing"] == "expired":
        expected_issues.append("ll_expired")

    checkpoints.append(GroundTruthCheckpoint(
        step_id="fill_form_dl", action_type=ActionType.FILL_FORM,
        description="Fill DL application (Form 4)", required=True, order_matters=True,
    ))
    optimal_steps += 1

    checkpoints.append(GroundTruthCheckpoint(
        step_id="pay_fee_dl", action_type=ActionType.PAY_FEE,
        description="Pay DL fee Rs.300", required=True,
    ))
    optimal_steps += 1

    checkpoints.append(GroundTruthCheckpoint(
        step_id="book_slot_dl", action_type=ActionType.BOOK_APPOINTMENT,
        description="Book RTO slot for driving test", required=True,
    ))
    optimal_steps += 1

    checkpoints.append(GroundTruthCheckpoint(
        step_id="take_driving", action_type=ActionType.TAKE_TEST,
        description="Take practical driving test", required=True, order_matters=True,
    ))
    forbidden.append("apply_dl_without_ll")
    optimal_steps += 1

    if complications["will_fail_driving_test"]:
        expected_issues.append("driving_test_fail")
        optimal_steps += 2

    checkpoints.append(GroundTruthCheckpoint(
        step_id="receive_dl", action_type=ActionType.CHECK_STATUS,
        description="Receive smart-card DL", required=True,
    ))
    optimal_steps += 1

    vehicle_class = complications["desired_class"]
    if complications["underage_for_class"]:
        vehicle_class = "two_wheeler_gearless"

    return GroundTruth(
        task_id=TASK_ID,
        citizen_id=citizen_id,
        required_checkpoints=checkpoints,
        forbidden_actions=forbidden,
        valid_completions=["dl_received"],
        expected_issues=expected_issues,
        optimal_steps=optimal_steps,
        correct_form_values={
            "applicant_name": citizen.name,
            "dob": complications["dob"],
            "address": citizen.current_address,
            "vehicle_class": vehicle_class,
        },
        correct_fee=LL_FEE + DL_FEE,
    )


class DrivingLicenceState:
    """Mutable internal state for a driving licence episode."""

    def __init__(self, citizen: CitizenProfile, complications: Dict[str, Any], ground_truth: GroundTruth):
        self.citizen = citizen
        self.complications = complications
        self.ground_truth = ground_truth
        self.citizen_id = ground_truth.citizen_id

        # Current phase
        self.current_phase = "learner_licence"  # or "driving_licence"

        # Phase 1: LL state
        self.ll_prereqs_checked = False
        self.ll_documents_compared = False
        self.ll_options_evaluated = False
        self.address_mismatch_detected = False
        self.address_fixed = False
        self.medical_missing_detected = False
        self.medical_obtained = False
        self.underage_detected = False
        self.vehicle_class_corrected = False
        self.ll_form_filled = False
        self.ll_form_data: Dict[str, Any] = {}
        self.ll_fee_paid = False
        self.ll_fee_amount: Optional[float] = None
        self.ll_appointment_booked = False
        self.written_test_taken = False
        self.written_test_passed = False
        self.written_test_attempts = 0
        self.ll_received = False
        self.ll_issue_day = 0

        # Phase 2: DL state
        self.dl_prereqs_checked = False
        self.dl_form_filled = False
        self.dl_form_data: Dict[str, Any] = {}
        self.dl_fee_paid = False
        self.dl_fee_amount: Optional[float] = None
        self.dl_appointment_booked = False
        self.driving_test_taken = False
        self.driving_test_passed = False
        self.driving_test_attempts = 0
        self.dl_received = False
        self.waited_for_practice = False

        # Tracking
        self.completed_steps: List[str] = []
        self.pending_issues: List[str] = []
        self.steps_taken = 0
        self.simulated_day = 0
        self.done = False
        self.diagnostic_before_execution = False
        self.action_history: List[Dict[str, Any]] = []
        self.forbidden_violations: List[str] = []
        self.invalid_action_count = 0

    def get_services_status(self) -> Dict[str, str]:
        if self.dl_received:
            return {"learner_licence": ServiceStatus.COMPLETED.value, "driving_licence": ServiceStatus.COMPLETED.value}
        elif self.current_phase == "driving_licence":
            return {"learner_licence": ServiceStatus.COMPLETED.value, "driving_licence": ServiceStatus.IN_PROGRESS.value}
        elif self.ll_received:
            return {"learner_licence": ServiceStatus.COMPLETED.value, "driving_licence": ServiceStatus.NOT_STARTED.value}
        elif self.pending_issues:
            return {"learner_licence": ServiceStatus.BLOCKED.value, "driving_licence": ServiceStatus.NOT_STARTED.value}
        else:
            return {"learner_licence": ServiceStatus.IN_PROGRESS.value, "driving_licence": ServiceStatus.NOT_STARTED.value}

    def get_progress(self) -> float:
        score = 0.0
        # Phase 1 = 50% of total
        if self.ll_prereqs_checked: score += 0.05
        if self.address_fixed or not self.complications["address_mismatch"]: score += 0.03
        if self.medical_obtained or not (self.complications["needs_medical_cert"] and not self.complications["has_medical_cert"]): score += 0.03
        if self.vehicle_class_corrected or not self.complications["underage_for_class"]: score += 0.02
        if self.ll_form_filled: score += 0.07
        if self.ll_fee_paid: score += 0.05
        if self.ll_appointment_booked: score += 0.05
        if self.written_test_passed: score += 0.10
        if self.ll_received: score += 0.10

        # Phase 2 = 50% of total
        if self.dl_prereqs_checked: score += 0.05
        if self.waited_for_practice or self.complications["ll_timing"] == "normal": score += 0.05
        if self.dl_form_filled: score += 0.07
        if self.dl_fee_paid: score += 0.05
        if self.dl_appointment_booked: score += 0.05
        if self.driving_test_passed: score += 0.13
        if self.dl_received: score += 0.10

        return min(score, 1.0)


def _get_available_actions(state: DrivingLicenceState) -> List[str]:
    actions = [ActionType.CHECK_STATUS.value, ActionType.CHECK_ELIGIBILITY.value]

    if state.current_phase == "learner_licence":
        if not state.ll_prereqs_checked:
            actions.append(ActionType.CHECK_PREREQUISITES.value)
        if not state.ll_documents_compared:
            actions.append(ActionType.COMPARE_DOCUMENTS.value)
        if state.pending_issues and not state.ll_options_evaluated:
            actions.append(ActionType.EVALUATE_OPTIONS.value)
        if state.address_mismatch_detected and not state.address_fixed:
            actions.append(ActionType.FIX_DOCUMENT.value)
        if state.medical_missing_detected and not state.medical_obtained:
            actions.append(ActionType.GATHER_DOCUMENT.value)
            actions.append(ActionType.FIX_DOCUMENT.value)
        if not state.ll_form_filled:
            actions.append(ActionType.FILL_FORM.value)
        if not state.ll_fee_paid:
            actions.append(ActionType.PAY_FEE.value)
        if not state.ll_appointment_booked:
            actions.append(ActionType.BOOK_APPOINTMENT.value)
        if state.ll_appointment_booked and not state.written_test_passed:
            actions.append(ActionType.TAKE_TEST.value)
    else:  # driving_licence phase
        if not state.dl_prereqs_checked:
            actions.append(ActionType.CHECK_PREREQUISITES.value)
        if not state.dl_form_filled:
            actions.append(ActionType.FILL_FORM.value)
        if not state.dl_fee_paid:
            actions.append(ActionType.PAY_FEE.value)
        if not state.dl_appointment_booked:
            actions.append(ActionType.BOOK_APPOINTMENT.value)
        if state.dl_appointment_booked and not state.driving_test_passed:
            actions.append(ActionType.TAKE_TEST.value)
        actions.append(ActionType.WAIT.value)

    return list(set(actions))


def _build_citizen_summary(citizen: CitizenProfile, state: DrivingLicenceState) -> str:
    aad = citizen.documents["aadhaar_card"].fields
    lines = [
        f"Citizen: {citizen.name}, Age: {citizen.age}, Gender: {citizen.gender}",
        f"Aadhaar: {aad['aadhaar_number']}, Address: {aad['address']}",
        f"Current Address: {citizen.current_address}",
        f"DOB: {aad['dob']}",
        f"Desired Vehicle Class: {citizen.attributes.get('vehicle_class', 'unknown')}",
        f"Current Phase: {state.current_phase.replace('_', ' ').title()}",
        f"Simulated Day: {state.simulated_day}",
    ]

    if state.ll_received:
        lines.append(f"[LL received on Day {state.ll_issue_day}, valid until Day {state.ll_issue_day + LL_VALIDITY_DAYS}]")
    if state.pending_issues:
        lines.append(f"Pending issues: {', '.join(state.pending_issues)}")
    for step in state.completed_steps:
        lines.append(f"[Completed: {step}]")

    return "\n".join(lines)


def _build_action_hints(state: DrivingLicenceState) -> str:
    """Phase-level status summary — describes WHERE in the process the citizen is.
    Does NOT prescribe specific actions. Agent reasons from available_actions,
    completed_steps, pending_issues, and services_status."""
    if state.current_phase == "learner_licence":
        if not state.ll_prereqs_checked:
            return "Learner's Licence — eligibility verification phase."
        if state.pending_issues:
            return "Learner's Licence — issue resolution phase."
        if not state.written_test_passed:
            return "Learner's Licence — application and examination phase."
        return "Learner's Licence phase complete."
    else:
        if not state.dl_prereqs_checked:
            return "Permanent Licence — eligibility verification phase."
        if not state.waited_for_practice and state.complications["ll_timing"] == "too_early":
            return f"Permanent Licence — mandatory practice period. Day {state.simulated_day}."
        if not state.driving_test_passed:
            return "Permanent Licence — application and examination phase."
        return "Permanent Licence — processing phase."


def handle_action(state: DrivingLicenceState, action: Action) -> Tuple[str, bool, Optional[str]]:
    """Process action and update state."""
    state.steps_taken += 1
    action_record = {"step": state.steps_taken, "action": action.action_type.value, "params": action.parameters}
    at = action.action_type

    # ── CHECK_PREREQUISITES ──
    if at == ActionType.CHECK_PREREQUISITES:
        if state.current_phase == "learner_licence" and not state.ll_prereqs_checked:
            state.ll_prereqs_checked = True
            state.diagnostic_before_execution = True
            issues = []

            if state.complications["underage_for_class"]:
                issues.append(f"UNDERAGE: Age {state.citizen.age} is below 18. Cannot apply for '{state.citizen.attributes.get('vehicle_class', 'unknown')}'. Only gearless two-wheeler allowed for age 16-17 (with guardian consent).")
                state.pending_issues.append("underage_for_class")
                state.underage_detected = True

            if state.complications["address_mismatch"]:
                issues.append(f"ADDRESS MISMATCH: Aadhaar address '{state.citizen.documents['aadhaar_card'].fields['address']}' ≠ Current '{state.citizen.current_address}'")
                state.pending_issues.append("address_mismatch")
                state.address_mismatch_detected = True

            if state.complications["needs_medical_cert"] and not state.complications["has_medical_cert"]:
                issues.append(f"MEDICAL CERT MISSING: Age {state.citizen.age} requires Form 1A medical certificate from registered practitioner")
                state.pending_issues.append("medical_cert_missing")
                state.medical_missing_detected = True

            if state.complications["needs_guardian_consent"]:
                issues.append("MINOR: Guardian consent form required for applicants under 18")

            state.completed_steps.append("check_prereqs_ll")
            action_record["success"] = True
            state.action_history.append(action_record)

            age_info = f"Age {state.citizen.age}: "
            if state.citizen.age < 16:
                age_info += "Not eligible for any driving licence."
            elif state.citizen.age < 18:
                age_info += "Eligible for gearless two-wheeler only (with guardian consent)."
            else:
                age_info += "Eligible for all vehicle classes."

            if issues:
                return f"LL Prerequisites check:\n{age_info}\nIssues:\n" + "\n".join(f"  - {i}" for i in issues), True, None
            return f"LL Prerequisites check complete. {age_info} All documents valid. Ready to proceed.", True, None

        elif state.current_phase == "driving_licence" and not state.dl_prereqs_checked:
            state.dl_prereqs_checked = True
            state.diagnostic_before_execution = True
            days_since_ll = state.simulated_day - state.ll_issue_day

            issues = []
            if days_since_ll < MANDATORY_WAIT_DAYS:
                if state.complications["ll_timing"] == "too_early":
                    issues.append(f"TOO EARLY: Only {days_since_ll} days since LL. Must wait {MANDATORY_WAIT_DAYS} days (mandatory practice period). Wait {MANDATORY_WAIT_DAYS - days_since_ll} more days.")
                    state.pending_issues.append("wait_for_practice")

            if days_since_ll > LL_VALIDITY_DAYS:
                issues.append(f"LL EXPIRED: LL issued {days_since_ll} days ago. Validity is {LL_VALIDITY_DAYS} days. Must re-apply for LL.")
                state.pending_issues.append("ll_expired")

            state.completed_steps.append("check_prereqs_dl")
            action_record["success"] = True
            state.action_history.append(action_record)

            if issues:
                return f"DL Prerequisites check (Day {state.simulated_day}, LL issued Day {state.ll_issue_day}):\n" + "\n".join(f"  - {i}" for i in issues), True, None
            return f"DL Prerequisites check: LL is {days_since_ll} days old (valid). Ready for DL application.", True, None

        action_record["success"] = True
        state.action_history.append(action_record)
        return "Prerequisites already checked for this phase.", True, None

    # ── COMPARE_DOCUMENTS ──
    if at == ActionType.COMPARE_DOCUMENTS:
        state.ll_documents_compared = True
        state.diagnostic_before_execution = True
        aad = state.citizen.documents["aadhaar_card"].fields
        lines = [
            f"Aadhaar Address: '{aad['address']}' vs Current: '{state.citizen.current_address}' → {'MATCH' if aad['address'] == state.citizen.current_address else 'MISMATCH'}",
            f"Age Proof DOB: '{state.citizen.documents['age_proof'].fields['dob']}' vs Aadhaar DOB: '{aad['dob']}' → MATCH",
            f"Medical Certificate: {'Present' if state.citizen.documents['medical_certificate'].status == DocumentStatus.PRESENT else 'MISSING' if state.citizen.age >= 40 else 'Not required (age < 40)'}",
            f"Guardian Consent: {'Present' if state.citizen.documents['guardian_consent'].status == DocumentStatus.PRESENT else 'Not required (age >= 18)' if state.citizen.age >= 18 else 'MISSING'}",
        ]

        if aad['address'] != state.citizen.current_address:
            state.address_mismatch_detected = True

        state.completed_steps.append("compare_documents")
        action_record["success"] = True
        state.action_history.append(action_record)
        return "Document comparison:\n" + "\n".join(f"  {l}" for l in lines), True, None

    # ── EVALUATE_OPTIONS ──
    if at == ActionType.EVALUATE_OPTIONS:
        state.ll_options_evaluated = True
        options = []
        if state.underage_detected:
            options.append("Underage: Switch vehicle class to 'two_wheeler_gearless' using fill_form with corrected class. Need guardian consent.")
        if state.address_mismatch_detected and not state.address_fixed:
            options.append("Address: Update Aadhaar address via UIDAI (Rs.50, ~5 days)")
        if state.medical_missing_detected and not state.medical_obtained:
            options.append("Medical: Visit registered practitioner for Form 1A certificate (Rs.200, ~1 day)")

        state.completed_steps.append("evaluate_options")
        action_record["success"] = True
        state.action_history.append(action_record)
        return "Resolution options:\n" + "\n".join(f"  - {o}" for o in options) if options else "No issues.", True, None

    # ── CHECK_ELIGIBILITY ──
    if at == ActionType.CHECK_ELIGIBILITY:
        state.diagnostic_before_execution = True
        age = state.citizen.age
        eligibility = []
        if age >= 18:
            eligibility.append("Gearless two-wheeler: ✓")
            eligibility.append("Geared two-wheeler: ✓")
            eligibility.append("Four-wheeler (LMV): ✓")
        elif age >= 16:
            eligibility.append("Gearless two-wheeler: ✓ (with guardian consent)")
            eligibility.append("Geared two-wheeler: ✗ (must be 18+)")
            eligibility.append("Four-wheeler (LMV): ✗ (must be 18+)")
        else:
            eligibility.append("Not eligible for any vehicle class (must be 16+)")

        msg = f"Eligibility for age {age}:\n" + "\n".join(f"  {e}" for e in eligibility)
        msg += f"\nLL Fee: Rs.{LL_FEE}, DL Fee: Rs.{DL_FEE}"
        msg += f"\nLL validity: {LL_VALIDITY_DAYS} days. Mandatory practice period: {MANDATORY_WAIT_DAYS} days."

        state.completed_steps.append("check_eligibility")
        action_record["success"] = True
        state.action_history.append(action_record)
        return msg, True, None

    # ── FIX_DOCUMENT ──
    if at == ActionType.FIX_DOCUMENT:
        target = action.parameters.get("target", "")

        if target == "aadhaar_address" and state.address_mismatch_detected:
            state.citizen.documents["aadhaar_card"].fields["address"] = state.citizen.current_address
            state.address_fixed = True
            state.simulated_day += 5
            if "address_mismatch" in state.pending_issues:
                state.pending_issues.remove("address_mismatch")
            state.completed_steps.append("fix_address")
            action_record["success"] = True
            state.action_history.append(action_record)
            return f"Aadhaar address updated. Cost: Rs.50. Time: 5 days. Day: {state.simulated_day}.", True, None

        if target == "vehicle_class":
            state.citizen.attributes["vehicle_class"] = "two_wheeler_gearless"
            state.vehicle_class_corrected = True
            if "underage_for_class" in state.pending_issues:
                state.pending_issues.remove("underage_for_class")
            state.completed_steps.append("correct_vehicle_class")
            action_record["success"] = True
            state.action_history.append(action_record)
            return "Vehicle class corrected to 'two_wheeler_gearless' (eligible for age 16-17).", True, None

        if target in ("medical", "medical_certificate", "medical_cert", "form_1a") and state.medical_missing_detected and not state.medical_obtained:
            state.citizen.documents["medical_certificate"].status = DocumentStatus.PRESENT
            state.medical_obtained = True
            state.simulated_day += 1
            if "medical_cert_missing" in state.pending_issues:
                state.pending_issues.remove("medical_cert_missing")
            state.completed_steps.append("obtain_medical_cert")
            action_record["success"] = True
            state.action_history.append(action_record)
            return "Medical certificate (Form 1A) obtained from registered practitioner. Cost: Rs.200.", True, None

        action_record["success"] = False
        state.invalid_action_count += 1
        state.action_history.append(action_record)
        return "", False, f"Unknown fix target: '{target}'. Valid: aadhaar_address, vehicle_class, medical."

    # ── GATHER_DOCUMENT ──
    if at == ActionType.GATHER_DOCUMENT:
        target = action.parameters.get("target", action.parameters.get("doc_type", ""))

        if target in ("medical_certificate", "medical_cert", "form_1a"):
            if state.medical_missing_detected:
                state.citizen.documents["medical_certificate"].status = DocumentStatus.PRESENT
                state.medical_obtained = True
                state.simulated_day += 1
                if "medical_cert_missing" in state.pending_issues:
                    state.pending_issues.remove("medical_cert_missing")
                state.completed_steps.append("obtain_medical_cert")
                action_record["success"] = True
                state.action_history.append(action_record)
                return "Medical certificate (Form 1A) obtained from registered practitioner. Cost: Rs.200.", True, None

        action_record["success"] = True
        state.action_history.append(action_record)
        return "Document gathered.", True, None

    # ── FILL_FORM ──
    if at == ActionType.FILL_FORM:
        form_data = action.parameters

        if state.current_phase == "learner_licence":
            if state.pending_issues:
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, f"Resolve pending issues first: {', '.join(state.pending_issues)}"

            state.ll_form_filled = True
            state.ll_form_data = form_data

            # Check vehicle class
            submitted_class = form_data.get("vehicle_class", state.citizen.attributes.get('vehicle_class', ''))
            if state.citizen.age < 18 and submitted_class != "two_wheeler_gearless":
                state.forbidden_violations.append("wrong_vehicle_class_underage")
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, f"Cannot apply for '{submitted_class}' at age {state.citizen.age}. Only 'two_wheeler_gearless' allowed."

            state.completed_steps.append("fill_form_ll")
            action_record["success"] = True
            state.action_history.append(action_record)
            return "Form 2 (LL Application) filled successfully.", True, None
        else:
            if not state.waited_for_practice and "wait_for_practice" in state.pending_issues:
                state.forbidden_violations.append("apply_dl_before_30_days")
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "Mandatory 30-day practice period has not been completed."

            state.dl_form_filled = True
            state.dl_form_data = form_data
            state.completed_steps.append("fill_form_dl")
            action_record["success"] = True
            state.action_history.append(action_record)
            return "Form 4 (DL Application) filled successfully.", True, None

    # ── PAY_FEE ──
    if at == ActionType.PAY_FEE:
        if state.current_phase == "learner_licence":
            if state.ll_fee_paid:
                state.invalid_action_count += 1
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "LL fee already paid."
            state.ll_fee_paid = True
            state.ll_fee_amount = action.parameters.get("amount", LL_FEE)
            state.completed_steps.append("pay_fee_ll")
            action_record["success"] = True
            state.action_history.append(action_record)
            return f"LL application fee of Rs.{LL_FEE} paid.", True, None
        else:
            if state.dl_fee_paid:
                state.invalid_action_count += 1
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "DL fee already paid."
            state.dl_fee_paid = True
            state.dl_fee_amount = action.parameters.get("amount", DL_FEE)
            state.completed_steps.append("pay_fee_dl")
            action_record["success"] = True
            state.action_history.append(action_record)
            return f"DL application fee of Rs.{DL_FEE} paid.", True, None

    # ── BOOK_APPOINTMENT ──
    if at == ActionType.BOOK_APPOINTMENT:
        if state.current_phase == "learner_licence":
            if not state.ll_form_filled:
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "Application form is incomplete."
            if not state.ll_fee_paid:
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "Application fee has not been paid."
            state.ll_appointment_booked = True
            state.simulated_day += 5
            state.completed_steps.append("book_slot_ll")
            action_record["success"] = True
            state.action_history.append(action_record)
            return f"RTO slot booked for written test on Day {state.simulated_day}.", True, None
        else:
            if not state.dl_form_filled:
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "Application form is incomplete."
            if not state.dl_fee_paid:
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "Application fee has not been paid."
            state.dl_appointment_booked = True
            state.simulated_day += 5
            state.completed_steps.append("book_slot_dl")
            action_record["success"] = True
            state.action_history.append(action_record)
            return f"RTO slot booked for driving test on Day {state.simulated_day}.", True, None

    # ── TAKE_TEST ──
    if at == ActionType.TAKE_TEST:
        test_type = action.parameters.get("test_type", "")

        if state.current_phase == "learner_licence" and test_type in ("written", ""):
            if not state.ll_appointment_booked:
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "No RTO appointment has been scheduled."

            state.written_test_taken = True
            state.written_test_attempts += 1

            if state.complications["will_fail_written_test"] and state.written_test_attempts == 1:
                state.completed_steps.append("written_test_failed")
                action_record["success"] = True
                state.action_history.append(action_record)
                return (
                    f"Written test attempt {state.written_test_attempts}: FAILED (scored 6/15, pass mark: 9/15).\n"
                    f"Can retake after 7 days."
                ), True, None
            else:
                state.written_test_passed = True
                state.ll_received = True
                state.ll_issue_day = state.simulated_day

                # Set timing for phase 2
                if state.complications["ll_timing"] == "too_early":
                    pass  # agent will try to apply early
                elif state.complications["ll_timing"] == "expired":
                    state.simulated_day += LL_VALIDITY_DAYS + 10  # fast-forward past expiry

                state.current_phase = "driving_licence"
                state.completed_steps.append("written_test_passed")
                state.completed_steps.append("ll_received")
                action_record["success"] = True
                state.action_history.append(action_record)
                return (
                    f"Written test: PASSED (scored 12/15).\n"
                    f"Learner's Licence issued on Day {state.ll_issue_day}.\n"
                    f"LL valid until Day {state.ll_issue_day + LL_VALIDITY_DAYS}.\n"
                    f"Mandatory 30-day practice period before applying for permanent DL.\n"
                    f"Now proceed to Phase 2: Permanent Driving Licence."
                ), True, None

        if state.current_phase == "driving_licence" and test_type in ("practical", "driving", ""):
            if not state.dl_appointment_booked:
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, "No RTO appointment has been scheduled for the driving test."

            # Check LL validity
            days_since_ll = state.simulated_day - state.ll_issue_day
            if days_since_ll > LL_VALIDITY_DAYS:
                action_record["success"] = False
                state.action_history.append(action_record)
                return "", False, f"LL has expired ({days_since_ll} days old, max {LL_VALIDITY_DAYS}). Must re-apply for LL."

            state.driving_test_taken = True
            state.driving_test_attempts += 1

            if state.complications["will_fail_driving_test"] and state.driving_test_attempts == 1:
                state.completed_steps.append("driving_test_failed")
                action_record["success"] = True
                state.action_history.append(action_record)
                return (
                    f"Driving test attempt {state.driving_test_attempts}: FAILED.\n"
                    f"Reasons: Did not clear H-track maneuver.\n"
                    f"Can retake after 7 days. LL still valid until Day {state.ll_issue_day + LL_VALIDITY_DAYS}."
                ), True, None
            else:
                state.driving_test_passed = True
                state.completed_steps.append("driving_test_passed")
                action_record["success"] = True
                state.action_history.append(action_record)
                return "Driving test: PASSED. DL will be dispatched to address.", True, None

        action_record["success"] = False
        state.invalid_action_count += 1
        state.action_history.append(action_record)
        return "", False, f"Invalid test_type '{test_type}' for current phase '{state.current_phase}'."

    # ── WAIT ──
    if at == ActionType.WAIT:
        days = action.parameters.get("days", MANDATORY_WAIT_DAYS)
        days = min(int(days), 60)  # cap at 60 days
        state.simulated_day += days

        if "wait_for_practice" in state.pending_issues:
            if state.simulated_day - state.ll_issue_day >= MANDATORY_WAIT_DAYS:
                state.waited_for_practice = True
                state.pending_issues.remove("wait_for_practice")
                state.completed_steps.append("practice_period_complete")
                action_record["success"] = True
                state.action_history.append(action_record)
                return f"Waited {days} days. Day {state.simulated_day}. Mandatory practice period complete. Ready for DL application.", True, None

        state.completed_steps.append("wait")
        action_record["success"] = True
        state.action_history.append(action_record)
        return f"Waited {days} days. Current day: {state.simulated_day}.", True, None

    # ── CHECK_STATUS ──
    if at == ActionType.CHECK_STATUS:
        if state.driving_test_passed and not state.dl_received:
            state.dl_received = True
            state.simulated_day += 15
            state.done = True
            state.completed_steps.append("dl_received")
            action_record["success"] = True
            state.action_history.append(action_record)
            return (
                f"Smart-card Driving Licence dispatched on Day {state.simulated_day}.\n"
                f"DL Number: DL-{hash(state.citizen.name + str(state.simulated_day)) % 9000000 + 1000000}\n"
                f"Vehicle Class: {state.citizen.attributes.get('vehicle_class', 'unknown')}\n"
                f"Valid for 20 years or until age 50 (whichever is later).\n"
                f"Task complete."
            ), True, None

        status = [
            f"Phase: {state.current_phase.replace('_', ' ').title()}",
            f"Day: {state.simulated_day}",
        ]
        if state.ll_received:
            days_since = state.simulated_day - state.ll_issue_day
            remaining = LL_VALIDITY_DAYS - days_since
            status.append(f"LL: Received Day {state.ll_issue_day} ({days_since} days ago, {max(0, remaining)} days remaining)")
            status.append(f"Practice period: {'Complete' if days_since >= MANDATORY_WAIT_DAYS else f'{MANDATORY_WAIT_DAYS - days_since} days remaining'}")

        for step in state.completed_steps[-5:]:
            status.append(f"[{step}]")

        action_record["success"] = True
        state.action_history.append(action_record)
        return "Status:\n" + "\n".join(f"  {s}" for s in status), True, None

    # ── SUBMIT_APPLICATION ──
    if at == ActionType.SUBMIT_APPLICATION:
        if state.current_phase == "learner_licence":
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "This action is not applicable in the current phase."
        action_record["success"] = False
        state.action_history.append(action_record)
        return "", False, "This action is not applicable in the current phase."

    # ── UNKNOWN ──
    state.invalid_action_count += 1
    action_record["success"] = False
    state.action_history.append(action_record)
    return "", False, f"Action '{at.value}' not applicable in driving licence flow."


def build_observation(state: DrivingLicenceState, result_msg: str, success: bool, error: Optional[str]) -> Observation:
    doc_summary = {}
    for doc_id, doc in state.citizen.documents.items():
        doc_summary[doc_id] = f"{doc.status.value} — {doc.fields}"

    return Observation(
        task_id=TASK_ID,
        task_description=(
            "Help the citizen obtain a driving licence in India. This is a two-phase process: "
            "first obtaining a Learner's Licence (LL), then a permanent Driving Licence (DL) "
            "after a mandatory practice period. Age restrictions, medical requirements, and "
            "document validity windows may apply depending on the applicant's profile."
        ),
        difficulty=DIFFICULTY,
        citizen_summary=_build_citizen_summary(state.citizen, state),
        citizen_documents=doc_summary,
        current_phase=state.current_phase,
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


def build_initial_observation(state: DrivingLicenceState) -> Observation:
    return build_observation(
        state,
        result_msg="Episode started. You are helping a citizen obtain a driving licence in India.",
        success=True,
        error=None,
    )
