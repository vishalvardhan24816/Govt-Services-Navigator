"""
Task 2 (MEDIUM): Fresh Passport Application

Source: https://passportindia.gov.in/AppOnlineProject/online/faqGeneral
Verified: April 2026

Flow: Check prerequisites → gather documents → detect mismatches → fix if needed
      → fill form → pay fee → book appointment → visit PSK → police verification → dispatch
Complications: name mismatch, address proof invalid, photo rejected, form errors, Aadhaar address outdated
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

TASK_ID = TaskId.PASSPORT_FRESH
DIFFICULTY = Difficulty.MEDIUM
MAX_STEPS = 25

FEE_MAP = {
    "normal_36": 1500,
    "normal_60": 2000,
    "tatkaal_36": 3500,
    "tatkaal_60": 4000,
}

FIRST_NAMES = ["Ravi", "Anjali", "Mohan", "Sita", "Karthik", "Lakshmi", "Arjun", "Divya", "Venkat", "Meera"]
LAST_NAMES = ["Shankar", "Iyer", "Reddy", "Nair", "Menon", "Pillai", "Das", "Mukherjee", "Chatterjee", "Bose"]
FATHER_NAMES_FIRST = ["Ram", "Krishna", "Suresh", "Mahesh", "Ganesh", "Shiva", "Vishnu", "Prakash", "Mohan", "Gopal"]

ADDRESSES = [
    "12, Indiranagar, Bangalore 560038",
    "45, Banjara Hills, Hyderabad 500034",
    "78, Koramangala, Bangalore 560034",
    "23, T Nagar, Chennai 600017",
    "56, Jubilee Hills, Hyderabad 500033",
]

OLD_ADDRESSES = [
    "99, Jayanagar, Bangalore 560041",
    "31, Madhapur, Hyderabad 500081",
    "67, Whitefield, Bangalore 560066",
    "44, Adyar, Chennai 600020",
    "12, Gachibowli, Hyderabad 500032",
]


def _generate_aadhaar_number(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(12))


def generate_citizen(rng: random.Random) -> Tuple[CitizenProfile, Dict[str, Any]]:
    """Generate a citizen profile with potential passport complications."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    full_name = f"{first} {last}"
    father_first = rng.choice(FATHER_NAMES_FIRST)
    father_name = f"{father_first} {last}"
    mother_name = f"{'Smt. ' if rng.random() > 0.5 else ''}{rng.choice(FIRST_NAMES)} {last}"
    age = rng.randint(21, 55)
    gender = rng.choice(["Male", "Female"])
    address = rng.choice(ADDRESSES)
    aadhaar_number = _generate_aadhaar_number(rng)
    dob = f"{2026 - age}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"

    complications: Dict[str, Any] = {
        "name_mismatch": False,
        "address_outdated": False,
        "address_proof_invalid": False,
        "photo_invalid": False,
        "form_error_field": None,
    }

    aadhaar_name = full_name
    aadhaar_address = address
    birth_cert_name = full_name
    birth_cert_father = father_name

    roll = rng.random()

    # Name mismatch between Aadhaar and birth certificate
    if roll < 0.25:
        complications["name_mismatch"] = True
        name_variants = [
            f"{first} {last[0]}",               # abbreviated last name
            f"{first.upper()} {last.upper()}",   # case difference (simulated)
            f"{first} {last} {'JR' if rng.random() > 0.5 else 'SR'}",  # suffix
        ]
        birth_cert_name = rng.choice(name_variants)
        if birth_cert_name == full_name:
            birth_cert_name = f"{first} {last[0]}"
        complications["aadhaar_name"] = aadhaar_name
        complications["birth_cert_name"] = birth_cert_name

    # Aadhaar address outdated
    if 0.20 < roll < 0.50 or roll < 0.10:
        complications["address_outdated"] = True
        aadhaar_address = rng.choice(OLD_ADDRESSES)
        complications["old_address"] = aadhaar_address
        complications["current_address"] = address

    # Address proof invalid (utility bill too old, rent not notarized)
    if 0.45 < roll < 0.65:
        complications["address_proof_invalid"] = True
        complications["proof_issue"] = rng.choice([
            "utility_bill_older_than_3_months",
            "rent_agreement_not_notarized",
            "document_not_in_applicant_name",
        ])

    # Photo doesn't meet specs
    if 0.60 < roll < 0.75:
        complications["photo_invalid"] = True
        complications["photo_issue"] = rng.choice([
            "spectacles_on",
            "ears_not_visible",
            "wrong_background_color",
            "shadows_on_face",
        ])

    # Form error (will trigger on fill_form if agent doesn't match fields)
    if rng.random() < 0.20:
        complications["form_error_field"] = rng.choice(["father_name", "dob", "present_address"])

    citizen = CitizenProfile(
        name=full_name,
        age=age,
        gender=gender,
        current_address=address,
        identifiers={"aadhaar_number": aadhaar_number},
        complication_flags={
            "name_mismatch": complications["name_mismatch"],
            "address_mismatch": complications["address_outdated"],
        },
        documents={
            "aadhaar_card": DocumentInfo(
                doc_type="aadhaar_card",
                status=DocumentStatus.PRESENT,
                fields={
                    "aadhaar_number": aadhaar_number,
                    "name": aadhaar_name,
                    "address": aadhaar_address,
                    "dob": dob,
                },
            ),
            "birth_certificate": DocumentInfo(
                doc_type="birth_certificate",
                status=DocumentStatus.PRESENT,
                fields={
                    "name": birth_cert_name,
                    "dob": dob,
                    "father_name": birth_cert_father,
                    "mother_name": mother_name,
                },
            ),
            "address_proof": DocumentInfo(
                doc_type="address_proof",
                status=DocumentStatus.PRESENT if not complications["address_proof_invalid"] else DocumentStatus.INVALID,
                fields={
                    "type": "utility_bill",
                    "address": address,
                    "issue": complications.get("proof_issue", "none"),
                },
            ),
            "passport_photo": DocumentInfo(
                doc_type="passport_photo",
                status=DocumentStatus.PRESENT if not complications["photo_invalid"] else DocumentStatus.INVALID,
                fields={
                    "issue": complications.get("photo_issue", "none"),
                },
            ),
        },
    )

    complications["father_name"] = father_name
    complications["mother_name"] = mother_name
    complications["dob"] = dob

    return citizen, complications


def compute_ground_truth(citizen: CitizenProfile, complications: Dict[str, Any]) -> GroundTruth:
    """Precompute the correct solution path for passport application."""
    citizen_id = str(uuid.uuid4())[:8]
    checkpoints: List[GroundTruthCheckpoint] = []
    forbidden: List[str] = []
    expected_issues: List[str] = []
    optimal_steps = 0

    # Always check prerequisites first
    checkpoints.append(GroundTruthCheckpoint(
        step_id="check_prereqs", action_type=ActionType.CHECK_PREREQUISITES,
        description="Verify all documents and details", required=True, order_matters=True,
    ))
    optimal_steps += 1

    # Fix Aadhaar address if outdated
    if complications["address_outdated"]:
        expected_issues.append("aadhaar_address_outdated")
        checkpoints.append(GroundTruthCheckpoint(
            step_id="fix_aadhaar_address", action_type=ActionType.FIX_DOCUMENT,
            description="Update Aadhaar address to current", required=True, order_matters=True,
        ))
        optimal_steps += 1

    # Fix name mismatch
    if complications["name_mismatch"]:
        expected_issues.append("name_mismatch_docs")
        checkpoints.append(GroundTruthCheckpoint(
            step_id="fix_name", action_type=ActionType.FIX_DOCUMENT,
            description="Fix name mismatch between Aadhaar and birth certificate", required=True,
        ))
        optimal_steps += 1

    # Fix address proof if invalid
    if complications["address_proof_invalid"]:
        expected_issues.append("address_proof_invalid")
        checkpoints.append(GroundTruthCheckpoint(
            step_id="fix_address_proof", action_type=ActionType.GATHER_DOCUMENT,
            description="Get valid address proof", required=True,
        ))
        optimal_steps += 1

    # Fix photo if invalid
    if complications["photo_invalid"]:
        expected_issues.append("photo_rejected")
        checkpoints.append(GroundTruthCheckpoint(
            step_id="fix_photo", action_type=ActionType.GATHER_DOCUMENT,
            description="Get valid passport photo", required=True,
        ))
        optimal_steps += 1

    # Gather documents
    checkpoints.append(GroundTruthCheckpoint(
        step_id="gather_docs", action_type=ActionType.GATHER_DOCUMENT,
        description="Verify all documents ready", required=True,
    ))
    optimal_steps += 1

    # Fill form
    checkpoints.append(GroundTruthCheckpoint(
        step_id="fill_form", action_type=ActionType.FILL_FORM,
        description="Fill passport application form", required=True, order_matters=True,
    ))
    optimal_steps += 1

    # Pay fee
    checkpoints.append(GroundTruthCheckpoint(
        step_id="pay_fee", action_type=ActionType.PAY_FEE,
        description="Pay passport application fee", required=True,
    ))
    optimal_steps += 1

    # Book appointment
    checkpoints.append(GroundTruthCheckpoint(
        step_id="book_appointment", action_type=ActionType.BOOK_APPOINTMENT,
        description="Book PSK appointment", required=True, order_matters=True,
    ))
    optimal_steps += 1

    # Submit at PSK
    checkpoints.append(GroundTruthCheckpoint(
        step_id="visit_psk", action_type=ActionType.SUBMIT_APPLICATION,
        description="Visit PSK and submit documents", required=True, order_matters=True,
    ))
    forbidden.append("submit_without_docs")
    forbidden.append("skip_appointment")
    optimal_steps += 1

    # Check status
    checkpoints.append(GroundTruthCheckpoint(
        step_id="check_dispatch", action_type=ActionType.CHECK_STATUS,
        description="Check passport dispatch status", required=True,
    ))
    optimal_steps += 1

    scheme_type = "normal"
    booklet_type = "36_pages"
    fee_key = f"{scheme_type}_{booklet_type.split('_')[0]}"
    correct_fee = FEE_MAP.get(fee_key, 1500)

    return GroundTruth(
        task_id=TASK_ID,
        citizen_id=citizen_id,
        required_checkpoints=checkpoints,
        forbidden_actions=forbidden,
        valid_completions=["passport_dispatched"],
        expected_issues=expected_issues,
        optimal_steps=optimal_steps,
        correct_form_values={
            "applicant_name": citizen.documents["aadhaar_card"].fields["name"],
            "dob": complications["dob"],
            "father_name": complications["father_name"],
            "mother_name": complications["mother_name"],
            "present_address": citizen.current_address,
            "application_type": "fresh",
            "booklet_type": booklet_type,
            "scheme_type": scheme_type,
        },
        correct_fee=correct_fee,
    )


class PassportState:
    """Mutable internal state for a passport application episode."""

    def __init__(self, citizen: CitizenProfile, complications: Dict[str, Any], ground_truth: GroundTruth):
        self.citizen = citizen
        self.complications = complications
        self.ground_truth = ground_truth
        self.citizen_id = ground_truth.citizen_id

        self.prerequisites_checked = False
        self.documents_compared = False
        self.documents_gathered = False
        self.options_evaluated = False

        # Issue detection
        self.name_mismatch_detected = False
        self.address_outdated_detected = False
        self.address_proof_issue_detected = False
        self.photo_issue_detected = False

        # Fixes
        self.name_fixed = False
        self.aadhaar_address_fixed = False
        self.address_proof_fixed = False
        self.photo_fixed = False

        # Process steps
        self.form_filled = False
        self.form_data: Dict[str, Any] = {}
        self.form_errors: List[str] = []
        self.fee_paid = False
        self.fee_amount: Optional[float] = None
        self.appointment_booked = False
        self.psk_visited = False
        self.police_verification_done = False
        self.passport_dispatched = False

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
        if self.passport_dispatched:
            return {"passport_fresh": ServiceStatus.COMPLETED.value}
        elif self.psk_visited:
            return {"passport_fresh": ServiceStatus.IN_PROGRESS.value}
        elif self.pending_issues:
            return {"passport_fresh": ServiceStatus.BLOCKED.value}
        elif self.form_filled:
            return {"passport_fresh": ServiceStatus.IN_PROGRESS.value}
        else:
            return {"passport_fresh": ServiceStatus.NOT_STARTED.value}

    def get_progress(self) -> float:
        score = 0.0
        if self.prerequisites_checked: score += 0.08
        if self.documents_compared: score += 0.05

        # Issue resolution progress
        total_issues = sum([
            self.complications["name_mismatch"],
            self.complications["address_outdated"],
            self.complications["address_proof_invalid"],
            self.complications["photo_invalid"],
        ])
        resolved = sum([self.name_fixed, self.aadhaar_address_fixed, self.address_proof_fixed, self.photo_fixed])
        if total_issues > 0:
            score += 0.15 * (resolved / total_issues)
        else:
            score += 0.15  # no issues = auto credit

        if self.documents_gathered: score += 0.07
        if self.form_filled and not self.form_errors: score += 0.15
        elif self.form_filled: score += 0.08
        if self.fee_paid: score += 0.10
        if self.appointment_booked: score += 0.10
        if self.psk_visited: score += 0.15
        if self.police_verification_done: score += 0.10
        if self.passport_dispatched: score += 0.05

        return min(score, 1.0)


def _get_available_actions(state: PassportState) -> List[str]:
    actions = [ActionType.CHECK_STATUS.value]

    if not state.prerequisites_checked:
        actions.append(ActionType.CHECK_PREREQUISITES.value)
    if not state.documents_compared:
        actions.append(ActionType.COMPARE_DOCUMENTS.value)
    actions.append(ActionType.CHECK_ELIGIBILITY.value)

    if any([state.name_mismatch_detected, state.address_outdated_detected,
            state.address_proof_issue_detected, state.photo_issue_detected]):
        if not state.options_evaluated:
            actions.append(ActionType.EVALUATE_OPTIONS.value)

    if state.name_mismatch_detected and not state.name_fixed:
        actions.append(ActionType.FIX_DOCUMENT.value)
    if state.address_outdated_detected and not state.aadhaar_address_fixed:
        actions.append(ActionType.FIX_DOCUMENT.value)

    if state.address_proof_issue_detected and not state.address_proof_fixed:
        actions.append(ActionType.GATHER_DOCUMENT.value)
        actions.append(ActionType.FIX_DOCUMENT.value)
    if state.photo_issue_detected and not state.photo_fixed:
        actions.append(ActionType.GATHER_DOCUMENT.value)
        actions.append(ActionType.FIX_DOCUMENT.value)
    if not state.documents_gathered:
        actions.append(ActionType.GATHER_DOCUMENT.value)

    if not state.form_filled:
        actions.append(ActionType.FILL_FORM.value)
    if not state.fee_paid:
        actions.append(ActionType.PAY_FEE.value)
    if not state.appointment_booked:
        actions.append(ActionType.BOOK_APPOINTMENT.value)
    if not state.psk_visited:
        actions.append(ActionType.SUBMIT_APPLICATION.value)

    return list(set(actions))


def _build_citizen_summary(citizen: CitizenProfile, state: PassportState) -> str:
    aad = citizen.documents["aadhaar_card"].fields
    bc = citizen.documents["birth_certificate"].fields

    lines = [
        f"Citizen: {citizen.name}, Age: {citizen.age}, Gender: {citizen.gender}",
        f"Aadhaar: {aad['aadhaar_number']}, Name: {aad['name']}, Address: {aad['address']}",
        f"Birth Certificate Name: {bc['name']}, Father: {bc['father_name']}, Mother: {bc['mother_name']}",
        f"Current Address: {citizen.current_address}",
        f"Task: Apply for fresh Indian passport",
    ]

    if state.pending_issues:
        lines.append(f"\nPending issues: {', '.join(state.pending_issues)}")
    for step in state.completed_steps:
        lines.append(f"[Completed: {step}]")

    return "\n".join(lines)


def _build_action_hints(state: PassportState) -> str:
    """Phase-level status summary — describes WHERE in the process the citizen is.
    Does NOT prescribe specific actions. Agent reasons from available_actions,
    completed_steps, pending_issues, and services_status."""
    if not state.prerequisites_checked:
        return "Document verification phase."
    if state.pending_issues and not all([state.name_fixed or not state.name_mismatch_detected,
                                          state.aadhaar_address_fixed or not state.address_outdated_detected,
                                          state.address_proof_fixed or not state.address_proof_issue_detected,
                                          state.photo_fixed or not state.photo_issue_detected]):
        return "Issue resolution phase."
    if not state.documents_gathered:
        return "Document assembly phase."
    if not state.psk_visited:
        return "Application preparation phase."
    if not state.passport_dispatched:
        return "Processing phase."
    return "Process complete."


def handle_action(state: PassportState, action: Action) -> Tuple[str, bool, Optional[str]]:
    """Process an action and update state."""
    state.steps_taken += 1
    action_record = {"step": state.steps_taken, "action": action.action_type.value, "params": action.parameters}
    at = action.action_type

    # ── CHECK_PREREQUISITES ──
    if at == ActionType.CHECK_PREREQUISITES:
        state.prerequisites_checked = True
        state.diagnostic_before_execution = True
        issues = []

        if state.complications["name_mismatch"] and not state.name_fixed:
            issues.append(f"NAME MISMATCH: Aadhaar name '{state.citizen.documents['aadhaar_card'].fields['name']}' ≠ Birth certificate name '{state.citizen.documents['birth_certificate'].fields['name']}'")
            if "name_mismatch" not in state.pending_issues:
                state.pending_issues.append("name_mismatch")
            state.name_mismatch_detected = True

        if state.complications["address_outdated"] and not state.aadhaar_address_fixed:
            issues.append(f"ADDRESS OUTDATED: Aadhaar address '{state.citizen.documents['aadhaar_card'].fields['address']}' ≠ Current address '{state.citizen.current_address}'")
            if "aadhaar_address_outdated" not in state.pending_issues:
                state.pending_issues.append("aadhaar_address_outdated")
            state.address_outdated_detected = True

        if state.complications["address_proof_invalid"] and not state.address_proof_fixed:
            issues.append(f"ADDRESS PROOF INVALID: {state.complications['proof_issue'].replace('_', ' ')}")
            if "address_proof_invalid" not in state.pending_issues:
                state.pending_issues.append("address_proof_invalid")
            state.address_proof_issue_detected = True

        if state.complications["photo_invalid"] and not state.photo_fixed:
            issues.append(f"PHOTO INVALID: {state.complications['photo_issue'].replace('_', ' ')}")
            if "photo_invalid" not in state.pending_issues:
                state.pending_issues.append("photo_invalid")
            state.photo_issue_detected = True

        state.completed_steps.append("check_prerequisites")
        action_record["success"] = True
        state.action_history.append(action_record)

        if issues:
            return "Prerequisites check complete. Issues found:\n" + "\n".join(f"  - {i}" for i in issues), True, None
        return "Prerequisites check complete. All documents valid and consistent. Ready to proceed.", True, None

    # ── COMPARE_DOCUMENTS ──
    if at == ActionType.COMPARE_DOCUMENTS:
        state.documents_compared = True
        state.diagnostic_before_execution = True
        aad = state.citizen.documents["aadhaar_card"].fields
        bc = state.citizen.documents["birth_certificate"].fields

        lines = [
            f"Aadhaar Name: '{aad['name']}' vs Birth Cert Name: '{bc['name']}' → {'MATCH' if aad['name'] == bc['name'] else 'MISMATCH'}",
            f"Aadhaar DOB: '{aad['dob']}' vs Birth Cert DOB: '{bc['dob']}' → {'MATCH' if aad['dob'] == bc['dob'] else 'MISMATCH'}",
            f"Aadhaar Address: '{aad['address']}' vs Current Address: '{state.citizen.current_address}' → {'MATCH' if aad['address'] == state.citizen.current_address else 'MISMATCH'}",
            f"Address Proof Status: {state.citizen.documents['address_proof'].status.value}",
            f"Photo Status: {state.citizen.documents['passport_photo'].status.value}",
        ]

        if aad['name'] != bc['name']:
            state.name_mismatch_detected = True
        if aad['address'] != state.citizen.current_address:
            state.address_outdated_detected = True

        state.completed_steps.append("compare_documents")
        action_record["success"] = True
        state.action_history.append(action_record)
        return "Document comparison:\n" + "\n".join(f"  {l}" for l in lines), True, None

    # ── EVALUATE_OPTIONS ──
    if at == ActionType.EVALUATE_OPTIONS:
        state.options_evaluated = True
        options = []
        if state.name_mismatch_detected and not state.name_fixed:
            options.append("Name mismatch: Update Aadhaar name via UIDAI (Rs.50, ~7 days)")
        if state.address_outdated_detected and not state.aadhaar_address_fixed:
            options.append("Address outdated: Update Aadhaar address via UIDAI online (Rs.50, ~5 days)")
        if state.address_proof_issue_detected and not state.address_proof_fixed:
            options.append("Address proof: Get recent utility bill OR use Aadhaar as address proof (if updated)")
        if state.photo_issue_detected and not state.photo_fixed:
            options.append("Photo: Get new passport photo meeting specifications (white bg, ears visible, no specs)")

        state.completed_steps.append("evaluate_options")
        action_record["success"] = True
        state.action_history.append(action_record)
        return "Resolution options:\n" + "\n".join(f"  - {o}" for o in options) if options else "No issues to resolve.", True, None

    # ── CHECK_ELIGIBILITY ──
    if at == ActionType.CHECK_ELIGIBILITY:
        state.diagnostic_before_execution = True
        msg = (
            f"Passport eligibility check:\n"
            f"  - Indian citizen: Yes\n"
            f"  - Age: {state.citizen.age} (eligible)\n"
            f"  - Has Aadhaar: Yes\n"
            f"  - Fee: Rs.1500 (normal 36pg) / Rs.2000 (60pg) / Rs.3500 (tatkaal)\n"
            f"  - Processing time: Normal ~30 days, Tatkaal ~7 days"
        )
        state.completed_steps.append("check_eligibility")
        action_record["success"] = True
        state.action_history.append(action_record)
        return msg, True, None

    # ── FIX_DOCUMENT ──
    if at == ActionType.FIX_DOCUMENT:
        target = action.parameters.get("target", "")

        if target == "aadhaar_name" and state.name_mismatch_detected:
            correct_name = state.citizen.documents["birth_certificate"].fields["name"]
            state.citizen.documents["aadhaar_card"].fields["name"] = correct_name
            state.name_fixed = True
            state.simulated_day += 7
            if "name_mismatch" in state.pending_issues:
                state.pending_issues.remove("name_mismatch")
            state.completed_steps.append("fix_aadhaar_name")
            action_record["success"] = True
            state.action_history.append(action_record)
            return f"Aadhaar name updated to '{correct_name}'. Cost: Rs.50. Time: 7 days.", True, None

        if target == "aadhaar_address" and state.address_outdated_detected:
            state.citizen.documents["aadhaar_card"].fields["address"] = state.citizen.current_address
            state.aadhaar_address_fixed = True
            state.simulated_day += 5
            if "aadhaar_address_outdated" in state.pending_issues:
                state.pending_issues.remove("aadhaar_address_outdated")
            state.completed_steps.append("fix_aadhaar_address")
            action_record["success"] = True
            state.action_history.append(action_record)
            return f"Aadhaar address updated to '{state.citizen.current_address}'. Cost: Rs.50. Time: 5 days.", True, None

        if target in ("photo", "passport_photo") and state.photo_issue_detected and not state.photo_fixed:
            state.citizen.documents["passport_photo"].status = DocumentStatus.PRESENT
            state.citizen.documents["passport_photo"].fields["issue"] = "none"
            state.photo_fixed = True
            if "photo_invalid" in state.pending_issues:
                state.pending_issues.remove("photo_invalid")
            state.completed_steps.append("fix_photo")
            action_record["success"] = True
            state.action_history.append(action_record)
            return "Obtained new passport photo meeting all specifications.", True, None

        if target in ("address_proof", "utility_bill") and state.address_proof_issue_detected and not state.address_proof_fixed:
            state.citizen.documents["address_proof"].status = DocumentStatus.PRESENT
            state.citizen.documents["address_proof"].fields["issue"] = "none"
            state.address_proof_fixed = True
            if "address_proof_invalid" in state.pending_issues:
                state.pending_issues.remove("address_proof_invalid")
            state.completed_steps.append("fix_address_proof")
            action_record["success"] = True
            state.action_history.append(action_record)
            return "Obtained valid address proof document.", True, None

        if not state.prerequisites_checked:
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Prerequisites have not been verified."

        action_record["success"] = False
        state.invalid_action_count += 1
        state.action_history.append(action_record)
        return "", False, f"Unknown or inapplicable fix target: '{target}'. Valid: aadhaar_name, aadhaar_address, photo, address_proof."

    # ── GATHER_DOCUMENT ──
    if at == ActionType.GATHER_DOCUMENT:
        target = action.parameters.get("target", action.parameters.get("doc_type", ""))

        if target in ("utility_bill_recent", "address_proof", "utility_bill"):
            if state.address_proof_issue_detected and not state.address_proof_fixed:
                state.citizen.documents["address_proof"].status = DocumentStatus.PRESENT
                state.citizen.documents["address_proof"].fields["issue"] = "none"
                state.address_proof_fixed = True
                if "address_proof_invalid" in state.pending_issues:
                    state.pending_issues.remove("address_proof_invalid")
                state.completed_steps.append("fix_address_proof")
                action_record["success"] = True
                state.action_history.append(action_record)
                return "Obtained recent utility bill as valid address proof.", True, None

        if target in ("passport_photo", "photo"):
            if state.photo_issue_detected and not state.photo_fixed:
                state.citizen.documents["passport_photo"].status = DocumentStatus.PRESENT
                state.citizen.documents["passport_photo"].fields["issue"] = "none"
                state.photo_fixed = True
                if "photo_invalid" in state.pending_issues:
                    state.pending_issues.remove("photo_invalid")
                state.completed_steps.append("fix_photo")
                action_record["success"] = True
                state.action_history.append(action_record)
                return "Obtained new passport photo meeting all specifications.", True, None

        if target in ("all", "verify_all", "documents"):
            state.documents_gathered = True
            state.completed_steps.append("gather_documents")
            action_record["success"] = True
            state.action_history.append(action_record)
            doc_status = {k: v.status.value for k, v in state.citizen.documents.items()}
            return f"All documents gathered and verified. Status: {doc_status}", True, None

        # Default: gather general
        state.documents_gathered = True
        state.completed_steps.append("gather_documents")
        action_record["success"] = True
        state.action_history.append(action_record)
        return "Documents gathered.", True, None

    # ── FILL_FORM ──
    if at == ActionType.FILL_FORM:
        form_data = action.parameters
        state.form_data = form_data
        state.form_filled = True
        state.form_errors = []

        # Validate form fields against ground truth
        gt = state.ground_truth.correct_form_values
        for field, expected in gt.items():
            submitted = form_data.get(field, "")
            if submitted and str(submitted).strip().lower() != str(expected).strip().lower():
                state.form_errors.append(f"Field '{field}': submitted '{submitted}' but expected '{expected}'")

        # Check for missing required fields
        required = ["applicant_name", "dob", "father_name", "present_address"]
        for f in required:
            if not form_data.get(f):
                state.form_errors.append(f"Required field '{f}' is missing")

        state.completed_steps.append("fill_form")
        action_record["success"] = True
        state.action_history.append(action_record)

        if state.form_errors:
            return (
                "Form filled with errors:\n" + "\n".join(f"  ⚠ {e}" for e in state.form_errors) +
                "\nYou can re-fill the form to correct these."
            ), True, None
        return "Passport application form filled successfully. All fields match supporting documents.", True, None

    # ── PAY_FEE ──
    if at == ActionType.PAY_FEE:
        if state.fee_paid:
            state.invalid_action_count += 1
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Fee already paid."

        amount = action.parameters.get("amount", 1500)
        state.fee_paid = True
        state.fee_amount = amount
        state.completed_steps.append("pay_fee")
        action_record["success"] = True
        state.action_history.append(action_record)
        return f"Payment of Rs.{amount} for passport application processed successfully.", True, None

    # ── BOOK_APPOINTMENT ──
    if at == ActionType.BOOK_APPOINTMENT:
        if not state.form_filled:
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Application form is incomplete."

        state.appointment_booked = True
        state.simulated_day += 3
        state.completed_steps.append("book_appointment")
        action_record["success"] = True
        state.action_history.append(action_record)
        return f"Appointment booked at nearest PSK for Day {state.simulated_day}. Bring all original documents.", True, None

    # ── SUBMIT_APPLICATION (Visit PSK) ──
    if at == ActionType.SUBMIT_APPLICATION:
        if not state.appointment_booked:
            state.forbidden_violations.append("skip_appointment")
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "No appointment has been scheduled."

        if not state.fee_paid:
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Application fee has not been paid."

        if not state.form_filled:
            state.forbidden_violations.append("submit_without_docs")
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, "Application form is incomplete."

        # Check if unresolved issues
        if state.pending_issues:
            state.forbidden_violations.append("submit_without_docs")
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, f"Cannot visit PSK with unresolved issues: {', '.join(state.pending_issues)}"

        if state.form_errors:
            action_record["success"] = False
            state.action_history.append(action_record)
            return "", False, f"Form has errors that will cause rejection at PSK: {'; '.join(state.form_errors)}. Re-fill the form."

        state.psk_visited = True
        state.simulated_day += 1
        state.completed_steps.append("visit_psk")
        action_record["success"] = True
        state.action_history.append(action_record)
        return (
            "PSK visit completed successfully.\n"
            "  - Documents verified ✓\n"
            "  - Biometrics captured ✓\n"
            "  - Application submitted ✓\n"
            "Police verification will be initiated automatically."
        ), True, None

    # ── CHECK_STATUS ──
    if at == ActionType.CHECK_STATUS:
        if state.psk_visited and not state.police_verification_done:
            state.police_verification_done = True
            state.simulated_day += 14
            state.completed_steps.append("police_verification")
            action_record["success"] = True
            state.action_history.append(action_record)
            return f"Police verification completed on Day {state.simulated_day}. Report: CLEAR. Passport being printed.", True, None

        if state.police_verification_done and not state.passport_dispatched:
            state.passport_dispatched = True
            state.simulated_day += 7
            state.done = True
            state.completed_steps.append("passport_dispatched")
            action_record["success"] = True
            state.action_history.append(action_record)
            return (
                f"Passport dispatched via Speed Post on Day {state.simulated_day}.\n"
                f"Passport Number: J{hash(state.citizen.name + str(state.simulated_day)) % 9000000 + 1000000}\n"
                f"Delivery expected within 3-5 business days.\n"
                f"Task complete."
            ), True, None

        status_parts = [
            f"Prerequisites: {'✓' if state.prerequisites_checked else '✗'}",
            f"Documents gathered: {'✓' if state.documents_gathered else '✗'}",
            f"Form filled: {'✓' if state.form_filled else '✗'}{' (with errors)' if state.form_errors else ''}",
            f"Fee paid: {'✓ Rs.' + str(state.fee_amount) if state.fee_paid else '✗'}",
            f"Appointment: {'✓' if state.appointment_booked else '✗'}",
            f"PSK visit: {'✓' if state.psk_visited else '✗'}",
        ]
        if state.pending_issues:
            status_parts.append(f"Pending issues: {', '.join(state.pending_issues)}")

        action_record["success"] = True
        state.action_history.append(action_record)
        return "Current status:\n" + "\n".join(f"  {s}" for s in status_parts), True, None

    # ── UNKNOWN ──
    state.invalid_action_count += 1
    action_record["success"] = False
    state.action_history.append(action_record)
    return "", False, f"Action '{at.value}' is not applicable at this stage of passport application."


def build_observation(state: PassportState, result_msg: str, success: bool, error: Optional[str]) -> Observation:
    doc_summary = {}
    for doc_id, doc in state.citizen.documents.items():
        doc_summary[doc_id] = f"{doc.status.value} — {doc.fields}"

    return Observation(
        task_id=TASK_ID,
        task_description=(
            "Help the citizen apply for a fresh Indian passport through the Passport Seva Kendra (PSK). "
            "The process involves document verification, application submission, and in-person verification. "
            "Common complications include identity document discrepancies, address proof issues, "
            "and photo specification requirements."
        ),
        difficulty=DIFFICULTY,
        citizen_summary=_build_citizen_summary(state.citizen, state),
        citizen_documents=doc_summary,
        current_phase="passport_application",
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


def build_initial_observation(state: PassportState) -> Observation:
    return build_observation(
        state,
        result_msg="Episode started. You are helping a citizen apply for a fresh Indian passport.",
        success=True,
        error=None,
    )
