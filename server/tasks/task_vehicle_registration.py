"""
Task 4 (EXPERT): New Vehicle Registration at RTO

Source: https://parivahan.gov.in/
Verified: April 2026

Flow: Verify insurance → gather docs (Form 20/21/22, PUC) → fill Form 20 →
      calculate & pay road tax + fees → book RTO appointment → submit →
      vehicle inspection → temporary reg → wait → permanent registration

Complications: insurance expired, address mismatch, invoice discrepancy,
               missing PUC, hypothecation (loan), inspection failure
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

TASK_ID = TaskId.VEHICLE_REGISTRATION
DIFFICULTY = Difficulty.EXPERT
MAX_STEPS = 35

FIRST_NAMES = ["Rajesh", "Priya", "Amit", "Sunita", "Vikram", "Neha", "Suresh", "Kavita", "Arun", "Deepa"]
LAST_NAMES = ["Kumar", "Sharma", "Singh", "Patel", "Verma", "Gupta", "Reddy", "Nair", "Joshi", "Rao"]

ADDRESSES = [
    "42, MG Road, Bangalore 560001",
    "15, Park Street, Kolkata 700016",
    "88, Nehru Place, New Delhi 110019",
    "33, Anna Salai, Chennai 600002",
    "7, FC Road, Pune 411004",
]

VEHICLE_TYPES = ["two_wheeler", "four_wheeler_petrol", "four_wheeler_diesel", "four_wheeler_ev"]
VEHICLE_NAMES = {
    "two_wheeler": ["Honda Activa 6G", "TVS Jupiter", "Royal Enfield Classic 350", "Bajaj Pulsar 150"],
    "four_wheeler_petrol": ["Maruti Swift", "Hyundai i20", "Tata Nexon", "Honda City"],
    "four_wheeler_diesel": ["Mahindra XUV700", "Toyota Innova", "Kia Seltos", "Tata Harrier"],
    "four_wheeler_ev": ["Tata Nexon EV", "MG ZS EV", "Hyundai Kona EV", "Mahindra XUV400 EV"],
}
DEALERS = ["AutoWorld Motors", "Prime Vehicles Pvt Ltd", "National Auto Sales", "Metro Car Hub"]
INSURANCE_PROVIDERS = ["ICICI Lombard", "HDFC ERGO", "Bajaj Allianz", "New India Assurance"]

ROAD_TAX_PCT = {
    "two_wheeler": 8,
    "four_wheeler_petrol": 10,
    "four_wheeler_diesel": 12,
    "four_wheeler_ev": 0,
}
REGISTRATION_FEE = {
    "two_wheeler": 300,
    "four_wheeler_petrol": 600,
    "four_wheeler_diesel": 600,
    "four_wheeler_ev": 600,
}
TEMP_REG_FEE = 200
NUMBER_PLATE_FEE = 500
SMART_CARD_FEE = 200
HYPOTHECATION_FEE = 1500

EX_SHOWROOM_RANGES = {
    "two_wheeler": (60000, 250000),
    "four_wheeler_petrol": (600000, 1500000),
    "four_wheeler_diesel": (900000, 2500000),
    "four_wheeler_ev": (1200000, 2000000),
}


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def _generate_aadhaar_number(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(12))


def _generate_chassis_number(rng: random.Random) -> str:
    chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    return "".join(rng.choices(chars, k=17))


def _generate_engine_number(rng: random.Random) -> str:
    chars = "ABCDEFGHKLMNPQRSTUVWXYZ0123456789"
    return "".join(rng.choices(chars, k=10))


def _calc_road_tax(vehicle_type: str, ex_showroom: int) -> int:
    pct = ROAD_TAX_PCT.get(vehicle_type, 10)
    return int(ex_showroom * pct / 100)


def _calc_total_fee(vehicle_type: str, ex_showroom: int, has_loan: bool) -> int:
    road_tax = _calc_road_tax(vehicle_type, ex_showroom)
    reg_fee = REGISTRATION_FEE.get(vehicle_type, 600)
    total = road_tax + reg_fee + TEMP_REG_FEE + NUMBER_PLATE_FEE + SMART_CARD_FEE
    if has_loan:
        total += HYPOTHECATION_FEE
    return total


# ──────────────────────────────────────────────────────────────────────
# CITIZEN PROFILE GENERATION
# ──────────────────────────────────────────────────────────────────────

def generate_citizen(rng: random.Random) -> Tuple[CitizenProfile, Dict[str, Any]]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    age = rng.randint(18, 55)
    address = rng.choice(ADDRESSES)
    aadhaar = _generate_aadhaar_number(rng)

    vehicle_type = rng.choice(VEHICLE_TYPES)
    vehicle_name = rng.choice(VEHICLE_NAMES[vehicle_type])
    dealer = rng.choice(DEALERS)
    lo, hi = EX_SHOWROOM_RANGES[vehicle_type]
    ex_showroom = rng.randint(lo // 1000, hi // 1000) * 1000
    chassis = _generate_chassis_number(rng)
    engine = _generate_engine_number(rng)
    insurance_provider = rng.choice(INSURANCE_PROVIDERS)

    # Complications (weighted)
    insurance_expired = rng.random() < 0.25
    address_mismatch = rng.random() < 0.20
    invoice_discrepancy = rng.random() < 0.15
    missing_puc = rng.random() < 0.15
    has_loan = rng.random() < 0.20
    inspection_failure = rng.random() < 0.10

    complications = {
        "insurance_expired": insurance_expired,
        "address_mismatch": address_mismatch,
        "invoice_discrepancy": invoice_discrepancy,
        "missing_puc": missing_puc,
        "hypothecation_required": has_loan,
        "inspection_failure": inspection_failure,
    }

    # Build address for Aadhaar (may differ if address_mismatch)
    aadhaar_address = rng.choice(ADDRESSES) if address_mismatch else address

    documents = {
        "aadhaar_card": DocumentInfo(
            doc_type="identity",
            status=DocumentStatus.PRESENT,
            fields={"number": aadhaar, "address": aadhaar_address, "holder_name": name},
        ),
        "dealer_invoice": DocumentInfo(
            doc_type="purchase",
            status=DocumentStatus.PRESENT,
            fields={
                "dealer": dealer,
                "vehicle": vehicle_name,
                "vehicle_type": vehicle_type,
                "ex_showroom": ex_showroom if not invoice_discrepancy else ex_showroom + rng.randint(10000, 50000),
                "actual_ex_showroom": ex_showroom,
                "chassis_number": chassis,
                "engine_number": engine,
                "holder_name": name,
            },
        ),
        "form_21": DocumentInfo(
            doc_type="sale_certificate",
            status=DocumentStatus.PRESENT,
            fields={"dealer": dealer, "vehicle": vehicle_name},
        ),
        "form_22": DocumentInfo(
            doc_type="roadworthiness",
            status=DocumentStatus.PRESENT,
            fields={
                "chassis_number": chassis,
                "engine_number": engine,
                "vehicle_type": vehicle_type,
            },
        ),
        "insurance_policy": DocumentInfo(
            doc_type="insurance",
            status=DocumentStatus.EXPIRED if insurance_expired else DocumentStatus.PRESENT,
            fields={
                "provider": insurance_provider,
                "valid": not insurance_expired,
                "vehicle": vehicle_name,
            },
        ),
        "puc_certificate": DocumentInfo(
            doc_type="pollution",
            status=DocumentStatus.NOT_AVAILABLE if missing_puc else DocumentStatus.PRESENT,
            fields={"valid": not missing_puc},
        ),
        "passport_photos": DocumentInfo(
            doc_type="photo",
            status=DocumentStatus.PRESENT,
            fields={},
        ),
    }

    if has_loan:
        documents["bank_noc"] = DocumentInfo(
            doc_type="loan",
            status=DocumentStatus.NOT_AVAILABLE,
            fields={"bank": "SBI", "loan_amount": int(ex_showroom * 0.8)},
        )

    gender = rng.choice(["Male", "Female"])
    citizen = CitizenProfile(
        name=name,
        age=age,
        gender=gender,
        current_address=address,
        documents=documents,
        identifiers={"aadhaar_number": aadhaar},
        attributes={"vehicle_class": vehicle_type},
        complication_flags={
            "address_mismatch": address_mismatch,
            "expired_document": insurance_expired,
            "missing_document": missing_puc,
        },
    )

    return citizen, complications


# ──────────────────────────────────────────────────────────────────────
# GROUND TRUTH
# ──────────────────────────────────────────────────────────────────────

def compute_ground_truth(citizen: CitizenProfile, comp: Dict[str, Any]) -> GroundTruth:
    checkpoints: List[GroundTruthCheckpoint] = []
    expected_issues: List[str] = []
    forbidden: List[str] = []
    step_idx = 0

    vehicle_type = citizen.attributes.get('vehicle_class', 'four_wheeler_petrol') or "four_wheeler_petrol"
    invoice = citizen.documents["dealer_invoice"]
    actual_price = invoice.fields.get("actual_ex_showroom", invoice.fields.get("ex_showroom", 0))
    has_loan = comp["hypothecation_required"]

    # Step 1: Check prerequisites
    checkpoints.append(GroundTruthCheckpoint(
        step_id=f"step_{step_idx}", action_type=ActionType.CHECK_PREREQUISITES,
        description="Check all prerequisites for vehicle registration",
    ))
    step_idx += 1

    # Handle complications
    if comp["insurance_expired"]:
        expected_issues.append("insurance_expired")
        checkpoints.append(GroundTruthCheckpoint(
            step_id=f"step_{step_idx}", action_type=ActionType.FIX_DOCUMENT,
            description="Renew or activate vehicle insurance",
        ))
        step_idx += 1
        forbidden.append("submit_without_insurance")

    if comp["address_mismatch"]:
        expected_issues.append("address_mismatch")
        checkpoints.append(GroundTruthCheckpoint(
            step_id=f"step_{step_idx}", action_type=ActionType.FIX_DOCUMENT,
            description="Resolve address mismatch on Aadhaar",
        ))
        step_idx += 1

    if comp["invoice_discrepancy"]:
        expected_issues.append("invoice_discrepancy")
        checkpoints.append(GroundTruthCheckpoint(
            step_id=f"step_{step_idx}", action_type=ActionType.FIX_DOCUMENT,
            description="Get corrected invoice from dealer",
        ))
        step_idx += 1

    if comp["missing_puc"]:
        expected_issues.append("missing_puc")
        checkpoints.append(GroundTruthCheckpoint(
            step_id=f"step_{step_idx}", action_type=ActionType.GATHER_DOCUMENT,
            description="Obtain PUC certificate from testing center",
        ))
        step_idx += 1

    if has_loan:
        expected_issues.append("hypothecation_required")
        checkpoints.append(GroundTruthCheckpoint(
            step_id=f"step_{step_idx}", action_type=ActionType.GATHER_DOCUMENT,
            description="Obtain bank NOC and Form 34 for hypothecation",
        ))
        step_idx += 1

    # Compare documents
    checkpoints.append(GroundTruthCheckpoint(
        step_id=f"step_{step_idx}", action_type=ActionType.COMPARE_DOCUMENTS,
        description="Verify all documents match (chassis, engine, name, address)",
    ))
    step_idx += 1

    # Fill Form 20
    checkpoints.append(GroundTruthCheckpoint(
        step_id=f"step_{step_idx}", action_type=ActionType.FILL_FORM,
        description="Fill Form 20 with vehicle and owner details",
    ))
    step_idx += 1

    # Pay fees
    total_fee = _calc_total_fee(vehicle_type, actual_price, has_loan)
    checkpoints.append(GroundTruthCheckpoint(
        step_id=f"step_{step_idx}", action_type=ActionType.PAY_FEE,
        description=f"Pay road tax + registration fee (total: ₹{total_fee})",
    ))
    step_idx += 1

    # Book appointment
    checkpoints.append(GroundTruthCheckpoint(
        step_id=f"step_{step_idx}", action_type=ActionType.BOOK_APPOINTMENT,
        description="Book RTO inspection appointment",
    ))
    step_idx += 1

    # Submit application
    checkpoints.append(GroundTruthCheckpoint(
        step_id=f"step_{step_idx}", action_type=ActionType.SUBMIT_APPLICATION,
        description="Submit Form 20 with all documents at RTO",
    ))
    step_idx += 1

    # Handle inspection failure
    if comp["inspection_failure"]:
        expected_issues.append("inspection_failure")
        checkpoints.append(GroundTruthCheckpoint(
            step_id=f"step_{step_idx}", action_type=ActionType.FIX_DOCUMENT,
            description="Resolve chassis/engine discrepancy after failed inspection",
        ))
        step_idx += 1

    # Wait for permanent registration
    checkpoints.append(GroundTruthCheckpoint(
        step_id=f"step_{step_idx}", action_type=ActionType.WAIT,
        description="Wait for permanent registration number",
    ))
    step_idx += 1

    # Check status (final)
    checkpoints.append(GroundTruthCheckpoint(
        step_id=f"step_{step_idx}", action_type=ActionType.CHECK_STATUS,
        description="Collect RC and verify registration details",
    ))
    step_idx += 1

    correct_form = {
        "owner_name": citizen.name,
        "address": citizen.current_address,
        "vehicle_type": vehicle_type,
        "vehicle_name": citizen.documents["dealer_invoice"].fields.get("vehicle", ""),
        "chassis_number": citizen.documents["form_22"].fields.get("chassis_number", ""),
        "engine_number": citizen.documents["form_22"].fields.get("engine_number", ""),
        "dealer": citizen.documents["dealer_invoice"].fields.get("dealer", ""),
    }

    return GroundTruth(
        task_id=TASK_ID,
        citizen_id=citizen.name,
        required_checkpoints=checkpoints,
        optimal_steps=step_idx,
        expected_issues=expected_issues,
        forbidden_actions=forbidden,
        correct_form_values=correct_form,
        correct_fee=total_fee,
    )


# ──────────────────────────────────────────────────────────────────────
# MUTABLE TASK STATE
# ──────────────────────────────────────────────────────────────────────

class VehicleRegistrationState:
    def __init__(self, citizen: CitizenProfile, comp: Dict[str, Any],
                 ground_truth: GroundTruth):
        self.citizen = citizen
        self.complications = comp
        self.ground_truth = ground_truth

        self.current_phase = "pre_registration"
        self.steps_taken = 0
        self.simulated_day = 0
        self.completed_steps: List[str] = []
        self.pending_issues: List[str] = []
        self.services_status: Dict[str, str] = {
            "insurance": "expired" if comp["insurance_expired"] else "valid",
            "puc": "missing" if comp["missing_puc"] else "valid",
            "form_20": "not_started",
            "road_tax": "not_paid",
            "rto_appointment": "not_booked",
            "application": "not_submitted",
            "inspection": "not_done",
            "temp_registration": "not_issued",
            "permanent_registration": "not_issued",
        }

        self.prerequisites_checked = False
        self.documents_compared = False
        self.form_filled = False
        self.form_data: Dict[str, Any] = {}
        self.fee_paid = False
        self.fee_amount: Optional[float] = None
        self.appointment_booked = False
        self.application_submitted = False
        self.inspection_done = False
        self.temp_reg_issued = False
        self.permanent_reg_issued = False
        self.insurance_fixed = False
        self.address_fixed = False
        self.invoice_fixed = False
        self.puc_obtained = False
        self.bank_noc_obtained = False
        self.inspection_issue_fixed = False

        # Grader-required tracking
        self.action_history: List[Dict[str, Any]] = []
        self.forbidden_violations: List[str] = []
        self.invalid_action_count = 0

    @property
    def done(self) -> bool:
        return self.permanent_reg_issued

    def get_services_status(self) -> Dict[str, str]:
        return dict(self.services_status)

    def get_progress(self) -> float:
        """Return 0-1 progress estimate."""
        total = 10.0
        done = 0.0
        if self.prerequisites_checked: done += 1
        if self.documents_compared: done += 1
        if self.form_filled: done += 1
        if self.fee_paid: done += 1
        if self.appointment_booked: done += 1
        if self.application_submitted: done += 1
        if self.inspection_done: done += 1
        if self.temp_reg_issued: done += 1
        if self.permanent_reg_issued: done += 2
        return min(done / total, 1.0)


# ──────────────────────────────────────────────────────────────────────
# ACTION HANDLERS
# ──────────────────────────────────────────────────────────────────────

def handle_action(state: VehicleRegistrationState, action: Action) -> Tuple[str, bool, Optional[str]]:
    at = action.action_type
    params = action.parameters

    state.steps_taken += 1
    action_record = {"action": at.value, "step": state.steps_taken, "params": params, "success": False}

    if at == ActionType.CHECK_PREREQUISITES:
        result = _handle_check_prerequisites(state)
    elif at == ActionType.COMPARE_DOCUMENTS:
        result = _handle_compare_documents(state)
    elif at == ActionType.EVALUATE_OPTIONS:
        result = _handle_evaluate_options(state)
    elif at == ActionType.CHECK_ELIGIBILITY:
        result = _handle_check_eligibility(state)
    elif at == ActionType.CHECK_STATUS:
        result = _handle_check_status(state)
    elif at == ActionType.GATHER_DOCUMENT:
        result = _handle_gather_document(state, params)
    elif at == ActionType.FILL_FORM:
        result = _handle_fill_form(state, params)
    elif at == ActionType.PAY_FEE:
        result = _handle_pay_fee(state, params)
    elif at == ActionType.BOOK_APPOINTMENT:
        result = _handle_book_appointment(state)
    elif at == ActionType.SUBMIT_APPLICATION:
        result = _handle_submit_application(state)
    elif at == ActionType.FIX_DOCUMENT:
        result = _handle_fix_document(state, params)
    elif at == ActionType.WAIT:
        result = _handle_wait(state, params)
    elif at == ActionType.TAKE_TEST:
        state.invalid_action_count += 1
        result = ("Vehicle registration does not require a test.", False,
                  "take_test is not applicable for vehicle registration")
    elif at == ActionType.APPEAL_REJECTION:
        result = _handle_appeal(state)
    else:
        state.invalid_action_count += 1
        result = (f"Unknown action: {at.value}", False, f"Action {at.value} not recognized")

    msg, success, error = result
    action_record["success"] = success
    state.action_history.append(action_record)
    return result


def _handle_check_prerequisites(state: VehicleRegistrationState) -> Tuple[str, bool, Optional[str]]:
    state.prerequisites_checked = True
    issues = []
    citizen = state.citizen
    comp = state.complications

    if comp["insurance_expired"] and not state.insurance_fixed:
        issues.append("insurance_expired")
    if comp["address_mismatch"] and not state.address_fixed:
        issues.append("address_mismatch")
    if comp["invoice_discrepancy"] and not state.invoice_fixed:
        issues.append("invoice_discrepancy")
    if comp["missing_puc"] and not state.puc_obtained:
        issues.append("missing_puc")
    if comp["hypothecation_required"] and not state.bank_noc_obtained:
        issues.append("hypothecation_required")

    state.pending_issues = issues
    state.completed_steps.append("check_prerequisites")

    vehicle_type = citizen.attributes.get('vehicle_class', 'four_wheeler_petrol') or "four_wheeler_petrol"
    invoice = citizen.documents["dealer_invoice"]
    vehicle_name = invoice.fields.get("vehicle", "Unknown Vehicle")

    if issues:
        msg = (
            f"Prerequisites check for {vehicle_name} ({vehicle_type}) registration:\n"
            f"Owner: {citizen.name}, Age: {citizen.age}\n"
            f"Dealer: {invoice.fields.get('dealer', 'N/A')}\n"
            f"ISSUES FOUND: {', '.join(issues)}\n"
        )
        if "insurance_expired" in issues:
            msg += "⚠ Vehicle insurance is expired or not activated. Must fix before proceeding.\n"
        if "address_mismatch" in issues:
            msg += "⚠ Address on Aadhaar doesn't match current address. Update required.\n"
        if "invoice_discrepancy" in issues:
            msg += "⚠ Dealer invoice amount doesn't match ex-showroom price. Get corrected invoice.\n"
        if "missing_puc" in issues:
            msg += "⚠ PUC certificate is missing. Obtain from authorized testing center.\n"
        if "hypothecation_required" in issues:
            msg += "⚠ Vehicle on loan — need bank NOC and Form 34 for hypothecation endorsement.\n"
        return (msg, True, None)
    else:
        return (
            f"All prerequisites met for {vehicle_name} ({vehicle_type}) registration.\n"
            f"Owner: {citizen.name}, Age: {citizen.age}\n"
            f"Dealer: {invoice.fields.get('dealer', 'N/A')}\n"
            f"No issues found. Proceed with document comparison and Form 20.",
            True, None,
        )


def _handle_compare_documents(state: VehicleRegistrationState) -> Tuple[str, bool, Optional[str]]:
    state.documents_compared = True
    state.completed_steps.append("compare_documents")

    citizen = state.citizen
    invoice = citizen.documents["dealer_invoice"]
    form22 = citizen.documents["form_22"]

    chassis_match = invoice.fields.get("chassis_number") == form22.fields.get("chassis_number")
    engine_match = invoice.fields.get("engine_number") == form22.fields.get("engine_number")

    msg = "Document comparison results:\n"
    msg += f"  Chassis number: {'✓ Match' if chassis_match else '✗ MISMATCH'}\n"
    msg += f"  Engine number: {'✓ Match' if engine_match else '✗ MISMATCH'}\n"
    msg += f"  Owner name on all docs: {citizen.name}\n"

    if state.complications["address_mismatch"] and not state.address_fixed:
        aadhaar_addr = citizen.documents["aadhaar_card"].fields.get("address", "")
        msg += f"  ⚠ Aadhaar address: {aadhaar_addr}\n"
        msg += f"  ⚠ Current address: {citizen.current_address}\n"
        msg += "  Address MISMATCH detected.\n"
    else:
        msg += "  Address: ✓ Match\n"

    if state.complications["invoice_discrepancy"] and not state.invoice_fixed:
        msg += f"  ⚠ Invoice shows ₹{invoice.fields.get('ex_showroom', 0)} but actual ex-showroom is ₹{invoice.fields.get('actual_ex_showroom', 0)}\n"

    return (msg, True, None)


def _handle_evaluate_options(state: VehicleRegistrationState) -> Tuple[str, bool, Optional[str]]:
    state.completed_steps.append("evaluate_options")
    citizen = state.citizen
    vehicle_type = citizen.attributes.get('vehicle_class', 'four_wheeler_petrol') or "four_wheeler_petrol"
    invoice = citizen.documents["dealer_invoice"]
    actual_price = invoice.fields.get("actual_ex_showroom", invoice.fields.get("ex_showroom", 0))
    has_loan = state.complications["hypothecation_required"]

    road_tax = _calc_road_tax(vehicle_type, actual_price)
    total = _calc_total_fee(vehicle_type, actual_price, has_loan)

    msg = "Fee breakdown for registration:\n"
    msg += f"  Road tax ({ROAD_TAX_PCT.get(vehicle_type, 10)}% of ₹{actual_price}): ₹{road_tax}\n"
    msg += f"  Registration fee: ₹{REGISTRATION_FEE.get(vehicle_type, 600)}\n"
    msg += f"  Temporary registration: ₹{TEMP_REG_FEE}\n"
    msg += f"  Number plates: ₹{NUMBER_PLATE_FEE}\n"
    msg += f"  Smart card RC: ₹{SMART_CARD_FEE}\n"
    if has_loan:
        msg += f"  Hypothecation fee: ₹{HYPOTHECATION_FEE}\n"
    msg += f"  TOTAL: ₹{total}\n"

    return (msg, True, None)


def _handle_check_eligibility(state: VehicleRegistrationState) -> Tuple[str, bool, Optional[str]]:
    state.completed_steps.append("check_eligibility")
    citizen = state.citizen
    eligible = citizen.age >= 18
    msg = f"Eligibility check: {'ELIGIBLE' if eligible else 'NOT ELIGIBLE'}\n"
    msg += f"  Age: {citizen.age} (minimum 18 required)\n"
    if not eligible:
        msg += "  ⚠ Owner must be 18+ to register a vehicle.\n"
    return (msg, True, None)


def _handle_check_status(state: VehicleRegistrationState) -> Tuple[str, bool, Optional[str]]:
    state.completed_steps.append("check_status")
    state.simulated_day += 1

    if not state.application_submitted:
        return ("Application not yet submitted. Complete the registration process first.",
                True, None)

    if not state.inspection_done:
        if state.appointment_booked and state.simulated_day >= 3:
            # Inspection happens
            if state.complications["inspection_failure"] and not state.inspection_issue_fixed:
                state.services_status["inspection"] = "failed"
                if "inspection_failure" not in state.pending_issues:
                    state.pending_issues.append("inspection_failure")
                return (
                    "INSPECTION RESULT: FAILED\n"
                    "Chassis number on vehicle doesn't match Form 22.\n"
                    "Contact dealer to resolve discrepancy and re-inspect.",
                    True, None,
                )
            else:
                state.inspection_done = True
                state.temp_reg_issued = True
                state.services_status["inspection"] = "passed"
                state.services_status["temp_registration"] = "issued"
                state.current_phase = "permanent_registration"
                return (
                    "INSPECTION RESULT: PASSED\n"
                    "Temporary registration issued.\n"
                    "Wait 7 days for permanent registration number.",
                    True, None,
                )
        else:
            return ("Waiting for inspection appointment date...", True, None)

    # After inspection — waiting for permanent registration
    if state.temp_reg_issued and state.simulated_day >= 10:
        state.permanent_reg_issued = True
        state.services_status["permanent_registration"] = "issued"
        return (
            "PERMANENT REGISTRATION COMPLETE\n"
            f"Registration number assigned for {state.citizen.documents['dealer_invoice'].fields.get('vehicle', 'vehicle')}.\n"
            "Registration Certificate (RC) is ready for collection.\n"
            "Affix permanent number plates within 7 days.",
            True, None,
        )

    return (
        f"Status: Day {state.simulated_day}\n"
        f"Inspection: {state.services_status['inspection']}\n"
        f"Temp Registration: {state.services_status['temp_registration']}\n"
        f"Permanent Registration: {state.services_status['permanent_registration']}",
        True, None,
    )


def _handle_gather_document(state: VehicleRegistrationState, params: Dict) -> Tuple[str, bool, Optional[str]]:
    target = params.get("target", "")
    state.completed_steps.append(f"gather_document:{target}")

    if target == "puc" or target == "puc_certificate":
        state.puc_obtained = True
        state.citizen.documents["puc_certificate"] = DocumentInfo(
            doc_type="pollution",
            status=DocumentStatus.PRESENT,
            fields={"valid": True},
        )
        state.services_status["puc"] = "valid"
        if "missing_puc" in state.pending_issues:
            state.pending_issues.remove("missing_puc")
        return ("PUC certificate obtained from authorized testing center.", True, None)

    if target == "bank_noc" or target == "form_34":
        state.bank_noc_obtained = True
        if "bank_noc" in state.citizen.documents:
            state.citizen.documents["bank_noc"] = DocumentInfo(
                doc_type="loan",
                status=DocumentStatus.PRESENT,
                fields=dict(state.citizen.documents["bank_noc"].fields),
            )
        if "hypothecation_required" in state.pending_issues:
            state.pending_issues.remove("hypothecation_required")
        return ("Bank NOC and Form 34 obtained for hypothecation endorsement.", True, None)

    if target == "all":
        if state.complications["missing_puc"] and not state.puc_obtained:
            state.puc_obtained = True
            if "missing_puc" in state.pending_issues:
                state.pending_issues.remove("missing_puc")
        if state.complications["hypothecation_required"] and not state.bank_noc_obtained:
            state.bank_noc_obtained = True
            if "hypothecation_required" in state.pending_issues:
                state.pending_issues.remove("hypothecation_required")
        return ("All required documents gathered.", True, None)

    return (f"Gathered document: {target}", True, None)


def _handle_fill_form(state: VehicleRegistrationState, params: Dict) -> Tuple[str, bool, Optional[str]]:
    if not state.prerequisites_checked:
        return ("", False, "Prerequisites have not been verified.")

    # Auto-populate form from citizen profile (server-side, any agent benefits)
    citizen = state.citizen
    invoice = citizen.documents.get("dealer_invoice", None)
    form_22 = citizen.documents.get("form_22", None)
    state.form_data = {
        "owner_name": params.get("owner_name", citizen.name),
        "address": params.get("address", citizen.current_address),
        "vehicle_type": params.get("vehicle_type", citizen.attributes.get('vehicle_class', 'four_wheeler_petrol') or "four_wheeler_petrol"),
        "vehicle_name": params.get("vehicle_name",
                                   invoice.fields.get("vehicle", "") if invoice else ""),
        "chassis_number": params.get("chassis_number",
                                     form_22.fields.get("chassis_number", "") if form_22 else ""),
        "engine_number": params.get("engine_number",
                                    form_22.fields.get("engine_number", "") if form_22 else ""),
        "dealer": params.get("dealer",
                             invoice.fields.get("dealer", "") if invoice else ""),
    }

    state.form_filled = True
    state.services_status["form_20"] = "filled"
    state.current_phase = "application"
    state.completed_steps.append("fill_form")
    return ("Form 20 (application for registration) filled successfully.", True, None)


def _handle_pay_fee(state: VehicleRegistrationState, params: Dict) -> Tuple[str, bool, Optional[str]]:
    if not state.form_filled:
        return ("", False, "Application form is incomplete.")

    citizen = state.citizen
    vehicle_type = citizen.attributes.get('vehicle_class', 'four_wheeler_petrol') or "four_wheeler_petrol"
    invoice = citizen.documents["dealer_invoice"]
    actual_price = invoice.fields.get("actual_ex_showroom", invoice.fields.get("ex_showroom", 0))
    has_loan = state.complications["hypothecation_required"]
    correct_total = _calc_total_fee(vehicle_type, actual_price, has_loan)

    state.fee_paid = True
    state.fee_amount = correct_total
    state.services_status["road_tax"] = "paid"
    state.completed_steps.append("pay_fee")
    return (f"Road tax and registration fees paid. Total: ₹{correct_total}", True, None)


def _handle_book_appointment(state: VehicleRegistrationState) -> Tuple[str, bool, Optional[str]]:
    if not state.fee_paid:
        return ("", False, "Registration fees have not been paid.")

    state.appointment_booked = True
    state.services_status["rto_appointment"] = "booked"
    state.completed_steps.append("book_appointment")
    return ("RTO vehicle inspection appointment booked. Present vehicle in 3 days.", True, None)


def _handle_submit_application(state: VehicleRegistrationState) -> Tuple[str, bool, Optional[str]]:
    # Validation checks
    if not state.prerequisites_checked:
        return ("", False, "Prerequisites have not been verified.")
    if not state.form_filled:
        return ("", False, "Application form is incomplete.")
    if not state.fee_paid:
        return ("", False, "Registration fees have not been paid.")
    if not state.appointment_booked:
        return ("", False, "No RTO appointment has been scheduled.")

    # Check blocking issues
    if state.complications["insurance_expired"] and not state.insurance_fixed:
        return ("", False, "Submission blocked: vehicle insurance is expired.")
    if state.complications["address_mismatch"] and not state.address_fixed:
        return ("", False, "Submission blocked: address mismatch on Aadhaar is unresolved.")
    if state.complications["invoice_discrepancy"] and not state.invoice_fixed:
        return ("", False, "Submission blocked: dealer invoice discrepancy is unresolved.")

    state.application_submitted = True
    state.services_status["application"] = "submitted"
    state.current_phase = "inspection"
    state.completed_steps.append("submit_application")
    return ("Application submitted at RTO with all documents. Vehicle inspection pending.", True, None)


def _handle_fix_document(state: VehicleRegistrationState, params: Dict) -> Tuple[str, bool, Optional[str]]:
    target = params.get("target", "")
    state.completed_steps.append(f"fix_document:{target}")

    if "insurance" in target:
        state.insurance_fixed = True
        state.citizen.documents["insurance_policy"] = DocumentInfo(
            doc_type="insurance",
            status=DocumentStatus.PRESENT,
            fields={**state.citizen.documents["insurance_policy"].fields, "valid": True},
        )
        state.services_status["insurance"] = "valid"
        if "insurance_expired" in state.pending_issues:
            state.pending_issues.remove("insurance_expired")
        return ("Vehicle insurance renewed/activated successfully.", True, None)

    if "address" in target or "aadhaar" in target:
        state.address_fixed = True
        state.citizen.documents["aadhaar_card"] = DocumentInfo(
            doc_type="identity",
            status=DocumentStatus.PRESENT,
            fields={**state.citizen.documents["aadhaar_card"].fields, "address": state.citizen.current_address},
        )
        if "address_mismatch" in state.pending_issues:
            state.pending_issues.remove("address_mismatch")
        return ("Aadhaar address updated to match current address.", True, None)

    if "invoice" in target:
        state.invoice_fixed = True
        invoice = state.citizen.documents["dealer_invoice"]
        actual = invoice.fields.get("actual_ex_showroom", invoice.fields.get("ex_showroom", 0))
        state.citizen.documents["dealer_invoice"] = DocumentInfo(
            doc_type="purchase",
            status=DocumentStatus.PRESENT,
            fields={**invoice.fields, "ex_showroom": actual},
        )
        if "invoice_discrepancy" in state.pending_issues:
            state.pending_issues.remove("invoice_discrepancy")
        return ("Corrected dealer invoice obtained.", True, None)

    if "chassis" in target or "engine" in target or "inspection" in target:
        state.inspection_issue_fixed = True
        if "inspection_failure" in state.pending_issues:
            state.pending_issues.remove("inspection_failure")
        return ("Chassis/engine discrepancy resolved with dealer. Ready for re-inspection.", True, None)

    if "puc" in target:
        state.puc_obtained = True
        state.citizen.documents["puc_certificate"] = DocumentInfo(
            doc_type="pollution", status=DocumentStatus.PRESENT, fields={"valid": True},
        )
        state.services_status["puc"] = "valid"
        if "missing_puc" in state.pending_issues:
            state.pending_issues.remove("missing_puc")
        return ("PUC certificate obtained from authorized testing center.", True, None)

    if "hypothecation" in target or "bank_noc" in target or "noc" in target:
        state.bank_noc_obtained = True
        if "hypothecation_required" in state.pending_issues:
            state.pending_issues.remove("hypothecation_required")
        return ("Bank NOC and Form 34 obtained for hypothecation endorsement.", True, None)

    # Generic fix
    return (f"Document issue '{target}' resolved.", True, None)


def _handle_wait(state: VehicleRegistrationState, params: Dict) -> Tuple[str, bool, Optional[str]]:
    days = params.get("days", 7)
    state.simulated_day += days
    state.completed_steps.append(f"wait:{days}")
    return (f"Waited {days} days. Current day: {state.simulated_day}", True, None)


def _handle_appeal(state: VehicleRegistrationState) -> Tuple[str, bool, Optional[str]]:
    state.completed_steps.append("appeal")
    return ("Appeal filed against RTO decision. Review in 15 business days.", True, None)


# ──────────────────────────────────────────────────────────────────────
# OBSERVATION BUILDERS
# ──────────────────────────────────────────────────────────────────────

def _available_actions(state: VehicleRegistrationState) -> List[str]:
    actions = ["check_prerequisites", "check_status", "compare_documents", "evaluate_options"]

    if state.prerequisites_checked:
        actions.extend(["gather_document", "fix_document", "fill_form"])
    if state.form_filled:
        actions.append("pay_fee")
    if state.fee_paid:
        actions.append("book_appointment")
    if state.appointment_booked and state.fee_paid and state.form_filled:
        actions.append("submit_application")
    if state.application_submitted:
        actions.append("wait")
    if state.services_status.get("inspection") == "failed":
        actions.append("appeal_rejection")

    return sorted(set(actions))


def _action_hints(state: VehicleRegistrationState) -> str:
    """Phase-level status summary — describes WHERE in the process the citizen is.
    Does NOT prescribe specific actions. Agent reasons from available_actions,
    completed_steps, pending_issues, and services_status."""
    if not state.prerequisites_checked:
        return "Pre-registration compliance verification phase."
    if state.pending_issues:
        return "Issue resolution phase."
    if not state.application_submitted:
        return "Application preparation phase."
    if not state.inspection_done:
        return f"Inspection and processing phase. Day {state.simulated_day}."
    if not state.permanent_reg_issued:
        return f"Registration issuance phase. Day {state.simulated_day}."
    return "Process complete."


def _progress_pct(state: VehicleRegistrationState) -> float:
    total = 10
    done = 0
    if state.prerequisites_checked: done += 1
    if state.documents_compared: done += 1
    if state.form_filled: done += 1
    if state.fee_paid: done += 1
    if state.appointment_booked: done += 1
    if state.application_submitted: done += 1
    if state.inspection_done: done += 1
    if state.temp_reg_issued: done += 1
    if state.permanent_reg_issued: done += 2
    # Deduct for unresolved issues
    if not state.pending_issues:
        done += 0  # no penalty
    return min(done / total, 1.0)


def _citizen_summary(citizen: CitizenProfile, comp: Dict[str, Any]) -> str:
    invoice = citizen.documents["dealer_invoice"]
    vehicle = invoice.fields.get("vehicle", "Unknown")
    vehicle_type = citizen.attributes.get('vehicle_class', 'four_wheeler_petrol') or "Unknown"
    price = invoice.fields.get("ex_showroom", 0)
    has_loan = comp["hypothecation_required"]

    lines = [
        f"Name: {citizen.name}, Age: {citizen.age}",
        f"Address: {citizen.current_address}",
        f"Aadhaar: {citizen.identifiers.get('aadhaar_number', '')}",
        f"Vehicle: {vehicle} ({vehicle_type})",
        f"Ex-showroom price: ₹{price}",
        f"Dealer: {invoice.fields.get('dealer', 'N/A')}",
        f"Loan vehicle: {'Yes' if has_loan else 'No'}",
    ]
    return "\n".join(lines)


def build_initial_observation(state: VehicleRegistrationState) -> Observation:
    citizen = state.citizen
    comp = state.complications

    return Observation(
        task_id=TASK_ID,
        task_description=(
            "Register a newly purchased vehicle at the Regional Transport Office (RTO). "
            "The process involves document verification, compliance checks, fee payment, "
            "vehicle inspection, and obtaining permanent registration. Complications may "
            "include expired insurance, missing compliance certificates, or identification discrepancies."
        ),
        difficulty=DIFFICULTY,
        citizen_summary=_citizen_summary(citizen, comp),
        citizen_documents={k: v.doc_type for k, v in citizen.documents.items()},
        current_phase="pre_registration",
        services_status=state.get_services_status(),
        completed_steps=[],
        pending_issues=[],
        available_actions=_available_actions(state),
        status_summary=_action_hints(state),
        progress_pct=0.0,
        steps_taken=0,
        max_steps=MAX_STEPS,
        simulated_day=0,
    )


def build_observation(state: VehicleRegistrationState, result_msg: str,
                      success: bool, error: Optional[str]) -> Observation:
    citizen = state.citizen
    comp = state.complications

    return Observation(
        task_id=TASK_ID,
        task_description=(
            "Register a newly purchased vehicle at the Regional Transport Office (RTO). "
            "The process involves document verification, compliance checks, fee payment, "
            "vehicle inspection, and obtaining permanent registration."
        ),
        difficulty=DIFFICULTY,
        citizen_summary=_citizen_summary(citizen, comp),
        citizen_documents={k: v.doc_type for k, v in citizen.documents.items()},
        current_phase=state.current_phase,
        services_status=state.get_services_status(),
        completed_steps=list(state.completed_steps),
        pending_issues=list(state.pending_issues),
        last_action=state.completed_steps[-1] if state.completed_steps else None,
        last_action_result=result_msg,
        last_action_success=success,
        last_action_error=error,
        available_actions=_available_actions(state),
        status_summary=_action_hints(state),
        progress_pct=_progress_pct(state),
        steps_taken=state.steps_taken,
        max_steps=MAX_STEPS,
        simulated_day=state.simulated_day,
    )


def is_task_done(state: VehicleRegistrationState) -> bool:
    return state.permanent_reg_issued
