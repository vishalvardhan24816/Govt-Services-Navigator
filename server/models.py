"""
Typed Pydantic models for the Government Services Navigator environment.

Defines the contract between:
  - Agent ↔ Environment (Action, Observation)
  - Environment ↔ Grader (Reward, TrajectoryStep)
  - Environment internals (CitizenProfile, ServiceState, GroundTruth)

Extends official OpenEnv base classes for spec compliance.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)


# ──────────────────────────────────────────────────────────────────────
# ENUMS — Typed constants for action types, service IDs, document types
# ──────────────────────────────────────────────────────────────────────

class TaskId(str, enum.Enum):
    PAN_AADHAAR_LINK = "pan_aadhaar_link"
    PASSPORT_FRESH = "passport_fresh"
    DRIVING_LICENCE = "driving_licence"
    VEHICLE_REGISTRATION = "vehicle_registration"


class ActionType(str, enum.Enum):
    # Diagnostic actions (show thinking / reasoning)
    CHECK_PREREQUISITES = "check_prerequisites"
    COMPARE_DOCUMENTS = "compare_documents"
    EVALUATE_OPTIONS = "evaluate_options"
    CHECK_ELIGIBILITY = "check_eligibility"
    CHECK_STATUS = "check_status"

    # Execution actions (do things)
    GATHER_DOCUMENT = "gather_document"
    FILL_FORM = "fill_form"
    PAY_FEE = "pay_fee"
    BOOK_APPOINTMENT = "book_appointment"
    SUBMIT_APPLICATION = "submit_application"
    FIX_DOCUMENT = "fix_document"
    TAKE_TEST = "take_test"
    APPEAL_REJECTION = "appeal_rejection"
    WAIT = "wait"


class ServiceStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentStatus(str, enum.Enum):
    MISSING = "missing"
    NOT_AVAILABLE = "not_available"
    PRESENT = "present"
    INVALID = "invalid"
    EXPIRED = "expired"
    SUBMITTED = "submitted"
    VERIFIED = "verified"


class Difficulty(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


# ──────────────────────────────────────────────────────────────────────
# CITIZEN PROFILE — Generated per episode
# ──────────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """A single document held by the citizen."""
    doc_type: str
    status: DocumentStatus = DocumentStatus.PRESENT
    fields: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class CitizenProfile(BaseModel):
    """Profile of the citizen the agent is helping. Generated on reset()."""
    name: str
    age: int
    gender: str
    current_address: str

    # Documents the citizen has (may have mismatches / issues)
    documents: Dict[str, DocumentInfo] = Field(default_factory=dict)

    # Generic complication flags (task-agnostic metadata for the grader/state endpoint)
    complication_flags: Dict[str, bool] = Field(default_factory=dict)

    # Task-specific identifiers (e.g. pan_number, aadhaar_number) — generic dict
    identifiers: Dict[str, str] = Field(default_factory=dict)

    # Task-specific attributes (e.g. vehicle_class) — generic dict
    attributes: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


# ──────────────────────────────────────────────────────────────────────
# ACTION — What the agent sends to the environment
# Extends official OpenEnv Action base class
# ──────────────────────────────────────────────────────────────────────

class Action(BaseAction):
    """
    An action the agent takes in the environment.
    Extends openenv.core.env_server.types.Action.

    action_type: The type of action (from ActionType enum)
    parameters: Action-specific parameters (e.g., which document to gather,
                which form fields to fill, which service to check)
    """
    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", frozen=True)


# ──────────────────────────────────────────────────────────────────────
# OBSERVATION — What the environment returns to the agent
# Extends official OpenEnv Observation base class
# ──────────────────────────────────────────────────────────────────────

class Observation(BaseObservation):
    """
    What the agent sees after each action (or on reset).
    Extends openenv.core.env_server.types.Observation.

    Inherits: done (bool), reward (float|None), metadata (Dict)

    Designed to be clear enough for any LLM (including Nemotron)
    to understand the situation and decide the next action.
    """
    # Task context
    task_id: TaskId
    task_description: str
    difficulty: Difficulty

    # Citizen info (what the agent knows)
    citizen_summary: str
    citizen_documents: Dict[str, str] = Field(default_factory=dict)

    # Current state
    current_phase: str = "initial"
    services_status: Dict[str, str] = Field(default_factory=dict)
    completed_steps: List[str] = Field(default_factory=list)
    pending_issues: List[str] = Field(default_factory=list)

    # Last action result
    last_action: Optional[str] = None
    last_action_result: str = ""
    last_action_success: bool = True
    last_action_error: Optional[str] = None

    # Available actions (critical for Nemotron to know what it can do)
    available_actions: List[str] = Field(default_factory=list)
    status_summary: str = ""

    # Progress
    progress_pct: float = 0.0
    steps_taken: int = 0
    max_steps: int = 25

    # Simulated time
    simulated_day: int = 0

    # Override base config to allow extra fields
    model_config = ConfigDict(extra="allow", frozen=True)


# ──────────────────────────────────────────────────────────────────────
# REWARD — Scoring output from the grader
# ──────────────────────────────────────────────────────────────────────

class DimensionScore(BaseModel):
    """Score for a single grading dimension."""
    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    feedback: str = ""

    model_config = {"frozen": True}


class Reward(BaseModel):
    """
    Final grading output. Returned at episode end.
    Contains both the aggregate score and per-dimension breakdown.
    """
    score: float = Field(ge=0.0, le=1.0)
    breakdown: List[DimensionScore] = Field(default_factory=list)
    feedback: str = ""
    trajectory_length: int = 0
    task_completed: bool = False

    model_config = {"frozen": True}


# ──────────────────────────────────────────────────────────────────────
# STEP RESULT — What step() returns (kept for backward compat, wraps Observation)
# ──────────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Return type of env.step(action) for our HTTP API."""
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


# ──────────────────────────────────────────────────────────────────────
# GROUND TRUTH — Precomputed correct solution (hidden from agent)
# ──────────────────────────────────────────────────────────────────────

class GroundTruthCheckpoint(BaseModel):
    """A required checkpoint the agent should hit."""
    step_id: str
    action_type: ActionType
    description: str
    required: bool = True
    order_matters: bool = False  # if True, must come before subsequent checkpoints

    model_config = {"frozen": True}


class GroundTruth(BaseModel):
    """
    Precomputed correct solution for a generated episode.
    The grader compares the agent's trajectory against this.
    The agent NEVER sees this — it's hidden state.
    """
    task_id: TaskId
    citizen_id: str  # unique ID for this episode's citizen

    # Required checkpoints (must-hit actions)
    required_checkpoints: List[GroundTruthCheckpoint] = Field(default_factory=list)

    # Forbidden actions (must-not-do)
    forbidden_actions: List[str] = Field(default_factory=list)

    # Valid end states
    valid_completions: List[str] = Field(default_factory=list)

    # Expected issues the agent should detect
    expected_issues: List[str] = Field(default_factory=list)

    # Optimal path length (for efficiency scoring)
    optimal_steps: int = 5

    # Correct form values (for form accuracy scoring)
    correct_form_values: Dict[str, Any] = Field(default_factory=dict)

    # Correct fee amount
    correct_fee: Optional[float] = None

    model_config = {"frozen": True}


# ──────────────────────────────────────────────────────────────────────
# TRAJECTORY — Record of agent's actions (for grading)
# ──────────────────────────────────────────────────────────────────────

class TrajectoryStep(BaseModel):
    """A single step in the agent's trajectory."""
    step_number: int
    action: Action
    observation_summary: str
    reward: float
    success: bool
    error: Optional[str] = None
    simulated_day: int = 0

    model_config = {"frozen": True}


class Trajectory(BaseModel):
    """Full record of an episode for grading."""
    task_id: TaskId
    citizen_id: str
    steps: List[TrajectoryStep] = Field(default_factory=list)
    total_reward: float = 0.0
    completed: bool = False
    ground_truth: Optional[GroundTruth] = None

    model_config = {"frozen": False}  # mutable — we append steps during episode


# ──────────────────────────────────────────────────────────────────────
# ENVIRONMENT STATE — Extends official OpenEnv State base class
# ──────────────────────────────────────────────────────────────────────

class EnvironmentState(BaseState):
    """
    Complete internal state of the environment.
    Extends openenv.core.env_server.types.State.
    Inherits: episode_id (str), step_count (int)

    Returned by state() endpoint. Includes everything needed
    to inspect or debug the environment.
    """
    task_id: Optional[TaskId] = None
    difficulty: Optional[Difficulty] = None
    citizen: Optional[CitizenProfile] = None
    current_phase: str = "not_started"
    services_status: Dict[str, ServiceStatus] = Field(default_factory=dict)
    documents_status: Dict[str, DocumentStatus] = Field(default_factory=dict)
    completed_steps: List[str] = Field(default_factory=list)
    pending_issues: List[str] = Field(default_factory=list)
    simulated_day: int = 0
    done: bool = False
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow", validate_assignment=True)
