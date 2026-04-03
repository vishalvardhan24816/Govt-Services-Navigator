"""
Government Services Navigator — Core Environment

Orchestrates task selection, citizen generation, state management,
and delegates to task-specific handlers.

Extends the official OpenEnv Environment ABC for spec compliance.

API:
  reset(seed, task=...) → Observation
  step(action)          → StepResult (via HTTP) / Observation (via OpenEnv)
  state                 → EnvironmentState (property)
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, Optional, Union

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from server.models import (
    Action,
    ActionType,
    Difficulty,
    EnvironmentState,
    Observation,
    ServiceStatus,
    StepResult,
    TaskId,
    Trajectory,
    TrajectoryStep,
)
from server.tasks import task_driving_licence as dl_task
from server.tasks import task_pan_aadhaar as pan_task
from server.tasks import task_passport as passport_task
from server.tasks import task_vehicle_registration as vr_task
from server.grader import grade_trajectory


class GovtServicesEnv(Environment):
    """
    Government Services Navigator environment.
    Extends openenv.core.env_server.Environment ABC.

    Manages episode lifecycle:
      1. reset(task=...) → generate citizen, compute ground truth, return initial obs
      2. step(action) → validate, execute, update state, return obs + reward
      3. state (property) → return full internal state for debugging

    Single-session environment (SUPPORTS_CONCURRENT_SESSIONS = False).
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    TASK_CONFIGS = {
        TaskId.PAN_AADHAAR_LINK: {
            "difficulty": Difficulty.EASY,
            "max_steps": pan_task.MAX_STEPS,
            "module": pan_task,
        },
        TaskId.PASSPORT_FRESH: {
            "difficulty": Difficulty.MEDIUM,
            "max_steps": passport_task.MAX_STEPS,
            "module": passport_task,
        },
        TaskId.DRIVING_LICENCE: {
            "difficulty": Difficulty.HARD,
            "max_steps": dl_task.MAX_STEPS,
            "module": dl_task,
        },
        TaskId.VEHICLE_REGISTRATION: {
            "difficulty": Difficulty.EXPERT,
            "max_steps": vr_task.MAX_STEPS,
            "module": vr_task,
        },
    }

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._rng = random.Random(seed)
        self._task_id: Optional[TaskId] = None
        self._task_state: Any = None  # PanAadhaarState | PassportState | DrivingLicenceState
        self._trajectory: Optional[Trajectory] = None
        self._done = False
        self._last_reward = 0.0
        self._episode_id: Optional[str] = None
        self._step_count = 0
        self._env_state = EnvironmentState()
        self._action_counts: Dict[str, int] = {}  # track per-action repeat counts
        self._ground_truth: Any = None

    @property
    def available_tasks(self):
        return [t.value for t in TaskId]

    def get_metadata(self) -> EnvironmentMetadata:
        """Return environment metadata for /metadata endpoint."""
        return EnvironmentMetadata(
            name="Government Services Navigator",
            description="Simulate navigating Indian government services: PAN-Aadhaar linking, passport application, and driving licence. Agent must diagnose issues, plan solutions, and execute multi-step bureaucratic workflows.",
            version="1.0.0",
        )

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> Observation:
        """
        Initialize a new episode.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Optional episode identifier.
            **kwargs: May contain 'task' key for task selection.

        Returns:
            Initial Observation with citizen profile and task description.
        """
        # Apply seed if provided
        if seed is not None:
            self._rng = random.Random(seed)

        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0

        # Extract task from kwargs (for HTTP API compat)
        task_name = kwargs.get("task", None)

        # Resolve task
        if task_name is None:
            self._task_id = self._rng.choice(list(TaskId))
        else:
            try:
                self._task_id = TaskId(task_name)
            except ValueError:
                valid = [t.value for t in TaskId]
                raise ValueError(f"Unknown task '{task_name}'. Valid: {valid}")

        config = self.TASK_CONFIGS[self._task_id]
        module = config["module"]

        # Generate citizen and compute ground truth
        citizen, complications = module.generate_citizen(self._rng)
        ground_truth = module.compute_ground_truth(citizen, complications)

        # Create task state
        if self._task_id == TaskId.PAN_AADHAAR_LINK:
            self._task_state = pan_task.PanAadhaarState(citizen, complications, ground_truth)
        elif self._task_id == TaskId.PASSPORT_FRESH:
            self._task_state = passport_task.PassportState(citizen, complications, ground_truth)
        elif self._task_id == TaskId.DRIVING_LICENCE:
            self._task_state = dl_task.DrivingLicenceState(citizen, complications, ground_truth)
        elif self._task_id == TaskId.VEHICLE_REGISTRATION:
            self._task_state = vr_task.VehicleRegistrationState(citizen, complications, ground_truth)

        # Init trajectory
        self._trajectory = Trajectory(
            task_id=self._task_id,
            citizen_id=ground_truth.citizen_id,
            ground_truth=ground_truth,
        )
        self._done = False
        self._last_reward = 0.0
        self._ground_truth = ground_truth
        self._action_counts = {}

        # Build initial observation
        obs = module.build_initial_observation(self._task_state)
        return obs

    def reset_for_http(self, task_name: Optional[str] = None, seed: Optional[int] = None) -> Observation:
        """HTTP-friendly reset that accepts task_name directly."""
        return self.reset(seed=seed, task=task_name)

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs) -> Observation:
        """
        Execute one action in the environment.
        Conforms to OpenEnv Environment.step() signature.

        Args:
            action: Action with action_type and parameters.
            timeout_s: Optional timeout (unused, for API compat).

        Returns:
            Observation with done/reward set (OpenEnv pattern).
        """
        result = self.step_for_http(action)
        # Return observation with done/reward baked in for OpenEnv compat
        return result.observation

    def step_for_http(self, action: Action) -> StepResult:
        """
        Execute one action — returns full StepResult for HTTP API.

        Args:
            action: Action with action_type and parameters.

        Returns:
            StepResult with observation, reward, done flag, and info dict.
        """
        if self._task_id is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        config = self.TASK_CONFIGS[self._task_id]
        module = config["module"]
        max_steps = config["max_steps"]

        # Track action usage for anti-spam.
        # Key includes ALL params so fix_document(insurance) ≠ fix_document(chassis)
        # and take_test(written) ≠ take_test(practical).
        param_sig = ",".join(f"{k}={v}" for k, v in sorted(action.parameters.items()) if v)
        action_key = f"{action.action_type.value}:{param_sig}" if param_sig else action.action_type.value
        self._action_counts[action_key] = self._action_counts.get(action_key, 0) + 1

        # Execute action via task handler
        result_msg, success, error = module.handle_action(self._task_state, action)

        # Check if task is done (from state or max steps)
        task_done = getattr(self._task_state, "done", False)
        steps_taken = self._task_state.steps_taken
        hit_max = steps_taken >= max_steps

        self._done = task_done or hit_max

        # Calculate progress reward with anti-hacking penalties
        progress = self._task_state.get_progress()
        optimal = self._ground_truth.optimal_steps if self._ground_truth else max_steps
        penalty = 0.0

        # Penalty 1: wasted steps — only kicks in after 2x optimal (generous buffer)
        threshold = optimal * 2
        if steps_taken > threshold:
            penalty += 0.01 * (steps_taken - threshold)  # 1% per step after 2x optimal

        # Penalty 2: repeating exact same action+params excessively (>3 uses).
        # State-advancing actions (check_status, wait) are excluded because each
        # call changes simulation day/status — repeats are legitimate process
        # management, not spam. The step count penalty already handles abuse.
        state_advancing = {"check_status", "wait"}
        action_uses = self._action_counts[action_key]
        if action.action_type.value not in state_advancing and action_uses > 3:
            penalty += 0.02 * (action_uses - 3)  # 2% per extra repeat beyond 3

        reward = max(progress - penalty, 0.0)

        # If done, compute final grader score
        info: Dict[str, Any] = {
            "success": success,
            "steps_taken": steps_taken,
            "simulated_day": self._task_state.simulated_day,
        }

        if self._done:
            final_score = grade_trajectory(self._task_id, self._task_state, self._trajectory)
            reward = final_score.score
            info["final_grade"] = {
                "score": final_score.score,
                "breakdown": [
                    {"name": d.name, "score": d.score, "weight": d.weight, "feedback": d.feedback}
                    for d in final_score.breakdown
                ],
                "feedback": final_score.feedback,
                "task_completed": final_score.task_completed,
            }
            if hit_max and not task_done:
                info["truncated"] = True
                info["truncation_reason"] = f"Hit maximum steps ({max_steps})"

        self._last_reward = reward
        self._step_count += 1

        # Record trajectory step
        if self._trajectory is not None:
            self._trajectory.steps.append(TrajectoryStep(
                step_number=steps_taken,
                action=action,
                observation_summary=result_msg[:200] if result_msg else (error or ""),
                reward=reward,
                success=success,
                error=error,
                simulated_day=self._task_state.simulated_day,
            ))
            self._trajectory.total_reward = reward
            self._trajectory.completed = task_done

        # Build observation with done/reward for OpenEnv compat
        obs = module.build_observation(self._task_state, result_msg, success, error)

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    @property
    def state(self) -> EnvironmentState:
        """Return complete internal state for debugging/inspection."""
        if self._task_id is None:
            return EnvironmentState(
                episode_id=self._episode_id,
                step_count=self._step_count,
            )

        ts = self._task_state
        config = self.TASK_CONFIGS[self._task_id]

        # Build trajectory records
        traj_records = []
        if self._trajectory:
            for step in self._trajectory.steps:
                traj_records.append({
                    "step": step.step_number,
                    "action": step.action.action_type.value,
                    "params": step.action.parameters,
                    "reward": step.reward,
                    "success": step.success,
                    "error": step.error,
                })

        current_phase = getattr(ts, "current_phase", "main")

        # Build services status
        services = ts.get_services_status()
        svc_status = {}
        for k, v in services.items():
            try:
                svc_status[k] = ServiceStatus(v)
            except ValueError:
                svc_status[k] = ServiceStatus.NOT_STARTED

        # Build documents status
        from server.models import DocumentStatus
        doc_status = {}
        for doc_id, doc in ts.citizen.documents.items():
            doc_status[doc_id] = doc.status

        return EnvironmentState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            difficulty=config["difficulty"],
            citizen=ts.citizen,
            current_phase=current_phase,
            services_status=svc_status,
            documents_status=doc_status,
            completed_steps=list(ts.completed_steps),
            pending_issues=list(ts.pending_issues),
            simulated_day=ts.simulated_day,
            done=self._done,
            trajectory=traj_records,
        )
