"""
Government Services Navigator — FastAPI Server

Exposes the OpenEnv-compliant API:
  GET  /health       → health check (returns {"status": "healthy"})
  GET  /metadata     → environment metadata (name, description)
  GET  /schema       → action/observation/state JSON schemas
  POST /reset        → start new episode
  POST /step         → take action
  GET  /state        → get current state
  GET  /tasks        → list available tasks
  POST /mcp          → MCP JSON-RPC endpoint (minimal)

Also provides main() entry point for `openenv serve` / `uv run server`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.env import GovtServicesEnv
from server.models import Action, ActionType, Observation, EnvironmentState

app = FastAPI(
    title="Government Services Navigator",
    description="OpenEnv RL environment for training AI agents to navigate Indian government services",
    version="1.0.0",
)

# Global environment instance (stateful — persists across HTTP requests)
env = GovtServicesEnv()


# ──────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


class ActionRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────────────
# STANDARD OPENENV ENDPOINTS (for openenv validate)
# ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint — returns 200 for automated pings."""
    return {"status": "healthy", "environment": "govt-services-navigator"}


@app.get("/health")
def health():
    """Health check — returns {"status": "healthy"} per OpenEnv spec."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """Environment metadata — required by openenv validate."""
    meta = env.get_metadata()
    return {
        "name": meta.name,
        "description": meta.description,
        "version": meta.version or "1.0.0",
    }


@app.get("/schema")
def schema():
    """Action/Observation/State JSON schemas — required by openenv validate."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EnvironmentState.model_json_schema(),
    }


@app.post("/mcp")
def mcp(request: Dict[str, Any] = {}):
    """Minimal MCP JSON-RPC endpoint — required by openenv validate."""
    return {
        "jsonrpc": "2.0",
        "error": {
            "code": -32601,
            "message": "MCP not supported. Use HTTP /reset and /step endpoints.",
        },
        "id": request.get("id"),
    }


# ──────────────────────────────────────────────────────────────────────
# STATEFUL ENVIRONMENT ENDPOINTS
# ──────────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """Start a new episode."""
    global env

    try:
        if request.seed is not None:
            env = GovtServicesEnv(seed=request.seed)

        obs = env.reset_for_http(task_name=request.task)
        return {
            "observation": obs.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
def step(request: ActionRequest):
    """Take an action in the environment."""
    try:
        try:
            action_type = ActionType(request.action_type)
        except ValueError:
            valid = [a.value for a in ActionType]
            raise ValueError(f"Unknown action_type '{request.action_type}'. Valid: {valid}")

        action = Action(action_type=action_type, parameters=request.parameters)
        result = env.step_for_http(action)

        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
def get_state():
    """Get current environment state."""
    try:
        s = env.state
        return {"state": s.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")


@app.get("/tasks")
def list_tasks():
    """List available tasks with difficulty levels."""
    return {
        "tasks": [
            {
                "id": "pan_aadhaar_link",
                "name": "PAN-Aadhaar Linking",
                "difficulty": "easy",
                "description": "Link PAN card with Aadhaar on the Income Tax portal. Identity documents may contain discrepancies that must be resolved.",
                "max_steps": 15,
            },
            {
                "id": "passport_fresh",
                "name": "Fresh Passport Application",
                "difficulty": "medium",
                "description": "Apply for a fresh Indian passport through the Passport Seva Kendra. Involves document verification, application, and in-person verification.",
                "max_steps": 25,
            },
            {
                "id": "driving_licence",
                "name": "Driving Licence (LL + DL)",
                "difficulty": "hard",
                "description": "Obtain a driving licence in India. Two-phase process with age restrictions, medical requirements, and document validity constraints.",
                "max_steps": 30,
            },
            {
                "id": "vehicle_registration",
                "name": "New Vehicle Registration (RTO)",
                "difficulty": "expert",
                "description": "Register a newly purchased vehicle at the RTO. Involves compliance checks, vehicle inspection, and obtaining permanent registration.",
                "max_steps": 35,
            },
        ]
    }


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT — required by openenv validate for multi-mode deployment
# ──────────────────────────────────────────────────────────────────────

def main():
    """Entry point for `uv run server` / `openenv serve`."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
