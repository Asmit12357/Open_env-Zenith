import sys
import os

# Absolute path injection — must be before any local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import uvicorn
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from openenv.core.env_server.http_server import create_app

from my_env.models import MyAction, MyObservation
from server.my_env_environment import MyEnvironment

# --- Create main OpenEnv FastAPI app ---
app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="MedicalTriageRL",
    max_concurrent_envs=5,
)

# Shared environment instance for /tasks and /grader endpoints
_env_instance = MyEnvironment()


# ------------------------------------------------------------------
# Standard endpoints
# ------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "status": "online",
        "environment": "Medical Triage RL",
        "description": "Multi-turn clinical interview RL environment. Agent asks clarifying questions then triages.",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grader"],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "medical-triage-rl"}


# ------------------------------------------------------------------
# /tasks — task catalog for validators and agents
# ------------------------------------------------------------------

@app.get("/tasks")
def get_tasks():
    """
    Returns the full task catalog.
    Validators call this to enumerate tasks and verify 3+ exist.
    """
    return _env_instance.get_task_catalog()


# ------------------------------------------------------------------
# /grader — on-demand grading endpoint
# ------------------------------------------------------------------

class GraderRequest(BaseModel):
    agent_choice: str
    turns_used: Optional[int] = 1
    task_seed: Optional[int] = 1


@app.post("/grader")
def grade_submission(request: GraderRequest):
    """
    Grade an agent's triage decision without running a full episode.
    Useful for validators to verify reward is in [0.0, 1.0].

    Body:
        agent_choice: one of "home care" | "clinic visit" | "urgent care" | "emergency"
        turns_used: how many turns the agent used (affects efficiency bonus)
        task_seed: which task to grade against (default 1)
    """
    # Load the appropriate task
    _env_instance.reset(seed=request.task_seed)

    result = _env_instance.grade(
        agent_choice=request.agent_choice,
        turns_used=request.turns_used,
    )

    if not (0.0 <= result["reward"] <= 1.0):
        raise HTTPException(
            status_code=500,
            detail=f"Reward {result['reward']} is out of valid range [0.0, 1.0]"
        )

    return JSONResponse(content=result)


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()