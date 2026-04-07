"""
Data models for the Medical Verifier Environment.
Defines the communication between the AI Agent and the Diagnostic Triage Logic.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field # Added Field here
from typing import Optional, Dict, Any # Added Dict and Any here

class MyAction(BaseModel):
    # Make fields optional or provide defaults to prevent "Field Required" crashes
    action_id: Optional[int] = 0
    treatment: Optional[str] = None
    message: Optional[str] = None

class MyObservation(Observation):
    """What the AI Agent sees and the feedback it receives."""

    echoed_message: str = Field(
        default="", 
        description="The symptoms or the feedback from the last diagnosis"
    )
    message_length: int = Field(
        default=0, 
        description="Technical field for internal tracking"
    )
    # Metadata is great for debugging rewards and clinical explanations
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Contains rewards, task IDs, and clinical explanations"
    )