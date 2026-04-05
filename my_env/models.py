"""
Data models for the Medical Verifier Environment.
Defines the communication between the AI Agent and the Diagnostic Triage Logic.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, Dict, Any


class MyAction(Action):
    """The AI Agent's diagnostic decision."""

    message: str = Field(
        ..., 
        description="The triage level: 'Home Care', 'Clinic', or 'Emergency'"
    )


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
    # We add metadata here so the agent can see its reward and the 'Correct' answer during training
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Contains rewards, task IDs, and clinical explanations"
    )
