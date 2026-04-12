"""
Data models for the Medical Triage RL Environment.
Supports multi-turn clinical interview: agent asks questions, then decides.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class MyAction(BaseModel):
    """
    Two action types:
    - ask: ask a clarifying question about the patient (e.g. "What is the pain level?")
    - triage: submit final triage decision (one of the 4 categories)
    """
    action_type: str = Field(
        default="triage",
        description="Either 'ask' (clarifying question) or 'triage' (final decision)"
    )
    message: Optional[str] = Field(
        default=None,
        description="The question text (if ask) or triage category (if triage)"
    )
    # Legacy fields kept for backward compatibility
    action_id: Optional[int] = Field(default=0)
    treatment: Optional[str] = Field(default=None)


class MyObservation(Observation):
    """What the agent sees after each action."""

    # Core symptom / feedback text
    echoed_message: str = Field(
        default="",
        description="Patient symptoms on reset; clinical feedback after each step"
    )
    message_length: int = Field(
        default=0,
        description="Internal tracking field"
    )

    # Multi-turn state
    turn: int = Field(
        default=0,
        description="Current turn number (0 = just reset)"
    )
    max_turns: int = Field(
        default=5,
        description="Maximum turns allowed before forced triage"
    )
    turns_remaining: int = Field(
        default=5,
        description="How many turns are left"
    )

    # Clinical context — revealed progressively as agent asks questions
    patient_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Clinical details revealed so far (vitals, history, etc.)"
    )

    # What the agent can do right now
    available_actions: List[str] = Field(
        default_factory=lambda: ["ask", "triage"],
        description="Valid action_types at this step"
    )

    # Rich metadata for debugging and reward shaping
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Task ID, step count, reward breakdown, clinical explanation"
    )