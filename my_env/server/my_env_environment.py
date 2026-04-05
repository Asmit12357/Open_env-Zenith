import json
import random
import os
from uuid import uuid4
from typing import Optional, Dict, Any
from pydantic import Field

from openenv.core.env_server.interfaces import Environment, Action, Observation
from openenv.core.env_server.types import State

class MyAction(Action):
    """The AI Agent's diagnostic decision."""
    action: str = Field(
        ..., 
        description="The triage level: 'Home Care', 'Clinic', or 'Emergency'"
    )

class MyObservation(Observation):
    """What the AI Agent sees and the feedback it receives."""
    echoed_message: str = Field(default="")
    message_length: int = Field(default=0)
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MyEnvironment(Environment):
    """
    Medical Triage RL Environment.
    Strictly follows the 0.0 - 1.0 Reward Scale for Scaler/Meta Hackathon.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize and load the symptom bank."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Load tasks from your friend's JSON file
        data_path = os.path.join(os.path.dirname(__file__), 'medical_tasks.json')
        try:
            with open(data_path, 'r') as f:
                self.tasks = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.tasks = [{"id": 0, "symptoms": "Test: Headache", "correct_triage": "Home Care"}]
            
        self.current_task = None

    def reset(self) -> MyObservation:
        """Starts a new patient case."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_task = random.choice(self.tasks)

        return MyObservation(
            echoed_message=f"NEW PATIENT: {self.current_task['symptoms']}",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: MyAction) -> MyObservation:
        """
        Calculates reward on a 0.0 to 1.0 scale as per hackathon rules.
        """
        self._state.step_count += 1
        
        agent_choice = action.message.strip().lower() 
        actual_choice = self.current_task['correct_triage'].lower()
        
        # --- ADJUSTED REWARD LOGIC (0.0 to 1.0) ---
        reward = 0.0
        
        if agent_choice == actual_choice:
            reward = 1.0  # Perfect Diagnosis
        else:
            # THE "ANTI-DOOM" PENALTY (Over-diagnosis)
            if actual_choice == "home care" and agent_choice == "emergency":
                reward = 0.1  # Low score for causing panic
            
            # THE "NEGLIGENCE" PENALTY (Under-diagnosis)
            elif actual_choice == "emergency" and agent_choice == "home care":
                reward = 0.0  # Zero for life-threatening mistake
            
            else:
                reward = 0.4  # Minor mismatch (e.g., Clinic vs Home Care)

        return MyObservation(
            echoed_message=f"Diagnosis: {agent_choice}. Reality: {actual_choice}",
            message_length=len(agent_choice),
            done=True, 
            reward=reward,
            metadata={
                "task_id": self.current_task.get("id", 0),
                "step": self._state.step_count
            },
        )

    @property
    def state(self) -> State:
        return self._state
