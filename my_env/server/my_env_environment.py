import json
import os
import random
from uuid import uuid4
from typing import Optional, Dict, Any
from pydantic import Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
# Ensure this relative import works with __init__.py in both folders
from ..models import MyAction, MyObservation

class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Path logic for JSON in same folder as this script
        data_path = os.path.join(os.path.dirname(__file__), 'medical_tasks.json')
        try:
            with open(data_path, 'r') as f:
                self.tasks = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback keys match the reset logic
            self.tasks = [{"id": 0, "symptoms": "Headache", "correct_triage": "Home Care"}]
            
        self.current_task = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
        
        # 1. Reproducible patient selection
        patient_index = seed % len(self.tasks) if seed is not None else 0
        self.current_task = self.tasks[patient_index]
        self._state.step_count = 0
        
        # 2. Store active task info in state for session safety
        self._state.metadata["active_task_id"] = self.current_task.get("id", patient_index)
       
        obs_text = self.current_task.get("symptoms", "No symptoms listed")
        
        return MyObservation(
            echoed_message=obs_text,
            message_length=len(obs_text),
            done=False,
            reward=0.0
        ), {}

    def step(self, action: MyAction) -> MyObservation:
        # 3. Increment step count immediately
        self._state.step_count += 1
        
        if not self.current_task:
            # Fallback if reset wasn't called properly
            self.current_task = self.tasks[0]
        
        # 4. Null-safe and whitespace-safe extraction
        msg = action.message if action.message else ""
        treat = action.treatment if action.treatment else ""
        
        # We strip both sides to ensure "Emergency " matches "emergency"
        agent_choice = (msg or treat).strip().lower()
        actual_choice = str(self.current_task.get('correct_triage', "")).strip().lower()
        
        # 5. Reward Logic (0.0 to 1.0)
        reward = 0.0
        if agent_choice == actual_choice:
            reward = 1.0
        else:
            if actual_choice == "home care" and agent_choice == "emergency":
                reward = 0.1  # Over-diagnosis penalty
            elif actual_choice == "emergency" and agent_choice == "home care":
                reward = 0.0  # Critical under-diagnosis penalty
            else:
                reward = 0.4  # Minor mismatch (e.g., Clinic vs Emergency)

        return MyObservation(
            echoed_message=f"Diagnosis: {agent_choice}. Reality: {actual_choice}",
            message_length=len(agent_choice),
            done=True, 
            reward=reward,
            metadata={
                "task_id": self._state.metadata.get("active_task_id", 0),
                "step": self._state.step_count,
                "explanation": self.current_task.get("explanation", "No explanation provided")
            },
        )

    @property
    def state(self) -> State:
        return self._state