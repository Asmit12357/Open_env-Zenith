import json
import os
from uuid import uuid4
from typing import Optional, Dict, Any
from pydantic import Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
# Ensure these match exactly what you named them in models.py
from models import MyAction, MyObservation 

class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Load tasks
        data_path = os.path.join(os.path.dirname(__file__), 'medical_tasks.json')
        try:
            with open(data_path, 'r') as f:
                self.tasks = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.tasks = [{"id": 0, "description": "Headache", "correct_triage": "Home Care"}]
            
        self.current_task = None

    def reset(self, seed=None, options=None):
        # 1. Initialize random generator if seed is provided
        if seed is not None:
            import random
            random.seed(seed)
        
        # 2. Pick the patient using the seed logic
        patient_index = seed % len(self.tasks) if seed is not None else 0
        self.current_task = self.tasks[patient_index]
        self._state.step_count = 0
        
        # 3. Construct Observation
        # Note: We use 'description' because that's what's in your JSON
        obs_text = self.current_task.get("description", "No description available")
        
        observation = MyObservation(
            echoed_message=obs_text,
            message_length=len(obs_text),
            done=False,
            reward=0.0
        )
        
        return observation, {}

    def step(self, action: MyAction) -> MyObservation:
        self._state.step_count += 1
        
        # FIX: Your MyAction model uses 'message' as an alias
        agent_choice = (action.message or action.treatment or "").strip().lower()
        actual_choice = self.current_task['correct_triage'].lower()
        
        # Reward Logic (0.0 to 1.0)
        reward = 0.0
        if agent_choice == actual_choice:
            reward = 1.0
        else:
            if actual_choice == "home care" and agent_choice == "emergency":
                reward = 0.1
            elif actual_choice == "emergency" and agent_choice == "home care":
                reward = 0.0
            else:
                reward = 0.4

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