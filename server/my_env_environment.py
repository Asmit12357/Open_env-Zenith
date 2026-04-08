import json
import os
import random
from uuid import uuid4
from typing import Optional, Dict, Any
from pydantic import Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from my_env.models import MyAction, MyObservation

class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        # Initializing state with a unique ID
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Absolute pathing to prevent FileNotFoundError in Docker
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'medical_tasks.json')
        
        try:
            with open(data_path, 'r') as f:
                self.tasks = json.load(f)
        except Exception:
            # Hard fallback to prevent 500 if file is missing
            self.tasks = [{"id": 1, "symptoms": "Minor headache", "correct_triage": "Home Care"}]
            
        self.current_task = self.tasks[0] # Default to first task
        self.active_task_id = self.current_task.get("id", 1)

    def reset(self, seed=None, options=None):
        # FIX: Ensure safe_seed is always an int
        safe_seed = int(seed) if (seed is not None) else 0
        random.seed(safe_seed)
        
        self._state.step_count = 0
        
        # FIX: Adjusted math so Seed 11 = ID 11 (if IDs start at 1)
        # Using (safe_seed - 1) handles the zero-index offset
        idx = (safe_seed - 1) % len(self.tasks) if safe_seed > 0 else 0
        self.current_task = self.tasks[idx]
        self.active_task_id = self.current_task.get("id", idx + 1)
        
        obs_text = self.current_task.get("symptoms", "No symptoms listed")
        
        # Return observation with reward 0 and done False
        return MyObservation(
            echoed_message=obs_text,
            message_length=len(obs_text),
            done=False,
            reward=0.0
        )

    def step(self, action: MyAction) -> MyObservation:
        """
        Executes a step. Validates agent input against expected triage categories.
        """
        self._state.step_count += 1
        
        # DEFENSIVE: If action is somehow None or empty
        if not action:
            return MyObservation(echoed_message="Error: No action provided", reward=0.0, done=True)

        # Extraction logic with fallbacks
        msg = getattr(action, 'message', "") or ""
        treat = getattr(action, 'treatment', "") or ""
        
        # Standardize the choice
        agent_choice = (msg or treat).strip().lower()
        actual_choice = str(self.current_task.get('correct_triage', "")).strip().lower()

        # Define the set of valid triage categories (nonsense check)
        valid_categories = {"home care", "clinic visit", "urgent care", "emergency"}
        
        # Reward Logic
        reward = 0.0

        if agent_choice not in valid_categories:
            # Complete nonsense gets nothing
            reward = 0.0
        elif agent_choice == actual_choice:
            # Perfect match
            reward = 1.0
        else:
            # Reward shaping for valid but incorrect categories
            if actual_choice == "home care" and agent_choice == "emergency":
                # Over-cautious (Safe but expensive)
                reward = 0.1
            elif actual_choice == "emergency" and agent_choice == "home care":
                # Dangerous mistake
                reward = 0.0
            else:
                # Valid triage category, but not the correct one
                reward = 0.5

        return MyObservation(
            echoed_message=f"Diagnosis: {agent_choice}. Reality: {actual_choice}",
            message_length=len(agent_choice),
            done=True, 
            reward=reward,
            metadata={
                "task_id": self.active_task_id,
                "step": self._state.step_count,
                "explanation": self.current_task.get("explanation", "N/A")
            },
        )

    @property
    def state(self) -> State:
        return self._state