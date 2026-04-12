"""
Medical Triage RL Environment — Multi-Turn Clinical Interview.

Episode flow:
  reset()  → patient arrives with vague/specific symptoms
  step(ask)    → agent asks a clarifying question, receives clinical detail
  step(ask)    → agent can ask up to (max_turns - 1) questions
  step(triage) → agent submits final triage decision → episode ends

Reward shaping:
  - Correct triage:       +1.0  (base)
  - Efficiency bonus:     +0.1 per unused turn (rewards asking fewer questions)
  - Step penalty:         -0.05 per ask action (encourages concise reasoning)
  - Wrong but valid:      +0.5  (adjacent category)
  - Dangerous mistake:    0.0   (e.g. home care for emergency)
  - Invalid action:       -0.1
  - Forced triage (timeout): scored on best guess, no efficiency bonus
"""

import json
import os
import random
from uuid import uuid4
from typing import Optional, Dict, Any, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from my_env.models import MyAction, MyObservation

VALID_TRIAGE = {"home care", "clinic visit", "urgent care", "emergency"}
MAX_TURNS = 5


class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "medical_tasks.json")

        try:
            with open(data_path, "r") as f:
                self.all_tasks = json.load(f)
        except Exception:
            # Hard fallback
            self.all_tasks = [
                {
                    "id": 1, "difficulty": "easy", "task_id": "task_easy",
                    "symptoms": "Sharp crushing chest pain, sweating, shortness of breath.",
                    "correct_triage": "emergency",
                    "explanation": "Classic heart attack presentation.",
                    "clarifying_info": {
                        "pain_level": "9/10", "duration": "20 minutes",
                        "vitals": "BP 160/100, HR 110", "history": "58yo smoker",
                        "additional": "Jaw pain present"
                    },
                    "red_flags": ["chest pain", "sweating"]
                }
            ]

        # Group tasks by task_id for easy lookup
        self.task_catalog = {}
        for t in self.all_tasks:
            tid = t.get("task_id", "task_easy")
            if tid not in self.task_catalog:
                self.task_catalog[tid] = []
            self.task_catalog[tid].append(t)

        self.current_task: Dict = self.all_tasks[0]
        self._turn: int = 0
        self._revealed_keys: List[str] = []
        self._asked_questions: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> MyObservation:
        safe_seed = int(seed) if seed is not None else 0
        random.seed(safe_seed)

        # Select task — seed maps into the full task list
        idx = (safe_seed - 1) % len(self.all_tasks) if safe_seed > 0 else 0
        self.current_task = self.all_tasks[idx]

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._turn = 0
        self._revealed_keys = []
        self._asked_questions = []

        symptoms = self.current_task.get("symptoms", "No symptoms listed.")

        return MyObservation(
            echoed_message=symptoms,
            message_length=len(symptoms),
            turn=0,
            max_turns=MAX_TURNS,
            turns_remaining=MAX_TURNS,
            patient_context={},
            available_actions=["ask", "triage"],
            done=False,
            reward=0.0,
            metadata={
                "task_id": self.current_task.get("task_id"),
                "difficulty": self.current_task.get("difficulty"),
                "step": 0,
                "hint": (
                    "You can 'ask' up to 4 clarifying questions, then submit your 'triage' decision. "
                    "Valid triage categories: home care | clinic visit | urgent care | emergency"
                ),
            },
        )

    def step(self, action: MyAction) -> MyObservation:
        self._state.step_count += 1
        self._turn += 1

        if not action:
            return self._error_obs("No action provided.", reward=-0.1)

        action_type = (getattr(action, "action_type", None) or "triage").strip().lower()
        message = (
            getattr(action, "message", None)
            or getattr(action, "treatment", None)
            or ""
        ).strip().lower().rstrip(".")

        # ---- FORCED TRIAGE: ran out of turns ----
        if self._turn >= MAX_TURNS and action_type == "ask":
            action_type = "triage"
            message = message or "emergency"  # force a guess

        # ---- ASK action ----
        if action_type == "ask":
            return self._handle_ask(message)

        # ---- TRIAGE action ----
        if action_type == "triage":
            return self._handle_triage(message)

        # ---- Unknown action type ----
        return self._error_obs(
            f"Unknown action_type '{action_type}'. Use 'ask' or 'triage'.",
            reward=-0.1,
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Task catalog (used by /tasks endpoint)
    # ------------------------------------------------------------------

    def get_task_catalog(self) -> Dict:
        return {
            "tasks": [
                {
                    "task_id": "task_easy",
                    "description": "Clear-cut presentations — correct triage should be obvious from symptoms alone.",
                    "difficulty": "easy",
                    "max_turns": MAX_TURNS,
                    "valid_actions": ["ask", "triage"],
                    "triage_categories": list(VALID_TRIAGE),
                },
                {
                    "task_id": "task_medium",
                    "description": "Moderate presentations — requires 1-2 clarifying questions for confident triage.",
                    "difficulty": "medium",
                    "max_turns": MAX_TURNS,
                    "valid_actions": ["ask", "triage"],
                    "triage_categories": list(VALID_TRIAGE),
                },
                {
                    "task_id": "task_hard",
                    "description": "Ambiguous or vague presentations — requires careful questioning to avoid dangerous mistakes.",
                    "difficulty": "hard",
                    "max_turns": MAX_TURNS,
                    "valid_actions": ["ask", "triage"],
                    "triage_categories": list(VALID_TRIAGE),
                },
            ]
        }

    # ------------------------------------------------------------------
    # Grader (used by /grader endpoint)
    # ------------------------------------------------------------------

    def grade(self, agent_choice: str, turns_used: int) -> Dict:
        agent_choice = agent_choice.strip().lower().rstrip(".")
        actual = self.current_task.get("correct_triage", "").strip().lower()
        reward = self._compute_reward(agent_choice, actual, turns_used)
        return {
            "agent_choice": agent_choice,
            "correct_choice": actual,
            "reward": reward,
            "is_correct": agent_choice == actual,
            "explanation": self.current_task.get("explanation", ""),
            "difficulty": self.current_task.get("difficulty", ""),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_ask(self, question: str) -> MyObservation:
        """Reveal one piece of clinical information based on the question."""
        clarifying = self.current_task.get("clarifying_info", {})
        info_keys = list(clarifying.keys())

        # Match question keywords to clinical info categories
        keyword_map = {
            "pain": "pain_level",
            "level": "pain_level",
            "scale": "pain_level",
            "duration": "duration",
            "long": "duration",
            "when": "duration",
            "start": "duration",
            "vital": "vitals",
            "bp": "vitals",
            "blood pressure": "vitals",
            "heart rate": "vitals",
            "temperature": "vitals",
            "temp": "vitals",
            "spo2": "vitals",
            "oxygen": "vitals",
            "history": "history",
            "age": "history",
            "medical": "history",
            "background": "history",
            "allerg": "history",
            "medication": "history",
            "additional": "additional",
            "more": "additional",
            "else": "additional",
            "other": "additional",
            "symptom": "additional",
        }

        matched_key = None
        for keyword, info_key in keyword_map.items():
            if keyword in question and info_key in clarifying and info_key not in self._revealed_keys:
                matched_key = info_key
                break

        # If no keyword match, reveal the next unrevealed piece in order
        if not matched_key:
            for key in info_keys:
                if key not in self._revealed_keys:
                    matched_key = key
                    break

        self._asked_questions.append(question)

        if matched_key and matched_key in clarifying:
            self._revealed_keys.append(matched_key)
            revealed_info = clarifying[matched_key]
            feedback = f"Clinical info — {matched_key.replace('_', ' ').title()}: {revealed_info}"
        else:
            feedback = "No additional clinical information available at this time."

        # Build up progressive patient context
        patient_ctx = {k: clarifying[k] for k in self._revealed_keys if k in clarifying}

        turns_left = MAX_TURNS - self._turn

        return MyObservation(
            echoed_message=feedback,
            message_length=len(feedback),
            turn=self._turn,
            max_turns=MAX_TURNS,
            turns_remaining=turns_left,
            patient_context=patient_ctx,
            available_actions=["ask", "triage"] if turns_left > 1 else ["triage"],
            done=False,
            reward=-0.05,  # small step penalty to encourage efficiency
            metadata={
                "task_id": self.current_task.get("task_id"),
                "difficulty": self.current_task.get("difficulty"),
                "step": self._state.step_count,
                "revealed_info": matched_key,
                "questions_asked": len(self._asked_questions),
                "reward_breakdown": {"step_penalty": -0.05},
            },
        )

    def _handle_triage(self, agent_choice: str) -> MyObservation:
        """Score the agent's final triage decision."""
        actual = self.current_task.get("correct_triage", "").strip().lower()
        turns_used = self._turn
        reward = self._compute_reward(agent_choice, actual, turns_used)

        result_text = (
            f"Triage submitted: '{agent_choice}'. "
            f"Correct answer: '{actual}'. "
            f"Reward: {reward:.2f}. "
            f"{self.current_task.get('explanation', '')}"
        )

        patient_ctx = {
            k: self.current_task["clarifying_info"][k]
            for k in self._revealed_keys
            if k in self.current_task.get("clarifying_info", {})
        }

        efficiency_bonus = max(0.0, (MAX_TURNS - turns_used) * 0.1) if agent_choice == actual else 0.0

        return MyObservation(
            echoed_message=result_text,
            message_length=len(agent_choice),
            turn=self._turn,
            max_turns=MAX_TURNS,
            turns_remaining=0,
            patient_context=patient_ctx,
            available_actions=[],
            done=True,
            reward=reward,
            metadata={
                "task_id": self.current_task.get("task_id"),
                "difficulty": self.current_task.get("difficulty"),
                "step": self._state.step_count,
                "agent_choice": agent_choice,
                "correct_choice": actual,
                "is_correct": agent_choice == actual,
                "questions_asked": len(self._asked_questions),
                "turns_used": turns_used,
                "explanation": self.current_task.get("explanation", ""),
                "reward_breakdown": {
                    "base_reward": 1.0 if agent_choice == actual else (
                        0.5 if agent_choice in VALID_TRIAGE else 0.0
                    ),
                    "efficiency_bonus": efficiency_bonus,
                    "step_penalties": -(len(self._asked_questions) * 0.05),
                    "total": reward,
                },
            },
        )

    def _compute_reward(self, agent_choice: str, actual: str, turns_used: int) -> float:
        """Dense reward function with efficiency shaping."""
        if agent_choice not in VALID_TRIAGE:
            return 0.0

        if agent_choice == actual:
            # Base reward + efficiency bonus for unused turns
            efficiency_bonus = max(0.0, (MAX_TURNS - turns_used) * 0.1)
            step_penalties = -(max(0, turns_used - 1) * 0.05)
            raw = 1.0 + efficiency_bonus + step_penalties
            return round(min(0.99, max(0.01, raw)), 4)

        # Dangerous under-triage
        if actual == "emergency" and agent_choice == "home care":
            return 0.0

        # Dangerous over-triage (wastes emergency resources)
        if actual == "home care" and agent_choice == "emergency":
            return round(min(0.99, max(0.01, 0.1)), 4)

        # Adjacent/reasonable but wrong
        return round(min(0.99, max(0.01, 0.5)), 4)

    def _error_obs(self, message: str, reward: float = -0.1) -> MyObservation:
        return MyObservation(
            echoed_message=f"Error: {message}",
            message_length=0,
            turn=self._turn,
            max_turns=MAX_TURNS,
            turns_remaining=max(0, MAX_TURNS - self._turn),
            patient_context={},
            available_actions=["ask", "triage"],
            done=False,
            reward=reward,
            metadata={"error": message, "step": self._state.step_count},
        )