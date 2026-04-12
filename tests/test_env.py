"""
Test suite for Medical Triage RL Environment.
Run with: pytest tests/ -v
"""

import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.my_env_environment import MyEnvironment, VALID_TRIAGE, MAX_TURNS
from my_env.models import MyAction, MyObservation


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def env():
    return MyEnvironment()


# ------------------------------------------------------------------
# Reset tests
# ------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(seed=1)
        assert isinstance(obs, MyObservation)

    def test_reset_has_symptoms(self, env):
        obs = env.reset(seed=1)
        assert len(obs.echoed_message) > 10, "Symptoms should be non-trivial"

    def test_reset_turn_is_zero(self, env):
        obs = env.reset(seed=1)
        assert obs.turn == 0

    def test_reset_not_done(self, env):
        obs = env.reset(seed=1)
        assert obs.done is False

    def test_reset_reward_is_zero(self, env):
        obs = env.reset(seed=1)
        assert obs.reward == 0.0

    def test_reset_has_available_actions(self, env):
        obs = env.reset(seed=1)
        assert "ask" in obs.available_actions
        assert "triage" in obs.available_actions

    def test_reset_different_seeds_give_different_patients(self, env):
        obs1 = env.reset(seed=1)
        obs2 = env.reset(seed=3)
        assert obs1.echoed_message != obs2.echoed_message


# ------------------------------------------------------------------
# Ask action tests
# ------------------------------------------------------------------

class TestAskAction:
    def test_ask_reveals_clinical_info(self, env):
        env.reset(seed=1)
        obs = env.step(MyAction(action_type="ask", message="What is the pain level?"))
        assert obs.reward == -0.05  # step penalty
        assert obs.done is False
        assert len(obs.echoed_message) > 5

    def test_ask_increments_turn(self, env):
        env.reset(seed=1)
        obs = env.step(MyAction(action_type="ask", message="How long have symptoms lasted?"))
        assert obs.turn == 1

    def test_ask_builds_patient_context(self, env):
        env.reset(seed=1)
        env.step(MyAction(action_type="ask", message="What are the vitals?"))
        obs = env.step(MyAction(action_type="ask", message="What is the history?"))
        assert len(obs.patient_context) >= 1

    def test_ask_reduces_turns_remaining(self, env):
        env.reset(seed=1)
        obs = env.step(MyAction(action_type="ask", message="Any additional symptoms?"))
        assert obs.turns_remaining == MAX_TURNS - 1

    def test_last_turn_forces_triage_only(self, env):
        env.reset(seed=1)
        # Burn through turns
        for _ in range(MAX_TURNS - 1):
            env.step(MyAction(action_type="ask", message="tell me more"))
        obs = env.step(MyAction(action_type="ask", message="one more question"))
        # Should have been forced to triage or turned into done
        assert obs.done is True or obs.turns_remaining == 0


# ------------------------------------------------------------------
# Triage action tests
# ------------------------------------------------------------------

class TestTriageAction:
    def test_correct_triage_scores_high(self, env):
        env.reset(seed=1)  # seed=1 → emergency case
        obs = env.step(MyAction(action_type="triage", message="emergency"))
        assert obs.reward >= 0.9, f"Correct triage should score >= 0.9, got {obs.reward}"
        assert obs.done is True

    def test_wrong_triage_scores_lower(self, env):
        env.reset(seed=1)  # emergency case
        obs = env.step(MyAction(action_type="triage", message="home care"))
        assert obs.reward <= 0.1, f"Dangerous mistake should score <= 0.1, got {obs.reward}"
        assert obs.done is True

    def test_adjacent_triage_scores_partial(self, env):
        env.reset(seed=1)  # emergency case
        obs = env.step(MyAction(action_type="triage", message="urgent care"))
        assert 0.3 <= obs.reward <= 0.7, f"Adjacent triage should be ~0.5, got {obs.reward}"

    def test_triage_ends_episode(self, env):
        env.reset(seed=1)
        obs = env.step(MyAction(action_type="triage", message="emergency"))
        assert obs.done is True

    def test_reward_always_in_valid_range(self, env):
        for seed in range(1, 11):
            env.reset(seed=seed)
            for triage in VALID_TRIAGE:
                env.reset(seed=seed)
                obs = env.step(MyAction(action_type="triage", message=triage))
                assert 0.0 <= obs.reward <= 1.0, (
                    f"Reward {obs.reward} out of range for seed={seed}, triage={triage}"
                )

    def test_efficiency_bonus_for_fewer_questions(self, env):
        """Triaging immediately should score higher than asking many questions first."""
        env.reset(seed=1)
        obs_immediate = env.step(MyAction(action_type="triage", message="emergency"))
        immediate_reward = obs_immediate.reward

        env.reset(seed=1)
        # Ask questions first
        for _ in range(3):
            env.step(MyAction(action_type="ask", message="tell me more"))
        obs_after_questions = env.step(MyAction(action_type="triage", message="emergency"))
        delayed_reward = obs_after_questions.reward

        assert immediate_reward >= delayed_reward, (
            "Immediate correct triage should reward >= delayed correct triage"
        )

    def test_invalid_triage_scores_zero(self, env):
        env.reset(seed=1)
        obs = env.step(MyAction(action_type="triage", message="i dont know"))
        assert obs.reward == 0.0


# ------------------------------------------------------------------
# Grader tests
# ------------------------------------------------------------------

class TestGrader:
    def test_grader_returns_correct_fields(self, env):
        env.reset(seed=1)
        result = env.grade("emergency", turns_used=1)
        assert "reward" in result
        assert "is_correct" in result
        assert "agent_choice" in result
        assert "correct_choice" in result
        assert "explanation" in result

    def test_grader_reward_in_range(self, env):
        for seed in range(1, 6):
            env.reset(seed=seed)
            for choice in VALID_TRIAGE:
                env.reset(seed=seed)
                result = env.grade(choice, turns_used=1)
                assert 0.0 <= result["reward"] <= 1.0, (
                    f"Grader reward {result['reward']} out of range"
                )

    def test_grader_correct_answer_is_correct(self, env):
        env.reset(seed=1)
        correct = env.current_task["correct_triage"]
        result = env.grade(correct, turns_used=1)
        assert result["is_correct"] is True
        assert result["reward"] >= 0.9


# ------------------------------------------------------------------
# Task catalog tests
# ------------------------------------------------------------------

class TestTaskCatalog:
    def test_catalog_has_three_tasks(self, env):
        catalog = env.get_task_catalog()
        assert len(catalog["tasks"]) == 3

    def test_catalog_has_difficulty_progression(self, env):
        catalog = env.get_task_catalog()
        difficulties = [t["difficulty"] for t in catalog["tasks"]]
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_catalog_tasks_have_required_fields(self, env):
        catalog = env.get_task_catalog()
        for task in catalog["tasks"]:
            assert "task_id" in task
            assert "description" in task
            assert "difficulty" in task
            assert "triage_categories" in task


# ------------------------------------------------------------------
# Multi-turn episode integration test
# ------------------------------------------------------------------

class TestFullEpisode:
    def test_full_easy_episode(self, env):
        """Easy task: ask one question then triage correctly."""
        obs = env.reset(seed=1)
        assert obs.done is False

        obs = env.step(MyAction(action_type="ask", message="What are the vitals?"))
        assert obs.done is False
        assert obs.turn == 1

        obs = env.step(MyAction(action_type="triage", message="emergency"))
        assert obs.done is True
        assert obs.reward >= 0.7

    def test_full_episode_step_count(self, env):
        env.reset(seed=1)
        env.step(MyAction(action_type="ask", message="pain level?"))
        env.step(MyAction(action_type="ask", message="duration?"))
        obs = env.step(MyAction(action_type="triage", message="emergency"))
        assert obs.turn == 3
        assert obs.done is True

    def test_episode_resets_cleanly(self, env):
        env.reset(seed=1)
        env.step(MyAction(action_type="triage", message="emergency"))

        # Reset and run again — should be clean
        obs = env.reset(seed=2)
        assert obs.turn == 0
        assert obs.done is False
        assert obs.patient_context == {}