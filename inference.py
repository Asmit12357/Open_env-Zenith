"""
Medical Triage RL — Inference Script
Runs a multi-turn clinical interview for each task.
The agent asks clarifying questions then submits a triage decision.

Log format (required by Scaler validator):
  [START] task=<id> env=medical_triage model=<model>
  [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<msg|null>
  [END]   task=<id> score=<float> steps=<n>
"""

import os
import sys
import json
import requests
from pathlib import Path


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

def load_and_configure():
    """Prioritize injected env vars (from Scaler validator) over .env file."""

    # Read injected vars FIRST — never let .env overwrite these
    injected_api_key = os.getenv("API_KEY")
    injected_base_url = os.getenv("API_BASE_URL")
    injected_model = os.getenv("MODEL_NAME")
    injected_env_url = os.getenv("ENV_URL")

    # .env as fallback only
    try:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if not os.getenv(key):  # never overwrite injected vars
                        os.environ[key] = value.strip().strip('"').strip("'")
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to load .env: {e}\n")

    env_url = injected_env_url or os.getenv("ENV_URL") or "http://127.0.0.1:7860"
    if all(x not in env_url for x in ["huggingface.co", "localhost", "127.0.0.1", "0.0.0.0"]):
        env_url = "https://asmit99-medical-triage-rl.hf.space"

    return {
        "API_KEY":      injected_api_key or os.getenv("HF_TOKEN") or "EMPTY_KEY",
        "API_BASE_URL": injected_base_url or "https://router.huggingface.co/v1",
        "MODEL_NAME":   injected_model or os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct",
        "ENV_URL":      env_url,
        "MAX_STEPS":    int(os.getenv("MAX_AGENT_STEPS", "5")),
    }


def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


# ------------------------------------------------------------------
# LLM helpers
# ------------------------------------------------------------------

def build_ask_prompt(symptoms: str, patient_context: dict, turn: int, turns_remaining: int) -> str:
    context_str = ""
    if patient_context:
        context_str = "\n\nClinical information gathered so far:\n"
        for k, v in patient_context.items():
            context_str += f"  - {k.replace('_', ' ').title()}: {v}\n"

    return (
        f"You are a medical triage assistant conducting a clinical interview.\n\n"
        f"Patient presentation: \"{symptoms}\"\n"
        f"{context_str}\n"
        f"This is turn {turn}. You have {turns_remaining} turns remaining.\n\n"
        f"You may ask ONE clarifying question to gather more clinical information, "
        f"or submit your final triage decision.\n\n"
        f"To ask a question, respond with:\n"
        f"  ACTION: ask\n"
        f"  MESSAGE: <your question about pain level, duration, vitals, history, or additional symptoms>\n\n"
        f"To submit a triage decision, respond with:\n"
        f"  ACTION: triage\n"
        f"  MESSAGE: <exactly one of: home care | clinic visit | urgent care | emergency>\n\n"
        f"If you have enough information or this is your last turn, submit your triage decision."
    )


def build_triage_prompt(symptoms: str, patient_context: dict) -> str:
    context_str = ""
    if patient_context:
        context_str = "\n\nClinical information gathered:\n"
        for k, v in patient_context.items():
            context_str += f"  - {k.replace('_', ' ').title()}: {v}\n"

    return (
        f"You are a medical triage assistant. Based on all available information, "
        f"submit your final triage decision.\n\n"
        f"Patient presentation: \"{symptoms}\"\n"
        f"{context_str}\n\n"
        f"Respond with ONLY one of these exact phrases:\n"
        f"home care\nclinic visit\nurgent care\nemergency"
    )


def parse_llm_response(content: str) -> tuple[str, str]:
    """
    Parse LLM response into (action_type, message).
    Handles both structured ACTION/MESSAGE format and plain triage responses.
    """
    content = content.strip()
    lines = content.split("\n")

    action_type = "triage"
    message = "emergency"

    for line in lines:
        line = line.strip()
        if line.lower().startswith("action:"):
            action_type = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("message:"):
            message = line.split(":", 1)[1].strip().lower().rstrip(".")

    # If no structured format found, treat whole response as triage
    if action_type == "triage" and message == "emergency":
        raw = content.lower().rstrip(".")
        valid = {"home care", "clinic visit", "urgent care", "emergency"}
        if raw in valid:
            message = raw
        elif "ask" in content.lower() and "?" in content:
            action_type = "ask"
            message = content

    return action_type, message


# ------------------------------------------------------------------
# Main episode runner
# ------------------------------------------------------------------

def run_episode(client, config: dict, task_id: str, seed: int) -> float:
    """Run one multi-turn episode. Returns final score."""
    env_url = config["ENV_URL"]
    model = config["MODEL_NAME"]

    print(f"[START] task={task_id} env=medical_triage model={model}", flush=True)

    # --- RESET ---
    symptoms = "Patient presents with unknown symptoms."
    try:
        r = requests.post(f"{env_url}/reset", json={"seed": seed}, timeout=20)
        if r.status_code == 200:
            data = r.json()
            obs = data.get("observation") if isinstance(data, dict) else data
            if isinstance(obs, dict):
                symptoms = obs.get("echoed_message") or symptoms
    except Exception as e:
        sys.stderr.write(f"[WARN] Reset error: {e}\n")

    # --- MULTI-TURN EPISODE ---
    step_num = 0
    final_score = 0.01
    patient_context = {}
    turns_remaining = config["MAX_STEPS"]
    done = False

    while not done and step_num < config["MAX_STEPS"]:
        step_num += 1
        turns_remaining = config["MAX_STEPS"] - step_num

        # Build prompt based on remaining turns
        if turns_remaining <= 0:
            prompt = build_triage_prompt(symptoms, patient_context)
        else:
            prompt = build_ask_prompt(symptoms, patient_context, step_num, turns_remaining)

        # LLM call
        action_type = "triage"
        message = "emergency"
        error_msg = None

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=18,
            )
            if resp and resp.choices:
                content = resp.choices[0].message.content or ""
                action_type, message = parse_llm_response(content)
        except Exception as e:
            error_msg = str(e)
            sys.stderr.write(f"[WARN] LLM error on step {step_num}: {e}\n")
            # Force triage on LLM failure to not waste turns
            action_type = "triage"

        # Send action to environment
        raw_reward = 0.0
        try:
            s_res = requests.post(
                f"{env_url}/step",
                json={"action": {"action_type": action_type, "message": message}},
                timeout=20,
            )
            if s_res.status_code == 200:
                s_data = s_res.json()
                raw_reward = s_data.get("reward", 0.0)
                done = s_data.get("done", False)

                # Update patient context from observation
                obs = s_data.get("observation", {})
                if isinstance(obs, dict) and obs.get("patient_context"):
                    patient_context.update(obs["patient_context"])
                if isinstance(obs, dict) and obs.get("turns_remaining") is not None:
                    turns_remaining = obs["turns_remaining"]
            else:
                sys.stderr.write(f"[WARN] Step failed ({s_res.status_code})\n")
        except Exception as e:
            sys.stderr.write(f"[WARN] Step error: {e}\n")

        step_score = max(0.01, min(0.99, safe_float(raw_reward)))

        # Update final score only on triage (terminal) step
        if done or action_type == "triage":
            final_score = step_score

        # Log every step in required format
        action_json = json.dumps({"action_type": action_type, "message": message})
        print(
            f"[STEP] step={step_num} action={action_json} "
            f"reward={step_score:.4f} done={str(done).lower()} "
            f"error={error_msg or 'null'}",
            flush=True,
        )

        if done:
            break

    print(f"[END] task={task_id} score={final_score:.4f} steps={step_num}", flush=True)
    return final_score


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def run_inference():
    config = load_and_configure()
    print(f"Connecting to environment at: {config['ENV_URL']}", flush=True)

    try:
        from openai import OpenAI
    except ImportError:
        sys.stderr.write("Fatal Error: openai library not found.\n")
        return

    # Initialize with injected credentials — required by Scaler validator
    client = OpenAI(
        base_url=config["API_BASE_URL"],
        api_key=config["API_KEY"],
    )

    # Three tasks with escalating difficulty
    # seed maps to task index: seed=1→easy, seed=3→medium, seed=5→hard
    tasks = [
        {"id": "task_1", "task_name": "task_easy",   "seed": 1},
        {"id": "task_2", "task_name": "task_medium",  "seed": 3},
        {"id": "task_3", "task_name": "task_hard",    "seed": 5},
    ]

    scores = []
    for task in tasks:
        score = run_episode(client, config, task["id"], task["seed"])
        scores.append(score)

    avg = sum(scores) / len(scores)
    sys.stderr.write(f"[INFO] Average score across tasks: {avg:.4f}\n")


if __name__ == "__main__":
    try:
        run_inference()
    except BaseException as e:
        sys.stderr.write(f"CRITICAL ERROR: {e}\n")
        sys.exit(0)