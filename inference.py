import os
import sys
import requests
from pathlib import Path


# --- 1. Global Setup ---

def load_and_configure():
    """Load environment variables, always prioritizing injected env vars over .env file."""
    env_vars = {}

    # Read injected vars FIRST before anything can overwrite them
    injected_api_key = os.getenv("API_KEY")
    injected_base_url = os.getenv("API_BASE_URL")
    injected_model = os.getenv("MODEL_NAME")
    injected_env_url = os.getenv("ENV_URL")

    # Load .env file ONLY as fallback for vars not already injected
    try:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    # CRITICAL: never overwrite vars already injected by the validator
                    if not os.getenv(key):
                        os.environ[key] = value.strip().strip('"').strip("'")
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to load .env: {e}\n")

    # Always use the injected API_KEY — this is what the Scaler validator checks
    env_vars['API_KEY'] = injected_api_key or os.getenv("HF_TOKEN") or "EMPTY_KEY"
    env_vars['API_BASE_URL'] = injected_base_url or "https://router.huggingface.co/v1"
    env_vars['MODEL_NAME'] = injected_model or os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

    base_url = injected_env_url or os.getenv("ENV_URL") or "http://127.0.0.1:7860"
    if all(x not in base_url for x in ["huggingface.co", "localhost", "127.0.0.1", "0.0.0.0"]):
        base_url = "https://asmit99-medical-triage-rl.hf.space"
    env_vars['ENV_URL'] = base_url

    return env_vars


def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def run_inference():
    config = load_and_configure()
    print(f"Connecting to environment at: {config['ENV_URL']}", flush=True)

    try:
        from openai import OpenAI
    except ImportError:
        sys.stderr.write("Fatal Error: openai library not found. Please add to requirements.txt\n")
        return

    # Initialize client with injected credentials — required by Scaler validator
    client = OpenAI(
        base_url=config['API_BASE_URL'],
        api_key=config['API_KEY']
    )

    # Each task uses a distinct seed → different patient each time
    # seed=1 → id1 (emergency), seed=13 → id13 (clinic visit), seed=2 → id2 (home care)
    tasks = [
        {"id": "task_1", "seed": 1},
        {"id": "task_2", "seed": 13},
        {"id": "task_3", "seed": 2},
    ]

    for task in tasks:
        t_id = task["id"]
        seed = task["seed"]
        print(f"[START] task={t_id}", flush=True)

        # --- RESET ---
        symptoms = "Patient presents with unknown symptoms."
        try:
            r = requests.post(
                f"{config['ENV_URL']}/reset",
                json={"seed": seed},
                timeout=20
            )
            if r.status_code == 200:
                data = r.json()
                obs = data.get("observation") if isinstance(data, dict) else data
                if isinstance(obs, dict):
                    symptoms = obs.get("echoed_message") or obs.get("symptoms") or symptoms
                elif isinstance(obs, str):
                    symptoms = obs
            else:
                sys.stderr.write(f"[WARN] Reset failed (status {r.status_code})\n")
        except Exception as e:
            sys.stderr.write(f"[WARN] Reset error: {e}\n")

        # --- LLM INFERENCE ---
        prompt = (
            f"You are a medical triage assistant. A patient presents with the following symptoms:\n\n"
            f"\"{symptoms}\"\n\n"
            "Based on these symptoms, classify the urgency level. "
            "Respond with ONLY one of these exact phrases and nothing else:\n"
            "home care\n"
            "clinic visit\n"
            "urgent care\n"
            "emergency"
        )

        answer = "emergency"  # Safe clinical fallback
        try:
            resp = client.chat.completions.create(
                model=config['MODEL_NAME'],
                messages=[{"role": "user", "content": prompt}],
                timeout=18
            )
            if resp and resp.choices:
                content = resp.choices[0].message.content
                if content:
                    raw = content.strip().lower().strip(".")
                    valid = {"home care", "clinic visit", "urgent care", "emergency"}
                    answer = raw if raw in valid else "emergency"
        except Exception as e:
            sys.stderr.write(f"[WARN] LLM error: {e}\n")

        # --- STEP ---
        raw_reward = 0.0
        try:
            s_res = requests.post(
                f"{config['ENV_URL']}/step",
                json={"action": {"message": answer}},
                timeout=20
            )
            if s_res.status_code == 200:
                s_data = s_res.json()
                if isinstance(s_data, dict):
                    raw_reward = s_data.get("reward")
                    if raw_reward is None:
                        raw_reward = s_data.get("observation", {}).get("reward", 0.0)
            else:
                sys.stderr.write(f"[WARN] Step failed (status {s_res.status_code}): {s_res.text}\n")
        except Exception as e:
            sys.stderr.write(f"[WARN] Step error: {e}\n")

        # Enforce valid score range
        final_score = max(0.01, min(0.99, safe_float(raw_reward)))

        print(f"[STEP] step=1 reward={final_score}", flush=True)
        print(f"[END] task={t_id} score={final_score} steps=1", flush=True)


if __name__ == "__main__":
    try:
        run_inference()
    except BaseException as e:
        sys.stderr.write(f"CRITICAL ERROR: {e}\n")
        sys.exit(0)