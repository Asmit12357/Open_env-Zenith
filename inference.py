import os
import sys
import time
import requests
from pathlib import Path
from openai import OpenAI

# --- 1. Load .env for Local Development (Friend's Suggestion) ---
def load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_dotenv()

# --- 2. Configuration (Prioritizing Scaler Variables) ---
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
# Use the injected API_KEY first, fallback to HF_TOKEN for local
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "EMPTY_KEY"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

# DIRECT PROG-URL
ENV_URL = "https://asmit99-medical-triage-rl.hf.space"

def run_inference():
    # MUST match the IDs in your openenv.yaml exactly
    task_ids = ["task_1", "task_2", "task_3"]

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        for t_id in task_ids:
            # Each task needs its own [START] block
            print(f"[START] task={t_id}", flush=True)
            
            # --- RESET ---
            try:
                r = requests.post(f"{ENV_URL}/reset", json={"seed": 42}, timeout=20)
                obs = r.json()
            except:
                obs = {"echoed_message": "emergency"} # Fallback

            # --- LLM CALL ---
            prompt = f"Symptoms: {obs.get('observation', {}).get('echoed_message', 'chest pain')}. Triage category?"
            answer = "emergency" # Default
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=15
                )
                answer = resp.choices[0].message.content.strip().lower()
            except:
                pass

            # --- STEP ---
            try:
                s_res = requests.post(f"{ENV_URL}/step", json={"action": {"message": answer}}, timeout=20)
                s_data = s_res.json()
                
                # Reward check for nested or flat JSON
                raw_reward = s_data.get("reward")
                if raw_reward is None:
                    raw_reward = s_data.get("observation", {}).get("reward", 0.0)
            except:
                raw_reward = 0.0

            # --- THE 0.01 - 0.99 RULE ---
            # Clamping prevents boundary crashes on 1.0 or 0.0
            safe_score = max(0.01, min(0.99, float(raw_reward)))

            print(f"[STEP] step=1 reward={safe_score}", flush=True)
            
            # Each task needs its own [END] block matching the ID
            print(f"[END] task={t_id} score={safe_score} steps=1", flush=True)

    except Exception as e:
        sys.stderr.write(f"Global Error: {str(e)}\n")

if __name__ == "__main__":
    run_inference()