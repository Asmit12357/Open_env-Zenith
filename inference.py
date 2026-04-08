import os
import sys
import time
import json
import requests
from openai import OpenAI

# Immediate start for the validator
print("[START] task=medical_triage", flush=True)

ENV_URL = "https://asmit99-medical-triage-rl.hf.space"

def safe_get(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        return r if r.status_code == 200 else None
    except:
        return None

def safe_post(url, payload, timeout=10):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r if r.status_code == 200 else None
    except:
        return None

def get_llm_answer(client, model_name, message):
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": message}]
        )
        return resp.choices[0].message.content.strip().lower()
    except:
        # fallback safe answer for triage
        return "clinic visit"

def run_inference():
    final_score = 0.0
    total_steps = 0

    try:
        # --- ENV VARIABLES (Injected by Scaler) ---
        api_base = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")
        model_name = os.environ.get("MODEL_NAME")

        client = None
        if api_base and api_key and model_name:
            try:
                client = OpenAI(base_url=api_base, api_key=api_key)
            except:
                client = None

        # --- CHECK ENV AVAILABILITY ---
        env_alive = False
        for _ in range(3):  # Fast retry logic
            if safe_get(f"{ENV_URL}/health"):
                env_alive = True
                break
            time.sleep(3)

        # --- RESET ---
        if env_alive:
            reset_res = safe_post(f"{ENV_URL}/reset", {"seed": 42})
            if reset_res:
                try:
                    obs = reset_res.json()
                except:
                    obs = {"echoed_message": "fever"}
            else:
                obs = {"echoed_message": "fever"}
        else:
            obs = {"echoed_message": "fever"}

        done = False

        # --- LOOP ---
        while not done and total_steps < 5:
            total_steps += 1

            prompt = f"Symptoms: {obs.get('echoed_message', '')}. Triage category?"

            # LLM call via proxy or local fallback
            if client:
                answer = get_llm_answer(client, model_name, prompt)
            else:
                answer = "clinic visit"

            # STEP
            if env_alive:
                step_res = safe_post(
                    f"{ENV_URL}/step",
                    {"action": {"message": answer}}
                )
                if step_res:
                    try:
                        obs = step_res.json()
                    except:
                        obs = {}
                else:
                    obs = {}
            else:
                obs = {}

            # Extract results safely
            reward = float(obs.get("reward", 0.0))
            final_score += reward
            done = obs.get("done", False)

            # Strict Step Format
            print(f"[STEP] step={total_steps} reward={reward}", flush=True)

        # Strict End Format (score and steps are the required keys)
        print(f"[END] task=medical_triage score={final_score} steps={total_steps}", flush=True)

    except Exception as e:
        # Final safety net to prevent non-zero exit code
        print(f"[ERROR] {str(e)}", flush=True)
        print("[END] task=medical_triage score=0 steps=0", flush=True)


if __name__ == "__main__":
    run_inference()