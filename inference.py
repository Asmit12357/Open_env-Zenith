import os
import sys
import time
import requests
from openai import OpenAI

# 1. IMMEDIATE START LOG - Must be exactly this format
print("[START] task=medical_triage", flush=True)

def wait_for_env(url, timeout=60):
    """Wait for the HF Space to wake up and return a 200 OK."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(5)
    return False

def run_inference():
    # ENSURE THIS IS PUBLIC AND RUNNING
    ENV_URL = "https://asmit99-medical-triage-rl.hf.space"
    
    if not wait_for_env(ENV_URL):
        sys.stderr.write("Error: Environment not reachable after 60s timeout\n")
        sys.exit(1)

    try:
        # 2. CAPTURE ALL POSSIBLE INJECTED VARIABLES
        api_base = os.environ.get("API_BASE_URL")
        # Try API_KEY first, then fall back to HF_TOKEN
        api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
        model_name = os.environ.get("MODEL_NAME")

        if not api_key:
            sys.stderr.write("Error: No API_KEY or HF_TOKEN found in environment\n")
            sys.exit(1)

        client = OpenAI(base_url=api_base, api_key=api_key)
        
        # 3. RESET ENVIRONMENT
        reset_req = requests.post(f"{ENV_URL}/reset", json={"seed": 42}, timeout=15)
        reset_req.raise_for_status() # Crash early if reset fails
        obs = reset_req.json()
        
        total_reward = 0.0
        step_count = 0
        done = False

        # 4. MAIN INTERACTION LOOP
        while not done and step_count < 10:
            step_count += 1
            
            # Strict Prompting to ensure agent matches your reward categories
            system_prompt = "You are a medical triage assistant. Reply with ONLY one of these four categories: 'home care', 'clinic visit', 'urgent care', or 'emergency'. Do not explain.Just the word."
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Symptoms: {obs.get('echoed_message', '')}"}
                ],
                temperature=0.0 # Force consistency
            )
            
            answer = response.choices[0].message.content.strip().lower()

            # 5. EXECUTE STEP
            step_req = requests.post(f"{ENV_URL}/step", json={"action": {"message": answer}}, timeout=15)
            step_req.raise_for_status()
            obs = step_req.json()
            
            current_reward = float(obs.get("reward", 0.0))
            total_reward += current_reward
            done = obs.get("done", False)

            # 6. STRUCTURED STEP LOG
            print(f"[STEP] step={step_count} reward={current_reward}", flush=True)

        # 7. FINAL END LOG - score must be the sum of rewards
        print(f"[END] task=medical_triage score={total_reward} steps={step_count}", flush=True)

    except Exception as e:
        sys.stderr.write(f"CRITICAL EXCEPTION: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    run_inference()