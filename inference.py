import os
import sys
import httpx
from openai import OpenAI

# 1. Import your environment and models
# Ensure these paths are correct relative to your repo root
from server.my_env_environment import MyEnvironment
from my_env.models import MyAction

# 2. CAPTURE INJECTED VARIABLES
try:
    API_BASE_URL = os.environ["API_BASE_URL"]
    MODEL_NAME = os.environ["MODEL_NAME"]
    API_KEY = os.environ["API_KEY"]
except KeyError as e:
    # If variables are missing, we print to stderr and exit
    sys.stderr.write(f"CRITICAL: Missing environment variable {e}\n")
    sys.exit(1)

# FIX FOR BYPASS ERROR: 
# Ensure the /v1 suffix exists so the OpenAI client routes correctly
if not API_BASE_URL.endswith("/v1"):
    API_BASE_URL = API_BASE_URL.rstrip("/") + "/v1"

# 3. INITIALIZE CLIENT
# Using httpx for timeout protection
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
    http_client=httpx.Client(timeout=30.0)
)

# Initialize your real environment
env = MyEnvironment()

def main():
    # [START] - Grader needs 'task=' key
    print("[START] task=medical_triage", flush=True)

    total_reward = 0.0
    step_count = 0

    try:
        # Step 1: Environment Reset
        # Seed 42 is standard for evaluation
        observation = env.reset(seed=42)
        
        # Extract symptoms safely
        symptoms = getattr(observation, 'echoed_message', "No symptoms listed")
        
        # Step 2: Get AI Prediction via Proxy
        prompt = (
            f"Patient Symptoms: {symptoms}\n"
            "Task: Categorize into exactly one of these: emergency, clinic visit, urgent care, home care.\n"
            "Constraint: Respond with ONLY the category name, nothing else."
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15,
            temperature=0.0
        )
        
        ai_prediction = response.choices[0].message.content.strip().lower()

        # Step 3: Environment Step
        action_obj = MyAction(message=ai_prediction)
        result_obs = env.step(action_obj)
        
        # Update trackers
        step_count += 1
        reward = getattr(result_obs, 'reward', 0.0)
        total_reward += reward
        
        # [STEP] - Grader needs 'step=' and 'reward=' keys
        print(f"[STEP] step={step_count} reward={reward}", flush=True)

    except Exception as e:
        sys.stderr.write(f"LOG: {str(e)}\n")
    
    finally:
        # [END] - Grader needs 'score=' and 'steps=' keys
        # This must print even if the try block fails
        print(f"[END] task=medical_triage score={total_reward} steps={step_count}", flush=True)

if __name__ == "__main__":
    main()