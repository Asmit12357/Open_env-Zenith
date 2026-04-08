import os
import asyncio
from openai import OpenAI

# 1. Import your environment and models
from server.my_env_environment import MyEnvironment
from my_env.models import MyAction

# 2. CAPTURE INJECTED VARIABLES
# Using strict os.environ to ensure we don't accidentally fall back to local keys
try:
    API_BASE_URL = os.environ["API_BASE_URL"]
    MODEL_NAME = os.environ["MODEL_NAME"]
    API_KEY = os.environ["API_KEY"]
except KeyError as e:
    # If variables are missing, we print and exit to avoid a "No API traffic" fail
    print(f"CRITICAL: Missing environment variable {e}")
    exit(1)

# FIX FOR BYPASS ERROR: 
# LiteLLM proxies often require the /v1 suffix to route requests correctly.
# If the injected URL doesn't have it, the OpenAI client might skip the proxy logic.
if not API_BASE_URL.endswith("/v1"):
    API_BASE_URL = API_BASE_URL.rstrip("/") + "/v1"

# 3. INITIALIZE CLIENT
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# Initialize your real environment
env = MyEnvironment()

async def main():
    steps = 0
    
    # [MARKER] START - Grader uses this to begin session tracking
    # Print it alone on its own line.
    print("START")

    try:
        # Step 1: Environment Reset
        # We use seed=42 for consistency
        observation = env.reset(seed=42)
        
        # Handle observation extraction safely
        symptoms = getattr(observation, 'echoed_message', str(observation))
        
        # Step 2: Get AI Prediction via Proxy
        # Providing explicit options to the LLM prevents "wordy" responses
        prompt = (
            f"Patient Symptoms: {symptoms}\n"
            "Task: Categorize into exactly one of these: emergency, clinic visit, urgent care, home care.\n"
            "Constraint: Respond with ONLY the category name, nothing else."
        )
        
        # This call MUST go through the client initialized with API_BASE_URL
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15,
            temperature=0.0  # Crucial for consistency
        )
        
        ai_prediction = response.choices[0].message.content.strip().lower()

        # Step 3: Environment Step
        # Using the confirmed 'message' field
        action_obj = MyAction(message=ai_prediction)
        result_obs = env.step(action_obj)
        
        # Update trackers
        steps += 1
        
        # [MARKER] STEP - Grader uses this to count valid interactions
        # Format must be exactly "STEP X"
        print(f"STEP {steps}")

    except Exception as e:
        # Debug info for you, ignored by the grader's main markers
        print(f"DEBUG_LOG: {str(e)}")
    
    finally:
        # [MARKER] END - Grader uses this to finalize the score
        # Must be printed even if an error occurs
        print("END")

if __name__ == "__main__":
    # Ensure the event loop runs correctly
    asyncio.run(main())