import os
import asyncio
import json
from openai import OpenAI

# 1. Import your actual environment and models
from server.my_env_environment import MyEnvironment
from my_env.models import MyAction

# 2. MANDATORY VARIABLES - Scaler Injected
# We remove the hardcoded fallbacks to ensure it ONLY uses the proxy
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
# FIX: Scaler uses "API_KEY", not "HF_TOKEN"
API_KEY = os.environ.get("API_KEY") 

# Initialize the OpenAI Client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# Initialize your real environment
env = MyEnvironment()

async def main():
    steps = 0
    rewards = []
    
    # [START] - Standardized for Grader
    print("START")

    try:
        # Step 1: Real Reset
        observation = env.reset(seed=42)
        # Note: Depending on your reset return type, you might need observation.echoed_message
        symptoms = getattr(observation, 'echoed_message', str(observation))
        
        # Step 2: Get AI Prediction
        prompt = f"Patient Symptoms: {symptoms}. Categorize this ONLY as: Emergency, Clinic, or Home Care. Respond with a single word."
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15,
            temperature=0.1 
        )
        
        ai_prediction = response.choices[0].message.content.strip().lower()

        # Step 3: Real Environment Step
        action_obj = MyAction(message=ai_prediction)
        result_obs = env.step(action_obj)
        
        # Update trackers
        steps += 1
        reward = result_obs.reward
        rewards.append(reward)
        done = result_obs.done

        # [STEP] - Standardized for Grader
        print(f"STEP {steps}")
        # Optional metadata logging
        # print(f"Action: {ai_prediction} | Reward: {reward}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
    
    finally:
        # [END] - Standardized for Grader
        print("END")

if __name__ == "__main__":
    asyncio.run(main())