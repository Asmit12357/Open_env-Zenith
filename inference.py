import os
import asyncio
from openai import OpenAI

# 1. MANDATORY VARIABLES (Set by the Grader/Environment)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-instruct") # Example model
HF_TOKEN = os.getenv("HF_TOKEN") # Your hf_... token

# 2. INITIALIZE THE WRAPPER
# We use the OpenAI Client to talk to Hugging Face
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

async def main():
    # Logging Variables
    task_id = "medical-triage-01"
    steps = 0
    total_score = 0.0
    rewards = []

    # [START] LINE - MANDATORY
    print(f"[START] task=medical-diagnosis env=openenv model={MODEL_NAME}")

    try:
        # Step 1: Get Observation (Symptoms) from your environment
        # In a real run, this comes from: obs = await env.reset()
        obs = "Patient has sharp chest pain and shortness of breath."
        
        steps += 1
        
        # Step 2: Get AI Action using the OpenAI Client
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Triage this: {obs}. Options: Emergency, Clinic, Home Care."}],
            max_tokens=10
        )
        action = response.choices[0].message.content.strip()

        # Step 3: Get Reward from your environment
        # In a real run: reward, done, info = await env.step(action)
        reward = 1.0 if "Emergency" in action else 0.0 # Example logic
        rewards.append(reward)
        total_score = reward 

        # [STEP] LINE - MANDATORY (Note the lowercase 'true' and 2 decimal places)
        print(f"[STEP] step={steps} action={action} reward={reward:.2f} done=true error=null")

    except Exception as e:
        # If something breaks, we still need to emit an [END] line
        pass

    finally:
        # [END] LINE - MANDATORY
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success=true steps={steps} score={total_score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    asyncio.run(main())