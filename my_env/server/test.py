import sys
import os

# Ensure the script can see the environment file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from my_env_environment import MyEnvironment, MyAction

env = MyEnvironment()
obs = env.reset()
print(f"Patient Symptoms: {obs.echoed_message}")

# Testing 'Emergency' choice
action = MyAction(message="Emergency")
res = env.step(action)
print(f"Reward received: {res.reward}")