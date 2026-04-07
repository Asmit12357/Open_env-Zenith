import sys
import os
import uvicorn
from openenv.core.env_server.http_server import create_app

# 1. Absolute Path Injection 
# This ensures Python can see 'my_env' folder sitting next to the 'server' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 2. Absolute Imports
# Since 'server' and 'my_env' are siblings at the root, 
# we import directly from the 'my_env' package.
from my_env.models import MyAction, MyObservation
from my_env.my_env_environment import MyEnvironment

# 3. Create the FastAPI app
app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="MedicalVerifier", 
    max_concurrent_envs=5,       
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Medical Triage RL Environment is Running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

def main():
    # Use 7860 as the hard default for Hugging Face compatibility
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()