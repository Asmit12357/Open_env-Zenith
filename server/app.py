import sys
import os

# 1. MOVED TO THE VERY TOP: Absolute Path Injection
# We must do this before any 'from my_env' imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import uvicorn
from openenv.core.env_server.http_server import create_app

# 2. Absolute Imports (Now Python knows where to look)
try:
    from my_env.models import MyAction, MyObservation
    from my_env.my_env_environment import MyEnvironment
except ImportError:
    # Local fallback if the path injection needs a nudge
    from models import MyAction, MyObservation
    from my_env_environment import MyEnvironment

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
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()