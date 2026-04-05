import sys
import os
import argparse
import uvicorn
from openenv.core.env_server.http_server import create_app

# 1. Path Injection
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 2. Imports
try:
    from models import MyAction, MyObservation
    from my_env_environment import MyEnvironment
except ImportError:
    from ..models import MyAction, MyObservation
    from .my_env_environment import MyEnvironment

# 3. Create the FastAPI app
app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="MedicalVerifier", 
    max_concurrent_envs=5,       
)

# --- ROUTES (Must be defined BEFORE the server starts) ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "Medical Triage RL Environment is Running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# -------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860) 
    args = parser.parse_args()
    main(port=args.port)