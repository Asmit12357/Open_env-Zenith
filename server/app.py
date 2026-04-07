import sys
import os
import uvicorn
from openenv.core.env_server.http_server import create_app

# 1. Path Injection (Ensures Python finds the sibling my_env folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 2. Absolute Imports
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

# ... (rest of your health routes)

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()