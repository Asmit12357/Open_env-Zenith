import argparse
import uvicorn
from openenv.core.env_server.http_server import create_app

# Handling imports for both local and package-level execution
try:
    from models import MyAction, MyObservation
    from server.my_env_environment import MyEnvironment
except ModuleNotFoundError:
    from models import MyAction, MyObservation
    from server.my_env_environment import MyEnvironment

# This creates the FastAPI server using your Medical Logic (MyEnvironment)
app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="MedicalVerifier", # Renamed for your project
    max_concurrent_envs=5,       # Increased to allow you and your friend to test at once
)

def main(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
