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

# This creates the FastAPI server using your Medical Logic
app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="MedicalVerifier", 
    max_concurrent_envs=5,       
)

# --- ADDED FOR HUGGING FACE HEALTH CHECK ---
@app.get("/health")
def health_check():
    return {"status": "ok"}
# ------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860): # Defaulted to 7860
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated default to 7860 for Hugging Face compatibility
    parser.add_argument("--port", type=int, default=7860) 
    args = parser.parse_args()
    main(port=args.port)
