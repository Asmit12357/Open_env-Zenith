import sys
import os

# Add the parent directory to sys.path so 'my_env' can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try the absolute import first
    from my_env.my_env_environment import MyEnvironment
except ImportError:
    # Fallback to a relative import if the first fails
    try:
        from ..my_env.my_env_environment import MyEnvironment
    except (ImportError, ValueError):
        # Last ditch effort for specific container structures
        import my_env.my_env_environment as env_mod
        MyEnvironment = env_mod.MyEnvironment
