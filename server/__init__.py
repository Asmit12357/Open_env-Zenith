import sys
import os

# This ensures the 'zenith' root is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 1. Import from the local folder (server)
from server.my_env_environment import MyEnvironment

# 2. Import from the sibling folder (my_env)
from my_env.models import MyAction, MyObservation
