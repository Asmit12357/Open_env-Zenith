FROM python:3.11-slim

# 1. Set the working directory to /app
WORKDIR /app

# 2. Copy requirements first to speed up builds
COPY requirements.txt .

# 3. Install dependencies 
# We add --upgrade to ensure you get the latest version of openenv
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 4. Copy your entire project (my_env, inference.py, etc.)
COPY . .

# 5. CRITICAL: Tell Python to treat /app as a root for imports
# This fixes the "ModuleNotFoundError: No module named 'openenv.core'"
ENV PYTHONPATH=/app

# 6. Create and switch to a non-root user (Hugging Face Security Requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 7. Expose the mandatory Hugging Face port
EXPOSE 7860

# 8. Start the FastAPI server
# This points to: zenith/my_env/server/app.py
CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]