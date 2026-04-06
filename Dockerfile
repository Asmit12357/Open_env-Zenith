FROM python:3.11-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Create the user FIRST so we can give them ownership
RUN useradd -m -u 1000 user

# 3. Install Python libraries
# Added 'pydantic' and 'typing-extensions' just in case
RUN pip install --no-cache-dir openenv-core openenv fastapi uvicorn pydantic

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy code and FIX PERMISSIONS
# The --chown=user:user is the secret sauce for Hugging Face
COPY --chown=user:user . .

# 5. Environment Variables
# Fixed to match your "my_env" folder name
ENV PYTHONPATH="/app:/app/my_env:/app/my_env/server:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true
# Tell Hugging Face we are using the standard port
ENV PORT=7860

# 6. Health check (Updated to port 7860)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER user

# Hugging Face usually expects 7860
EXPOSE 7860

# 7. Start the server (Using the correct folder 'my_env')
CMD ["sh", "-c", "uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000"]