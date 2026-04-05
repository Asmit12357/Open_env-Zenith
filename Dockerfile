FROM python:3.11-slim

WORKDIR /app

# Install the core library explicitly
RUN pip install --no-cache-dir openenv-core openenv fastapi uvicorn

# Copy your requirements and install others
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (including your openenv.yaml now in root)
COPY . .

# Critical for resolving 'openenv.core' and your local 'my_env'
ENV PYTHONPATH=/app

RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

# We start the server directly. 
# HF will use this to communicate with your OpenEnv logic.
CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]