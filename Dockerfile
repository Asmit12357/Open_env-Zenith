FROM python:3.11-slim

# Install uv (the tool your logs show you're using)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory to root
WORKDIR /app

# Copy EVERYTHING from your local zenith folder to the container /app
COPY . .

# Install dependencies using uv
# This looks for pyproject.toml or requirements.txt in the root
RUN uv sync --no-editable

# Ensure the container runs as a non-root user (Mandatory for HF)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/app/.venv/bin:$PATH"

# Expose the correct HF port
EXPOSE 7860

# Start the server
# We use 'my_env.server.app:app' because your code is in that subfolder
CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]