# --- STAGE 1: BUILDER ---
FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy your local code into the builder
COPY . /app/zenith
WORKDIR /app/zenith

# Install uv and sync dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable


# --- STAGE 2: FINAL RUNTIME ---
FROM ghcr.io/meta-pytorch/openenv-base:latest

# Hugging Face requirement: Run as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR /app

# Copy the environment and code from the builder stage
COPY --from=builder /app/zenith/.venv /app/.venv
COPY --from=builder /app/zenith /app/zenith

# Set critical paths
ENV PATH="/app/.venv/bin:$HOME/.local/bin:$PATH"
ENV PYTHONPATH="/app/zenith:/app/zenith/my_env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server on HF port 7860
CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]