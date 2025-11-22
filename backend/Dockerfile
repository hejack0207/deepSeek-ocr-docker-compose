# Use runtime image instead of devel to reduce size
# FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Clean cache in the same layer
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    # Configure Python
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    # Update pip in the same layer
    && pip install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple pip setuptools wheel \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Directorio de trabajo
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Install all Python dependencies in a single optimized layer
RUN --mount=type=cache,target=/root/.cache/pip \
    # PyTorch primero (versión específica para CUDA 11.8)
    pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118 \
    # Install other dependencies in the same layer
    && pip install --no-cache-dir -r requirements.txt \
    # Clean up temporary pip files
    && rm -rf /tmp/* /root/.cache/pip/*

RUN apt-get update && apt-get install -y --no-install-recommends python3-dev

RUN --mount=type=cache,target=/root/.cache/pip \
    # Try flash-attention (may fail, not critical)
    pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation \
    && rm -rf /tmp/* /root/.cache/pip/*

# Copy application code (use .dockerignore to exclude unnecessary files)
COPY . .

# Create directories and clean up unnecessary files in a single layer
RUN mkdir -p /app/uploads /app/outputs /root/.cache/huggingface \
    # Remove unnecessary files from source code
    && find /app -type f -name "*.pyc" -delete \
    && find /app -type d -name "__pycache__" -delete \
    && rm -rf /app/.git /app/.pytest_cache /app/.mypy_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Startup command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
