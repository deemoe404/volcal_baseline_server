FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

## ---- System packages -----------------------------------------------------
RUN apt-get update && \
    # —— Build dependencies ——
    apt-get install -y --no-install-recommends \
        build-essential cmake git \
    # —— Python 3.10 & dev packages ——
        python3.10 python3.10-dev python3.10-distutils python3-pip \
    # —— root CA certificates ——
        ca-certificates \
    # —— Open3D dependencies ——
        libx11-6 libxext6 libsm6 libglib2.0-0 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# FastAPI application dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Pipeline dependencies
COPY volcal_baseline/requirements.txt /tmp/volcal_requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/volcal_requirements.txt

COPY . .
RUN python3 -m pip install -v --editable volcal_baseline/py4dgeo

# Start the FastAPI server
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
