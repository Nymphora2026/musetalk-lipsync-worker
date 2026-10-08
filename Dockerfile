# MuseTalk 1.5 isolated GPU worker.
# Official code: TMElyralab/MuseTalk @ 0a89dec45a0192b824e3cf4daf96c239440c5ed8
# Target: cheapest practical 16 GB NVIDIA (T4 / RTX 2000 Ada). Not A100/H100.
# Weights are NOT baked. They live on /models (network volume / persistent cache).

ARG MUSETALK_REVISION=0a89dec45a0192b824e3cf4daf96c239440c5ed8

FROM python:3.10-slim-bookworm AS schema
WORKDIR /opt/worker
COPY contract.py handler.py pins.json ./
RUN python contract.py && python handler.py --validate-only
CMD ["python", "handler.py", "--validate-only"]

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu
ARG MUSETALK_REVISION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUSETALK_ROOT=/opt/MuseTalk \
    MUSETALK_MODEL_DIR=/models \
    MUSETALK_RESULT_CACHE=/models/results \
    HF_HOME=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf \
    TORCH_HOME=/models/torch \
    FFMPEG_PATH=ffmpeg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip git ffmpeg \
      libgl1 libglib2.0-0 libsndfile1 ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
         torch==2.2.2 torchvision==0.17.2 \
         --index-url https://download.pytorch.org/whl/cu121

WORKDIR /opt/worker
COPY requirements-gpu.txt pins.json /opt/worker/
RUN python -m pip install --no-cache-dir -r /opt/worker/requirements-gpu.txt \
    && python -m pip install --no-cache-dir -U openmim \
    && mim install mmengine \
    && mim install "mmcv>=2.0.1,<2.2.0" \
    && mim install "mmdet>=3.1.0,<3.3.0" \
    && mim install "mmpose>=1.1.0,<1.4.0"

RUN git clone https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk \
    && git -C /opt/MuseTalk checkout ${MUSETALK_REVISION} \
    && ln -sfn /models /opt/MuseTalk/models \
    && python -m pip install --no-cache-dir -e /opt/MuseTalk --no-deps

COPY contract.py handler.py run_inference.py download_models.sh /opt/worker/
RUN chmod +x /opt/worker/download_models.sh \
    && mkdir -p /models /tmp/musetalk

WORKDIR /opt/worker
# First request may download ~4.3GB into /models. Later requests reuse the volume.
CMD ["python", "-u", "handler.py"]
