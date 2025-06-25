FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install numpy==1.24.4
RUN pip install torch==1.13.0 torchvision==0.14.0
RUN pip install mmcv==2.0.1
RUN pip install mmengine==0.7.4
RUN pip install mmdet==3.0.0
RUN pip install gradio ultralytics

WORKDIR /workspace
