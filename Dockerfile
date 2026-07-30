FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/hf-cache MODEL_ID=Muno459/fastconformer-quran MODEL_REVISION=main
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY handler.py .
CMD ["python3", "-u", "handler.py"]
