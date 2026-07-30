FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/.hf \
    MODEL_DIR=/models/quran-asr

WORKDIR /app

COPY requirements.txt .
# Use the official CPU-only wheel; PyPI's Linux torch wheel pulls CUDA/NVIDIA
# runtime packages even though this RunPod endpoint has no GPU. Torchaudio is
# intentionally absent: preprocessing is implemented and regression-checked
# in app.py, avoiding its OpenMP/system-library runtime dependency.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.7.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py diagnose.py ./
COPY tests ./tests

# Asset-independent regressions run during image build. Gated model assets are
# validated by the offline diagnostic before the image is released.
RUN pytest -q tests

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
