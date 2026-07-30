FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/.hf \
    MODEL_DIR=/models/quran-asr

WORKDIR /app

COPY requirements.txt .
# Use the official CPU-only wheels; PyPI's Linux torch wheel pulls CUDA/NVIDIA
# runtime packages even though this RunPod endpoint has no GPU.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.7.1 torchaudio==2.7.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
