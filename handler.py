"""Stateless Runpod Serverless handler for FastConformer-Quran ONNX."""
from __future__ import annotations
import base64, io, os
from pathlib import Path
from typing import Any
import librosa, numpy as np, onnxruntime as ort, runpod, sentencepiece as spm, soundfile as sf
from huggingface_hub import hf_hub_download

MODEL_ID = os.environ.get("MODEL_ID", "Muno459/fastconformer-quran")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/model"))
RATE = 16_000

def download(name: str) -> Path:
    token = os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN is required; attach the Runpod Secret.")
    return Path(hf_hub_download(repo_id=MODEL_ID, filename=name, revision=os.environ.get("MODEL_REVISION", "main"), token=token, local_dir=MODEL_DIR))

opts = ort.SessionOptions(); opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SESSION = ort.InferenceSession(str(download("onnx/model.fp16.onnx")), sess_options=opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
TOKENIZER = spm.SentencePieceProcessor(model_file=str(download("tokenizer.model")))

def features(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=RATE, n_fft=512, hop_length=160, win_length=400, n_mels=80, fmin=0, fmax=8000, power=2.0)
    mel = np.log(np.maximum(mel, 1e-10)).astype(np.float32)
    return ((mel - mel.mean(1, keepdims=True)) / (mel.std(1, keepdims=True) + 1e-5))[None]

def decode(logprobs: np.ndarray) -> str:
    blank, prev, out = TOKENIZER.get_piece_size(), None, []
    for token in np.argmax(logprobs[0], axis=-1).tolist():
        if token != prev and token != blank: out.append(token)
        prev = token
    return TOKENIZER.decode(out).strip()

def handler(job: dict[str, Any]) -> dict[str, Any]:
    encoded = job.get("input", {}).get("audio_base64")
    if not isinstance(encoded, str): return {"error": "input.audio_base64 is required (base64 WAV audio)."}
    try:
        audio, rate = sf.read(io.BytesIO(base64.b64decode(encoded, validate=True)), dtype="float32", always_2d=False)
        if audio.ndim == 2: audio = audio.mean(1)
        if rate != RATE: audio = librosa.resample(audio, orig_sr=rate, target_sr=RATE)
        x = features(np.asarray(audio, dtype=np.float32))
        logprobs = SESSION.run(["logprobs"], {"audio_signal": x, "length": np.asarray([x.shape[-1]], dtype=np.int64)})[0]
        return {"text": decode(logprobs), "audio_seconds": round(len(audio)/RATE, 3), "model": MODEL_ID}
    except Exception as exc: return {"error": str(exc)}

runpod.serverless.start({"handler": handler})
