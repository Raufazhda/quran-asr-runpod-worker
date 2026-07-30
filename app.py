"""Cache-aware Quran ASR WebSocket service for RunPod Load Balancer."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from huggingface_hub import snapshot_download

MODEL_ID = "Muno459/fastconformer-quran-streaming"
MODEL_FILE = "model_streaming_with_encoder.q8.onnx"
ASSET_FILES = [MODEL_FILE, "streaming_global_cmvn.npz", "tokenizer.model", "config.json"]
SAMPLE_RATE = 16_000
N_LAYERS, D_MODEL, LEFT_CACHE, TIME_CACHE = 17, 512, 70, 8
CHUNK_MEL, BLANK_ID = 112, 1024


class ModelRuntime:
    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.tokenizer: spm.SentencePieceProcessor | None = None
        self.cmvn: np.lib.npyio.NpzFile | None = None

    def load(self) -> None:
        model_dir = Path(os.getenv("MODEL_DIR", "/models/quran-asr"))
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=MODEL_ID, local_dir=model_dir, allow_patterns=ASSET_FILES,
                          token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
        self.session = ort.InferenceSession(str(model_dir / MODEL_FILE), providers=["CPUExecutionProvider"])
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
        self.cmvn = np.load(model_dir / "streaming_global_cmvn.npz")


runtime = ModelRuntime()


def log_mel(wav: np.ndarray) -> np.ndarray:
    wav_tensor = torch.tensor(np.asarray(wav, dtype=np.float32))
    if wav_tensor.numel() == 0:
        return np.empty((80, 0), dtype=np.float32)
    wav_tensor = torch.cat([wav_tensor[:1], wav_tensor[1:] - 0.97 * wav_tensor[:-1]])
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512,
        win_length=400, hop_length=160, n_mels=80, power=2.0,
        window_fn=torch.hann_window, norm="slaney", mel_scale="slaney")(wav_tensor)
    return torch.log(mel + 2**-24).numpy()


@dataclass
class StreamingSession:
    cache_last_channel: np.ndarray = field(default_factory=lambda: np.zeros((1, N_LAYERS, LEFT_CACHE, D_MODEL), np.float32))
    cache_last_time: np.ndarray = field(default_factory=lambda: np.zeros((1, N_LAYERS, D_MODEL, TIME_CACHE), np.float32))
    cache_last_channel_len: np.ndarray = field(default_factory=lambda: np.zeros((1,), np.int64))
    audio: np.ndarray = field(default_factory=lambda: np.empty((0,), np.float32))
    mel_consumed: int = 0
    pending_mel: np.ndarray = field(default_factory=lambda: np.empty((80, 0), np.float32))
    token_ids: list[int] = field(default_factory=list)
    previous_ctc_id: int = BLANK_ID

    def append_pcm(self, payload: bytes) -> None:
        if len(payload) % 2:
            raise ValueError("PCM payload must contain whole signed-16-bit samples")
        self.audio = np.concatenate((self.audio, np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0))

    def _new_stable_mel(self, final: bool) -> None:
        assert runtime.cmvn is not None
        mel = log_mel(self.audio)
        stable = mel.shape[1] if final else max(0, mel.shape[1] - 2)
        if stable <= self.mel_consumed:
            return
        fresh = (mel[:, self.mel_consumed:stable] - runtime.cmvn["tlog_mean"][:, None]) / (runtime.cmvn["tlog_std"][:, None] + 1e-5)
        self.pending_mel = np.concatenate((self.pending_mel, fresh.astype(np.float32)), axis=1)
        self.mel_consumed = stable

    def _infer(self, features: np.ndarray) -> None:
        assert runtime.session is not None
        logprobs, self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = runtime.session.run(
            ["logprobs", "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"],
            {"audio_signal": features[None].astype(np.float32), "length": np.array([features.shape[1]], np.int64),
             "cache_last_channel": self.cache_last_channel, "cache_last_time": self.cache_last_time,
             "cache_last_channel_len": self.cache_last_channel_len})
        for token in logprobs[0].argmax(axis=-1).tolist():
            if token != BLANK_ID and token != self.previous_ctc_id:
                self.token_ids.append(int(token))
            self.previous_ctc_id = int(token)

    def process(self, final: bool = False) -> str:
        self._new_stable_mel(final)
        while self.pending_mel.shape[1] >= CHUNK_MEL:
            self._infer(self.pending_mel[:, :CHUNK_MEL]); self.pending_mel = self.pending_mel[:, CHUNK_MEL:]
        if final and self.pending_mel.shape[1]:
            self._infer(self.pending_mel); self.pending_mel = np.empty((80, 0), np.float32)
        assert runtime.tokenizer is not None
        return runtime.tokenizer.decode(self.token_ids)


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.load()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": runtime.session is not None}


@app.websocket("/ws/asr")
async def asr(websocket: WebSocket) -> None:
    await websocket.accept()
    state, started = StreamingSession(), False
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect": break
            if message.get("text") is not None:
                control = json.loads(message["text"])
                if control.get("type") == "start" and not started and control.get("sample_rate") == SAMPLE_RATE and control.get("format") == "pcm_s16le":
                    started = True
                elif control.get("type") == "stop" and started:
                    await websocket.send_json({"type":"final", "text":state.process(True), "is_final":True}); break
                else:
                    await websocket.send_json({"type":"error", "detail":"Send start, PCM frames, then stop"})
            elif message.get("bytes") is not None:
                if not started:
                    await websocket.send_json({"type":"error", "detail":"Send start before PCM frames"}); continue
                state.append_pcm(message["bytes"])
                await websocket.send_json({"type":"partial", "text":state.process(), "is_final":False})
    except (WebSocketDisconnect, ValueError):
        pass
