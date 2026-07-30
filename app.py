"""Cache-aware Quran ASR streaming with final pronunciation diagnostics.

The ONNX model consumes normalized 80-bin log-mel features.  Model weights,
tokenizer, CMVN, the pronunciation checkpoint, and the publisher's scorer are
loaded exactly once per worker.  Mutable encoder/decoder state is per session.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import importlib.util
import io
import json
import logging
import os
import resource
import sys
import time
import traceback
import unicodedata
import wave
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
import sentencepiece as spm
import torch
from fastapi import FastAPI, Header, HTTPException, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

MODEL_ID = "Muno459/fastconformer-quran-streaming"
MODEL_FILE = "model_streaming_with_encoder.q8.onnx"
HEAD_FILE = "head/pronunciation_head.pt"
SCORER_FILE = "tajweed/head_scorer.py"
ASSET_FILES = [
    MODEL_FILE,
    HEAD_FILE,
    SCORER_FILE,
    "streaming_global_cmvn.npz",
    "tokenizer.model",
    "config.json",
]
SAMPLE_RATE = 16_000
N_LAYERS, D_MODEL, LEFT_CACHE, TIME_CACHE = 17, 512, 70, 8
CHUNK_MEL = 112
MIN_MODEL_MEL = 10
BLANK_ID = 1024
OUTPUT_HOP_S = 0.080
DIAGNOSTIC_PCM_CHUNK = 3_200  # 200 ms delivery chunks; model chunks remain 112 mel frames.
CPU_THREADS = max(1, int(os.getenv("CPU_THREADS", "4")))

EXPECTED_INPUTS = {
    "audio_signal": ("tensor(float)", ("B", 80, "T_in")),
    "length": ("tensor(int64)", ("B",)),
    "cache_last_channel": ("tensor(float)", ("B", 17, 70, 512)),
    "cache_last_time": ("tensor(float)", ("B", 17, 512, 8)),
    "cache_last_channel_len": ("tensor(int64)", ("B",)),
}
REQUIRED_OUTPUTS = {
    "logprobs",
    "encoder_output",
    "encoded_lengths",
    "cache_last_channel_next",
    "cache_last_time_next",
    "cache_last_channel_next_len",
}
NORMALIZATION_RULES = [
    "Unicode NFKC",
    "remove tatweel",
    "remove Arabic harakat, Quran annotation/waqf marks, and superscript alef",
    "normalize alef variants (أ, إ, آ, ٱ) to ا",
    "normalize alif maqsura ى to ي",
    "collapse whitespace",
]

logger = logging.getLogger("quran_asr")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")


def log_event(event: str, **fields: Any) -> None:
    """Emit credential-free structured logs."""
    safe = {"event": event, **fields}
    sensitive_names = {
        "authorization",
        "cookie",
        "secret",
        "password",
        "api_key",
        "huggingface_hub_token",
        "hf_token",
        "access_token",
    }
    for key in tuple(safe):
        lowered = key.lower()
        if lowered in sensitive_names or any(
            lowered.endswith(f"_{name}") for name in sensitive_names
        ):
            safe[key] = "[REDACTED]"
    logger.info(json.dumps(safe, ensure_ascii=False, default=str))


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return round(value / (1024 if os.uname().sysname == "Linux" else 1024 * 1024), 2)


def tensor_spec(value: Any) -> dict[str, Any]:
    return {"name": value.name, "dtype": value.type, "shape": list(value.shape)}


def _shape_matches(actual: list[Any], expected: tuple[Any, ...]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(want == got for got, want in zip(actual, expected) if isinstance(want, int))


def load_python_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ModelRuntime:
    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.tokenizer: spm.SentencePieceProcessor | None = None
        self.cmvn: dict[str, np.ndarray] | None = None
        self.pronunciation_scorer: Any | None = None
        self.loading = False
        self.load_error: str | None = None
        self.model_dir: Path | None = None
        self.input_specs: list[dict[str, Any]] = []
        self.output_specs: list[dict[str, Any]] = []
        self.load_metrics: dict[str, Any] = {}
        self.ctc_aligner_ready = False

    @property
    def asr_model_loaded(self) -> bool:
        return self.session is not None

    @property
    def tokenizer_loaded(self) -> bool:
        return self.tokenizer is not None

    @property
    def cmvn_loaded(self) -> bool:
        return self.cmvn is not None

    @property
    def pronunciation_head_loaded(self) -> bool:
        return self.pronunciation_scorer is not None

    @property
    def ready(self) -> bool:
        return all(
            (
                self.asr_model_loaded,
                self.tokenizer_loaded,
                self.cmvn_loaded,
                self.pronunciation_head_loaded,
                self.ctc_aligner_ready,
            )
        )

    def _validate_onnx_contract(self) -> None:
        assert self.session is not None
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_specs = [tensor_spec(item) for item in inputs]
        self.output_specs = [tensor_spec(item) for item in outputs]
        actual_inputs = {item.name: item for item in inputs}
        actual_outputs = {item.name for item in outputs}
        if set(actual_inputs) != set(EXPECTED_INPUTS):
            raise RuntimeError(f"unexpected ONNX inputs: {sorted(actual_inputs)}")
        if not REQUIRED_OUTPUTS.issubset(actual_outputs):
            raise RuntimeError(f"missing ONNX outputs: {sorted(REQUIRED_OUTPUTS - actual_outputs)}")
        for name, (dtype, shape) in EXPECTED_INPUTS.items():
            item = actual_inputs[name]
            if item.type != dtype or not _shape_matches(list(item.shape), shape):
                raise RuntimeError(
                    f"invalid ONNX input {name}: dtype={item.type}, shape={item.shape}"
                )
        by_name = {item.name: item for item in outputs}
        if by_name["logprobs"].type != "tensor(float)" or not _shape_matches(
            list(by_name["logprobs"].shape), ("B", "T_out", BLANK_ID + 1)
        ):
            raise RuntimeError("logprobs must be float [B,T,1025]")
        if by_name["encoder_output"].type != "tensor(float)" or not _shape_matches(
            list(by_name["encoder_output"].shape), ("B", D_MODEL, "T_out")
        ):
            raise RuntimeError("encoder_output must be float [B,512,T]")

    def load(self) -> None:
        started = time.perf_counter()
        model_dir = Path(os.getenv("MODEL_DIR", "/models/quran-asr"))
        model_dir.mkdir(parents=True, exist_ok=True)
        download_started = time.perf_counter()
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=model_dir,
            allow_patterns=ASSET_FILES,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
            local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
        )
        download_s = time.perf_counter() - download_started
        self.model_dir = model_dir

        onnx_started = time.perf_counter()
        torch.set_num_threads(CPU_THREADS)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = CPU_THREADS
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_dir / MODEL_FILE),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._validate_onnx_contract()
        onnx_s = time.perf_counter() - onnx_started

        tokenizer_started = time.perf_counter()
        tokenizer = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
        if tokenizer.get_piece_size() != BLANK_ID:
            raise RuntimeError(
                f"tokenizer vocab {tokenizer.get_piece_size()} does not match blank id {BLANK_ID}"
            )
        self.tokenizer = tokenizer
        tokenizer_s = time.perf_counter() - tokenizer_started

        with np.load(model_dir / "streaming_global_cmvn.npz") as cmvn_file:
            cmvn = {name: np.asarray(cmvn_file[name], dtype=np.float32) for name in cmvn_file.files}
        for name in ("tlog_mean", "tlog_std"):
            if name not in cmvn or cmvn[name].shape != (80,):
                raise RuntimeError(f"invalid CMVN tensor {name}")
        if np.any(cmvn["tlog_std"] <= 0):
            raise RuntimeError("CMVN standard deviation must be positive")
        self.cmvn = cmvn

        checkpoint_started = time.perf_counter()
        scorer_path = model_dir / SCORER_FILE
        module = load_python_module(scorer_path, "publisher_head_scorer")
        scorer = module.HeadPronunciationScorer(model_dir / HEAD_FILE, device="cpu")
        feature_table = scorer.feature_table
        if tuple(feature_table.shape) != (BLANK_ID + 1, 16):
            raise RuntimeError(f"unexpected feature_table shape {tuple(feature_table.shape)}")
        model_enc_dim = scorer.model.mlp[0].in_features - scorer.model.tok_emb.embedding_dim - 16
        if model_enc_dim != D_MODEL:
            raise RuntimeError(f"pronunciation head encoder dimension is {model_enc_dim}, not 512")
        self.pronunciation_scorer = scorer
        checkpoint_s = time.perf_counter() - checkpoint_started

        self.ctc_aligner_ready = True
        self.load_metrics = {
            "download_s": round(download_s, 3),
            "onnx_load_s": round(onnx_s, 3),
            "tokenizer_load_s": round(tokenizer_s, 3),
            "checkpoint_load_s": round(checkpoint_s, 3),
            "total_s": round(time.perf_counter() - started, 3),
            "peak_rss_mb": peak_rss_mb(),
            "cpu_threads": CPU_THREADS,
        }
        log_event(
            "runtime_loaded",
            **self.load_metrics,
            inputs=self.input_specs,
            outputs=self.output_specs,
        )


runtime = ModelRuntime()


def _hz_to_mel_slaney(freq: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3.0
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return torch.where(
        freq >= min_log_hz,
        min_log_mel + torch.log(freq / min_log_hz) / logstep,
        mels,
    )


def _mel_filterbank() -> torch.Tensor:
    all_freqs = torch.linspace(0, SAMPLE_RATE // 2, 512 // 2 + 1)
    m_min = _hz_to_mel_slaney(torch.tensor(0.0))
    m_max = _hz_to_mel_slaney(torch.tensor(float(SAMPLE_RATE // 2)))
    m_pts = torch.linspace(m_min, m_max, 82)
    # Inverse of Slaney's piecewise hz->mel conversion.
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    f_pts = torch.where(
        m_pts >= min_log_mel,
        min_log_hz * torch.exp(logstep * (m_pts - min_log_mel)),
        f_sp * m_pts,
    )
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[:, None] - all_freqs[None, :]
    down = (-slopes[:-2]) / f_diff[:-1, None]
    up = slopes[2:] / f_diff[1:, None]
    fb = torch.clamp(torch.minimum(down, up), min=0.0)
    fb *= (2.0 / (f_pts[2:] - f_pts[:-2]))[:, None]
    return fb


MEL_FILTERBANK = _mel_filterbank()


def log_mel(wav: np.ndarray) -> np.ndarray:
    """NeMo-compatible 80-bin log-mel without a second model dependency."""
    tensor = torch.as_tensor(np.asarray(wav, dtype=np.float32))
    if tensor.numel() == 0:
        return np.empty((80, 0), dtype=np.float32)
    tensor = torch.cat((tensor[:1], tensor[1:] - 0.97 * tensor[:-1]))
    spectrum = torch.stft(
        tensor,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=torch.hann_window(400),
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    ).abs().pow(2.0)
    mel = MEL_FILTERBANK.to(spectrum.device) @ spectrum
    return torch.log(mel + 2**-24).cpu().numpy().astype(np.float32)


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("ـ", "")
    normalized: list[str] = []
    for char in text:
        code = ord(char)
        if (
            unicodedata.category(char) in {"Mn", "Me"}
            or 0x0610 <= code <= 0x061A
            or 0x064B <= code <= 0x065F
            or code == 0x0670
            or 0x06D6 <= code <= 0x06ED
        ):
            continue
        normalized.append({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"}.get(char, char))
    return " ".join("".join(normalized).split())


@dataclass
class TokenInterval:
    token_id: int
    token_str: str
    start_s: float
    end_s: float
    asr_confidence: float
    start_frame: int
    end_frame: int
    word_index: int | None = None


def _valid_token_id(token_id: int) -> bool:
    if not 0 <= token_id < BLANK_ID:
        return False
    assert runtime.tokenizer is not None
    return not (
        runtime.tokenizer.is_unknown(token_id)
        or runtime.tokenizer.is_control(token_id)
        or runtime.tokenizer.is_unused(token_id)
    )


def _assign_word_indices(intervals: list[TokenInterval]) -> None:
    word_index = -1
    for index, interval in enumerate(intervals):
        if interval.token_str.startswith("▁") or index == 0:
            word_index += 1
        interval.word_index = word_index


def greedy_intervals(logprobs: np.ndarray) -> tuple[list[int], list[TokenInterval]]:
    assert runtime.tokenizer is not None
    best = logprobs.argmax(axis=-1)
    intervals: list[TokenInterval] = []
    token_ids: list[int] = []
    frame = 0
    while frame < len(best):
        token_id = int(best[frame])
        end = frame + 1
        while end < len(best) and int(best[end]) == token_id:
            end += 1
        if _valid_token_id(token_id):
            confidence = float(np.exp(logprobs[frame:end, token_id]).mean())
            token_ids.append(token_id)
            intervals.append(
                TokenInterval(
                    token_id=token_id,
                    token_str=runtime.tokenizer.id_to_piece(token_id),
                    start_s=frame * OUTPUT_HOP_S,
                    end_s=end * OUTPUT_HOP_S,
                    asr_confidence=round(confidence, 4),
                    start_frame=frame,
                    end_frame=end,
                )
            )
        frame = end
    _assign_word_indices(intervals)
    return token_ids, intervals


def ctc_forced_align(logprobs: np.ndarray, token_ids: list[int]) -> list[TokenInterval]:
    """Viterbi CTC alignment of expected SentencePiece ids to model frames."""
    assert runtime.tokenizer is not None
    if not token_ids:
        return []
    if any(not _valid_token_id(token_id) for token_id in token_ids):
        raise ValueError("reference contains unknown/control/out-of-vocabulary token")
    frame_count = int(logprobs.shape[0])
    states: list[int] = [BLANK_ID]
    for token_id in token_ids:
        states.extend((token_id, BLANK_ID))
    state_count = len(states)
    if frame_count < len(token_ids):
        raise ValueError(
            f"alignment impossible: {frame_count} encoder frames for {len(token_ids)} tokens"
        )
    previous = np.full(state_count, -np.inf, dtype=np.float64)
    previous[0] = float(logprobs[0, BLANK_ID])
    if state_count > 1:
        previous[1] = float(logprobs[0, states[1]])
    backpointers = np.full((frame_count, state_count), -1, dtype=np.int32)
    for frame in range(1, frame_count):
        current = np.full(state_count, -np.inf, dtype=np.float64)
        for state_index, token_id in enumerate(states):
            candidates = [(previous[state_index], state_index)]
            if state_index > 0:
                candidates.append((previous[state_index - 1], state_index - 1))
            if (
                state_index > 1
                and token_id != BLANK_ID
                and token_id != states[state_index - 2]
            ):
                candidates.append((previous[state_index - 2], state_index - 2))
            score, origin = max(candidates, key=lambda item: item[0])
            current[state_index] = score + float(logprobs[frame, token_id])
            backpointers[frame, state_index] = origin
        previous = current
    final_candidates = [state_count - 1]
    if state_count > 1:
        final_candidates.append(state_count - 2)
    final_state = max(final_candidates, key=lambda index: previous[index])
    if not np.isfinite(previous[final_state]):
        raise ValueError("no finite CTC alignment path")
    state_path = np.empty(frame_count, dtype=np.int32)
    state_path[-1] = final_state
    for frame in range(frame_count - 1, 0, -1):
        origin = backpointers[frame, state_path[frame]]
        if origin < 0:
            raise ValueError("broken CTC alignment backpointer")
        state_path[frame - 1] = origin

    intervals: list[TokenInterval] = []
    for token_index, token_id in enumerate(token_ids):
        target_state = token_index * 2 + 1
        frames = np.flatnonzero(state_path == target_state)
        if frames.size == 0:
            raise ValueError(f"token {token_index} was not assigned an encoder frame")
        start = int(frames[0])
        end = int(frames[-1]) + 1
        confidence = float(np.exp(logprobs[frames, token_id]).mean())
        intervals.append(
            TokenInterval(
                token_id=token_id,
                token_str=runtime.tokenizer.id_to_piece(token_id),
                start_s=start * OUTPUT_HOP_S,
                end_s=end * OUTPUT_HOP_S,
                asr_confidence=round(confidence, 4),
                start_frame=start,
                end_frame=end,
            )
        )
    _assign_word_indices(intervals)
    validate_intervals(intervals, frame_count)
    return intervals


def validate_intervals(intervals: list[TokenInterval], frame_count: int) -> None:
    previous_end = 0
    for interval in intervals:
        if not _valid_token_id(interval.token_id):
            raise ValueError(f"invalid pronunciation token id {interval.token_id}")
        if not (0 <= interval.start_frame < interval.end_frame <= frame_count):
            raise ValueError(f"out-of-range token interval {interval}")
        if interval.start_frame < previous_end:
            raise ValueError("token intervals overlap or are unordered")
        previous_end = interval.end_frame


@dataclass
class InferenceBundle:
    logprobs: np.ndarray
    encoder_features: np.ndarray
    chunk_latencies_ms: list[float]
    chunk_count: int


@dataclass
class StreamingSession:
    """All mutable recognizer and model cache state for exactly one connection."""

    cache_last_channel: np.ndarray = field(
        default_factory=lambda: np.zeros((1, N_LAYERS, LEFT_CACHE, D_MODEL), np.float32)
    )
    cache_last_time: np.ndarray = field(
        default_factory=lambda: np.zeros((1, N_LAYERS, D_MODEL, TIME_CACHE), np.float32)
    )
    cache_last_channel_len: np.ndarray = field(default_factory=lambda: np.zeros((1,), np.int64))
    audio: np.ndarray = field(default_factory=lambda: np.empty((0,), np.float32))
    mel_consumed: int = 0
    pending_mel: np.ndarray = field(default_factory=lambda: np.empty((80, 0), np.float32))
    logprob_chunks: list[np.ndarray] = field(default_factory=list)
    encoder_chunks: list[np.ndarray] = field(default_factory=list)
    chunk_latencies_ms: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    previous_ctc_id: int = BLANK_ID
    finalized: bool = False

    def append_float(self, samples: np.ndarray) -> None:
        if self.finalized:
            raise ValueError("session is already finalized")
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("audio contains NaN or infinity")
        self.audio = np.concatenate((self.audio, values))

    def append_pcm(self, payload: bytes) -> None:
        if len(payload) % 2:
            raise ValueError("PCM payload must contain whole signed-16-bit samples")
        self.append_float(np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0)

    def _new_stable_mel(self, final: bool) -> None:
        assert runtime.cmvn is not None
        estimated_stable = max(0, 1 + len(self.audio) // 160 - (0 if final else 2))
        needed = CHUNK_MEL - self.pending_mel.shape[1]
        if not final and estimated_stable - self.mel_consumed < needed:
            return
        mel = log_mel(self.audio)
        stable_frames = mel.shape[1] if final else max(0, mel.shape[1] - 2)
        if stable_frames <= self.mel_consumed:
            return
        mean = runtime.cmvn["tlog_mean"][:, None]
        std = runtime.cmvn["tlog_std"][:, None]
        fresh = (mel[:, self.mel_consumed:stable_frames] - mean) / (std + 1e-5)
        self.pending_mel = np.concatenate(
            (self.pending_mel, fresh.astype(np.float32)), axis=1
        )
        self.mel_consumed = stable_frames

    def _infer(self, features: np.ndarray) -> None:
        assert runtime.session is not None
        original_length = features.shape[1]
        if original_length < MIN_MODEL_MEL:
            features = np.pad(features, ((0, 0), (0, MIN_MODEL_MEL - original_length)))
        inputs = {
            "audio_signal": features[None].astype(np.float32),
            "length": np.array([original_length], dtype=np.int64),
            "cache_last_channel": self.cache_last_channel,
            "cache_last_time": self.cache_last_time,
            "cache_last_channel_len": self.cache_last_channel_len,
        }
        started = time.perf_counter()
        (
            logprobs,
            encoder_output,
            encoded_lengths,
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = runtime.session.run(
            [
                "logprobs",
                "encoder_output",
                "encoded_lengths",
                "cache_last_channel_next",
                "cache_last_time_next",
                "cache_last_channel_next_len",
            ],
            inputs,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        self.chunk_latencies_ms.append(latency_ms)
        encoded_length = int(encoded_lengths[0])
        logprobs = np.asarray(logprobs[0, :encoded_length], dtype=np.float32)
        encoder = np.asarray(encoder_output[0, :, :encoded_length].T, dtype=np.float32)
        if logprobs.shape != (encoded_length, BLANK_ID + 1):
            raise RuntimeError(f"invalid logprobs shape {logprobs.shape}")
        if encoder.shape != (encoded_length, D_MODEL):
            raise RuntimeError(f"invalid encoder_output shape {encoder.shape}")
        self.logprob_chunks.append(logprobs)
        self.encoder_chunks.append(encoder)
        for token in logprobs.argmax(axis=-1).tolist():
            token = int(token)
            if token != BLANK_ID and token != self.previous_ctc_id and _valid_token_id(token):
                self.token_ids.append(token)
            self.previous_ctc_id = token
        log_event(
            "chunk_inference",
            chunk_index=len(self.logprob_chunks) - 1,
            input_mel_frames=original_length,
            output_frames=encoded_length,
            latency_ms=round(latency_ms, 3),
        )

    def process(self, final: bool = False) -> str:
        if final and self.finalized:
            raise ValueError("session final flush was already performed")
        self._new_stable_mel(final=final)
        while self.pending_mel.shape[1] >= CHUNK_MEL:
            self._infer(self.pending_mel[:, :CHUNK_MEL])
            self.pending_mel = self.pending_mel[:, CHUNK_MEL:]
        if final and self.pending_mel.shape[1]:
            self._infer(self.pending_mel)
            self.pending_mel = np.empty((80, 0), np.float32)
        if final:
            self.finalized = True
        assert runtime.tokenizer is not None
        return runtime.tokenizer.decode(self.token_ids)

    def bundle(self) -> InferenceBundle:
        if not self.finalized:
            raise ValueError("final flush is required before diagnostics")
        logprobs = (
            np.concatenate(self.logprob_chunks, axis=0)
            if self.logprob_chunks
            else np.empty((0, BLANK_ID + 1), np.float32)
        )
        encoder = (
            np.concatenate(self.encoder_chunks, axis=0)
            if self.encoder_chunks
            else np.empty((0, D_MODEL), np.float32)
        )
        if logprobs.shape[0] != encoder.shape[0]:
            raise RuntimeError("logprobs and encoder_output time axes differ")
        return InferenceBundle(
            logprobs=logprobs,
            encoder_features=encoder,
            chunk_latencies_ms=list(self.chunk_latencies_ms),
            chunk_count=len(self.logprob_chunks),
        )


def _percentile(values: list[float], percentile: float) -> float:
    return round(float(np.percentile(values, percentile)), 3) if values else 0.0


def _word_error_pairs(reference: str, recognized: str) -> dict[int, str]:
    reference_words = normalize_arabic(reference).split()
    recognized_words = normalize_arabic(recognized).split()
    errors: dict[int, str] = {}
    for index, word in enumerate(reference_words):
        if index >= len(recognized_words) or recognized_words[index] != word:
            errors[index] = recognized_words[index] if index < len(recognized_words) else ""
    return errors


def _score_tokens(
    bundle: InferenceBundle, reference_text: str, recognized_text: str
) -> tuple[list[TokenInterval], list[dict[str, Any]], list[str]]:
    assert runtime.tokenizer is not None
    assert runtime.pronunciation_scorer is not None
    warnings = ["THRESHOLD_DOCUMENTATION_MISMATCH"]
    reference_ids = runtime.tokenizer.encode(reference_text, out_type=int)
    recognized_ids, _ = greedy_intervals(bundle.logprobs)
    alignment_ids = reference_ids
    if (
        normalize_arabic(reference_text) == normalize_arabic(recognized_text)
        and reference_ids != recognized_ids
    ):
        # SentencePiece uses identity normalization and its vocabulary contains
        # non-canonical Arabic combining-mark order (for example shadda before
        # fatha). Canonically equivalent input may otherwise force-align to
        # near-zero posterior ids. Preserve/report the raw difference, but pool
        # the actual model token sequence for pronunciation scoring.
        alignment_ids = recognized_ids
        warnings.append(
            "TOKENIZATION_ERROR: canonically equivalent text produced different "
            "SentencePiece ids; alignment used recognized ids"
        )
    try:
        intervals = ctc_forced_align(bundle.logprobs, alignment_ids)
    except ValueError as error:
        warnings.append(f"ALIGNMENT_ERROR: {error}")
        _, intervals = greedy_intervals(bundle.logprobs)
    validate_intervals(intervals, bundle.encoder_features.shape[0])
    scores = runtime.pronunciation_scorer.score(
        bundle.encoder_features, intervals, output_hop_s=OUTPUT_HOP_S
    )
    if len(scores) != len(intervals):
        raise RuntimeError(
            f"pronunciation scorer returned {len(scores)} scores for {len(intervals)} intervals"
        )
    token_rows: list[dict[str, Any]] = []
    for interval, score in zip(intervals, scores):
        row = asdict(score)
        row.update(
            {
                "token": interval.token_str,
                "asr_confidence": interval.asr_confidence,
                "start_ms": round(interval.start_s * 1000),
                "end_ms": round(interval.end_s * 1000),
                "word_index": interval.word_index,
            }
        )
        token_rows.append(row)
    return intervals, token_rows, warnings


def analyze_bundle(
    bundle: InferenceBundle,
    reference_text: str,
    *,
    finalization_ms: float,
) -> dict[str, Any]:
    assert runtime.tokenizer is not None
    decoded_ids, _ = greedy_intervals(bundle.logprobs)
    recognized = runtime.tokenizer.decode(decoded_ids)
    normalized = normalize_arabic(recognized)
    normalized_reference = normalize_arabic(reference_text)
    intervals, token_rows, warnings = _score_tokens(bundle, reference_text, recognized)
    word_errors = _word_error_pairs(reference_text, recognized)
    reference_words = reference_text.split()
    normalized_reference_words = normalized_reference.split()
    normalized_recognized_words = normalized.split()

    words: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for word_index, reference_word in enumerate(reference_words):
        related = [row for row in token_rows if row["word_index"] == word_index]
        recognized_word = (
            normalized_recognized_words[word_index]
            if word_index < len(normalized_recognized_words)
            else ""
        )
        if related:
            start_ms = min(row["start_ms"] for row in related)
            end_ms = max(row["end_ms"] for row in related)
            probability = min(row["prob_correct"] for row in related)
            worst = min(related, key=lambda row: row["prob_correct"])
            deviation = worst["deviation"]
            asr_confidence = round(
                float(np.mean([row["asr_confidence"] for row in related])), 4
            )
        else:
            start_ms = end_ms = None
            probability = None
            deviation = "unknown"
            asr_confidence = 0.0
            worst = None
        sequence_error = word_index in word_errors
        error_type: str | None = None
        explanation = "Urutan kata dan pronunciation score tidak menunjukkan masalah."
        diagnosis_confidence = "sedang"
        if sequence_error:
            error_type = "ASR_ERROR"
            explanation = (
                "Transcript berbeda dari referensi setelah normalisasi; ini belum membuktikan "
                "kesalahan bacaan karena ASR atau alignment dapat menjadi penyebab."
            )
            diagnosis_confidence = "tinggi"
        elif deviation in {"minor", "major"}:
            error_type = "PRONUNCIATION_ERROR"
            explanation = (
                "Urutan kata benar, tetapi pronunciation head memberi probabilitas rendah "
                f"pada token {worst['token']!r}; false positive tetap mungkin."
            )
            diagnosis_confidence = "rendah" if deviation == "minor" else "sedang"
        row = {
            "word_index": word_index,
            "reference": reference_word,
            "recognized": recognized_word,
            "normalized_reference": (
                normalized_reference_words[word_index]
                if word_index < len(normalized_reference_words)
                else ""
            ),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "asr_confidence": asr_confidence,
            "pronunciation_probability": probability,
            "deviation": deviation,
            "suspect_token": worst["token"] if worst else None,
            "error_type": error_type,
            "explanation": explanation,
            "diagnosis_confidence": diagnosis_confidence,
            "pipeline_suspected": bool(sequence_error or deviation == "minor"),
        }
        words.append(row)
        if error_type:
            errors.append(row)

    pronunciation_ok = sum(row["deviation"] == "ok" for row in token_rows)
    pronunciation_minor = sum(row["deviation"] == "minor" for row in token_rows)
    pronunciation_major = sum(row["deviation"] == "major" for row in token_rows)
    return {
        "recognized_text": recognized,
        "normalized_text": normalized,
        "sequence_correct": normalized == normalized_reference,
        "tokens": token_rows,
        "words": words,
        "errors": errors,
        "summary": {
            "asr_errors": sum(row["error_type"] == "ASR_ERROR" for row in errors),
            "alignment_errors": sum(warning.startswith("ALIGNMENT_ERROR") for warning in warnings),
            "tokenization_errors": sum(
                warning.startswith("TOKENIZATION_ERROR") for warning in warnings
            ),
            "pronunciation_ok": pronunciation_ok,
            "pronunciation_minor": pronunciation_minor,
            "pronunciation_major": pronunciation_major,
            "pipeline_warnings": warnings,
        },
        "performance": {
            "chunk_count": bundle.chunk_count,
            "chunk_latency_ms": [round(value, 3) for value in bundle.chunk_latencies_ms],
            "mean_chunk_latency_ms": round(
                float(np.mean(bundle.chunk_latencies_ms)), 3
            )
            if bundle.chunk_latencies_ms
            else 0.0,
            "p95_chunk_latency_ms": _percentile(bundle.chunk_latencies_ms, 95),
            "finalization_ms": round(finalization_ms, 3),
            "encoder_frames": int(bundle.encoder_features.shape[0]),
            "alignment_token_count": len(intervals),
            "pronunciation_token_count": len(token_rows),
        },
    }


def _run_mode(
    audio: np.ndarray,
    reference_text: str,
    *,
    external_chunk_samples: int | None,
) -> tuple[dict[str, Any], list[str]]:
    state = StreamingSession()
    partials: list[str] = []
    if external_chunk_samples is None:
        state.append_float(audio)
    else:
        for start in range(0, len(audio), external_chunk_samples):
            state.append_float(audio[start : start + external_chunk_samples])
            partials.append(state.process(final=False))
    started = time.perf_counter()
    state.process(final=True)
    finalization_ms = (time.perf_counter() - started) * 1000
    return (
        analyze_bundle(state.bundle(), reference_text, finalization_ms=finalization_ms),
        partials,
    )


def run_diagnostics(
    audio: np.ndarray,
    *,
    surah: int,
    ayah: int,
    reference_text: str,
    source: str,
    qari: str,
) -> dict[str, Any]:
    if not runtime.ready:
        raise RuntimeError("runtime is not ready")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not audio.size:
        raise ValueError("audio is empty")
    if len(audio) / SAMPLE_RATE > 300:
        raise ValueError("diagnostic audio exceeds five minutes")
    process = psutil.Process()
    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()
    full, _ = _run_mode(audio, reference_text, external_chunk_samples=None)
    streaming, partials = _run_mode(
        audio,
        reference_text,
        external_chunk_samples=DIAGNOSTIC_PCM_CHUNK,
    )
    wall_s = time.perf_counter() - wall_start
    cpu_end = process.cpu_times()
    cpu_s = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    transcript_match = full["normalized_text"] == streaming["normalized_text"]
    token_count_match = len(full["tokens"]) == len(streaming["tokens"])
    paired_tokens = list(zip(full["tokens"], streaming["tokens"]))
    token_ids_match = token_count_match and all(
        first["token_id"] == second["token_id"] for first, second in paired_tokens
    )
    max_timestamp_delta_ms = max(
        (
            max(
                abs(first["start_ms"] - second["start_ms"]),
                abs(first["end_ms"] - second["end_ms"]),
            )
            for first, second in paired_tokens
        ),
        default=0,
    )
    max_probability_delta = max(
        (
            abs(first["prob_correct"] - second["prob_correct"])
            for first, second in paired_tokens
        ),
        default=0.0,
    )
    partial_changes = sum(
        current != previous for previous, current in zip(partials, partials[1:])
    )
    partial_retractions = sum(
        bool(previous) and not current.startswith(previous)
        for previous, current in zip(partials, partials[1:])
    )
    comparison_warnings: list[str] = []
    if not transcript_match or not token_ids_match or partial_retractions:
        comparison_warnings.append("STREAMING_BOUNDARY_ERROR")
        streaming["summary"]["pipeline_warnings"].append("STREAMING_BOUNDARY_ERROR")
    result = {
        "reference": {
            "surah": surah,
            "ayah": ayah,
            "text": reference_text,
            "normalized_text": normalize_arabic(reference_text),
            "normalization_rules": NORMALIZATION_RULES,
        },
        "audio": {
            "source": source,
            "qari": qari,
            "duration_s": round(len(audio) / SAMPLE_RATE, 3),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "encoding": "PCM float32 after WAV decode",
        },
        "full_file": full,
        "streaming": {**streaming, "partials": partials},
        "comparison": {
            "transcript_match": transcript_match,
            "token_count_match": token_count_match,
            "token_ids_match": token_ids_match,
            "alignment_match": token_ids_match and max_timestamp_delta_ms == 0,
            "pronunciation_scores_match": max_probability_delta <= 0.0001,
            "max_timestamp_delta_ms": max_timestamp_delta_ms,
            "max_pronunciation_probability_delta": round(max_probability_delta, 6),
            "partial_changes": partial_changes,
            "partial_retractions": partial_retractions,
            "full_normalized_text": full["normalized_text"],
            "streaming_normalized_text": streaming["normalized_text"],
            "warnings": comparison_warnings,
        },
        "server_metrics": {
            "wall_s": round(wall_s, 3),
            "cpu_time_s": round(cpu_s, 3),
            "cpu_percent_of_one_core": round(100 * cpu_s / max(wall_s, 1e-9), 2),
            "rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "peak_rss_mb": peak_rss_mb(),
        },
    }
    # Stable convenience fields for diagnostic clients; detailed comparison
    # remains available under full_file/streaming.
    result.update(
        {
            "recognized_text": streaming["recognized_text"],
            "normalized_text": streaming["normalized_text"],
            "sequence_correct": streaming["sequence_correct"],
            "tokens": streaming["tokens"],
            "errors": streaming["errors"],
            "summary": streaming["summary"],
        }
    )
    log_event(
        "diagnostic_complete",
        surah=surah,
        ayah=ayah,
        audio_duration_s=result["audio"]["duration_s"],
        full_chunks=full["performance"]["chunk_count"],
        streaming_chunks=streaming["performance"]["chunk_count"],
        transcript_match=transcript_match,
        alignment_tokens=streaming["performance"]["alignment_token_count"],
        pronunciation_tokens=streaming["performance"]["pronunciation_token_count"],
        wall_s=result["server_metrics"]["wall_s"],
        peak_rss_mb=result["server_metrics"]["peak_rss_mb"],
        cpu_percent=result["server_metrics"]["cpu_percent_of_one_core"],
    )
    return result


def decode_pcm16_wav(payload: bytes) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            if wav_file.getframerate() != SAMPLE_RATE:
                raise ValueError(f"sample rate must be {SAMPLE_RATE}")
            if wav_file.getnchannels() != 1:
                raise ValueError("audio must be mono")
            if wav_file.getsampwidth() != 2 or wav_file.getcomptype() != "NONE":
                raise ValueError("audio must be uncompressed signed 16-bit PCM WAV")
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error as error:
        raise ValueError(f"invalid WAV: {error}") from error
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


class DiagnosticRequest(BaseModel):
    audio_wav_base64: str = Field(min_length=4)
    surah: int = Field(ge=1, le=114)
    ayah: int = Field(ge=1)
    reference_text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    qari: str = Field(min_length=1)


async def load_runtime() -> None:
    """Load once without blocking health/readiness routes during cold start."""
    try:
        await asyncio.to_thread(runtime.load)
    except Exception as error:
        runtime.load_error = f"{type(error).__name__}: {error}"
        log_event(
            "runtime_load_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        runtime.loading = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.loading = True
    runtime.load_error = None
    load_task = asyncio.create_task(load_runtime())
    try:
        yield
    finally:
        if not load_task.done():
            load_task.cancel()
            with suppress(asyncio.CancelledError):
                await load_task


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if runtime.ready else ("error" if runtime.load_error else "loading"),
        "asr_model_loaded": runtime.asr_model_loaded,
        "pronunciation_head_loaded": runtime.pronunciation_head_loaded,
        "ctc_aligner_ready": runtime.ctc_aligner_ready,
        "tokenizer_loaded": runtime.tokenizer_loaded,
        "cmvn_loaded": runtime.cmvn_loaded,
        "loading": runtime.loading,
        "load_error": runtime.load_error,
        "model_id": MODEL_ID,
        "model_file": MODEL_FILE,
        "inputs": runtime.input_specs,
        "outputs": runtime.output_specs,
        "load_metrics": runtime.load_metrics,
    }


@app.get("/ping")
def ping() -> Response:
    """204 until every required component is ready; 200 only when all are ready."""
    if runtime.ready:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/diagnostics/audio")
async def diagnostics_audio(
    request: DiagnosticRequest,
    x_diagnostics_key: str | None = Header(default=None),
) -> dict[str, Any]:
    configured_key = os.getenv("DIAGNOSTICS_API_KEY")
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="diagnostic endpoint is disabled",
        )
    if not x_diagnostics_key or not hmac.compare_digest(x_diagnostics_key, configured_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")
    if not runtime.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="loading")
    try:
        payload = base64.b64decode(request.audio_wav_base64, validate=True)
        audio = decode_pcm16_wav(payload)
        return await asyncio.to_thread(
            run_diagnostics,
            audio,
            surah=request.surah,
            ayah=request.ayah,
            reference_text=request.reference_text,
            source=request.source,
            qari=request.qari,
        )
    except (ValueError, base64.binascii.Error) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))
    except Exception as error:
        log_event(
            "diagnostic_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="diagnostic processing failed",
        )


@app.websocket("/ws/asr")
async def asr(websocket: WebSocket) -> None:
    await websocket.accept()
    if not runtime.ready:
        await websocket.send_json({"type": "error", "detail": "Model is still loading"})
        await websocket.close(code=1013, reason="Model is still loading")
        return
    state = StreamingSession()
    started = False
    reference: dict[str, Any] | None = None
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                control = json.loads(message["text"])
                if control.get("type") == "start":
                    if (
                        started
                        or control.get("sample_rate") != SAMPLE_RATE
                        or control.get("format") != "pcm_s16le"
                    ):
                        await websocket.send_json(
                            {
                                "type": "error",
                                "detail": "Expected one 16 kHz pcm_s16le start message",
                            }
                        )
                        continue
                    started = True
                    if control.get("reference_text"):
                        reference = {
                            "surah": int(control.get("surah", 1)),
                            "ayah": int(control.get("ayah", 1)),
                            "reference_text": str(control["reference_text"]),
                            "source": str(control.get("source", "websocket")),
                            "qari": str(control.get("qari", "unknown")),
                        }
                elif control.get("type") == "stop" and started:
                    if reference:
                        result = await asyncio.to_thread(
                            run_diagnostics,
                            state.audio,
                            **reference,
                        )
                        await websocket.send_json(
                            {
                                "type": "final",
                                "text": result["streaming"]["recognized_text"],
                                "is_final": True,
                                "diagnostics": result,
                            }
                        )
                    else:
                        final_started = time.perf_counter()
                        text = state.process(final=True)
                        finalization_ms = (time.perf_counter() - final_started) * 1000
                        diagnostics = analyze_bundle(
                            state.bundle(),
                            text,
                            finalization_ms=finalization_ms,
                        )
                        await websocket.send_json(
                            {
                                "type": "final",
                                "text": text,
                                "is_final": True,
                                "diagnostics": diagnostics,
                            }
                        )
                    break
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "detail": "Send start, binary PCM frames, then stop",
                        }
                    )
            elif message.get("bytes") is not None:
                if not started:
                    await websocket.send_json(
                        {"type": "error", "detail": "Send start before PCM frames"}
                    )
                    continue
                state.append_pcm(message["bytes"])
                await websocket.send_json(
                    {"type": "partial", "text": state.process(), "is_final": False}
                )
    except WebSocketDisconnect:
        pass
    except Exception as error:
        log_event(
            "websocket_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        with suppress(Exception):
            await websocket.send_json({"type": "error", "detail": "processing failed"})
    finally:
        with suppress(Exception):
            await websocket.close()
