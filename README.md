# Quran ASR Runpod worker

Runpod Serverless worker for the 219 MB FP16 ONNX export of `Muno459/fastconformer-quran`.

Attach the Runpod secret `hf_quran_asr_read` as environment variable `HF_TOKEN`. Start with the lowest-cost available 16 GB GPU pool, 5 GB container disk, zero active workers, and two maximum workers.

```json
{"input":{"audio_base64":"<base64 WAV>"}}
```

This is a stateless complete-clip endpoint. Persistent microphone streaming needs a session-affine Pod/WebSocket service.
