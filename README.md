# Quran ASR streaming + pronunciation diagnostics

RunPod Load Balancer service on port `8000` for
`Muno459/fastconformer-quran-streaming`.

The worker downloads these gated assets once at startup:

- `model_streaming_with_encoder.q8.onnx`
- `head/pronunciation_head.pt`
- `tajweed/head_scorer.py`
- `streaming_global_cmvn.npz`
- `tokenizer.model`
- `config.json`

`HUGGINGFACE_HUB_TOKEN` must be a RunPod Secret; never bake it into the image.
The checkpoint is loaded once with `HeadPronunciationScorer(..., device="cpu")`.

## Readiness and monitoring

- `GET /ping`: HTTP 204 until ONNX, tokenizer, CMVN, CTC aligner, and
  pronunciation head are all ready; HTTP 200 afterwards.
- `GET /health`: component flags, sanitized load error, model tensor contract,
  and load timings.

The verified ONNX contract is:

| Direction | Name | dtype | shape |
|---|---|---|---|
| input | `audio_signal` | float32 | `[B,80,T_in]` |
| input | `length` | int64 | `[B]` |
| input | `cache_last_channel` | float32 | `[B,17,70,512]` |
| input | `cache_last_time` | float32 | `[B,17,512,8]` |
| input | `cache_last_channel_len` | int64 | `[B]` |
| output | `logprobs` | float32 | `[B,T_out,1025]` |
| output | `encoder_output` | float32 | `[B,512,T_out]` |
| output | `encoded_lengths` | int64 | `[B]` |
| output | `cache_last_channel_next` | float32 | `[B,17,*,512]` |
| output | `cache_last_time_next` | float32 | `[B,17,512,*]` |
| output | `cache_last_channel_next_len` | int64 | `[B]` |

Every inference step requests both `logprobs` and `encoder_output`. Mutable
cache state is per WebSocket. Final diagnostics concatenate encoder frames,
perform CTC forced alignment, pool 512-dimensional features per token, and run
the pronunciation head once on the final alignment.

## Diagnostic modes

`POST /diagnostics/audio` accepts a base64-encoded 16 kHz mono PCM16 WAV plus
reference metadata. It is disabled unless `DIAGNOSTICS_API_KEY` is configured
and requires the same value in `X-Diagnostics-Key`.

The same pipeline can be run without opening an HTTP endpoint:

```sh
python diagnose.py sample.wav \
  --surah 112 --ayah 1 \
  --reference 'قُلْ هُوَ اللَّهُ أَحَدٌ' \
  --source 'EveryAyah' --qari 'Abdullah Basfar'
```

The response contains full-file delivery, simulated 200 ms PCM streaming,
partial transcripts, final token/word diagnosis, latency, CPU/RAM, and a
comparison. Internally both modes use the official 112-mel-frame model chunk;
feeding a whole utterance as one ONNX call is invalid for this cache-aware
export.

## Build

```sh
docker build --no-cache \
  -t REGISTRY/OWNER/quran-asr-streaming:diagnostic-pronunciation-v3-final .
docker push REGISTRY/OWNER/quran-asr-streaming:diagnostic-pronunciation-v3-final
```

The Docker build runs all asset-independent regressions. Before releasing the
image, run `diagnose.py` against the real gated assets and a known Quran sample.

## RunPod settings

- CPU: 5 GHz Compute-Optimized, 4 vCPU / 8 GB
- Runtime thread cap: `CPU_THREADS=4`
- Active workers: 0; maximum workers: 3
- Autoscaling: request count, one request per worker
- Idle timeout: 20 seconds
- HTTP port: 8000
- Health endpoint: `/ping`
