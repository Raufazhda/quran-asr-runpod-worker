from __future__ import annotations

from dataclasses import dataclass
import asyncio

import numpy as np


def _ready_runtime(module):
    module.runtime.session = object()
    module.runtime.tokenizer = module.spm.SentencePieceProcessor()
    module.runtime.cmvn = {
        "tlog_mean": np.zeros(80, np.float32),
        "tlog_std": np.ones(80, np.float32),
    }
    module.runtime.pronunciation_scorer = object()
    module.runtime.ctc_aligner_ready = True


def test_ping_waits_for_every_component(app_module):
    app_module.runtime.session = object()
    app_module.runtime.tokenizer = app_module.spm.SentencePieceProcessor()
    app_module.runtime.cmvn = {
        "tlog_mean": np.zeros(80, np.float32),
        "tlog_std": np.ones(80, np.float32),
    }
    app_module.runtime.ctc_aligner_ready = True
    app_module.runtime.pronunciation_scorer = None
    assert app_module.ping().status_code == 204
    app_module.runtime.pronunciation_scorer = object()
    assert app_module.ping().status_code == 200
    health = app_module.health()
    assert health["asr_model_loaded"] is True
    assert health["pronunciation_head_loaded"] is True
    assert health["ctc_aligner_ready"] is True
    assert health["tokenizer_loaded"] is True
    assert health["cmvn_loaded"] is True


def test_diagnostic_http_route_is_disabled_without_secret(app_module, monkeypatch):
    monkeypatch.delenv("DIAGNOSTICS_API_KEY", raising=False)
    request = app_module.DiagnosticRequest(
        audio_wav_base64="AAAA",
        surah=112,
        ayah=1,
        reference_text="قل هو الله أحد",
        source="fixture",
        qari="Basfar",
    )
    try:
        asyncio.run(app_module.diagnostics_audio(request, None))
    except app_module.HTTPException as error:
        assert error.status_code == 503
    else:
        raise AssertionError("diagnostic endpoint must be disabled without a secret")


def test_dynamic_scorer_import_supports_dataclass_annotations(app_module, tmp_path):
    scorer = tmp_path / "scorer.py"
    scorer.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Score:\n"
        "    value: float\n",
        encoding="utf-8",
    )
    module = app_module.load_python_module(scorer, "temporary_scorer")
    assert module.Score(0.4).value == 0.4


def test_infer_requests_logprobs_and_encoder_output(app_module):
    class FakeSession:
        requested = None

        def run(self, outputs, inputs):
            self.requested = outputs
            return [
                np.zeros((1, 1, 1025), np.float32),
                np.zeros((1, 512, 1), np.float32),
                np.array([1], np.int64),
                np.ones((1, 17, 70, 512), np.float32),
                np.ones((1, 17, 512, 8), np.float32),
                np.ones((1,), np.int64),
            ]

    _ready_runtime(app_module)
    fake = FakeSession()
    app_module.runtime.session = fake
    state = app_module.StreamingSession()
    state._infer(np.zeros((80, 112), np.float32))
    assert "logprobs" in fake.requested
    assert "encoder_output" in fake.requested
    assert state.encoder_chunks[0].shape == (1, 512)


def test_encoder_cache_is_per_session(app_module):
    first = app_module.StreamingSession()
    second = app_module.StreamingSession()
    first.cache_last_channel[0, 0, 0, 0] = 9
    assert second.cache_last_channel[0, 0, 0, 0] == 0


def test_ctc_collapse_deduplicates_across_chunk_boundary(app_module):
    class FakeSession:
        calls = 0

        def run(self, outputs, inputs):
            token = 3
            logprobs = np.full((1, 1, 1025), -20, np.float32)
            logprobs[0, 0, token] = 0
            self.calls += 1
            return [
                logprobs,
                np.zeros((1, 512, 1), np.float32),
                np.array([1], np.int64),
                inputs["cache_last_channel"],
                inputs["cache_last_time"],
                inputs["cache_last_channel_len"],
            ]

    _ready_runtime(app_module)
    app_module.runtime.session = FakeSession()
    state = app_module.StreamingSession()
    state._infer(np.zeros((80, 112), np.float32))
    state._infer(np.zeros((80, 112), np.float32))
    assert state.token_ids == [3]


def test_forced_alignment_excludes_blank_and_has_ordered_intervals(app_module):
    _ready_runtime(app_module)
    logprobs = np.full((5, 1025), -20, np.float32)
    path = [1024, 3, 1024, 4, 1024]
    for frame, token_id in enumerate(path):
        logprobs[frame, token_id] = 0
    intervals = app_module.ctc_forced_align(logprobs, [3, 4])
    assert [item.token_id for item in intervals] == [3, 4]
    assert all(item.token_id != app_module.BLANK_ID for item in intervals)
    assert intervals[0].end_frame <= intervals[1].start_frame


def test_short_final_chunk_is_padded_but_length_is_original(app_module):
    class FakeSession:
        seen_shape = None
        seen_length = None

        def run(self, outputs, inputs):
            self.seen_shape = inputs["audio_signal"].shape
            self.seen_length = inputs["length"].tolist()
            return [
                np.zeros((1, 1, 1025), np.float32),
                np.zeros((1, 512, 1), np.float32),
                np.array([1], np.int64),
                inputs["cache_last_channel"],
                inputs["cache_last_time"],
                inputs["cache_last_channel_len"],
            ]

    _ready_runtime(app_module)
    fake = FakeSession()
    app_module.runtime.session = fake
    app_module.StreamingSession()._infer(np.zeros((80, 1), np.float32))
    assert fake.seen_shape == (1, 80, app_module.MIN_MODEL_MEL)
    assert fake.seen_length == [1]


def test_full_file_still_uses_official_112_mel_chunks(app_module, monkeypatch):
    class FakeSession:
        seen_lengths = []

        def run(self, outputs, inputs):
            length = int(inputs["length"][0])
            self.seen_lengths.append(length)
            encoded = max(1, (length - 1) // 8)
            logprobs = np.full((1, encoded, 1025), -20, np.float32)
            logprobs[:, :, 1024] = 0
            return [
                logprobs,
                np.zeros((1, 512, encoded), np.float32),
                np.array([encoded], np.int64),
                inputs["cache_last_channel"],
                inputs["cache_last_time"],
                inputs["cache_last_channel_len"],
            ]

    _ready_runtime(app_module)
    fake = FakeSession()
    app_module.runtime.session = fake
    monkeypatch.setattr(
        app_module,
        "log_mel",
        lambda audio: np.zeros((80, 250), np.float32),
    )
    state = app_module.StreamingSession()
    state.append_float(np.zeros(40_000, np.float32))
    state.process(final=True)
    assert fake.seen_lengths == [112, 112, 26]
    assert max(fake.seen_lengths) == app_module.CHUNK_MEL


def test_canonical_text_tokenization_mismatch_uses_recognized_ids(app_module):
    @dataclass
    class Score:
        phoneme: str
        token_id: int
        start_s: float
        end_s: float
        prob_correct: float
        deviation: str

    class EquivalentTokenizer(app_module.spm.SentencePieceProcessor):
        def id_to_piece(self, token_id):
            return {3: "▁ا", 4: "ب", 5: "ب"}.get(token_id, super().id_to_piece(token_id))

        def encode(self, text, out_type=int):
            return [3, 4]

        def decode(self, token_ids):
            return "اب"

    class RecordingScorer:
        seen_ids = None

        def score(self, features, intervals, output_hop_s):
            self.seen_ids = [interval.token_id for interval in intervals]
            return [
                Score(
                    phoneme=interval.token_str,
                    token_id=interval.token_id,
                    start_s=interval.start_s,
                    end_s=interval.end_s,
                    prob_correct=0.9,
                    deviation="ok",
                )
                for interval in intervals
            ]

    _ready_runtime(app_module)
    app_module.runtime.tokenizer = EquivalentTokenizer()
    scorer = RecordingScorer()
    app_module.runtime.pronunciation_scorer = scorer
    logprobs = np.full((5, 1025), -20, np.float32)
    for frame, token_id in enumerate([1024, 3, 1024, 5, 1024]):
        logprobs[frame, token_id] = 0
    bundle = app_module.InferenceBundle(
        logprobs=logprobs,
        encoder_features=np.zeros((5, 512), np.float32),
        chunk_latencies_ms=[],
        chunk_count=1,
    )
    _, _, warnings = app_module._score_tokens(bundle, "اب", "اب")
    assert scorer.seen_ids == [3, 5]
    assert any(item.startswith("TOKENIZATION_ERROR") for item in warnings)


def test_custom_mel_matches_torchaudio_reference(app_module):
    import pytest
    import torch

    torchaudio = pytest.importorskip("torchaudio")

    rng = np.random.default_rng(7)
    audio = rng.normal(0, 0.03, 16_000).astype(np.float32)
    tensor = torch.tensor(audio)
    tensor = torch.cat([tensor[:1], tensor[1:] - 0.97 * tensor[:-1]])
    reference = torch.log(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
            power=2.0,
            window_fn=torch.hann_window,
            norm="slaney",
            mel_scale="slaney",
        )(tensor)
        + 2**-24
    ).numpy()
    actual = app_module.log_mel(audio)
    assert reference.shape == actual.shape
    assert np.allclose(reference, actual, atol=2e-5, rtol=2e-5)


def test_scorer_result_is_included_in_final_analysis(app_module):
    @dataclass
    class Score:
        phoneme: str
        token_id: int
        start_s: float
        end_s: float
        prob_correct: float
        deviation: str

    class FakeScorer:
        def score(self, features, intervals, output_hop_s):
            return [
                Score(
                    phoneme=interval.token_str.replace("▁", ""),
                    token_id=interval.token_id,
                    start_s=interval.start_s,
                    end_s=interval.end_s,
                    prob_correct=0.35,
                    deviation="minor",
                )
                for interval in intervals
            ]

    _ready_runtime(app_module)
    app_module.runtime.pronunciation_scorer = FakeScorer()
    logprobs = np.full((5, 1025), -20, np.float32)
    for frame, token_id in enumerate([1024, 3, 1024, 4, 1024]):
        logprobs[frame, token_id] = 0
    bundle = app_module.InferenceBundle(
        logprobs=logprobs,
        encoder_features=np.zeros((5, 512), np.float32),
        chunk_latencies_ms=[1.0],
        chunk_count=1,
    )
    result = app_module.analyze_bundle(bundle, "اب", finalization_ms=2.0)
    assert result["tokens"]
    assert result["tokens"][0]["prob_correct"] == 0.35
    assert result["summary"]["pronunciation_minor"] == 2
    assert "THRESHOLD_DOCUMENTATION_MISMATCH" in result["summary"]["pipeline_warnings"]
