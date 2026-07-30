from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class FakeSentencePieceProcessor:
    def __init__(self, *args, **kwargs):
        self.pieces = ["<unk>", "<s>", "</s>", "▁ا", "ب"] + [
            f"x{index}" for index in range(5, 1024)
        ]

    def get_piece_size(self):
        return 1024

    def id_to_piece(self, token_id):
        return self.pieces[token_id]

    def decode(self, token_ids):
        return "".join(self.pieces[token_id].replace("▁", " ") for token_id in token_ids).strip()

    def encode(self, text, out_type=int):
        return [3, 4]

    def is_unknown(self, token_id):
        return token_id == 0

    def is_control(self, token_id):
        return token_id in {1, 2}

    def is_unused(self, token_id):
        return False


fake_sentencepiece = ModuleType("sentencepiece")
fake_sentencepiece.SentencePieceProcessor = FakeSentencePieceProcessor
sys.modules.setdefault("sentencepiece", fake_sentencepiece)


@pytest.fixture
def app_module():
    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("quran_app_test", project / "app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
