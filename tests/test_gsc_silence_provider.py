from pathlib import Path

import torch
import torchaudio

from src.evaluation.gsc import GSCFewShotProvider
from src.evaluation.protocols import EDGESPOT_TARGET_WORDS, EvaluationProtocol, GSC_POSITIVE_WORDS


def _write_wav(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = torch.full((1, 16000), value)
    torchaudio.save(str(path), wav, 16000)


def test_edgespot_exact_partition_uses_true_silence():
    proto = EvaluationProtocol(dataset="gsc", mode="edgespot_exact")
    pos, neg = proto.get_partitions(0)
    assert "_silence_" in pos
    assert "marvin" not in pos
    assert len(pos) == 11
    assert len(neg) == 25
    assert not (set(neg) & set(GSC_POSITIVE_WORDS))
    assert set(pos) == set(EDGESPOT_TARGET_WORDS)


def test_gsc_provider_silence_from_background_noise(tmp_path):
    gsc = tmp_path / "gsc"
    lines_val = []
    lines_test = []
    for word in ("yes", "no"):
        for i in range(6):
            rel = f"{word}/val_{i}.wav"
            _write_wav(gsc / rel, 0.01)
            lines_val.append(rel)
        rel_test = f"{word}/test_0.wav"
        _write_wav(gsc / rel_test, 0.02)
        lines_test.append(rel_test)
        _write_wav(gsc / word / "train_0.wav", 0.03)

    _write_wav(gsc / "_background_noise_" / "noise.wav", 0.0)
    (gsc / "validation_list.txt").write_text("\n".join(lines_val), encoding="utf-8")
    (gsc / "testing_list.txt").write_text("\n".join(lines_test), encoding="utf-8")

    provider = GSCFewShotProvider(gsc, feature_type="mel")
    provider.validate_words(["yes", "_silence_"], min_support=5)
    silence, names = provider.get_support_samples("_silence_", 3, seed=123)
    assert silence.shape == (3, 1, 40, 101)
    assert all(name.startswith("_silence_/") for name in names)
