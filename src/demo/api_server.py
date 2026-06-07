"""FastAPI backend for the premium KWS web UI.

Run:  python -m src.demo.api_server
Open: http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import struct

from src.features.mel import MelSpectrogramExtractor
from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN
from src.models.edgespot_full import EdgeSpotFull
from src.streaming.enrollment import (
    EmbeddingBackend,
    EnrollmentProfile,
    build_enrollment_profile,
    crop_to_active_region,
    pad_or_trim as enrollment_pad_or_trim,
)
from src.streaming.robust_engine import RobustStreamingKWS, StreamingDecisionConfig
from src.demo.artifacts import artifact_markdown, discover_artifacts

# -- Globals --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
ENROLL_DIR = PROJECT_ROOT / "data" / "enroll_profiles"
GSC_DIR = PROJECT_ROOT / "data" / "gsc_v2"


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def first_existing(pattern: str) -> Path | None:
    matches = sorted(PROJECT_ROOT.glob(pattern))
    for match in matches:
        if match.exists():
            return match
    return None


TOP500_EPOCH13 = (
    PROJECT_ROOT / "server" / "final_kws_artifacts_package" / "checkpoints" /
    "edgespot_full_t4_scaf_ge2e_top500_full_v1" / "epoch_13.pt"
)
MICROSET_EPOCH05 = first_existing(
    "server/**/checkpoints/edgespot_full_t4_scaf_ge2e_microset_en_v1/epoch_05.pt"
) or (
    PROJECT_ROOT / "server" / "DoAnTotNghiep_output" / "checkpoints" /
    "edgespot_full_t4_scaf_ge2e_microset_en_v1" / "epoch_05.pt"
)
LEGACY_DSCNN = PROJECT_ROOT / "checkpoints" / "triplet" / "best_v2_margin1.0_colab.pt"

MODEL_PROFILES = {
    "top500_epoch13": {
        "label": "Top500 Full - EdgeSpotFull T4 + SCAF+GE2E - epoch 13",
        "short_label": "Top500 epoch13",
        "description": "Top500 full checkpoint available in this repo, useful for broad-coverage demos.",
        "description_en": "Top500 full checkpoint available in this repo, useful for broad-coverage demos.",
        "description_vi": "Checkpoint Top500 full đang có sẵn trong repo, phù hợp để demo độ phủ rộng.",
        "checkpoint": TOP500_EPOCH13,
        "model_family": "edgespot_full",
        "edge_tau": 4,
        "feature_type": "mel",
        "threshold_hint": 0.30,
        "metrics": [
            {"label": "ACC@1%FAR", "value": "86.68%"},
            {"label": "ACC@5%FAR", "value": "88.87%"},
            {"label": "F1", "value": "81.71%"},
        ],
        "notes": "Top500 full run, selected by GSC-dev among available epoch 1-13 checkpoints.",
        "notes_en": "Top500 full run, selected by GSC-dev among available epoch 1-13 checkpoints.",
        "notes_vi": "Run Top500 full, chọn theo GSC-dev trong các checkpoint epoch 1-13 hiện có.",
    },
    "microset_epoch05": {
        "label": "Microset - EdgeSpotFull T4 + SCAF+GE2E - epoch 05",
        "short_label": "Microset epoch05",
        "description": "Frozen Microset checkpoint for the thesis report and test100 comparison.",
        "description_en": "Frozen Microset checkpoint for the thesis report and test100 comparison.",
        "description_vi": "Checkpoint Microset đã khóa cho báo cáo thesis, dùng để so sánh với kết quả test100.",
        "checkpoint": MICROSET_EPOCH05,
        "model_family": "edgespot_full",
        "edge_tau": 4,
        "feature_type": "mel",
        "threshold_hint": 0.30,
        "metrics": [
            {"label": "ACC@5%FAR", "value": "86.12%"},
            {"label": "KW-ACC", "value": "77.66%"},
            {"label": "F1", "value": "82.41%"},
        ],
        "notes": "Frozen Microset thesis checkpoint evaluated on GSC test100.",
        "notes_en": "Frozen Microset thesis checkpoint evaluated on GSC test100.",
        "notes_vi": "Checkpoint Microset đã khóa, đánh giá trên GSC test100.",
    },
    "legacy_dscnn": {
        "label": "Legacy DSCNN-L Triplet",
        "short_label": "DSCNN legacy",
        "description": "Older baseline for comparing demo behavior when the checkpoint is available.",
        "description_en": "Older baseline for comparing demo behavior when the checkpoint is available.",
        "description_vi": "Baseline cũ để đối chiếu hành vi demo nếu checkpoint còn tồn tại.",
        "checkpoint": LEGACY_DSCNN,
        "model_family": "dscnn",
        "edge_tau": 4,
        "feature_type": "mfcc",
        "threshold_hint": 0.80,
        "metrics": [
            {"label": "Baseline", "value": "DSCNN-L"},
            {"label": "Feature", "value": "MFCC"},
        ],
        "notes": "Older local demo baseline, if the checkpoint is present.",
        "notes_en": "Older local demo baseline, if the checkpoint is present.",
        "notes_vi": "Baseline demo cũ trên máy local, dùng khi checkpoint còn tồn tại.",
    },
}

ENV_CKPT = os.environ.get("KWS_CHECKPOINT")
if ENV_CKPT:
    CKPT = resolve_project_path(ENV_CKPT)
    ACTIVE_MODEL_PROFILE_ID = "custom"
    MODEL_PROFILES["custom"] = {
        "label": f"Custom checkpoint - {CKPT.name}",
        "checkpoint": CKPT,
        "model_family": os.environ.get("KWS_MODEL_FAMILY", "auto"),
        "edge_tau": int(os.environ.get("KWS_EDGE_TAU", "4")),
        "feature_type": "auto",
        "threshold_hint": None,
        "notes": "Loaded from KWS_CHECKPOINT environment variable.",
    }
else:
    requested_profile = os.environ.get("KWS_MODEL_PROFILE", "top500_epoch13")
    if requested_profile not in MODEL_PROFILES or not resolve_project_path(MODEL_PROFILES[requested_profile]["checkpoint"]).exists():
        requested_profile = next(
            (pid for pid, profile in MODEL_PROFILES.items()
             if resolve_project_path(profile["checkpoint"]).exists()),
            "legacy_dscnn",
        )
    ACTIVE_MODEL_PROFILE_ID = requested_profile
    active_profile = MODEL_PROFILES[ACTIVE_MODEL_PROFILE_ID]
    CKPT = resolve_project_path(active_profile["checkpoint"])

MODEL_FAMILY = os.environ.get(
    "KWS_MODEL_FAMILY",
    MODEL_PROFILES.get(ACTIVE_MODEL_PROFILE_ID, {}).get("model_family", "auto"),
)
EDGE_TAU = int(os.environ.get(
    "KWS_EDGE_TAU",
    str(MODEL_PROFILES.get(ACTIVE_MODEL_PROFILE_ID, {}).get("edge_tau", 4)),
))
FEATURE_TYPE = "mfcc"
INPUT_SHAPE = "(1, 47, 10)"

encoder: torch.nn.Module | None = None
mfcc_ext: object | None = None
embedding_backend: EmbeddingBackend | None = None
enrollment_profile = EnrollmentProfile()
profile_version = 0
prototypes: dict[str, torch.Tensor] = {}
sample_count: dict[str, int] = {}
sample_embeddings: dict[str, list[torch.Tensor]] = {}
sample_waveforms: dict[str, list[torch.Tensor]] = {}
proto_thresholds: dict[str, float] = {}

ALPHA_THRESHOLD = 2.0
THR_FLOOR, THR_CEIL = 0.35, 1.25
MIN_ACCEPT_MARGIN = 0.05


@dataclass(frozen=True)
class DetectionPolicy:
    threshold: float
    use_per_class: bool
    close_word_guard: bool
    accept_margin: float

    def settings(self, engine: str) -> dict:
        return {
            "threshold": self.threshold,
            "use_per_class": self.use_per_class,
            "close_word_guard": self.close_word_guard,
            "accept_margin": self.accept_margin,
            "engine": engine,
            "model_profile": ACTIVE_MODEL_PROFILE_ID,
            "model_label": MODEL_PROFILES.get(ACTIVE_MODEL_PROFILE_ID, {}).get("short_label", ACTIVE_MODEL_PROFILE_ID),
        }


def coerce_form_bool(value) -> bool:
    if hasattr(value, "default"):
        value = value.default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def form_default(value, default=None):
    return value.default if hasattr(value, "default") else value


def build_detection_policy(
    threshold: float,
    use_per_class: bool,
    use_close_word_guard: bool,
    accept_margin: float | None = None,
) -> DetectionPolicy:
    accept_margin = form_default(accept_margin, None)
    close_word_guard = coerce_form_bool(use_close_word_guard)
    if close_word_guard and accept_margin is not None:
        margin = max(0.0, float(accept_margin))
    else:
        margin = MIN_ACCEPT_MARGIN if close_word_guard else 0.0
    return DetectionPolicy(
        threshold=float(threshold),
        use_per_class=coerce_form_bool(use_per_class),
        close_word_guard=close_word_guard,
        accept_margin=margin,
    )

KNOWN_GSC_WORDS = sorted([
    "yes","no","up","down","left","right","on","off","stop","go",
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "bed","bird","cat","dog","happy","house","marvin","sheila","tree","wow",
    "backward","forward","follow","learn","visual",
])

MICROSET_DEMO_WORDS = [
    "yes", "stop", "happy", "bird", "dog", "tree", "marvin", "four", "learn", "wow", "sheila", "zero",
    "down", "left", "right", "off", "one", "two", "three", "five", "six", "seven", "eight", "nine",
    "bed", "cat", "house", "backward", "forward", "follow", "visual",
]

GSC_OPEN_SET_17_KNOWN = [
    "yes", "stop", "happy", "bird", "dog", "tree", "marvin", "four", "learn",
    "wow", "sheila", "zero", "down", "left", "right", "off", "three",
]

GSC_OPEN_SET_17_UNKNOWN = [
    "no", "go", "up", "on", "one", "two", "five", "six", "seven", "eight",
    "nine", "bed", "cat", "house", "backward", "forward", "follow",
]

GSC_OPEN_SET_HELDOUT = ["visual"]
GSC_OPEN_SET_PRESET_ID = "gsc_17_17"

OPEN_SET_PRESETS = {
    GSC_OPEN_SET_PRESET_ID: {
        "id": GSC_OPEN_SET_PRESET_ID,
        "label": "GSC Open-Set 17/17",
        "known_words": GSC_OPEN_SET_17_KNOWN,
        "unknown_words": GSC_OPEN_SET_17_UNKNOWN,
        "heldout_words": GSC_OPEN_SET_HELDOUT,
    }
}

WORD_PRESETS = {
    "GSC Open-Set 17/17": ",".join(GSC_OPEN_SET_17_KNOWN),
    "Microset 31 / 50-word demo": ",".join(MICROSET_DEMO_WORDS),
    "IoT (yes/no/...)": "yes,no,stop,go,up,down,left,right,on,off",
    "Diverse phonetic": "yes,no,stop,happy,bird,dog,tree,marvin,four,learn",
    "Numbers": "zero,one,two,three,four,five,six,seven,eight,nine",
    "Names + commands": "marvin,sheila,stop,go,yes,no,happy,wow",
}


# -- Init -----------------------------------------------------
def init_model():
    global encoder, mfcc_ext, embedding_backend, MODEL_FAMILY, FEATURE_TYPE, INPUT_SHAPE
    ckpt = None
    if CKPT.exists():
        ckpt = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
        if MODEL_FAMILY == "auto":
            MODEL_FAMILY = ckpt.get("model_family", "dscnn")
    elif MODEL_FAMILY == "auto":
        MODEL_FAMILY = "dscnn"

    if MODEL_FAMILY == "edgespot_full":
        mfcc_ext = MelSpectrogramExtractor()
        encoder = EdgeSpotFull(tau=EDGE_TAU, embedding_dim=64, use_pcen=True)
        FEATURE_TYPE = "mel"
        INPUT_SHAPE = "(1, 40, 101)"
    elif MODEL_FAMILY == "dscnn":
        mfcc_ext = MFCCExtractor(n_mfcc=40, num_features=10, sample_rate=SR)
        encoder = DSCNN(model_size="L", feature_mode="NORM", input_shape=(47, 10))
        FEATURE_TYPE = "mfcc"
        INPUT_SHAPE = "(1, 47, 10)"
    else:
        raise ValueError(f"Unsupported KWS_MODEL_FAMILY={MODEL_FAMILY!r}")

    if ckpt is not None:
        encoder.load_state_dict(ckpt["model_state_dict"])
        ep = ckpt.get("epoch", "?")
        ls = ckpt.get("loss", 0)
        print(f"  Model: {CKPT.name} (family={MODEL_FAMILY}, epoch={ep}, loss={ls:.6f})")
    else:
        print(f"  WARNING: {CKPT} not found - random weights")
    encoder = encoder.to(DEVICE).eval()
    embedding_backend = EmbeddingBackend(encoder, mfcc_ext, DEVICE, SR)
    print(f"  Device: {DEVICE}, Params: {sum(p.numel() for p in encoder.parameters()):,}")


# -- Audio helpers --------------------------------------------
def bytes_to_wav(data: bytes) -> torch.Tensor | None:
    buf = io.BytesIO(data)
    try:
        wav, sr = torchaudio.load(buf)
    except Exception:
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size == 0:
            return None
        wav = torch.from_numpy(arr).unsqueeze(0)
        sr = SR
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def pad_trim(wav: torch.Tensor) -> torch.Tensor:
    return enrollment_pad_or_trim(wav, SR)


def embed(wav_1s: torch.Tensor) -> torch.Tensor:
    if embedding_backend is not None:
        return embedding_backend.embed(wav_1s)
    mfcc = mfcc_ext.extract(wav_1s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = F.normalize(encoder(mfcc), p=2, dim=-1)
    return emb.squeeze(0).cpu()


def recompute(word: str):
    if sample_waveforms.get(word) and embedding_backend is not None:
        rebuild_enrollment_profile()
        return

    embs = sample_embeddings.get(word, [])
    if not embs:
        prototypes.pop(word, None)
        proto_thresholds.pop(word, None)
        sample_count[word] = 0
        return
    stacked = torch.stack(embs)
    prototypes[word] = stacked.mean(0)
    sample_count[word] = len(embs)
    dists = [torch.dist(e, prototypes[word], p=2).item() for e in embs]
    mean_d = float(np.mean(dists))
    std_d = float(np.std(dists)) if len(dists) > 1 else 0.0
    raw = mean_d + ALPHA_THRESHOLD * max(std_d, 1e-3)
    proto_thresholds[word] = max(THR_FLOOR, min(THR_CEIL, raw))


def normalize_waveform(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.float().cpu()


def load_wav_file(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    return normalize_waveform(wav)


def quality_to_dict(quality) -> dict:
    return {
        "accepted": bool(quality.accepted),
        "reason": quality.reason,
        "duration_ms": round(float(quality.duration_ms), 1),
        "active_ms": round(float(quality.active_ms), 1),
        "rms_dbfs": round(float(quality.rms_dbfs), 1),
        "peak": round(float(quality.peak), 4),
        "snr_proxy_db": round(float(quality.snr_proxy_db), 1),
    }


def select_diverse_files(files: list[Path], k: int) -> list[Path]:
    if k <= 0 or len(files) <= k:
        return files[:max(k, 0)]
    step = len(files) / k
    return [files[min(len(files) - 1, int(i * step))] for i in range(k)]


def parse_word_list(value: str) -> list[str]:
    words = []
    seen = set()
    for raw in value.replace("\n", ",").replace("\t", ",").split(","):
        word = raw.strip().lower()
        if not word or word in seen:
            continue
        words.append(word)
        seen.add(word)
    return words


def enrolled_word_names() -> list[str]:
    words = set(prototypes)
    if profile_ready():
        words.update(enrollment_profile.keywords)
    return sorted(words)


def gsc_word_files(word: str) -> list[Path]:
    word_dir = GSC_DIR / word
    if not word_dir.exists() or not word_dir.is_dir():
        return []
    return sorted(word_dir.glob("*.wav"))


def sample_gsc_files(word: str, k: int, rng: random.Random) -> tuple[list[Path], int]:
    files = gsc_word_files(word)
    if not files:
        return [], 0
    shuffled = files[:]
    rng.shuffle(shuffled)
    return shuffled[:max(0, k)], len(files)


def resolve_open_set_split(
    preset: str,
    known_words: str,
    unknown_words: str,
) -> tuple[list[str], list[str], list[str], str]:
    preset = form_default(preset, "manual") or "manual"
    known_words = form_default(known_words, "") or ""
    unknown_words = form_default(unknown_words, "") or ""
    preset_id = (preset or "").strip().lower()
    if preset_id in OPEN_SET_PRESETS:
        spec = OPEN_SET_PRESETS[preset_id]
        return (
            list(spec["known_words"]),
            list(spec["unknown_words"]),
            list(spec["heldout_words"]),
            preset_id,
        )

    known = parse_word_list(known_words)
    if not known:
        known = enrolled_word_names()
    unknown = parse_word_list(unknown_words)
    return known, unknown, [], "manual"


def project_relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def collect_open_set_examples(
    known_words: list[str],
    unknown_words: list[str],
    samples_per_word: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(int(seed))
    enrolled_set = set(enrolled_word_names())
    known_set = set(known_words)
    skipped_unknown_words = [word for word in unknown_words if word in known_set]
    unknown_candidates = [word for word in unknown_words if word not in known_set]

    examples = []
    meta = {
        "missing_known_words": [],
        "missing_unknown_words": [],
        "skipped_unknown_words": skipped_unknown_words,
        "short_known_words": [],
        "short_unknown_words": [],
    }

    def add_examples(word: str, kind: str, expected: str) -> None:
        missing_key = "missing_known_words" if kind == "known" else "missing_unknown_words"
        short_key = "short_known_words" if kind == "known" else "short_unknown_words"
        if kind == "known" and word not in enrolled_set:
            meta[missing_key].append(word)
            return
        files, available = sample_gsc_files(word, samples_per_word, rng)
        if not files:
            meta[missing_key].append(word)
            return
        if available < samples_per_word:
            meta[short_key].append({
                "word": word,
                "available": available,
                "requested": samples_per_word,
            })
        for path in files:
            wav = pad_trim(load_wav_file(path))
            examples.append({
                "kind": kind,
                "word": word,
                "expected": expected,
                "file": path.name,
                "path": project_relative_path(path),
                "embedding": embed(wav),
            })

    for word in known_words:
        add_examples(word, "known", word)
    for word in unknown_candidates:
        add_examples(word, "unknown", "unknown")
    return examples, meta


def evaluate_open_set_examples(
    examples: list[dict],
    policy: DetectionPolicy,
    candidate_words: list[str],
) -> dict:
    results = []
    for item in examples:
        score = rounded_score_payload(
            score_embedding(
                item["embedding"],
                policy.threshold,
                policy.use_per_class,
                min_margin=policy.accept_margin,
                candidate_words=candidate_words,
            )
        )
        predicted = score["keyword"]
        correct = predicted == item["expected"]
        status = "correct" if correct else (
            "false_accept" if item["kind"] == "unknown" and predicted != "unknown"
            else "false_reject" if item["kind"] == "known" and predicted == "unknown"
            else "wrong_keyword"
        )
        result = {
            "kind": item["kind"],
            "word": item["word"],
            "expected": item["expected"],
            "predicted": predicted,
            "correct": correct,
            "status": status,
            "file": item["file"],
            "path": item["path"],
            "detected": score["detected"],
            "best_label": score["best_label"],
            "distance": score["distance"],
            "threshold": score["threshold"],
            "margin": score["margin"],
            "accept_margin": round(policy.accept_margin, 4),
            "confidence": score["confidence"],
            "second_label": score["second_label"],
            "top_3": score["top_3"],
        }
        results.append(result)

    known_results = [r for r in results if r["kind"] == "known"]
    unknown_results = [r for r in results if r["kind"] == "unknown"]
    known_correct = sum(1 for r in known_results if r["correct"])
    unknown_rejected = sum(1 for r in unknown_results if r["correct"])
    false_accepts = [r for r in unknown_results if r["predicted"] != "unknown"]
    known_misses = [r for r in known_results if not r["correct"]]
    known_false_rejects = [r for r in known_results if r["predicted"] == "unknown"]
    total = len(results)
    correct_total = known_correct + unknown_rejected

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    keyword_acc = rate(known_correct, len(known_results))
    unknown_reject_acc = rate(unknown_rejected, len(unknown_results))
    balanced = None
    if keyword_acc is not None and unknown_reject_acc is not None:
        balanced = round(0.5 * keyword_acc + 0.5 * unknown_reject_acc, 4)

    summary = {
        "known_tested": len(known_results),
        "unknown_tested": len(unknown_results),
        "total": total,
        "correct": correct_total,
        "known_correct": known_correct,
        "known_misses": len(known_misses),
        "unknown_rejected": unknown_rejected,
        "false_accepts": len(false_accepts),
        "false_rejects": len(known_false_rejects),
        "keyword_acc": keyword_acc,
        "unknown_reject_acc": unknown_reject_acc,
        "false_accept_rate": rate(len(false_accepts), len(unknown_results)),
        "false_reject_rate": rate(len(known_false_rejects), len(known_results)),
        "open_set_acc": rate(correct_total, total),
        "balanced_score": balanced,
    }
    return {
        "summary": summary,
        "results": results,
        "false_accepts": false_accepts,
        "known_misses": known_misses,
    }


def parse_float_values(value: str, default: list[float]) -> list[float]:
    values = []
    for raw in str(value or "").replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(round(max(0.0, float(raw)), 4))
        except ValueError:
            continue
    return values or default


def parse_bool_values(value: str, default: list[bool]) -> list[bool]:
    values = []
    for raw in str(value or "").replace(";", ",").split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"1", "true", "yes", "on"}:
            values.append(True)
        elif token in {"0", "false", "no", "off"}:
            values.append(False)
    deduped = []
    for item in values:
        if item not in deduped:
            deduped.append(item)
    return deduped or default


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    start = max(0.0, float(start))
    stop = max(start, float(stop))
    step = max(0.01, float(step))
    values = []
    current = start
    while current <= stop + (step / 2):
        values.append(round(current, 3))
        current += step
    return values


def calibration_rank_balanced(row: dict) -> tuple:
    return (
        row.get("balanced_score") if row.get("balanced_score") is not None else -1,
        -(row.get("false_accept_rate") or 0),
        -(row.get("false_reject_rate") or 0),
        row.get("keyword_acc") if row.get("keyword_acc") is not None else -1,
    )


def profile_ready() -> bool:
    return bool(enrollment_profile.keywords)


def sync_legacy_from_profile() -> None:
    prototypes.clear()
    sample_embeddings.clear()
    proto_thresholds.clear()
    sample_count.clear()

    for label, profile in enrollment_profile.keywords.items():
        prototypes[label] = profile.prototype.detach().cpu()
        proto_thresholds[label] = float(profile.threshold)
        sample_embeddings[label] = [
            emb.detach().cpu() for emb in profile.exemplars
        ]
        sample_count[label] = len(sample_waveforms.get(label, [])) or len(profile.qualities)


def rebuild_enrollment_profile() -> None:
    global enrollment_profile, profile_version

    if embedding_backend is None:
        return

    active_samples = {
        label: wavs for label, wavs in sample_waveforms.items() if wavs
    }
    if not active_samples:
        enrollment_profile = EnrollmentProfile()
        sync_legacy_from_profile()
        profile_version += 1
        return

    rebuilt = build_enrollment_profile(
        active_samples,
        embedding_backend,
        views_per_sample=5,
        threshold_alpha=ALPHA_THRESHOLD,
        threshold_floor=THR_FLOOR,
        threshold_ceil=THR_CEIL,
        target_far=0.01,
    )
    merged = dict(enrollment_profile.keywords)
    merged.update(rebuilt.keywords)
    enrollment_profile = EnrollmentProfile(merged)
    sync_legacy_from_profile()
    profile_version += 1


def score_embedding(
    embedding: torch.Tensor,
    threshold: float,
    use_per_class: bool,
    min_margin: float = MIN_ACCEPT_MARGIN,
    candidate_words: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict:
    if profile_ready():
        result = enrollment_profile.score(
            embedding,
            min_margin=min_margin,
            threshold_scale=1.0,
        )
        dists = result.distances
    else:
        dists = {
            word: torch.cdist(embedding.unsqueeze(0), proto.unsqueeze(0)).item()
            for word, proto in prototypes.items()
        }
    if candidate_words is not None:
        candidate_set = set(candidate_words)
        dists = {word: dist for word, dist in dists.items() if word in candidate_set}

    ordered = sorted(dists.items(), key=lambda item: item[1])
    if not ordered:
        return {
            "detected": False,
            "keyword": "unknown",
            "best_label": "unknown",
            "distance": float("inf"),
            "threshold": threshold,
            "margin": 0.0,
            "confidence": 0.0,
            "second_label": None,
            "all_distances": {},
            "top_3": [],
        }

    best_label, best_dist = ordered[0]
    second_label = ordered[1][0] if len(ordered) > 1 else None
    second_dist = ordered[1][1] if len(ordered) > 1 else best_dist + 2.0
    margin = second_dist - best_dist
    if use_per_class and profile_ready() and best_label in enrollment_profile.keywords:
        eff_thr = enrollment_profile.keywords[best_label].threshold
    elif use_per_class:
        eff_thr = proto_thresholds.get(best_label, threshold)
    else:
        eff_thr = threshold
    detected = best_dist <= eff_thr and margin >= min_margin
    dist_score = max(0.0, 1.0 - best_dist / max(eff_thr, 1e-8))
    margin_score = max(0.0, min(1.0, margin / 0.50))
    confidence = 0.75 * dist_score + 0.25 * margin_score

    return {
        "detected": detected,
        "keyword": best_label if detected else "unknown",
        "best_label": best_label,
        "distance": float(best_dist),
        "threshold": float(eff_thr),
        "margin": float(margin),
        "confidence": float(confidence),
        "second_label": second_label,
        "all_distances": {word: float(dist) for word, dist in ordered},
        "top_3": [{"word": word, "dist": float(dist)} for word, dist in ordered[:3]],
    }


def rounded_score_payload(score: dict) -> dict:
    return {
        "detected": bool(score["detected"]),
        "keyword": score["keyword"],
        "best_label": score["best_label"],
        "distance": round(score["distance"], 4),
        "threshold": round(score["threshold"], 3),
        "margin": round(score["margin"], 4),
        "confidence": round(score["confidence"], 3),
        "second_label": score["second_label"],
        "all_distances": {
            word: round(dist, 4) for word, dist in score["all_distances"].items()
        },
        "top_3": [
            {"word": item["word"], "dist": round(item["dist"], 4)}
            for item in score["top_3"]
        ],
    }


def make_stream_engine() -> RobustStreamingKWS | None:
    if embedding_backend is None or not profile_ready():
        return None
    cfg = StreamingDecisionConfig(
        sample_rate=SR,
        min_margin=MIN_ACCEPT_MARGIN,
        min_votes=2,
        cooldown_ms=900,
        chunk_process_stride_ms=250,
    )
    return RobustStreamingKWS(embedding_backend, enrollment_profile, config=cfg)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0a0e14", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# -- FastAPI app ----------------------------------------------
app = FastAPI(title="Few-Shot KWS API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

WEB_DIR = Path(__file__).parent / "web"
UI_DIST_DIR = Path(__file__).parent / "ui" / "dist"


@app.get("/")
async def index():
    react_index = UI_DIST_DIR / "index.html"
    if react_index.exists():
        return FileResponse(react_index)
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
if (UI_DIST_DIR / "assets").exists():
    app.mount("/ui/assets", StaticFiles(directory=str(UI_DIST_DIR / "assets")), name="ui-assets")


def clear_enrollment_state() -> None:
    global enrollment_profile, profile_version
    prototypes.clear()
    sample_count.clear()
    sample_embeddings.clear()
    sample_waveforms.clear()
    proto_thresholds.clear()
    enrollment_profile = EnrollmentProfile()
    profile_version += 1


def has_rebuildable_samples() -> bool:
    return any(bool(wavs) for wavs in sample_waveforms.values())


def rebuild_enrollment_for_current_model() -> dict:
    global enrollment_profile
    if not has_rebuildable_samples():
        clear_enrollment_state()
        return {"policy": "clear", "rebuilt": False, "reason": "no_waveform_samples"}

    prototypes.clear()
    sample_count.clear()
    sample_embeddings.clear()
    proto_thresholds.clear()
    enrollment_profile = EnrollmentProfile()
    rebuild_enrollment_profile()
    return {
        "policy": "rebuild",
        "rebuilt": True,
        "keywords": len(prototypes),
        "samples": sum(len(wavs) for wavs in sample_waveforms.values()),
    }


def model_profile_payload(profile_id: str, profile: dict) -> dict:
    checkpoint = resolve_project_path(profile["checkpoint"])
    return {
        "id": profile_id,
        "label": profile.get("label", profile_id),
        "short_label": profile.get("short_label", profile.get("label", profile_id)),
        "description": profile.get("description", ""),
        "description_en": profile.get("description_en", profile.get("description", "")),
        "description_vi": profile.get("description_vi", profile.get("description", "")),
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "exists": checkpoint.exists(),
        "active": profile_id == ACTIVE_MODEL_PROFILE_ID,
        "model_family": profile.get("model_family", "auto"),
        "edge_tau": profile.get("edge_tau", 4),
        "feature_type": profile.get("feature_type", "auto"),
        "threshold_hint": profile.get("threshold_hint"),
        "metrics": profile.get("metrics", []),
        "notes": profile.get("notes", ""),
        "notes_en": profile.get("notes_en", profile.get("notes", "")),
        "notes_vi": profile.get("notes_vi", profile.get("notes", "")),
    }


def current_model_info_payload() -> dict:
    active_profile = MODEL_PROFILES.get(ACTIVE_MODEL_PROFILE_ID, {})
    info = {
        "active_profile": ACTIVE_MODEL_PROFILE_ID,
        "profile_label": active_profile.get("label", ACTIVE_MODEL_PROFILE_ID),
        "profile_short_label": active_profile.get("short_label", ACTIVE_MODEL_PROFILE_ID),
        "profile_description": active_profile.get("description", ""),
        "profile_description_en": active_profile.get("description_en", active_profile.get("description", "")),
        "profile_description_vi": active_profile.get("description_vi", active_profile.get("description", "")),
        "profile_metrics": active_profile.get("metrics", []),
        "threshold_hint": active_profile.get("threshold_hint"),
        "architecture": "EdgeSpotFull T4" if MODEL_FAMILY == "edgespot_full" else "DSCNN-L",
        "parameters": sum(p.numel() for p in encoder.parameters()) if encoder is not None else 0,
        "embedding_dim": getattr(encoder, "embedding_dim", None),
        "feature_type": FEATURE_TYPE,
        "input_shape": INPUT_SHAPE,
        "device": str(DEVICE),
        "checkpoint": CKPT.name if CKPT.exists() else "none",
        "checkpoint_path": str(CKPT),
        "deployment_engine": "robust_state_machine" if profile_ready() else "legacy_window",
        "profile_version": profile_version,
        "can_rebuild_on_switch": has_rebuildable_samples(),
    }
    evals = {}
    for name, rel in [("gsc_fixed", "results/gsc_fixed_results.json"),
                      ("gsc_random", "results/gsc_random_results.json"),
                      ("kshot", "results/kshot_ablation.json")]:
        p = PROJECT_ROOT / rel
        if p.exists():
            try:
                evals[name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    info["evaluations"] = evals
    return info


# -- Enrollment endpoints -------------------------------------
@app.get("/api/presets")
async def get_presets():
    return {
        "presets": WORD_PRESETS,
        "gsc_words": KNOWN_GSC_WORDS,
        "open_set_presets": OPEN_SET_PRESETS,
    }


@app.get("/api/enroll/status")
async def enroll_status():
    items = {}
    for w in prototypes:
        qualities = []
        if w in enrollment_profile.keywords:
            qualities = [
                quality_to_dict(q) for q in enrollment_profile.keywords[w].qualities
            ]
        items[w] = {
            "count": sample_count.get(w, 0),
            "threshold": round(proto_thresholds.get(w, 0), 3),
            "profile": "robust" if w in enrollment_profile.keywords else "legacy",
            "qualities": qualities,
        }
    return {
        "enrolled": items,
        "total": len(prototypes),
        "profile_version": profile_version,
        "streaming": "robust_state_machine" if profile_ready() else "legacy_window",
        "can_rebuild_on_switch": has_rebuildable_samples(),
    }


@app.post("/api/enroll/gsc")
async def enroll_gsc(words: str = Form(...), k: int = Form(5)):
    results = []
    word_list = [w.strip().lower() for w in words.split(",") if w.strip()]
    changed_words = []
    for word in word_list:
        d = GSC_DIR / word
        if not d.exists():
            results.append({"word": word, "status": "not_found"})
            continue
        files = select_diverse_files(sorted(d.glob("*.wav")), k)
        if not files:
            results.append({"word": word, "status": "no_files"})
            continue

        wavs = []
        qualities = []
        embs = []
        for f in files:
            wav = load_wav_file(f)
            cropped, _, quality = crop_to_active_region(wav, SR)
            wavs.append(cropped)
            qualities.append(quality_to_dict(quality))
            if embedding_backend is None:
                embs.append(embed(pad_trim(cropped)))

        if embedding_backend is not None:
            sample_waveforms[word] = wavs
            changed_words.append(word)
        else:
            sample_embeddings[word] = embs
            recompute(word)

        results.append({
            "word": word, "status": "ok",
            "samples": len(files),
            "threshold": None,
            "qualities": qualities,
        })

    if changed_words:
        rebuild_enrollment_profile()

    for item in results:
        if item.get("status") == "ok":
            word = item["word"]
            item["threshold"] = round(proto_thresholds.get(word, 0), 3)

    return {"results": results, "enrolled": len(prototypes)}


@app.post("/api/enroll/mic")
async def enroll_mic(keyword: str = Form(...), audio: UploadFile = File(...)):
    keyword = keyword.strip().lower()
    if not keyword:
        return JSONResponse({"error": "No keyword name"}, 400)
    data = await audio.read()
    wav = bytes_to_wav(data)
    if wav is None:
        return JSONResponse({"error": "Invalid audio"}, 400)
    wav = normalize_waveform(wav)
    cropped, _, quality = crop_to_active_region(wav, SR)
    quality_payload = quality_to_dict(quality)
    if not quality.accepted:
        return JSONResponse({
            "error": f"Audio quality rejected: {quality.reason}",
            "quality": quality_payload,
        }, 422)

    if embedding_backend is not None:
        sample_waveforms.setdefault(keyword, []).append(cropped)
        rebuild_enrollment_profile()
    else:
        e = embed(pad_trim(cropped))
        sample_embeddings.setdefault(keyword, []).append(e)
        recompute(keyword)

    return {
        "word": keyword,
        "count": sample_count[keyword],
        "threshold": round(proto_thresholds.get(keyword, 0), 3),
        "quality": quality_payload,
    }


@app.post("/api/enroll/clear")
async def clear_all():
    clear_enrollment_state()
    return {"status": "cleared"}


@app.post("/api/enroll/save")
async def save_profile(name: str = Form("default")):
    if not prototypes:
        return JSONResponse({"error": "Nothing to save"}, 400)
    ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    path = ENROLL_DIR / f"{name}.json"
    payload = {
        "version": 2,
        "profile": enrollment_profile.to_dict() if profile_ready() else None,
        "labels": list(prototypes.keys()),
        "sample_count": dict(sample_count),
        "embeddings": {k: v.tolist() for k, v in prototypes.items()},
        "sample_embeddings": {
            k: [e.tolist() for e in lst] for k, lst in sample_embeddings.items()
        },
        "proto_thresholds": dict(proto_thresholds),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"saved": name, "keywords": len(prototypes)}


@app.post("/api/enroll/load")
async def load_profile(name: str = Form("default")):
    global enrollment_profile, profile_version
    path = ENROLL_DIR / f"{name}.json"
    if not path.exists():
        return JSONResponse({"error": f"Profile '{name}' not found"}, 404)
    payload = json.loads(path.read_text(encoding="utf-8"))
    prototypes.clear(); sample_count.clear()
    sample_embeddings.clear(); sample_waveforms.clear(); proto_thresholds.clear()

    if payload.get("profile"):
        enrollment_profile = EnrollmentProfile.from_dict(payload["profile"])
        sync_legacy_from_profile()
        saved_counts = payload.get("sample_count", {})
        for label, count in saved_counts.items():
            if label in sample_count:
                sample_count[label] = int(count)
        profile_version += 1
        return {
            "loaded": name,
            "keywords": len(prototypes),
            "profile": "robust",
        }

    enrollment_profile = EnrollmentProfile()
    saved_embs = payload.get("sample_embeddings", {})
    for label, vec in payload.get("embeddings", {}).items():
        if label in saved_embs:
            sample_embeddings[label] = [
                torch.tensor(e, dtype=torch.float32) for e in saved_embs[label]
            ]
            recompute(label)
        else:
            prototypes[label] = torch.tensor(vec, dtype=torch.float32)
            sample_count[label] = payload.get("sample_count", {}).get(label, 0)
            t = payload.get("proto_thresholds", {}).get(label)
            if t is not None:
                proto_thresholds[label] = float(t)
    profile_version += 1
    return {"loaded": name, "keywords": len(prototypes), "profile": "legacy"}


@app.get("/api/profiles")
async def list_profiles():
    if not ENROLL_DIR.exists():
        return {"profiles": []}
    return {"profiles": sorted(p.stem for p in ENROLL_DIR.glob("*.json"))}


# -- Detection endpoints --------------------------------------
@app.post("/api/detect/single")
async def detect_single(audio: UploadFile = File(...),
                        threshold: float = Form(0.6),
                        use_per_class: bool = Form(True),
                        use_close_word_guard: bool = Form(True)):
    if not prototypes:
        return JSONResponse({"error": "No keywords enrolled"}, 400)
    data = await audio.read()
    wav = bytes_to_wav(data)
    if wav is None:
        return JSONResponse({"error": "Invalid audio"}, 400)
    wav = pad_trim(wav)
    e = embed(wav)

    policy = build_detection_policy(threshold, use_per_class, use_close_word_guard)
    score = rounded_score_payload(
        score_embedding(
            e,
            policy.threshold,
            policy.use_per_class,
            min_margin=policy.accept_margin,
        )
    )

    # MFCC data for frontend rendering
    mfcc = mfcc_ext.extract(wav).squeeze(0).numpy().tolist()

    return {
        **score,
        "mfcc": mfcc,
        "settings": policy.settings("single"),
    }


@app.post("/api/detect/long")
async def detect_long(audio: UploadFile = File(...),
                      threshold: float = Form(0.7),
                      use_per_class: bool = Form(True),
                      use_close_word_guard: bool = Form(True),
                      seg_method: str = Form("Energy"),
                      min_duration_ms: int = Form(200)):
    if not prototypes:
        return JSONResponse({"error": "No keywords enrolled"}, 400)
    data = await audio.read()
    wav = bytes_to_wav(data)
    if wav is None:
        return JSONResponse({"error": "Invalid audio"}, 400)

    policy = build_detection_policy(threshold, use_per_class, use_close_word_guard)

    if embedding_backend is not None and profile_ready() and policy.use_per_class:
        cfg = StreamingDecisionConfig(
            sample_rate=SR,
            min_segment_ms=max(120, min(5000, min_duration_ms)),
            min_margin=policy.accept_margin,
            min_votes=2,
            cooldown_ms=300,
        )
        engine = RobustStreamingKWS(embedding_backend, enrollment_profile, config=cfg)
        events = engine.process_file(wav)
        results = []
        for event in events:
            start = max(0, int(event["start_sec"] * SR))
            end = min(wav.shape[-1], int(event["end_sec"] * SR))
            event_wav = wav[..., start:end] if end > start else wav
            score = rounded_score_payload(
                score_embedding(
                    embed(event_wav),
                    policy.threshold,
                    policy.use_per_class,
                    min_margin=policy.accept_margin,
                )
            )
            results.append({
                "t0": round(event["start_sec"], 2),
                "t1": round(event["end_sec"], 2),
                "speech_t0": round(event["speech_start_sec"], 2),
                "speech_t1": round(event["speech_end_sec"], 2),
                "keyword": event["keyword"],
                "best_label": score.get("best_label", event["keyword"]),
                "distance": round(event["distance"], 4),
                "threshold": round(event["threshold"], 3),
                "margin": round(event["margin"], 4),
                "accept_margin": round(policy.accept_margin, 4),
                "close_word_guard": policy.close_word_guard,
                "confidence": round(event["confidence"], 3),
                "second_label": event["second_label"],
                "detected": True,
                "top_3": score["top_3"],
            })

        return {
            "duration": round(wav.shape[-1] / SR, 1),
            "segments": len(results),
            "results": results,
            "sequence": [r["keyword"] for r in results],
            "engine": "robust_state_machine",
            "settings": policy.settings("robust_state_machine"),
        }

    total = wav.shape[-1]
    min_dur = max(80, min(5000, min_duration_ms))

    if seg_method == "Silero VAD":
        segments = _vad_segments(wav, min_dur)
        if not segments:
            segments = _energy_segments(wav, min_dur)
    else:
        segments = _energy_segments(wav, min_dur)

    results = []
    for start, end in segments:
        seg = wav[..., start:end]
        seg_1s = pad_trim(seg)
        e = embed(seg_1s)
        score = rounded_score_payload(
            score_embedding(
                e,
                policy.threshold,
                policy.use_per_class,
                min_margin=policy.accept_margin,
            )
        )
        results.append({
            "t0": round(start / SR, 2),
            "t1": round(end / SR, 2),
            "keyword": score["keyword"],
            "best_label": score.get("best_label", score["keyword"]),
            "distance": score["distance"],
            "threshold": score["threshold"],
            "margin": score["margin"],
            "accept_margin": round(policy.accept_margin, 4),
            "close_word_guard": policy.close_word_guard,
            "confidence": score["confidence"],
            "second_label": score["second_label"],
            "detected": score["detected"],
            "top_3": score["top_3"],
        })

    preds = [r["keyword"] for r in results if r["detected"]]
    return {
        "duration": round(total / SR, 1),
        "segments": len(results),
        "results": results,
        "sequence": preds,
        "engine": "legacy_segments",
        "settings": policy.settings("legacy_segments"),
    }


def _energy_segments(wav, min_dur_ms):
    mono = wav.mean(dim=0) if wav.dim() == 2 else wav.squeeze(0)
    total = mono.shape[-1]
    frame = int(SR * 0.03)
    hop = int(SR * 0.01)
    if total < frame:
        return [(0, total)] if total > 0 else []
    starts = list(range(0, total - frame + 1, hop))
    energies = [float(torch.sqrt(torch.mean(mono[s:s+frame]**2)).item()) for s in starts]
    mx = max(energies) if energies else 0
    if mx <= 1e-6:
        return []
    thr = max(0.0005, mx * 0.08)
    active, cur = [], None
    for s, e in zip(starts, energies):
        if e >= thr and cur is None:
            cur = s
        elif e < thr and cur is not None:
            active.append((cur, s + frame))
            cur = None
    if cur is not None:
        active.append((cur, starts[-1] + frame))
    gap = int(SR * 0.35)
    merged = []
    for s, e in active:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    pad = int(SR * 0.2)
    minl = int(SR * min_dur_ms / 1000)
    return [(max(0, s-pad), min(total, e+pad))
            for s, e in merged if min(total, e+pad) - max(0, s-pad) >= minl]


def _vad_segments(wav, min_dur_ms):
    try:
        from src.streaming.vad_engine import SileroVAD
        vad = SileroVAD(threshold=0.5, min_speech_ms=min(min_dur_ms, 250), device=DEVICE)
        ts = vad.get_speech_timestamps(wav.squeeze(0) if wav.dim() == 2 else wav)
        minl = int(SR * min_dur_ms / 1000) // 2
        return [(int(t["start"]), int(t["end"])) for t in ts
                if int(t["end"]) - int(t["start"]) >= minl]
    except Exception:
        return []


# -- Open-set sampled evaluation ------------------------------
@app.post("/api/open-set/test")
async def open_set_test(
    unknown_words: str = Form("cat,bed,house,wow,sheila"),
    known_words: str = Form(""),
    preset: str = Form("manual"),
    samples_per_word: int = Form(5),
    threshold: float = Form(0.6),
    use_per_class: bool = Form(True),
    use_close_word_guard: bool = Form(True),
    accept_margin: float | None = Form(None),
    seed: int = Form(1234),
):
    if not prototypes:
        return JSONResponse({"error": "No keywords enrolled"}, 400)

    samples_per_word = max(1, min(100, int(samples_per_word)))
    split_known, split_unknown, heldout_words, preset_id = resolve_open_set_split(
        preset, known_words, unknown_words
    )
    candidate_words = [
        word for word in split_known if word in set(enrolled_word_names())
    ]
    policy = build_detection_policy(
        threshold,
        use_per_class,
        use_close_word_guard,
        accept_margin=accept_margin,
    )
    examples, meta = collect_open_set_examples(
        split_known,
        split_unknown,
        samples_per_word,
        int(seed),
    )
    evaluated = evaluate_open_set_examples(examples, policy, candidate_words)
    summary = evaluated["summary"]

    if summary["known_tested"] == 0:
        return JSONResponse({
            "error": "No enrolled keywords with GSC audio were found",
            "known_words": split_known,
            "unknown_words": split_unknown,
            "heldout_words": heldout_words,
            "candidate_words": candidate_words,
            **meta,
        }, 400)
    if summary["unknown_tested"] == 0:
        return JSONResponse({
            "error": "No unknown GSC audio was found",
            "known_words": split_known,
            "unknown_words": split_unknown,
            "heldout_words": heldout_words,
            "candidate_words": candidate_words,
            **meta,
        }, 400)

    return {
        "settings": policy.settings("open_set_gsc_sampled"),
        "preset": preset_id,
        "known_words": split_known,
        "unknown_words": split_unknown,
        "heldout_words": heldout_words,
        "candidate_words": candidate_words,
        "summary": summary,
        "results": evaluated["results"],
        "false_accepts": evaluated["false_accepts"],
        "known_misses": evaluated["known_misses"],
        **meta,
    }


@app.post("/api/open-set/calibrate")
async def open_set_calibrate(
    unknown_words: str = Form(""),
    known_words: str = Form(""),
    preset: str = Form(GSC_OPEN_SET_PRESET_ID),
    samples_per_word: int = Form(5),
    seed: int = Form(1234),
    threshold_min: float = Form(0.10),
    threshold_max: float = Form(1.20),
    threshold_step: float = Form(0.05),
    accept_margin_values: str = Form("0.00,0.02,0.05,0.08,0.10"),
    use_per_class_options: str = Form("true,false"),
):
    if not prototypes:
        return JSONResponse({"error": "No keywords enrolled"}, 400)

    samples_per_word = max(1, min(100, int(samples_per_word)))
    split_known, split_unknown, heldout_words, preset_id = resolve_open_set_split(
        preset, known_words, unknown_words
    )
    candidate_words = [
        word for word in split_known if word in set(enrolled_word_names())
    ]
    examples, meta = collect_open_set_examples(
        split_known,
        split_unknown,
        samples_per_word,
        int(seed),
    )
    base_policy = build_detection_policy(0.3, True, False)
    base_eval = evaluate_open_set_examples(examples, base_policy, candidate_words)
    if base_eval["summary"]["known_tested"] == 0:
        return JSONResponse({
            "error": "No enrolled keywords with GSC audio were found",
            "known_words": split_known,
            "unknown_words": split_unknown,
            "heldout_words": heldout_words,
            "candidate_words": candidate_words,
            **meta,
        }, 400)
    if base_eval["summary"]["unknown_tested"] == 0:
        return JSONResponse({
            "error": "No unknown GSC audio was found",
            "known_words": split_known,
            "unknown_words": split_unknown,
            "heldout_words": heldout_words,
            "candidate_words": candidate_words,
            **meta,
        }, 400)

    thresholds = threshold_grid(threshold_min, threshold_max, threshold_step)
    margins = parse_float_values(accept_margin_values, [0.0, 0.02, 0.05, 0.08, 0.10])
    per_class_values = parse_bool_values(use_per_class_options, [True, False])
    rows = []
    for thr in thresholds:
        for margin in margins:
            for use_per_class in per_class_values:
                policy = DetectionPolicy(
                    threshold=float(thr),
                    use_per_class=bool(use_per_class),
                    close_word_guard=margin > 0,
                    accept_margin=float(margin),
                )
                evaluated = evaluate_open_set_examples(examples, policy, candidate_words)
                row = {
                    "threshold": round(policy.threshold, 3),
                    "use_per_class": policy.use_per_class,
                    "close_word_guard": policy.close_word_guard,
                    "accept_margin": round(policy.accept_margin, 4),
                    **evaluated["summary"],
                }
                rows.append(row)

    rows.sort(key=calibration_rank_balanced, reverse=True)
    best_balanced = rows[0]
    best_open_set = max(rows, key=lambda row: (
        row.get("unknown_reject_acc") if row.get("unknown_reject_acc") is not None else -1,
        row.get("keyword_acc") if row.get("keyword_acc") is not None else -1,
        -(row.get("false_accept_rate") or 0),
        -(row.get("false_reject_rate") or 0),
    ))
    best_keyword = max(rows, key=lambda row: (
        row.get("keyword_acc") if row.get("keyword_acc") is not None else -1,
        row.get("unknown_reject_acc") if row.get("unknown_reject_acc") is not None else -1,
        -(row.get("false_reject_rate") or 0),
        -(row.get("false_accept_rate") or 0),
    ))

    return {
        "settings": {
            "engine": "open_set_gsc_calibration",
            "threshold_min": float(threshold_min),
            "threshold_max": float(threshold_max),
            "threshold_step": float(threshold_step),
            "accept_margin_values": margins,
            "use_per_class_options": per_class_values,
        },
        "preset": preset_id,
        "known_words": split_known,
        "unknown_words": split_unknown,
        "heldout_words": heldout_words,
        "candidate_words": candidate_words,
        "best_balanced": best_balanced,
        "best_open_set": best_open_set,
        "best_keyword": best_keyword,
        "rows": rows,
        **meta,
    }


# -- Batch evaluation -----------------------------------------
@app.post("/api/detect/batch")
async def detect_batch(
    labels_file: UploadFile = File(...),
    threshold: float = Form(0.6),
    use_per_class: bool = Form(True),
    use_close_word_guard: bool = Form(True),
):
    """Batch evaluation: upload a TXT with lines `filename,expected_keyword`
    and corresponding audio files in enrolled GSC data or provide them.
    The system detects each and compares vs ground truth."""
    if not prototypes:
        return JSONResponse({"error": "No keywords enrolled"}, 400)

    policy = build_detection_policy(threshold, use_per_class, use_close_word_guard)
    txt = (await labels_file.read()).decode("utf-8", errors="replace")
    lines = [l.strip() for l in txt.splitlines() if l.strip() and not l.startswith("#")]

    results = []
    correct = 0
    total = 0

    for line in lines:
        # Parse: filename,expected  OR  filename\texpected
        parts = line.replace("\t", ",").split(",")
        if len(parts) < 2:
            continue
        fname = parts[0].strip()
        expected = parts[1].strip().lower()

        # Try to find the audio file in GSC directory
        audio_path = None
        for candidate in [
            GSC_DIR / expected / fname,
            GSC_DIR / expected / (fname + ".wav"),
            GSC_DIR / fname,
            PROJECT_ROOT / fname,
        ]:
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            results.append({
                "file": fname, "expected": expected,
                "predicted": "-", "distance": 0,
                "status": "file_not_found", "correct": False,
            })
            total += 1
            continue

        try:
            w_t, sr = torchaudio.load(str(audio_path))
            if sr != SR:
                w_t = torchaudio.transforms.Resample(sr, SR)(w_t)
            w_t = pad_trim(w_t)
            e = embed(w_t)

            score = rounded_score_payload(
                score_embedding(
                    e,
                    policy.threshold,
                    policy.use_per_class,
                    min_margin=policy.accept_margin,
                )
            )
            predicted = score["keyword"]
            is_correct = predicted == expected

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "file": fname, "expected": expected,
                "predicted": predicted, "distance": score["distance"],
                "threshold": score["threshold"],
                "margin": score["margin"],
                "confidence": score["confidence"],
                "status": "ok", "correct": is_correct,
            })
        except Exception as ex:
            total += 1
            results.append({
                "file": fname, "expected": expected,
                "predicted": "-", "distance": 0,
                "status": f"error: {ex}", "correct": False,
            })

    accuracy = (correct / total * 100) if total > 0 else 0
    return {
        "settings": policy.settings("legacy_batch"),
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "results": results,
    }


# -- Model info -----------------------------------------------
@app.get("/api/model/profiles")
async def model_profiles():
    return {
        "active": ACTIVE_MODEL_PROFILE_ID,
        "can_rebuild_on_switch": has_rebuildable_samples(),
        "profiles": [
            model_profile_payload(profile_id, profile)
            for profile_id, profile in MODEL_PROFILES.items()
        ],
    }


@app.post("/api/model/select")
async def select_model_profile(
    profile_id: str = Form(...),
    enrollment_policy: str = Form("clear"),
):
    global CKPT, MODEL_FAMILY, EDGE_TAU, ACTIVE_MODEL_PROFILE_ID

    if profile_id not in MODEL_PROFILES:
        return JSONResponse({"error": f"Unknown model profile: {profile_id}"}, 404)
    if enrollment_policy not in {"clear", "rebuild"}:
        return JSONResponse({"error": f"Unsupported enrollment_policy: {enrollment_policy}"}, 400)

    profile = MODEL_PROFILES[profile_id]
    checkpoint = resolve_project_path(profile["checkpoint"])
    if not checkpoint.exists():
        return JSONResponse({"error": f"Checkpoint not found: {checkpoint}"}, 404)

    CKPT = checkpoint
    MODEL_FAMILY = profile.get("model_family", "auto")
    EDGE_TAU = int(profile.get("edge_tau", 4))
    ACTIVE_MODEL_PROFILE_ID = profile_id
    init_model()
    if enrollment_policy == "rebuild":
        enrollment_status = rebuild_enrollment_for_current_model()
    else:
        clear_enrollment_state()
        enrollment_status = {"policy": "clear", "rebuilt": False}
    return {
        "status": "ok",
        "active": ACTIVE_MODEL_PROFILE_ID,
        "enrollment": enrollment_status,
        "model": current_model_info_payload(),
    }


@app.get("/api/model/info")
async def model_info():
    return current_model_info_payload()


# -- Artifacts and report export ------------------------------
@app.get("/api/artifacts/status")
async def artifacts_status():
    return discover_artifacts(PROJECT_ROOT)


@app.post("/api/export/session-report")
async def export_session_report(title: str = Form("Few-Shot KWS Demo Session")):
    artifact_status = discover_artifacts(PROJECT_ROOT)
    model = current_model_info_payload()
    enrollment = await enroll_status()
    lines = [
        f"# {title}",
        "",
        "## Model",
        "",
        f"- Active profile: `{model.get('active_profile')}`",
        f"- Label: {model.get('profile_label')}",
        f"- Checkpoint: `{model.get('checkpoint_path')}`",
        f"- Feature type: `{model.get('feature_type')}`",
        f"- Device: `{model.get('device')}`",
        "",
        "## Enrollment",
        "",
        f"- Keyword count: {enrollment.get('total', 0)}",
        f"- Streaming engine: `{enrollment.get('streaming')}`",
        "",
    ]
    for word, item in enrollment.get("enrolled", {}).items():
        lines.append(f"- `{word}`: {item.get('count', 0)} samples, threshold={item.get('threshold')}")
    lines.extend([
        "",
        "## Artifact Status",
        "",
        artifact_markdown(artifact_status, lang="en"),
    ])
    return {
        "format": "markdown",
        "markdown": "\n".join(lines),
        "artifacts": artifact_status,
        "model": model,
        "enrollment": enrollment,
    }


# -- Streaming WebSocket --------------------------------------
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    stream_engine = make_stream_engine()
    stream_profile_version = profile_version

    # Legacy fallback: 1-second windows with 0.5-second stride.
    buffer = torch.zeros(0)
    window_size = SR  # 16000 samples = 1 second
    stride = SR // 2  # 8000 samples = 0.5 second
    cooldown = 0  # Prevent duplicate detections

    try:
        while True:
            data = await ws.receive_bytes()
            # Browser sends Float32 PCM
            n_samples = len(data) // 4
            chunk = torch.tensor(
                struct.unpack(f"{n_samples}f", data), dtype=torch.float32
            )

            if stream_profile_version != profile_version:
                stream_engine = make_stream_engine()
                stream_profile_version = profile_version

            if stream_engine is not None:
                events = stream_engine.process_chunk(chunk.unsqueeze(0))
                for event in events:
                    start = max(0, int(event["start_sec"] * SR))
                    end = max(start + 1, int(event["end_sec"] * SR))
                    top_3 = [{
                        "word": event["keyword"],
                        "dist": round(event["distance"], 4),
                    }]
                    if event["second_label"]:
                        top_3.append({
                            "word": event["second_label"],
                            "dist": round(event["distance"] + event["margin"], 4),
                        })
                    await ws.send_json({
                        "detected": True,
                        "keyword": event["keyword"],
                        "state": event.get("state", "detected"),
                        "distance": round(event["distance"], 4),
                        "threshold": round(event["threshold"], 3),
                        "margin": round(event["margin"], 4),
                        "confidence": round(event["confidence"], 3),
                        "second_label": event["second_label"],
                        "start_sec": round(start / SR, 2),
                        "end_sec": round(end / SR, 2),
                        "start_ms": event.get("start_ms", round(start / SR * 1000)),
                        "end_ms": event.get("end_ms", round(end / SR * 1000)),
                        "timestamp": event.get("timestamp", time.time()),
                        "top_3": top_3,
                        "engine": "robust_state_machine",
                    })
                continue

            buffer = torch.cat([buffer, chunk])

            # Process when we have enough samples
            while buffer.shape[0] >= window_size:
                window = buffer[:window_size].unsqueeze(0)
                buffer = buffer[stride:]  # Slide by stride

                if cooldown > 0:
                    cooldown -= 1
                    continue

                if not prototypes:
                    continue

                score = rounded_score_payload(score_embedding(embed(window), 0.7, True))

                result = {
                    **score,
                    "state": "detected" if score["detected"] else "rejected",
                    "start_ms": None,
                    "end_ms": None,
                    "timestamp": time.time(),
                    "engine": "legacy_window",
                }
                await ws.send_json(result)

                if score["detected"]:
                    cooldown = 2  # Skip 2 windows (~1s) after detection

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")


# -- Main -----------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Few-Shot KWS - Premium Web UI")
    print("=" * 60)
    init_model()
    print("\n  Starting server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
