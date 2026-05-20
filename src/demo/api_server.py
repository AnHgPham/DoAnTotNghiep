"""FastAPI backend for the premium KWS web UI.

Run:  python -m src.demo.api_server
Open: http://127.0.0.1:8000
"""

import base64
import io
import json
import sys
import time
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

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN
from src.streaming.enrollment import (
    EmbeddingBackend,
    EnrollmentProfile,
    build_enrollment_profile,
    crop_to_active_region,
    pad_or_trim as enrollment_pad_or_trim,
)
from src.streaming.robust_engine import RobustStreamingKWS, StreamingDecisionConfig

# -- Globals --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
CKPT = PROJECT_ROOT / "checkpoints" / "triplet" / "best_v2_margin1.0_colab.pt"
ENROLL_DIR = PROJECT_ROOT / "data" / "enroll_profiles"
GSC_DIR = PROJECT_ROOT / "data" / "gsc_v2"

encoder: DSCNN | None = None
mfcc_ext: MFCCExtractor | None = None
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

KNOWN_GSC_WORDS = sorted([
    "yes","no","up","down","left","right","on","off","stop","go",
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "bed","bird","cat","dog","happy","house","marvin","sheila","tree","wow",
    "backward","forward","follow","learn","visual",
])

WORD_PRESETS = {
    "IoT (yes/no/...)": "yes,no,stop,go,up,down,left,right,on,off",
    "Diverse phonetic": "yes,no,stop,happy,bird,dog,tree,marvin,four,learn",
    "Numbers": "zero,one,two,three,four,five,six,seven,eight,nine",
    "Names + commands": "marvin,sheila,stop,go,yes,no,happy,wow",
}


# -- Init -----------------------------------------------------
def init_model():
    global encoder, mfcc_ext, embedding_backend
    mfcc_ext = MFCCExtractor(n_mfcc=40, num_features=10, sample_rate=SR)
    encoder = DSCNN(model_size="L", feature_mode="NORM", input_shape=(47, 10))
    if CKPT.exists():
        ckpt = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(ckpt["model_state_dict"])
        ep = ckpt.get("epoch", "?")
        ls = ckpt.get("loss", 0)
        print(f"  Model: {CKPT.name} (epoch={ep}, loss={ls:.6f})")
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


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# -- Enrollment endpoints -------------------------------------
@app.get("/api/presets")
async def get_presets():
    return {"presets": WORD_PRESETS, "gsc_words": KNOWN_GSC_WORDS}


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
    global enrollment_profile, profile_version
    prototypes.clear()
    sample_count.clear()
    sample_embeddings.clear()
    sample_waveforms.clear()
    proto_thresholds.clear()
    enrollment_profile = EnrollmentProfile()
    profile_version += 1
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
                        use_per_class: bool = Form(True)):
    if not prototypes:
        return JSONResponse({"error": "No keywords enrolled"}, 400)
    data = await audio.read()
    wav = bytes_to_wav(data)
    if wav is None:
        return JSONResponse({"error": "Invalid audio"}, 400)
    wav = pad_trim(wav)
    e = embed(wav)

    score = rounded_score_payload(score_embedding(e, threshold, use_per_class))

    # MFCC data for frontend rendering
    mfcc = mfcc_ext.extract(wav).squeeze(0).numpy().tolist()

    return {
        **score,
        "mfcc": mfcc,
    }


@app.post("/api/detect/long")
async def detect_long(audio: UploadFile = File(...),
                      threshold: float = Form(0.7),
                      use_per_class: bool = Form(True),
                      seg_method: str = Form("Energy"),
                      min_duration_ms: int = Form(200)):
    if not prototypes:
        return JSONResponse({"error": "No keywords enrolled"}, 400)
    data = await audio.read()
    wav = bytes_to_wav(data)
    if wav is None:
        return JSONResponse({"error": "Invalid audio"}, 400)

    if embedding_backend is not None and profile_ready() and use_per_class:
        cfg = StreamingDecisionConfig(
            sample_rate=SR,
            min_segment_ms=max(120, min(5000, min_duration_ms)),
            min_margin=MIN_ACCEPT_MARGIN,
            min_votes=2,
            cooldown_ms=900,
        )
        engine = RobustStreamingKWS(embedding_backend, enrollment_profile, config=cfg)
        events = engine.process_file(wav)
        results = []
        for event in events:
            start = max(0, int(event["start_sec"] * SR))
            end = min(wav.shape[-1], int(event["end_sec"] * SR))
            event_wav = wav[..., start:end] if end > start else wav
            score = rounded_score_payload(
                score_embedding(embed(event_wav), threshold, True)
            )
            results.append({
                "t0": round(event["start_sec"], 2),
                "t1": round(event["end_sec"], 2),
                "speech_t0": round(event["speech_start_sec"], 2),
                "speech_t1": round(event["speech_end_sec"], 2),
                "keyword": event["keyword"],
                "distance": round(event["distance"], 4),
                "threshold": round(event["threshold"], 3),
                "margin": round(event["margin"], 4),
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
        score = rounded_score_payload(score_embedding(e, threshold, use_per_class))
        results.append({
            "t0": round(start / SR, 2),
            "t1": round(end / SR, 2),
            "keyword": score["keyword"],
            "distance": score["distance"],
            "threshold": score["threshold"],
            "margin": score["margin"],
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


# -- Batch evaluation -----------------------------------------
@app.post("/api/detect/batch")
async def detect_batch(
    labels_file: UploadFile = File(...),
    threshold: float = Form(0.6),
    use_per_class: bool = Form(True),
):
    """Batch evaluation: upload a TXT with lines `filename,expected_keyword`
    and corresponding audio files in enrolled GSC data or provide them.
    The system detects each and compares vs ground truth."""
    if not prototypes:
        return JSONResponse({"error": "No keywords enrolled"}, 400)

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

            score = rounded_score_payload(score_embedding(e, threshold, use_per_class))
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
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "results": results,
    }


# -- Model info -----------------------------------------------
@app.get("/api/model/info")
async def model_info():
    info = {
        "architecture": "DSCNN-L",
        "parameters": sum(p.numel() for p in encoder.parameters()),
        "embedding_dim": encoder.embedding_dim,
        "input_shape": "(1, 47, 10)",
        "device": str(DEVICE),
        "checkpoint": CKPT.name if CKPT.exists() else "none",
        "deployment_engine": "robust_state_machine" if profile_ready() else "legacy_window",
        "profile_version": profile_version,
    }
    # Load eval results
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
