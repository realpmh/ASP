#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags
from providers import build_transcript_provider
from scoring import build_scorer

ROOT = Path(__file__).resolve().parents[1]


def load_audio(path: Path):
    x, sr = sf.read(path, always_2d=True)
    x = x.astype(np.float32)
    if sr != 16000:
        x = librosa.resample(x.T, orig_sr=sr, target_sr=16000).T
        sr = 16000
    return x, sr


def energy_segments(x_mono, sr):
    frame = int(0.030 * sr)
    hop = int(0.010 * sr)
    rms = librosa.feature.rms(y=x_mono, frame_length=max(512, frame), hop_length=hop, center=True)[0]
    floor = float(np.percentile(rms, 15))
    hi = float(np.percentile(rms, 90))
    threshold = max(floor * 5.0, floor + 0.16 * (hi - floor), 0.006)
    active = rms > threshold

    gap_frames = int(0.32 / (hop / sr))
    active2 = active.copy()
    idx = np.where(active)[0]
    if len(idx):
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= gap_frames:
                active2[a:b+1] = True

    ranges = []
    start = None
    for i, v in enumerate(active2):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(active2) - 1):
            end = i if not v else i + 1
            s = max(0, int(start * hop - 0.10 * sr))
            e = min(len(x_mono), int(end * hop + 0.15 * sr))
            if (e - s) / sr >= 0.75:
                ranges.append([s, e])
            start = None

    merged = []
    for s, e in ranges:
        if merged and (s - merged[-1][1]) / sr < 0.36:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged, threshold


def voice_feature(mono, sr):
    if len(mono) < sr // 2:
        mono = np.pad(mono, (0, sr // 2 - len(mono)))
    mono = librosa.util.normalize(mono)
    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=20, n_fft=512, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    feat = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
    ]).astype(np.float32)
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    f0 = librosa.yin(mono, fmin=60, fmax=350, sr=sr)
    f0 = f0[np.isfinite(f0)]
    pitch = float(np.median(f0)) if len(f0) else 120.0
    return {"embedding": feat, "pitch": pitch}


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def voice_similarity(a, b):
    mfcc_sim = (cosine(a["embedding"], b["embedding"]) + 1.0) / 2.0
    ratio = max(a["pitch"], b["pitch"]) / max(1e-6, min(a["pitch"], b["pitch"]))
    pitch_sim = math.exp(-5.0 * abs(math.log(ratio)))
    return 0.42 * mfcc_sim + 0.58 * pitch_sim


def build_profiles(manifest):
    profiles = {}
    for st in manifest["students"]:
        wav = ROOT / "data" / "enroll" / f"{st['id']}.wav"
        x, sr = load_audio(wav)
        profiles[st["id"]] = voice_feature(x.mean(axis=1), sr)
    return profiles


def estimate_delay(left, right, sr):
    n = min(len(left), len(right))
    if n < 256:
        return 0, 0.0
    l = left[:n] - float(np.mean(left[:n]))
    r = right[:n] - float(np.mean(right[:n]))
    max_lag = int(0.0010 * sr)
    c = correlate(l, r, mode="full", method="fft")
    lags = correlation_lags(len(l), len(r), mode="full")
    mask = np.abs(lags) <= max_lag
    c2 = c[mask]
    l2 = lags[mask]
    lag = int(l2[int(np.argmax(c2))])
    peak = float(np.max(c2) / (np.sqrt(np.sum(l*l) * np.sum(r*r)) + 1e-9))
    return lag, peak


def location_from_lag(lag):
    if lag >= 3:
        return "左侧"
    if lag <= -3:
        return "右侧"
    return "中间"


def analyze(audio_path: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    x, sr = load_audio(audio_path)
    mono = x.mean(axis=1)
    segs, threshold = energy_segments(mono, sr)
    profiles = build_profiles(manifest)
    by_id = {s["id"]: s for s in manifest["students"]}
    transcript_provider = build_transcript_provider()
    transcript_segments = transcript_provider.transcribe(audio_path, manifest)
    scorer = build_scorer()

    results = []
    for i, (s, e) in enumerate(segs):
        chunk = x[s:e]
        feat = voice_feature(chunk.mean(axis=1), sr)
        sims = {sid: voice_similarity(feat, p) for sid, p in profiles.items()}
        sid = max(sims, key=sims.get)
        st = by_id[sid]
        lag, corr = estimate_delay(chunk[:, 0], chunk[:, 1], sr) if chunk.shape[1] >= 2 else (0, 0.0)
        loc = location_from_lag(lag)
        text = transcript_segments[i].text if i < len(transcript_segments) else ""
        sc = scorer.score(text, manifest["rubric"])
        results.append({
            "index": i + 1,
            "start": round(s / sr, 2),
            "end": round(e / sr, 2),
            "duration": round((e - s) / sr, 2),
            "speaker_id": sid,
            "speaker": st["name"],
            "speaker_confidence": round(sims[sid], 3),
            "voice_similarity": {k: round(v, 3) for k, v in sims.items()},
            "location": loc,
            "delay_samples": lag,
            "delay_ms": round(lag / sr * 1000, 3),
            "location_correlation": round(corr, 3),
            "text": text,
            "score": sc,
        })

    summary = {}
    total_dur = sum(r["duration"] for r in results) or 1.0
    for st in manifest["students"]:
        rs = [r for r in results if r["speaker_id"] == st["id"]]
        seconds = round(sum(r["duration"] for r in rs), 2)
        summary[st["id"]] = {
            "name": st["name"],
            "seat": st["seat"],
            "turns": len(rs),
            "seconds": seconds,
            "share": round(seconds / total_dur * 100, 1),
            "avg_score": round(sum(r["score"]["total"] for r in rs) / len(rs), 1) if rs else 0.0,
        }

    overall = round(sum(r["score"]["total"] for r in results) / len(results), 1) if results else 0.0
    return {
        "title": manifest["title"],
        "audio": str(audio_path),
        "sample_rate": sr,
        "channels": int(x.shape[1]),
        "detected_turns": len(results),
        "vad_threshold": round(float(threshold), 5),
        "overall_score": overall,
        "segments": results,
        "students": list(summary.values()),
        "rubric": manifest["rubric"],
        "runtime": {
            "speaker": "MFCC voiceprint enrollment",
            "location": "bounded inter-channel cross-correlation",
            "transcription": transcript_provider.name,
            "scoring": scorer.name,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("audio")
    a.add_argument("manifest")
    args = ap.parse_args()
    if args.cmd == "analyze":
        out = analyze(Path(args.audio), Path(args.manifest))
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
