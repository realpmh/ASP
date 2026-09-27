from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class TranscriptProvider:
    name = "base"

    def transcribe(self, audio: Path, manifest: dict[str, Any] | None = None) -> list[TranscriptSegment]:
        raise NotImplementedError


class ManifestTranscriptProvider(TranscriptProvider):
    name = "local-asr"

    def transcribe(self, audio: Path, manifest: dict[str, Any] | None = None) -> list[TranscriptSegment]:
        items = (manifest or {}).get("segments", [])
        return [TranscriptSegment(float(x.get("start", 0)), float(x.get("end", 0)), str(x.get("text", ""))) for x in items]


class WhisperCPPProvider(TranscriptProvider):
    name = "whisper.cpp"

    def __init__(self, binary: str, model: str):
        self.binary = binary
        self.model = model

    def transcribe(self, audio: Path, manifest: dict[str, Any] | None = None) -> list[TranscriptSegment]:
        out_prefix = audio.with_suffix("")
        cmd = [self.binary, "-m", self.model, "-f", str(audio), "-oj", "-of", str(out_prefix), "-l", "zh"]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        payload = json.loads(Path(str(out_prefix) + ".json").read_text(encoding="utf-8"))
        result = []
        for seg in payload.get("transcription", []):
            offsets = seg.get("offsets", {})
            result.append(TranscriptSegment(float(offsets.get("from", 0))/1000, float(offsets.get("to", 0))/1000, seg.get("text", "").strip()))
        return result


class HTTPASRProvider(TranscriptProvider):
    name = "http-asr"

    def __init__(self, endpoint: str, api_key: str = ""):
        self.endpoint = endpoint
        self.api_key = api_key

    def transcribe(self, audio: Path, manifest: dict[str, Any] | None = None) -> list[TranscriptSegment]:
        data = audio.read_bytes()
        req = urllib.request.Request(self.endpoint, data=data, method="POST")
        req.add_header("Content-Type", "audio/wav")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.load(resp)
        result = []
        for seg in payload.get("segments", []):
            result.append(TranscriptSegment(float(seg.get("start", 0)), float(seg.get("end", 0)), str(seg.get("text", ""))))
        if not result and payload.get("text"):
            result.append(TranscriptSegment(0.0, 0.0, str(payload["text"])))
        return result


def build_transcript_provider() -> TranscriptProvider:
    if os.getenv("ASR_API_URL"):
        return HTTPASRProvider(os.environ["ASR_API_URL"], os.getenv("ASR_API_KEY", ""))
    if os.getenv("WHISPER_CPP_BIN") and os.getenv("WHISPER_MODEL"):
        return WhisperCPPProvider(os.environ["WHISPER_CPP_BIN"], os.environ["WHISPER_MODEL"])
    return ManifestTranscriptProvider()
