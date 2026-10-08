"""CPU-safe request/response contract. No torch. No GPU."""

from __future__ import annotations

import json
from typing import Any

ALLOWED_STATUSES = {"queued", "processing", "completed", "failed", "rejected"}
MAX_MEDIA_BYTES = 80 * 1024 * 1024
REQUIRED_REQUEST = ("id", "source_video", "aligned_audio")
REQUIRED_TIMINGS = (
    "cold_start_ms",
    "model_load_ms",
    "preprocess_ms",
    "inference_ms",
    "encode_ms",
    "upload_ms",
    "total_ms",
    "billed_ms",
)


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("invalid_request")
    for key in REQUIRED_REQUEST:
        if key not in payload:
            raise ValueError("invalid_request")
    source = payload["source_video"]
    audio = payload["aligned_audio"]
    if not isinstance(source, dict) or not isinstance(audio, dict):
        raise ValueError("invalid_request")
    if not _has_media(source) or not _has_media(audio):
        raise ValueError("invalid_request")
    if source.get("mime") not in (None, "video/mp4"):
        raise ValueError("unsupported_source")
    if audio.get("mime") not in (None, "audio/wav"):
        raise ValueError("unsupported_audio")
    settings = payload.get("settings") or {}
    if settings.get("version") not in (None, "v15"):
        raise ValueError("unsupported_version")
    return payload


def validate_response(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("invalid_response")
    if payload.get("status") not in ALLOWED_STATUSES:
        raise ValueError("invalid_response")
    if "id" not in payload:
        raise ValueError("invalid_response")
    timings = payload.get("timings")
    if timings is not None:
        if not isinstance(timings, dict):
            raise ValueError("invalid_response")
        for key in REQUIRED_TIMINGS:
            if key not in timings:
                raise ValueError("invalid_response")
    if payload.get("status") == "completed" and not _has_media(payload.get("output_video") or {}):
        raise ValueError("invalid_response")
    return payload


def decode_b64_size(value: str) -> int:
    padding = value.count("=")
    return max(0, (len(value) * 3) // 4 - padding)


def _has_media(item: dict[str, Any]) -> bool:
    b64 = item.get("b64")
    url = item.get("url")
    if isinstance(b64, str) and b64.strip():
        if decode_b64_size(b64) > MAX_MEDIA_BYTES:
            raise ValueError("media_too_large")
        return True
    if isinstance(url, str) and url.startswith("https://"):
        return True
    return False


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


if __name__ == "__main__":
    sample = {
        "id": "job-1",
        "source_video": {"b64": "AAAA", "mime": "video/mp4"},
        "aligned_audio": {"b64": "AAAA", "mime": "audio/wav"},
        "settings": {"version": "v15", "use_float16": True, "batch_size": 8},
    }
    validate_request(sample)
    validate_response({
        "id": "job-1",
        "status": "queued",
        "timings": {key: 0 for key in REQUIRED_TIMINGS},
    })
    print(json.dumps({"stage": "musetalk-contract", "pass": True}))
