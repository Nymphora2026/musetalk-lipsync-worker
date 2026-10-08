"""RunPod/generic GPU entry. Inference is lazy-imported so CPU schema checks stay free."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from contract import REQUIRED_TIMINGS, validate_request, validate_response

PROCESS_STARTED = time.monotonic()
MODELS_LOADED = False
MODEL_LOAD_MS = 0
RESULT_CACHE = Path(os.environ.get("MUSETALK_RESULT_CACHE", "/models/results"))


def handler(event: dict[str, Any]) -> dict[str, Any]:
    billed_started = time.monotonic()
    payload = event.get("input") if isinstance(event, dict) and "input" in event else event
    validate_request(payload)
    work = Path(tempfile.mkdtemp(prefix="musetalk-"))
    try:
        source = work / "source.mp4"
        audio = work / "aligned.wav"
        output = work / "lipsynced.mp4"
        _write_media(payload["source_video"], source)
        _write_media(payload["aligned_audio"], audio)
        identity = payload.get("identity") or _identity(source, audio)
        cached = RESULT_CACHE / f"{_safe(identity)}.mp4"
        timings = {key: 0 for key in REQUIRED_TIMINGS}
        timings["cold_start_ms"] = int((time.monotonic() - PROCESS_STARTED) * 1000)

        if cached.is_file():
            output.write_bytes(cached.read_bytes())
        else:
            global MODELS_LOADED, MODEL_LOAD_MS
            load_started = time.monotonic()
            infer = _inference()
            if not MODELS_LOADED:
                infer.ensure_models()
                MODEL_LOAD_MS = int((time.monotonic() - load_started) * 1000)
                MODELS_LOADED = True
            timings["model_load_ms"] = MODEL_LOAD_MS
            measured = infer.run(source, audio, output, payload.get("settings") or {})
            timings.update(measured)
            RESULT_CACHE.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output, cached)

        response = {
            "id": payload["id"],
            "status": "completed",
            "identity": identity,
            "output_video": {
                "b64": base64.b64encode(output.read_bytes()).decode("ascii"),
                "mime": "video/mp4",
            },
            "timings": {
                **timings,
                "total_ms": int((time.monotonic() - billed_started) * 1000),
                "billed_ms": int((time.monotonic() - billed_started) * 1000),
            },
        }
        return validate_response(response)
    except Exception as error:
        return _failed(payload, billed_started, error)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _failed(payload: dict[str, Any], billed_started: float, error: Exception) -> dict[str, Any]:
    message = str(error)
    safe = message if message in {"invalid_request", "unsupported_source", "unsupported_audio", "unsupported_version", "media_too_large"} else "Lip sync failed."
    return {
        "id": payload.get("id", "unknown"),
        "status": "failed",
        "error": safe,
        "timings": {key: 0 for key in REQUIRED_TIMINGS} | {
            "total_ms": int((time.monotonic() - billed_started) * 1000),
            "billed_ms": int((time.monotonic() - billed_started) * 1000),
        },
    }


def _write_media(item: dict[str, Any], dest: Path) -> None:
    if item.get("b64"):
        dest.write_bytes(base64.b64decode(item["b64"]))
        return
    url = item["url"]
    if not str(url).startswith("https://"):
        raise ValueError("invalid_request")
    request = Request(url, method="GET")
    with urlopen(request, timeout=60) as response:  # noqa: S310 - https-only above
        dest.write_bytes(response.read())


def _identity(source: Path, audio: Path) -> str:
    return hashlib.sha256(source.read_bytes() + b"|" + audio.read_bytes()).hexdigest()


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)[:128]


def _inference():
    from run_inference import MuseTalkInference
    return MuseTalkInference()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        sample = {
            "id": "job-1",
            "source_video": {"b64": "AAAA", "mime": "video/mp4"},
            "aligned_audio": {"b64": "AAAA", "mime": "audio/wav"},
            "settings": {"version": "v15"},
        }
        validate_request(sample)
        print(json.dumps({"stage": "musetalk-handler-schema", "pass": True}))
        return
    import runpod
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
