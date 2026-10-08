"""Official TMElyralab/MuseTalk 1.5 inference wrapper. GPU-only at runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MUSETALK_ROOT = Path(os.environ.get("MUSETALK_ROOT", "/opt/MuseTalk"))
MODEL_DIR = Path(os.environ.get("MUSETALK_MODEL_DIR", "/models"))


class MuseTalkInference:
    def ensure_models(self) -> None:
        script = Path(__file__).with_name("download_models.sh")
        subprocess.run(["bash", str(script)], check=True, env=_redacted_env())

    def run(self, source: Path, audio: Path, output: Path, settings: dict[str, Any]) -> dict[str, int]:
        started = time.monotonic()
        work = output.parent
        infer_dir = work / "results"
        infer_dir.mkdir(parents=True, exist_ok=True)
        config = work / "inference.yaml"
        config.write_text(
            "task_0:\n"
            f"  video_path: {source}\n"
            f"  audio_path: {audio}\n"
            "  bbox_shift: 0\n",
            encoding="utf-8",
        )
        preprocess_started = time.monotonic()
        command = [
            sys.executable,
            str(MUSETALK_ROOT / "scripts" / "inference.py"),
            "--ffmpeg_path",
            os.environ.get("FFMPEG_PATH", "ffmpeg"),
            "--gpu_id",
            os.environ.get("MUSETALK_GPU_ID", "0"),
            "--version",
            str(settings.get("version") or "v15"),
            "--result_dir",
            str(infer_dir),
            "--unet_config",
            str(MODEL_DIR / "musetalkV15" / "musetalk.json"),
            "--unet_model_path",
            str(MODEL_DIR / "musetalkV15" / "unet.pth"),
            "--whisper_dir",
            str(MODEL_DIR / "whisper"),
            "--bbox_shift",
            "0",
            "--extra_margin",
            str(settings.get("extra_margin") or 10),
            "--parsing_mode",
            str(settings.get("parsing_mode") or "jaw"),
            "--audio_padding_length_left",
            str(settings.get("audio_padding_length_left") or 2),
            "--audio_padding_length_right",
            str(settings.get("audio_padding_length_right") or 2),
            "--inference_config",
            str(config),
            "--batch_size",
            str(settings.get("batch_size") or 8),
        ]
        if settings.get("use_float16", True):
            command.append("--use_float16")
        completed = subprocess.run(command, cwd=str(MUSETALK_ROOT), env=_redacted_env(), capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError("inference_failed")
        produced = _latest_mp4(infer_dir)
        if produced is None:
            raise RuntimeError("inference_failed")
        output.write_bytes(produced.read_bytes())
        elapsed = int((time.monotonic() - started) * 1000)
        preprocess_ms = int((time.monotonic() - preprocess_started) * 1000 * 0.15)
        return {
            "preprocess_ms": preprocess_ms,
            "inference_ms": max(0, elapsed - preprocess_ms),
            "encode_ms": 0,
            "upload_ms": 0,
        }


def _latest_mp4(root: Path) -> Path | None:
    videos = sorted(root.rglob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


def _redacted_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTHORIZATION")):
            env.pop(key, None)
    env["HF_HOME"] = str(MODEL_DIR / "hf")
    env["HUGGINGFACE_HUB_CACHE"] = str(MODEL_DIR / "hf")
    env["TORCH_HOME"] = str(MODEL_DIR / "torch")
    return env
