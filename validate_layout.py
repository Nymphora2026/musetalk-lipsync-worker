"""CPU-only layout/contract checks. No GPU. No model download."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from contract import validate_request, validate_response, REQUIRED_TIMINGS

ROOT = Path(__file__).resolve().parent
OFFICIAL_WHISPER_FILES = ("config.json", "pytorch_model.bin", "preprocessor_config.json")
SOURCE_FILES = (
    "Dockerfile",
    "download_models.sh",
    "handler.py",
    "run_inference.py",
    "contract.py",
    "pins.json",
)


def main() -> None:
    pins = json.loads((ROOT / "pins.json").read_text(encoding="utf-8"))
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    download = (ROOT / "download_models.sh").read_text(encoding="utf-8")

    assert pins["torch"] == "2.1.2", "pins.json torch must match Dockerfile 2.1.2"
    assert pins["torchvision"] == "0.16.2", "pins.json torchvision must match Dockerfile 0.16.2"
    assert pins["mmcv"] == "2.1.0", "pins.json mmcv must match Dockerfile 2.1.0"
    assert pins["cuda"] == "12.1", "pins.json cuda must match cu121"
    assert "2.2.2" not in json.dumps(pins), "stale torch 2.2.2 metadata must be removed"
    assert "torch==2.1.2" in dockerfile and "torchvision==0.16.2" in dockerfile
    assert 'mim install "mmcv==2.1.0"' in dockerfile

    vae = pins["checkpoints"]["sd_vae_ft_mse"]
    assert vae["dest"] == "sd-vae", "VAE dest must be sd-vae"
    assert 'root / "sd-vae"' in download, "download_models.sh must populate /models/sd-vae"
    assert "sd-vae-ft-mse" in download and "symlink_to" in download
    assert "ln -sfn /models /opt/MuseTalk/models" in dockerfile
    assert 'ENV PYTHONPATH="/opt/MuseTalk"' in dockerfile
    assert "snapshot_download" not in download

    whisper_files = tuple(pins["checkpoints"]["whisper_tiny"]["files"])
    assert whisper_files == OFFICIAL_WHISPER_FILES, "whisper files must match official MuseTalk download_weights.sh"

    for path in SOURCE_FILES:
        text = (ROOT / path).read_text(encoding="utf-8")
        mac_home = "/Users/" + "macbook"
        assert mac_home not in text, f"Mac path leaked in {path}"
        assert "C:\\Users\\" not in text, f"Windows path leaked in {path}"

    with tempfile.TemporaryDirectory(prefix="musetalk-layout-") as raw:
        models = Path(raw) / "models"
        vae_dir = models / "sd-vae"
        vae_dir.mkdir(parents=True)
        (vae_dir / "config.json").write_text("{}", encoding="utf-8")
        (models / "sd-vae-ft-mse").symlink_to(vae_dir)
        musetalk = Path(raw) / "opt" / "MuseTalk"
        musetalk.mkdir(parents=True)
        (musetalk / "models").symlink_to(models)
        via_musetalk = musetalk / "models" / "sd-vae" / "config.json"
        alias = musetalk / "models" / "sd-vae-ft-mse"
        assert via_musetalk.is_file(), "MuseTalk models symlink must expose sd-vae"
        assert alias.resolve() == vae_dir.resolve(), "sd-vae-ft-mse must alias sd-vae"

    validate_request({
        "id": "layout-check",
        "source_video": {"b64": "AAAA", "mime": "video/mp4"},
        "aligned_audio": {"b64": "AAAA", "mime": "audio/wav"},
        "settings": {"version": "v15"},
    })
    validate_response({
        "id": "layout-check",
        "status": "queued",
        "timings": {key: 0 for key in REQUIRED_TIMINGS},
    })
    print(json.dumps({
        "stage": "musetalk-layout",
        "pass": True,
        "vae_dest": "sd-vae",
        "whisper_files": list(OFFICIAL_WHISPER_FILES),
        "pythonpath": "/opt/MuseTalk",
        "cwd": os.getcwd(),
    }))


if __name__ == "__main__":
    main()
