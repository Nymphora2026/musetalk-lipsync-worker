#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MUSETALK_MODEL_DIR:-/models}"
mkdir -p "$MODEL_DIR"

python - <<'PY'
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

pins = json.loads(Path("/opt/worker/pins.json").read_text(encoding="utf-8"))
root = Path(os.environ.get("MUSETALK_MODEL_DIR", "/models"))
root.mkdir(parents=True, exist_ok=True)

def cached_file(repo: str, revision: str, name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    downloaded = Path(hf_hub_download(repo_id=repo, filename=name, revision=revision))
    shutil.copy2(downloaded, dest)

ck = pins["checkpoints"]
cached_file(ck["musetalk_v15"]["repo"], ck["musetalk_v15"]["revision"], "musetalkV15/unet.pth", root / "musetalkV15" / "unet.pth")
cached_file(ck["musetalk_v15"]["repo"], ck["musetalk_v15"]["revision"], "musetalkV15/musetalk.json", root / "musetalkV15" / "musetalk.json")
# Official inference loads models/sd-vae (vae_type="sd-vae"). VAE class default is sd-vae-ft-mse.
cached_file(ck["sd_vae_ft_mse"]["repo"], ck["sd_vae_ft_mse"]["revision"], "config.json", root / "sd-vae" / "config.json")
cached_file(ck["sd_vae_ft_mse"]["repo"], ck["sd_vae_ft_mse"]["revision"], "diffusion_pytorch_model.bin", root / "sd-vae" / "diffusion_pytorch_model.bin")
alias = root / "sd-vae-ft-mse"
if alias.exists() and not alias.is_symlink():
    pass
elif not alias.exists():
    alias.symlink_to(root / "sd-vae")
for name in ck["whisper_tiny"]["files"]:
    cached_file(ck["whisper_tiny"]["repo"], ck["whisper_tiny"]["revision"], name, root / "whisper" / Path(name).name)
cached_file(ck["dwpose"]["repo"], ck["dwpose"]["revision"], "dw-ll_ucoco_384.pth", root / "dwpose" / "dw-ll_ucoco_384.pth")
cached_file(ck["face_parse_bisent"]["repo"], ck["face_parse_bisent"]["revision"], "79999_iter.pth", root / "face-parse-bisent" / "79999_iter.pth")
cached_file(ck["face_parse_bisent"]["repo"], ck["face_parse_bisent"]["revision"], "resnet18-5c106cde.pth", root / "face-parse-bisent" / "resnet18-5c106cde.pth")
print("models-ready")
PY
