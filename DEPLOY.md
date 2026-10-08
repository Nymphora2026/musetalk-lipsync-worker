# MuseTalk 1.5 GPU worker — deployment contract

Do not deploy from this document automatically. Do not create paid RunPod resources until a human starts ONE <$1 test.

## Isolation

This package is the only GPU stack:

- `workers/lipsync-musetalk/`

Do not install CUDA/PyTorch/MuseTalk into React, Supabase Edge Functions, or `workers/media/`.

Product path:

```
Product → MuseTalkProvider → MuseTalk GPU endpoint → GPU host adapter
```

RunPod Serverless is the first host adapter (`MUSETALK_GPU_HOST=runpod`). A generic HTTPS adapter (`MUSETALK_GPU_HOST=generic`) can replace it without changing product logic.

## Pinned revisions

| Item | Pin |
|---|---|
| MuseTalk code | `0a89dec45a0192b824e3cf4daf96c239440c5ed8` (official TMElyralab/MuseTalk) |
| MuseTalk version | 1.5 (`--version v15`) |
| Weights repo | `TMElyralab/MuseTalk` @ `2bcb936e2fddb4d86db4c62fd45b387d0c061571` |
| VAE | `stabilityai/sd-vae-ft-mse` @ `31f26fdeee1355a5c34592e401dd41e45d25a493` |
| Python | 3.10 |
| PyTorch | 2.2.2 + CUDA 12.1 |
| ffmpeg | distro 6.x in the CUDA 22.04 image |

See `pins.json`.

## Container image

- Build context: `workers/lipsync-musetalk/`
- GPU target: `gpu` (CUDA 12.1 runtime)
- CPU schema target: `schema` (no GPU, no spend)
- Entry point: `python -u handler.py` (`runpod.serverless.start`)
- Handler: `handler.handler`
- Official inference: `/opt/MuseTalk/scripts/inference.py`

### Size (estimated; measure on first image push)

| Layer | Baked into image? | Size |
|---|---|---|
| CUDA 12.1 runtime + Python + ffmpeg | yes | ~3–4 GB |
| PyTorch 2.2.2 cu121 + torchvision | yes | ~3–4 GB |
| mmcv / mmdet / mmpose | yes | ~1–2 GB |
| MuseTalk code @ pinned SHA | yes | <50 MB |
| Worker scripts | yes | <1 MB |
| **Image without weights** | | **~10–14 GB** |
| musetalkV15 UNet | persistent `/models` | ~3.4 GB |
| sd-vae-ft-mse | persistent `/models` | ~0.3 GB |
| whisper-tiny | persistent `/models` | ~0.15 GB |
| DWPose | persistent `/models` | ~0.4 GB |
| face-parse-bisent | persistent `/models` | ~0.05 GB |
| **Checkpoint cache** | | **~4.3 GB** |

Weights are **not** baked. Baking would inflate every cold image pull by ~4 GB.

## Model cache

- Mount a persistent volume at `/models`.
- First worker on an empty volume downloads checkpoints once (`download_models.sh`).
- Later workers on the same volume skip the download.
- Completed visual results cache at `/models/results/{identity}.mp4` so a retry of the same identity does not rerun inference.
- Cold start implications:
  - empty volume: minutes (checkpoint download + model load)
  - warm volume, cold GPU: tens of seconds (model load only)
  - warm worker: seconds
- Scale-to-zero: idle workers stop; billed time is execution + configured idle timeout. Keep idle timeout short for the $1 test (for example 5–10 seconds after the job).

## Cheapest suitable 16 GB GPU class

Do not require A100/H100. Official MuseTalk 1.5 fp16 documents 4 GB VRAM.

First choice for this test:

**NVIDIA T4 16 GB** on RunPod Serverless (or the current cheapest 16 GB Serverless SKU if T4 is missing from the queue).

Acceptable substitutes in the same class: RTX 2000 Ada 16 GB, RTX A4000 16 GB.

Do not pick 24 GB L4 unless 16 GB is unavailable; L4 is fine technically but is not the 16 GB target.

Prior public RunPod 16 GB Serverless list price used for later measurement: **$0.58/hour**. Do not treat that as the billed rate until the actual invoice/SKU is recorded.

Cost formula after the one run:

```
actual GPU cost = billed_seconds × actual_GPU_USD_per_second
```

Do not estimate a production unit cost before that measurement.

## Request schema

```json
{
  "id": "job-uuid",
  "identity": "sourceHash|audioHash|musetalk|musetalk-1.5|settings+pins|musetalk-lipsync-v1",
  "source_video": { "b64": "<base64 mp4>", "mime": "video/mp4" },
  "aligned_audio": { "b64": "<base64 wav>", "mime": "audio/wav" },
  "settings": {
    "version": "v15",
    "use_float16": true,
    "batch_size": 8,
    "extra_margin": 10,
    "parsing_mode": "jaw",
    "audio_padding_length_left": 2,
    "audio_padding_length_right": 2
  }
}
```

`url` may replace `b64` and must be `https://` (short-lived signed URL). HTTP is rejected.

Inputs are **original source MP4 + approved aligned dubbed WAV**. Do not send the lip-unmodified dubbed MP4 as the MuseTalk video input.

## Response schema

```json
{
  "id": "job-uuid",
  "status": "completed",
  "identity": "...",
  "output_video": { "b64": "<base64 mp4>", "mime": "video/mp4" },
  "timings": {
    "cold_start_ms": 0,
    "model_load_ms": 0,
    "preprocess_ms": 0,
    "inference_ms": 0,
    "encode_ms": 0,
    "upload_ms": 0,
    "total_ms": 0,
    "billed_ms": 0
  }
}
```

MuseTalk's muxed audio is **not** authoritative. The CPU worker remuxes this visual with the approved aligned WAV via `muxDubbedMp4`.

Statuses: `queued | processing | completed | failed | rejected`.

## Environment variables (GPU container)

| Name | Required | Purpose |
|---|---|---|
| `MUSETALK_ROOT` | no | default `/opt/MuseTalk` |
| `MUSETALK_MODEL_DIR` | no | default `/models` |
| `MUSETALK_RESULT_CACHE` | no | default `/models/results` |
| `MUSETALK_GPU_ID` | no | default `0` |
| `HF_HOME` | no | default `/models/hf` |
| `HUGGINGFACE_HUB_TOKEN` | no | only if a future gated file appears; current pins are public |

Do not put product secrets in the GPU worker. Signed URLs in the request body must never be logged.

## Environment variables (CPU product worker)

| Name | Required for the $1 test |
|---|---|
| `MUSETALK_GPU_ENDPOINT` | yes (RunPod endpoint id or generic HTTPS origin) |
| `MUSETALK_GPU_HOST` | `runpod` or `generic` |
| `MUSETALK_GPU_TOKEN` | yes for RunPod (`RUNPOD_API_KEY` may be copied here) |
| `MUSETALK_ALLOW_SUBMIT` | **must be `1`**. Unset = no GPU call |
| `MUSETALK_MAX_USD` | default `1` |
| `MUSETALK_GPU_HOURLY_USD` | actual SKU $/hour, recorded at test time |
| `LIPSYNC_PROVIDER` | leave unset (defaults to musetalk). Do not set `sync` |

`SYNC_API_KEY` must stay empty.

## Temporary disk

15-second 640×360 source: extract frames + latents + encode. Provision **at least 8 GB**, prefer **10 GB**, worker disk. Do not persist temp files; `handler.py` deletes the work directory.

## Scale-to-zero principles

- Min workers: 0
- Max workers: 1 for the experimental test
- Execution timeout: enough for cold start + 15s inference (10–15 minutes)
- Idle timeout: as short as the platform allows
- FlashBoot / cached workers: optional; not required for one test
- Network volume attached at `/models`

## Privacy

- Prefer private object bytes (base64 for this 15s clip) or short-lived HTTPS signed URLs
- Never log signed URLs, raw media, or secrets
- Delete temporary media after processing
- GPU adapter redacts `url`, `b64`, tokens, and keys before any error log

## Spend guard

Maximum experimental budget: **USD $1**.
This repository does not deploy, does not start a RunPod endpoint, and does not submit inference unless a human sets `MUSETALK_ALLOW_SUBMIT=1` after the endpoint exists.
