# MuseTalk 1.5 dependency license audit

Prepared for one experimental 15-second quality test. Not legal advice.
Unresolved items are listed as REVIEW REQUIRED. Do not treat those as cleared.

Pinned code: `TMElyralab/MuseTalk` revision `0a89dec45a0192b824e3cf4daf96c239440c5ed8`.
Pinned MuseTalk weights repo: `TMElyralab/MuseTalk` revision `2bcb936e2fddb4d86db4c62fd45b387d0c061571`.

| Component | Source | License signal | Classification |
|---|---|---|---|
| MuseTalk code | TMElyralab/MuseTalk LICENSE | MIT | **OK FOR COMMERCIAL USE** |
| MuseTalk 1.5 UNet (`musetalkV15/unet.pth`, `musetalk.json`) | TMElyralab/MuseTalk on Hugging Face (`license:mit`) plus authors' commercial-use statement in the project LICENSE | MIT / authors permit commercial weights | **OK FOR COMMERCIAL USE** |
| sd-vae-ft-mse | stabilityai/sd-vae-ft-mse | Hugging Face card currently tags `license:mit`. This VAE historically shipped under CreativeML OpenRAIL-M. MuseTalk's LICENSE also claims MIT for listed weights. | **REVIEW REQUIRED** |
| openai/whisper-tiny | openai/whisper-tiny @ `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | Hugging Face card: Apache-2.0. Upstream OpenAI Whisper GitHub: MIT. | **ATTRIBUTION REQUIRED** |
| DWPose `dw-ll_ucoco_384.pth` | yzd-v/DWPose @ `1a7144101628d69ee7a3768d1ee3a094070dc388`; mmpose stack | Apache-2.0 | **ATTRIBUTION REQUIRED** |
| face-parse-bisent (`79999_iter.pth`, ResNet-18) | Unofficial HF mirror ManyOtherFunctions/face-parse-bisent @ `0073b233a5a3c4b1377d4dbf49245017938a72b5`; original zllrunning/face-parsing.PyTorch | Mirror card: WTFPL. Original repo: MIT. Weights originated from CelebAMask-HQ training. | **REVIEW REQUIRED** (mirror + dataset provenance) + **ATTRIBUTION REQUIRED** |
| S3FD face detector | Commonly pulled via MuseTalk/mmpose/face-alignment paths (`yxlijun/S3FD.pytorch` and similar ports) | No project LICENSE file in the widely used PyTorch port | **REVIEW REQUIRED** |
| mmcv / mmengine / mmdet / mmpose | OpenMMLab | Apache-2.0 | **ATTRIBUTION REQUIRED** |
| PyTorch / torchvision | Meta | BSD-style | **ATTRIBUTION REQUIRED** |
| diffusers / transformers | Hugging Face | Apache-2.0 | **ATTRIBUTION REQUIRED** |
| MuseTalk testdata / demo assets | MuseTalk README | Non-commercial testdata notice | **BLOCKED** for product shipping. Not used in this worker. |
| ByteDance/LatentSync SyncNet weights | LatentSync | Not downloaded. Official `scripts/inference.py` does not load them. | Not in runtime path. **Do not install.** |

## Unresolved

1. **sd-vae-ft-mse**: current HF tag is MIT; historical OpenRAIL-M terms may still be argued to apply. Do not treat as cleared for production until counsel reviews.
2. **S3FD**: license underspecified in the PyTorch ports MuseTalk's face pipeline historically uses. Do not treat as cleared.
3. **face-parse-bisent**: official MuseTalk Windows download switched to the ManyOtherFunctions HF mirror (WTFPL tag). Original parser is MIT; CelebAMask-HQ dataset terms are a separate review. Do not treat the mirror as the original author's distribution.

A 15-second internal quality test can proceed with those two items flagged. A commercial launch cannot hide them.
