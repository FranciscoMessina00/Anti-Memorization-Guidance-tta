# Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2509.14934) *Francisco Messina, Francesca Ronchini, Luca Comanducci, Paolo Bestagini, Fabio Antonacci*

This repository contains the code used in the paper *[Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance](https://arxiv.org/abs/2509.14934)*. It builds on Stability AI’s Stable Audio Open 1.0 and adds *Anti-Memorization Guidance (AMG)* during sampling. The main entry point for generation is `my_infer.py`.

References to the paper and base model are at the end of this document.

## Quick overview

- Goal: Generate audio from text prompts while discouraging replication of training data via AMG.
- Core script: `my_infer.py` (configure guidance and scheduling, then run).
- Internal hook: `stable_audio_tools/inference/my_generation.py` implements custom sampling and loads precomputed CLAP embeddings from `embeddings_new.json`.

## 1) Environment setup (conda)

Use the provided `environment.yml` to create a conda env:

```bash
conda env create -f environment.yml -n sao-amg
conda activate sao-amg
```

Notes:
- A recent NVIDIA driver and CUDA toolkit are recommended for GPU inference. CPU is supported but slow.
- If you prefer pip-only, mirror the dependencies from `environment.yml` (PyTorch/torchaudio versions must match).

## 2) CLAP installation and import path

AMG relies on CLAP audio/text embeddings. You have two options:

1) Install CLAP as a Python package (recommended for most users):
	 - LAION-CLAP: `pip install laion-clap`
	 - Or follow the official CLAP repo instructions.

2) Use a local CLAP clone and adjust the import path.
	 - In `stable_audio_tools/inference/my_generation.py` there’s a small section near the top that modifies `sys.path` to point to a local CLAP checkout (e.g., `CLAP/src`).
	 - If you don’t have a local copy, remove that relocation block or change it to the correct path in your setup to avoid “module not found” errors.

If you hit CLAP import issues, first try option (1) and remove the `sys.path` relocation.

## 3) Precomputed dataset embeddings (embeddings_new.json)

`my_generation.py` loads `embeddings_new.json`, which contains precomputed CLAP embeddings for the Stable Audio Open dataset. These are used by the AMG term during denoising.

- You may add more entries manually using the same JSON format and the default CLAP checkpoint to compute embeddings.
- To see which audio IDs were used in the dataset and their sources, consult the Stable Audio Open 1.0 card (CSV references to Freesound):
	- https://huggingface.co/stabilityai/stable-audio-open-1.0

Minimal JSON format example:

```json
{
  "1234": {
    "embedding": [0.0123, -0.0456, 0.1546],
    "conditioning": {
      "prompt": "An acoustic drum loop, 110 bpm",
      "seconds_start": 0.0,
      "seconds_total": 10.0
    }
  }
}
```
This metadata is useful for reproducibility and analysis; AMG primarily consumes the `embedding` vectors during guidance, and `prompt` for the caption deduplication guidance.

## 4) Running generation (my_infer.py)

Open `my_infer.py` and set your parameters:

- `prompt`: Your text description.
- `total_duration`: Target duration in seconds.
- `denoising_steps`: Number of diffusion steps.
- `cfg_scale`: Classifier-free guidance scale.
- `c1, c2, c3`: AMG guidance weights (larger typically stronger effect):
	- `c1`: Despecification guidance (slows down the CF guidance)
	- `c2`: Caption deduplication guidance (uses closest training example's prompt as a negative prompt)
	- `c3`: Dissimilarity guidance (pushes away from nearest neighbors in embedding space)
- `lambda_min, lambda_max`: Parabolic scheduling bounds for AMG along the denoising trajectory.
- `sampler_type`: Custom sampler (default: `my-dpmpp-3m-sde`).
- `sigma_min, sigma_max`: Noise schedule bounds.

Then run:

```bash
python my_infer.py
```

The script will save a waveform to `audio.wav` in the repository root by default.

### What my_infer.py does

- Loads Stable Audio Open 1.0 via `get_pretrained_model("stabilityai/stable-audio-open-1.0")`.
- Configures sampling with the parameters you specify.
- Calls `my_generate_diffusion_cond(...)` in `my_generation.py`, which:
	- Loads CLAP embeddings from `embeddings_new.json`.
	- Applies AMG guided denoising using `c1, c2, c3` and `lambda_min/lambda_max`.
	- Returns the generated audio.

## 5) Troubleshooting

- CLAP import errors: Remove or adjust the CLAP `sys.path` override in `my_generation.py`, or install CLAP via pip.
- Torchaudio/Torch ABI mismatch: Ensure `torch` and `torchaudio` versions match (e.g., reinstall both from the same CUDA/CPU channel or wheel index).
- CUDA OOM: Lower `denoising_steps`, reduce `total_duration`, or run on CPU (slow).
- No output / silent audio: Check `sigma_min/sigma_max` and `cfg_scale` are reasonable. Start with the defaults in `my_infer.py`.

## 6) Data and licensing

- Stable Audio Open 1.0 dataset and model card: https://huggingface.co/stabilityai/stable-audio-open-1.0
- Base repository: https://github.com/Stability-AI/stable-audio-tools

Please follow the licensing terms of each dependency and dataset.

## 7) Citation

If you use this code in academic work, please cite:

Thesis work:

```
@misc{
	messina2025mitigatingdatareplicationtexttoaudio,
	title={Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance}, 
	author={Francisco Messina and Francesca Ronchini and Luca Comanducci and Paolo Bestagini and Fabio Antonacci},
	year={2025},
	eprint={2509.14934},
	archivePrefix={arXiv},
	primaryClass={eess.AS},
	url={https://arxiv.org/abs/2509.14934}, 
}
```

Stable Audio Open:

```
@misc{
	evans2024stableaudioopen,
	title={Stable Audio Open}, 
	author={Zach Evans and Julian D. Parker and CJ Carr and Zack Zukowski and Josiah Taylor and Jordi Pons},
	year={2024},
	eprint={2407.14358},
	archivePrefix={arXiv},
	primaryClass={cs.SD},
	url={https://arxiv.org/abs/2407.14358}, 
}
```

---

For questions or reproducibility details (e.g., exact `c1/c2/c3` and scheduling configurations used for the paper experiments), you can inspect `my_infer.py` in this repository, the AMG logic within `stable_audio_tools/inference/my_generation.py`, and the reference paper.

