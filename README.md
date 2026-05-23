# etch

Text-to-image generation on Apple Silicon using [Candle](https://github.com/huggingface/candle). Supports FLUX.1 and SDXL pipelines with Metal acceleration.

## Models

| Flag | Model | Size | Steps | License |
|------|-------|------|-------|---------|
| `--model schnell` | FLUX.1-schnell | ~25 GB | 4 | Apache 2.0 |
| `--model dev` | FLUX.1-dev | ~25 GB | 50 | Non-commercial |
| `--model araminta` | the-araminta-experiment-fv5-sdxl | ~7 GB | 20 | — |

Weights are downloaded automatically from HuggingFace Hub on first run and cached in `~/.cache/huggingface/`.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain (`rustup`)
- HuggingFace account for gated models (FLUX.1-dev)

## Build

```bash
# Apple Silicon (Metal GPU)
cargo build --release --features metal

# CPU only
cargo build --release
```

## Usage

### FLUX.1-schnell

```bash
./target/release/etch \
  --model schnell \
  --prompt "A rusty robot walking on a beach"
```

### FLUX.1-dev

```bash
./target/release/etch \
  --model dev \
  --prompt "A cyberpunk cat in neon-lit Tokyo, highly detailed" \
  --n-steps 50
```

### SDXL (Araminta)

```bash
./target/release/etch \
  --model araminta \
  --prompt "portrait of a woman, cinematic lighting, 8k" \
  --uncond-prompt "blurry, low quality, deformed, ugly" \
  --guidance-scale 7.5
```

### With LoRA

```bash
./target/release/etch \
  --model araminta \
  --prompt "portrait in style of xyz" \
  --lora /path/to/lora.safetensors \
  --lora-scale 0.8
```

### With a local model directory

```bash
./target/release/etch \
  --model araminta \
  --prompt "a fantasy landscape" \
  --local-model /path/to/sdxl-diffusers
```

The directory must be in diffusers format. Convert from a single `.safetensors` file with:

```python
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_single_file("model.safetensors")
pipe.save_pretrained("/path/to/sdxl-diffusers")
```

## Schedulers (SDXL only)

| Value | Description |
|-------|-------------|
| `euler-a` | Euler Ancestral — stochastic, fast, good all-rounder (default) |
| `euler-a-karras` | Euler Ancestral + Karras sigma spacing |
| `dpm2m-karras` | DPM++ 2M Karras — smooth, great at 20–30 steps |

```bash
./target/release/etch \
  --model araminta \
  --prompt "scenic mountain landscape" \
  --scheduler dpm2m-karras \
  --n-steps 25
```

## All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `schnell` | `schnell` / `dev` / `araminta` |
| `--prompt` | — | Text prompt |
| `--uncond-prompt` | `""` | Negative prompt (SDXL only) |
| `--height` | 768 / 1024 | Output height in pixels |
| `--width` | 1360 / 1024 | Output width in pixels |
| `--n-steps` | 4 / 50 / 20 | Denoising steps |
| `--guidance-scale` | `7.5` | CFG scale (SDXL only) |
| `--scheduler` | `euler-a` | Sampler type (SDXL only) |
| `--clip-skip` | `1` | CLIP layers to skip from end (SDXL only) |
| `--seed` | random | Random seed for reproducibility |
| `--output` | `out-<rand>.png` | Output file path |
| `--lora` | — | Path to LoRA `.safetensors` (SDXL only) |
| `--lora-scale` | `1.0` | LoRA strength |
| `--local-model` | — | Local diffusers model dir (SDXL only, overrides HF download) |
| `--cpu` | — | Force CPU instead of Metal |

## LoRA format

Supports standard kohya_ss safetensors format:

```
lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight
lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight
lora_unet_down_blocks_0_attentions_0_to_q.alpha  (optional)
```

LoRA weights are merged into the UNet before inference — no runtime overhead.

## Troubleshooting

### Одно и то же лицо на всех изображениях

Это особенность модели Araminta, а не баг. Несколько способов получить разнообразие:

**Задай явный сид** — без `--seed` генератор инициализируется одинаково:
```bash
--seed 1234
--seed 9999
```

**Детализируй промпт** — без конкретики модель падает в "средний" образ из обучающих данных:
```
"portrait of an elderly asian woman with wrinkles, gray hair, warm smile"
```

**Снизь guidance scale** — высокий CFG усиливает bias датасета:
```bash
--guidance-scale 5.5
```

**Используй негативный промпт:**
```bash
--uncond-prompt "same face, identical features, clone, blurry, low quality"
```

## Download sizes (first run)

**FLUX.1-schnell / dev:**
- `flux1-schnell.safetensors` — 24 GB
- `ae.safetensors` — 335 MB
- `google/t5-v1_1-xxl` — 10 GB
- `openai/clip-vit-large-patch14` — 1.7 GB

**Araminta (SDXL):**
- UNet — 5 GB
- Text encoders (×2) — 4 GB
- VAE — 335 MB
- Tokenizers — small
