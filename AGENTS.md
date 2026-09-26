# AGENTS.md — etch

> Agent-oriented guide for the `etch` codebase. Assumes the reader knows nothing about the project.

## Project overview

`etch` is a single-crate Rust CLI for text-to-image generation using [Candle](https://github.com/huggingface/candle). It supports two independent diffusion pipelines:

- **FLUX.1** (`src/flux/`) — `schnell`, `dev`, `schnell-gguf`, `dev-gguf`
- **SDXL** (`src/sdxl/`) — `araminta` (`John6666/the-araminta-experiment-fv5-sdxl`)

Backends: Metal on Apple Silicon, CUDA on NVIDIA, or CPU fallback. Weights download automatically from HuggingFace Hub on first run and cache in `~/.cache/huggingface/`.

- Package: `etch` v0.1.0
- Rust edition: 2024
- Toolchain: `stable` (pinned in `rust-toolchain.toml`)
- Author: Rust Wizard
- License: not specified in `Cargo.toml`; individual models have their own licenses (FLUX.1-dev is non-commercial)

## Technology stack

Key crates (see `Cargo.toml`):

| Crate | Purpose |
|-------|---------|
| `candle-core`, `candle-nn`, `candle-transformers` 0.11 | ML framework and model implementations |
| `hf-hub` 0.4 (tokio feature) | HuggingFace Hub downloads |
| `clap` 4 (derive) | CLI parsing |
| `image` 0.25 (png, jpeg only) | Saving output images |
| `tokenizers` 0.21 | Text tokenization |
| `indicatif` 0.17 | Progress bars |
| `tracing`, `tracing-subscriber` | Logging |
| `anyhow` | Error handling |
| `serde_json` | JSONL run logs |
| `rand` | Seed generation |
| `sha2`, `dirs` | Embedding cache keys and cache dir (`~/.cache/etch/`) |
| `ctrlc` | Graceful Ctrl-C handling (`src/signals.rs`) |

Release profile: `opt-level = 3`, `lto = true`, `codegen-units = 1`.

## Build & run commands

```bash
# macOS (Apple Silicon, Metal)
cargo build --release --features metal

# Linux / WSL2 (NVIDIA, CUDA)
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cargo build --release --features cuda

# CPU only
cargo build --release
```

Run the binary (the `--` separator is required when using `cargo run`):

```bash
cargo run --release --features metal -- --prompt "a cat" --model schnell

# Or run the built binary directly
./target/release/etch --prompt "a cat" --model schnell
```

Cargo features:

- `metal` — enables `candle-core/metal`, `candle-nn/metal`, `candle-transformers/metal`
- `cuda` — enables `candle-core/cuda`, `candle-nn/cuda`, `candle-transformers/cuda`
- Default features: empty (CPU only)

Build notes:

- WSL2 OOM workaround: `CARGO_BUILD_JOBS=1 cargo build --release --features cuda`
- `metal` and `cuda` are **mutually exclusive**: enabling both fails the build with a `compile_error!` guard in `src/device.rs`.

## Lint & format

```bash
cargo fmt --check
cargo clippy --features metal   # or --features cuda
```

- `src/main.rs` starts with `#![deny(clippy::unwrap_used)]`. Prefer `?` or `.expect("message")` with a descriptive message.
- Toolchain is pinned to `stable` in `rust-toolchain.toml`.
- Run `cargo clippy --features metal` (or `cuda`) locally before pushing GPU-related changes; CI does **not** compile those feature combinations.

## Testing

**No test suite currently exists.** `cargo test` runs zero tests.

Planned tests are documented in `IMPROVEMENTS.md`:

- Unit tests for `te_lora_base_to_weight_key`, `ldm_lora_base_to_unet_key`, `greedy_tokenize` in `src/lora.rs`
- Unit tests for `build_karras_schedule` monotonicity/boundaries in `src/schedulers.rs`
- Integration smoke test: run with `--cpu --n-steps 1` to verify pipeline startup

When adding tests, use `cargo test` for unit tests and consider the smoke test target above.

## CI / deployment

Two GitHub Actions workflows live in `.github/workflows/`:

1. **`ci.yml`** — runs on push/PR to `master` on `ubuntu-latest`:
   - `cargo fmt --check`
   - `cargo check`
   - `cargo clippy -- -D warnings`
   - `cargo test`
   - This runs **without** GPU features, so Metal/CUDA code paths are not validated in CI.

2. **`mirror.yml`** — mirrors `master` and `v*` tags to `gitverse.ru:rustwizard/etch.git` via SSH deploy key (`secrets.GITVERSE_SSH_KEY`).

There is no release packaging, Docker image, or crates.io publishing workflow. Default branch is `master`.

## Architecture & code organization

Source tree (`src/**/*.rs`):

| File | Responsibility |
|------|----------------|
| `main.rs` | Entry point: parse CLI, init logging, pick device, resolve seeds, dispatch pipeline, log results |
| `cli.rs` | `Args` struct, enums (`Model`, `SamplerType`, `DtypeArg`, `Quantization`), seed-range parsing, output path generation |
| `device.rs` | `pick_device(cpu)` — Metal/CUDA/CPU selection |
| `pipeline.rs` | `Pipeline` trait and `for_model()` dispatch |
| `hub.rs` | HuggingFace Hub fetch helpers + model-size logging |
| `image.rs` | Save a 3-channel tensor to PNG or JPEG |
| `logger.rs` | Append run results to `log.jsonl` |
| `lora.rs` | LoRA weight merging for UNet and text encoders |
| `progress.rs` | Shared `indicatif` denoising progress bar |
| `schedulers.rs` | SDXL Karras Euler-A and DPM++ 2M Karras schedulers |
| `signals.rs` | Ctrl-C handling: interrupt flag polled at step/seed boundaries |
| `cache.rs` | Text embedding disk cache (avoid reloading T5/CLIP on repeat prompts) |
| `vae_tiling.rs` | Tiled VAE decode for memory-constrained high-res images |
| `flux/mod.rs` | FLUX module exports |
| `flux/run.rs` | FLUX pipeline implementation |
| `flux/gguf.rs` | GGUF quantized `VarBuilder` loader |
| `flux/model.rs` | Enum wrapper around full vs quantized FLUX DiT |
| `sdxl/mod.rs` | SDXL module exports |
| `sdxl/run.rs` | SDXL pipeline implementation |
| `sdxl/clip.rs` | SDXL dual CLIP embedding helper |

### Runtime flow

1. `main.rs` parses `Args`, validates that `--seed` and `--seed-range` are mutually exclusive, and installs the Ctrl-C handler (`signals::install()`).
2. Initializes `tracing_subscriber` (verbose mode shows timestamps + targets; otherwise minimal).
3. Calls `device::pick_device(args.cpu)`.
4. Resolves seed list: explicit `--seed`, parsed `--seed-range START-END`, or a single random `u64`.
5. Chooses dtype: `BF16` on GPU by default, `F32` on CPU by default, or `--dtype` override.
6. Calls `pipeline.prepare(...)` **once** — loads all weights (text encoders, DiT/UNet, VAE) and reusable state. A `prepare` failure is fatal.
7. For each seed:
   - If the interrupt flag is set (Ctrl-C), exits with code 130.
   - Sets seed on the device (warns on failure; CPU ignores it).
   - Builds per-seed output path via `cli::output_for_seed`.
   - Creates parent directories.
   - Calls `pipeline.generate(...)` — reuses the prepared weights; the denoising loops poll the interrupt flag at step boundaries.
   - On success, logs elapsed time and writes a JSONL entry.
   - On failure: if interrupted, exits with code 130 (no failure log entry); otherwise logs the error, writes a failure JSONL entry, and **continues** to the next seed.

### Pipeline dispatch (`src/pipeline.rs`)

```rust
pub trait Pipeline {
    /// Load all weights and reusable state once (fatal on failure).
    fn prepare(&mut self, args: &Args, device: &Device, dtype: DType) -> Result<()>;
    /// Produce a single image reusing the prepared state. May be called many times.
    fn generate(&self, args: &Args) -> Result<()>;
}

pub fn for_model(model: Model) -> Box<dyn Pipeline> {
    match model {
        Model::Schnell | Model::Dev | Model::SchnellGguf | Model::DevGguf => {
            Box::new(crate::flux::FluxPipeline::default())
        }
        Model::Araminta => Box::new(crate::sdxl::SdxlPipeline::default()),
    }
}
```

The two-phase split lets batch runs (`--seed-range`) load weights once instead of per seed. Device and dtype are stored in the prepared state, not passed to `generate`.

### FLUX pipeline (`src/flux/run.rs`)

- Validates `height % 16 == 0` and `width % 16 == 0`.
- Defaults: height 768, width 1360.
- Default steps: 4 for `schnell`, 50 for `dev`.
- Loads T5-XXL (`mcmonkey/google_t5-v1_1-xxl_encoderonly`) and CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) in **parallel threads** via `std::thread::scope`.
- Full FLUX DiT loads BF16 safetensors; GGUF loads quantized weights from HF `city96` repos or a local `--gguf` path.
- For GGUF, the DiT device is forced to CPU and dtype to F32.
- Guidance tensor is `None` for `schnell` (distilled) and `Some(Tensor::full(flux_guidance))` for `dev`.
- `prepare` loads text embeddings (or reuses the disk cache), DiT, and VAE; per-seed state (latents, img_ids, vec) is rebuilt in `generate`. The DiT stays resident during VAE decode so batch runs reuse it — see the memory gotcha below.
- Latents are cast to F32 before VAE decode (the VAE always runs in F32).
- VAE decode uses F32 and can be offloaded to CPU with `--vae-cpu`.

### SDXL pipeline (`src/sdxl/run.rs`)

- Validates `height % 8 == 0` and `width % 8 == 0`.
- Defaults: height 768, width 1024 (non-standard; SDXL spec is 1024×1024).
- Default steps: 20.
- Model source: `John6666/the-araminta-experiment-fv5-sdxl`, or a local diffusers directory via `--local-model`.
- Schedulers: `euler-a`, `euler-a-karras`, `dpm2m-karras`. Custom Karras schedulers live in `src/schedulers.rs`.
- Loads two text encoders (CLIP-1 ViT-L/14, CLIP-2 ViT-bigG) and concatenates embeddings along hidden dim to 2048.
- Supports LoRA for UNet and both text encoders (`--lora`, `--lora-scale`).
- Uses `sd_config.build_unet(...)` normally, but when LoRA is present constructs a hardcoded `UNet2DConditionModelConfig` because LoRA-loaded weights cannot use the standard `build_unet` path.
- `prepare` loads embeddings (or reuses the disk cache), UNet, and VAE; the UNet stays resident during decode for batch reuse. The scheduler is recreated per `generate` call (`Dpm2mKarrasScheduler` carries mutable per-step state).

### Device selection (`src/device.rs`)

```rust
pub fn pick_device(cpu: bool) -> Device
```

- `--cpu` → `Device::Cpu`
- compiled with `metal` → `Device::new_metal(0)`, fallback to CPU on failure
- compiled with `cuda` → `Device::new_cuda(0)`, fallback to CPU on failure
- otherwise → CPU

`metal` and `cuda` cannot be enabled together — a `compile_error!` guard at the top of `device.rs` rejects the combination at build time.

### Model loading (`src/hub.rs`)

- `fetch(repo, filename)` wraps `hf_hub::ApiRepo::get()` and logs an info message. First run downloads; later runs return cached paths from `~/.cache/huggingface/`.
- `log_model_size(path, label)` prints file size in MB/GB after each weight file loads.

## CLI overview

Defined in `src/cli.rs` (clap derive). Key flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--prompt` | `"A rusty robot walking on a beach"` | |
| `--uncond-prompt` | `""` | Negative prompt, SDXL only |
| `--height` | 768 (FLUX) / 1024 (SDXL) | |
| `--width` | 1360 (FLUX) / 1024 (SDXL) | |
| `--n-steps` | 4 / 50 / 20 | Per model |
| `--seed` | random | |
| `--seed-range` | — | `START-END`, e.g. `0-100` |
| `--output` | `out/out-{seed}-{rand}.png` | |
| `--model` | `schnell` | `schnell`, `dev`, `schnell-gguf`, `dev-gguf`, `araminta` |
| `--guidance-scale` | `7.5` | SDXL only |
| `--flux-guidance` | `3.5` | FLUX dev only |
| `--cpu` | — | Force CPU |
| `--lora` | — | SDXL only, `.safetensors` |
| `--lora-scale` | `1.0` | 0.0–1.0 |
| `--local-model` | — | SDXL only, diffusers dir |
| `--clip-skip` | `1` | SDXL only, 1–12 |
| `--scheduler` | `euler-a` | SDXL only |
| `--gguf` | — | Local FLUX GGUF file |
| `--quantization` | `q8` | `q8` / `q4` |
| `--dtype` | GPU: bf16, CPU: f32 | `f32`, `bf16`, `f16`; GGUF ignores this |
| `--vae-cpu` | — | Decode VAE on CPU |
| `--verbose` | — | Verbose logging |
| `--sequential-te` | — | Load T5 then CLIP sequentially (reduces peak VRAM ~9 GB) |
| `--vae-tile-size` | `0` (disabled) | Tile size in latent px for tiled VAE decode (0 = disabled). Recommended: 64 (SDXL), 128 (FLUX) |
| `--vae-tile-overlap` | `8` | Overlap between VAE tiles in latent pixels |

## Development conventions

- `src/main.rs` denies `clippy::unwrap_used`. Use `?` or `.expect("reason")` with a message.
- Use `cargo fmt --check` and `cargo clippy --features metal` before pushing.
- `// SAFETY:` comments explain `unsafe { VarBuilder::from_mmaped_safetensors(...) }` usage; the underlying files are owned by the HF cache or local model directory and are not modified during inference.
- Prefer minimal, focused changes that match the existing style.
- Default branch is `master`.

## Key gotchas

- **Seed not reproducible on CPU** — `Device::set_seed` only works for GPU. CPU PRNG ignores it.
- **`metal + cuda` is a compile error** — a `compile_error!` guard in `src/device.rs` rejects builds with both GPU features enabled; pick exactly one backend.
- **GGUF models force dtype=F32 and DiT on CPU** — quantized weights dequantize to F32; mixing BF16 tensors with F32 weights fails in matmul. The DiT also runs on CPU to avoid device mismatches and Metal OOM.
- **`--guidance-scale` is SDXL-only** (typical range ~5–10). FLUX dev uses `--flux-guidance` (default 3.5); `schnell` ignores guidance because it is distilled. These are separate flags with different semantics.
- **SDXL default resolution is 768×1024**, not the SDXL-standard 1024×1024. FLUX defaults to 768×1360.
- **LoRA config is hardcoded** — when `--lora` is used, `UNet2DConditionModelConfig` is manually constructed in `sdxl/run.rs` because LoRA-loaded weights cannot use the standard `build_unet` path.
- **`log.jsonl` appends are atomic** — each entry is formatted into a single string and written with one `write_all` on an append-mode file (`O_APPEND` / `FILE_APPEND_DATA`), so parallel runs cannot interleave lines. Do not switch back to `writeln!` on the raw `File` — its formatting adaptor can split a line across multiple syscalls.
- **Batch mode continues on failure** — if one seed fails, the loop proceeds to the next seed.
- **Batch mode keeps weights resident** — `prepare` loads DiT/UNet **and** VAE once; the denoiser stays in memory during VAE decode (tradeoff for fast batch runs). Mitigations on tight memory: `--vae-cpu`, `--vae-tile-size`, GGUF models.
- **Ctrl-C** — first press stops gracefully at the next denoising step / seed boundary (exit code 130, progress bar cleared, no failure log entry); second press exits immediately. Long phases without checkpoints (model load, VAE decode) only respond to the second press.
- **VAE CPU offload** — `--vae-cpu` avoids Metal pool growth / CUDA OOM from VAE intermediate activations; decode is slower (~10–30s for 1024×1024).
- **Embedding cache** — repeated runs with the same prompt skip T5/CLIP loading by caching embeddings to `~/.cache/etch/embeddings/`. Cache key includes prompt, model type, compute dtype, and for SDXL also LoRA info, clip_skip, and the CFG batch layout (`cfg1`/`cfg2` — embeddings are batch-2 when `--guidance-scale > 1`). Anything that changes embedding shape or dtype **must** be added to the key or cached tensors will collide across configurations. Cache is saved in safetensors format.
- **`--sequential-te`** — loads T5-XXL and CLIP sequentially instead of in parallel. Peak VRAM during text encoding drops from ~11 GB to ~1.5 GB in FLUX. Slightly slower startup (~2–4s).
- **Tiled VAE decode** — splits latent into overlapping tiles before VAE decode. Prevents OOM on high-res images (1536×1536+). Adds 10–50% decode overhead depending on resolution. Use `--vae-tile-size 64` (SDXL) or `128` (FLUX).

## Security considerations

- **Network downloads:** model weights are downloaded from HuggingFace Hub via `hf-hub`. Ensure users trust the configured repos (`black-forest-labs/FLUX.1-schnell`, `black-forest-labs/FLUX.1-dev`, `city96` GGUF variants, `openai/clip-vit-large-patch14`, `mcmonkey/google_t5-v1_1-xxl_encoderonly`, `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`, `John6666/the-araminta-experiment-fv5-sdxl`). Gated models (e.g., FLUX.1-dev) require a HuggingFace account/token.
- **Local file paths:** `--gguf`, `--lora`, and `--local-model` accept arbitrary filesystem paths. No additional validation is performed beyond opening the files.
- **Memory-mapped weights:** `VarBuilder::from_mmaped_safetensors` is used with `// SAFETY:` comments. The files must not be modified while mapped.
- **`log.jsonl` concurrency:** entries are appended atomically (single `write_all` on an append-mode file), so parallel runs do not corrupt the log. Keep it that way when modifying `logger.rs`.
- **No sandboxing:** the CLI runs diffusion models locally with full CPU/GPU access. Be cautious with untrusted model files (especially custom GGUF or LoRA weights).
- **SSH deploy key:** `.github/workflows/mirror.yml` uses `secrets.GITVERSE_SSH_KEY` to push to Gitverse. Protect this secret and rotate it if exposed.
