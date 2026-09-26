# etch development conventions

## Branch naming

No "phase" in branch names. Descriptive: `feature/ssd-streaming`, `feature/cache-embeddings`.

## Workflow

1. Show diff/plan before making changes
2. Wait for approval
3. Implement
4. Show results, wait for approval
5. Commit only when told

Never commit autonomously. Give the user the `git commit` command.

## Code style

- `#![deny(clippy::unwrap_used)]` — use `?` or `.expect("msg")`
- `cargo fmt` and `cargo clippy --features metal` before commit
- Rust edition 2024
- Toolchain: stable (rust-toolchain.toml)
- Match existing style: no comments unless asked, same import grouping

## Debugging streaming forward

When a streaming forward pass fails:
1. Add debug logging to the failing function
2. Let the user rebuild and run
3. Read the output to understand the mismatch
4. Fix the issue, show the diff, get approval

Common issues:
- Layer ordering (BTreeMap sorts alphabetically)
- Wrong tensor name suffixes
- Missing weights (filtered out or grouped incorrectly)

## Dependencies

- candle-core/nn/transformers 0.11
- Check API availability before using: `grep -r "pub fn conv2d" ~/.cargo/registry/src/*/candle-nn-0.11.0/src/`
- candle-nn: `Conv2d::new(w,b,config).forward(x)`, `GroupNorm::new(...).forward(x)`, `Linear::new(w,Some(b)).forward(x)`
- candle_nn::ops: `silu`, `softmax`, `layer_norm` (last dim only, eps: f32)

## GPU

- Primary target: Apple Silicon Metal
- Default dtype: BF16 on GPU, F32 on CPU
- Build: `cargo build --release --features metal`
