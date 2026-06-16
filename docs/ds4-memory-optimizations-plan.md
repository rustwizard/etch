# Объединённый план: экономия памяти в etch (DS4-подход)

## Цель

Позволить `etch` запускать большие diffusion-модели (FLUX, SDXL) на машинах с
ограниченной памятью, адаптировав идеи `antirez/ds4` под dense diffusion
трансформеры.

Приоритетные конфигурации:
- **FLUX + Metal (Apple Silicon)**
- **SDXL + Metal/CUDA**

## Почему не прямой порт DS4

`ds4` использует:
- **MoE sparsity** — только ~6/256+ экспертов активны на токен.
- **Disk KV-cache** — кэш промежуточных состояний LLM.
- **SSD streaming** — неактивные MoE-эксперты хранятся на SSD.

В `etch` другая картина:
- Модели **dense** — все веса используются на каждом denoising step.
- Нет KV-cache.
- Основная память уходит на text encoders, DiT/UNet и VAE.
- Candle 0.10 не даёт 2-bit кастомную квантизацию; для FLUX DiT уже есть
  GGUF (Q8/Q4), для SDXL/encoders — нет готового quantized-пути.

Поэтому адаптируем DS4-подходы:
1. Кэшировать тяжёлые text embeddings (аналог disk KV-cache).
2. Разгружать слои DiT/UNet на CPU/SSD (аналог SSD streaming).
3. По возможности распределять слои по хостам (distributed).

---

## Общие принципы

- Каждая фаза — отдельный mergeable milestone.
- Сохраняем `#![deny(clippy::unwrap_used)]`.
- `cargo fmt --check` и `cargo clippy --features metal` перед коммитом.
- Smoke test с `--n-steps 1` для каждой фазы.
- Обновляем `AGENTS.md` при новых флагах.

---

## Phase 1: Conservative — disk cache эмбеддингов + sequential text encoders + tiled VAE

**Цель:** быстро снизить пиковую память и ускорить повторные запуски.

### 1.1 Core embedding cache

Создать `src/cache.rs`:

```rust
pub struct CacheKey {
    pub prompt: String,
    pub uncond_prompt: Option<String>,
    pub model: String,
    pub clip_skip: Option<usize>,
    pub lora: Option<(String, f64)>, // hash of LoRA file + scale
}

pub struct EmbeddingCache {
    root: PathBuf,
}
```

- Ключ — `sha256` от нормализованных полей.
- Хранение в `safetensors` (`~/.cache/etch/embeddings/` или `--cache-embeddings-dir`).
- API: `get(key, device) -> Option<HashMap<String, Tensor>>`, `set(key, tensors)`.

**Изменения:** `src/cache.rs`, регистрация в `main.rs`.

### 1.2 Интеграция в FLUX

В `src/flux/run.rs`:

1. Сформировать `CacheKey`.
2. При hit — загрузить `t5_emb`, `clip_emb` с диска, пропустить загрузку encoders.
3. При miss — загрузить T5/CLIP, посчитать, сохранить в кэш.

**Критерий:** повторный запуск с тем же промптом не логирует
`Fetching model.safetensors` для T5/CLIP.

**Ожидаемый эффект:**

| Scenario | Time spent on text encoders | Notes |
|----------|----------------------------|-------|
| Cache miss (cold) | 20–30 s | Загрузка T5 (~9.5 GB) + CLIP (~1.2 GB) с SSD/HF cache + forward |
| Cache hit (warm) | <1 s | Чтение 2 small safetensors с диска: T5 emb (~4 MB) + CLIP emb (~1 MB) |
| Batch / `--seed-range` same prompt | ~20–30 s saved per image | Особенно выигрышно при генерации нескольких сидов |

Для SDXL dual CLIP экономия аналогичная: ViT-L/14 + ViT-bigG в сумме
~3.5–4 GB, загрузка и forward занимают 10–20 s.

### 1.3 Интеграция в SDXL

В `src/sdxl/run.rs` аналогично для dual CLIP (`emb1`, `emb2`).
Хеш LoRA-файла входит в ключ.

### 1.4 Последовательная загрузка text encoders

Добавить флаг `--sequential-te`.

В FLUX:
- Загрузить T5, посчитать `t5_emb`, сохранить, `drop(model)`.
- Загрузить CLIP, посчитать `clip_emb`, сохранить.
- Убрать `std::thread::scope` при включённом флаге.

**Ожидаемый эффект:**

| Metric | Parallel TE (default) | Sequential TE | Saving |
|--------|----------------------|---------------|--------|
| Peak VRAM during text encoding | ~10.7 GB (T5 ~9.5 GB + CLIP ~1.2 GB) | ~1.5–2.0 GB (CLIP only + activations) | **~8–9 GB** |
| Wall time (first run) | baseline | +2–4 s (T5 unload + CLIP load) | slightly slower |
| Wall time (cached run) | baseline | same as cached | no change |

T5-XXL занимает ~9.5 GB в BF16, CLIP ViT-L/14 — ~1.2 GB. Последовательная
загрузка снимает основной пик, освобождая VRAM для DiT/UNet.

### 1.5 Tiled VAE decode

Создать `src/vae_tiling.rs`:

```rust
pub fn tiled_decode(
    vae: &AutoEncoder,
    latent: &Tensor,
    tile_size: usize,
    overlap: usize,
) -> Result<Tensor>
```

- Разбивает latent на перекрывающиеся плитки.
- Декодирует каждую.
- Склеивает с feathering в overlap-зоне.

Флаги: `--vae-tile-size`, `--vae-tile-overlap`.
Интеграция в `src/flux/run.rs` и `src/sdxl/run.rs`.

**Критерий:** на 1024×1024 tiled decode даёт изображение, визуально
совпадающее с обычным decode.

**Ожидаемый эффект:**

| Resolution | Regular VAE | Tiled VAE | Overhead | Notes |
|------------|-------------|-----------|----------|-------|
| 512×512 | baseline | +5–10% | small | Мало выгоды |
| 1024×1024 | baseline | +10–30% | moderate | Предотвращает пик VRAM |
| 1536×1536 | may OOM | +20–50% | significant | Позволяет вообще декодировать |
| 2048+ | likely OOM | slower but works | — | Без tiling невозможно |

Overhead возникает из-за overlap-зон, которые декодируются дважды, и
склейки плиток. На малых разрешениях выгода незначительна; основное
применение — 1536×1536 и выше, где обычный decode либо OOM, либо
сильно растёт Metal memory pool.

### Phase 1 Performance Targets

| Optimization | What it saves | Target |
|--------------|---------------|--------|
| `--sequential-te` | Peak VRAM during text encoding | **~8–9 GB** (с ~10.7 GB до ~1.5–2.0 GB) |
| Embedding cache | Wall time on repeat prompts | **20–30 s** per image with same prompt |
| Tiled VAE 1024×1024 | Peak VAE VRAM | prevents pool growth; **< 30% slowdown** |
| Tiled VAE 1536×1536 | Ability to decode | no OOM; **20–50% slowdown acceptable** |

### Критерии завершения Phase 1

- `cargo fmt --check` и `cargo clippy --features metal` чистые.
- Smoke test: `--model schnell --prompt "a cat" --n-steps 1`.
- Повторный запуск использует кэш.
- `--sequential-te` снижает peak VRAM на ~8–9 GB.
- Tiled VAE работает на 1024×1024 и 1536×1536.
- Обновить `AGENTS.md`.

---

## Phase 2: Aggressive — layer-wise streaming / offloading DiT/UNet

**Цель:** запускать модели, которые не влезают в VRAM, ценой скорости.

Эта фаза объединяет:
- **новый план:** `Offloadable` trait и `CpuOffload<T>` wrapper,
- **старый план:** `OffsetIndex`, `AsyncReader`, `LayerStream` с double-buffering
  для disk-target, `LayerSpec` / `WeightRef`, разбивку слоёв.

### 2.1 Общий интерфейс (`src/offload.rs`)

```rust
pub enum OffloadTarget {
    Cpu,
    Disk,
}

pub trait Offloadable {
    fn to_compute(&mut self, device: &Device) -> Result<()>;
    fn to_offload(&mut self, device: &Device) -> Result<()>;
}
```

Два независимых backend'а:
- **Disk streaming** (`src/ssd/`, `--offload-target disk`) — читает слои с SSD
  по запросу, почти не использует CPU RAM. Основной путь Phase 2.
- **CPU offloading** (`src/cpu_offload.rs`, `--offload-target cpu`) — держит все
  веса в CPU RAM и клонирует активный слой на GPU. Альтернатива, если CPU RAM
  достаточна и хочется избежать SSD I/O.

### 2.2 Safetensors Offset Index (`src/ssd/offset_index.rs`)

Из старого плана:

```rust
pub struct TensorMeta {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub offset_bytes: u64,
    pub size_bytes: u64,
}

pub struct OffsetIndex {
    file_path: PathBuf,
    data_start: u64,
    tensors: HashMap<String, TensorMeta>,
}

impl OffsetIndex {
    pub fn from_file(path: &Path) -> Result<Self>;
    pub fn read_tensor(&self, name: &str, device: &Device) -> Result<Tensor>;
}
```

- Парсим safetensors header, строим map name → offset/size.
- Читаем отдельные тензоры через `pread` без загрузки всего файла.

### 2.3 Async I/O and Staging Buffer (`src/ssd/async_reader.rs`)

Из старого плана:

```rust
pub enum IoRequest {
    LoadLayer { layer_id: usize, tensor_metas: Vec<(String, TensorMeta)> },
    Shutdown,
}

pub enum IoResult {
    LayerReady { layer_id: usize, buffers: Vec<(String, Vec<u8>)> },
    Error(String),
}

pub struct AsyncReader {
    tx: SyncSender<IoRequest>,
    rx: Receiver<IoResult>,
    worker: JoinHandle<()>,
}
```

- Bounded `sync_channel` для backpressure.
- Фоновый thread читает слой целиком через `pread`.

### 2.4 LayerStream (`src/ssd/layer_stream.rs`)

Из старого плана:

```rust
pub struct LayerStream {
    reader: AsyncReader,
    index: Arc<OffsetIndex>,
    device: Device,
    dtype: DType,
    layer_specs: Vec<LayerSpec>,
    total_layers: usize,
    current_layer_id: usize,
}

impl LayerStream {
    pub fn begin(&mut self);
    pub fn next_layer(&mut self) -> Result<Vec<Tensor>>;
    pub fn drop_layer(&self, tensors: Vec<Tensor>);
}
```

- `begin()` запускает загрузку слоя 0.
- `next_layer()` ждёт текущий слой, запускает prefetch следующего.
- `drop_layer()` освобождает GPU-память.

### 2.5 Weight Descriptor and Layer Structure (`src/ssd/weight_descriptor.rs`)

Из старого плана:

```rust
pub struct WeightRef {
    pub name: String,
    pub description: &'static str,
}

pub struct LayerSpec {
    pub id: usize,
    pub name: String,
    pub weights: Vec<WeightRef>,
    pub estimated_bytes: u64,
}

pub trait LayerStructure {
    fn layer_specs(&self) -> Vec<LayerSpec>;
}
```

- Описывает, какие тензоры к какому слою принадлежат.
- Нужно для FLUX DiT и SDXL UNet.

### 2.6 SDXL UNet layer breakdown

Из старого плана. ~80 слоёв:

| Layer ID | Name | Type | Approx Size (BF16) |
|----------|------|------|--------------------|
| 0 | `conv_in` | Conv2d 4→320 | ~0.01 MB |
| 1–2 | `down_0_b0/b1` | ResNetBlock (320ch) | ~1 MB each |
| 3 | `down_0_downsample` | Conv2d | ~1 MB |
| 4–5 | `down_1_b0/b1` | ResNetBlock (640ch) | ~4 MB each |
| 6–9 | `down_1_t*` | BasicTransformer (640ch) | ~10 MB each |
| 10 | `down_1_downsample` | Conv2d | ~4 MB |
| 11–22 | `down_2_*` | ResNetBlock + Transformer (1280ch) | ~15–25 MB |
| 23 | `mid_resnet` | ResNetBlock (1280ch) | ~15 MB |
| 24–33 | `mid_t*` | BasicTransformer (1280ch) | ~25 MB × 10 |
| 34–78 | `up_*` | ResNetBlock + Transformer | ~1–25 MB |
| 79 | `conv_out` | Conv2d 320→4 | ~0.001 MB |

**Largest layer:** ~25 MB.  
**Peak VRAM:** ~2 GB (vs ~8 GB).

### 2.7 FLUX DiT layer breakdown

Из старого плана. ~62 слоя:

| Layer ID | Name | Size (BF16) | Size (F32/GGUF) |
|----------|------|-------------|-----------------|
| 0 | `img_in` | ~0.8 MB | ~1.5 MB |
| 1 | `txt_in` | ~25 MB | ~50 MB |
| 2 | `time_in` | ~18 MB | ~36 MB |
| 3 | `vector_in` | ~4.5 MB | ~9 MB |
| 4 | `guidance_in` (dev) | ~18 MB | ~36 MB |
| 5–23 | `double_blocks.0`–`.18` | ~380 MB each | ~760 MB each |
| 24–61 | `single_blocks.0`–`.37` | ~200 MB each | ~400 MB each |
| 62 | `final_layer` | ~18 MB | ~36 MB |

**Largest layer:** ~380 MB BF16.  
**Peak VRAM:** ~2.5 GB (vs ~25 GB).

### 2.8 Modified forward pass — SDXL

Из старого плана. Реализовать `sdxl_forward_streaming()` в
`src/sdxl/forward_streaming.rs`:

```rust
conv_in(input) →
for each down_block: resnets → attentions → downsample →
mid_block: resnet → attention →
for each up_block: resnets → attentions → upsample (with skip connections) →
conv_out → output
```

- Skip connections (latents) остаются в GPU — мало весят (~13 MB на 1024×1024).
- Используем `LayerStream` для загрузки/выгрузки весов каждого слоя.

### 2.9 Modified forward pass — FLUX

Из старого плана. Реализовать `flux_forward_streaming()` в
`src/flux/forward_streaming.rs`:

```rust
// Pre-loaded constant layers (small, stay resident):
let (mut img, mut txt, y_vec, t_vec, guidance_vec, pe) = load_input_layers(stream)?;

// Double stream blocks
for i in 0..config.depth {
    let w = stream.next_layer()?;
    let (img_out, txt_out) = double_stream_block_forward(&img, &txt, &y_vec, &t_vec, &pe, &w)?;
    img = (img + img_out)?;
    txt = (txt + txt_out)?;
    drop(w);
}

// Single stream blocks
for _ in 0..config.depth_single_blocks {
    let w = stream.next_layer()?;
    let img_out = single_stream_block_forward(&img, &y_vec, &t_vec, &pe, &w)?;
    img = (img + img_out)?;
    drop(w);
}

// Final layer
let final_w = stream.next_layer()?;
let output = final_layer_forward(&img, &txt, &y_vec, &t_vec, &final_w)?;
```

- Small input layers (~65 MB BF16) остаются резидентными.
- Streaming идёт по DoubleStreamBlock и SingleStreamBlock.

### 2.10 CPU offloading path (`src/cpu_offload.rs`)

Отдельная стратегия, не смешанная с disk streaming.

Для `--offload-target cpu`:

- Загружаем safetensors в CPU RAM (`candle_core::safetensors::load` на CPU).
- Строим модель через `VarBuilder::from_tensors`.
- `CpuOffload<T>` оборачивает каждый блок:
  - `to_compute()` — клонирует веса блока на GPU.
  - `forward()` — выполняет forward на GPU.
  - `to_offload()` — дропает GPU-копию.

**Trade-off:**
- **Быстрее**, чем disk streaming, потому что нет повторного чтения с SSD
  на каждый denoising step.
- **Требует больше CPU RAM:** ~7 GB для SDXL, ~24 GB для FLUX BF16
  (или ~12 GB/48 GB при F32).
- **Подходит**, если модель не влезает в VRAM, но влезает в RAM.

**Когда использовать:**
- Metal Mac с 36–64 GB unified memory — FLUX/SDXL в RAM, активный слой на GPU.
- CUDA машина с 32+ GB системной RAM, но 8–16 GB VRAM.

**Когда НЕ использовать:**
- Машины, где даже CPU RAM меньше размера модели — тогда только disk streaming.

### 2.11 CLI flags

В `src/cli.rs`:

```rust
/// Enable layer-wise offloading / streaming.
/// Alias for --offload-target disk.
#[arg(long)]
pub ssd_streaming: bool,

/// Offload target: cpu or disk.
#[arg(long, value_enum)]
pub offload_target: Option<OffloadTarget>,

/// CPU RAM / disk buffer for prefetching layers (default: 4 GB)
#[arg(long, default_value = "4")]
pub ssd_buffer_gb: usize,

/// Disable async prefetch
#[arg(long)]
pub ssd_no_prefetch: bool,

/// Keep small input layers resident (default: true)
#[arg(long, default_value = "true")]
pub ssd_keep_resident: bool,
```

### 2.12 Интеграция в pipeline

В `src/flux/run.rs` и `src/sdxl/run.rs`:

```rust
let noise_pred = match args.offload_target {
    Some(OffloadTarget::Disk) => {
        forward_streaming_disk(..., &mut layer_stream)?
    }
    Some(OffloadTarget::Cpu) => {
        forward_streaming_cpu(..., &mut cpu_offload_model)?
    }
    None => unet.forward(...), // or model.forward(...)
};
```

Два независимых кода forward:
- `forward_streaming_disk()` — использует `LayerStream`, считывает слои с SSD.
- `forward_streaming_cpu()` — использует `CpuOffload<T>`, двигает слои CPU↔GPU.

### Критерии завершения Phase 2

- SDXL araminta 1024×1024, 20 steps, streaming — **VRAM < 3 GB**.
- FLUX schnell 768×1360, 4 steps, streaming — **VRAM < 6 GB**.
- Pixel-identical output: streaming vs non-streaming (same seed).
- Streaming runtime: **< 2×** non-streaming for schnell (4 steps).
- `cargo clippy --features metal` чистый.

---

## Phase 3: Distributed — разделение слоёв по хостам

**Статус:** не включается в активный roadmap. Рассматривать только как
future research, если появится веский use case.

### Почему ROI низкий для diffusion

- В LLM distributed даёт pipeline-параллелизм: машина A считает токен N,
  машина B параллельно считает токен N+1. В diffusion такого нет:
  каждый denoising step требует полного прохода через **все** слои.
- На каждом шаге активации должны пройти все хосты. Даже при 0.5 ms/hop
  на Thunderbolt и 20–50 шагах overhead заметный.
- С offload/streaming на одной машине можно запустить модель, не влезающую
  в VRAM, без сети, без координации, без отказоустойчивости.
- Единственный реальный сценарий distributed — модель не влезает даже в
  суммарную RAM нескольких машин. Но для FLUX/SDXL это крайний edge case.

### Если всё же понадобится

- `src/distributed/protocol.rs`: `Hello`, `Work`, `Result`, `Heartbeat`.
- `src/bin/etch-worker.rs`: worker-процесс.
- `src/distributed/coordinator.rs`: управление route.
- Флаги: `--dist-role`, `--dist-listen`, `--dist-coordinator`, `--dist-layers`.

---

## Work Plan

### Неделя 1: Phase 1

1. `src/cache.rs` — `EmbeddingCache` + `CacheKey`.
2. Интеграция кэша в `src/flux/run.rs`.
3. Интеграция кэша в `src/sdxl/run.rs`.
4. `--sequential-te` для FLUX.
5. `src/vae_tiling.rs` + флаги + интеграция.
6. Smoke tests, обновление `AGENTS.md`.

### Неделя 2–3: Phase 2 — инфраструктура

1. `src/ssd/offset_index.rs`.
2. `src/ssd/async_reader.rs`.
3. `src/ssd/layer_stream.rs`.
4. `src/ssd/weight_descriptor.rs`.
5. `src/offload.rs` + `CpuOffload<T>`.
6. CLI flags (`--ssd-streaming`, `--offload-target`, etc.).

### Неделя 4: Phase 2 — SDXL streaming

1. `src/sdxl/forward_streaming.rs`.
2. `src/sdxl/offload.rs` / `SdxlUnetStructure`.
3. Интеграция в `src/sdxl/run.rs`.
4. Сравнение output с non-streaming.
5. Prefetch benchmark.

### Неделя 5: Phase 2 — FLUX streaming

1. `src/flux/forward_streaming.rs`.
2. `src/flux/offload.rs` / `FluxStructure`.
3. Интеграция в `src/flux/run.rs`.
4. Benchmark schnell на ограниченной памяти.

### Неделя 6+: Phase 3 — только по явному запросу

1. Пересмотреть ROI для конкретного use case.
2. Если оправдано: distributed protocol, `etch-worker`, coordinator.
3. Локальный и двухмашинный smoke test.

**По умолчанию Phase 3 не реализуется.**

---

## Риски и mitigation

| Риск | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Хрупкость ручного forward | High | High | Интеграционные тесты pixel-identical; фиксировать Candle версию. |
| Слишком медленно для FLUX dev | High | Medium | Документировать как "schnell/emergency mode". |
| macOS mmap issues | Low | Critical | Metal-only path; не использовать CPU streaming. |
| LoRA несовместим со streaming | Medium | Medium | Применять LoRA на CPU до streaming. |
| Prefetch не даёт speedup на Metal | Medium | Medium | Замерять; отключать `--ssd-no-prefetch` если вредит. |
| Разные имена тензоров schnell/dev | Low | Medium | Автоопределение из safetensors header. |

---

## Success Criteria

### Phase 1

- Повторный запуск с тем же промптом использует embedding cache.
- `--sequential-te` снижает peak VRAM при text encoding.
- Tiled VAE даёт визуально идентичный результат на 1024×1024.

### Phase 2

- SDXL streaming: **VRAM < 3 GB** на 1024×1024, 20 steps.
- FLUX schnell streaming: **VRAM < 6 GB** на 768×1360, 4 steps.
- **Pixel-identical** output vs non-streaming (same seed).
- schnell streaming runtime **< 2×** non-streaming.

### Phase 3 (future research)

- Решение о реализации принимается отдельно после Phase 1 и Phase 2.
- Должен быть конкретный use case, где offload на одной машине недостаточен.

---

## Out of Scope

- **GGUF + streaming** — сложнее из-за dequantization + CPU path.
- **Streaming T5-XXL** — используется один раз, проще закэшировать.
- **Streaming VAE** — покрывается tiled VAE и `--vae-cpu`.
- **Automatic memory budget** как в ds4 — пока нет expert cache.
- **Distributed inference over network** — ROI для diffusion низкий,
  см. Phase 3.

---

## Рекомендация

1. **Сначала Phase 1** — быстрые, безопасные победы.
2. **Затем Phase 2 — disk streaming** (`--offload-target disk`).
   Использовать детали старого плана (`OffsetIndex`, `AsyncReader`,
   `LayerStream`, `LayerSpec`) для качественной реализации.
3. **CPU offloading** (`--offload-target cpu`) — добавить как альтернативный
   режим, если появится запрос или если машина имеет достаточно RAM.
4. **Phase 3 (distributed) не делать** — ROI для diffusion низкий.
   Оставить как future research, если появится веский use case.
