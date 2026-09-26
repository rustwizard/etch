# Что можно улучшить в `etch`

---

## 🏗 Архитектура и структура

- [ ] **Вынести ручной конфиг SDXL UNet при LoRA**  
  `unet_2d::UNet2DConditionModelConfig` создаётся хардкодом в `sdxl/run.rs`. Лучше загружать из `config.json` диффузеров.

- [ ] **Согласовать WIP-код в `feature/ssd-streaming`**  
  `src/sdxl/forward_streaming.rs` и `unet_structure.rs` ожидают именованные веса (`&[(String, Tensor)]`), а закоммиченный `LayerStream::next_layer()` возвращает `Vec<Tensor>` — 23 ошибки компиляции. Файлы намеренно не подключены к `sdxl/mod.rs` (см. коммит `e0f8240`).

---

## 🛡 Надёжность

- [x] **Безопасный `append` в `log.jsonl`** — сделано в PR #24.  
  Строка целиком форматируется и пишется одним `write_all` в файл с `append(true)`: `O_APPEND` (POSIX) / `FILE_APPEND_DATA` (Windows) гарантируют атомарность. Явная блокировка (`fd-lock`/`fs2`) не понадобилась.

- [x] **Graceful shutdown по `Ctrl+C`** — сделано в PR #25.  
  Модуль `src/signals.rs` (крейт `ctrlc`): первое нажатие останавливает генерацию на границе шага денойзинга и между сидами (код выхода 130, прерванный прогон не логируется как failed), второе — немедленный выход.

---

## 🧪 Тестирование и инфраструктура

- [ ] **Добавить unit-тесты**
  - `te_lora_base_to_weight_key` и `ldm_lora_base_to_unet_key` (в `lora.rs`)
  - `build_karras_schedule` (монотонность, граничные значения) (в `schedulers.rs`)
  - `greedy_tokenize` (в `lora.rs`)

- [ ] **Добавить интеграционный smoke-test**  
  Запуск с `--cpu --n-steps 1` для проверки, что пайплайн не падает на старте.

---

## 🔧 Мелочи

- [x] **Явный выбор Metal/CUDA через CLI** (`--metal` / `--cuda`) — потеряло актуальность после PR #26.  
  `metal + cuda` теперь `compile_error!`, бинарь всегда собирается ровно с одним backend'ом, выбирать в рантайме нечего.

- [ ] **doc-комментарии для `apply_lora` / `apply_te_lora`** в `lora.rs` — семантика возвращаемого HashMap и side effects не очевидны.

- [ ] **Ссылки на paper для констант в `schedulers.rs`** — `BETA_START=0.00085`, `BETA_END=0.012`, `TRAIN_STEPS=1000` без пояснения откуда.

- [ ] **Высота/ширина SDXL по умолчанию** — `height=768, width=1024` нестандартны (SDXL spec: 1024×1024). Нужен комментарий или смена дефолтов.

- [x] **`compile_error!` для `metal + cuda`** — сделано в PR #26.  
  Guard в `src/device.rs` с внятным сообщением; раньше Metal молча побеждал через порядок `cfg`.

- [ ] **README: добавить раздел для разработчиков**  
  `cargo test`, `cargo clippy`, как добавить новую модель.

---

## ✅ Недавние улучшения (этот цикл)

- PR #21 — **candle 0.10.2 → 0.11.0**: Metal concurrent dispatch, ускоренный gemv, фиксы SDPA/RMSNorm/readback-race, безопасный GGUF-лоадер.
- PR #22 — **reuse весов в батч-режиме**: `Pipeline` разделён на `prepare` (загрузка один раз) / `generate` (на каждый сид); заодно фикс BF16→F32 каста латента перед VAE decode в FLUX.
- PR #23 — **ключ кэша эмбеддингов**: добавлены режим CFG (`cfg1`/`cfg2`) и dtype — раньше прогон с `--guidance-scale 1.0` и `7.5` на одном промпте делили один кэш (падение на `chunk(2, 0)`).
- PR #27 — **кэш feather-масок в tiled VAE decode**: маска зависит только от `(out_h, out_w, feather-флаги)`, внутренние тайлы переиспользуют одну вместо пересчёта и re-upload на каждый тайл.
