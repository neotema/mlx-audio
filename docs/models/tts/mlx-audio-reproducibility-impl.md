# mlx-audio VoxCPM2 Reproducibility — Implementation Spec

This document specifies the **upstream (mlx-audio)** and **consumer (pf-voxcpm-generator)** work required to make VoxCPM2 voice-design synthesis reproducible via `seed` and controllable via diffusion `temperature`.

**Status (2026-06):**

| Layer | seed | temperature |
|-------|------|-------------|
| pf-voxcpm-generator | wired (HTTP payload + run reports) | wired (HTTP payload + run reports) |
| mlx-audio server | **not implemented** | schema exists; **not forwarded to VoxCPM2 DiT** |

Until mlx-audio implements the upstream changes below, sending `seed` / `temperature` is **forward-compatible** but has **no effect** on audio output.

Related: `docs/voice-design-prompt-handoff.md` §6.

---

## Problem

VoxCPM2 zero-shot output is stochastic. Product mitigations today:

- `voice_design.n_best.voxcpm2` (default 3): parallel samples + QC rank
- Clone mode stabilizes timbre after a seed ref exists

For **reproducible A/B** (cfg/steps/prompt comparisons), we need:

1. **`seed`** — identical HTTP request → identical WAV (given same mlx-audio + model revision)
2. **`temperature`** — diffusion noise scale in `feat_decoder.sample()` (DiT Euler solver)

---

## Current mlx-audio behavior (Blaizzy/mlx-audio `main`, verified 2026-06)

### HTTP API

`POST /v1/audio/speech` accepts `SpeechRequest`:

```python
class SpeechRequest(BaseModel):
    model: str
    input: str
    instruct: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    temperature: float | None = 0.7   # present
    # seed: NOT present
    ...
```

`TTSExecutionAdapter.run_serial()` passes `temperature` into `model.generate(...)`.

### VoxCPM2 model

File: `mlx_audio/tts/models/voxcpm2/voxcpm2.py`

- `generate(..., **kwargs)` accepts `temperature` in kwargs but **does not use it**
- Calls `self.feat_decoder.sample(..., cfg_value=cfg_value)` without `temperature`

File: `mlx_audio/tts/models/voxcpm2/dit.py`

```python
def sample(self, mu, n_timesteps, patch_size, cond, temperature=1.0, cfg_value=1.0):
    z = mx.random.normal((B, self.in_channels, T)) * temperature
    ...
```

- Initial noise uses `mx.random.normal` with **no seed control**
- `temperature` defaults to `1.0` and is never overridden from HTTP

### `voxcpm` nested config

pf-voxcpm-generator sends:

```json
{
  "model": "mlx-community/VoxCPM2-8bit",
  "input": "...",
  "instruct": "...",
  "temperature": 1.0,
  "seed": 42,
  "voxcpm": {
    "cfg_value": 2.2,
    "inference_timesteps": 12,
    "temperature": 1.0,
    "seed": 42
  }
}
```

Upstream `SpeechRequest` **does not declare** `voxcpm` or `seed`; Pydantic v2 ignores unknown fields by default. Local deployments may patch this — production should treat upstream support as explicit.

---

## Required mlx-audio changes

Target repo: [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)

### 1. Extend `SpeechRequest` (`mlx_audio/server.py`)

```python
class VoxCPMOptions(BaseModel):
    cfg_value: float | None = None
    inference_timesteps: int | None = None
    temperature: float | None = None
    seed: int | None = None

class SpeechRequest(BaseModel):
    ...
    seed: int | None = None
    voxcpm: VoxCPMOptions | None = None
```

Resolution order inside adapter (recommended):

```text
effective_seed       = request.seed or (request.voxcpm.seed if request.voxcpm else None)
effective_temperature = request.temperature or (request.voxcpm.temperature if request.voxcpm else None)
effective_cfg         = request.voxcpm.cfg_value if request.voxcpm else None
effective_timesteps   = request.voxcpm.inference_timesteps if request.voxcpm else None
```

### 2. Seed MLX RNG before generation (`TTSExecutionAdapter.run_serial`)

```python
import mlx.core as mx

seed = effective_seed
if seed is not None:
    mx.random.seed(int(seed))

for result in model.generate(speech_request.input, **generate_kwargs):
    ...
```

**Notes:**

- Call `mx.random.seed()` immediately before each `generate()` invocation (serial and batch paths).
- Document that seed affects **DiT initial noise** and any other `mx.random` draws in the forward path.
- Batch endpoint: either reject mixed seeds or seed per sequence index (`seed + i`).

### 3. Forward `temperature` and VoxCPM knobs in `generate_kwargs`

In `TTSExecutionAdapter.run_serial`, merge VoxCPM-specific kwargs when model is VoxCPM2:

```python
generate_kwargs = {
    ...
    "temperature": effective_temperature,
    "inference_timesteps": effective_timesteps,
    "cfg_value": effective_cfg,
    "seed": effective_seed,
}
generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
```

Filter with `inspect.signature(model.generate).parameters` (existing pattern).

### 4. Plumb through `VoxCPM2.generate()` (`voxcpm2.py`)

Add explicit parameters (keep `**kwargs` compat):

```python
def generate(
    self,
    text: str,
    ...
    inference_timesteps: int = 10,
    cfg_value: float = 2.0,
    temperature: float = 1.0,
    seed: int | None = None,
    ...
):
    if seed is not None:
        mx.random.seed(int(seed))
    ...
    pred_feat = self.feat_decoder.sample(
        mu=dit_h,
        n_timesteps=inference_timesteps,
        patch_size=self.patch_size,
        cond=cond_in,
        cfg_value=cfg_value,
        temperature=temperature,
    )
```

Apply the same in the autoregressive loop (each DiT call uses the same `temperature`; re-seeding per step is **not** recommended).

### 5. Tests (mlx-audio)

Add to `mlx_audio/tts/tests/test_voxcpm_integration.py` or new `test_voxcpm2_reproducibility.py`:

```python
def test_voxcpm2_same_seed_same_audio():
    model = load_model("mlx-community/VoxCPM2-8bit")
    kwargs = dict(text="测试。", instruct="Male, mid-twenties.", seed=123, temperature=1.0)
    a = next(model.generate(**kwargs)).audio
    b = next(model.generate(**kwargs)).audio
    assert np.allclose(a, b)

def test_voxcpm2_different_seed_different_audio():
    ...
```

Add server test: `POST /v1/audio/speech` with `seed` returns identical bytes on repeat.

### 6. Optional: LM sampling temperature

`SpeechRequest.temperature` is also used by autoregressive TTS models for token sampling. For VoxCPM2, scope **`temperature` to DiT noise only** and document in API docs to avoid breaking Kokoro/Qwen adapters.

---

## pf-voxcpm-generator (implemented)

### Wire format

`utils/voxcpm2_voice_design.build_speech_payload()` sends:

| Field | Location | When |
|-------|----------|------|
| `temperature` | top-level + `voxcpm.temperature` | when set |
| `seed` | top-level + `voxcpm.seed` | when set |
| `cfg_value` | `voxcpm.cfg_value` | existing |
| `inference_timesteps` | `voxcpm.inference_timesteps` | existing |

### Synthesis params

`VoiceDesignSynthesisParams` fields:

```python
temperature: float | None = None
seed: int | None = None
```

Stored in run reports under `synthesis_params` and request records.

### Config (`config/app.defaults.yaml`)

```yaml
voice_design:
  # voxcpm2_temperature: 1.0   # optional; omit = mlx-audio default
  # voxcpm2_seed: 42           # optional fixed seed for all calls
  deterministic_voxcpm2_seed: false
  voxcpm2_seed_base: 0
```

When `deterministic_voxcpm2_seed: true`, seeds derive from `(cast_key, candidate_id, attempt, n_best_index)` via `utils/voxcpm2_reproducibility.derive_voxcpm2_seed()` — each n-best candidate gets a **distinct but stable** seed.

### Key modules

| Module | Role |
|--------|------|
| `utils/voxcpm2_reproducibility.py` | seed derivation + settings resolution |
| `utils/voxcpm2_voice_design.py` | HTTP payload |
| `utils/voice_design_backend.py` | design/dry-ref synthesis |
| `scripts/design_cast_refs.py` | n-best per-candidate seeds |

---

## Verification checklist

After mlx-audio patch is deployed:

```bash
# 1. Same seed → identical WAV
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/VoxCPM2-8bit",
    "input": "滨江市，凌晨两点三十分。",
    "instruct": "Male, mid-twenties. Low deep voice...",
    "seed": 42,
    "temperature": 1.0,
    "voxcpm": {"cfg_value": 2.2, "inference_timesteps": 12},
    "response_format": "wav"
  }' --output a.wav

# repeat → b.wav; cmp a.wav b.wav should be silent

# 2. Product path with deterministic n-best
# config: voice_design.deterministic_voxcpm2_seed: true
uv run python scripts/run_episode_voice_design.py run \
  --episode output/sample_mystery_novel/episodes/episode_01 \
  --roles narrator --design-backend voxcpm2
# Inspect run report: synthesis_params.seed present per n-best candidate
```

---

## PR submission template (mlx-audio)

**Title:** VoxCPM2: forward seed + temperature to DiT sampler; accept voxcpm nested config

**Summary:**

- Add `seed` and `voxcpm` to `SpeechRequest`
- Seed `mx.random` before VoxCPM2 generation when `seed` is set
- Pass `temperature` to `feat_decoder.sample()`
- Integration tests for reproducibility

**Breaking changes:** None (all new fields optional)

**References:**

- pf-voxcpm-generator consumer: `utils/voxcpm2_voice_design.py`
- OpenBMB VoxCPM2 docs: https://voxcpm.readthedocs.io/en/latest/usage_guide.html
