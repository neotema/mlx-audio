# VoxCPM2 Voice Design

Voice design sets speaker identity via `instruct` (HTTP: `instructions`). No reference audio.

Model prepends `({instruct}){text}` internally. Write `instruct` as a single English clause chain; keep `input` as the script only.

## Six Dimensions

Define each dimension before writing the prompt. Omit nothing that affects casting.

| Dimension | Covers | Examples |
|-----------|--------|------------|
| Gender | Sex presentation | male, female |
| Age | Life stage | mid-twenties, elderly |
| Timbre | Pitch and resonance | low, deep, chest resonance, vocal fry |
| Pace | Speed and rhythm | moderate, slow, natural pauses |
| Mood | Delivery emotion | restrained, tense, calm with underlying anxiety |
| Accent | Language and register | colloquial Mandarin, unpolished, non-broadcast |

**Rules**

- One clause per dimension, joined in table order.
- Use concrete acoustic words (pitch, pace, resonance), not role labels alone.
- Mood = how it sounds, not plot summary.
- Accent = dialect/register, not character biography.
- Script emotion lives in `input`, not `instruct`.

## Prompt Template

```
{Gender}, {age}. {Timbre}. {Pace}. {Mood}. {Accent}, {scene/use-case if needed}.
```

## Example: Suspense Radio Drama Lead

**Brief:** male lead, 20s, deep, everyday accent, mystery radio drama.

| Dimension | Setting |
|-----------|---------|
| Gender | Male |
| Age | Mid-twenties |
| Timbre | Low, deep, warm chest resonance, slight vocal fry |
| Pace | Moderate-slow, natural pauses |
| Mood | Restrained suspense; calm surface, underlying tension |
| Accent | Colloquial Mandarin, unpolished, non-broadcast; late-night solo narration |

**`instructions`**

```
Male, mid-twenties. Low deep voice with warm chest resonance and slight vocal fry. Moderate slow conversational pace with natural pauses. Restrained suspenseful mood, calm surface with underlying tension. Everyday colloquial Mandarin, unpolished and non-broadcast, like a real person narrating alone late at night in a mystery radio drama.
```

**`input`**

```
那天晚上，我照例在十一点半关灯。楼道里忽然传来脚步声——很轻，但每一步都像是踩在我后颈上。我屏住呼吸，对着门缝问了一句：谁？外面停了半秒，有人用和我一模一样的声音回答：是我。
```

### HTTP

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/VoxCPM2-8bit",
    "instructions": "Male, mid-twenties. Low deep voice with warm chest resonance and slight vocal fry. Moderate slow conversational pace with natural pauses. Restrained suspenseful mood, calm surface with underlying tension. Everyday colloquial Mandarin, unpolished and non-broadcast, like a real person narrating alone late at night in a mystery radio drama.",
    "input": "那天晚上，我照例在十一点半关灯。",
    "response_format": "wav",
    "inference_timesteps": 12,
    "cfg_value": 2.2
  }' --output out.wav
```

### Python

```python
from mlx_audio.tts.utils import load

model = load("mlx-community/VoxCPM2-8bit")
instruct = (
    "Male, mid-twenties. Low deep voice with warm chest resonance and slight vocal fry. "
    "Moderate slow conversational pace with natural pauses. "
    "Restrained suspenseful mood, calm surface with underlying tension. "
    "Everyday colloquial Mandarin, unpolished and non-broadcast, "
    "like a real person narrating alone late at night in a mystery radio drama."
)
result = next(model.generate(text="那天晚上，我照例在十一点半关灯。", instruct=instruct))
```

## Generation Knobs

| Parameter | Voice design note |
|-----------|-------------------|
| `inference_timesteps` | 10 default; 12 for steadier quality |
| `cfg_value` | ≥ 2.0; 2.0–2.5 typical |
| `warmup_patches` | 0–1; model may lower automatically when `instruct` is set |

## Tuning

| Goal | Adjust |
|------|--------|
| More pressure | Mood: stronger tension; Pace: slower + longer pauses |
| More everyday | Accent: more colloquial; Timbre: less resonance |
| Younger | Age down; Timbre: lighter, less fry |
| More broadcast | Accent: remove unpolished / non-broadcast |

Change one dimension per iteration; compare outputs.
