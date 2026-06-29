import base64
import io
import threading
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mlx_audio.server import (
    ModelLoadExecutionAdapter,
    _build_tts_generate_kwargs,
    _normalize_tts_generate_kwargs,
    _resolve_voxcpm_params,
    _seed_mlx_rng,
    _validate_speech_request,
    app,
)
from mlx_audio.server import SpeechRequest
from mlx_audio.server_inference import InferenceBroker


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_model_provider():
    with patch(
        "mlx_audio.server.model_provider", new_callable=AsyncMock
    ) as mock_provider:
        mock_provider.load_model = MagicMock()
        yield mock_provider


def test_normalize_tts_generate_kwargs_maps_max_tokens_for_sesame():
    def generate(
        text,
        voice=None,
        max_audio_length_ms: float = 90_000,
        stream: bool = False,
    ):
        del text, voice, max_audio_length_ms, stream

    model = MagicMock()
    model.generate = generate

    kwargs = _normalize_tts_generate_kwargs(
        model,
        {
            "voice": "conversational_a",
            "max_tokens": 200,
            "stream": False,
            "lang_code": "a",
        },
    )

    assert kwargs == {
        "voice": "conversational_a",
        "max_audio_length_ms": 200 * 80,
        "stream": False,
    }
    assert "max_tokens" not in kwargs
    assert "lang_code" not in kwargs


def test_preflight_load_runs_on_broker_worker_thread():
    broker = InferenceBroker(idle_poll_s=0.01)
    broker.register_adapter("load", ModelLoadExecutionAdapter())
    load_threads: list[int] = []

    def tracked_load(model_name: str):
        load_threads.append(threading.get_ident())
        return MagicMock()

    with patch("mlx_audio.server._load_model_for_inference", side_effect=tracked_load):
        handle = broker.submit(
            endpoint_kind="load",
            model_name="test-model",
            payload=None,
        )
        deadline = handle.result_queue.get(timeout=2.0)
        while deadline.kind != "done":
            deadline = handle.result_queue.get(timeout=2.0)

    broker.stop_and_join()
    assert len(load_threads) == 1


def test_tts_preflight_and_inference_share_broker_thread():
    """Regression: model load must not run on asyncio's default thread pool."""
    from fastapi.testclient import TestClient

    from mlx_audio.server import app

    load_threads: list[int] = []
    generate_threads: list[int] = []

    mock_tts = MagicMock()

    def tracked_generate(text, **kwargs):
        del kwargs
        generate_threads.append(threading.get_ident())
        import numpy as np

        from mlx_audio.tests.test_server import MockAudioResult

        yield MockAudioResult(np.zeros(16000, dtype=np.float32), 16000)

    mock_tts.generate = tracked_generate

    with (
        patch("mlx_audio.server.model_provider") as mock_provider,
        patch(
            "mlx_audio.server._load_model_for_inference",
            side_effect=lambda name: (
                load_threads.append(threading.get_ident()) or mock_tts
            ),
        ),
        patch("mlx_audio.server.INFERENCE_BROKER", None),
    ):
        mock_provider.load_model = MagicMock(return_value=mock_tts)
        client = TestClient(app)
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "Marvis-AI/marvis-tts-250m-v0.2-MLX-4bit",
                "input": "短文本测试",
                "voice": "conversational_a",
            },
        )

    assert response.status_code == 200
    assert load_threads, "expected preflight load on broker thread"
    assert generate_threads, "expected generate on broker thread"
    assert load_threads[0] == generate_threads[0]


def _encode_wav_base64(samples: np.ndarray, sample_rate: int = 16000) -> str:
    buffer = io.BytesIO()
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return base64.b64encode(buffer.getvalue()).decode()


def test_build_tts_generate_kwargs_supports_voxcpm2_fields():
    def generate(
        text,
        instruct=None,
        ref_audio=None,
        ref_text=None,
        prompt_text=None,
        prompt_audio=None,
        inference_timesteps=10,
        cfg_value=2.0,
        warmup_patches=0,
        max_tokens=2000,
    ):
        del (
            text,
            instruct,
            ref_audio,
            ref_text,
            prompt_text,
            prompt_audio,
            inference_timesteps,
            cfg_value,
            warmup_patches,
            max_tokens,
        )

    model = MagicMock()
    model.generate = generate
    model.sample_rate = 48000

    ref_b64 = _encode_wav_base64(np.zeros(1600, dtype=np.float32))
    prompt_b64 = _encode_wav_base64(np.ones(1600, dtype=np.float32))

    speech_request = SpeechRequest(
        model="mlx-community/VoxCPM2-8bit",
        input="Continue from here.",
        instructions="A calm narrator voice",
        ref_audio=ref_b64,
        ref_text="reference transcript",
        prompt_text="The previous sentence spoken aloud",
        prompt_audio=prompt_b64,
        inference_timesteps=12,
        cfg_value=2.5,
        warmup_patches=1,
        max_tokens=1800,
    )

    kwargs = _build_tts_generate_kwargs(model, speech_request)

    assert kwargs["instruct"] == "A calm narrator voice"
    assert kwargs["ref_text"] == "reference transcript"
    assert kwargs["prompt_text"] == "The previous sentence spoken aloud"
    assert kwargs["inference_timesteps"] == 12
    assert kwargs["cfg_value"] == 2.5
    assert kwargs["warmup_patches"] == 1
    assert kwargs["max_tokens"] == 1800
    assert isinstance(kwargs["ref_audio"], mx.array)
    assert isinstance(kwargs["prompt_audio"], mx.array)


def test_resolve_voxcpm_params_prefers_top_level():
    speech_request = SpeechRequest(
        model="mlx-community/VoxCPM2-8bit",
        input="Hello",
        seed=11,
        temperature=0.9,
        inference_timesteps=8,
        cfg_value=2.1,
        voxcpm={
            "seed": 22,
            "temperature": 1.5,
            "inference_timesteps": 12,
            "cfg_value": 3.0,
        },
    )

    params = _resolve_voxcpm_params(speech_request)

    assert params == {
        "seed": 11,
        "temperature": 0.9,
        "inference_timesteps": 8,
        "cfg_value": 2.1,
    }


def test_resolve_voxcpm_params_falls_back_to_nested():
    speech_request = SpeechRequest(
        model="mlx-community/VoxCPM2-8bit",
        input="Hello",
        voxcpm={
            "seed": 42,
            "temperature": 1.0,
            "inference_timesteps": 12,
            "cfg_value": 2.2,
        },
    )

    params = _resolve_voxcpm_params(speech_request)

    assert params == {
        "seed": 42,
        "temperature": 1.0,
        "inference_timesteps": 12,
        "cfg_value": 2.2,
    }


def test_build_tts_generate_kwargs_omits_default_temperature_for_voxcpm2():
    def generate(text, **kwargs):
        del text, kwargs

    model = MagicMock()
    model.generate = generate
    model.__class__.__module__ = "mlx_audio.tts.models.voxcpm2.voxcpm2"

    speech_request = SpeechRequest(
        model="mlx-community/VoxCPM2-8bit",
        input="测试。",
    )

    kwargs = _build_tts_generate_kwargs(model, speech_request)

    assert "temperature" not in kwargs
    assert "seed" not in kwargs


def test_build_tts_generate_kwargs_forwards_voxcpm_seed_and_temperature():
    def generate(text, seed=None, temperature=1.0, **kwargs):
        del text, seed, temperature, kwargs

    model = MagicMock()
    model.generate = generate
    model.__class__.__module__ = "mlx_audio.tts.models.voxcpm2.voxcpm2"

    speech_request = SpeechRequest(
        model="mlx-community/VoxCPM2-8bit",
        input="测试。",
        voxcpm={"seed": 123, "temperature": 1.0},
    )

    kwargs = _build_tts_generate_kwargs(model, speech_request)

    assert kwargs["seed"] == 123
    assert kwargs["temperature"] == 1.0


def test_seed_mlx_rng_offsets_by_sequence_index():
    _seed_mlx_rng(100, sequence_index=0)
    a = mx.random.normal((4,))
    _seed_mlx_rng(100, sequence_index=0)
    b = mx.random.normal((4,))
    _seed_mlx_rng(100, sequence_index=1)
    c = mx.random.normal((4,))

    assert np.allclose(np.array(a), np.array(b))
    assert not np.allclose(np.array(a), np.array(c))


def test_tts_run_serial_seeds_before_generate():
    from mlx_audio.server import SpeechRequest, TTSExecutionAdapter
    from mlx_audio.tests.test_server import sync_mock_audio_stream_generator

    adapter = TTSExecutionAdapter()
    mock_model = MagicMock()
    mock_model.generate = MagicMock(wraps=sync_mock_audio_stream_generator)
    mock_model.__class__.__module__ = "mlx_audio.tts.models.voxcpm2.voxcpm2"

    speech_request = SpeechRequest(
        model="mlx-community/VoxCPM2-8bit",
        input="测试。",
        seed=42,
        temperature=1.0,
        response_format="wav",
    )
    payload = MagicMock()
    payload.request = speech_request
    request = MagicMock()
    request.payload = payload
    request.cancel_event.is_set.return_value = False
    request.model_name = "mlx-community/VoxCPM2-8bit"
    request.emit_done = MagicMock()

    with (
        patch.object(adapter, "_get_model_for_request", return_value=mock_model),
        patch("mlx_audio.server._seed_mlx_rng") as mock_seed,
        patch.object(adapter, "_emit_audio"),
    ):
        adapter.run_serial(request)

    mock_seed.assert_called_once_with(42, sequence_index=0)
    _, kwargs = mock_model.generate.call_args
    assert kwargs["seed"] == 42
    assert kwargs["temperature"] == 1.0


def test_tts_execution_adapter_batch_fallback_seeds_with_sequence_index():
    from mlx_audio.server import SpeechRequest, TTSExecutionAdapter
    from mlx_audio.tests.test_server import sync_mock_audio_stream_generator

    adapter = TTSExecutionAdapter()
    mock_model = MagicMock()
    mock_model.__class__.__module__ = "mlx_audio.tts.models.voxcpm2.voxcpm2"
    mock_model.generate = MagicMock(wraps=sync_mock_audio_stream_generator)

    requests = []
    for idx in range(2):
        speech_request = SpeechRequest(
            model="mlx-community/VoxCPM2-8bit",
            input=f"line {idx}",
            seed=10,
            temperature=1.0,
        )
        payload = MagicMock()
        payload.request = speech_request
        request = MagicMock()
        request.payload = payload
        request.cancel_event.is_set.return_value = False
        request.model_name = "mlx-community/VoxCPM2-8bit"
        request.emit_done = MagicMock()
        requests.append(request)

    with (
        patch.object(adapter, "_get_model_for_request", return_value=mock_model),
        patch.object(adapter, "_can_call_batch_generate", return_value=False),
        patch("mlx_audio.server._seed_mlx_rng") as mock_seed,
        patch.object(adapter, "_emit_audio"),
    ):
        adapter.run_batch(requests)

    assert mock_seed.call_args_list == [
        ((10,), {"sequence_index": 0}),
        ((10,), {"sequence_index": 1}),
    ]


def test_validate_speech_request_requires_prompt_pair():
    with pytest.raises(HTTPException) as exc_info:
        _validate_speech_request(
            SpeechRequest(
                model="mlx-community/VoxCPM2-8bit",
                input="Hello",
                prompt_text="previous text only",
            )
        )
    assert "prompt_text and prompt_audio" in exc_info.value.detail


def test_tts_speech_rejects_partial_continuation_payload(client, mock_model_provider):
    mock_tts_model = MagicMock()
    mock_model_provider.load_model = MagicMock(return_value=mock_tts_model)

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "mlx-community/VoxCPM2-8bit",
            "input": "Hello",
            "prompt_text": "Only text, no audio",
        },
    )

    assert response.status_code == 400
    mock_tts_model.generate.assert_not_called()


def test_tts_speech_forwards_voxcpm2_kwargs(client, mock_model_provider):
    from mlx_audio.tests.test_server import MockAudioResult, sync_mock_audio_stream_generator

    mock_tts_model = MagicMock()
    mock_tts_model.generate = MagicMock(wraps=sync_mock_audio_stream_generator)
    mock_tts_model.sample_rate = 48000
    mock_model_provider.load_model = MagicMock(return_value=mock_tts_model)

    ref_b64 = _encode_wav_base64(np.zeros(1600, dtype=np.float32))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "mlx-community/VoxCPM2-8bit",
            "input": "This continues seamlessly.",
            "instructions": "A young woman, gentle voice",
            "ref_audio": ref_b64,
            "ref_text": "placeholder",
            "inference_timesteps": 10,
            "cfg_value": 2.0,
        },
    )

    assert response.status_code == 200
    _, kwargs = mock_tts_model.generate.call_args
    assert kwargs["instruct"] == "A young woman, gentle voice"
    assert kwargs["ref_text"] == "placeholder"
    assert kwargs["inference_timesteps"] == 10
    assert kwargs["cfg_value"] == 2.0
    assert isinstance(kwargs["ref_audio"], mx.array)
