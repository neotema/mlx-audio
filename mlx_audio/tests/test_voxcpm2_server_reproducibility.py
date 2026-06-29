import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mlx_audio.server import app

VOXCPM2_MODEL = os.environ.get(
    "VOXCPM2_MLX_MODEL", "mlx-community/VoxCPM2-8bit"
)


@pytest.fixture(scope="module")
def voxcpm2_client():
    from mlx_audio.tts.utils import load

    model = load(VOXCPM2_MODEL)

    with patch("mlx_audio.server.model_provider") as mock_provider:
        mock_provider.load_model = MagicMock(return_value=model)
        with patch("mlx_audio.server._load_model_for_inference", return_value=model):
            with patch("mlx_audio.server.INFERENCE_BROKER", None):
                yield TestClient(app)


def test_tts_speech_same_seed_same_wav_bytes(voxcpm2_client):
    payload = {
        "model": VOXCPM2_MODEL,
        "input": "滨江市，凌晨两点三十分。",
        "instruct": "Male, mid-twenties. Low deep voice.",
        "seed": 42,
        "temperature": 1.0,
        "voxcpm": {"cfg_value": 2.2, "inference_timesteps": 10},
        "response_format": "wav",
        "max_tokens": 64,
    }

    first = voxcpm2_client.post("/v1/audio/speech", json=payload)
    second = voxcpm2_client.post("/v1/audio/speech", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.content == second.content
    assert len(first.content) > 44
