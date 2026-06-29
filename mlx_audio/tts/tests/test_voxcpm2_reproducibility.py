import os

import numpy as np
import pytest

VOXCPM2_MODEL = os.environ.get(
    "VOXCPM2_MLX_MODEL", "mlx-community/VoxCPM2-8bit"
)


@pytest.fixture(scope="module")
def voxcpm2_model():
    from mlx_audio.tts.utils import load

    return load(VOXCPM2_MODEL)


def test_voxcpm2_same_seed_same_audio(voxcpm2_model):
    kwargs = dict(
        text="测试。",
        instruct="Male, mid-twenties.",
        seed=123,
        temperature=1.0,
        inference_timesteps=10,
        cfg_value=2.0,
        max_tokens=64,
    )
    audio_a = np.array(next(voxcpm2_model.generate(**kwargs)).audio)
    audio_b = np.array(next(voxcpm2_model.generate(**kwargs)).audio)

    assert np.allclose(audio_a, audio_b)


def test_voxcpm2_different_seed_different_audio(voxcpm2_model):
    base = dict(
        text="测试。",
        instruct="Male, mid-twenties.",
        temperature=1.0,
        inference_timesteps=10,
        cfg_value=2.0,
        max_tokens=64,
    )
    audio_a = np.array(next(voxcpm2_model.generate(**base, seed=1)).audio)
    audio_b = np.array(next(voxcpm2_model.generate(**base, seed=2)).audio)

    assert not np.allclose(audio_a, audio_b)


def test_voxcpm2_temperature_affects_output(voxcpm2_model):
    base = dict(
        text="测试。",
        instruct="Male, mid-twenties.",
        seed=999,
        inference_timesteps=10,
        cfg_value=2.0,
        max_tokens=64,
    )
    audio_low = np.array(next(voxcpm2_model.generate(**base, temperature=0.5)).audio)
    audio_high = np.array(
        next(voxcpm2_model.generate(**base, temperature=1.5)).audio
    )

    overlap = min(len(audio_low), len(audio_high))
    assert overlap > 0
    assert not np.allclose(audio_low[:overlap], audio_high[:overlap])
