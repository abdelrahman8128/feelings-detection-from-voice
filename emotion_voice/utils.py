import os
import math
import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple

DEFAULT_SR = 16000
DEFAULT_DURATION = 4.0  # seconds
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 320  # ~20ms at 16k
FMIN = 50
FMAX = 8000

_mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=DEFAULT_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    center=True,
)
_amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

def load_audio_mono_16k(path: str, target_sr: int = DEFAULT_SR) -> np.ndarray:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32)

def pad_or_trim(y: np.ndarray, sr: int = DEFAULT_SR, duration: float = DEFAULT_DURATION) -> np.ndarray:
    target_len = int(sr * duration)
    if y.shape[0] < target_len:
        pad = target_len - y.shape[0]
        y = np.pad(y, (0, pad), mode="constant")
    elif y.shape[0] > target_len:
        y = y[:target_len]
    return y

def wav_to_logmel_tensor(y: np.ndarray, sr: int = DEFAULT_SR) -> torch.Tensor:
    # y: (T,) float32
    y_t = torch.from_numpy(y).unsqueeze(0)  # (1, T)
    mel = _mel_spec(y_t)  # (1, n_mels, time)
    logmel = _amplitude_to_db(mel + 1e-10)
    return logmel

def preprocess_wav_to_logmel(path: str) -> torch.Tensor:
    y = load_audio_mono_16k(path, DEFAULT_SR)
    y = pad_or_trim(y, DEFAULT_SR, DEFAULT_DURATION)
    logmel = wav_to_logmel_tensor(y, DEFAULT_SR)
    return logmel

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location=None) -> dict:
    return torch.load(path, map_location=map_location)
