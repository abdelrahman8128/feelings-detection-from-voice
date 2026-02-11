import os
import re
import glob
import zipfile
import requests
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .utils import (
    preprocess_wav_to_logmel,
    DEFAULT_SR,
    DEFAULT_DURATION,
    N_MELS,
)

RAVDESS_SPEECH_ZIP_URL = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)
RAVDESS_SPEECH_ZIP_NAME = "Audio_Speech_Actors_01-24.zip"
RAVDESS_SPEECH_DIR = "Audio_Speech_Actors_01-24"

# Emotion code (3rd field in filename) mapping to label index and name
EMOTION_CODE_TO_NAME = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
EMOTION_NAME_TO_ID = {name: i for i, name in enumerate(EMOTION_CODE_TO_NAME.values())}
EMOTION_CODE_TO_ID = {code: EMOTION_NAME_TO_ID[name] for code, name in EMOTION_CODE_TO_NAME.items()}

RAVDESS_FILENAME_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$")


def _download_file(url: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = 1024 * 1024
        downloaded = 0
        with open(dst_path, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    downloaded += len(part)
    return dst_path


def ensure_ravdess(data_dir: str) -> str:
    """
    Ensure RAVDESS speech subset is present under data_dir.
    Returns path to extracted directory containing actor folders.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, RAVDESS_SPEECH_ZIP_NAME)
    extract_dir = os.path.join(data_dir, RAVDESS_SPEECH_DIR)

    # If already extracted with actors present, return
    if os.path.isdir(extract_dir) and len(glob.glob(os.path.join(extract_dir, "Actor_*"))) > 0:
        return extract_dir

    # If zip exists, extract; else download and extract
    if not os.path.isfile(zip_path):
        _download_file(RAVDESS_SPEECH_ZIP_URL, zip_path)
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    return extract_dir


def _parse_emotion_from_filename(fname: str) -> Optional[int]:
    base = os.path.basename(fname)
    m = RAVDESS_FILENAME_RE.match(base)
    if not m:
        return None
    emo_code = m.group(3)
    return EMOTION_CODE_TO_ID.get(emo_code)


def _parse_actor_from_filename(fname: str) -> Optional[int]:
    base = os.path.basename(fname)
    m = RAVDESS_FILENAME_RE.match(base)
    if not m:
        return None
    actor = int(m.group(7))
    return actor


class RAVDESSDataset(Dataset):
    def __init__(self, root_dir: str, split: str, actors: List[int]):
        self.root_dir = root_dir
        self.split = split
        self.actors = set(actors)
        pattern = os.path.join(root_dir, "Actor_*", "*.wav")
        files = glob.glob(pattern)
        items = []
        for f in files:
            actor = _parse_actor_from_filename(f)
            label = _parse_emotion_from_filename(f)
            if actor is None or label is None:
                continue
            if actor in self.actors:
                items.append((f, label))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        logmel = preprocess_wav_to_logmel(path)  # (1, n_mels, T)
        return logmel, label


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    root = ensure_ravdess(data_dir)
    # Speaker-disjoint split
    train_actors = list(range(1, 21))
    val_actors = [21, 22]
    test_actors = [23, 24]

    train_set = RAVDESSDataset(root, "train", train_actors)
    val_set = RAVDESSDataset(root, "val", val_actors)
    test_set = RAVDESSDataset(root, "test", test_actors)

    def collate(batch):
        xs, ys = zip(*batch)
        # pad to max time in batch
        max_t = max(x.shape[-1] for x in xs)
        x_pad = []
        for x in xs:
            pad_t = max_t - x.shape[-1]
            if pad_t > 0:
                x = torch.nn.functional.pad(x, (0, pad_t))
            x_pad.append(x)
        X = torch.stack(x_pad, dim=0)  # (B, 1, n_mels, T)
        y = torch.tensor(ys, dtype=torch.long)
        return X, y

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    return train_loader, val_loader, test_loader
