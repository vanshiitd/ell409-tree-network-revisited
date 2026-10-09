"""Paths, device, logging and feature-cache helpers shared by the scripts."""
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
for _d in (RESULTS, FIGURES, RESULTS / "trunks", RESULTS / "trees"):
    _d.mkdir(parents=True, exist_ok=True)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Logger:
    def __init__(self, name):
        self.f = open(RESULTS / f"{name}.log", "a")
        self.t0 = time.time()

    def __call__(self, msg):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def save_json(name, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=conv))


def load_json(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def feature_path(dataset, trunk):
    return DATA / f"feats_{dataset}_{trunk}.npz"


def load_features(dataset, trunk, dev):
    """Returns dict split -> (features fp16 tensor on device, labels numpy)."""
    d = np.load(feature_path(dataset, trunk))
    return {s: (torch.tensor(d[f"F_{s}"], device=dev), d[f"y_{s}"]) for s in ("tree", "val", "test")}
