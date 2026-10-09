"""Dataset loading, caching and splitting for CIFAR-10 and Microsoft's Kaggle Cats vs Dogs.

Both datasets are converted once to uint8 numpy arrays (N, H, W, 3) and cached in data/*.npz.
Every image is used in exactly one of four disjoint splits:

    trunk  -- trains the shared feature extractor (never seen by the tree)
    tree   -- trains the tree-type network
    val    -- model selection / reduced-error pruning
    test   -- reported test accuracy only
"""
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CATSDOGS_CLASSES = ["cat", "dog"]


# ----------------------------------------------------------------------------- CIFAR-10
def load_cifar10():
    """Returns (X_train, y_train, X_test, y_test) as uint8 (N,32,32,3) / int64."""
    cache = DATA / "cifar10.npz"
    if cache.exists():
        d = np.load(cache)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    base = DATA / "cifar-10-batches-py"
    if base.exists() or (DATA / "cifar-10-python.tar.gz").exists():
        # original python pickles from https://www.cs.toronto.edu/~kriz/cifar.html
        if not base.exists():
            with tarfile.open(DATA / "cifar-10-python.tar.gz") as t:
                t.extractall(DATA)

        def batch(name):
            with open(base / name, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            return X.astype(np.uint8), np.array(d[b"labels"], dtype=np.int64)

        parts = [batch(f"data_batch_{i}") for i in range(1, 6)]
        Xtr = np.concatenate([p[0] for p in parts])
        ytr = np.concatenate([p[1] for p in parts])
        Xte, yte = batch("test_batch")
    else:
        # identical data from the Hugging Face mirror uoft-cs/cifar10 (PNG bytes in parquet files)
        import io
        import pyarrow.parquet as pq
        from PIL import Image

        def part(name):
            t = pq.read_table(DATA / "cifar10_hf" / f"{name}.parquet").to_pydict()
            X = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
            return X.astype(np.uint8), np.array(t["label"], dtype=np.int64)

        Xtr, ytr = part("train")
        Xte, yte = part("test")
    np.savez(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


# ----------------------------------------------------------------------------- Cats vs Dogs
def _load_one(args):
    path, size = args
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < 16:
                return None
            s = min(w, h)                                   # centre square crop, then resize
            left, top = (w - s) // 2, (h - s) // 2
            im = im.crop((left, top, left + s, top + s)).resize((size, size), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception:                                       # the archive contains a few corrupt files
        return None


def load_catsdogs(size=64):
    """Returns (X, y) for all decodable images, y = 0 cat / 1 dog."""
    cache = DATA / f"catsdogs{size}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    from multiprocessing import Pool
    base = DATA / "catsdogs" / "PetImages"
    jobs, labels = [], []
    for lab, cls in enumerate(["Cat", "Dog"]):
        for fn in sorted(os.listdir(base / cls)):
            if fn.lower().endswith(".jpg"):
                jobs.append((str(base / cls / fn), size))
                labels.append(lab)
    with Pool(8) as pool:
        imgs = pool.map(_load_one, jobs, chunksize=64)
    keep = [i for i, im in enumerate(imgs) if im is not None]
    X = np.stack([imgs[i] for i in keep])
    y = np.array([labels[i] for i in keep], dtype=np.int64)
    np.savez(cache, X=X, y=y)
    print(f"cats vs dogs: kept {len(keep)} / {len(jobs)} images")
    return X, y


# ----------------------------------------------------------------------------- splits
def get_splits(dataset, seed=0):
    """Returns dict split -> (X uint8, y). Splits are disjoint and stratified."""
    rng = np.random.default_rng(seed)
    if dataset == "cifar10":
        Xtr, ytr, Xte, yte = load_cifar10()
        sizes = dict(trunk=15000, val=5000)                 # remaining 30,000 -> tree
        X, y = Xtr, ytr
        test = (Xte, yte)
    elif dataset == "catsdogs":
        X, y = load_catsdogs()
        idx = rng.permutation(len(y))
        te, rest = idx[:5000], idx[5000:]
        test = (X[te], y[te])
        X, y = X[rest], y[rest]
        sizes = dict(trunk=6000, val=2000)                  # remaining ~12,000 -> tree
    else:
        raise ValueError(dataset)
    idx = rng.permutation(len(y))
    a, b = sizes["trunk"], sizes["trunk"] + sizes["val"]
    out = dict(trunk=(X[idx[:a]], y[idx[:a]]), val=(X[idx[a:b]], y[idx[a:b]]),
               tree=(X[idx[b:]], y[idx[b:]]), test=test)
    return out


def class_names(dataset):
    return CIFAR_CLASSES if dataset == "cifar10" else CATSDOGS_CLASSES
