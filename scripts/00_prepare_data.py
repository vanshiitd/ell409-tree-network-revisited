"""Step 0: decode both datasets once into cached uint8 arrays (data/cifar10.npz, data/catsdogs64.npz).

Expects data/cifar10_hf/{train,test}.parquet (Hugging Face mirror uoft-cs/cifar10) or the original
data/cifar-10-python.tar.gz, and data/catsdogs/PetImages/{Cat,Dog}/*.jpg (Microsoft's Kaggle Cats and Dogs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from treenet.data import load_catsdogs, load_cifar10

if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_cifar10()
    print("cifar10:", Xtr.shape, Xte.shape)
    X, y = load_catsdogs()
    print("catsdogs:", X.shape, np.bincount(y))
