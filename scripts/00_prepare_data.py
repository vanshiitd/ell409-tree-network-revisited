"""Step 0: decode both datasets once and cache them as data/cifar10.npz and data/catsdogs64.npz."""
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
