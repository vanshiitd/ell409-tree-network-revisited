"""Step 3: double descent in node width.

Grows a full gated tree for each width on a fixed subset, with and without label noise, and evaluates it
at several depth caps (depth 0 is the root network alone).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from treenet.common import RESULTS, Logger, device, load_features, load_json, save_json
from treenet.tree import HierarchicalTreeNetwork, TreeTypeNetwork, spectral_hierarchy

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="*", default=["catsdogs", "cifar10"])
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--noise", nargs="*", type=float, default=[0.0, 0.2])
ap.add_argument("--widths", nargs="*", type=int,
                default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
ap.add_argument("--seeds", type=int, default=1)
args = ap.parse_args()

log = Logger("03_double_descent")
dev = device()
CAPS = [0, 1, 2, 3, 4, 6, 8, 100]
out = load_json("double_descent") if (RESULTS / "double_descent.json").exists() else {}
trunks = load_json("trunks")

for dataset in args.datasets:
    feats = load_features(dataset, "split", dev)
    (Ftr_all, ytr_all), (Fte, yte) = feats["tree"], feats["test"]
    K = int(ytr_all.max()) + 1
    hier = spectral_hierarchy(np.array(trunks["cifar10_split"]["val_confusion"])) if dataset == "cifar10" else None
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        sub = np.sort(np.concatenate([rng.choice(np.where(ytr_all == c)[0], args.n // K, replace=False)
                                      for c in range(K)]))
        Fsub = Ftr_all[torch.as_tensor(sub, device=dev)]
        for eta in args.noise:
            y = ytr_all[sub].copy()
            flip = rng.random(len(y)) < eta
            y[flip] = (y[flip] + rng.integers(1, K, flip.sum())) % K     # always a different class
            key = f"{dataset}_eta{eta}_seed{seed}"
            rows = out.get(key, [])
            done = {r["width"] for r in rows}
            for w in args.widths:
                if w in done:
                    continue
                t0 = time.time()
                if dataset == "cifar10":
                    model = HierarchicalTreeNetwork(hier, width=w, variant="gated", seed=seed).fit(Fsub, y)
                else:
                    model = TreeTypeNetwork(width=w, variant="gated", seed=seed).fit(Fsub, y)
                r = dict(width=w, seconds=round(time.time() - t0, 1), caps={})
                for cap in CAPS:
                    if dataset == "cifar10":
                        s = model.summary(cap)
                    else:
                        s = dict(nodes=len(model.nodes(cap)), params=model.n_params(cap))
                    ptr = model.predict(Fsub, cap).cpu().numpy().astype(int)
                    pte = model.predict(Fte, cap).cpu().numpy().astype(int)
                    r["caps"][str(cap)] = dict(nodes=s["nodes"], params=s["params"],
                                               train_err=float((ptr != y).mean()),
                                               train_err_clean=float((ptr != ytr_all[sub]).mean()),
                                               test_err=float((pte != yte).mean()))
                rows.append(r)
                c0, cf = r["caps"]["0"], r["caps"]["100"]
                log(f"{key} w={w:<4} root: train {c0['train_err']:.3f} test {c0['test_err']:.3f} | "
                    f"tree: {cf['nodes']} nodes, train {cf['train_err']:.3f} test {cf['test_err']:.3f} "
                    f"({r['seconds']}s)")
                out[key] = sorted(rows, key=lambda r: r["width"])
                save_json("double_descent", out)
                del model
log("done")
