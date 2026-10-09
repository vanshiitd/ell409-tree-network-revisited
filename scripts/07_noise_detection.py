"""Step 7 (extra): the tree as a label-noise detector.

A gated tree routes every training sample along a single path. Samples the early, general-purpose nodes
already classify correctly stop high in the tree; a mislabelled sample contradicts its neighbours and can
only be fitted by a deep, highly specific node. We therefore score each training sample by the depth at
which the tree finally resolves it, and test how well that score finds labels we flipped on purpose.

Also lists the deepest-resolved samples of the real (un-noised) Cats vs Dogs training set.

Outputs: results/noise_detection.json, figures/noise_detection.pdf, figures/deepest_catsdogs.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from treenet.common import FIGURES, RESULTS, Logger, device, load_features, load_json, save_json
from treenet.data import get_splits
from treenet.style import AQUA, BLUE, INK, INK2, MUTED, ORANGE, setup
from treenet.tree import (HierarchicalTreeNetwork, TreeTypeNetwork, leaves, node_predict,
                          spectral_hierarchy)

setup()
log = Logger("07_noise_detection")
dev = device()
trunks = load_json("trunks")
W, N, ETA = (int(sys.argv[1]) if len(sys.argv) > 1 else 8), 4000, 0.2


def resolve(tree, Fx):
    """For each sample of a gated binary tree: depth of the last node on its path and the size of that
    node's training set (its 'leaf size')."""
    depth = np.zeros(len(Fx), dtype=np.int64)
    size = np.zeros(len(Fx), dtype=np.int64)

    def rec(node, rows):
        if len(rows) == 0:
            return
        depth[rows] = node.depth
        size[rows] = len(rows)
        if node.const is not None:
            return
        g = node_predict(node.net, Fx[torch.as_tensor(rows, device=dev)]).cpu().numpy()
        if node.A is not None:
            rec(node.A, rows[~g])
        if node.B is not None:
            rec(node.B, rows[g])

    rec(tree.root, np.arange(len(Fx)))
    return depth, size


def resolve_depth(tree, Fx):
    return resolve(tree, Fx)[0]


def sample_scores(model, Fx, y, dataset):
    """Returns (resolution depth, leaf size); for CIFAR-10 depth is summed and leaf size minimised over
    the class-hierarchy splits a sample passes through."""
    if dataset != "cifar10":
        return resolve(model, Fx)
    total = np.zeros(len(y), dtype=np.int64)
    smallest = np.full(len(y), 10 ** 9, dtype=np.int64)
    for key, tree in model.splits.items():
        h = model.h
        for c in key[1:]:
            h = h[int(c)]
        rows = np.where(np.isin(y, leaves(h[0]) + leaves(h[1])))[0]
        d, sz = resolve(tree, Fx[torch.as_tensor(rows, device=dev)])
        total[rows] += d
        smallest[rows] = np.minimum(smallest[rows], sz)
    return total, smallest


res = {}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
for ax, dataset in zip(axes, ("catsdogs", "cifar10")):
    feats = load_features(dataset, "split", dev)
    Ftr_all, ytr_all = feats["tree"]
    K = int(ytr_all.max()) + 1
    rng = np.random.default_rng(7)
    sub = np.sort(np.concatenate([rng.choice(np.where(ytr_all == c)[0], N // K, replace=False) for c in range(K)]))
    Fx = Ftr_all[torch.as_tensor(sub, device=dev)]
    y_clean = ytr_all[sub]
    y = y_clean.copy()
    flip = rng.random(len(y)) < ETA
    y[flip] = (y[flip] + rng.integers(1, K, flip.sum())) % K
    if dataset == "cifar10":
        hier = spectral_hierarchy(np.array(trunks["cifar10_split"]["val_confusion"]))
        model = HierarchicalTreeNetwork(hier, width=W, variant="gated", seed=3).fit(Fx, y)
    else:
        model = TreeTypeNetwork(width=W, variant="gated", seed=3).fit(Fx, y)
    depth, leaf = sample_scores(model, Fx, y, dataset)
    auc_depth = roc_auc_score(flip, depth)
    score = -np.log(leaf)                       # small final leaf -> suspicious label
    auc = roc_auc_score(flip, score)
    k = int(flip.sum())
    top = np.argsort(-score, kind="stable")[:k]
    prec = float(flip[top].mean())
    res[dataset] = dict(n=len(y), flipped=k, auc_leaf_size=float(auc), auc_depth=float(auc_depth),
                        precision_at_k=prec, leaf_size_median_clean=float(np.median(leaf[~flip])),
                        leaf_size_median_flipped=float(np.median(leaf[flip])),
                        mean_depth_clean=float(depth[~flip].mean()), mean_depth_flipped=float(depth[flip].mean()),
                        train_acc=float((model.predict(Fx).cpu().numpy() == y).mean()))
    log(f"{dataset}: leaf-size AUC {auc:.3f} (depth AUC {auc_depth:.3f}), precision@{k} {prec:.3f}, "
        f"median leaf size clean {np.median(leaf[~flip]):.0f} flipped {np.median(leaf[flip]):.0f}")
    bins = np.linspace(0, np.log10(leaf.max()) + 0.1, 25)
    ax.hist(np.log10(leaf[~flip]), bins=bins, color=BLUE, alpha=0.75, label="correct label", density=True)
    ax.hist(np.log10(leaf[flip]), bins=bins, color=ORANGE, alpha=0.65, label="flipped label", density=True)
    ax.set_xlabel("log10 size of the final node's sample set" + (" (smallest over splits)" if dataset == "cifar10" else ""))
    ax.set_ylabel("fraction of samples")
    ax.set_title(f"({'a' if dataset == 'catsdogs' else 'b'}) {'Cats vs Dogs' if dataset == 'catsdogs' else 'CIFAR-10'}: "
                 f"AUC {auc:.2f}", loc="left", color=INK)
    ax.legend(loc="upper right")
fig.tight_layout(w_pad=2)
fig.savefig(FIGURES / f"noise_detection_w{W}.pdf")
plt.close(fig)

# deepest-resolved samples of the real Cats vs Dogs training set (main width-8 tree)
feats = load_features("catsdogs", "split", dev)
Ftr, ytr = feats["tree"]
model = torch.load(RESULTS / "trees" / "catsdogs_split_gated_w8.pt", map_location=dev, weights_only=False)
depth, leaf = resolve(model, Ftr)
Xtree = get_splits("catsdogs")["tree"][0]
order = np.lexsort((-depth, leaf))[:16]           # smallest final sets first, deeper first on ties
fig, axes = plt.subplots(2, 8, figsize=(6.6, 2.0))
for ax, i in zip(axes.flat, order):
    ax.imshow(Xtree[i])
    ax.set_title(f"{'dog' if ytr[i] else 'cat'}, set {leaf[i]}", fontsize=6.5, color=INK, pad=2)
    ax.axis("off")
fig.tight_layout(pad=0.2)
fig.savefig(FIGURES / "deepest_catsdogs.pdf")
plt.close(fig)
res["catsdogs_real_smallest_sets"] = [dict(index=int(i), label=int(ytr[i]), depth=int(depth[i]), leaf=int(leaf[i]))
                                     for i in order]
save_json(f"noise_detection_w{W}", res)
log("done")
