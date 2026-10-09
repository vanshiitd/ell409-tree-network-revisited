"""Step 4: receptive fields of the tree's nodes.

For every node of the width-8 gated trees (split trunk) we route the training samples through the tree to
recover each node's own training set S_v and targets t_v, then compute

  * Grad-CAM at the trunk tap (8x8): where in the image the node's decision comes from,
  * a spatial-concentration score: share of Grad-CAM mass in the central 4x4 cells (of 8x8),
  * activation maximisation through the frozen trunk: the input the node responds to most,
  * the node's most strongly activating training images (with Grad-CAM overlays),
  * for CIFAR-10, which classes the node must fire on.

Outputs: figures/rf_<dataset>.pdf, figures/rf_centrality.pdf, results/receptive_fields.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from treenet.common import FIGURES, RESULTS, Logger, device, feature_path, save_json
from treenet.data import class_names, get_splits
from treenet.style import BLUE, INK, INK2, MUTED, ORANGE, setup
from treenet.tree import leaves, node_predict
from treenet.trunk import Trunk

setup()
log = Logger("04_receptive_fields")
dev = device()
WIDTH = 8
out = {}


def route(tree, Fx, t):
    """Training set (row indices into Fx) and targets of every node of a gated tree."""
    sets = {}

    def rec(node, rows, tt):
        sets[node.name] = (node, rows, tt)
        if node.const is not None or len(rows) == 0:
            return
        g = node_predict(node.net, Fx[rows]).cpu().numpy()
        if node.A is not None:
            rec(node.A, rows[~g], tt[~g])
        if node.B is not None:
            rec(node.B, rows[g], 1 - tt[g])

    rec(tree.root, np.arange(len(t)), t.astype(np.int64))
    return sets


def node_input(trunk, mu, sd, x):
    return (trunk.tap(x) - mu) / sd


def gradcam(trunk, mu, sd, net, X_uint8):
    """Grad-CAM maps (B, 8, 8) of a node's logit at the trunk tap, plus the logits."""
    maps, logits = [], []
    for k in range(0, len(X_uint8), 256):
        x = trunk.normalize(torch.tensor(X_uint8[k:k + 256], device=dev))
        f = trunk.tap(x).detach().requires_grad_(True)
        z = net((f - mu) / sd)
        g, = torch.autograd.grad(z.sum(), f)
        alpha = g.mean((2, 3), keepdim=True)
        cam = F.relu((alpha * f).sum(1))
        cam = cam / (cam.flatten(1).sum(1)[:, None, None] + 1e-8)
        maps.append(cam.detach().cpu())
        logits.append(z.detach().cpu())
    return torch.cat(maps).numpy(), torch.cat(logits).numpy()


def activation_max(trunk, mu, sd, net, size, steps=300, seed=0):
    """Input that maximises the node's logit (jitter + L2 + total-variation regularisation)."""
    torch.manual_seed(seed)
    x = (0.1 * torch.randn(1, 3, size, size, device=dev)).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    for s in range(steps):
        dx, dy = np.random.randint(-2, 3, 2)
        xj = torch.roll(x, shifts=(int(dx), int(dy)), dims=(2, 3))
        z = net(node_input(trunk, mu, sd, xj))
        tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        loss = -z.mean() + 0.05 * (x ** 2).mean() + 0.5 * tv
        opt.zero_grad()
        loss.backward()
        opt.step()
    img = x.detach()[0] * trunk.std[:, None, None] + trunk.mean[:, None, None]
    img = img.permute(1, 2, 0).cpu().numpy()
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def centrality(cams):
    """Share of Grad-CAM mass in the central 4x4 block of the 8x8 grid (uniform map: 0.25)."""
    return cams[:, 2:6, 2:6].sum((1, 2))


all_cent = []
for dataset in (sys.argv[1:] or ("catsdogs", "cifar10")):
    names = class_names(dataset)
    sp = get_splits(dataset)
    Xtree, ytree = sp["tree"]
    size = Xtree.shape[1]
    trunk = Trunk(in_size=size, n_classes=len(names)).to(dev)
    trunk.load_state_dict(torch.load(RESULTS / "trunks" / f"{dataset}_split.pt", map_location=dev))
    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    d = np.load(feature_path(dataset, "split"))
    mu = torch.tensor(d["mu"], device=dev)
    sd = torch.tensor(d["sd"], device=dev)
    Ftree = torch.tensor(d["F_tree"], device=dev)
    model = torch.load(RESULTS / "trees" / f"{dataset}_split_gated_w{WIDTH}.pt", map_location=dev,
                       weights_only=False)

    if dataset == "cifar10":
        trees = []
        for key, tree in model.splits.items():
            h = model.h
            for c in key[1:]:
                h = h[int(c)]
            L, R = leaves(h[0]), leaves(h[1])
            rows = np.where(np.isin(ytree, L + R))[0]
            trees.append((key, tree, rows, np.isin(ytree[rows], R).astype(np.int64),
                          ("/".join(names[c] for c in L), "/".join(names[c] for c in R))))
    else:
        trees = [("T", model, np.arange(len(ytree)), ytree.astype(np.int64), ("cat", "dog"))]

    records, panels = [], []
    for key, tree, rows, t, side_names in trees:
        sets = route(tree, Ftree[torch.as_tensor(rows, device=dev)], t)
        for nname, (node, nrows, tt) in sets.items():
            if node.net is None or len(nrows) < 20 or tt.sum() < 5:
                continue
            gidx = rows[nrows]
            pos = gidx[tt == 1]
            sel = pos if len(pos) <= 600 else np.random.default_rng(0).choice(pos, 600, replace=False)
            cams, z = gradcam(trunk, mu, sd, node.net, Xtree[sel])
            fire = z > 0
            cent = float(centrality(cams[fire]).mean()) if fire.any() else float("nan")
            classes = np.bincount(ytree[pos], minlength=len(names)).tolist()
            rec = dict(split=key, node=nname, depth=node.depth, n=len(nrows), n_pos=int(tt.sum()),
                       errors=node.errors, centrality=cent, positive_classes=classes,
                       sides=side_names)
            records.append(rec)
            all_cent.append((dataset, node.depth, cent, int(tt.sum())))
            if (dataset == "catsdogs" and node.depth <= 2) or \
               (dataset == "cifar10" and ((key == "H" and node.depth <= 1) or (node.depth == 0 and key in ("H0", "H1")))):
                panels.append((rec, node, sel, cams, z))
    out[dataset] = records
    log(f"{dataset}: {len(records)} nodes analysed, {len(panels)} shown")

    # ---- figure: one row per selected node
    panels = panels[:7]
    ntop = 5
    fig, axes = plt.subplots(len(panels), 2 + ntop, figsize=(7.0, 1.12 * len(panels) + 0.3),
                             gridspec_kw=dict(wspace=0.05, hspace=0.42))
    for row, (rec, node, sel, cams, z) in enumerate(panels):
        ax = axes[row]
        am = activation_max(trunk, mu, sd, node.net, size, seed=row)
        ax[0].imshow(am)
        mean_cam = cams[z > 0].mean(0) if (z > 0).any() else cams.mean(0)
        ax[1].imshow(mean_cam, cmap="magma", interpolation="bilinear")
        top = np.argsort(-z)[:ntop]
        for j, i in enumerate(top):
            img = Xtree[sel[i]].astype(np.float32) / 255.0
            cam = torch.tensor(cams[i])[None, None]
            cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
            cam = cam / (cam.max() + 1e-8)
            ax[2 + j].imshow(img * (0.2 + 0.8 * cam[..., None]))   # dark = ignored by the node
        neg, posn = rec["sides"]
        role = {"": "root", "A": "child A (rescues misses)", "B": "child B (vetoes false alarms)"}
        kind = role[""] if rec["node"] == "r" else role[rec["node"][-1]]
        tgt = posn if rec["node"].count("B") % 2 == 0 else neg
        label = f"{rec['split']}:{rec['node']}  depth {rec['depth']}  {kind};  fires on '{tgt}'" if dataset == "cifar10" \
            else f"{rec['node']}  depth {rec['depth']}  {kind};  fires on '{tgt}'"
        label += f";  n={rec['n']}, positives={rec['n_pos']}, centrality={rec['centrality']:.2f}"
        ax[0].set_title(label, loc="left", fontsize=6.3, color=INK, pad=2)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for s in a.spines.values():
                s.set_visible(False)
    axes[0][0].text(0.5, 1.45, "preferred input", transform=axes[0][0].transAxes, ha="center", fontsize=6.5,
                    color=INK2)
    axes[0][1].text(0.5, 1.45, "mean Grad-CAM", transform=axes[0][1].transAxes, ha="center", fontsize=6.5,
                    color=INK2)
    axes[0][4].text(0.5, 1.45, "most strongly activating training images (darkened where Grad-CAM is low)",
                    transform=axes[0][4].transAxes, ha="center", fontsize=6.5, color=INK2)
    fig.savefig(FIGURES / f"rf_{dataset}.pdf")
    plt.close(fig)
    log(f"  wrote figures/rf_{dataset}.pdf")

# ---- figure: spatial concentration vs depth
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
for ax, dataset, title in zip(axes, ("catsdogs", "cifar10"), ("(a) Cats vs Dogs", "(b) CIFAR-10 (all 9 splits)")):
    pts = [(d, c, n) for ds, d, c, n in all_cent if ds == dataset and not np.isnan(c)]
    if not pts:
        continue
    dd, cc, nn = map(np.array, zip(*pts))
    ax.scatter(dd + np.random.default_rng(0).uniform(-0.15, 0.15, len(dd)), cc, s=8 + 30 * np.sqrt(nn / nn.max()),
               color=BLUE, alpha=0.55, edgecolors="white", linewidths=0.4)
    for dep in np.unique(dd):
        ax.plot([dep - 0.3, dep + 0.3], [np.median(cc[dd == dep])] * 2, color=ORANGE, lw=2)
    ax.axhline(0.25, color=MUTED, ls=":", lw=0.9)
    ax.text(ax.get_xlim()[1] if False else dd.max() + 0.2, 0.255, "uniform", color=INK2, fontsize=7, va="bottom", ha="right")
    ax.set_xlabel("node depth")
    ax.set_title(title, loc="left", color=INK)
axes[0].set_ylabel("Grad-CAM mass in central 4x4")
fig.tight_layout()
fig.savefig(FIGURES / "rf_centrality.pdf")
plt.close(fig)
save_json("receptive_fields", out)
log("done")
