"""Step 5: shrinking the network.

1. width sweep: total parameters of the full tree for different node widths
2. magnitude pruning inside nodes, keeping every node's training decisions exactly
3. validation pruning of subtrees (results come from step 2)
"""
import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from treenet.common import RESULTS, Logger, device, load_features, load_json, save_json
from treenet.tree import (HierarchicalTreeNetwork, NodeNet, TreeTypeNetwork, count_params, leaves,
                          node_predict, spectral_hierarchy)

ap = argparse.ArgumentParser()
ap.add_argument("--widths", nargs="*", type=int, default=None,
                help="default: 1..64 for Cats vs Dogs; 16, 32, 64 for CIFAR-10 (w=8 is taken from step 2)")
ap.add_argument("--prune_width", type=int, default=8)
ap.add_argument("--skip_sweep", action="store_true")
args = ap.parse_args()

log = Logger("05_minimize")
dev = device()
trunks = load_json("trunks")
res = load_json("minimize") if (RESULTS / "minimize.json").exists() else {}


def build(dataset, Ftr, ytr, width, seed=0):
    if dataset == "cifar10":
        hier = spectral_hierarchy(np.array(trunks["cifar10_split"]["val_confusion"]))
        return HierarchicalTreeNetwork(hier, width=width, variant="gated", seed=seed).fit(Ftr, ytr)
    return TreeTypeNetwork(width=width, variant="gated", seed=seed).fit(Ftr, ytr)


def acc(model, Fx, y):
    return float((model.predict(Fx).cpu().numpy().astype(int) == y.astype(int)).mean())


def prune_node(net, Fx, g, levels=(0.5, 0.75, 0.9, 0.95, 0.98), rounds=15, lr=2e-3):
    """Largest sparsity at which the node, after a masked fine-tune, still reproduces its decisions g.
    Returns (pruned_net, nonzero_weights, sparsity)."""
    params = [net.reduce.weight, net.fc1.weight, net.fc2.weight]
    best = copy.deepcopy(net)
    best_s = 0.0
    tgt = g.float()
    n = len(g)
    n_pos = float(tgt.sum())
    pw = torch.tensor(min(1e4, max(1e-4, (n - n_pos) / max(n_pos, 1.0))), device=Fx.device)
    for s in levels:
        cand = copy.deepcopy(best)
        cp = [cand.reduce.weight, cand.fc1.weight, cand.fc2.weight]
        allw = torch.cat([p.detach().abs().flatten() for p in cp])
        k = int(s * allw.numel())
        if k == 0:
            continue
        thr = allw.kthvalue(k).values
        masks = [(p.detach().abs() > thr).float() for p in cp]
        with torch.no_grad():
            for p, m in zip(cp, masks):
                p.mul_(m)
        opt = torch.optim.Adam(cand.parameters(), lr=lr)
        ok = bool((node_predict(cand, Fx) == g).all())
        bs = min(512, n)
        for _ in range(rounds):
            if ok:
                break
            cand.train()
            for _ in range(max(n // bs, 20)):
                b = torch.randint(0, n, (bs,), device=Fx.device)
                loss = F.binary_cross_entropy_with_logits(cand(Fx[b].float()), tgt[b], pos_weight=pw)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for p, m in zip(cp, masks):
                        p.mul_(m)
            ok = bool((node_predict(cand, Fx) == g).all())
        if not ok:
            break
        best, best_s = cand, s
    nz = sum(int((p != 0).sum()) for p in [best.reduce.weight, best.fc1.weight, best.fc2.weight]) + \
        sum(p.numel() for p in [best.reduce.bias, best.fc1.bias, best.fc2.bias])
    return best, nz, best_s


def prune_tree_weights(tree, Fx, t):
    """Run prune_node on every node of a gated tree, top-down."""
    stats = []

    def rec(node, rows):
        if node.const is not None or len(rows) == 0:
            return
        Fv = Fx[rows]
        g = node_predict(node.net, Fv)
        new, nz, s = prune_node(node.net, Fv, g)
        stats.append((count_params(node.net), nz, s))
        node.net = new
        if node.A is not None:
            rec(node.A, rows[~g])
        if node.B is not None:
            rec(node.B, rows[g])

    rec(tree.root, torch.arange(len(t), device=Fx.device))
    return stats


for dataset in ("catsdogs", "cifar10"):
    feats = load_features(dataset, "split", dev)
    (Ftr, ytr), (Fva, yva), (Fte, yte) = feats["tree"], feats["val"], feats["test"]
    r = res.get(dataset, {})

    if not args.skip_sweep:
        sweep = r.get("width_sweep", [])
        done = {row["width"] for row in sweep}
        main = load_json("trees").get(f"{dataset}_split_gated_w8")
        if 8 not in done and main is not None:              # reuse the width-8 tree from step 2
            sweep.append(dict(width=8, node_params=count_params(NodeNet(8)), nodes=main["summary"]["nodes"],
                              networks=main["summary"]["networks"], params=main["summary"]["params"],
                              depth=main["summary"]["depth"], train=main["train"], test=main["test"],
                              pruned={k: main["pruned"][k] for k in ("nodes", "params", "train", "test")},
                              seconds=main["seconds"]))
            done.add(8)
        widths = args.widths or ([1, 2, 4, 8, 16, 32, 64] if dataset == "catsdogs" else [16, 32, 64])
        for w in widths:
            if w in done:
                continue
            t0 = time.time()
            m = build(dataset, Ftr, ytr, w)
            s = m.summary()
            pm = copy.deepcopy(m)
            removed = pm.prune(Fva, yva)
            row = dict(width=w, node_params=count_params(NodeNet(w)),
                       nodes=s["nodes"], networks=s["networks"], params=s["params"], depth=s["depth"],
                       train=acc(m, Ftr, ytr), test=acc(m, Fte, yte),
                       pruned=dict(nodes=pm.summary()["nodes"], params=pm.summary()["params"],
                                   train=acc(pm, Ftr, ytr), test=acc(pm, Fte, yte)),
                       seconds=round(time.time() - t0, 1))
            sweep.append(row)
            log(f"{dataset} width {w}: {row['nodes']} nodes, {row['params']} params, train {row['train']:.4f} "
                f"test {row['test']:.4f} | pruned {row['pruned']['nodes']} nodes test {row['pruned']['test']:.4f} "
                f"({row['seconds']}s)")
            r["width_sweep"] = sorted(sweep, key=lambda x: x["width"])
            res[dataset] = r
            save_json("minimize", res)

    # weight pruning inside the width-8 tree
    model = torch.load(RESULTS / "trees" / f"{dataset}_split_gated_w{args.prune_width}.pt", map_location=dev,
                       weights_only=False)
    t0 = time.time()
    before = acc(model, Ftr, ytr), acc(model, Fte, yte)
    stats = []
    if dataset == "cifar10":
        for key, tree in model.splits.items():
            h = model.h
            for c in key[1:]:
                h = h[int(c)]
            L, R = leaves(h[0]), leaves(h[1])
            rows = np.where(np.isin(ytr, L + R))[0]
            stats += prune_tree_weights(tree, Ftr[torch.as_tensor(rows, device=dev)], np.isin(ytr[rows], R))
    else:
        stats = prune_tree_weights(model, Ftr, ytr)
    dense = sum(a for a, _, _ in stats)
    nonzero = sum(b for _, b, _ in stats)
    after = acc(model, Ftr, ytr), acc(model, Fte, yte)
    r["weight_pruning"] = dict(width=args.prune_width, nodes=len(stats), dense_params=dense, nonzero_params=nonzero,
                               sparsity_hist=np.histogram([s for _, _, s in stats],
                                                          bins=[-0.01, 0.25, 0.6, 0.8, 0.92, 0.96, 1.0])[0].tolist(),
                               train_before=before[0], test_before=before[1],
                               train_after=after[0], test_after=after[1], seconds=round(time.time() - t0, 1))
    log(f"{dataset} weight pruning: {dense} -> {nonzero} non-zero node weights "
        f"({100 * (1 - nonzero / dense):.1f}% removed); train {before[0]:.4f}->{after[0]:.4f}, "
        f"test {before[1]:.4f}->{after[1]:.4f}")
    torch.save(model, RESULTS / "trees" / f"{dataset}_split_gated_w{args.prune_width}_sparse.pt")
    res[dataset] = r
    save_json("minimize", res)
log("done")
