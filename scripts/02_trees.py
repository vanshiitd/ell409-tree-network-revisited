"""Step 2: grow the trees until they fit the training set.

Records accuracy, size, the growth curve over depth, validation pruning, and a single network with the
same number of parameters as a baseline. CIFAR-10 uses the class hierarchy, Cats vs Dogs one binary tree.
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from treenet.common import RESULTS, Logger, device, load_features, load_json, save_json
from treenet.data import class_names
from treenet.tree import (HierarchicalTreeNetwork, TreeTypeNetwork, count_params, hierarchy_str,
                          spectral_hierarchy)

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=8)
ap.add_argument("--only", nargs="*", default=None, help="subset of config names to run")
ap.add_argument("--flat_only", action="store_true", help="recompute only the one-network baseline")
args = ap.parse_args()

log = Logger("02_trees")
dev = device()
trunks = load_json("trunks")
CONFIGS = [
    ("catsdogs", "split", "gated"), ("catsdogs", "split", "faithful"),
    ("catsdogs", "transfer", "gated"), ("catsdogs", "random", "gated"),
    ("cifar10", "split", "gated"), ("cifar10", "split", "faithful"), ("cifar10", "random", "gated"),
]


class FlatNet(nn.Module):
    """Node architecture with n_out outputs, used as the single-network baseline."""

    def __init__(self, width, n_out, c_in=64, spatial=8):
        super().__init__()
        k = max(1, math.ceil(width / 4))
        self.reduce = nn.Conv2d(c_in, k, 1)
        self.fc1 = nn.Linear(k * spatial * spatial, width)
        self.fc2 = nn.Linear(width, n_out)

    def forward(self, f):
        return self.fc2(F.relu(self.fc1(F.relu(self.reduce(f)).flatten(1))))


def flat_params(width, n_out):
    return count_params(FlatNet(width, n_out))


def train_flat(Fx, y, n_out, target_params, epochs=200, bs=512, lr=3e-3, seed=0):
    width = 1
    while flat_params(width + 1, n_out) <= target_params:
        width += 1
    torch.manual_seed(seed)
    net = FlatNet(width, n_out).to(Fx.device)
    yt = torch.as_tensor(y, device=Fx.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(y)
    for ep in range(epochs):
        perm = torch.randperm(n, device=Fx.device)
        for k in range(0, n, bs):
            b = perm[k:k + bs]
            out = net(Fx[b].float())
            loss = F.cross_entropy(out, yt[b]) if n_out > 1 else \
                F.binary_cross_entropy_with_logits(out.squeeze(1), yt[b].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        loss.item()                    # sync once per epoch, otherwise MPS runs out of memory
    net.eval()
    return net, width


@torch.no_grad()
def flat_predict(net, Fx, n_out):
    out = torch.cat([net(Fx[k:k + 8192].float()) for k in range(0, len(Fx), 8192)])
    return (out.argmax(1) if n_out > 1 else (out.squeeze(1) > 0).long()).cpu().numpy()


def acc(model, Fx, y, cap=None):
    p = model.predict(Fx, cap).cpu().numpy()
    return float((p.astype(int) == np.asarray(y).astype(int)).mean())


results = load_json("trees") if (RESULTS / "trees.json").exists() else {}
hier = None
for dataset, trunk, variant in CONFIGS:
    name = f"{dataset}_{trunk}_{variant}_w{args.width}"
    if args.only and name not in args.only and f"{dataset}_{trunk}_{variant}" not in args.only:
        continue
    log(f"== {name}")
    feats = load_features(dataset, trunk, dev)
    (Ftr, ytr), (Fva, yva), (Fte, yte) = feats["tree"], feats["val"], feats["test"]
    if args.flat_only:
        r = results[name]
        n_out = 10 if dataset == "cifar10" else 1
        fnet, fw = train_flat(Ftr, ytr, n_out, r["summary"]["params"])
        r["flat"] = dict(width=fw, params=count_params(fnet),
                         train=float((flat_predict(fnet, Ftr, n_out) == ytr).mean()),
                         test=float((flat_predict(fnet, Fte, n_out) == yte).mean()))
        log(f"  capacity-matched single network (width {fw}, {r['flat']['params']} params): "
            f"train {r['flat']['train']:.4f} test {r['flat']['test']:.4f}")
        save_json("trees", results)
        continue
    t0 = time.time()
    if dataset == "cifar10":
        conf = np.array(trunks["cifar10_split"]["val_confusion"])
        hier = spectral_hierarchy(conf)
        log(f"  class hierarchy: {hierarchy_str(hier, class_names('cifar10'))}")
        model = HierarchicalTreeNetwork(hier, width=args.width, variant=variant, log=log).fit(Ftr, ytr)
        max_d = max(t.depth() for t in model.splits.values())
        summ = model.summary()
        summ["per_split"] = {k: t.summary() for k, t in model.splits.items()}
    else:
        model = TreeTypeNetwork(width=args.width, variant=variant, log=log).fit(Ftr, ytr)
        max_d = model.depth()
        summ = model.summary()
    r = dict(dataset=dataset, trunk=trunk, variant=variant, width=args.width, summary=summ,
             train=acc(model, Ftr, ytr), val=acc(model, Fva, yva), test=acc(model, Fte, yte),
             seconds=round(time.time() - t0, 1))
    log(f"  train {r['train']:.4f}  val {r['val']:.4f}  test {r['test']:.4f}  {summ['nodes']} nodes, "
        f"depth {summ['depth']}, {summ['params']} params, {r['seconds']}s")

    # growth curve over depth
    growth = []
    for d in range(0, max_d + 1):
        s = model.summary(d) if dataset == "cifar10" else dict(nodes=len(model.nodes(d)),
                                                              params=model.n_params(d))
        growth.append(dict(depth=d, nodes=s["nodes"], params=s["params"], train=acc(model, Ftr, ytr, d),
                           val=acc(model, Fva, yva, d), test=acc(model, Fte, yte, d)))
    r["growth"] = growth

    # per-node record
    trees = model.splits if dataset == "cifar10" else {"T": model}
    r["nodes"] = {k: [dict(name=n.name, depth=n.depth, n=n.n, n_pos=n.n_pos, errors=n.errors,
                           const=n.const, rounds=n.rounds, seconds=round(n.seconds, 2))
                      for n in t.root.nodes()] for k, t in trees.items()}
    torch.save(model, RESULTS / "trees" / f"{name}.pt")

    # validation pruning
    if variant == "gated":
        pm = copy.deepcopy(model)
        removed = pm.prune(Fva, yva)
        ps = pm.summary()
        r["pruned"] = dict(removed=removed, nodes=ps["nodes"], params=ps["params"], depth=ps["depth"],
                           train=acc(pm, Ftr, ytr), val=acc(pm, Fva, yva), test=acc(pm, Fte, yte))
        log(f"  pruned: -{removed} nodes -> {ps['nodes']} nodes, {ps['params']} params; "
            f"train {r['pruned']['train']:.4f} test {r['pruned']['test']:.4f}")
        torch.save(pm, RESULTS / "trees" / f"{name}_pruned.pt")

    # single network with the same parameter count
    n_out = 10 if dataset == "cifar10" else 1
    fnet, fw = train_flat(Ftr, ytr, n_out, summ["params"])
    r["flat"] = dict(width=fw, params=count_params(fnet),
                     train=float((flat_predict(fnet, Ftr, n_out) == ytr).mean()),
                     test=float((flat_predict(fnet, Fte, n_out) == yte).mean()))
    log(f"  capacity-matched single network (width {fw}, {r['flat']['params']} params): "
        f"train {r['flat']['train']:.4f} test {r['flat']['test']:.4f}")
    results[name] = r
    save_json("trees", results)
    del feats, Ftr, Fva, Fte
    if dev.type == "mps":
        torch.mps.empty_cache()
log("done")
