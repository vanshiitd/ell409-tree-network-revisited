"""Tree-type network with a small neural network at every node (after Jayadeva).

A node g fits the frozen trunk features. Its samples fall into C1 (g=0, t=0), C2 (g=1, t=1),
C3 (g=0, t=1, a miss) and C4 (g=1, t=0, a false alarm). Child A corrects C3 and child B corrects C4.

variant="faithful": A and B are trained on all of the parent's samples, y = (g or A) and not B.
variant="gated": A only sees g=0 and B only g=1 (with flipped labels), y = A(x) if g(x)=0 else not B(x).
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeNet(nn.Module):
    """1x1 conv (64 -> k) -> ReLU -> linear (h) -> ReLU -> linear (1), with h = width, k = ceil(width / 4)."""

    def __init__(self, width=16, c_in=64, spatial=8):
        super().__init__()
        k = max(1, math.ceil(width / 4))
        self.width, self.k = width, k
        self.reduce = nn.Conv2d(c_in, k, 1)
        self.fc1 = nn.Linear(k * spatial * spatial, width)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, f):
        z = F.relu(self.reduce(f)).flatten(1)
        return self.fc2(F.relu(self.fc1(z))).squeeze(1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def calibrate(z, t):
    """Threshold on the logits that minimises training errors, excluding constant outputs.
    Returns (tau, errors); the node predicts z > tau."""
    order = torch.argsort(z, descending=True)
    zs, ts = z[order], t[order]
    n, n_pos = len(t), t.sum()
    tp = torch.cumsum(ts, 0)
    k = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    errors = (n_pos - tp) + (k - tp)
    errors = errors[:-1]                           # exclude "all positive" (constant)
    if len(errors) == 0:
        return z.min() - 1, int(min(n_pos.item(), n - n_pos.item()))
    valid = zs[:-1] > zs[1:]
    errors = torch.where(valid, errors, torch.full_like(errors, float("inf")))
    j = int(torch.argmin(errors).item())
    if not torch.isfinite(errors[j]):              # all logits tied: no split exists -> majority label
        if n_pos.item() > n / 2:
            return zs[-1] - 1, int(n - n_pos.item())
        return zs[0] + 1, int(n_pos.item())
    tau = (zs[j] + zs[j + 1]) / 2
    return tau, int(errors[j].item())


def train_node(Fx, t, width, seed=0, lr=3e-3, bs=512, max_rounds=120, patience=20, min_steps=20):
    """Fit one node with a class-balanced logistic loss, then calibrate its threshold.
    Returns (net, pred_bool, n_errors, rounds_used)."""
    n = len(t)
    torch.manual_seed(seed)
    net = NodeNet(width).to(Fx.device)
    n_pos = float(t.sum().item())
    pos_weight = torch.tensor(min(1e4, max(1e-4, (n - n_pos) / max(n_pos, 1.0))), device=Fx.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = min(bs, n)
    steps_per_round = max(math.ceil(n / bs), min_steps)
    best_err, best_state, best_tau, stall = n + 1, None, 0.0, 0
    for rnd in range(max_rounds):
        net.train()
        for _ in range(steps_per_round):
            b = torch.randint(0, n, (bs,), device=Fx.device)
            loss = F.binary_cross_entropy_with_logits(net(Fx[b].float()), t[b], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        tau, err = calibrate(node_logit(net, Fx), t)
        if err < best_err:
            best_err, stall, best_tau = err, 0, float(tau)
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stall += 1
        if best_err == 0 or stall >= patience:
            break
    net.load_state_dict(best_state)
    with torch.no_grad():
        net.fc2.bias -= best_tau                   # fold the threshold into the output bias
    net.eval()
    pred = node_predict(net, Fx)
    return net, pred, int((pred != (t > 0.5)).sum().item()), rnd + 1


@torch.no_grad()
def node_predict(net, Fx, bs=8192):
    net.eval()
    return torch.cat([net(Fx[k:k + bs].float()) > 0 for k in range(0, len(Fx), bs)])


@torch.no_grad()
def node_logit(net, Fx, bs=8192):
    net.eval()
    return torch.cat([net(Fx[k:k + bs].float()) for k in range(0, len(Fx), bs)])


@dataclass
class TreeNode:
    name: str
    depth: int
    n: int = 0
    n_pos: int = 0
    errors: int = 0
    const: int = None          # pure sample set: constant output, no network
    net: NodeNet = None
    A: "TreeNode" = None
    B: "TreeNode" = None
    rounds: int = 0
    seconds: float = 0.0
    idx: np.ndarray = field(default=None, repr=False)
    target: np.ndarray = field(default=None, repr=False)

    def nodes(self):
        out = [self]
        for c in (self.A, self.B):
            if c is not None:
                out += c.nodes()
        return out


class TreeTypeNetwork:
    """Binary tree-type network of small neural-network nodes."""

    def __init__(self, width=16, variant="gated", max_depth=40, seed=0, keep_indices=False, log=None):
        assert variant in ("gated", "faithful")
        self.width, self.variant, self.max_depth, self.seed = width, variant, max_depth, seed
        self.keep_indices = keep_indices
        self.log = log
        self.root = None
        self._count = 0

    def __getstate__(self):
        d = self.__dict__.copy()
        d["log"] = None
        return d

    def fit(self, Fx, y, idx=None):
        t0 = time.time()
        y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=Fx.device)
        idx = np.arange(len(y)) if idx is None else np.asarray(idx)
        self.root = self._grow(Fx, y, idx, 0, "r")
        self.fit_seconds = time.time() - t0
        return self

    def _grow(self, Fx, t, idx, depth, name):
        node = TreeNode(name=name, depth=depth, n=len(t), n_pos=int(t.sum().item()))
        if self.keep_indices:
            node.idx, node.target = idx, t.cpu().numpy().astype(np.int8)
        if node.n_pos in (0, node.n):
            node.const = int(node.n_pos > 0)
            return node
        t0 = time.time()
        seed = self.seed * 100003 + self._count
        self._count += 1
        net, pred, err, rounds = train_node(Fx, t, self.width, seed=seed)
        for retry in range(1, 6):                               # constant output, e.g. a dead ReLU
            if not (bool(pred.all()) or not bool(pred.any())):
                break
            net, pred, err, rounds = train_node(Fx, t, self.width, seed=seed + 7 * retry,
                                                lr=1e-2 if retry % 2 else 3e-3)
        node.net, node.errors, node.rounds, node.seconds = net, err, rounds, time.time() - t0
        if self.log and depth <= 2:
            self.log(f"      node {name:<8} n={node.n:<6} pos={node.n_pos:<6} errors={err:<5} "
                     f"({node.seconds:.1f}s)")
        # a constant node would hand its children the same problem, so stop here
        no_progress = err > 0 and (bool(pred.all()) or not bool(pred.any()))
        if no_progress and self.log:
            self.log(f"      node {name}: no progress (n={node.n}, pos={node.n_pos}, errors={err})")
        if err == 0 or depth >= self.max_depth or no_progress:
            return node
        tb = t > 0.5
        if self.variant == "gated":
            mA, mB = ~pred, pred
            if (tb & mA).any():
                node.A = self._grow(Fx[mA], t[mA], idx[mA.cpu().numpy()], depth + 1, name + "A")
            if (~tb & mB).any():
                node.B = self._grow(Fx[mB], 1 - t[mB], idx[mB.cpu().numpy()], depth + 1, name + "B")
        else:
            c3, c4 = (~pred) & tb, pred & ~tb
            if c3.any():
                node.A = self._grow(Fx, c3.float(), idx, depth + 1, name + "A")
            if c4.any():
                node.B = self._grow(Fx, c4.float(), idx, depth + 1, name + "B")
        return node

    @torch.no_grad()
    def predict(self, Fx, max_depth=None):
        cap = self.max_depth if max_depth is None else max_depth
        return self._predict(self.root, Fx, cap)

    def _predict(self, node, Fx, cap):
        if len(Fx) == 0:
            return torch.zeros(0, dtype=torch.bool, device=Fx.device)
        if node.const is not None:
            return torch.full((len(Fx),), bool(node.const), device=Fx.device)
        g = node_predict(node.net, Fx)
        if node.depth >= cap:
            return g
        if self.variant == "gated":
            out = g.clone()
            if node.A is not None:
                m = ~g
                out[m] = self._predict(node.A, Fx[m], cap)
            if node.B is not None:
                m = g
                out[m] = ~self._predict(node.B, Fx[m], cap)
            return out
        a = self._predict(node.A, Fx, cap) if node.A is not None else torch.zeros_like(g)
        b = self._predict(node.B, Fx, cap) if node.B is not None else torch.zeros_like(g)
        return (g | a) & ~b

    def nodes(self, max_depth=None):
        cap = self.max_depth if max_depth is None else max_depth
        return [n for n in self.root.nodes() if n.depth <= cap]

    def n_params(self, max_depth=None):
        return sum(count_params(n.net) if n.net is not None else 1 for n in self.nodes(max_depth))

    def depth(self):
        return max(n.depth for n in self.root.nodes())

    def summary(self):
        ns = self.root.nodes()
        return dict(nodes=len(ns), networks=sum(n.net is not None for n in ns), depth=self.depth(),
                    params=self.n_params(), fit_seconds=round(getattr(self, "fit_seconds", 0.0), 1),
                    root_errors=self.root.errors, width=self.width, variant=self.variant)

    @torch.no_grad()
    def prune(self, Fv, yv):
        """Reduced-error pruning, bottom-up. Gated variant only."""
        assert self.variant == "gated"
        yv = torch.as_tensor(np.asarray(yv), dtype=torch.bool, device=Fv.device)
        return self._prune(self.root, Fv, yv)

    def _prune(self, node, Fv, yv):
        if node.const is not None or (node.A is None and node.B is None) or len(Fv) == 0:
            if len(Fv) == 0 and (node.A is not None or node.B is not None):
                k = len(node.nodes()) - 1
                node.A = node.B = None
                return k
            return 0
        g = node_predict(node.net, Fv)
        removed = 0
        if node.A is not None:
            removed += self._prune(node.A, Fv[~g], yv[~g])
        if node.B is not None:
            removed += self._prune(node.B, Fv[g], ~yv[g])
        with_children = (self._predict(node, Fv, 10 ** 9) == yv).sum().item()
        alone = (g == yv).sum().item()
        if alone >= with_children:
            k = len(node.nodes()) - 1
            node.A = node.B = None
            removed += k
        return removed


def spectral_hierarchy(confusion, classes=None):
    """Recursively bisect the classes along the Fiedler vector of the confusion graph, keeping
    confused classes together. Returns nested tuples."""
    C = np.asarray(confusion, dtype=float)
    classes = list(range(len(C))) if classes is None else list(classes)
    if len(classes) == 1:
        return classes[0]
    if len(classes) == 2:
        return (classes[0], classes[1])
    A = C[np.ix_(classes, classes)]
    A = A + A.T
    np.fill_diagonal(A, 0)
    A += 1e-6
    d = A.sum(1)
    L = np.eye(len(classes)) - A / np.sqrt(np.outer(d, d))
    w, v = np.linalg.eigh(L)
    f = v[:, 1] / np.sqrt(d)
    order = np.argsort(f)
    # best normalised cut along the Fiedler ordering, sides of size >= 2
    best, best_k = None, None
    for k in range(1, len(classes)):
        if min(k, len(classes) - k) < min(2, len(classes) // 2):
            continue
        S, T = order[:k], order[k:]
        cut = A[np.ix_(S, T)].sum()
        val = cut / A[S].sum() + cut / A[T].sum()
        if best is None or val < best:
            best, best_k = val, k
    left = sorted(classes[i] for i in order[:best_k])
    right = sorted(classes[i] for i in order[best_k:])
    return (spectral_hierarchy(C, left), spectral_hierarchy(C, right))


def leaves(h):
    return [h] if isinstance(h, (int, np.integer)) else leaves(h[0]) + leaves(h[1])


def hierarchy_str(h, names):
    if isinstance(h, (int, np.integer)):
        return names[h]
    return "(" + hierarchy_str(h[0], names) + " | " + hierarchy_str(h[1], names) + ")"


class HierarchicalTreeNetwork:
    """Multiclass tree: a binary class hierarchy where each split is a TreeTypeNetwork trained on the
    samples of the classes below it."""

    def __init__(self, hierarchy, width=16, variant="gated", max_depth=40, seed=0, log=None):
        self.h, self.width, self.variant, self.max_depth, self.seed, self.log = \
            hierarchy, width, variant, max_depth, seed, log
        self.splits = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["log"] = None
        return d

    def fit(self, Fx, y):
        t0 = time.time()
        y = np.asarray(y)
        self._fit(self.h, Fx, y, np.arange(len(y)), "H")
        self.fit_seconds = time.time() - t0
        return self

    def _fit(self, h, Fx, y, idx, key):
        if isinstance(h, (int, np.integer)):
            return
        L, R = leaves(h[0]), leaves(h[1])
        mask = np.isin(y[idx], L + R)
        sub = idx[mask]
        t = np.isin(y[sub], R).astype(np.float32)
        if self.log:
            self.log(f"    split {key}: {len(L)} vs {len(R)} classes, {len(sub)} samples")
        tree = TreeTypeNetwork(self.width, self.variant, self.max_depth,
                               seed=self.seed + 31 * len(self.splits), log=self.log)
        sel = torch.as_tensor(sub, device=Fx.device)
        tree.fit(Fx[sel], t, idx=sub)
        self.splits[key] = tree
        self._fit(h[0], Fx, y, sub[t == 0], key + "0")
        self._fit(h[1], Fx, y, sub[t == 1], key + "1")

    @torch.no_grad()
    def predict(self, Fx, max_depth=None):
        out = torch.zeros(len(Fx), dtype=torch.long, device=Fx.device)
        self._predict(self.h, Fx, torch.arange(len(Fx), device=Fx.device), "H", out, max_depth)
        return out

    def _predict(self, h, Fx, rows, key, out, max_depth):
        if len(rows) == 0:
            return
        if isinstance(h, (int, np.integer)):
            out[rows] = int(h)
            return
        go_right = self.splits[key].predict(Fx[rows], max_depth)
        self._predict(h[0], Fx, rows[~go_right], key + "0", out, max_depth)
        self._predict(h[1], Fx, rows[go_right], key + "1", out, max_depth)

    def summary(self, max_depth=None):
        trees = self.splits.values()
        return dict(splits=len(self.splits),
                    nodes=sum(len(t.nodes(max_depth)) for t in trees),
                    networks=sum(sum(n.net is not None for n in t.nodes(max_depth)) for t in trees),
                    depth=max(t.depth() for t in trees),
                    params=sum(t.n_params(max_depth) for t in trees),
                    fit_seconds=round(getattr(self, "fit_seconds", 0.0), 1),
                    width=self.width, variant=self.variant)

    def prune(self, Fv, yv):
        yv = np.asarray(yv)
        removed = 0
        for key, tree in self.splits.items():
            h = self.h
            for c in key[1:]:
                h = h[int(c)]
            L, R = leaves(h[0]), leaves(h[1])
            m = np.isin(yv, L + R)
            sel = torch.as_tensor(np.where(m)[0], device=Fv.device)
            removed += tree.prune(Fv[sel], np.isin(yv[m], R))
        return removed
