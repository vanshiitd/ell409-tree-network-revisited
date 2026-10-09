"""Tree-type network whose nodes are small neural networks (Jayadeva's construction, generalised).

Every node v owns a sample set S_v and binary targets t_v, and fits a small network g_v on the frozen
trunk features. Its training samples fall into the four categories of Table I of the original paper:

    C1: g=0, t=0 (correct)   C2: g=1, t=1 (correct)   C3: g=0, t=1 (miss)   C4: g=1, t=0 (false alarm)

Child A corrects C3 and child B corrects C4. Two ways of training the children are implemented:

  variant="faithful"  -- the construction used in the first assignment: A and B are trained on ALL of S_v,
                         A with target 1 on C3 and 0 elsewhere, B with target 1 on C4 and 0 elsewhere.
                         Prediction: y = (g OR A) AND NOT B.

  variant="gated"     -- uses the "don't care" (X) entries of Table II exactly. A's output only matters
                         where g=0 and B's only where g=1, so A is trained on {g=0} = C1 u C3 with the true
                         labels and B on {g=1} = C2 u C4 with flipped labels. Prediction:
                         y = A(x) if g(x)=0 else NOT B(x).
                         The children's sample sets PARTITION S_v, so each level of the tree costs one pass
                         over the data instead of one pass per node, and class imbalance at deep nodes is
                         milder.

Both variants reach zero training error when allowed to grow (a node that can isolate one point always
makes progress), which is the overfitting regime the assignment asks for.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------- node
class NodeNet(nn.Module):
    """A small neural network on the trunk's 64x8x8 feature map:
         1x1 conv (64 -> k) -> ReLU -> flatten (64k) -> linear (h) -> ReLU -> linear (1)
       with h = width and k = ceil(width / 4)."""

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
    """Error-minimising threshold on the node's logits (the paper's 'minimise the number of
    misclassified samples'), restricted to non-constant decisions so the node always splits its set.
    Returns (threshold tau, errors). The node then predicts z > tau."""
    order = torch.argsort(z, descending=True)
    zs, ts = z[order], t[order]
    n, n_pos = len(t), t.sum()
    tp = torch.cumsum(ts, 0)                       # predicting the top-k as positive
    k = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    errors = (n_pos - tp) + (k - tp)               # misses + false alarms
    errors = errors[:-1]                           # exclude "all positive" (constant)
    if len(errors) == 0:
        return z.min() - 1, int(min(n_pos.item(), n - n_pos.item()))
    # thresholds only between distinct logits
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
    """Fit one node to (Fx, t): class-balanced logistic loss for the ranking, then an error-minimising
    threshold (calibrate). Trains until the node makes no training errors or stops improving.
    Fx: (n, 64, 8, 8) float16 tensor on the device, t: (n,) float tensor in {0,1}.
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


# ----------------------------------------------------------------------------------------- tree
@dataclass
class TreeNode:
    name: str
    depth: int
    n: int = 0                 # training samples seen by the node
    n_pos: int = 0
    errors: int = 0            # node's own training errors (before its children correct them)
    const: int = None          # set for a pure sample set: constant output, no network
    net: NodeNet = None
    A: "TreeNode" = None
    B: "TreeNode" = None
    rounds: int = 0
    seconds: float = 0.0
    idx: np.ndarray = field(default=None, repr=False)   # global indices of the node's training samples
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

    # ---------------------------------------------------------------- training
    def fit(self, Fx, y, idx=None):
        """Fx: (N,64,8,8) float16 tensor on device; y: (N,) {0,1} numpy/tensor; idx: optional global indices."""
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
        if node.n_pos in (0, node.n):                         # pure set: a constant unit suffices
            node.const = int(node.n_pos > 0)
            return node
        t0 = time.time()
        seed = self.seed * 100003 + self._count
        self._count += 1
        net, pred, err, rounds = train_node(Fx, t, self.width, seed=seed)
        for retry in range(1, 6):                               # constant output (e.g. a dead ReLU): retry
            if not (bool(pred.all()) or not bool(pred.any())):
                break
            net, pred, err, rounds = train_node(Fx, t, self.width, seed=seed + 7 * retry,
                                                lr=1e-2 if retry % 2 else 3e-3)
        node.net, node.errors, node.rounds, node.seconds = net, err, rounds, time.time() - t0
        if self.log and depth <= 2:
            self.log(f"      node {name:<8} n={node.n:<6} pos={node.n_pos:<6} errors={err:<5} "
                     f"({node.seconds:.1f}s)")
        # a node that outputs a constant hands its children the same problem again -> stop there
        no_progress = err > 0 and (bool(pred.all()) or not bool(pred.any()))
        if no_progress and self.log:
            self.log(f"      node {name}: no progress (n={node.n}, pos={node.n_pos}, errors={err})")
        if err == 0 or depth >= self.max_depth or no_progress:
            return node
        tb = t > 0.5
        if self.variant == "gated":
            mA, mB = ~pred, pred
            if (tb & mA).any():                                # C3 non-empty -> child A on {g=0}
                node.A = self._grow(Fx[mA], t[mA], idx[mA.cpu().numpy()], depth + 1, name + "A")
            if (~tb & mB).any():                               # C4 non-empty -> child B on {g=1}
                node.B = self._grow(Fx[mB], 1 - t[mB], idx[mB.cpu().numpy()], depth + 1, name + "B")
        else:
            c3, c4 = (~pred) & tb, pred & ~tb
            if c3.any():
                node.A = self._grow(Fx, c3.float(), idx, depth + 1, name + "A")
            if c4.any():
                node.B = self._grow(Fx, c4.float(), idx, depth + 1, name + "B")
        return node

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, Fx, max_depth=None):
        """Boolean predictions; max_depth truncates the tree (nodes at that depth act as leaves)."""
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

    # ---------------------------------------------------------------- bookkeeping
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

    # ---------------------------------------------------------------- pruning
    @torch.no_grad()
    def prune(self, Fv, yv):
        """Reduced-error pruning (bottom-up): replace a node's subtree by the node itself whenever that does
        not lower accuracy on the validation samples that reach it. Only for the gated variant, whose
        routing makes the samples reaching each node well defined."""
        assert self.variant == "gated"
        yv = torch.as_tensor(np.asarray(yv), dtype=torch.bool, device=Fv.device)
        removed = self._prune(self.root, Fv, yv)
        return removed

    def _prune(self, node, Fv, yv):
        if node.const is not None or (node.A is None and node.B is None) or len(Fv) == 0:
            if len(Fv) == 0 and (node.A is not None or node.B is not None):   # no validation evidence
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


# ----------------------------------------------------------------------------- multiclass
def spectral_hierarchy(confusion, classes=None):
    """Recursively bisect the classes so that the most-confused classes stay together (Fiedler vector of
    the normalised Laplacian of the symmetrised confusion graph). Returns nested tuples."""
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
    # choose the cut along the Fiedler ordering that minimises the normalised cut, sides of size >= 2
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
    """Multiclass tree (Section 5.3 of the first report): a binary class hierarchy whose every split is a
    binary tree-type network trained on the samples of the classes below it. 100% training accuracy at
    every split implies 100% training accuracy overall, because every training sample is routed correctly
    at every level."""

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
        """Prune each split's tree on the validation samples whose true class belongs to that split."""
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
