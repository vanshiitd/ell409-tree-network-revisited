"""Shared convolutional trunk.

The trunk is trained ONCE, on its own disjoint data split, and then frozen. Every tree node reads the
same mid-level feature map ("tap") produced by the trunk, so the expensive convolutional features are
paid for once instead of once per node.

    32x32 input :            stage1 (32ch) -> 16x16 -> stage2 (64ch) -> 8x8  = TAP (64 x 8 x 8)
    64x64 input : stem(32ch) -> 32x32 -> stage1 -> 16x16 -> stage2 -> 8x8  = TAP
    pretraining only:        TAP -> stage3 (128ch) -> 4x4 -> global average pool -> linear
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class Trunk(nn.Module):
    def __init__(self, in_size=32, n_classes=10, widths=(32, 64, 128)):
        super().__init__()
        w1, w2, w3 = widths
        stem = [conv_bn(3, w1), nn.MaxPool2d(2)] if in_size == 64 else []
        cin = w1 if in_size == 64 else 3
        self.pre = nn.Sequential(*stem,
                                 conv_bn(cin, w1), conv_bn(w1, w1), nn.MaxPool2d(2),
                                 conv_bn(w1, w2), conv_bn(w2, w2), nn.MaxPool2d(2))
        self.post = nn.Sequential(conv_bn(w2, w3), conv_bn(w3, w3), nn.MaxPool2d(2))
        self.head = nn.Linear(w3, n_classes)
        self.in_size = in_size
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))

    def normalize(self, x_uint8_nhwc):
        x = x_uint8_nhwc.permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    def tap(self, x):            # x: normalised NCHW float
        return self.pre(x)

    def forward(self, x):
        return self.head(self.post(self.pre(x)).mean((2, 3)))


def _augment(x):
    """Random crop (pad 4, reflect) + horizontal flip, on a normalised NCHW batch."""
    B, C, H, W = x.shape
    p = H // 8
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    i = torch.randint(0, 2 * p + 1, (1,)).item()
    j = torch.randint(0, 2 * p + 1, (1,)).item()
    x = xp[:, :, i:i + H, j:j + W]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    return x


def train_trunk(X, y, n_classes, device, epochs=30, bs=128, lr=0.05, wd=5e-4, seed=0, log=print):
    """Supervised training of the trunk on its own split. X: uint8 NHWC numpy."""
    torch.manual_seed(seed)
    net = Trunk(in_size=X.shape[1], n_classes=n_classes).to(device)
    xf = X.reshape(-1, 3).astype(np.float64) / 255.0
    net.mean.copy_(torch.tensor(xf.mean(0), dtype=torch.float32))
    net.std.copy_(torch.tensor(xf.std(0), dtype=torch.float32))
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    steps = epochs * math.ceil(len(y) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(y), device=device)
        t0, tot, correct = time.time(), 0.0, 0
        for k in range(0, len(y), bs):
            b = perm[k:k + bs]
            xb = _augment(net.normalize(Xt[b]))
            out = net(xb)
            loss = F.cross_entropy(out, yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            correct += (out.argmax(1) == yt[b]).sum().item()
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"    trunk epoch {ep + 1}/{epochs}: loss {tot / len(y):.3f} "
                f"train-acc {correct / len(y):.3f} ({time.time() - t0:.1f}s)")
    net.eval()
    return net


def random_trunk(X, n_classes, device, seed=0):
    """Untrained trunk; BatchNorm statistics are calibrated on X so the random features are well scaled."""
    torch.manual_seed(seed)
    net = Trunk(in_size=X.shape[1], n_classes=n_classes).to(device)
    xf = X.reshape(-1, 3).astype(np.float64) / 255.0
    net.mean.copy_(torch.tensor(xf.mean(0), dtype=torch.float32))
    net.std.copy_(torch.tensor(xf.std(0), dtype=torch.float32))
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None                     # cumulative average over the calibration pass
    net.train()
    Xt = torch.tensor(X, device=device)
    with torch.no_grad():
        for k in range(0, len(X), 500):
            net(net.normalize(Xt[k:k + 500]))
    net.eval()
    return net


@torch.no_grad()
def predict_logits(net, X, device, bs=1000):
    net.eval()
    out = []
    for k in range(0, len(X), bs):
        xb = torch.tensor(X[k:k + bs], device=device)
        out.append(net(net.normalize(xb)).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def extract_tap(net, X, device, bs=1000, resize_to=None):
    """Frozen mid-level features (N, 64, 8, 8) as float16 numpy."""
    net.eval()
    out = []
    for k in range(0, len(X), bs):
        xb = torch.tensor(X[k:k + bs], device=device)
        x = net.normalize(xb)
        if resize_to is not None and x.shape[-1] != resize_to:
            x = F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False,
                              antialias=True)
        out.append(net.tap(x).half().cpu())
    return torch.cat(out).numpy()


def n_params(module):
    return sum(p.numel() for p in module.parameters())
