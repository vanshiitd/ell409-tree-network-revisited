"""Step 1: train the trunks and cache their features.

split: trained on the dataset's trunk split. random: untrained. transfer: the CIFAR-10 trunk applied to
Cats vs Dogs resized to 32x32.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from treenet.common import Logger, RESULTS, device, feature_path, save_json
from treenet.data import get_splits
from treenet.trunk import extract_tap, n_params, predict_logits, random_trunk, train_trunk

log = Logger("01_trunks")
dev = device()
log(f"device: {dev}")
summary = {}


def cache_features(dataset, name, net, splits, resize_to=None):
    feats = {s: extract_tap(net, splits[s][0], dev, resize_to=resize_to) for s in ("tree", "val", "test")}
    # standardise per channel with the tree split's statistics
    mu = feats["tree"].astype(np.float32).mean(axis=(0, 2, 3), keepdims=True)
    sd = feats["tree"].astype(np.float32).std(axis=(0, 2, 3), keepdims=True) + 1e-6
    out = {}
    for s in feats:
        out[f"F_{s}"] = ((feats[s].astype(np.float32) - mu) / sd).astype(np.float16)
        out[f"y_{s}"] = splits[s][1]
    np.savez(feature_path(dataset, name), mu=mu, sd=sd, **out)
    log(f"  cached {feature_path(dataset, name).name}: " +
        ", ".join(f"{s} {out['F_' + s].shape}" for s in feats))


trunks = {}
for dataset, n_classes in (("cifar10", 10), ("catsdogs", 2)):
    log(f"== {dataset}")
    sp = get_splits(dataset)
    for s, (X, y) in sp.items():
        log(f"  split {s:<5}: {X.shape}, class counts {np.bincount(y).tolist()}")
    Xtr, ytr = sp["trunk"]

    log("  training the supervised trunk on the trunk split")
    net = train_trunk(Xtr, ytr, n_classes, dev, epochs=40, log=log)
    torch.save(net.state_dict(), RESULTS / "trunks" / f"{dataset}_split.pt")
    trunks[dataset] = net
    accs = {}
    for s in ("trunk", "val", "test"):
        pred = predict_logits(net, sp[s][0], dev).argmax(1)
        accs[s] = float((pred == sp[s][1]).mean())
    val_pred = predict_logits(net, sp["val"][0], dev).argmax(1)
    conf = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(conf, (sp["val"][1], val_pred), 1)
    log(f"  trunk accuracy: {accs}")
    summary[f"{dataset}_split"] = dict(accuracy=accs, params=n_params(net),
                                       params_tap=n_params(net.pre), val_confusion=conf.tolist())
    cache_features(dataset, "split", net, sp)

    log("  random (untrained) trunk")
    rnet = random_trunk(Xtr, n_classes, dev)
    torch.save(rnet.state_dict(), RESULTS / "trunks" / f"{dataset}_random.pt")
    summary[f"{dataset}_random"] = dict(params_tap=n_params(rnet.pre))
    cache_features(dataset, "random", rnet, sp)

    if dataset == "catsdogs":
        log("  transfer trunk: CIFAR-10 trunk on images resized to 32x32")
        cnet = trunks["cifar10"]
        cache_features(dataset, "transfer", cnet, sp, resize_to=32)
        summary["catsdogs_transfer"] = dict(params_tap=n_params(cnet.pre), source="cifar10_split")

save_json("trunks", summary)
log("done")
