# Tree-type networks of small neural networks: CIFAR-10 and Cats vs Dogs

ELL409 (Machine Learning and Optimization), IIT Delhi. Graded assignment *"Tree type network revisited"*.
Vansh Saini (2023EE10656).

This repository builds Jayadeva's tree-type network with every node replaced by a small neural network.
It grows the tree until it fits the training set (CIFAR-10 and Cats vs Dogs), looks for double descent
beyond the interpolation threshold, visualises what each node responds to, and shrinks the network.
The write-up is `report/report.pdf`.

## How the model works

```
image ──► frozen CNN trunk (trained once, on a split the tree never sees) ──► 64×8×8 feature map
                                                                                │
                      ┌─────────────────────── tree of small node networks ◄────┘
                      │   node g_v: 1×1 conv (64→k) → ReLU → linear (64k→w) → ReLU → linear (w→1)
                      │   child A corrects g_v's misses (C3), child B its false alarms (C4)
                      └── grown until every training sample is classified correctly
```

* **Features.** A small CNN trunk is trained once, on a disjoint *trunk split*, and frozen. Every node
  reads the same mid-level feature map, so the convolutional features are paid for once. Variants: the
  dataset's own split (main), an untrained random trunk, and (for Cats vs Dogs) the CIFAR-10 trunk.
* **Nodes.** Small networks, about 1.2k parameters at the default width w = 8, trained with a
  class-balanced logistic loss until they make no errors on their own sample set (or stop improving).
* **Children.** Two variants of the child construction:
  * `faithful`, as in the first assignment: A and B are trained on the parent's whole sample set.
  * `gated`: uses the paper's "don't care" entries exactly. A is trained only where the parent outputs 0,
    B only where it outputs 1, so the children partition the parent's data.
* **Multiclass (CIFAR-10).** A class hierarchy is found by spectral bisection of the trunk's validation
  confusion matrix. Each of its 9 splits is a binary tree-type network.

## Main results

| dataset | trunk | children | nodes | depth | train | test | test after pruning | one net, same size |
|---|---|---|---|---|---|---|---|---|
| Cats vs Dogs | own split | gated | 11 | 5 | 99.99% | 86.1% | 88.8% (1 node) | 88.9% |
| Cats vs Dogs | own split | faithful | 104 | 40 | 99.98% | 84.4% | – | 88.8% |
| CIFAR-10 | own split | gated | 96 | 23 | 100% | 66.7% | 71.4% (13 nodes) | 76.7% |
| CIFAR-10 | own split | faithful | 453 | 40 | 95.16% | 66.5% | – | 80.3% |

* 99.99% is the maximum on Cats vs Dogs: Flickr's "photo unavailable" placeholder appears three times
  in the training split with labels dog, dog, cat.
* With 20% label noise, test error shows double descent in node width. It peaks where a single node
  first interpolates (Cats vs Dogs 31.9%, CIFAR-10 about 58%) and falls again for wider nodes (24.1%, 35.5%).
* Deep correction nodes move their Grad-CAM attention from the object to the background and memorise
  single images. Validation pruning removes 86–91% of the nodes and raises test accuracy.
* With small nodes (w = 2), the size of a sample's final node set flags flipped labels (AUC 0.66 / 0.74).

## Layout

| path | contents |
|---|---|
| `treenet/data.py` | dataset loading, caching, disjoint trunk / tree / val / test splits |
| `treenet/trunk.py` | the shared CNN trunk, its training, random trunk, feature extraction |
| `treenet/tree.py` | node network, node training, binary tree-type network (both variants), pruning, hierarchical multiclass tree |
| `treenet/style.py`, `treenet/common.py` | plotting style, paths, device, logging |
| `scripts/00_prepare_data.py` | decode both datasets into cached arrays |
| `scripts/01_trunks.py` | train trunks, cache frozen features |
| `scripts/02_trees.py` | grow the trees, growth curves, pruning, baselines |
| `scripts/03_double_descent.py` | width sweep with and without label noise |
| `scripts/04_receptive_fields.py` | Grad-CAM, activation maximisation and top images for every node |
| `scripts/05_minimize.py` | width sweep for the smallest interpolating tree, weight pruning inside nodes |
| `scripts/06_figures.py` | summary figures and LaTeX tables |
| `scripts/07_noise_detection.py` | extra: the tree as a detector of mislabelled training examples |
| `results/` | logs and JSON results of every run |
| `figures/` | all figures used in the report |
| `report/` | LaTeX source and PDF of the report |

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the data into `data/`:

* CIFAR-10: either `cifar-10-python.tar.gz` from https://www.cs.toronto.edu/~kriz/cifar.html, or the
  two parquet files of the Hugging Face mirror `uoft-cs/cifar10` into `data/cifar10_hf/{train,test}.parquet`.
* Cats vs Dogs: Microsoft's `kagglecatsanddogs_5340.zip`, extracted to `data/catsdogs/PetImages/`.

Then run the steps in order:

```bash
python scripts/00_prepare_data.py
python scripts/01_trunks.py
python scripts/02_trees.py
python scripts/03_double_descent.py
python scripts/04_receptive_fields.py
python scripts/05_minimize.py
python scripts/06_figures.py
python scripts/07_noise_detection.py 2   # node width 2 (and 8 for the negative control)
```

Everything was run on an Apple M5 laptop (PyTorch MPS backend). CUDA and CPU work too. All seeds are fixed.
