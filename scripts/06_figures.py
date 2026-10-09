"""Step 6: summary figures and LaTeX tables for the report, from results/*.json.

Outputs: figures/{growth,double_descent,minimize,tree_catsdogs,hierarchy_cifar10}.pdf, report/tables/*.tex
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

from treenet.common import FIGURES, RESULTS, ROOT, load_json
from treenet.data import class_names
from treenet.style import AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, YELLOW, setup

setup()
TAB = ROOT / "report" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
trees = load_json("trees")
trunks = load_json("trunks")
W = 8
NICE = dict(catsdogs="Cats vs Dogs", cifar10="CIFAR-10")


def pct(x):
    return f"{100 * x:.1f}"


def ptrain(x):
    return "100" if x >= 0.99995 else f"{100 * x:.2f}"


# ------------------------------------------------------------------ main results table
rows = []
order = [("catsdogs", "split", "gated"), ("catsdogs", "split", "faithful"), ("catsdogs", "transfer", "gated"),
         ("catsdogs", "random", "gated"), ("cifar10", "split", "gated"), ("cifar10", "split", "faithful"),
         ("cifar10", "random", "gated")]
trunk_label = dict(split="own split", transfer="CIFAR-10", random="random")
body = [r"\begin{tabular}{@{}l l l r r r r r r r@{}}", r"\toprule",
        r" & & & \multicolumn{4}{c}{grown tree (until error-free or depth 40)} & \multicolumn{2}{c}{after pruning} & one net\\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}",
        r"dataset & trunk & children & nodes & depth & train & test & nodes & test & test\\", r"\midrule"]
last = None
for ds, tr, var in order:
    k = f"{ds}_{tr}_{var}_w{W}"
    if k not in trees:
        continue
    r = trees[k]
    s = r["summary"]
    pr = r.get("pruned")
    name = NICE[ds] if ds != last else ""
    if last is not None and ds != last:
        body.append(r"\addlinespace")
    last = ds
    body.append(f"{name} & {trunk_label[tr]} & {var} & {s['nodes']} & {s['depth']} & {ptrain(r['train'])} & "
                f"{pct(r['test'])} & {pr['nodes'] if pr else '--'} & {pct(pr['test']) if pr else '--'} & "
                f"{pct(r['flat']['test'])}\\\\")
body += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_main.tex").write_text("\n".join(body) + "\n")

# ------------------------------------------------------------------ growth curves
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
for ax, ds, ttl in zip(axes, ("catsdogs", "cifar10"), ("(a) Cats vs Dogs", "(b) CIFAR-10")):
    g = trees.get(f"{ds}_split_gated_w{W}")
    if g is None:
        continue
    gr = g["growth"]
    n = [x["nodes"] for x in gr]
    ax.plot(n, [100 * x["train"] for x in gr], "-o", color=BLUE, mec="white", mew=0.6, ms=3.5,
            label="train, gated")
    ax.plot(n, [100 * x["test"] for x in gr], "-o", color=ORANGE, mec="white", mew=0.6, ms=3.5,
            label="test, gated")
    f = trees.get(f"{ds}_split_faithful_w{W}")
    if f is not None:
        gf = f["growth"]
        nf = [x["nodes"] for x in gf]
        ax.plot(nf, [100 * x["train"] for x in gf], "--", color=BLUE, lw=1.1, label="train, faithful")
        ax.plot(nf, [100 * x["test"] for x in gf], "--", color=ORANGE, lw=1.1, label="test, faithful")
    te = 100 * trunks[f"{ds}_split"]["accuracy"]["test"]
    ax.axhline(te, color=MUTED, ls=":", lw=1)
    ax.text(0.02, te + 0.4, "end-to-end trunk CNN (test)", color=INK2, fontsize=6.8, va="bottom",
            transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
    ax.set_xscale("log")
    ax.set_xlabel("nodes in the tree truncated at depth d = 0, 1, 2, ...")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(ttl, loc="left", color=INK)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=7)
fig.tight_layout(w_pad=2)
fig.savefig(FIGURES / "growth.pdf")
plt.close(fig)

# ------------------------------------------------------------------ double descent
if (RESULTS / "double_descent.json").exists():
    dd = load_json("double_descent")
    keys = [k for k in ("catsdogs_eta0.0_seed0", "catsdogs_eta0.2_seed0", "cifar10_eta0.0_seed0",
                        "cifar10_eta0.2_seed0") if k in dd]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8))
    for ax, k in zip(axes.flat, keys):
        rows = dd[k]
        w = np.array([r["width"] for r in rows])
        caps = [("0", "root network only", BLUE), ("2", "tree, depth $\\leq$ 2", AQUA), ("100", "full tree", ORANGE)]
        for cap, lab, col in caps:
            te = [100 * r["caps"][cap]["test_err"] for r in rows]
            ax.plot(w, te, "-o", color=col, ms=3.2, mec="white", mew=0.5, label=f"test, {lab}")
        tr0 = [100 * r["caps"]["0"]["train_err"] for r in rows]
        ax.plot(w, tr0, "--", color=BLUE, lw=1.1, label="train, root only")
        interp = [r["width"] for r in rows if r["caps"]["0"]["train_err"] == 0]
        if interp:
            ax.axvline(interp[0], color=MUTED, ls=":", lw=1)
            ax.text(interp[0] * 1.08, ax.get_ylim()[1] * 0.97 if False else 2, "root\ninterpolates", color=INK2,
                    fontsize=6.5, va="bottom")
        ds, eta = k.split("_")[0], float(k.split("_eta")[1].split("_")[0])
        ax.set_xscale("log", base=2)
        ax.set_title(f"{NICE[ds]}, {int(100 * eta)}% label noise", loc="left", color=INK)
        ax.set_xlabel("node width w")
        ax.set_ylabel("error (%)")
        ax.set_ylim(bottom=0)
    axes.flat[0].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "double_descent.pdf")
    plt.close(fig)

# ------------------------------------------------------------------ minimisation
if (RESULTS / "minimize.json").exists():
    mz = load_json("minimize")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ds, col in (("catsdogs", BLUE), ("cifar10", ORANGE)):
        sw = mz.get(ds, {}).get("width_sweep", [])
        if not sw:
            continue
        w = np.array([r["width"] for r in sw])
        prm = np.array([r["params"] for r in sw])
        ok = np.array([r["train"] >= 0.9999 for r in sw])          # reached the best reachable training accuracy
        axes[0].plot(w, prm, "-", color=col, lw=1.2, label=NICE[ds])
        axes[0].plot(w[ok], prm[ok], "o", color=col, mec="white", mew=0.6)
        axes[0].plot(w[~ok], prm[~ok], "o", mfc="white", mec=col, mew=1.2)
        for r in sw:
            axes[0].annotate(str(r["nodes"]), (r["width"], r["params"]), textcoords="offset points",
                             xytext=(0, 5), ha="center", fontsize=6, color=INK2)
        axes[1].plot(w, [100 * r["test"] for r in sw], "-o", color=col, mec="white", mew=0.6, ms=3.5,
                     label=f"{NICE[ds]}, full tree")
        axes[1].plot(w, [100 * r["pruned"]["test"] for r in sw], "--", color=col, lw=1.1,
                     label=f"{NICE[ds]}, after pruning")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("node width w (labels: number of nodes)")
    axes[0].set_ylabel("node parameters in the tree")
    axes[0].set_title("(a) size of the grown tree", loc="left", color=INK)
    axes[0].legend(loc="upper left")
    axes[0].text(0.03, 0.66, "hollow marker: depth cap hit\nbefore 100% train", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=6.3, color=INK2)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("node width w")
    axes[1].set_ylabel("test accuracy (%)")
    axes[1].set_title("(b) test accuracy", loc="left", color=INK)
    axes[1].set_ylim(58, 92)
    axes[1].legend(loc="lower left", fontsize=6.2)
    fig.tight_layout(w_pad=2)
    fig.savefig(FIGURES / "minimize.pdf")
    plt.close(fig)

    def num(x):
        return f"{x:,}".replace(",", "{,}")

    def ptr(x):
        return "100" if x >= 0.99995 else f"{100 * x:.2f}"

    body = [r"\begin{tabular}{@{}l l r r r r r@{}}", r"\toprule",
            r"dataset & model & nodes & node parameters & non-zero & train (\%) & test (\%)\\", r"\midrule"]
    for ds in ("catsdogs", "cifar10"):
        m = mz.get(ds, {})
        main = trees.get(f"{ds}_split_gated_w{W}")
        if not main:
            continue
        s_ = main["summary"]
        body.append(f"{NICE[ds]} & full tree, $w={W}$ & {s_['nodes']} & {num(s_['params'])} & {num(s_['params'])} & "
                    f"{ptr(main['train'])} & {pct(main['test'])}\\\\")
        sw = m.get("width_sweep", [])
        interp = [r for r in sw if r["train"] >= 0.9999]
        if interp:
            best = min(interp, key=lambda r: r["params"])
            if best["width"] != W:
                body.append(f" & smallest interpolating tree, $w={best['width']}$ & {best['nodes']} & "
                            f"{num(best['params'])} & {num(best['params'])} & {ptr(best['train'])} & "
                            f"{pct(best['test'])}\\\\")
        wp = m.get("weight_pruning")
        if wp:
            body.append(f" & $w={W}$ + magnitude pruning inside nodes & {s_['nodes']} & {num(wp['dense_params'])} & "
                        f"{num(wp['nonzero_params'])} & {ptr(wp['train_after'])} & {pct(wp['test_after'])}\\\\")
        pr = main.get("pruned")
        if pr:
            body.append(f" & $w={W}$ + reduced-error pruning & {pr['nodes']} & {num(pr['params'])} & "
                        f"{num(pr['params'])} & {ptr(pr['train'])} & {pct(pr['test'])}\\\\")
        if ds == "catsdogs":
            body.append(r"\addlinespace")
    body += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_minimize.tex").write_text("\n".join(body) + "\n")


# ------------------------------------------------------------------ tree diagram (Cats vs Dogs, top levels)
def draw_tree(nodes, ax, max_depth=4, title=""):
    by = {n["name"]: n for n in nodes}
    pos = {}

    def layout(name, x0, x1):
        n = by[name]
        pos[name] = ((x0 + x1) / 2, -n["depth"])
        kids = [c for c in (name + "A", name + "B") if c in by and by[c]["depth"] <= max_depth]
        if len(kids) == 2:
            layout(kids[0], x0, (x0 + x1) / 2)
            layout(kids[1], (x0 + x1) / 2, x1)
        elif len(kids) == 1:
            layout(kids[0], x0 + (x1 - x0) * (0.0 if kids[0].endswith("A") else 0.5),
                   x0 + (x1 - x0) * (0.5 if kids[0].endswith("A") else 1.0))

    layout("r", 0, 1)
    for name, (x, y) in pos.items():
        for c in (name + "A", name + "B"):
            if c in pos:
                cx, cy = pos[c]
                ax.plot([x, cx], [y, cy], color=AQUA if c.endswith("A") else ORANGE, lw=1.1, zorder=1)
    hidden = [n for n in nodes if n["depth"] > max_depth]
    for name, (x, y) in pos.items():
        n = by[name]
        if n["const"] is not None:
            ax.scatter([x], [y], s=18, marker="s", color=MUTED, zorder=2)
            continue
        ax.scatter([x], [y], s=26 + 90 * np.sqrt(n["n"] / by["r"]["n"]), color=BLUE, edgecolors="white",
                   linewidths=0.6, zorder=2)
        if n["depth"] <= 2:
            ax.text(x, y + 0.22, f"n={n['n']}\nerr={n['errors']}", ha="center", va="bottom", fontsize=5.6,
                    color=INK2)
    ax.set_ylim(-max_depth - 0.6, 1.0)
    ax.axis("off")
    if hidden:
        ax.text(0.5, -max_depth - 0.55, f"+ {len(hidden)} deeper nodes not drawn", ha="center", fontsize=6.5,
                color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)


main = trees.get(f"catsdogs_split_gated_w{W}")
if main:
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    draw_tree(main["nodes"]["T"], ax, max_depth=5,
              title="Cats vs Dogs tree (gated, w=8): aqua edges = child A (rescues misses), "
                    "orange = child B (vetoes false alarms), grey squares = constant units")
    fig.savefig(FIGURES / "tree_catsdogs.pdf")
    plt.close(fig)

# ------------------------------------------------------------------ CIFAR-10 class hierarchy with per-split tree sizes
main = trees.get(f"cifar10_split_gated_w{W}")
if main:
    from treenet.tree import spectral_hierarchy
    names = class_names("cifar10")
    hier = spectral_hierarchy(np.array(trunks["cifar10_split"]["val_confusion"]))
    per = main["summary"]["per_split"]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    xs = {}
    order_leaves = []

    def collect(h):
        if isinstance(h, int):
            order_leaves.append(h)
        else:
            collect(h[0])
            collect(h[1])

    collect(hier)
    for i, c in enumerate(order_leaves):
        xs[c] = i

    def draw(h, key, depth):
        if isinstance(h, int):
            x, y = xs[h], 0
            ax.text(x, y - 0.15, names[h], rotation=40, ha="right", va="top", fontsize=7, color=INK)
            return x, y
        xl, yl = draw(h[0], key + "0", depth + 1)
        xr, yr = draw(h[1], key + "1", depth + 1)
        y = max(yl, yr) + 1
        ax.plot([xl, xl, xr, xr], [yl, y, y, yr], color=MUTED, lw=1)
        s = per[key]
        x = (xl + xr) / 2
        ax.scatter([x], [y], s=20 + 4 * s["nodes"], color=BLUE, edgecolors="white", zorder=3)
        ax.text(x, y + 0.18, f"{key}: {s['nodes']} nodes", ha="center", va="bottom", fontsize=6, color=INK2)
        return x, y

    draw(hier, "H", 0)
    ax.axis("off")
    ax.set_title("CIFAR-10 class hierarchy (spectral bisection of the trunk's confusion matrix); "
                 "marker size = nodes in that split's tree", loc="left", color=INK, fontsize=8)
    fig.savefig(FIGURES / "hierarchy_cifar10.pdf")
    plt.close(fig)

# ------------------------------------------------------------------ receptive-field centrality by depth
if (RESULTS / "receptive_fields.json").exists():
    rf = load_json("receptive_fields")
    bins = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 99, "11+")]
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    for off, (ds, col) in zip((-0.12, 0.12), (("catsdogs", BLUE), ("cifar10", ORANGE))):
        for b, (lo, hi, lab) in enumerate(bins):
            v = [r["centrality"] for r in rf[ds] if lo <= r["depth"] <= hi and r["centrality"] == r["centrality"]]
            if not v:
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([b + off] * 2, [q1, q3], color=col, lw=2.2, alpha=0.5, solid_capstyle="round")
            ax.plot([b + off], [med], "o", color=col, mec="white", mew=0.6, label=NICE[ds] if b == 0 else None)
            ax.text(b + off, q3 + 0.012, str(len(v)), ha="center", fontsize=5.8, color=INK2)
    ax.axhline(0.25, color=MUTED, ls=":", lw=0.9)
    ax.text(5.45, 0.255, "uniform", color=INK2, fontsize=6.5, ha="right", va="bottom")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b[2] for b in bins])
    ax.set_xlabel("node depth (numbers: nodes per bin)")
    ax.set_ylabel("Grad-CAM mass in central 4x4")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "rf_centrality.pdf")
    plt.close(fig)
print("figures and tables written")
