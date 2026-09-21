"""
Gera o relatório das LSTMs (compra/venda) a partir dos artefatos gravados por
lstm_pipeline.py (`--stage train`), no mesmo formato do relatório v5 do RL:
um .md em docs/ + figuras PNG em docs/img/.

Lê, para cada (fold, lado) em <run-dir>/<tag>/fold<k>_<lado>/ :
  state.json        histórico por época (loss/val_loss/acc/val_acc), tempos
  metrics.json      métricas val/teste do melhor modelo + configuração
  predictions.npz   y e probabilidades de treino(subamostra)/val/teste, com
                    pregão e combinação (sigma, Re, Ri) de cada amostra
e <run-dir>/<tag>/label_balance.csv (balanceamento de labels por combinação).
NÃO precisa dos modelos (.keras) nem dos dados de tick: copie do cluster só o
que é leve, ex.:
  rsync -av --exclude '*.keras' --exclude '*.h5' \
      <cluster>:/scratch/ppg-lncc/$USER/lstm_runs/ ./lstm_runs/

Uso:
  python src/report_lstm.py --run-dir lstm_runs --tag lstm_base
  python src/report_lstm.py --run-dir lstm_runs --tag lstm_base --compare   # + comparação entre todos os experimentos
Saída: docs/relatorio_lstm_<tag>.md e docs/img/lstm_<tag>_fig*.png
(Requer pandas, numpy, scikit-learn e matplotlib.)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# --- estilo (paleta categórica/sequencial/divergente do dataviz, superfície clara) ---
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_TRAIN, C_VAL, C_TEST = "#2a78d6", "#eb6834", "#1baf7a"     # identidade fixa: treino/val/teste
FOLD_CMAP = LinearSegmentedColormap.from_list("folds", ["#9ec5f4", "#104281"])   # ordinal: claro->escuro
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#2a78d6", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])  # laranja - neutro - azul
SIDE_PT = {"buy": "Compra", "sell": "Venda"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.6,
    "font.size": 9,
})


def set_ticks(ax, axis, pos, labels, **kw):
    """set_xticks/yticks com rótulos, compatível com matplotlib antigos (< 3.5)."""
    if axis == "x":
        ax.set_xticks(list(pos))
        ax.set_xticklabels(labels, **kw)
    else:
        ax.set_yticks(list(pos))
        ax.set_yticklabels(labels, **kw)


def fold_colors(n):
    return [FOLD_CMAP(i / max(n - 1, 1)) for i in range(n)]


# --------------------------------------------------------------------------
# leitura
# --------------------------------------------------------------------------

def load_run(run_dir):
    """{(fold, lado): {'state', 'metrics', 'pred'}} para as tarefas do experimento."""
    tasks = {}
    for d in sorted(Path(run_dir).glob("fold*_*")):
        m = re.fullmatch(r"fold(\d+)_(buy|sell)", d.name)
        if not m:
            continue
        t = {}
        for key, fname in (("state", "state.json"), ("metrics", "metrics.json")):
            if (d / fname).exists():
                with open(d / fname, encoding="utf-8") as f:
                    t[key] = json.load(f)
        if (d / "predictions.npz").exists():
            with np.load(d / "predictions.npz") as z:
                t["pred"] = {k: z[k] for k in z.files}
        tasks[(int(m.group(1)), m.group(2))] = t
    return tasks


def done_tasks(tasks):
    return {k: v for k, v in tasks.items() if "metrics" in v and "pred" in v}


# --------------------------------------------------------------------------
# métricas
# --------------------------------------------------------------------------

def auc_day_bootstrap(y, p, orders, n_boot=300, seed=0):
    """IC95% da AUC por bootstrap sobre os PREGÕES (amostras do mesmo dia são
    correlacionadas; reamostrar amostras soltas daria um IC estreito demais)."""
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    order_idx = np.argsort(orders, kind="stable")
    y, p, orders = y[order_idx], p[order_idx], orders[order_idx]
    days, starts = np.unique(orders, return_index=True)
    groups = np.split(np.arange(len(y)), starts[1:])
    aucs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def _key(g):
    return (round(float(g[0]), 6), round(float(g[1]), 6), round(float(g[2]), 6))


def combo_rate_baseline(pred, bal, side):
    """AUC de um 'modelo' que só conhece a COMBINAÇÃO (sigma, Re, Ri) da amostra:
    score = taxa de lucro histórica da combinação nos pregões de desenvolvimento
    (label_balance.csv; não usa o teste). Mede o quanto do AUC vem só do sweep."""
    if bal is None:
        return float("nan")
    b = bal[bal["lado"] == side]
    rate = {_key((r.sigma, r.Re, r.Ri)): r.taxa_lucro for r in b.itertuples()}
    by_ci = np.array([rate.get(_key(g), np.nan) for g in pred["grid"]])
    sc = by_ci[pred["combo_test"]]
    ok = ~np.isnan(sc)
    return safe_auc(pred["y_test"][ok], sc[ok]) if ok.any() else float("nan")


def within_combo_auc(pred, mask=None):
    """AUC média DENTRO de cada combinação (ponderada por n): só compara
    oportunidades da mesma combinação, então a taxa base do sweep não ajuda."""
    y, sc, c = pred["y_test"], pred["p_test"], pred["combo_test"]
    num = den = 0.0
    for ci in np.unique(c):
        if mask is not None and not mask[ci]:
            continue
        m = c == ci
        a = safe_auc(y[m], sc[m])
        if not np.isnan(a):
            num += a * m.sum()
            den += m.sum()
    return num / den if den else float("nan")


def combo_mask(pred, common):
    """Máscara (por índice de combinação) das combinações presentes em `common`."""
    return np.array([_key(g) in common for g in pred["grid"]])


def auc_on_common(pred, common):
    """AUC no teste restrita às combinações comuns a todos os experimentos
    (necessário para comparar grades diferentes)."""
    m = combo_mask(pred, common)[pred["combo_test"]]
    return safe_auc(pred["y_test"][m], pred["p_test"][m]) if m.any() else float("nan")


# --------------------------------------------------------------------------
# utilidades de saída
# --------------------------------------------------------------------------

def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("—" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


class Figs:
    def __init__(self, img_dir, prefix):
        self.dir, self.prefix, self.n = Path(img_dir), prefix, 0
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name):
        self.n += 1
        fname = f"{self.prefix}_fig{self.n}_{name}.png"
        fig.savefig(self.dir / fname, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return f"img/{fname}"


def sides_present(tasks):
    return [s for s in ("buy", "sell") if any(k[1] == s for k in tasks)]


# --------------------------------------------------------------------------
# figuras
# --------------------------------------------------------------------------

def fig_training(tasks, folds, sides, metric, figs, name, ylabel):
    """Curvas de treino (azul) x validação (laranja) por época; linha tracejada = melhor época."""
    fig, axes = plt.subplots(len(sides), len(folds), figsize=(2.6 * len(folds) + 0.6, 2.4 * len(sides) + 0.5),
                             squeeze=False, sharey="row")
    for i, side in enumerate(sides):
        for j, fold in enumerate(folds):
            ax = axes[i][j]
            st = tasks.get((fold, side), {}).get("state")
            if st and st["history"]:
                h = pd.DataFrame(st["history"])
                ax.plot(h["epoch"], h[metric], color=C_TRAIN, label="treino")
                ax.plot(h["epoch"], h["val_" + metric], color=C_VAL, label="validação")
                best = int(h["val_loss"].idxmin())
                ax.axvline(h["epoch"].iloc[best], color=MUTED, lw=0.8, ls="--")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if metric == "acc":
                ax.axhline(0.5, color=AXIS, lw=0.8)
            ax.set_title(f"{SIDE_PT[side]} · fold {fold}")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(sides) - 1:
                ax.set_xlabel("época")
    axes[0][0].legend(loc="best")
    fig.tight_layout()
    return figs.save(fig, name)


def fig_roc_folds(done, folds, sides, figs):
    cols = fold_colors(len(folds))
    fig, axes = plt.subplots(len(sides), 2, figsize=(8.2, 3.6 * len(sides)), squeeze=False)
    for i, side in enumerate(sides):
        for j, split in enumerate(("val", "test")):
            ax = axes[i][j]
            for c, fold in zip(cols, folds):
                p = done.get((fold, side), {}).get("pred")
                if p is None or len(np.unique(p["y_" + split])) < 2:
                    continue
                fpr, tpr, _ = roc_curve(p["y_" + split], p["p_" + split])
                ax.plot(fpr, tpr, color=c, lw=1.3, label=f"fold {fold} (AUC {roc_auc_score(p['y_' + split], p['p_' + split]):.3f})")
            ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
            ax.set_title(f"{SIDE_PT[side]} · {'validação' if split == 'val' else 'teste'}")
            ax.set_xlabel("taxa de falsos positivos")
            ax.set_ylabel("taxa de verdadeiros positivos")
            ax.set_aspect("equal")
            ax.legend(loc="lower right")
    fig.tight_layout()
    return figs.save(fig, "roc_por_fold")


def fig_roc_splits(done, folds, sides, figs):
    """Último fold (o que mais viu dados): treino x validação x teste na mesma curva ROC."""
    fold = max(f for f, _ in done)
    fig, axes = plt.subplots(1, len(sides), figsize=(4.4 * len(sides), 4.0), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        p = done.get((fold, side), {}).get("pred")
        if p is not None:
            for split, col, lab in (("train", C_TRAIN, "treino"), ("val", C_VAL, "validação"), ("test", C_TEST, "teste")):
                y, s = p["y_" + split], p["p_" + split]
                if len(np.unique(y)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y, s)
                ax.plot(fpr, tpr, color=col, label=f"{lab} (AUC {roc_auc_score(y, s):.3f})")
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
        ax.set_title(f"{SIDE_PT[side]} · fold {fold}")
        ax.set_xlabel("taxa de falsos positivos")
        ax.set_ylabel("taxa de verdadeiros positivos")
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
    fig.tight_layout()
    return figs.save(fig, "roc_treino_val_teste")


def fig_confusion(done, folds, sides, figs):
    fig, axes = plt.subplots(len(sides), len(folds), figsize=(2.5 * len(folds) + 0.4, 2.6 * len(sides) + 0.4), squeeze=False)
    for i, side in enumerate(sides):
        for j, fold in enumerate(folds):
            ax = axes[i][j]
            ax.grid(False)
            p = done.get((fold, side), {}).get("pred")
            if p is None:
                ax.axis("off")
                continue
            cm = confusion_matrix(p["y_test"], (p["p_test"] >= 0.5).astype(int), labels=[0, 1])
            norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(norm, cmap=SEQ_CMAP, vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{cm[a, b]}\n{norm[a, b]:.0%}", ha="center", va="center",
                            color="white" if norm[a, b] > 0.55 else INK, fontsize=8)
            set_ticks(ax, "x", [0, 1], ["prev. prej.", "prev. lucro"], fontsize=7)
            set_ticks(ax, "y", [0, 1], ["real prej.", "real lucro"], fontsize=7)
            ax.set_title(f"{SIDE_PT[side]} · fold {fold}")
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.tight_layout()
    return figs.save(fig, "matriz_confusao_teste")


def fig_lift(done, folds, sides, figs, bal=None):
    """Taxa de lucro das oportunidades aceitas, aceitando só as de maior probabilidade."""
    cols = fold_colors(len(folds))
    fig, axes = plt.subplots(1, len(sides), figsize=(4.6 * len(sides), 3.6), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        base = []
        ref_done = False
        for c, fold in zip(cols, folds):
            p = done.get((fold, side), {}).get("pred")
            if p is None:
                continue
            if bal is not None and not ref_done:
                # referência: ordenar só pela taxa histórica da combinação (sem olhar o mercado)
                b = bal[bal["lado"] == side]
                rate = {_key((r.sigma, r.Re, r.Ri)): r.taxa_lucro for r in b.itertuples()}
                by_ci = np.array([rate.get(_key(g), np.nan) for g in p["grid"]])
                sc = by_ci[p["combo_test"]]
                ok = ~np.isnan(sc)
                if ok.any():
                    yo = p["y_test"][ok][np.argsort(-sc[ok], kind="stable")]
                    kk = np.arange(1, len(yo) + 1)
                    keep0 = kk >= 200
                    ax.plot((kk / len(yo))[keep0], (np.cumsum(yo) / kk)[keep0], color=C_VAL, lw=1.4, ls="--",
                            label="só a combinação")
                    ref_done = True
            o = np.argsort(-p["p_test"])
            y = p["y_test"][o]
            k = np.arange(1, len(y) + 1)
            frac, rate = k / len(y), np.cumsum(y) / k
            keep = k >= 200
            ax.plot(frac[keep], rate[keep], color=c, lw=1.3, label=f"fold {fold}")
            base.append(y.mean())
        if base:
            ax.axhline(np.mean(base), color=MUTED, ls="--", lw=1)
            ax.text(0.99, np.mean(base), "taxa base", color=INK2, ha="right", va="bottom", fontsize=8)
        ax.set_xscale("log")
        ax.set_title(f"{SIDE_PT[side]} · teste")
        ax.set_xlabel("fração das oportunidades aceitas (maior prob. primeiro)")
        ax.set_ylabel("taxa de lucro das aceitas")
        ax.legend(loc="best")
    fig.tight_layout()
    return figs.save(fig, "lift_teste")


def fig_calibration(done, folds, sides, figs):
    fig, axes = plt.subplots(1, len(sides), figsize=(4.2 * len(sides), 3.8), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        p = np.concatenate([done[(f, side)]["pred"]["p_test"] for f in folds if (f, side) in done])
        y = np.concatenate([done[(f, side)]["pred"]["y_test"] for f in folds if (f, side) in done])
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        mp = [p[b == i].mean() for i in range(len(edges) - 1)]
        my = [y[b == i].mean() for i in range(len(edges) - 1)]
        ax.plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1)
        ax.plot(mp, my, color=C_TEST, marker="o", ms=5)
        ax.set_title(f"{SIDE_PT[side]} · teste (folds juntos, decis)")
        ax.set_xlabel("probabilidade prevista (média do decil)")
        ax.set_ylabel("taxa de lucro observada")
    fig.tight_layout()
    return figs.save(fig, "calibracao_teste")


def fig_folds_auc(done, folds, sides, figs):
    fig, axes = plt.subplots(1, len(sides), figsize=(4.4 * len(sides), 3.4), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        for split, col, lab in (("val", C_VAL, "validação"), ("test", C_TEST, "teste")):
            xs, ys = [], []
            for f in folds:
                p = done.get((f, side), {}).get("pred")
                if p is not None:
                    xs.append(f)
                    ys.append(safe_auc(p["y_" + split], p["p_" + split]))
            ax.plot(xs, ys, color=col, marker="o", ms=5, label=lab)
        ax.axhline(0.5, color=AXIS, lw=1, ls="--")
        ax.set_xticks(folds)
        ax.set_title(f"{SIDE_PT[side]}")
        ax.set_xlabel("fold (mais dados de treino →)")
        ax.set_ylabel("AUC")
        ax.legend(loc="best")
    fig.tight_layout()
    return figs.save(fig, "auc_por_fold")


def heat_panels(df, value_col, title_fmt, cmap, center, span, figs, name, suptitle, cbar_label):
    """Um painel por sigma; linhas = Re, colunas = Ri; célula = valor (n)."""
    sigmas = sorted(df["sigma"].unique())
    res, ris = sorted(df["Re"].unique()), sorted(df["Ri"].unique())
    ncols = min(len(sigmas), 3)
    nrows = int(np.ceil(len(sigmas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols + 0.9, 2.8 * nrows + 0.6), squeeze=False)
    im = None
    for k, sg in enumerate(sigmas):
        ax = axes[k // ncols][k % ncols]
        ax.grid(False)
        grid = np.full((len(res), len(ris)), np.nan)
        ns = np.zeros((len(res), len(ris)), dtype=int)
        for _, r in df[df["sigma"] == sg].iterrows():
            grid[res.index(r["Re"]), ris.index(r["Ri"])] = r[value_col]
            ns[res.index(r["Re"]), ris.index(r["Ri"])] = int(r["n"])
        im = ax.imshow(grid, cmap=cmap, vmin=center - span, vmax=center + span, aspect="auto")
        for a in range(len(res)):
            for b in range(len(ris)):
                if not np.isnan(grid[a, b]):
                    ax.text(b, a, title_fmt.format(grid[a, b]) + f"\n(n={ns[a, b]})", ha="center", va="center",
                            fontsize=7, color=INK)
        set_ticks(ax, "x", range(len(ris)), [f"{v:g}" for v in ris])
        set_ticks(ax, "y", range(len(res)), [f"{v:g}" for v in res])
        ax.set_xlabel("Ri (risco)")
        ax.set_ylabel("Re (realização)")
        ax.set_title(f"sigma = {sg:g}")
        for sp in ax.spines.values():
            sp.set_visible(False)
    for k in range(len(sigmas), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(suptitle, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label=cbar_label)
    return figs.save(fig, name)


def combo_table(done, folds, side, split="test"):
    """AUC por combinação (sigma, Re, Ri): média entre folds das AUC calculadas em cada fold."""
    rows = {}
    grid = None
    for f in folds:
        p = done.get((f, side), {}).get("pred")
        if p is None:
            continue
        grid = p["grid"]
        y, s, c = p["y_" + split], p["p_" + split], p["combo_" + split]
        for ci in np.unique(c):
            m = c == ci
            rows.setdefault(int(ci), {"auc": [], "n": int(m.sum()), "rate": float(y[m].mean())})["auc"].append(safe_auc(y[m], s[m]))
    out = []
    for ci, v in rows.items():
        out.append({"sigma": grid[ci][0], "Re": grid[ci][1], "Ri": grid[ci][2],
                    "auc": float(np.nanmean(v["auc"])) if not np.all(np.isnan(v["auc"])) else float("nan"),
                    "n": v["n"], "taxa_lucro": v["rate"]})
    return pd.DataFrame(out)


def fig_compare(all_runs, figs, common):
    tags = sorted(all_runs)
    sides = sorted({k[1] for t in tags for k in done_tasks(all_runs[t])})
    fig, axes = plt.subplots(1, len(sides), figsize=(4.8 * len(sides), 0.75 * len(tags) + 1.9), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        for i, tag in enumerate(tags):
            d = done_tasks(all_runs[tag])
            v = [auc_on_common(t["pred"], common) for k, t in d.items() if k[1] == side]
            if not v:
                continue
            ax.errorbar(np.nanmean(v), i, xerr=np.nanstd(v), fmt="o", color=C_TEST, ecolor=AXIS, capsize=3)
            ax.text(np.nanmean(v), i - 0.22, f"{np.nanmean(v):.3f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.axvline(0.5, color=AXIS, ls="--", lw=1)
        set_ticks(ax, "y", range(len(tags)), tags)
        ax.set_ylim(len(tags) - 0.5, -0.7)   # primeiro experimento no topo
        ax.set_title(f"{SIDE_PT[side]} · AUC no teste")
        ax.set_xlabel("AUC nas combinações comuns (média ± desvio entre folds)")
    fig.tight_layout()
    return figs.save(fig, "comparacao_experimentos")


# --------------------------------------------------------------------------
# relatório
# --------------------------------------------------------------------------

def build_report(run_root, tag, out_dir, compare=False, n_boot=300, exclude=("lstm_smoke",), notes=None, compare_prefix=""):
    run_dir = Path(run_root) / tag
    tasks = load_run(run_dir)
    done = done_tasks(tasks)
    if not done:
        raise SystemExit(f"Nenhuma tarefa concluída (com metrics.json + predictions.npz) em {run_dir}")
    folds = sorted({f for f, _ in tasks})
    dfolds = sorted({f for f, _ in done})
    sides = sides_present(done)
    figs = Figs(Path(out_dir) / "img", f"lstm_{tag}")
    cfg = next(iter(done.values()))["metrics"].get("config", {})
    L = []   # linhas do markdown

    bal_path = run_dir / "label_balance.csv"
    bal = pd.read_csv(bal_path) if bal_path.exists() else None

    # ---------------- tabela por fold ----------------
    rows = []
    for (fold, side), t in sorted(done.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        m, p, st = t["metrics"], t["pred"], t["state"]
        lo, hi = auc_day_bootstrap(p["y_test"], p["p_test"], p["order_test"], n_boot=n_boot)
        h = pd.DataFrame(st["history"])
        rate = float(p["y_test"].mean())
        rows.append({
            "lado": SIDE_PT[side], "fold": fold, "épocas": m["epochs"],
            "melhor época": int(h["val_loss"].idxmin()) + 1, "n treino": m["n_train"],
            "n val": m["val"]["n"], "n teste": m["test"]["n"],
            "AUC val": m["val"]["auc"], "AUC teste": m["test"]["auc"],
            "IC95% teste": f"[{lo:.3f}; {hi:.3f}]",
            "AUC só-combo": combo_rate_baseline(p, bal, side), "AUC intra-combo": within_combo_auc(p),
            "acc teste": m["test"]["acc"], "acc majoritária": max(rate, 1 - rate),
            "bal_acc teste": m["test"]["bal_acc"], "prec teste": m["test"]["prec"], "rec teste": m["test"]["rec"],
            "min treinando": h["sec"].sum() / 60,
            "_lo": lo, "_hi": hi, "_side": side,
        })
    tab = pd.DataFrame(rows)

    # ---------------- resumo automático (só fatos) ----------------
    L += [f"# Relatório LSTM — experimento `{tag}`", "",
          "> Gerado por `src/report_lstm.py` a partir dos artefatos gravados pelo pipeline "
          "(`state.json`, `metrics.json`, `predictions.npz`). Tudo aqui é medido; nenhuma interpretação é gerada "
          "automaticamente — a leitura dos resultados fica a cargo de quem analisa.", "", "## 0. Resumo", ""]
    for side in sides:
        d = tab[tab["_side"] == side]
        auc = d["AUC teste"].to_numpy()
        gain = (d["acc teste"] - d["acc majoritária"]).mean()
        base_auc, intra = d["AUC só-combo"].mean(), d["AUC intra-combo"].mean()
        L.append(f"- **{SIDE_PT[side]}**: AUC no teste = **{auc.mean():.3f}** em média entre {len(d)} fold(s) "
                 f"(desvio {auc.std(ddof=1) if len(auc) > 1 else float('nan'):.3f}; mín {auc.min():.3f}, máx {auc.max():.3f}); "
                 f"{int((auc > 0.5).sum())} de {len(d)} folds acima de 0,5. "
                 f"Último fold: {d.iloc[-1]['AUC teste']:.3f} {d.iloc[-1]['IC95% teste']} (IC95% por bootstrap sobre os pregões). "
                 f"Acurácia no teste menos a do classificador majoritário: {gain:+.3f} (média). "
                 f"Referências: AUC de um score que só conhece a combinação (sigma, Re, Ri) = {base_auc:.3f}; "
                 f"AUC do modelo apenas dentro de cada combinação = {intra:.3f}.")
    L += ["", "Uma AUC de 0,5 é o acaso; o IC95% que contém 0,5 significa que, com esses dados, não dá para distinguir o "
          "modelo do acaso naquele fold. `AUC só-combo` usa como score a taxa de lucro histórica da combinação (sem olhar o "
          "mercado): é o que se ganha só por saber quais parâmetros geraram a oportunidade. `AUC intra-combo` compara apenas "
          "oportunidades da mesma combinação, então o efeito do sweep é removido.", ""]

    if notes:   # leitura manual (arquivo .md à parte, para sobreviver à regeneração do relatório)
        L += ["---", "", Path(notes).read_text(encoding="utf-8").strip(), ""]

    # ---------------- 1. configuração ----------------
    L += ["---", "", "## 1. Configuração e execução", ""]
    if cfg:
        g = np.array(cfg["grid"])
        L += [md_table(pd.DataFrame([
            ["Objetivo", "classificar cada oportunidade (entrada da heurística) como Lucro (1) ou Prejuízo (0); "
                         "label = sinal do lucro realizado da negociação (SG se atinge o alvo, −SL se atinge o stop, "
                         "resultado a mercado se fechada no fim do pregão)"],
            ["Sweep (sigma × Re × Ri)", f"{len(g)} combinações: sigma {sorted(set(map(float, g[:, 0])))}, Re {sorted(set(map(float, g[:, 1])))}, Ri {sorted(set(map(float, g[:, 2])))}"],
            ["Features (por tick)", f"{len(cfg['features'])}: " + ", ".join(cfg["features"])],
            ["Janela", f"{cfg['n_ticks']} ticks anteriores à entrada"],
            ["Rede", f"2× LSTM({cfg['units']}) + Dense(1, sigmoid)" + (f", dropout {cfg['dropout']}" if cfg["dropout"] else "")],
            ["Treino", f"Adam, binary_crossentropy, batch {cfg['batch_size']}, até {cfg['max_epochs']} épocas, "
                       f"early stopping (paciência {cfg['patience']}, melhor val_loss); "
                       + ("class_weight" if cfg.get("class_weight") else "oversampling" if cfg.get("oversampling") else "sem balanceamento")
                       + (f"; semente {cfg['seed']}" if cfg.get("seed") else "")],
            ["Validação", "TimeSeriesSplit expanding window sobre os pregões (split por dia); scaler e balanceamento ajustados só no treino do fold"],
        ], columns=["Item", "Valor"])), ""]
        jr = []
        for f in dfolds:
            c = next(t for k, t in done.items() if k[0] == f)["metrics"]["config"]
            jr.append([f, f"{c['train_orders'][0]}–{c['train_orders'][1]} ({c['n_train_days']})",
                       f"{c['val_orders'][0]}–{c['val_orders'][1]} ({c['n_val_days']})",
                       f"{c['test_orders'][0]}–{c['test_orders'][1]} ({c['n_test_days']})"])
        L += ["**Janelas** (pregões, com o nº de dias entre parênteses; o teste é o mesmo em todos os folds):", "",
              md_table(pd.DataFrame(jr, columns=["fold", "treino", "validação", "teste"])), ""]
    n_missing = len(tasks) - len(done)
    if n_missing:
        L += [f"> ⚠️ {n_missing} tarefa(s) ainda sem `metrics.json`/`predictions.npz` (não concluídas): "
              + ", ".join(f"fold{f}-{s}" for (f, s) in sorted(tasks) if (f, s) not in done) + ".", ""]

    # ---------------- 2. balanceamento ----------------
    if bal is not None:
        L += ["---", "", "## 2. Balanceamento dos labels por combinação", "",
              "Taxa de lucro de cada combinação (pregões de desenvolvimento). Laranja = maioria de prejuízos, "
              "azul = maioria de lucros; o cinza é 50%.", ""]
        for side in sides:
            b = bal[bal["lado"] == side].rename(columns={"taxa_lucro": "v"})
            if b.empty:
                continue
            L += [f"![Balanceamento {side}]({heat_panels(b, 'v', '{:.0%}', DIV_CMAP, 0.5, 0.25, figs, 'balanco_' + side, 'Taxa de lucro — ' + SIDE_PT[side], 'taxa de lucro')})", ""]
        tot = bal.groupby("lado")[["n", "lucros", "prejuizos", "fechados_forcado"]].sum().reset_index()
        tot["taxa de lucro"] = tot["lucros"] / tot["n"]
        tot["lado"] = tot["lado"].map(SIDE_PT)
        L += [md_table(tot, "{:.3f}"), ""]

    # ---------------- 3. treino ----------------
    L += ["---", "", "## 3. Curvas de treino e validação", "",
          "Linha tracejada = melhor época (menor `val_loss`, a que o early stopping restaura).", ""]
    L += [f"![Loss]({fig_training(tasks, folds, sides, 'loss', figs, 'curva_loss', 'binary crossentropy')})", "",
          "> A loss de **treino** do Keras já incorpora o `class_weight` (ponderada), enquanto a de validação não; "
          "por isso as duas não são diretamente comparáveis em nível — o que importa é a forma (quando a validação "
          "para de melhorar e sobe).", "",
          f"![Acurácia]({fig_training(tasks, folds, sides, 'acc', figs, 'curva_acuracia', 'acurácia')})", ""]

    # ---------------- 4. resultados ----------------
    L += ["---", "", "## 4. Resultados: validação e teste", ""]
    show = tab.drop(columns=["_lo", "_hi", "_side"]).copy()
    L += [md_table(show), "",
          "`acc majoritária` é a acurácia de um classificador que sempre prevê a classe mais comum no teste — o "
          "piso que a acurácia do modelo precisa superar. `IC95%` = bootstrap sobre os pregões do teste.", "",
          f"![AUC por fold]({fig_folds_auc(done, dfolds, sides, figs)})", "",
          "### Curva ROC", "",
          f"![ROC por fold]({fig_roc_folds(done, dfolds, sides, figs)})", "",
          f"![ROC treino/val/teste]({fig_roc_splits(done, dfolds, sides, figs)})", "",
          "A distância entre a curva de treino e as de validação/teste (último fold) mede o sobreajuste.", "",
          "### Matriz de confusão (teste, limiar 0,5)", "",
          "Cada célula: contagem e % da linha (classe real).", "",
          f"![Matriz de confusão]({fig_confusion(done, dfolds, sides, figs)})", "",
          "### Utilidade para operar: aceitar só as melhores oportunidades", "",
          "Ordena as oportunidades do teste pela probabilidade prevista de lucro e mostra a taxa de lucro das "
          "aceitas conforme se aceita mais ou menos delas. Se o modelo discrimina, a curva começa acima da taxa base "
          "e decai até ela. A linha tracejada laranja é a referência **só-combo**: aceitar as oportunidades na ordem da taxa "
          "de lucro histórica de cada combinação (sigma, Re, Ri), sem olhar o mercado. Só o que fica acima dela é "
          "informação além do sweep.", "",
          f"![Lift]({fig_lift(done, dfolds, sides, figs, bal)})", "",
          "### Calibração", "",
          f"![Calibração]({fig_calibration(done, dfolds, sides, figs)})", "",
          "> Com `class_weight` as probabilidades não são calibradas para a taxa real; o que vale é o ranking (AUC/lift), "
          "não o limiar 0,5.", ""]

    # ---------------- 5. por combinação ----------------
    L += ["---", "", "## 5. Onde o modelo discrimina: AUC por combinação (sigma, Re, Ri)", "",
          "AUC no teste de cada combinação (média entre folds). Azul > 0,5; laranja < 0,5; células com menos de "
          "uma centena de amostras são ruidosas.", ""]
    for side in sides:
        ct = combo_table(done, dfolds, side)
        if ct.empty:
            continue
        L += [f"![AUC por combinação {side}]({heat_panels(ct, 'auc', '{:.2f}', DIV_CMAP, 0.5, 0.15, figs, 'auc_combo_' + side, 'AUC no teste por combinação — ' + SIDE_PT[side], 'AUC')})", ""]

    # ---------------- 6. comparação ----------------
    if compare:
        all_runs = {d.name: load_run(d) for d in sorted(Path(run_root).glob("*"))
                    if d.is_dir() and d.name not in exclude and d.name.startswith(compare_prefix) and load_run(d)}
        if len(all_runs) > 1:
            # combinações comuns a TODOS os experimentos: grades diferentes geram conjuntos de
            # teste diferentes, então só nelas a comparação é justa
            grids = [set(_key(g) for g in next(iter(done_tasks(tk).values()))["pred"]["grid"]) for tk in all_runs.values()]
            common = set.intersection(*grids)
            crow = []
            for t_name, tk in sorted(all_runs.items()):
                d = done_tasks(tk)
                for side in ("buy", "sell"):
                    ks = [k for k in d if k[1] == side]
                    if not ks:
                        continue
                    v = [d[k]["metrics"]["test"]["auc"] for k in ks]
                    vc = [auc_on_common(d[k]["pred"], common) for k in ks]
                    ic = [within_combo_auc(d[k]["pred"], combo_mask(d[k]["pred"], common)) for k in ks]
                    crow.append({"experimento": t_name, "lado": SIDE_PT[side], "folds": len(v),
                                 "combos": len(next(iter(d.values()))["pred"]["grid"]),
                                 "AUC teste (todas)": np.mean(v),
                                 "AUC teste (combos comuns)": np.mean(vc),
                                 "desvio (folds)": np.std(vc, ddof=1) if len(vc) > 1 else float("nan"),
                                 "AUC intra-combo (comuns)": np.mean(ic),
                                 "bal_acc teste": np.mean([d[k]["metrics"]["test"]["bal_acc"] for k in ks])})
            L += ["---", "", "## 6. Comparação entre experimentos", "",
                  f"Comparação feita nas **{len(common)} combinações comuns** a todos os experimentos (as grades diferem, "
                  "e o conjunto de teste de cada um depende da sua grade; na coluna `todas` cada experimento usa a própria).", "",
                  f"![Comparação]({fig_compare(all_runs, figs, common)})", "", md_table(pd.DataFrame(crow)), ""]
            if "lstm_base" in all_runs and "lstm_seed2" in all_runs:
                db, ds = done_tasks(all_runs["lstm_base"]), done_tasks(all_runs["lstm_seed2"])
                diffs = {sd: [abs(auc_on_common(db[k]["pred"], common) - auc_on_common(ds[k]["pred"], common))
                              for k in db if k in ds and k[1] == sd] for sd in ("buy", "sell")}
                L += ["**Ruído entre execuções** (`lstm_base` × `lstm_seed2`, só muda a semente): diferença absoluta média da AUC por fold = "
                      + "; ".join(f"{SIDE_PT[sd]} {np.mean(v):.3f} (máx {np.max(v):.3f})" for sd, v in diffs.items() if v) + ". "
                      "Diferenças entre experimentos menores que isso não devem ser lidas como melhora.", ""]
            else:
                L += ["Diferenças menores que o desvio entre folds (ou que a variação entre sementes) não devem ser lidas como melhora.", ""]

    # ---------------- limitações ----------------
    L += ["---", "", "## 7. Limitações e ressalvas", "",
          "- **É classificação, não P/L.** O label vem da regra fixa SG/SL da heurística, sem custos de execução "
          "(meio-spread, taxas, atraso). Um bom AUC aqui não implica lucro operável; o passo seguinte seria medir o "
          "P/L (com custos) de operar só as oportunidades aceitas pelo modelo.",
          "- **Amostras correlacionadas.** Combinações diferentes da grade no mesmo pregão compartilham quase as mesmas "
          "entradas, e janelas de 120 ticks de entradas próximas se sobrepõem; o número efetivo de amostras "
          "independentes é bem menor que `n`. Por isso os ICs são por pregão.",
          "- **Um único período de teste** (os últimos pregões, fixos), e poucos folds de validação: a variação entre "
          "folds mostra a instabilidade temporal, mas não substitui mais dados.",
          "- **Balanceamento por construção.** A taxa de lucro depende do sweep (sobretudo de Ri); um modelo pode "
          "aprender a taxa por combinação a partir das features SL/SG. A seção 5 mostra se há discriminação dentro de cada combinação.",
          "", "## Apêndice: reprodução", "",
          "```", f"python src/report_lstm.py --run-dir {run_root} --tag {tag}" + (" --compare" if compare else ""),
          "```", ""]

    out = Path(out_dir) / f"relatorio_lstm_{tag}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Relatório: {out}  ({figs.n} figuras em {figs.dir})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="lstm_runs", help="pasta que contém uma subpasta por experimento (tag)")
    ap.add_argument("--tag", default="lstm_base")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--compare", action="store_true", help="inclui a comparação com os demais experimentos de --run-dir")
    ap.add_argument("--boot", type=int, default=300, help="reamostragens do bootstrap por pregão")
    ap.add_argument("--exclude", nargs="*", default=["lstm_smoke"],
                    help="experimentos ignorados na comparação (padrão: o teste rápido lstm_smoke)")
    ap.add_argument("--compare-prefix", default="",
                    help="só compara experimentos cujo nome começa com este prefixo (ex.: lstm_v2_)")
    ap.add_argument("--notes", default=None, help="arquivo .md com a leitura manual, inserido logo após o resumo")
    a = ap.parse_args()
    build_report(a.run_dir, a.tag, a.out_dir, compare=a.compare, n_boot=a.boot, exclude=tuple(a.exclude), notes=a.notes,
                 compare_prefix=a.compare_prefix)
