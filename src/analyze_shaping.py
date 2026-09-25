"""
Figuras da rodada de shaping de recompensa (docs/relatorio_diagnostico_sintetico.md, §8):
10 execuções (`syn_shape_{A,B}_{none,opp_flat,opp_wrong,pbrs,regret}_s0`, 30M timesteps),
par sintético cointegrado, versão hedgeada. A = features da própria versão hedgeada
(spread_compra/venda); B = só preços + tempo cíclico, sem spread (shaping ensinado pelo
spread verdadeiro, informação privilegiada só na recompensa).

Só lê CSV/JSON já gravados pelo pipeline. Não precisa dos dados de tick.

Uso (a partir da raiz do repo, no WSL/venv com matplotlib):
    python src/analyze_shaping.py [backup]
    # default: sdumont_backup_shaping/pairs-trading-rl  ou sdumont_backup_shaping (ambos testados)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BACKUP = sys.argv[1] if len(sys.argv) > 1 else "sdumont_backup_shaping"
if not os.path.isdir(os.path.join(BACKUP, "src")):
    BACKUP = os.path.join(BACKUP, "pairs-trading-rl")
RDIR = os.path.join(BACKUP, "src", "runs")
OUT_IMG = os.path.join("docs", "img")
os.makedirs(OUT_IMG, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
KIND_COLOR = {"none": "#8a8985", "opp_flat": "#eda100", "opp_wrong": "#e34948",
              "pbrs": "#2a78d6", "regret": "#4a3aa7"}
KIND_ORDER = ["none", "opp_flat", "opp_wrong", "pbrs", "regret"]
KIND_LABEL = {"none": "controle (sem shaping)", "opp_flat": "opp_flat", "opp_wrong": "opp_wrong",
              "pbrs": "pbrs", "regret": "regret"}
ROUND_LABEL = {"A": "A: features da versão hedgeada", "B": "B: só preços + tempo (sem spread)"}
OPT_RULE = {"val": 69.0, "test": 62.9}    # R$/dia, mesmos 30 dias do baseline (ver §2)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 1.8, "font.size": 9,
    "axes.axisbelow": True,
})


def rdir(tag):
    return os.path.join(RDIR, f"syn_shape_{tag}_s0", "fold1")


def load_ck(tag):
    return pd.read_csv(os.path.join(rdir(tag), "eval", "checkpoint_curve.csv"))


def load_summary_row(tag, label):
    s = pd.read_csv(os.path.join(rdir(tag), "eval", "summary.csv"))
    return s[s.label == label].iloc[0]


def fig_checkpoints():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, rnd in zip(axs, ("A", "B")):
        for kind in KIND_ORDER:
            ck = load_ck(f"{rnd}_{kind}")
            ax.plot(ck.timesteps / 1e6, ck.test_pnl_total / 30, "o-", ms=4,
                    color=KIND_COLOR[kind], label=KIND_LABEL[kind])
        ax.axhline(OPT_RULE["test"], color=INK, lw=1.1, ls="--", label="regra ótima causal")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(ROUND_LABEL[rnd])
        ax.set_xlabel("timesteps (milhões)")
    axs[0].set_ylabel("P/L real médio por dia, teste (R$)")
    axs[1].legend(loc="lower right", ncol=1)
    fig.suptitle("pbrs converge cedo e fica estável; os outros formatos oscilam (30 pregões de teste)", y=1.03, fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_IMG, "v7_fig1_shaping_checkpoints.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", path)


def fig_fracao():
    rows = []
    for rnd in ("A", "B"):
        for kind in KIND_ORDER:
            tag = f"{rnd}_{kind}"
            row = load_summary_row(tag, "best_val_test_fee1.0")
            rows.append({"rodada": rnd, "kind": kind, "pnl_dia": row.real_pnl_hedged_brl / 30,
                        "fracao": row.real_pnl_hedged_brl / 30 / OPT_RULE["test"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.35
    xa = range(len(KIND_ORDER))
    for i, rnd in enumerate(("A", "B")):
        sub = df[df.rodada == rnd].set_index("kind").loc[KIND_ORDER]
        colors = [KIND_COLOR[k] for k in KIND_ORDER]
        ax.bar([x + (i - 0.5) * w for x in xa], sub.fracao * 100, w * 0.92,
               color=colors, alpha=(1.0 if rnd == "A" else 0.55),
               edgecolor=INK, linewidth=0.4 if rnd == "A" else 0.0,
               hatch=None if rnd == "A" else "////")
    ax.axhline(100, color=INK, lw=1.0, ls="--")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(list(xa))
    ax.set_xticklabels([KIND_LABEL[k] for k in KIND_ORDER], rotation=15, ha="right")
    ax.set_ylabel("% da regra ótima causal capturado (teste)")
    ax.set_title("Fração da regra ótima capturada, por formato de shaping")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#8a8985", label="A: features hedged"),
                       Patch(facecolor="#8a8985", alpha=0.55, hatch="////", label="B: só preços")],
              loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT_IMG, "v7_fig2_shaping_fracao.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", path)
    return df


if __name__ == "__main__":
    fig_checkpoints()
    df = fig_fracao()
    print(df.round(3).to_string(index=False))
