"""
Figuras do relatório do experimento sintético (docs/relatorio_diagnostico_sintetico.md):
dinâmica de treino (recompensa, entropia, variância explicada, negócios) e curva de
checkpoints (val/teste) contra a regra ótima, para uma execução (default: syn_coint_s0).

Só lê CSV/JSON já gravados pelo pipeline. Não precisa dos dados de tick.

Uso (a partir da raiz do repo, no WSL/venv com matplotlib):
    python src/analyze_synthetic.py [backup] [run_tag]
    # defaults: sdumont_backup_sintetico/pairs-trading-rl  e  syn_coint_s0
"""

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BACKUP = sys.argv[1] if len(sys.argv) > 1 else os.path.join("sdumont_backup_sintetico", "pairs-trading-rl")
RUN = sys.argv[2] if len(sys.argv) > 2 else "syn_coint_s0"
RDIR = os.path.join(BACKUP, "src", "runs", RUN, "fold1")
OUT_IMG = os.path.join("docs", "img")
os.makedirs(OUT_IMG, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
C_VAL, C_TEST = "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 1.6, "font.size": 9,
    "axes.axisbelow": True,
})


def save(fig, name):
    path = os.path.join(OUT_IMG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", path)


def load_progress():
    frames = []
    for p in sorted(glob.glob(os.path.join(RDIR, "logs", "chunk*", "progress.csv")),
                    key=lambda s: int(re.search(r"chunk(\d+)", s).group(1))):
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def fig_collapse():
    p = load_progress()
    p = p.dropna(subset=["rollout/ep_rew_mean"]).reset_index(drop=True)
    x = p["time/total_timesteps"] / 1e6
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
    axs[0].plot(x, p["rollout/ep_rew_mean"], color=SLOT[0])
    axs[0].set_title("Recompensa/episódio (política estocástica, R$)")
    axs[0].set_yscale("symlog", linthresh=100)
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[1].plot(x, -p["train/entropy_loss"], color=SLOT[1])
    axs[1].set_title("Entropia (nats; máx. ln 3 ≈ 1,10)")
    axs[1].set_yscale("symlog", linthresh=0.01)
    axs[2].plot(x, p["train/explained_variance"], color=SLOT[2])
    axs[2].set_title("Variância explicada do crítico")
    axs[2].axhline(1, color=INK2, lw=0.6, ls=":")
    t = p.dropna(subset=["train/trades_per_episode"])
    axs[3].plot(t["time/total_timesteps"] / 1e6, t["train/trades_per_episode"], "o-", ms=3, color=SLOT[3])
    axs[3].set_title("Negócios por episódio (treino)")
    axs[3].set_yscale("log")
    for ax in axs:
        ax.set_xlabel("timesteps (milhões)")
        ax.axvline(3.4, color=INK, lw=0.8, ls="--", alpha=0.5)
    axs[3].annotate("colapso\n(atualização 35/611,\n3,4M timesteps)", xy=(3.4, 1.0), xytext=(14, 40),
                    fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    fig.suptitle(f"Colapso de entropia em {RUN} (estado raw, par sintético cointegrado)", y=1.08, fontsize=11)
    fig.tight_layout()
    save(fig, "v6_fig1_colapso_entropia.png")


def fig_checkpoints(rule_day=62.6):
    ck = pd.read_csv(os.path.join(RDIR, "eval", "checkpoint_curve.csv"))
    x = ck["timesteps"] / 1e6
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, ck["val_pnl_total"] / 30, "o-", color=C_VAL, label="validação (30 dias)")
    ax.plot(x, ck["test_pnl_total"] / 30, "s-", color=C_TEST, label="teste (30 dias)")
    ax.axhline(rule_day, color=INK, lw=1.2, ls="--", label=f"regra ótima causal ({rule_day:+.1f} R$/dia)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-16, 78)
    ax.set_xlabel("timesteps (milhões)")
    ax.set_ylabel("P/L real médio por dia (R$)")
    ax.set_title(f"{RUN}: o agente nunca chega perto da regra ótima", pad=10)
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    save(fig, "v6_fig2_checkpoints_vs_regra.png")


if __name__ == "__main__":
    fig_collapse()
    fig_checkpoints()
