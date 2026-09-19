"""
Análise dos resultados da v4 (rodadas `v4` = 50M e `v4-long` = 100M timesteps):
gera as figuras (docs/img/v4_*.png) e as tabelas em markdown
(<backup>/analysis/tab_*.md) usadas por docs/relatorio_resultados_v4.md.

Só lê CSV/JSON já gravados pelo pipeline (state.json, val_curve.csv,
logs/chunk*/progress.csv, eval/*) + o CSV de baselines direcionais gerado por
hold_baselines.py. Não precisa dos dados de tick nem do stable-baselines3.

Uso (a partir da raiz do repo):
    python src/analyze_v4.py [pasta_do_backup]      # default: sdumont_backup_v4
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BACKUP = sys.argv[1] if len(sys.argv) > 1 else "sdumont_backup_v4"
RUNS_DIR = os.path.join(BACKUP, "src", "runs")
OUT_TAB = os.path.join(BACKUP, "analysis")
OUT_IMG = os.path.join("docs", "img")
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_IMG, exist_ok=True)

RUNS = ["v4", "v4-long"]
FOLDS = [1, 2, 3]
RUN_LABEL = {"v4": "v4 (50M)", "v4-long": "v4-long (100M)"}

# --- paleta (dataviz: ordem categórica fixa; run = cinza/violeta, conjunto = azul/laranja/aqua)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
C_TRAIN, C_VAL, C_TEST = "#2a78d6", "#eb6834", "#1baf7a"
C_RUN = {"v4": "#8a8985", "v4-long": "#4a3aa7"}
C_LONG, C_SHORT = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 1.6, "font.size": 9,
})


def save(fig, name):
    path = os.path.join(OUT_IMG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", path)


def br(text):
    """Formato numérico pt-BR (milhar '.', decimal ','), trocando os separadores do formato en-US."""
    return str(text).translate(str.maketrans(",.", ".,"))


def write_tab(name, df, floatfmt=None):
    lines = []
    cols = list(df.columns)
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for i in range(len(df)):
        cells = []
        for c in cols:
            v = df[c].iloc[i]
            if isinstance(v, (float, np.floating)):
                f = (floatfmt or {}).get(c, "{:,.1f}")
                cells.append("—" if pd.isna(v) else br(f.format(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(OUT_TAB, f"tab_{name}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    df.to_csv(os.path.join(OUT_TAB, f"tab_{name}.csv"), index=False)


# --------------------------------------------------------------------------
# carga
# --------------------------------------------------------------------------

def rdir(run, fold):
    return os.path.join(RUNS_DIR, run, f"fold{fold}")


def load_state(run, fold):
    with open(os.path.join(rdir(run, fold), "state.json")) as fh:
        return json.load(fh)


def load_meta(run, fold):
    with open(os.path.join(rdir(run, fold), "metadata.json")) as fh:
        return json.load(fh)


def load_progress(run, fold):
    frames = []
    for p in sorted(glob.glob(os.path.join(rdir(run, fold), "logs", "chunk*", "progress.csv")),
                    key=lambda x: int(os.path.basename(os.path.dirname(x))[5:])):
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def load_days(run, fold, label):
    p = os.path.join(rdir(run, fold), "eval", f"{label}_days.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load_trades(run, fold, label):
    p = os.path.join(rdir(run, fold), "eval", f"{label}_trades.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


SUMMARY = pd.concat([
    pd.read_csv(os.path.join(rdir(r, f), "eval", "summary.csv")).assign(run=r, fold=f)
    for r in RUNS for f in FOLDS], ignore_index=True)

HOLD_PATH = os.path.join(OUT_TAB, "hold_baselines.csv")
HOLD = pd.read_csv(HOLD_PATH) if os.path.exists(HOLD_PATH) else None
if HOLD is None:
    print("AVISO: hold_baselines.csv ausente -- rode src/hold_baselines.py primeiro")


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def window_orders(run, fold, which):
    m = load_meta(run, fold)
    return m["val_orders"] if which == "val" else m["test_orders"]


# --------------------------------------------------------------------------
# T1/T2: folds e execução
# --------------------------------------------------------------------------

rows = []
for f in FOLDS:
    m = load_meta("v4-long", f)
    dv = load_days("v4-long", f, "best_val_val_fee2.5")
    dt = load_days("v4-long", f, "best_val_test_fee2.5")
    rows.append({
        "Fold": f, "Treino": f"pregões 1–{m['train_orders']['last']} ({m['train_orders']['n']} dias)",
        "Validação": f"{m['val_orders'][0]}–{m['val_orders'][-1]} ({dv['date'].min()} a {dv['date'].max()})",
        "Teste": f"{m['test_orders'][0]}–{m['test_orders'][-1]} ({dt['date'].min()} a {dt['date'].max()})",
    })
write_tab("folds", pd.DataFrame(rows))

rows = []
for r in RUNS:
    for f in FOLDS:
        s = load_state(r, f)
        ch = pd.DataFrame(s["chunks"])
        rows.append({
            "Run": RUN_LABEL[r], "Fold": f, "Timesteps": s["num_timesteps"] / 1e6,
            "Chunks (jobs de treino)": len(ch),
            "Steps/s (médio)": ch["steps_per_sec"].mean(),
            "Tempo de treino (min)": ch["learn_seconds"].sum() / 60,
            "Tempo de validação periódica (min)": ch["eval_seconds"].sum() / 60,
            "Melhor validação (pts)": s["best_val_score"],
            "Passo do melhor ckpt (M)": s["best_val_steps"] / 1e6,
        })
write_tab("execucao", pd.DataFrame(rows), {
    "Timesteps": "{:,.1f}M", "Steps/s (médio)": "{:,.0f}", "Tempo de treino (min)": "{:,.0f}",
    "Tempo de validação periódica (min)": "{:,.0f}", "Melhor validação (pts)": "{:,.1f}",
    "Passo do melhor ckpt (M)": "{:,.0f}M"})

# --------------------------------------------------------------------------
# FIG 1: treino -- recompensa e negócios por episódio
# --------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex="col")
for j, f in enumerate(FOLDS):
    for r in RUNS:
        p = load_progress(r, f)
        x = p["time/total_timesteps"] / 1e6
        axes[0, j].plot(x, p["rollout/ep_rew_mean"], color=C_RUN[r], lw=1.3,
                        ls="--" if r == "v4" else "-", label=RUN_LABEL[r])
        axes[1, j].plot(x, p["train/trades_per_episode"], color=C_RUN[r], lw=1.3,
                        ls="--" if r == "v4" else "-", label=RUN_LABEL[r])
    axes[0, j].set_yscale("symlog", linthresh=100)
    axes[0, j].axhline(0, color=INK2, lw=0.7)
    axes[1, j].set_yscale("log")
    axes[0, j].set_title(f"Fold {f} ({load_meta('v4', f)['train_orders']['n']} pregões de treino)")
    axes[1, j].set_xlabel("timesteps de treino (milhões)")
axes[0, 0].set_ylabel("recompensa média por episódio (pts)\n(política estocástica, escala symlog)")
axes[1, 0].set_ylabel("negócios fechados por episódio\n(treino, escala log)")
axes[0, 0].legend(loc="lower right")
fig.suptitle("Treino: a política aprende primeiro a parar de operar", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
save(fig, "v4_fig1_treino_recompensa_negocios.png")

# --------------------------------------------------------------------------
# FIG 2: diagnósticos do PPO (run longa)
# --------------------------------------------------------------------------

metrics = [("train/entropy_loss", "entropia (−loss)", "linear"),
           ("train/explained_variance", "variância explicada do crítico", "linear"),
           ("train/value_loss", "value loss", "log"),
           ("train/approx_kl", "KL aproximado", "log")]
fig, axes = plt.subplots(4, 3, figsize=(13, 9), sharex="col")
for j, f in enumerate(FOLDS):
    p = load_progress("v4-long", f)
    x = p["time/total_timesteps"] / 1e6
    for i, (col, lab, sc) in enumerate(metrics):
        y = -p[col] if col == "train/entropy_loss" else p[col]
        axes[i, j].plot(x, y, color=C_RUN["v4-long"], lw=1.2)
        axes[i, j].set_yscale(sc)
        if j == 0:
            axes[i, j].set_ylabel(lab)
    axes[0, j].set_title(f"Fold {f} — v4-long")
    axes[3, j].set_xlabel("timesteps de treino (milhões)")
fig.suptitle("Diagnósticos do PPO (run v4-long)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
save(fig, "v4_fig2_diagnosticos_ppo.png")

rows = []
for r in RUNS:
    for f in FOLDS:
        p = load_progress(r, f)
        ts = p["time/total_timesteps"]
        last = p[ts >= ts.max() - 10e6]
        tr = p["train/trades_per_episode"].rolling(10, min_periods=3).mean()
        below = p.loc[(tr < 50) & (p["time/total_timesteps"] > 1e6), "time/total_timesteps"]
        pos = p.loc[(p["rollout/ep_rew_mean"].rolling(10, min_periods=3).mean() > 0), "time/total_timesteps"]
        ent = -p.loc[(ts - 10e6).abs().idxmin(), "train/entropy_loss"]
        ev_a = p[(ts >= 10e6) & (ts <= 25e6)]["train/explained_variance"].mean()
        ev_b = last["train/explained_variance"].mean()
        rows.append({
            "Run": RUN_LABEL[r], "Fold": f,
            "Recompensa/episódio no início (pts)": p["rollout/ep_rew_mean"].dropna().iloc[0],
            "Recompensa/episódio, últimos 10M (pts)": last["rollout/ep_rew_mean"].mean(),
            "Negócios/episódio no início": p["train/trades_per_episode"].dropna().iloc[0],
            "Negócios/episódio, últimos 10M": last["train/trades_per_episode"].mean(),
            "Passo em que negócios/ep. < 50 (M)": below.iloc[0] / 1e6 if len(below) else np.nan,
            "Passo em que recompensa > 0 (M)": pos.iloc[0] / 1e6 if len(pos) else np.nan,
            "Entropia @10M (máx. 1,10)": ent,
            "Var. explicada 10–25M": ev_a, "Var. explicada, últimos 10M": ev_b,
        })
write_tab("treino", pd.DataFrame(rows), {
    "Recompensa/episódio no início (pts)": "{:,.0f}", "Recompensa/episódio, últimos 10M (pts)": "{:,.0f}",
    "Negócios/episódio no início": "{:,.0f}", "Negócios/episódio, últimos 10M": "{:,.1f}",
    "Passo em que negócios/ep. < 50 (M)": "{:,.0f}M", "Passo em que recompensa > 0 (M)": "{:,.0f}M",
    "Entropia @10M (máx. 1,10)": "{:.4f}", "Var. explicada 10–25M": "{:.2f}",
    "Var. explicada, últimos 10M": "{:.2f}"})

# --------------------------------------------------------------------------
# FIG 3 / FIG 4: treino x validação x teste por checkpoint  (P/L médio por dia e negócios/dia)
# --------------------------------------------------------------------------

def curves(run, fold):
    v = pd.read_csv(os.path.join(rdir(run, fold), "val_curve.csv"))
    c = pd.read_csv(os.path.join(rdir(run, fold), "eval", "checkpoint_curve.csv"))
    return v, c


for fig_name, kind in (("v4_fig3_curvas_treino_val_teste.png", "pl"),
                       ("v4_fig4_negocios_por_dia_checkpoints.png", "tr")):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex="col", sharey="row" if kind == "tr" else False)
    for i, r in enumerate(RUNS):
        for j, f in enumerate(FOLDS):
            ax = axes[i, j]
            v, c = curves(r, f)
            n_val = len(load_meta(r, f)["val_orders"])
            n_test = len(load_meta(r, f)["test_orders"])
            if kind == "pl":
                ax.plot(v["timesteps"] / 1e6, v["train_subset_pnl_mean_day"], color=C_TRAIN, marker="o", ms=3.5,
                        label="treino (10 pregões fixos)")
                ax.plot(v["timesteps"] / 1e6, v["val_pnl_mean_day"], color=C_VAL, marker="o", ms=3.5,
                        label="validação (15 pregões)")
                ax.plot(c["timesteps"] / 1e6, c["test_pnl_total"] / n_test, color=C_TEST, marker="s", ms=3.5,
                        label="teste (15 pregões) — só relatório")
                ax.axhline(0, color=INK2, lw=0.7)
                if j == 0:
                    ax.set_ylabel(f"{RUN_LABEL[r]}\nP/L médio por dia (pts)")
            else:
                ax.plot(v["timesteps"] / 1e6, v["train_subset_trades_per_day"], color=C_TRAIN, marker="o", ms=3.5,
                        label="treino (10 pregões fixos)")
                ax.plot(v["timesteps"] / 1e6, v["val_trades_per_day"], color=C_VAL, marker="o", ms=3.5,
                        label="validação")
                ax.plot(c["timesteps"] / 1e6, c["test_trades_per_day"], color=C_TEST, marker="s", ms=3.5,
                        label="teste")
                if j == 0:
                    ax.set_ylabel(f"{RUN_LABEL[r]}\nnegócios por dia")
            sel = load_state(r, f)["best_val_steps"] / 1e6
            ax.axvline(sel, color=C_VAL, ls=":", lw=1.2)
            if i == 0:
                ax.set_title(f"Fold {f}")
            if i == 1:
                ax.set_xlabel("timesteps de treino (milhões)")
    h, l = axes[0, 0].get_legend_handles_labels()
    ttl = ("Treino × validação × teste ao longo do treino (linha pontilhada = checkpoint escolhido pela validação)"
           if kind == "pl" else "Negócios por dia ao longo do treino (linha pontilhada = checkpoint escolhido)")
    fig.suptitle(ttl, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, fig_name)

rows = []
for r in RUNS:
    for f in FOLDS:
        v, c = curves(r, f)
        n_test = len(load_meta(r, f)["test_orders"])
        rows.append({
            "Run": RUN_LABEL[r], "Fold": f, "Pontos de validação": len(v), "Pontos de teste": len(c),
            "Treino (10 pregões), P/L/dia": v["train_subset_pnl_mean_day"].mean(),
            "Validação, P/L/dia": v["val_pnl_mean_day"].mean(),
            "Teste, P/L/dia": (c["test_pnl_total"] / n_test).mean(),
            "Validação: pontos > 0": f"{(v['val_pnl_total'] > 0).sum()}/{len(v)}",
            "Teste: pontos > 0": f"{(c['test_pnl_total'] > 0).sum()}/{len(c)}",
        })
write_tab("curvas_medias", pd.DataFrame(rows), {
    "Treino (10 pregões), P/L/dia": "{:,.0f}", "Validação, P/L/dia": "{:,.0f}", "Teste, P/L/dia": "{:,.0f}"})

# --------------------------------------------------------------------------
# T3: resultados principais (val e teste) com IC bootstrap
# --------------------------------------------------------------------------

rows = []
for r in RUNS:
    for f in FOLDS:
        for model in ("best_val", "last"):
            for st in ("val", "test"):
                d = load_days(r, f, f"{model}_{st}_fee2.5")
                lo, hi = boot_ci(d["pnl"])
                rows.append({
                    "Run": RUN_LABEL[r], "Fold": f, "Modelo": "melhor-de-validação" if model == "best_val" else "último",
                    "Conjunto": "validação" if st == "val" else "teste",
                    "P/L total (pts)": d["pnl"].sum(), "R$": d["pnl"].sum() * 0.20,
                    "P/L médio/dia": d["pnl"].mean(), "IC95% médio/dia": br(f"[{lo:,.0f}; {hi:,.0f}]"),
                    "Dias +": f"{(d['pnl'] > 0).sum()}/{len(d)}",
                    "Negócios/dia": d["n_trades"].sum() / len(d),
                    "Acerto": d["n_wins"].sum() / max(d["n_trades"].sum(), 1) * 100,
                })
MAIN = pd.DataFrame(rows)
write_tab("principal", MAIN, {"P/L total (pts)": "{:,.0f}", "R$": "{:,.0f}", "P/L médio/dia": "{:,.0f}",
                              "Negócios/dia": "{:,.1f}", "Acerto": "{:.0f}%"})

# --- agregado nos 3 folds (45 dias de teste / 45 de validação)
rows = []
for r in RUNS:
    for model in ("best_val", "last"):
        for st in ("val", "test"):
            d = pd.concat([load_days(r, f, f"{model}_{st}_fee2.5") for f in FOLDS])
            lo, hi = boot_ci(d["pnl"])
            t = stats.ttest_1samp(d["pnl"], 0.0)
            rows.append({"Run": RUN_LABEL[r], "Modelo": "melhor-de-validação" if model == "best_val" else "último",
                         "Conjunto": "validação" if st == "val" else "teste", "Dias": len(d),
                         "P/L total (pts)": d["pnl"].sum(), "P/L médio/dia": d["pnl"].mean(),
                         "IC95% médio/dia": br(f"[{lo:,.0f}; {hi:,.0f}]"), "p-valor (t, média=0)": t.pvalue,
                         "Negócios/dia": d["n_trades"].sum() / len(d)})
write_tab("agregado", pd.DataFrame(rows), {"P/L total (pts)": "{:,.0f}", "P/L médio/dia": "{:,.0f}",
                                           "p-valor (t, média=0)": "{:.2f}", "Negócios/dia": "{:,.1f}"})

# --------------------------------------------------------------------------
# T4: baselines (por fold, conjunto)
# --------------------------------------------------------------------------

rows = []
for f in FOLDS:
    for st in ("val", "test"):
        orders = window_orders("v4-long", f, st)
        row = {"Fold": f, "Conjunto": "validação" if st == "val" else "teste"}
        if HOLD is not None:
            h = HOLD[HOLD["order"].isin(orders)]
            row["Comprar e segurar"] = h["long_hold_pnl"].sum()
            row["Vender e segurar"] = h["short_hold_pnl"].sum()
            row["Variação do WIN no período (pts)"] = h["win_move"].sum()
        for name, key in (("Regra de limiar (z=1)", "baseline_threshold1.0"), ("Aleatório", "baseline_random")):
            row[name] = SUMMARY[(SUMMARY.label == f"{key}_{st}_fee2.5") & (SUMMARY.fold == f) &
                                (SUMMARY.run == "v4")]["pnl_total"].iloc[0]
        for r in RUNS:
            row[f"Modelo {RUN_LABEL[r]}"] = SUMMARY[(SUMMARY.label == f"best_val_{st}_fee2.5") & (SUMMARY.fold == f) &
                                                    (SUMMARY.run == r)]["pnl_total"].iloc[0]
        rows.append(row)
write_tab("baselines", pd.DataFrame(rows), {c: "{:,.0f}" for c in [
    "Comprar e segurar", "Vender e segurar", "Variação do WIN no período (pts)",
    "Regra de limiar (z=1)", "Aleatório", "Modelo v4 (50M)", "Modelo v4-long (100M)"]})

# --------------------------------------------------------------------------
# FIG 5: contexto de mercado + janelas (buy&hold vs modelo)
# --------------------------------------------------------------------------

if HOLD is not None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    H = HOLD.sort_values("order")
    ax.plot(H["order"], H["win_move"].cumsum(), color=INK2, lw=1.4)
    for f in FOLDS:
        vo, to = window_orders("v4", f, "val"), window_orders("v4", f, "test")
        ax.axvspan(vo[0] - 0.5, vo[-1] + 0.5, color=C_VAL, alpha=0.25, lw=0,
                   label="validação" if f == 1 else None)
        ax.axvspan(to[0] - 0.5, to[-1] + 0.5, color=C_TEST, alpha=0.25, lw=0,
                   label="teste" if f == 1 else None)
    for f, n in zip(FOLDS, (100, 150, 200)):
        ax.axvline(n + 0.5, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("pregão (ordem cronológica)")
    ax.set_ylabel("variação acumulada do mid do WIN (pts)")
    ax.set_title("Contexto de mercado: soma das variações diárias do WIN")
    ax.legend(loc="lower left")

    ax = axes[1]
    labels, vals = [], {"long": [], "short": [], "v4": [], "v4-long": []}
    for f in FOLDS:
        for st in ("val", "test"):
            orders = window_orders("v4", f, st)
            h = HOLD[HOLD["order"].isin(orders)]
            labels.append(f"F{f} {'val' if st == 'val' else 'teste'}")
            vals["long"].append(h["long_hold_pnl"].sum())
            vals["short"].append(h["short_hold_pnl"].sum())
            for r in RUNS:
                vals[r].append(SUMMARY[(SUMMARY.label == f"best_val_{st}_fee2.5") & (SUMMARY.fold == f) &
                                       (SUMMARY.run == r)]["pnl_total"].iloc[0])
    x = np.arange(len(labels))
    w = 0.2
    ax.bar(x - 1.5 * w, vals["long"], w, color=C_LONG, label="comprar e segurar")
    ax.bar(x - 0.5 * w, vals["short"], w, color=C_SHORT, label="vender e segurar")
    ax.bar(x + 0.5 * w, vals["v4"], w, color=C_RUN["v4"], label="modelo v4 (melhor-de-val)")
    ax.bar(x + 1.5 * w, vals["v4-long"], w, color=C_RUN["v4-long"], label="modelo v4-long (melhor-de-val)")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("P/L total da janela (pts)")
    ax.set_title("Modelo × posição direcional fixa no dia (já com custos)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5)
    fig.tight_layout()
    save(fig, "v4_fig5_contexto_mercado_baselines.png")

# --------------------------------------------------------------------------
# FIG 6: curvas de equity (P/L acumulado por pregão) val | teste
# --------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
for i, r in enumerate(RUNS):
    for j, f in enumerate(FOLDS):
        ax = axes[i, j]
        dv = load_days(r, f, "best_val_val_fee2.5")
        dt = load_days(r, f, "best_val_test_fee2.5")
        dlv = load_days(r, f, "last_val_fee2.5")
        dlt = load_days(r, f, "last_test_fee2.5")
        n_v, n_t = len(dv), len(dt)
        xs = np.arange(1, n_v + n_t + 1)
        ax.plot(xs, np.cumsum(np.r_[dv["pnl"], dt["pnl"]]), color=INK, lw=2.0, label="melhor-de-validação")
        ax.plot(xs, np.cumsum(np.r_[dlv["pnl"], dlt["pnl"]]), color=INK, lw=1.3, ls="--", label="último")
        if HOLD is not None:
            orders = list(dv["order"]) + list(dt["order"])
            h = HOLD.set_index("order").loc[orders]
            ax.plot(xs, np.cumsum(h["long_hold_pnl"].to_numpy()), color=C_LONG, lw=1.3, label="comprar e segurar")
            ax.plot(xs, np.cumsum(h["short_hold_pnl"].to_numpy()), color=C_SHORT, lw=1.3, label="vender e segurar")
        ax.axvspan(0.5, n_v + 0.5, color=C_VAL, alpha=0.10, lw=0)
        ax.axvspan(n_v + 0.5, n_v + n_t + 0.5, color=C_TEST, alpha=0.10, lw=0)
        ax.axhline(0, color=INK2, lw=0.7)
        if i == 0:
            ax.set_title(f"Fold {f}")
        if j == 0:
            ax.set_ylabel(f"{RUN_LABEL[r]}\nP/L acumulado (pts)")
        if i == 1:
            ax.set_xlabel("pregão (fundo laranja = validação, verde = teste)")
axes[0, 0].legend(loc="upper left", fontsize=7.5)
fig.suptitle("P/L acumulado por pregão: modelo × posição direcional fixa", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
save(fig, "v4_fig6_equity.png")

# --------------------------------------------------------------------------
# T5 / FIG 7 / FIG 8: é arbitragem?  decomposição, exposição, dispersão P/L x mercado
# --------------------------------------------------------------------------

rows = []
for r in RUNS:
    for f in FOLDS:
        for st in ("val", "test"):
            s = SUMMARY[(SUMMARY.label == f"best_val_{st}_fee2.5") & (SUMMARY.fold == f) & (SUMMARY.run == r)].iloc[0]
            d = load_days(r, f, f"best_val_{st}_fee2.5")
            one = (d["n_trades"] <= 1).mean() * 100
            dur = (d["mean_duration"] / d["n_ticks"]).mean() * 100
            rows.append({
                "Run": RUN_LABEL[r], "Fold": f, "Conjunto": "validação" if st == "val" else "teste",
                "P/L bruto": s["gross_mtm_total"], "Convergência do spread": s["pnl_spread_component"],
                "Direcional (preço justo)": s["pnl_fair_component"], "Custos": s["cost_total"],
                "Exposição líquida média": s["mean_net_exposure"] * 100,
                "Corr. P/L × mov. WIN": s["corr_gross_vs_win_move"],
                "Dias com ≤1 negócio": one, "Duração média (% do pregão)": dur,
            })
DEC = pd.DataFrame(rows)
write_tab("decomposicao", DEC, {
    "P/L bruto": "{:,.0f}", "Convergência do spread": "{:,.0f}", "Direcional (preço justo)": "{:,.0f}",
    "Custos": "{:,.0f}", "Exposição líquida média": "{:+.0f}%", "Corr. P/L × mov. WIN": "{:+.2f}",
    "Dias com ≤1 negócio": "{:.0f}%", "Duração média (% do pregão)": "{:.0f}%"})

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, st in zip(axes, ("val", "test")):
    sub = DEC[DEC["Conjunto"] == ("validação" if st == "val" else "teste")].reset_index(drop=True)
    x = np.arange(len(sub))
    w = 0.26
    ax.bar(x - w, sub["Convergência do spread"], w, color=C_TRAIN, label="convergência do spread")
    ax.bar(x, sub["Direcional (preço justo)"], w, color=C_VAL, label="direcional (preço justo)")
    ax.bar(x + w, -sub["Custos"], w, color="#8a8985", label="custos (negativo)")
    ax.plot(x, sub["P/L bruto"] - sub["Custos"], ls="none", marker="D", ms=6, color=INK, label="P/L líquido")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{'v4' if 'long' not in a else 'v4-long'}\nF{b}" for a, b in zip(sub["Run"], sub["Fold"])])
    ax.set_title("Validação" if st == "val" else "Teste")
axes[0].set_ylabel("pts (soma dos 15 pregões)")
axes[0].legend(loc="upper right", fontsize=7.5)
fig.suptitle("De onde vem o P/L bruto: convergência do spread × movimento do preço justo", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
save(fig, "v4_fig7_decomposicao_pl.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharex=True, sharey=True)
for ax, r in zip(axes, RUNS):
    for st, col, mk in (("val", C_VAL, "o"), ("test", C_TEST, "s")):
        xs, ys = [], []
        for f in FOLDS:
            d = load_days(r, f, f"best_val_{st}_fee2.5")
            xs += list(d["win_move"])
            ys += list(d["pnl"])
        ax.scatter(xs, ys, s=22, color=col, marker=mk, alpha=0.85, label="validação" if st == "val" else "teste")
    lim = np.array(ax.get_xlim())
    ax.plot(lim, -lim, color=C_SHORT, lw=1.2, ls="--", label="vender e segurar (y = −x)")
    ax.plot(lim, lim, color=C_LONG, lw=1.2, ls="--", label="comprar e segurar (y = x)")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_title(RUN_LABEL[r])
    ax.set_xlabel("variação do mid do WIN no pregão (pts)")
axes[0].set_ylabel("P/L do modelo no pregão (pts)")
axes[0].legend(loc="upper right", fontsize=7.5)
fig.suptitle("P/L diário do modelo × movimento do WIN no dia (45 pregões de validação + 45 de teste)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
save(fig, "v4_fig8_pl_vs_movimento.png")

# --- acerto de direção nos pregões com posição praticamente única
rows = []
for r in RUNS:
    for st in ("val", "test"):
        hit = n = 0
        for f in FOLDS:
            d = load_days(r, f, f"best_val_{st}_fee2.5")
            dd = d[d["net_exposure"].abs() >= 0.9]
            hit += int((np.sign(dd["net_exposure"]) == np.sign(dd["win_move"])).sum())
            n += len(dd)
        p = stats.binomtest(hit, n, 0.5).pvalue if n else np.nan
        rows.append({"Run": RUN_LABEL[r], "Conjunto": "validação" if st == "val" else "teste",
                     "Dias com posição ≥90% do pregão numa direção": n, "Direção correta": hit,
                     "Taxa": hit / n * 100 if n else np.nan, "p-valor (binomial, 50%)": p})
write_tab("direcao", pd.DataFrame(rows), {"Taxa": "{:.0f}%", "p-valor (binomial, 50%)": "{:.2f}"})

# --------------------------------------------------------------------------
# FIG 9: event study
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for ax, r in zip(axes, RUNS):
    for f, col in zip(FOLDS, (C_TRAIN, C_VAL, C_TEST)):
        parts = []
        for st in ("val", "test"):
            t = load_trades(r, f, f"best_val_{st}_fee2.5")
            parts.append(t)
        t = pd.concat(parts)
        offs = [50, 100, 200, 400, 800]
        y = [0.0] + [t[f"ev_{o}"].mean() for o in offs]
        ax.plot([0] + offs, y, color=col, marker="o", ms=4, label=f"fold {f} (n={len(t)} negócios)")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_title(RUN_LABEL[r])
    ax.set_xlabel("ticks após a entrada")
axes[0].set_ylabel("Δ mispricing a favor da posição (pts)")
axes[0].legend(loc="upper left")
fig.suptitle("Event study: o desalinhamento WIN × BOVA converge depois da entrada? (validação + teste)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
save(fig, "v4_fig9_event_study.png")

# --------------------------------------------------------------------------
# FIG 10: perfil dos negócios (duração, hora de entrada)
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for r in RUNS:
    fr, hrs = [], []
    for f in FOLDS:
        for st in ("val", "test"):
            t = load_trades(r, f, f"best_val_{st}_fee2.5")
            d = load_days(r, f, f"best_val_{st}_fee2.5").set_index("order")
            fr += list(t["duration"] / t["order"].map(d["n_ticks"]) * 100)
            hrs += list(t["hour_entry"])
    axes[0].hist(fr, bins=np.linspace(0, 100, 21), histtype="step", lw=1.8, color=C_RUN[r], label=RUN_LABEL[r])
    cnt = pd.Series(hrs).value_counts().sort_index()
    axes[1].plot(cnt.index, cnt.values, marker="o", ms=4, color=C_RUN[r], label=RUN_LABEL[r])
axes[0].set_xlabel("duração do negócio (% do pregão)")
axes[0].set_ylabel("nº de negócios")
axes[0].set_title("Duração dos negócios")
axes[0].legend()
axes[1].set_xlabel("hora de entrada")
axes[1].set_title("Hora de entrada dos negócios")
# negócios por dia (todos os pregões de val+teste)
for r in RUNS:
    nt = pd.concat([load_days(r, f, f"best_val_{st}_fee2.5") for f in FOLDS for st in ("val", "test")])["n_trades"]
    cnt = nt.value_counts().sort_index()
    axes[2].plot(cnt.index, cnt.values, marker="o", ms=4, color=C_RUN[r], label=RUN_LABEL[r])
axes[2].set_xlabel("negócios no pregão")
axes[2].set_ylabel("nº de pregões")
axes[2].set_title("Negócios por pregão (90 pregões)")
fig.suptitle("Perfil dos negócios do melhor-de-validação (val + teste, 3 folds)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
save(fig, "v4_fig10_perfil_negocios.png")

# --------------------------------------------------------------------------
# T6: robustez -- custo, latência
# --------------------------------------------------------------------------

rows = []
for r in RUNS:
    for f in FOLDS:
        for st in ("val", "test"):
            row = {"Run": RUN_LABEL[r], "Fold": f, "Conjunto": "validação" if st == "val" else "teste"}
            for lab, key in (("fee 0", f"cost_sens_{st}_fee0.0"), ("fee 0,5", f"cost_sens_{st}_fee0.5"),
                             ("fee 2,5 (treino)", f"best_val_{st}_fee2.5")):
                row[lab] = SUMMARY[(SUMMARY.label == key) & (SUMMARY.fold == f) & (SUMMARY.run == r)]["pnl_total"].iloc[0]
            lat = SUMMARY[(SUMMARY.label == f"latency1_{st}_fee2.5") & (SUMMARY.fold == f) & (SUMMARY.run == r)].iloc[0]
            row["latência 1 tick: P/L"] = lat["pnl_total"]
            row["latência 1 tick: negócios/dia"] = lat["trades_per_day"]
            rows.append(row)
write_tab("robustez", pd.DataFrame(rows), {
    "fee 0": "{:,.0f}", "fee 0,5": "{:,.0f}", "fee 2,5 (treino)": "{:,.0f}",
    "latência 1 tick: P/L": "{:,.0f}", "latência 1 tick: negócios/dia": "{:,.0f}"})

# --------------------------------------------------------------------------
# T7: a validação é informativa sobre o teste?  (correlação entre checkpoints)
# --------------------------------------------------------------------------

allc = []
for r in RUNS:
    for f in FOLDS:
        c = pd.read_csv(os.path.join(rdir(r, f), "eval", "checkpoint_curve.csv"))
        c["run"], c["fold"] = r, f
        allc.append(c)
allc = pd.concat(allc)
uniq = allc.drop_duplicates(subset=["fold", "timesteps", "val_pnl_total", "test_pnl_total"])
rows = []
for f in FOLDS:
    u = uniq[uniq.fold == f]
    rho = stats.spearmanr(u["val_pnl_total"], u["test_pnl_total"])
    rows.append({"Escopo": f"fold {f}", "Checkpoints": len(u), "ρ de Spearman (val × teste)": rho.statistic,
                 "p-valor": rho.pvalue})
rho = stats.spearmanr(uniq["val_pnl_total"], uniq["test_pnl_total"])
rows.append({"Escopo": "todos", "Checkpoints": len(uniq), "ρ de Spearman (val × teste)": rho.statistic,
             "p-valor": rho.pvalue})
write_tab("val_vs_teste", pd.DataFrame(rows), {"ρ de Spearman (val × teste)": "{:+.2f}", "p-valor": "{:.2f}"})

rows = []
for r in RUNS:
    for f in FOLDS:
        c = allc[(allc.run == r) & (allc.fold == f)]
        sel = MAIN[(MAIN["Run"] == RUN_LABEL[r]) & (MAIN["Fold"] == f) & (MAIN["Modelo"] == "melhor-de-validação")]
        rows.append({
            "Run": RUN_LABEL[r], "Fold": f,
            "Val do escolhido": sel[sel["Conjunto"] == "validação"]["P/L total (pts)"].iloc[0],
            "Val médio dos checkpoints": c["val_pnl_total"].mean(),
            "Teste do escolhido": sel[sel["Conjunto"] == "teste"]["P/L total (pts)"].iloc[0],
            "Teste médio dos checkpoints": c["test_pnl_total"].mean(),
            "Checkpoints com teste > 0": f"{(c['test_pnl_total'] > 0).sum()}/{len(c)}",
        })
write_tab("selecao", pd.DataFrame(rows), {k: "{:,.0f}" for k in [
    "Val do escolhido", "Val médio dos checkpoints", "Teste do escolhido", "Teste médio dos checkpoints"]})

# --------------------------------------------------------------------------
# FIG 11: validação × teste por checkpoint (dispersão)
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.2, 5.2))
for f, col in zip(FOLDS, (C_TRAIN, C_VAL, C_TEST)):
    u = uniq[uniq.fold == f]
    ax.scatter(u["val_pnl_total"], u["test_pnl_total"], s=30, color=col, label=f"fold {f}")
ax.axhline(0, color=INK2, lw=0.7)
ax.axvline(0, color=INK2, lw=0.7)
ax.set_xlabel("P/L de validação do checkpoint (pts, 15 pregões)")
ax.set_ylabel("P/L de teste do mesmo checkpoint (pts, 15 pregões)")
ax.set_title(f"Validação prevê o teste? ρ de Spearman = {rho.statistic:+.2f} (p = {rho.pvalue:.2f})")
ax.legend()
fig.tight_layout()
save(fig, "v4_fig11_val_vs_teste.png")

# --------------------------------------------------------------------------
# T8: entrada por faixa de sinal (agregado nos folds, melhor-de-validação, val+teste)
# --------------------------------------------------------------------------

rows = []
for r in RUNS:
    parts = []
    for f in FOLDS:
        for st in ("val", "test"):
            p = os.path.join(rdir(r, f), "eval", f"best_val_{st}_fee2.5_entry_buckets.csv")
            b = pd.read_csv(p)
            b["fold"] = f
            parts.append(b)
    b = pd.concat(parts)
    for name, g in b.groupby("bucket", sort=False):
        n = g["n"].sum()
        rows.append({"Run": RUN_LABEL[r], "Faixa do sinal na entrada (z)": name, "Negócios": n,
                     "P/L médio (pts)": (g["pnl_mean"] * g["n"]).sum() / n if n else np.nan,
                     "Convergência média (pts)": (g["d_misp_mean"] * g["n"]).sum() / n if n else np.nan})
write_tab("faixas_sinal", pd.DataFrame(rows), {"Negócios": "{:,.0f}", "P/L médio (pts)": "{:,.0f}",
                                               "Convergência média (pts)": "{:,.0f}"})

# --------------------------------------------------------------------------
# T9: quanto o modelo negocia vs regra de limiar/aleatório (negócios/dia)
# --------------------------------------------------------------------------

rows = []
for f in FOLDS:
    row = {"Fold": f}
    for name, key in (("Aleatório", "baseline_random"), ("Regra de limiar", "baseline_threshold1.0")):
        row[f"{name}"] = SUMMARY[(SUMMARY.label == f"{key}_test_fee2.5") & (SUMMARY.fold == f) &
                                 (SUMMARY.run == "v4")]["trades_per_day"].iloc[0]
    for r in RUNS:
        row[f"Modelo {RUN_LABEL[r]}"] = SUMMARY[(SUMMARY.label == f"best_val_test_fee2.5") & (SUMMARY.fold == f) &
                                                (SUMMARY.run == r)]["trades_per_day"].iloc[0]
    rows.append(row)
write_tab("negocios_dia", pd.DataFrame(rows), {c: "{:,.1f}" for c in [
    "Aleatório", "Regra de limiar", "Modelo v4 (50M)", "Modelo v4-long (100M)"]})

# --------------------------------------------------------------------------
# T10: perfil dos negócios (melhor-de-validação, val + teste, 3 folds)
# --------------------------------------------------------------------------

rows = []
for r in RUNS:
    tr_all, day_all = [], []
    for f in FOLDS:
        for st in ("val", "test"):
            t = load_trades(r, f, f"best_val_{st}_fee2.5")
            d = load_days(r, f, f"best_val_{st}_fee2.5")
            t = t.assign(frac=t["duration"] / t["order"].map(d.set_index("order")["n_ticks"]))
            tr_all.append(t)
            day_all.append(d)
    t = pd.concat(tr_all)
    d = pd.concat(day_all)
    rows.append({
        "Run": RUN_LABEL[r], "Negócios": len(t), "Pregões": len(d),
        "Pregões com exatamente 1 negócio": f"{(d['n_trades'] == 1).sum()} ({(d['n_trades'] == 1).mean() * 100:.0f}%)",
        "Pregões com ≥ 10 negócios": f"{(d['n_trades'] >= 10).sum()} ({(d['n_trades'] >= 10).mean() * 100:.0f}%)",
        "Entradas às 10h–11h": f"{(t['hour_entry'] <= 11).mean() * 100:.0f}%",
        "Negócios < 5% do pregão": f"{(t['frac'] < 0.05).mean() * 100:.0f}%",
        "Negócios ≥ 95% do pregão": f"{(t['frac'] >= 0.95).mean() * 100:.0f}%",
        "Duração mediana (% do pregão)": t["frac"].median() * 100,
        "P/L médio por negócio (pts)": t["pnl"].mean(),
        "Ganho médio": t.loc[t["pnl"] > 0, "pnl"].mean(), "Perda média": t.loc[t["pnl"] <= 0, "pnl"].mean(),
    })
write_tab("perfil", pd.DataFrame(rows), {"Duração mediana (% do pregão)": "{:.1f}%", "P/L médio por negócio (pts)": "{:,.0f}",
                                         "Ganho médio": "{:,.0f}", "Perda média": "{:,.0f}"})

# --------------------------------------------------------------------------
# T11: regra de limiar -- quanto da convergência acontece no preço do WIN
# --------------------------------------------------------------------------

rows = []
for f in FOLDS:
    for st in ("val", "test"):
        x = SUMMARY[(SUMMARY.label == f"baseline_threshold1.0_{st}_fee2.5") & (SUMMARY.fold == f) &
                    (SUMMARY.run == "v4")].iloc[0]
        rows.append({"Fold": f, "Conjunto": "validação" if st == "val" else "teste",
                     "Negócios": x["n_trades"], "Convergência total do spread": x["pnl_spread_component"],
                     "Efeito do preço justo na posição": x["pnl_fair_component"],
                     "P/L bruto (movimento do WIN)": x["gross_mtm_total"], "Custos": x["cost_total"],
                     "% da convergência capturada no WIN": x["gross_mtm_total"] / x["pnl_spread_component"] * 100,
                     "Convergência por negócio (pts)": x["pnl_spread_component"] / x["n_trades"],
                     "Custo por negócio (pts)": x["cost_total"] / x["n_trades"]})
write_tab("limiar", pd.DataFrame(rows), {
    "Negócios": "{:,.0f}", "Convergência total do spread": "{:,.0f}", "Efeito do preço justo na posição": "{:,.0f}",
    "P/L bruto (movimento do WIN)": "{:,.0f}", "Custos": "{:,.0f}", "% da convergência capturada no WIN": "{:.0f}%",
    "Convergência por negócio (pts)": "{:.1f}", "Custo por negócio (pts)": "{:.1f}"})

print("\nOK -- tabelas em", OUT_TAB)
