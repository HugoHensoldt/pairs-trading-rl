"""
Análise da campanha v5 (recompensa hedgeada e de convergência do spread):
6 execuções (hedged_s0..s3 a 100M timesteps; spread_s0 e spreadx_s0, pilotos de
30M) -- gera as figuras (docs/img/v5_*.png) e as tabelas em markdown
(<backup>/analysis/tab_*.md) usadas por docs/relatorio_resultados_v5.md.

Só lê CSV/JSON gravados pelo pipeline (state.json, val_curve.csv,
logs/chunk*/progress.csv, eval/*) e, se existir, a pasta de contexto gerada por
opportunity_analysis.py / hold_baselines_hedged.py nos pregões de val/teste.

Uso (a partir da raiz do repo, no WSL/venv com matplotlib e scipy):
    python src/analyze_campanha.py [backup] [pasta_contexto]
    # defaults: sdumont_backup_campanha  e  <backup>/analysis/contexto
"""

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BACKUP = sys.argv[1] if len(sys.argv) > 1 else "sdumont_backup_campanha"
CTX = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BACKUP, "analysis", "contexto")
RUNS_DIR = os.path.join(BACKUP, "src", "runs")
OUT_TAB = os.path.join(BACKUP, "analysis")
OUT_IMG = os.path.join("docs", "img")
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_IMG, exist_ok=True)

HEDGED = ["hedged_s0", "hedged_s1", "hedged_s2", "hedged_s3"]
SPREAD = ["spread_s0", "spreadx_s0"]
ALL_RUNS = HEDGED + SPREAD
SEEDNAME = {r: f"seed {r[-1]}" for r in HEDGED}
STEPS_HEDGE = 100e6

# --- paleta de referência da skill dataviz: ordem categórica fixa (slots 1-4 validados p/ linhas e barras)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # azul, laranja, aqua, amarelo
C_SEED = dict(zip(HEDGED, SLOT))
MARK = dict(zip(HEDGED, ["o", "s", "^", "D"]))
C_VAL, C_TEST, C_TRAIN = "#eb6834", "#1baf7a", "#8a8985"    # conjuntos (val laranja, teste aqua, treino cinza)
C_VAR = {"hedged_s0": "#2a78d6", "spread_s0": "#eb6834", "spreadx_s0": "#1baf7a"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 1.6, "font.size": 9,
    "axes.axisbelow": True,
})

from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


def symlog_axis(ax, thresh, ticks):
    """Eixo y symlog com ticks explícitos e rótulos legíveis (sem minor ticks)."""
    ax.set_yscale("symlog", linthresh=thresh)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(f"{v:,.0f}")))


def save(fig, name):
    path = os.path.join(OUT_IMG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", path)


def br(text):
    """Formato numérico pt-BR (milhar '.', decimal ',')."""
    return str(text).translate(str.maketrans(",.", ".,"))


def write_tab(name, df, floatfmt=None):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
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
    print("tab ->", name)


# --------------------------------------------------------------------------- carga
def rdir(run):
    return os.path.join(RUNS_DIR, run, "fold1")


def load_state(run):
    with open(os.path.join(rdir(run), "state.json")) as fh:
        return json.load(fh)


def load_meta(run):
    with open(os.path.join(rdir(run), "metadata.json")) as fh:
        return json.load(fh)


def load_progress(run):
    frames = []
    for p in sorted(glob.glob(os.path.join(rdir(run), "logs", "chunk*", "progress.csv")),
                    key=lambda s: int(re.search(r"chunk(\d+)", s).group(1))):
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_summary(run):
    return pd.read_csv(os.path.join(rdir(run), "eval", "summary.csv"))


def load_days(run, label):
    p = os.path.join(rdir(run), "eval", f"{label}_days.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load_trades(run, label):
    p = os.path.join(rdir(run), "eval", f"{label}_trades.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load_valcurve(run):
    return pd.read_csv(os.path.join(rdir(run), "val_curve.csv"))


def load_ckcurve(run):
    return pd.read_csv(os.path.join(rdir(run), "eval", "checkpoint_curve.csv"))


def row(summary, label):
    r = summary[summary["label"] == label]
    return r.iloc[0] if len(r) else None


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def smooth(y, w=15):
    return pd.Series(y).rolling(w, min_periods=1, center=True).median().to_numpy()


SUM = {r: load_summary(r) for r in ALL_RUNS}
STATE = {r: load_state(r) for r in ALL_RUNS}


def real(run, label):
    r = row(SUM[run], label)
    return np.nan if r is None else float(r["real_pnl_hedged_brl"])


# --------------------------------------------------------------------------- tabelas
def tab_execucao():
    rows = []
    for r in ALL_RUNS:
        st, meta = STATE[r], load_meta(r)
        ch = st["chunks"]
        learn = sum(c["learn_seconds"] for c in ch)
        evals = sum(c["eval_seconds"] for c in ch)
        rows.append({"Execução": r, "Env": meta.get("env_kind", "?"),
                     "Timesteps (M)": st["num_timesteps"] / 1e6, "Chunks": len(ch),
                     "Steps/s (mediana)": float(np.median([c["steps_per_sec"] for c in ch])),
                     "Treino puro (min)": (learn - evals) / 60, "Validações periódicas (min)": evals / 60,
                     "Startup (s, mediana)": float(np.median([c["startup_seconds"] for c in ch])),
                     "Melhor ckpt de validação (M)": st["best_val_steps"] / 1e6,
                     "Métrica de seleção": st.get("best_val_metric", meta.get("select_metric", "?"))})
    df = pd.DataFrame(rows)
    write_tab("execucao", df, {"Timesteps (M)": "{:,.1f}", "Steps/s (mediana)": "{:,.0f}",
                               "Startup (s, mediana)": "{:,.0f}"})
    return df


def tab_resultados():
    rows = []
    for r in HEDGED:
        for model, lab in (("melhor-de-validação", "best_val"), ("último", "last")):
            for cj, sname in (("val", "validação"), ("test", "teste")):
                s = row(SUM[r], f"{lab}_{cj}_fee1.0")
                rows.append({"Execução": r, "Modelo": model, "Conjunto": sname,
                             "P/L real (R$)": s["real_pnl_hedged_brl"],
                             "P/L médio/dia (R$)": s["pnl_mean_day"],
                             "Perna WIN (R$)": s["pnl_win_leg_total"], "Perna BOVA11 (R$)": s["pnl_bova_leg_total"],
                             "Bruto do par (R$)": s["gross_mtm_total"], "Custos (R$)": s["cost_total"],
                             "Negócios/dia": s["trades_per_day"], "Acerto": 100 * s["win_rate"],
                             "Duração média (ticks)": s["mean_trade_duration_ticks"]})
    df = pd.DataFrame(rows)
    write_tab("resultados_hedged", df, {"Negócios/dia": "{:,.1f}", "Acerto": "{:,.0f}%",
                                        "Duração média (ticks)": "{:,.0f}"})
    return df


def daily_stats():
    """Estatística por dia (best_val): por seed e agregada entre seeds."""
    out, per_set = [], {}
    for cj, sname in (("val", "validação"), ("test", "teste")):
        mat = {}
        for r in HEDGED:
            d = load_days(r, f"best_val_{cj}_fee1.0")
            mat[r] = d.set_index("order")["real_pnl_brl"]
            x = mat[r].to_numpy()
            lo, hi = boot_ci(x)
            top3 = np.sort(x)[::-1][:3].sum() / x.sum() if x.sum() > 0 else np.nan
            out.append({"Execução": r, "Conjunto": sname, "P/L médio/dia (R$)": x.mean(),
                        "IC95% inferior": lo, "IC95% superior": hi,
                        "Dias positivos": f"{(x > 0).sum()}/{len(x)}",
                        "p (média=0)": stats.ttest_1samp(x, 0.0).pvalue,
                        "3 melhores dias / total": 100 * top3,
                        "corr(P/L do dia, movimento do WIN)": np.corrcoef(x, d["win_move"].to_numpy())[0, 1]})
        df = pd.DataFrame(mat)
        per_set[cj] = df
        x = df.mean(axis=1).to_numpy()
        lo, hi = boot_ci(x)
        out.append({"Execução": "média das 4 seeds", "Conjunto": sname, "P/L médio/dia (R$)": x.mean(),
                    "IC95% inferior": lo, "IC95% superior": hi, "Dias positivos": f"{(x > 0).sum()}/{len(x)}",
                    "p (média=0)": stats.ttest_1samp(x, 0.0).pvalue,
                    "3 melhores dias / total": 100 * (np.sort(x)[::-1][:3].sum() / x.sum()),
                    "corr(P/L do dia, movimento do WIN)": np.corrcoef(
                        x, load_days("hedged_s0", f"best_val_{cj}_fee1.0")["win_move"].to_numpy())[0, 1]})
    df = pd.DataFrame(out)
    write_tab("estatistica_diaria", df, {"IC95% inferior": "{:,.1f}", "IC95% superior": "{:,.1f}",
                                         "p (média=0)": "{:.3f}", "3 melhores dias / total": "{:,.0f}%",
                                         "corr(P/L do dia, movimento do WIN)": "{:+.2f}"})
    return df, per_set


def tab_robustez():
    rows = []
    for r in HEDGED:
        for cj, sname in (("val", "validação"), ("test", "teste")):
            rows.append({"Execução": r, "Conjunto": sname,
                         "Custo ×0 (bruto do par)": real(r, f"cost_sens_{cj}_fee0.0"),
                         "Custo ×0,5": real(r, f"cost_sens_{cj}_fee0.5"),
                         "Especificação (×1)": real(r, f"best_val_{cj}_fee1.0"),
                         "Especificação + meio-spread bid/ask": real(r, f"spreadcost_{cj}_fee1.0"),
                         "Latência de 1 tick": real(r, f"latency1_{cj}_fee1.0")})
    df = pd.DataFrame(rows)
    write_tab("robustez", df)
    return df


def tab_baselines():
    s = SUM["hedged_s0"]
    rows = []
    for name, lab in (("Flat (não operar)", "baseline_flat"), ("Regra de limiar |z|>1", "baseline_threshold1.0"),
                      ("Aleatório", "baseline_random")):
        v, t = row(s, f"{lab}_val_fee1.0"), row(s, f"{lab}_test_fee1.0")
        rows.append({"Baseline": name, "P/L validação (R$)": v["real_pnl_hedged_brl"],
                     "P/L teste (R$)": t["real_pnl_hedged_brl"], "Negócios/dia (teste)": t["trades_per_day"]})
    df = pd.DataFrame(rows)
    write_tab("baselines", df, {"Negócios/dia (teste)": "{:,.1f}"})
    return df


def tab_checkpoints():
    rows, pooled_v, pooled_t = [], [], []
    for r in HEDGED:
        ck = load_ckcurve(r)
        rho, p = stats.spearmanr(ck["val_pnl_total"], ck["test_pnl_total"])
        best = STATE[r]["best_val_steps"]
        chosen = ck.loc[(ck["timesteps"] - best).abs().idxmin()]
        rows.append({"Execução": r, "Checkpoints": len(ck), "ρ de Spearman (val × teste)": rho, "p": p,
                     "Teste do escolhido (R$)": chosen["test_pnl_total"],
                     "Teste médio dos checkpoints (R$)": ck["test_pnl_total"].mean(),
                     "Teste do último (R$)": ck["test_pnl_total"].iloc[-1],
                     "Checkpoints com teste > 0": f"{(ck['test_pnl_total'] > 0).sum()}/{len(ck)}",
                     "Checkpoints com val > 0": f"{(ck['val_pnl_total'] > 0).sum()}/{len(ck)}"})
        pooled_v += list(ck["val_pnl_total"]); pooled_t += list(ck["test_pnl_total"])
    rho, p = stats.spearmanr(pooled_v, pooled_t)
    rows.append({"Execução": "todos (80 pontos)", "Checkpoints": len(pooled_v),
                 "ρ de Spearman (val × teste)": rho, "p": p, "Teste do escolhido (R$)": np.nan,
                 "Teste médio dos checkpoints (R$)": float(np.mean(pooled_t)), "Teste do último (R$)": np.nan,
                 "Checkpoints com teste > 0": f"{(np.array(pooled_t) > 0).sum()}/{len(pooled_t)}",
                 "Checkpoints com val > 0": f"{(np.array(pooled_v) > 0).sum()}/{len(pooled_v)}"})
    df = pd.DataFrame(rows)
    write_tab("checkpoints", df, {"ρ de Spearman (val × teste)": "{:+.2f}", "p": "{:.3f}"})
    return df


def tab_spread():
    rows = []
    for r in ["hedged_s0"] + SPREAD:
        for cj, sname in (("val", "validação"), ("test", "teste")):
            s = row(SUM[r], f"best_val_{cj}_fee1.0")
            rows.append({"Execução": r, "Conjunto": sname, "Unidade da recompensa": s["unit"],
                         "P/L de treino (recompensa)": s["pnl_total"],
                         "P/L REAL hedgeado (R$)": s["real_pnl_hedged_brl"],
                         "Convergência bruta": s["gross_mtm_total"], "Custo de treino": s["cost_total"],
                         "Negócios/dia": s["trades_per_day"], "Acerto (recompensa)": 100 * s["win_rate"],
                         "Duração média (ticks)": s["mean_trade_duration_ticks"]})
    df = pd.DataFrame(rows)
    write_tab("spread", df, {"Negócios/dia": "{:,.0f}", "Acerto (recompensa)": "{:,.0f}%",
                             "Duração média (ticks)": "{:,.0f}"})
    return df


def trade_profile():
    frames = []
    for r in HEDGED:
        for cj in ("val", "test"):
            t = load_trades(r, f"best_val_{cj}_fee1.0")
            t["run"], t["set"] = r, cj
            frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    rows = []
    for cj, sname in (("val", "validação"), ("test", "teste")):
        x = t[t["set"] == cj]
        rows.append({"Conjunto": sname, "Negócios (4 seeds)": len(x),
                     "P/L líquido médio por negócio (R$)": x["pnl"].mean(),
                     "Mediana (R$)": x["pnl"].median(),
                     "Acerto": 100 * (x["pnl"] > 0).mean(),
                     "Ganho médio (R$)": x.loc[x["pnl"] > 0, "pnl"].mean(),
                     "Perda média (R$)": x.loc[x["pnl"] <= 0, "pnl"].mean(),
                     "Duração mediana (ticks)": x["duration"].median(),
                     "Duração p10–p90 (ticks)": br(f"{x['duration'].quantile(.1):,.0f}–{x['duration'].quantile(.9):,.0f}"),
                     "Comprado no par": 100 * (x["direction"] > 0).mean(),
                     "Convergência média do spread (pts)": x["d_misp_dir"].mean()})
    df = pd.DataFrame(rows)
    write_tab("perfil_negocios", df, {"Acerto": "{:,.0f}%", "Duração mediana (ticks)": "{:,.0f}",
                                      "Comprado no par": "{:,.0f}%"})
    return df, t


# --------------------------------------------------------------------------- contexto
def parse_ctx():
    """Números do oráculo / regra causal / hold hedgeado nos pregões de val e teste."""
    if not os.path.isdir(CTX):
        return None
    res = {}
    for cj in ("val", "test"):
        for suf, key in (("", "spec"), ("_spread", "spreadcost")):
            p = os.path.join(CTX, f"opp_{cj}{suf}.log")
            if not os.path.exists(p):
                continue
            txt = open(p, encoding="utf-8", errors="ignore").read()
            d = {}
            m = re.search(r"teto de P/L por dia com previsão perfeita: média R\$ ([\d.\-]+) \(mín ([\d.\-]+), máx ([\d.\-]+)\); "
                          r"negócios/dia: ([\d.]+)", txt)
            if m:
                d["oracle_day"], d["oracle_min"], d["oracle_max"], d["oracle_trades"] = map(float, m.groups())
            for mg in re.finditer(r"margem ([\d.]+)x:\s+([\d.]+) negócios/dia \| P/L líquido total R\$\s+([\d.\-]+) \(\s*([\d.\-]+)/dia\)"
                                  r" \| acerto médio ([\d.]+|nan)%", txt):
                d[f"rule_{mg.group(1)}"] = (float(mg.group(2)), float(mg.group(3)), float(mg.group(4)))
            m = re.search(r"custo de ida e volta ≈ ([\d.]+) pontos", txt)
            if m:
                d["cost_pts"] = float(m.group(1))
            res[(cj, key)] = d
    p = os.path.join(CTX, "hold_hedged.csv")
    hold = pd.read_csv(p) if os.path.exists(p) else None
    return res, hold


def tab_contexto(ctx, per_set):
    if ctx is None:
        return None
    res, hold = ctx
    rows = []
    nd = {"val": 30, "test": 30}
    agent = {cj: per_set[cj].mean(axis=1).mean() for cj in ("val", "test")}
    for cj, sname in (("val", "validação"), ("test", "teste")):
        r_spec, r_sp = res.get((cj, "spec"), {}), res.get((cj, "spreadcost"), {})
        rows.append({"Referência": "Agentes hedged (média das 4 seeds, melhor-de-validação)", "Conjunto": sname,
                     "R$/dia (custos da especificação)": agent[cj],
                     "R$/dia (com meio-spread bid/ask)": float(np.mean(
                         [real(r, f"spreadcost_{cj}_fee1.0") for r in HEDGED])) / nd[cj]})
        if "oracle_day" in r_spec:
            rows.append({"Referência": "Teto com previsão perfeita (oráculo, não operável)", "Conjunto": sname,
                         "R$/dia (custos da especificação)": r_spec["oracle_day"],
                         "R$/dia (com meio-spread bid/ask)": r_sp.get("oracle_day", np.nan)})
        for m in ("1.0", "1.5", "2.0"):
            if f"rule_{m}" in r_spec:
                rows.append({"Referência": f"Regra causal 'spread − custos' (limiar {m}× o custo)", "Conjunto": sname,
                             "R$/dia (custos da especificação)": r_spec[f"rule_{m}"][2],
                             "R$/dia (com meio-spread bid/ask)": r_sp.get(f"rule_{m}", (0, 0, np.nan))[2]})
        if hold is not None:
            h = hold[(hold["order"] >= (362 if cj == "val" else 392)) & (hold["order"] <= (391 if cj == "val" else 421))]
            for act, nm in (("long_par", "Sempre comprado no par (segurar o dia)"),
                            ("short_par", "Sempre vendido no par (segurar o dia)")):
                a = h[(h["action"] == act) & (~h["spread_cost"])]["pnl_brl"].sum() / nd[cj]
                b = h[(h["action"] == act) & (h["spread_cost"])]["pnl_brl"].sum() / nd[cj]
                rows.append({"Referência": nm, "Conjunto": sname,
                             "R$/dia (custos da especificação)": a, "R$/dia (com meio-spread bid/ask)": b})
        s = SUM["hedged_s0"]
        rows.append({"Referência": "Regra de limiar |z|>1 (baseline do pipeline)", "Conjunto": sname,
                     "R$/dia (custos da especificação)": float(row(s, f"baseline_threshold1.0_{cj}_fee1.0")["real_pnl_hedged_brl"]) / nd[cj],
                     "R$/dia (com meio-spread bid/ask)": np.nan})
    df = pd.DataFrame(rows)
    write_tab("contexto", df)
    return df


# --------------------------------------------------------------------------- figuras
def fig_treino():
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for r in HEDGED:
        p = load_progress(r)
        x = p["time/total_timesteps"] / 1e6
        ax = axs[0, 0]
        ok = p["rollout/ep_rew_mean"].notna()
        ax.plot(x[ok], smooth(p.loc[ok, "rollout/ep_rew_mean"] / 5.0), color=C_SEED[r], label=SEEDNAME[r])
        ok = p["train/trades_per_episode"].notna()
        axs[0, 1].plot(x[ok], smooth(p.loc[ok, "train/trades_per_episode"]), color=C_SEED[r])
        ok = p["train/entropy_loss"].notna()
        axs[1, 0].plot(x[ok], -smooth(p.loc[ok, "train/entropy_loss"]), color=C_SEED[r])
        ok = p["train/explained_variance"].notna()
        axs[1, 1].plot(x[ok], smooth(p.loc[ok, "train/explained_variance"]), color=C_SEED[r])
    axs[0, 0].set_title("Recompensa média por episódio (R$, política estocástica)")
    axs[0, 0].set_ylim(-60, 200)
    axs[0, 0].axhline(0, color=INK2, lw=0.8)
    axs[0, 0].text(0.02, 0.06, "início ≈ −100.000 R$ (fora da escala)",
                   transform=axs[0, 0].transAxes, fontsize=7.5, color=INK2)
    axs[0, 1].set_title("Negócios por episódio (pregão) no treino")
    axs[0, 1].set_yscale("log")
    axs[1, 0].set_title("Entropia da política (nats; máximo ln 3 = 1,10)")
    axs[1, 1].set_title("Variância explicada do crítico")
    for ax in axs.ravel():
        ax.set_xlabel("timesteps (milhões)")
    axs[0, 0].legend(ncol=2, loc="lower right")
    fig.suptitle("Treino do par hedgeado (4 seeds, 100M timesteps, ent_coef = 0,01)", y=1.0, fontsize=11)
    fig.tight_layout()
    save(fig, "v5_fig1_treino.png")


def fig_val_teste():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, r in zip(axs, HEDGED):
        ck, vc = load_ckcurve(r), load_valcurve(r)
        x = ck["timesteps"] / 1e6
        ax.plot(vc["timesteps"] / 1e6, vc["train_subset_pnl_mean_day"], color=C_TRAIN, marker="o", ms=3, label="treino (10 pregões)")
        ax.plot(x, ck["val_pnl_total"] / 30, color=C_VAL, marker="s", ms=3, label="validação (30)")
        ax.plot(x, ck["test_pnl_total"] / 30, color=C_TEST, marker="^", ms=3, label="teste (30)")
        ax.axhline(0, color=INK2, lw=0.8)
        b = STATE[r]["best_val_steps"] / 1e6
        ax.axvline(b, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"{SEEDNAME[r]} (melhor de validação: {b:.0f}M)")
        ax.set_xlabel("timesteps (milhões)")
    axs[0].set_ylabel("P/L real médio por dia (R$)")
    axs[0].legend(loc="upper right")
    fig.suptitle("Treino × validação × teste por checkpoint (o teste é só para relatório)", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "v5_fig2_treino_val_teste.png")


def fig_robustez(rob):
    cats = ["Custo ×0\n(bruto do par)", "Custo ×0,5", "Especificação\n(×1)", "+ meio-spread\nbid/ask", "Latência\n1 tick"]
    cols = ["Custo ×0 (bruto do par)", "Custo ×0,5", "Especificação (×1)", "Especificação + meio-spread bid/ask", "Latência de 1 tick"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    for ax, (sname, cj) in zip(axs, (("validação", "validação"), ("teste", "teste"))):
        for r in HEDGED:
            v = rob[(rob["Execução"] == r) & (rob["Conjunto"] == sname)].iloc[0][cols].to_numpy(dtype=float)
            ax.plot(range(5), v, color=C_SEED[r], marker=MARK[r], ms=5, label=SEEDNAME[r])
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_xticks(range(5)); ax.set_xticklabels(cats)
        ax.set_title(f"P/L real do melhor-de-validação — {sname} (30 pregões)")
        symlog_axis(ax, 500, [-3000, -1000, -300, 0, 300, 1000, 3000, 10000, 20000])
        ax.set_ylabel("R$ (escala symlog)")
    axs[0].legend(loc="upper right")
    fig.suptitle("O lucro some com custo realista e com 1 tick de atraso", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "v5_fig3_robustez.png")


def fig_equity(per_set, ctx):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (cj, sname) in zip(axs, (("val", "validação"), ("test", "teste"))):
        df = per_set[cj]
        x = np.arange(1, len(df) + 1)
        for r in HEDGED:
            ax.plot(x, df[r].cumsum(), color=C_SEED[r], marker=MARK[r], ms=3, label=SEEDNAME[r])
        ax.axhline(0, color=INK, lw=1.0, label="flat (não operar)")
        if ctx is not None and ctx[1] is not None:
            hold = ctx[1]
            lo, hi = (362, 391) if cj == "val" else (392, 421)
            h = hold[(hold["order"] >= lo) & (hold["order"] <= hi) & (~hold["spread_cost"])]
            for act, nm, ls in (("long_par", "sempre comprado no par", "--"), ("short_par", "sempre vendido no par", ":")):
                y = h[h["action"] == act].sort_values("order")["pnl_brl"].cumsum().to_numpy()
                ax.plot(x, y, color=C_TRAIN, ls=ls, label=nm)
        ax.set_title(f"P/L real acumulado — {sname}")
        ax.set_xlabel("dia (ordem cronológica)")
        ax.set_ylabel("R$")
    axs[0].legend(loc="upper left", ncol=1)
    fig.tight_layout()
    save(fig, "v5_fig4_equity.png")


def fig_decomposicao():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    comp = [("Perna WIN", "pnl_win_leg_total", SLOT[0]), ("Perna BOVA11", "pnl_bova_leg_total", SLOT[1]),
            ("Custos (sinal −)", "cost_total", SLOT[2]), ("Líquido", "real_pnl_hedged_brl", SLOT[3])]
    for ax, (cj, sname) in zip(axs, (("val", "validação"), ("test", "teste"))):
        w = 0.2
        for k, (nm, col, c) in enumerate(comp):
            vals = []
            for r in HEDGED:
                v = float(row(SUM[r], f"best_val_{cj}_fee1.0")[col])
                vals.append(-v if col == "cost_total" else v)
            ax.bar(np.arange(4) + (k - 1.5) * w, vals, w * 0.9, color=c, label=nm)
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_xticks(range(4)); ax.set_xticklabels([SEEDNAME[r] for r in HEDGED])
        ax.set_title(f"Decomposição do P/L do melhor-de-validação — {sname} (R$)")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.28)
    axs[0].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "v5_fig5_decomposicao.png")


def fig_selecao():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True, sharey=True)
    for ax, r in zip(axs, HEDGED):
        ck = load_ckcurve(r)
        ax.scatter(ck["val_pnl_total"], ck["test_pnl_total"], color=C_SEED[r], s=22)
        best = STATE[r]["best_val_steps"]
        c = ck.loc[(ck["timesteps"] - best).abs().idxmin()]
        ax.scatter([c["val_pnl_total"]], [c["test_pnl_total"]], s=90, facecolors="none", edgecolors=INK, linewidths=1.4)
        rho, p = stats.spearmanr(ck["val_pnl_total"], ck["test_pnl_total"])
        ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{SEEDNAME[r]}: ρ = {rho:+.2f}")
        ax.set_xlabel("P/L de validação (R$)")
    axs[0].set_ylabel("P/L de teste (R$)")
    fig.suptitle("A validação prevê o teste? Cada ponto é um checkpoint (círculo = escolhido pela validação)", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "v5_fig6_selecao.png")


def fig_spread():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for r in ["hedged_s0"] + SPREAD:
        vc = load_valcurve(r)
        x = vc["timesteps"] / 1e6
        axs[0].plot(x, vc["val_trades_per_day"], color=C_VAR[r], marker="o", ms=3, label=r)
        axs[1].plot(x, vc["val_real_pnl_hedged_brl"], color=C_VAR[r], marker="o", ms=3, label=r)
    axs[0].set_yscale("log"); axs[0].set_title("Negócios por dia na validação (escala log)")
    symlog_axis(axs[1], 1000, [-2e6, -1e6, -1e5, -1e4, -1e3, 0, 1e3, 1e4]); axs[1].axhline(0, color=INK, lw=0.9)
    axs[1].set_title("P/L REAL do par hedgeado na validação (R$, symlog)")
    for ax in axs:
        ax.set_xlabel("timesteps (milhões)")
    axs[0].legend(loc="center right")
    fig.suptitle("Recompensa de convergência do spread: o agente opera milhares de vezes e perde dinheiro real", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "v5_fig7_spread.png")


def fig_perfil(trades):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    t = trades
    axs[0].hist(t["pnl"].clip(-60, 120), bins=60, color=SLOT[0])
    axs[0].axvline(0, color=INK, lw=1.0)
    axs[0].set_title("P/L líquido por negócio (R$, cortado em −60/+120)")
    axs[0].set_xlabel("R$"); axs[0].set_ylabel("negócios (4 seeds, val+teste)")
    axs[1].hist(np.log10(t["duration"].clip(lower=1)), bins=40, color=SLOT[1])
    axs[1].set_title("Duração dos negócios (ticks, escala log)")
    axs[1].set_xlabel("ticks")
    axs[1].set_xticks([0, 1, 2, 3, 4]); axs[1].set_xticklabels(["1", "10", "100", "1.000", "10.000"])
    g = t.groupby("hour_entry").agg(n=("pnl", "size"), mean=("pnl", "mean"), total=("pnl", "sum"))
    axs[2].bar(g.index, g["total"], color=SLOT[2])
    axs[2].axhline(0, color=INK, lw=1.0)
    axs[2].set_title("P/L total por hora de entrada (R$)")
    axs[2].set_xlabel("hora do pregão")
    for h, row_ in g.iterrows():
        axs[2].text(h, row_["total"], f"n={int(row_['n'])}", ha="center", va="bottom" if row_["total"] >= 0 else "top", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "v5_fig8_perfil_negocios.png")
    return g


def fig_event():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    ev_rows = []
    for r in HEDGED:
        ev = pd.concat([pd.read_csv(os.path.join(rdir(r), "eval", f"best_val_{cj}_fee1.0_eventstudy.csv")) for cj in ("val", "test")])
        g = ev[ev["group"] == "all"].groupby("offset_ticks").apply(
            lambda d: np.average(d["mean_directed_d_misp"], weights=d["n"]), include_groups=False)
        axs[0].plot(g.index, g.values, color=C_SEED[r], marker=MARK[r], ms=4, label=SEEDNAME[r])
        ev_rows.append(pd.DataFrame({"run": r, "offset": g.index, "d_misp": g.values}))
        b = pd.concat([pd.read_csv(os.path.join(rdir(r), "eval", f"best_val_{cj}_fee1.0_entry_buckets.csv")) for cj in ("val", "test")])
        b = b.groupby("bucket").apply(lambda d: np.average(d["pnl_mean"], weights=d["n"]), include_groups=False)
        order = ["z<0", "0<=z<1", "1<=z<2", "z>=2"]
        b = b.reindex(order)
        axs[1].plot(range(4), b.values, color=C_SEED[r], marker=MARK[r], ms=4, label=SEEDNAME[r])
    axs[0].axhline(0, color=INK, lw=1.0); axs[1].axhline(0, color=INK, lw=1.0)
    axs[0].set_title("Variação do spread em torno da entrada (pts, a favor da posição)")
    axs[0].set_xlabel("ticks em relação à entrada (negativo = antes)")
    axs[1].set_title("P/L médio por negócio (R$) × força do sinal na entrada")
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(["z < 0", "0 ≤ z < 1", "1 ≤ z < 2", "z ≥ 2"])
    axs[1].set_xlabel("z orientado do spread (val+teste)")
    axs[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "v5_fig9_event_study.png")
    pd.concat(ev_rows).to_csv(os.path.join(OUT_TAB, "tab_event_study.csv"), index=False)


def fig_contexto(ctx_df):
    if ctx_df is None:
        return
    d = ctx_df[ctx_df["Conjunto"] == "teste"].copy()
    d = d[~d["Referência"].str.startswith("Regra de limiar")]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["R$/dia (custos da especificação)"], 0.38, color=SLOT[0], label="custos da especificação")
    ax.barh(y + 0.2, d["R$/dia (com meio-spread bid/ask)"], 0.38, color=SLOT[1], label="+ meio-spread bid/ask")
    names = (d["Referência"]
             .str.replace(r"Regra causal 'spread − custos' \(limiar ([\d.]+)× o custo\)", lambda m: "Regra causal, limiar " + m.group(1) + "× o custo", regex=True)
             .str.replace(r" \((média|oráculo|segurar).*\)$", "", regex=True))
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis(); ax.axvline(0, color=INK, lw=1.0)
    ax.set_xlabel("R$ por dia (pregões de teste)")
    ax.set_title("Agentes hedged × referências, no mesmo período de teste (30 pregões, 1 contrato)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "v5_fig10_contexto.png")


def main():
    t1 = tab_execucao()
    res = tab_resultados()
    dstats, per_set = daily_stats()
    rob = tab_robustez()
    base = tab_baselines()
    ck = tab_checkpoints()
    sp = tab_spread()
    prof, trades = trade_profile()
    ctx = parse_ctx()
    ctx_df = tab_contexto(ctx, per_set)
    fig_treino(); fig_val_teste(); fig_robustez(rob); fig_equity(per_set, ctx)
    fig_decomposicao(); fig_selecao(); fig_spread(); hours = fig_perfil(trades); fig_event()
    fig_contexto(ctx_df)
    hours.to_csv(os.path.join(OUT_TAB, "tab_hora_entrada.csv"))
    print("ok")


if __name__ == "__main__":
    main()
