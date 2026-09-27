"""
Análise da avaliação walk-forward da v5 (3 folds: treino de 300/360/420 pregões, cada um
com 30 de validação e 30 de teste; ver docs/relatorio_walkforward.md). Fold 360 é o
mesmo da campanha v5 (reaproveita hedged_s0..s3, sem retreinar); folds 300 e 420 vêm da
rodada nova (`slurm/submit_walkforward.sh`, RUN_TAG=wf_<300|420>_s<0|1>).

Só lê CSV/JSON já gravados pelo pipeline (state.json, metadata.json, eval/summary.csv,
eval/*_days.csv) + computa a referência "segurar o par" na hora, chamando
hold_baselines_hedged.run_hold nos mesmos pregões de cada fold (precisa dos dados de
tick, TICK_DATA_DIR). Um fold sem backup disponível é pulado (aviso no console), o
resto da análise segue com o que houver.

Uso (a partir da raiz do repo, no WSL/venv com matplotlib e scipy; TICK_DATA_DIR
apontando para os JSONs de tick):
    python src/analyze_walkforward.py
    python src/analyze_walkforward.py --v5-backup sdumont_backup_campanha --wf-backup sdumont_backup_walkforward
"""

import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hold_baselines_hedged import run_hold

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
FOLD_COLOR = {300: SLOT[0], 360: SLOT[1], 420: SLOT[2]}
OUT_IMG = os.path.join("docs", "img")
OUT_TAB = os.path.join("docs", "analysis_walkforward")

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 1.6, "font.size": 9,
    "axes.axisbelow": True,
})


def rdir(backup, run_tag, fold_num):
    for cand in (os.path.join(backup, "src", "runs", run_tag, f"fold{fold_num}"),
                os.path.join(backup, "pairs-trading-rl", "src", "runs", run_tag, f"fold{fold_num}")):
        if os.path.isdir(cand):
            return cand
    return None


def load_run(backup, run_tag, fold_num):
    d = rdir(backup, run_tag, fold_num)
    if d is None or not os.path.exists(os.path.join(d, "eval", "summary.csv")):
        return None
    meta = json.load(open(os.path.join(d, "metadata.json")))
    summary = pd.read_csv(os.path.join(d, "eval", "summary.csv"))
    days = {}
    for label in ("best_val_test_fee1.0", "best_val_val_fee1.0", "spreadcost_test_fee1.0",
                  "latency1_test_fee1.0"):
        p = os.path.join(d, "eval", f"{label}_days.csv")
        if os.path.exists(p):
            days[label] = pd.read_csv(p)
    return {"dir": d, "meta": meta, "summary": summary, "days": days}


def row_of(summary, label):
    r = summary[summary.label == label]
    return r.iloc[0] if len(r) else None


def boot_ci(x, n=10000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x))).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def hold_ref(orders, cache):
    """P/L médio/dia da referência 'segurar o par' (melhor entre long/short, custos da
    especificação), computado nos pregões `orders` -- memoizado em `cache` (dict) porque
    val/teste de folds diferentes não se repetem, mas o mesmo fold em 2 seeds sim."""
    key = tuple(orders)
    if key not in cache:
        rows = [run_hold(o, a, False) for o in orders for a in (1, 2)]
        df = pd.DataFrame(rows)
        by_action = df.groupby("action")["pnl_brl"].sum()
        best = by_action.idxmax()
        cache[key] = {"acao": best, "pnl_total": float(by_action[best]),
                     "pnl_dia": float(by_action[best] / len(orders))}
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-backup", default="sdumont_backup_campanha")
    ap.add_argument("--wf-backup", default="sdumont_backup_walkforward")
    a = ap.parse_args()
    os.makedirs(OUT_TAB, exist_ok=True)
    os.makedirs(OUT_IMG, exist_ok=True)

    # (tamanho_treino, fold_num_no_plano) -> [(run_tag, seed, backup, fold_num_da_execução), ...]
    # hedged_s0..s3 (v5) rodaram com TRAIN_SIZES=360 SOZINHO -> fold 1 na pasta deles;
    # wf_300/420 rodam com TRAIN_SIZES="300,360,420" -> fold 1/2/3 conforme o tamanho.
    plan = {
        300: [(f"wf_300_s{s}", s, a.wf_backup, 1) for s in (0, 1)],
        360: [(f"hedged_s{s}", s, a.v5_backup, 1) for s in (0, 1, 2, 3)],
        420: [(f"wf_420_s{s}", s, a.wf_backup, 3) for s in (0, 1)],
    }

    hold_cache = {}
    rows = []
    pooled_test_pnl_by_size = {}
    ck_by_size = {}
    for size, entries in plan.items():
        for run_tag, seed, backup, fold_num in entries:
            run = load_run(backup, run_tag, fold_num)
            if run is None:
                print(f"[pendente] {run_tag} (fold {fold_num}, treino={size}) não encontrada em {backup}; pulando")
                continue
            meta, summary, days = run["meta"], run["summary"], run["days"]
            tr, va, te = meta["train_orders"], meta["val_orders"], meta["test_orders"]
            bv_val, bv_test = row_of(summary, "best_val_val_fee1.0"), row_of(summary, "best_val_test_fee1.0")
            sc_test = row_of(summary, "spreadcost_test_fee1.0")
            lat_test = row_of(summary, "latency1_test_fee1.0")
            d_test = days.get("best_val_test_fee1.0")
            corr = win_move = np.nan
            if d_test is not None and len(d_test) > 2:
                corr = float(np.corrcoef(d_test.real_pnl_brl, d_test.win_move)[0, 1])
                win_move = float(d_test.win_move.sum())
                pooled_test_pnl_by_size.setdefault(size, []).extend(d_test.real_pnl_brl.tolist())
            ck_path = os.path.join(run["dir"], "eval", "checkpoint_curve.csv")
            if os.path.exists(ck_path):
                ck_by_size.setdefault(size, {})[(run_tag, seed)] = pd.read_csv(ck_path)
            hb = hold_ref(te, hold_cache)
            n_test = len(te)
            rows.append({
                "treino_dias": size, "execução": run_tag, "seed": seed,
                "treino": f"{tr['first']}-{tr['last']}", "val": f"{va[0]}-{va[-1]}",
                "teste": f"{te[0]}-{te[-1]}",
                "pnl_val_R$": bv_val.real_pnl_hedged_brl if bv_val is not None else np.nan,
                "pnl_teste_R$": bv_test.real_pnl_hedged_brl if bv_test is not None else np.nan,
                "pnl_teste_R$_dia": (bv_test.real_pnl_hedged_brl / n_test) if bv_test is not None else np.nan,
                "trades_dia_teste": bv_test.trades_per_day if bv_test is not None else np.nan,
                "pct_dias_positivos": float((d_test.real_pnl_brl > 0).mean()) if d_test is not None else np.nan,
                "corr_pnl_x_movimento_win": corr,
                "pnl_teste_meiospread_R$": sc_test.real_pnl_hedged_brl if sc_test is not None else np.nan,
                "pnl_teste_latencia1_R$": lat_test.real_pnl_hedged_brl if lat_test is not None else np.nan,
                "baseline_flat_R$": 0.0,
                "baseline_segurar_par_R$_dia": hb["pnl_dia"], "baseline_segurar_par_acao": hb["acao"],
                "movimento_win_teste_pts": win_move, "n_dias_teste": n_test,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("Nenhum backup encontrado ainda -- nada para analisar (rode depois que a rodada terminar).")
        return
    df.to_csv(os.path.join(OUT_TAB, "tab_folds.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.round(2).to_string(index=False))

    agg_rows = []
    for size, pnls in sorted(pooled_test_pnl_by_size.items()):
        pnls = np.array(pnls)
        lo, hi = boot_ci(pnls)
        agg_rows.append({"treino_dias": size, "dias_de_teste_agrupados": len(pnls),
                         "pnl_medio_dia_R$": pnls.mean(), "ic95_inf": lo, "ic95_sup": hi,
                         "pct_dias_positivos": float((pnls > 0).mean())})
    all_pnls = np.concatenate(list(pooled_test_pnl_by_size.values())) if pooled_test_pnl_by_size else np.array([])
    if len(all_pnls):
        lo, hi = boot_ci(all_pnls)
        agg_rows.append({"treino_dias": "todos", "dias_de_teste_agrupados": len(all_pnls),
                         "pnl_medio_dia_R$": all_pnls.mean(), "ic95_inf": lo, "ic95_sup": hi,
                         "pct_dias_positivos": float((all_pnls > 0).mean())})
    n_folds_pos = sum(1 for size, pnls in pooled_test_pnl_by_size.items() if np.mean(pnls) > 0)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(os.path.join(OUT_TAB, "tab_agregado.csv"), index=False)
    print("\n--- agregado ---")
    print(agg.round(2).to_string(index=False))
    print(f"folds (agregando as seeds) com P/L médio de teste > 0: {n_folds_pos}/{len(pooled_test_pnl_by_size)}")

    # figura: P/L de teste acumulado por fold (uma linha por seed, tracejada = média do fold)
    fig, ax = plt.subplots(figsize=(9, 5))
    for size in sorted(pooled_test_pnl_by_size):
        entries = plan[size]
        curves = []
        for run_tag, seed, backup, fold_num in entries:
            run = load_run(backup, run_tag, fold_num)
            if run is None:
                continue
            d_test = run["days"].get("best_val_test_fee1.0")
            if d_test is None:
                continue
            cum = d_test.sort_values("order").real_pnl_brl.cumsum().to_numpy()
            curves.append(cum)
            ax.plot(range(1, len(cum) + 1), cum, color=FOLD_COLOR[size], alpha=0.35, lw=1.1)
        if curves:
            m = min(len(c) for c in curves)
            mean_curve = np.mean([c[:m] for c in curves], axis=0)
            ax.plot(range(1, m + 1), mean_curve, color=FOLD_COLOR[size], lw=2.4,
                    label=f"treino={size} dias (média de {len(curves)} seed(s))")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlabel("dia de teste (ordem cronológica)")
    ax.set_ylabel("P/L real acumulado (R$)")
    ax.set_title("P/L de teste acumulado por fold (linhas finas = seeds; grossa = média)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT_IMG, "v8_fig1_walkforward_equity.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", path)


if __name__ == "__main__":
    main()
