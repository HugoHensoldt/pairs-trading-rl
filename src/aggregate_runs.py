"""
Agrega os resultados de avaliação da v4 entre seeds.

Lê `runs/<prefixo>*/fold<N>/eval/summary.csv` (um por seed/fold, gerado por
run_eval_job) e produz, por fold x modelo x conjunto (val/test), média e
desvio-padrão entre seeds das métricas principais, incluindo a decomposição
do P/L bruto (convergência do spread vs componente direcional).

Uso (de dentro de src/, ou apontando RUNS_DIR):
    python aggregate_runs.py                 # prefixo padrão: v4_s
    python aggregate_runs.py v4_s runs       # prefixo e diretório
Saída: imprime as tabelas e grava <runs>/aggregate_summary.csv.
"""

import glob
import os
import re
import sys

import pandas as pd

METRICS = [
    "pnl_total", "real_pnl_hedged_brl", "trades_per_day", "win_rate", "gross_mtm_total", "cost_total",
    "pnl_spread_component", "pnl_fair_component", "corr_gross_vs_win_move",
    "mean_net_exposure",
]


def load(prefix, runs_dir):
    frames = []
    pattern = os.path.join(runs_dir, f"{prefix}*", "fold*", "eval", "summary.csv")
    for path in sorted(glob.glob(pattern)):
        m = re.search(rf"{re.escape(prefix)}(\w+?)[/\\]fold(\d+)[/\\]eval", path)
        if not m:
            continue
        df = pd.read_csv(path)
        df.insert(0, "seed", m.group(1))
        df.insert(1, "fold", int(m.group(2)))
        frames.append(df)
    if not frames:
        raise SystemExit(f"nenhum summary.csv encontrado em {pattern}")
    return pd.concat(frames, ignore_index=True)


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "v4_s"
    runs_dir = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("RUNS_DIR", "runs")
    df = load(prefix, runs_dir)

    metrics = [m for m in METRICS if m in df.columns]
    # só o fee de treino (2.5 por padrão) nas tabelas principais; latência e
    # sensibilidade a custo têm labels próprios e ficam na saída completa
    grp = df.groupby(["fold", "model", "set", "label"], sort=True)
    agg = grp[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = grp["seed"].nunique()
    agg = agg.reset_index()

    out = os.path.join(runs_dir, "aggregate_summary.csv")
    agg.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    for fold, sub in agg.groupby("fold"):
        print(f"\n=== Fold {fold} ({int(sub['n_seeds'].max())} seed(s)) ===")
        view = sub[["model", "set", "label", "n_seeds", "pnl_total_mean", "pnl_total_std",
                    "trades_per_day_mean", "pnl_spread_component_mean",
                    "pnl_fair_component_mean"]]
        print(view.to_string(index=False))
    print(f"\nTabela completa gravada em {out}")

    # dispersão entre seeds do melhor-de-validação no teste: quanto do
    # resultado é sorte de seed?
    sel = df[(df.model == "best_val") & (df.set == "test") & (df.label.str.startswith("best_val"))]
    if len(sel):
        print("\nP/L de teste do best_val por seed (pts):")
        print(sel.pivot_table(index="fold", columns="seed", values="pnl_total").to_string())


if __name__ == "__main__":
    main()
