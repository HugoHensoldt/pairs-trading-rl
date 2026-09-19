"""
Decide, a partir dos resultados já gravados, qual é a PRÓXIMA execução da
campanha (usado por slurm/submit_adaptive.sh, que roda no cluster sem depender
de um agente/PC ligado).

Ordem e regras (objetivas):
  1. hedged  seed 0  100M    -- versão mais promissora (recompensa = dinheiro real)
  2. spread  seed 0   30M    -- piloto curto (a exploração da recompensa aparece cedo)
  3. spreadx seed 0   30M    -- piloto curto
  4. hedged  seeds 1 e 2  100M   -- sem condição (medir variância)
  5. hedged  seed 3  100M    -- só se pelo menos 1 das seeds 0-2 tiver P/L REAL
                                hedgeado > 0 em validação E em teste (best_val)
  6. spread/spreadx seeds 1 e 2  100M -- só se o piloto (seed 0) NÃO for
                                "explorável" E tiver P/L real de validação > 0
"Explorável" = mediana de negócios/dia na validação > 1000 e mediana do P/L
real de validação < 0 ao longo dos checkpoints (o agente colhe oscilação de
cotação que não vira dinheiro). Máximo de 4 seeds por variante. Uma execução
que foi tentada e não completou NÃO é repetida (evita laço infinito).

Uso:
    python campaign_decide.py next   --runs-dir runs --attempted tentadas.txt
        # imprime "<variante> <seed> <timesteps>" ou "NONE"
    python campaign_decide.py report --runs-dir runs
        # imprime uma tabela em markdown com o que já completou
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

MAIN_STEPS = 100_000_000
PILOT_STEPS = 30_000_000
MAX_SEEDS = 4
EXPLORABLE_TRADES = 1000


def run_dir(runs_dir, variant, seed):
    return os.path.join(runs_dir, f"{variant}_s{seed}", "fold1")


def tag(variant, seed):
    return f"{variant}_s{seed}"


def _best_rows(summary):
    m = summary["label"].str.match(r"^best_val_(val|test)_fee")
    val = summary[m & (summary["set"] == "val")]
    test = summary[m & (summary["set"] == "test")]
    return val, test


def metrics(d):
    """Métricas de uma execução COMPLETA (senão None)."""
    st_path = os.path.join(d, "state.json")
    sm_path = os.path.join(d, "eval", "summary.csv")
    if not (os.path.exists(st_path) and os.path.exists(sm_path)):
        return None
    try:
        if not json.load(open(st_path)).get("done", False):
            return None
        summary = pd.read_csv(sm_path)
    except Exception:
        return None
    val, test = _best_rows(summary)
    if val.empty or test.empty:
        return None
    col = "real_pnl_hedged_brl"
    out = {"real_val": float(val[col].iloc[0]), "real_test": float(test[col].iloc[0]),
           "trades_val": float(val["trades_per_day"].iloc[0]),
           "trades_test": float(test["trades_per_day"].iloc[0]),
           "explorable": False, "med_trades_curve": float("nan")}
    vc_path = os.path.join(d, "val_curve.csv")
    if os.path.exists(vc_path):
        vc = pd.read_csv(vc_path)
        if "val_trades_per_day" in vc and "val_real_pnl_hedged_brl" in vc and len(vc):
            out["med_trades_curve"] = float(vc["val_trades_per_day"].median())
            out["explorable"] = bool(vc["val_trades_per_day"].median() > EXPLORABLE_TRADES
                                     and vc["val_real_pnl_hedged_brl"].median() < 0)
    return out


def _read_attempted(path):
    if path and os.path.exists(path):
        return {ln.strip() for ln in open(path) if ln.strip()}
    return set()


def next_run(runs_dir, attempted, max_seeds=MAX_SEEDS):
    """Próxima (variante, seed, timesteps) ou None."""
    def pending(variant, seed):
        return tag(variant, seed) not in attempted and metrics(run_dir(runs_dir, variant, seed)) is None

    fixed = [("hedged", 0, MAIN_STEPS), ("spread", 0, PILOT_STEPS), ("spreadx", 0, PILOT_STEPS),
             ("hedged", 1, MAIN_STEPS), ("hedged", 2, MAIN_STEPS)]
    for variant, seed, steps in fixed:
        if pending(variant, seed):
            return variant, seed, steps

    # hedged, seed 3: só se alguma das 3 primeiras foi positiva em validação E teste
    hed = [metrics(run_dir(runs_dir, "hedged", s)) for s in range(3)]
    hed_ok = [m for m in hed if m is not None]
    if (max_seeds > 3 and len(hed_ok) >= 1
            and any(m["real_val"] > 0 and m["real_test"] > 0 for m in hed_ok)
            and pending("hedged", 3)):
        return "hedged", 3, MAIN_STEPS

    # variantes de spread: estende só se o piloto não for explorável e for positivo
    for variant in ("spread", "spreadx"):
        pilot = metrics(run_dir(runs_dir, variant, 0))
        if pilot is not None and not pilot["explorable"] and pilot["real_val"] > 0:
            for seed in (1, 2):
                if seed < max_seeds and pending(variant, seed):
                    return variant, seed, MAIN_STEPS
    return None


def report(runs_dir):
    rows = []
    for variant in ("hedged", "spread", "spreadx"):
        for seed in range(MAX_SEEDS):
            m = metrics(run_dir(runs_dir, variant, seed))
            if m is None:
                continue
            rows.append({"execução": tag(variant, seed),
                         "P/L real val (R$)": f"{m['real_val']:.1f}",
                         "P/L real teste (R$)": f"{m['real_test']:.1f}",
                         "negócios/dia val": f"{m['trades_val']:.1f}",
                         "mediana negócios/dia (curva)": f"{m['med_trades_curve']:.1f}",
                         "explorável": "sim" if m["explorable"] else "não"})
    if not rows:
        return "(nenhuma execução completa ainda)"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["next", "report"])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--attempted", default=None)
    ap.add_argument("--max-seeds", type=int, default=MAX_SEEDS)
    a = ap.parse_args()
    if a.cmd == "next":
        r = next_run(a.runs_dir, _read_attempted(a.attempted), a.max_seeds)
        print("NONE" if r is None else f"{r[0]} {r[1]} {r[2]}")
    else:
        print(report(a.runs_dir))


if __name__ == "__main__":
    main()
