"""
Baselines direcionais "comprar e segurar" / "vender e segurar" por pregão.

Para cada pregão, o agente entra no primeiro tick (long ou short, pagando
meio-spread + fee) e só sai no fim do dia -- P/L exato pelo mesmo
ArbitrageTradingEnv usado no treino/avaliação. Serve de referência para
separar "o modelo arbitra" de "o modelo tem viés direcional num período em
que o mercado subiu/caiu" (ver docs/relatorio_resultados_v4.md).

Uso (a partir de src/, com TICK_DATA_DIR apontando pros dados):
    python hold_baselines.py 1 230 ../sdumont_backup_v4/analysis/hold_baselines.csv
"""

import os
import sys

import numpy as np
import pandas as pd

from rl_trading_pipeline import ArbitrageTradingEnv, process_day_cached

FEE = float(os.environ.get("TRANSACTION_FEE", "2.5"))


def _hold_pnl(df, action, fee):
    env = ArbitrageTradingEnv(df, np.zeros((len(df), 2), dtype=np.float32), transaction_fee=fee)
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return float(env.total_reward)


def _one_day(order):
    df = process_day_cached(order)
    mid = ((df['bid'] + df['ask']) / 2).to_numpy()
    fair = ((df['Wajusto'] + df['Wbjusto']) / 2).to_numpy()
    return {
        "order": order,
        "date": str(pd.to_datetime(df['datahora']).iloc[0].date()),
        "n_ticks": len(df),
        "win_move": float(mid[-1] - mid[0]),
        "fair_move": float(fair[-1] - fair[0]),
        "misp_open": float(mid[0] - fair[0]),
        "misp_close": float(mid[-1] - fair[-1]),
        "long_hold_pnl": _hold_pnl(df, 1, FEE),
        "short_hold_pnl": _hold_pnl(df, 2, FEE),
    }


if __name__ == "__main__":
    first, last, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    orders = list(range(first, last + 1))
    n_workers = int(os.environ.get("EVAL_WORKERS", "4"))
    if n_workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n_workers) as pool:
            rows = pool.map(_one_day, orders, chunksize=1)
    else:
        rows = [_one_day(o) for o in orders]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} pregões -> {out}")
