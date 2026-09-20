"""
Baselines de par hedgeado "segurar o dia todo": em cada pregão, fica sempre
COMPRADO no par (long WIN / short BOVA11) ou sempre VENDIDO no par, com a
contabilidade do HedgedPairEnv (custos da especificação; abre no primeiro tick
e fecha no fim do pregão). É a referência "sem habilidade": o agente precisa
superar isso, além do flat (0).

Uso (de src/):
    python hold_baselines_hedged.py <primeiro_pregao> <ultimo_pregao> <saida.csv>
"""
import sys

import numpy as np
import pandas as pd

from rl_trading_pipeline import HedgedPairEnv, process_day_cached


def run_hold(order, action, spread_cost=False):
    df = process_day_cached(order)
    env = HedgedPairEnv(df, np.zeros((len(df), 2)), reward_scale=1.0,
                        include_spread_cost=spread_cost)
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return {"order": order, "action": {1: "long_par", 2: "short_par"}[action],
            "spread_cost": spread_cost, "pnl_brl": env.total_reward,
            "gross_brl": env.gross_mtm, "win_leg": env.mtm_win_leg, "bova_leg": env.mtm_bova_leg}


if __name__ == "__main__":
    first, last, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rows = [run_hold(o, a, sc) for o in range(first, last + 1) for a in (1, 2) for sc in (False, True)]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} linhas -> {out}")
