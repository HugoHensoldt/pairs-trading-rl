"""
Referências para o teste com dados SINTÉTICOS (ver synthetic_pair.py).

Como o spread verdadeiro X_t é conhecido (coluna `true_x`), dá para calcular:
  1) a melhor regra CAUSAL de limiar sobre o spread verdadeiro (entra em -a se
     X < -a, sai quando X cruza 0; o limiar a é escolhido nos dias de treino) e
     seu P/L nos dias de avaliação: o que um arbitrador com informação perfeita
     do modelo consegue SEM ver o futuro;
  2) o teto do oráculo (programação dinâmica com previsão perfeita);
  3) opcionalmente, o quanto do P/L da regra ótima o agente treinado captura nos
     mesmos pregões (lê eval/*_days.csv da pasta da execução).

Contabilidade: HedgedPairEnv (par hedgeado, custos da especificação, preço mid),
a mesma dos treinos.

Uso (de src/):
    SYNTH_KIND=coint python synthetic_benchmark.py --train-days 30 --eval-days 30
    SYNTH_KIND=null  python synthetic_benchmark.py                # controle negativo
    python synthetic_benchmark.py --run-dir runs/<tag>/fold1      # + captura do agente
"""

import argparse
import os

import numpy as np
import pandas as pd

from synthetic_pair import SYNTH_BASE, synth_config

MULTS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


def run_rule_day(order, a_pts, exit_pts=0.0, spread_cost=False):
    """P/L (R$) e nº de negócios de UM pregão com a regra de limiar sobre o spread verdadeiro."""
    from rl_trading_pipeline import HedgedPairEnv, process_day_cached
    df = process_day_cached(order)
    x = df["true_x"].to_numpy(dtype=float)
    n = len(df)
    env = HedgedPairEnv(df, np.zeros((n, 2)), reward_scale=1.0, include_spread_cost=spread_cost)
    env.reset()
    done = False
    while not done:
        t, pos, xt = env.t, env.position, x[env.t]
        if pos == 0:
            action = 1 if xt < -a_pts else (2 if xt > a_pts else 0)
        elif pos == 1:
            action = 1 if xt < -exit_pts else 0
        else:
            action = 2 if xt > exit_pts else 0
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.total_reward, env.n_trades_closed


def oracle_day_pnl(order, H=2000):
    from opportunity_analysis import oracle_day
    from rl_trading_pipeline import process_day_cached
    df = process_day_cached(order)
    a = {c: df[c].to_numpy(dtype=float) for c in ("bid", "ask", "bbid", "bask")}
    a["W"] = 0.5 * (a["bid"] + a["ask"])
    a["B"] = 0.5 * (a["bbid"] + a["bask"])
    a["hour"] = pd.to_datetime(df["datahora"]).dt.hour.to_numpy()
    a["order"] = order
    return oracle_day(a, H, False)[-1]


def best_threshold(train_orders, spread_std):
    rows = []
    for m in MULTS:
        pnl = [run_rule_day(o, m * spread_std)[0] for o in train_orders]
        rows.append((m, float(np.mean(pnl))))
    best = max(rows, key=lambda r: r[1])
    return best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=30)
    ap.add_argument("--eval-days", type=int, default=30)
    ap.add_argument("--eval-start", type=int, default=None, help="índice do 1º dia de avaliação (default: logo após o treino)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--oracle-days", type=int, default=10, help="dias usados para o teto do oráculo (lento)")
    a = ap.parse_args()

    cfg = synth_config()
    tr = [SYNTH_BASE + i for i in range(a.train_days)]
    e0 = a.train_days if a.eval_start is None else a.eval_start
    ev = [SYNTH_BASE + e0 + i for i in range(a.eval_days)]
    print(f"Sintético: kind={cfg['kind']} spread_std={cfg['spread_std']} pts, meia-vida={cfg['half_life']} ticks, "
          f"deriva={cfg['drift']} pts/pregão, rollover a cada {cfg['roll_every']}")
    (m_best, pnl_best), rows = best_threshold(tr, cfg["spread_std"])
    print("Limiar (múltiplos do desvio do spread) -> P/L médio/dia no TREINO (R$): "
          + "  ".join(f"{m:.1f}σ: {p:+.1f}" for m, p in rows))
    print(f"Limiar escolhido no treino: {m_best:.1f}σ = {m_best * cfg['spread_std']:.0f} pts")
    res = [run_rule_day(o, m_best * cfg["spread_std"]) for o in ev]
    rule = np.array([r[0] for r in res])
    print(f"REGRA ÓTIMA CAUSAL (spread verdadeiro) na avaliação: {rule.mean():+.1f} R$/dia "
          f"({rule.sum():+.0f} R$ em {len(ev)} dias), {np.mean([r[1] for r in res]):.1f} negócios/dia, "
          f"dias positivos {int((rule > 0).sum())}/{len(rule)}")
    res_sc = [run_rule_day(o, m_best * cfg["spread_std"], spread_cost=True)[0] for o in ev]
    print(f"  mesma regra com meio-spread bid/ask nas duas pernas: {np.mean(res_sc):+.1f} R$/dia")
    orc = [oracle_day_pnl(o) for o in ev[:a.oracle_days]]
    print(f"TETO DO ORÁCULO (previsão perfeita, {len(orc)} dias): {np.mean(orc):+.1f} R$/dia")

    if a.run_dir:
        for cj in ("val", "test"):
            p = os.path.join(a.run_dir, "eval", f"best_val_{cj}_fee1.0_days.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            orders = d["order"].tolist()
            rl = np.array([run_rule_day(o, m_best * cfg["spread_std"])[0] for o in orders])
            ag = d["real_pnl_brl"].to_numpy()
            print(f"AGENTE ({cj}, {len(d)} dias): {ag.mean():+.1f} R$/dia | regra ótima nos mesmos dias: {rl.mean():+.1f} R$/dia "
                  f"| fração capturada: {ag.mean() / rl.mean():.0%}" if rl.mean() != 0 else f"AGENTE ({cj}): {ag.mean():+.1f} R$/dia")


if __name__ == "__main__":
    main()
