"""
Teste do SpreadRewardEnv (recompensa = convergência do spread, só custo do WIN).

Parte 1 (sintética, valores calculados à mão):
  - long (q=+1) usa s_buy = ask_WIN - Wbjusto: recompensa |s(t-1)| - |s(t)|;
  - short (q=-1) usa s_sell = bid_WIN - Wajusto;
  - flat = 0; custo do WIN = R$ 0,25 / 0,20 = 1,25 pts por transação
    (2,5 pts ida e volta), sem custo de BOVA11 e sem meio-spread;
  - preço subindo SEM convergir do spread não gera recompensa;
  - virar de sentido = fechar + abrir (dois custos); fechamento forçado no fim;
  - o P/L REAL hedgeado (diagnóstico) coincide com o HedgedPairEnv para as
    mesmas ações.
Parte 2 (dia real): invariantes de contabilidade.

Uso:  python src/smoke_test_spread_env.py
"""

import numpy as np
import pandas as pd

from rl_trading_pipeline import (
    SpreadRewardEnv, SpreadRewardEnvX, HedgedPairEnv, process_day_cached, build_state_features,
    fit_feature_scaler, apply_scaler,
)

TOL = 1e-9
WIN_COST_PTS = 0.25 / 0.20        # 1,25 pts por transação


def make_df(win_ask, win_bid, wbj, waj, bova=None):
    n = len(win_ask)
    b = np.full(n, 120.0) if bova is None else np.asarray(bova, float)
    return pd.DataFrame({
        "bid": win_bid, "ask": win_ask, "bbid": b, "bask": b,
        "Wbjusto": wbj, "Wajusto": waj,
    }).astype(float)


def run(env, actions):
    env.reset()
    rewards = []
    for a in actions:
        _, r, term, _, _ = env.step(a)
        rewards.append(r)
        if term:
            break
    return np.array(rewards)


def test_synthetic():
    n = 12
    # sem meio-spread: ask == bid == mid do WIN
    W = np.array([120000, 120000, 120010, 120020, 120020, 120000, 120000, 120000, 120000, 120000, 120000, 120000], float)
    # justo fixo em 120030: s_buy = s_sell = W - 120030  (negativo: WIN "barato")
    fair = np.full(n, 120030.0)
    df = make_df(W, W, fair, fair)
    s = W - fair                                                    # -30,-30,-20,-10,-10,-30,...

    # (a) long: abre em t=0; em t=1 (s: -30 -> -30) recompensa 0; t=2 (-30 -> -20): +10; t=3 (-20 -> -10): +10
    env = SpreadRewardEnv(df, np.zeros((n, 2)), reward_scale=1.0)
    r = run(env, [1, 1, 1, 1, 0])
    assert abs(r[0] + WIN_COST_PTS) < TOL, r[0]                     # só custo de abertura
    assert abs(r[1]) < TOL
    assert abs(r[2] - 10) < TOL and abs(r[3] - 10) < TOL
    # t=4: MtM (s: -10 -> -10) = 0, depois fecha (ação 0): custo 1,25
    assert abs(r[4] + WIN_COST_PTS) < TOL, r[4]
    assert env.n_trades_closed == 1
    assert abs(env.trade_pnls[0] - (20 - 2 * WIN_COST_PTS)) < TOL, env.trade_pnls
    assert abs(env.gross_mtm - 20) < TOL
    assert abs(env.total_reward - (20 - 2 * WIN_COST_PTS)) < TOL

    # (b) o spread DIVERGE (|s| aumenta) => recompensa negativa: WIN cai de 120020 para 120000 em t=5
    env = SpreadRewardEnv(df, np.zeros((n, 2)), reward_scale=1.0)
    r = run(env, [1, 1, 1, 1, 1, 1])                                # long até t=5
    assert abs(r[5] - (abs(s[4]) - abs(s[5]))) < TOL and r[5] < 0   # |-10| - |-30| = -20

    # (c) short usa s_sell: com s_sell > 0 (WIN caro) e WIN caindo, |s| encolhe => positivo
    fair2 = np.full(n, 119990.0)                                    # s_sell = W - 119990 > 0
    df2 = make_df(W, W, fair2, fair2)
    env = SpreadRewardEnv(df2, np.zeros((n, 2)), reward_scale=1.0)
    r = run(env, [2, 2, 2, 2, 2, 2])
    s2 = W - fair2                                                  # 10,10,20,30,30,10
    assert abs(r[5] - (abs(s2[4]) - abs(s2[5]))) < TOL and r[5] > 0  # |30| - |10| = +20

    # (d) flat: nunca há recompensa nem custo
    env = SpreadRewardEnv(df, np.zeros((n, 2)), reward_scale=1.0)
    assert abs(run(env, [0] * 8).sum()) < TOL

    # (e) virar de sentido = fechar + abrir (dois custos de WIN = 2,5 pts)
    env = SpreadRewardEnv(df, np.zeros((n, 2)), reward_scale=1.0)
    r = run(env, [1, 1, 2, 2])
    assert abs(r[2] - ((abs(s[1]) - abs(s[2])) - 2 * WIN_COST_PTS)) < TOL, r[2]
    assert env.n_trades_closed == 1 and env.position == -1

    # (f) fechamento forçado no fim do pregão
    env = SpreadRewardEnv(df, np.zeros((n, 2)), reward_scale=1.0)
    run(env, [1] * 30)
    assert env.position == 0 and env.n_trades_closed == 1

    # (g) multiplicador de custo 0 => sem custo; reward_scale só escala o que o agente vê
    env = SpreadRewardEnv(df, np.zeros((n, 2)), transaction_fee=0.0, reward_scale=1.0)
    r = run(env, [1, 1, 1, 1, 0])
    assert abs(r.sum() - 20) < TOL
    env = SpreadRewardEnv(df, np.zeros((n, 2)), reward_scale=3.0)
    r = run(env, [1, 0])
    assert abs(r[0] + 3 * WIN_COST_PTS) < TOL and abs(env.total_reward + 2 * WIN_COST_PTS) < TOL

    # (h) BOVA11 NÃO entra na recompensa: mudar o preço do BOVA não muda o reward
    b_move = np.linspace(120, 140, n)
    e1 = SpreadRewardEnv(make_df(W, W, fair, fair), np.zeros((n, 2)), reward_scale=1.0)
    e2 = SpreadRewardEnv(make_df(W, W, fair, fair, bova=b_move), np.zeros((n, 2)), reward_scale=1.0)
    acts = [1, 1, 1, 1, 0]
    assert np.allclose(run(e1, acts), run(e2, acts))
    assert abs(e1.real_pnl_brl - e2.real_pnl_brl) > 1.0            # mas o P/L REAL diagnóstico muda

    # (i) observação: [z1, z2, posição, convergência acumulada / 2,5 pts]
    env = SpreadRewardEnv(df, np.zeros((n, 2)), reward_scale=1.0)
    env.reset()
    env.step(1)
    obs, *_ = env.step(1)
    obs, *_ = env.step(1)                                           # convergência acumulada = +10
    assert obs.shape == (4,) and obs[2] == 1.0
    assert abs(obs[3] - 10 / 2.5) < 1e-6, obs[3]
    # (j) exec_cost / SpreadRewardEnvX: soma o meio-spread do WIN (aqui 4 pts => 2 pts) por transação
    ask = W + 2.0
    bid = W - 2.0
    dfx = make_df(ask, bid, fair, fair)
    for env in (SpreadRewardEnv(dfx, np.zeros((n, 2)), reward_scale=1.0, exec_cost=True),
                SpreadRewardEnvX(dfx, np.zeros((n, 2)), reward_scale=1.0)):
        r = run(env, [1, 1, 1, 1, 0])
        assert abs(r[0] + WIN_COST_PTS + 2.0) < TOL, r[0]          # abre: taxa + meio-spread
        assert abs(r[4] + WIN_COST_PTS + 2.0) < TOL, r[4]          # fecha: taxa + meio-spread
        assert abs(env.trade_pnls[0] - (env.gross_mtm - 2 * (WIN_COST_PTS + 2.0))) < TOL
    base = SpreadRewardEnv(dfx, np.zeros((n, 2)), reward_scale=1.0)   # padrão: sem meio-spread
    r = run(base, [1, 1, 1, 1, 0])
    assert abs(r[0] + WIN_COST_PTS) < TOL
    print("parte 1 (sintética): OK")


def test_real_diag_matches_hedged(order=1):
    """O P/L real diagnóstico do SpreadRewardEnv == recompensa do HedgedPairEnv
    (mesmas ações, mesmos custos) em um dia real."""
    df = process_day_cached(order)
    raw = build_state_features(df)
    mean, std = fit_feature_scaler([raw])
    feat = apply_scaler(raw, mean, std)
    rng = np.random.default_rng(1)
    acts = rng.integers(0, 3, size=len(df) - 1)
    # política "persistente": troca de ação só de vez em quando, senão todos os custos dominam
    acts = np.repeat(rng.integers(0, 3, size=len(acts) // 400 + 1), 400)[: len(df) - 1]

    h = HedgedPairEnv(df, feat, reward_scale=1.0)
    s = SpreadRewardEnv(df, feat, reward_scale=1.0)
    run(h, acts)
    run(s, acts)
    assert h.n_trades_closed == s.n_trades_closed and h.n_trades_closed > 3
    assert abs(h.total_reward - s.real_pnl_brl) < 1e-6, (h.total_reward, s.real_pnl_brl)
    assert abs(h.mtm_win_leg - s.mtm_win_leg) < 1e-6 and abs(h.mtm_bova_leg - s.mtm_bova_leg) < 1e-6
    assert abs(sum(s.trade_real_pnls) - s.real_pnl_brl) < 1e-6
    # invariantes da recompensa de treino
    assert abs(s.total_reward - (s.gross_mtm - s.n_trades_closed * 2 * WIN_COST_PTS)) < 1e-6
    assert abs(sum(s.trade_pnls) - s.total_reward) < 1e-6
    print(f"dia real ({s.n_trades_closed} negócios): convergência bruta {s.gross_mtm:.1f} pts, "
          f"custo WIN {s.n_trades_closed * 2 * WIN_COST_PTS:.1f} pts, recompensa {s.total_reward:.1f} pts | "
          f"P/L real hedgeado {s.real_pnl_brl:.2f} R$ (= HedgedPairEnv {h.total_reward:.2f})")
    print("parte 2 (dia real, coerência com HedgedPairEnv): OK")


if __name__ == "__main__":
    test_synthetic()
    test_real_diag_matches_hedged()
