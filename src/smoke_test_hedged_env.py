"""
Teste do HedgedPairEnv (recompensa do par WIN + BOVA11 hedgeado).

Parte 1 (sintética, valores calculados à mão):
  - N_BOVA = P_WIN / (5 * P_BOVA) calculado na abertura e FIXO até fechar;
  - custos só ao abrir/fechar: 0,25 + 0,000230 * V_BOVA (R$) em cada lado;
  - long WIN = short BOVA11 e short WIN = long BOVA11 (P/L com sinais certos);
  - virar de sentido = fechar + abrir (dois custos);
  - fechamento forçado no fim do pregão;
  - recompensa devolvida ao agente = R$ * reward_scale, mas o P/L acumulado
    (total_reward) é em R$.
Parte 2 (dia real): invariantes de contabilidade e qualidade do hedge.

Uso:  python src/smoke_test_hedged_env.py
"""

import numpy as np
import pandas as pd

from rl_trading_pipeline import (
    HedgedPairEnv, process_day_cached, build_state_features, fit_feature_scaler,
    apply_scaler, HEDGE_FACTOR, WIN_COST_PER_SIDE_BRL, BOVA_COST_PCT_PER_SIDE,
)

TOL = 1e-9


def make_env(win_mid, bova_mid, **kw):
    n = len(win_mid)
    df = pd.DataFrame({
        "bid": win_mid, "ask": win_mid, "bbid": bova_mid, "bask": bova_mid,
        "Wajusto": win_mid, "Wbjusto": win_mid,
    }).astype(float)
    kw.setdefault("reward_scale", 1.0)
    return HedgedPairEnv(df, np.zeros((n, 2)), **kw)


def run(env, actions):
    env.reset()
    rewards = []
    for a in actions:
        _, r, term, _, _ = env.step(a)
        rewards.append(r)
        if term:
            break
    return np.array(rewards)


def cost(v):
    return WIN_COST_PER_SIDE_BRL + BOVA_COST_PCT_PER_SIDE * v


def test_synthetic():
    # WIN e BOVA com o MESMO retorno relativo por tick => par perfeitamente
    # hedgeado (N = 200 ações; ΔWIN=10 pts -> R$2,00; ΔBOVA=0,01 -> R$2,00)
    W = np.array([120000, 120010, 120020, 120010, 120000, 120000, 120000, 120000, 120000, 120000], float)
    B = np.array([120.00, 120.01, 120.02, 120.01, 120.00, 120.00, 120.00, 120.00, 120.00, 120.00])
    n_bova = 120000 / (HEDGE_FACTOR * 120.0)
    assert abs(n_bova - 200.0) < TOL

    # (a) long: abre em t=0, mantém, fecha em t=3 (ação 0)
    env = make_env(W, B)
    r = run(env, [1, 1, 1, 0])
    open_cost = cost(200 * 120.00)
    close_cost = cost(200 * 120.01)
    assert abs(r[0] + open_cost) < TOL, (r[0], open_cost)          # só custo de abertura
    assert abs(r[1]) < 1e-9 and abs(r[2]) < 1e-9                    # hedge neutro, sem custo ao manter
    assert abs(r[3] + close_cost) < 1e-9, (r[3], close_cost)
    assert abs(env.gross_mtm) < 1e-9                                # par hedgeado: MtM ~ 0
    assert env.n_trades_closed == 1
    assert abs(env.trade_pnls[0] + open_cost + close_cost) < 1e-9
    assert abs(env.total_reward - r.sum()) < 1e-9

    # (b) sem hedge efetivo (BOVA parado): long WIN ganha só a perna WIN
    B_flat = np.full(10, 120.0)
    env = make_env(W, B_flat)
    r = run(env, [1, 1, 1, 0])
    assert abs(env.mtm_win_leg - 0.2 * 10) < 1e-9 and abs(env.mtm_bova_leg) < 1e-9
    # short WIN: sinais invertidos
    env = make_env(W, B_flat)
    run(env, [2, 2, 2, 0])
    assert abs(env.mtm_win_leg + 0.2 * 10) < 1e-9

    # (c) long WIN = SHORT BOVA: BOVA sobe 10 -> perna BOVA perde 200*10
    B_jump = np.array([120, 130, 130, 130, 130, 130, 130, 130, 130, 130], float)
    env = make_env(np.full(10, 120000.0), B_jump)
    run(env, [1, 1, 0])
    assert abs(env.mtm_bova_leg + 200 * 10) < 1e-9, env.mtm_bova_leg   # N FIXO = 200 (não recalcula em t=1)
    env = make_env(np.full(10, 120000.0), B_jump)
    run(env, [2, 2, 0])                                                # short WIN = LONG BOVA
    assert abs(env.mtm_bova_leg - 200 * 10) < 1e-9

    # (d) virar de sentido = fechar + abrir (dois custos), sem custo ao manter
    env = make_env(W, B)
    r = run(env, [1, 1, 2, 2])
    assert abs(r[2] + cost(200 * 120.02) + cost(200 * 120.02)) < 1e-9, r[2]  # fecha long + abre short em t=2
    assert env.n_trades_closed == 1 and env.position == -1

    # (e) fechamento forçado no fim: posição aberta até o fim é fechada com custo
    env = make_env(W, B)
    r = run(env, [1] * 20)
    assert env.position == 0 and env.n_trades_closed == 1
    assert abs(env.trade_pnls[0] - (env.gross_mtm - cost(24000) - cost(200 * 120.0))) < 1e-9

    # (f) escala da recompensa: agente vê R$ * scale; total_reward fica em R$
    env = make_env(W, B, reward_scale=5.0)
    r = run(env, [1, 0])
    assert abs(r[0] + 5.0 * cost(24000)) < 1e-9
    assert abs(env.total_reward + cost(24000) + cost(200 * 120.01)) < 1e-9

    # (g) multiplicador de custo 0 => sem custo algum
    env = make_env(W, B, transaction_fee=0.0)
    r = run(env, [1, 1, 1, 0])
    assert abs(r.sum()) < 1e-9

    # (h) observação: [z1, z2, posição, P/L do par / custo nominal ida-e-volta]
    env = make_env(W, B_flat)
    env.reset()
    obs, *_ = env.step(1)
    assert obs.shape == (4,) and obs[2] == 1.0
    obs, *_ = env.step(1)                                              # MtM WIN = 0,2*10 = 2,0
    rt = 2 * cost(24000)
    assert abs(obs[3] - 2.0 / rt) < 1e-6, (obs[3], 2.0 / rt)
    print("parte 1 (sintética): OK")


def test_real_day():
    df = process_day_cached(1)
    raw = build_state_features(df)
    mean, std = fit_feature_scaler([raw])
    feat = apply_scaler(raw, mean, std)

    rng = np.random.default_rng(0)
    env = HedgedPairEnv(df, feat, reward_scale=1.0)
    obs, _ = env.reset()
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(int(rng.integers(0, 3)))
        total += r
        done = term or trunc
        if not done:
            assert np.all(np.isfinite(obs)) and obs[2] in (-1.0, 0.0, 1.0)
    assert abs(total - env.total_reward) < 1e-6
    assert abs(env.mtm_win_leg + env.mtm_bova_leg - env.gross_mtm) < 1e-6
    assert env.gross_mtm - env.total_reward >= -1e-6                   # custos >= 0
    assert abs(sum(env.trade_pnls) - env.total_reward) < 1e-6          # termina flat
    print(f"dia real, política aleatória: {env.n_trades_closed} negócios, "
          f"bruto {env.gross_mtm:.1f} R$, líquido {env.total_reward:.1f} R$")

    # qualidade do hedge: fica comprado o dia todo e compara a variação da
    # perna WIN com a soma das pernas
    env = HedgedPairEnv(df, feat, reward_scale=1.0)
    env.reset()
    wins, pair = [], []
    prev_w, prev_b = 0.0, 0.0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(1)
        wins.append(env.mtm_win_leg - prev_w)
        pair.append(env.mtm_win_leg + env.mtm_bova_leg - prev_w - prev_b)
        prev_w, prev_b = env.mtm_win_leg, env.mtm_bova_leg
        done = term or trunc
    wins, pair = np.array(wins), np.array(pair)
    n_b = (df["ask"].iloc[0] + df["bid"].iloc[0]) / (2 * HEDGE_FACTOR * (df["bask"].iloc[0] + df["bbid"].iloc[0]) / 2)
    print(f"dia real, long o dia todo: N_BOVA(abertura) = {n_b:.1f} ações; "
          f"desvio por tick: perna WIN {wins.std():.4f} R$ vs par {pair.std():.4f} R$ "
          f"(o hedge remove {1 - pair.std() / max(wins.std(), 1e-12):.0%} da variância por tick em desvio)")
    print("parte 2 (dia real): OK")


if __name__ == "__main__":
    test_synthetic()
    test_real_day()
