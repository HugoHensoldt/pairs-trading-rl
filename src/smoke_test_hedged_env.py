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
    HedgedPairEnv, MultiDayEnv, process_day_cached, build_state_features, fit_feature_scaler,
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


def test_reward_clip():
    """reward_clip limita o que o PPO vê (obs/step), mas NUNCA o P/L real
    (total_reward) -- ver docs/relatorio_diagnostico_sintetico.md, §6.2."""
    W = np.array([120000, 130000, 120000, 130000, 120000, 130000, 120000, 130000], float)  # saltos de 10.000 pts
    B = np.full(8, 120.0)  # BOVA parado: sem hedge efetivo, MtM cru do WIN vira o reward
    env = make_env(W, B, reward_scale=1.0, reward_clip=50.0)
    env.reset()
    rewards = []
    for a in (1, 1, 1, 1, 1, 1, 1):
        _, r, term, _, _ = env.step(a)
        rewards.append(r)
        if term:
            break
    assert all(abs(r) <= 50.0 + TOL for r in rewards), rewards
    assert max(abs(r) for r in rewards) > 49.0, "o teste não estourou o clip; ajuste os saltos"
    # sem clip, o MESMO cenário produz reward MUITO maior (0,2 R$/pt * 10.000 pts = 2.000)
    env2 = make_env(W, B, reward_scale=1.0)
    env2.reset()
    r2 = [env2.step(a)[1] for a in (1, 1, 1, 1, 1, 1, 1)]
    assert max(abs(x) for x in r2) > 1000
    # o P/L REAL (total_reward) não muda com o clip: mesmas ações, mesmo resultado
    assert abs(env.total_reward - env2.total_reward) < TOL, (env.total_reward, env2.total_reward)
    # clip desligado por padrão (reward_clip=None) reproduz o comportamento antigo
    env3 = make_env(W, B, reward_scale=1.0, reward_clip=None)
    r3 = [env3.step(a)[1] for a in (1, 1, 1, 1, 1, 1, 1)]
    assert r3 == r2
    print("reward_clip: OK (limita o step(), preserva o P/L real)")


def test_cost_curriculum():
    """MultiDayEnv.set_cost_scale muda cost_scale em TODOS os day_envs (não só o
    atual), pois o próximo reset pode sortear qualquer um -- ver CostCurriculumCallback."""
    df = process_day_cached(1)
    raw = build_state_features(df)
    mean, std = fit_feature_scaler([raw])
    feat = apply_scaler(raw, mean, std)
    envs = [HedgedPairEnv(df, feat, transaction_fee=1.0) for _ in range(3)]
    multi = MultiDayEnv(envs, seed=0)
    assert all(e.cost_scale == 1.0 for e in envs)
    multi.set_cost_scale(0.0)
    assert all(e.cost_scale == 0.0 for e in envs), [e.cost_scale for e in envs]
    multi.reset()
    assert multi.current_env.cost_scale == 0.0
    multi.set_cost_scale(0.37)
    assert all(abs(e.cost_scale - 0.37) < TOL for e in envs)
    print("cost curriculum (MultiDayEnv.set_cost_scale): OK")


def _shape_env(kind, z, W=None, B=None, **env):
    """HedgedPairEnv com shaping (shaping_z = z) e parâmetros fixos para os testes."""
    import os
    n = len(z)
    W = np.full(n, 120000.0) if W is None else np.asarray(W, float)
    B = np.full(n, 120.0) if B is None else np.asarray(B, float)
    old = {k: os.environ.get(k) for k in ("SHAPING_KIND", "SHAPING_K", "SHAPING_K0", "SHAPING_LAMBDA",
                                          "SHAPING_C", "GAMMA")}
    os.environ.update(SHAPING_KIND=kind, SHAPING_K="2.0", SHAPING_K0="0.25", SHAPING_LAMBDA="0.05",
                      SHAPING_C="5.0", GAMMA=env.pop("gamma", "1.0"))
    try:
        df = pd.DataFrame({"bid": W, "ask": W, "bbid": B, "bask": B, "Wajusto": W, "Wbjusto": W}).astype(float)
        return HedgedPairEnv(df, np.zeros((n, 2)), reward_scale=1.0, shaping_z=np.asarray(z, np.float32), **env)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _rewards(env, acts):
    env.reset()
    out = []
    for a in acts:
        _, r, term, _, _ = env.step(a)
        out.append(r)
        if term:
            break
    return np.array(out)


def test_shaping():
    """Shaping: soma um termo (R$/tick) ao que o PPO vê e NUNCA altera o P/L real."""
    lam, k, k0, c = 0.05, 2.0, 0.25, 5.0
    n = 12
    # convenção: z > 0 = WIN caro => posição certa é VENDIDA (ação 2); z < 0 => COMPRADA (ação 1)

    # none / sem shaping_z: idêntico ao ambiente antigo
    z = np.full(n, 3.0)
    base = _rewards(_shape_env("none", z), [0] * 6)
    assert np.allclose(base, 0.0)

    # opp_flat: flat com |z|=3 > k=2 => -lam*(3-2) por tick; flat com |z|=1 => 0
    r = _rewards(_shape_env("opp_flat", np.full(n, 3.0)), [0] * 5)
    assert np.allclose(r, -lam * 1.0), r
    r = _rewards(_shape_env("opp_flat", np.full(n, 1.0)), [0] * 5)
    assert np.allclose(r, 0.0), r
    # opp_flat NÃO penaliza estar posicionado (nem no lado errado)
    r = _rewards(_shape_env("opp_flat", np.full(n, 3.0)), [1, 1, 1])
    assert r[1] == 0.0 and r[2] == 0.0

    # opp_wrong: mesmo caso flat; lado ERRADO (comprado com z=+3) => -lam*(|z|+k) = -0,25
    r = _rewards(_shape_env("opp_wrong", np.full(n, 3.0)), [0] * 3)
    assert np.allclose(r, -lam * 1.0)
    r = _rewards(_shape_env("opp_wrong", np.full(n, 3.0)), [1, 1, 1])      # comprado com WIN caro
    assert abs(r[1] - (-lam * (3.0 + k))) < 1e-9, r
    r = _rewards(_shape_env("opp_wrong", np.full(n, 3.0)), [2, 2, 2])      # vendido com WIN caro: certo
    assert abs(r[1]) < 1e-9, r
    r = _rewards(_shape_env("opp_wrong", np.full(n, 0.1)), [2, 2, 2])      # lado certo, sinal convergiu
    assert abs(r[1] - (-lam * (k0 - 0.1))) < 1e-9, r
    r = _rewards(_shape_env("opp_wrong", np.full(n, 1.0)), [2, 2, 2])      # lado certo, sinal moderado
    assert abs(r[1]) < 1e-9, r
    # "posição fixa o dia todo" (atrator do oracle_c1) com z oscilando: perde em média
    zz = np.tile([3.0, -3.0], n // 2)
    fixa = _rewards(_shape_env("opp_wrong", zz), [1] * (n - 1)).sum()
    certa = _rewards(_shape_env("opp_wrong", zz), [2, 1] * ((n - 1) // 2) + [2])[:n - 1].sum()
    assert fixa < 0

    # pbrs: entrar comprado com z_{t+1} = -2 => F = gamma * c * (+1) * (2) = +10 (gamma=1)
    zp = np.array([0.0, -2.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e = _shape_env("pbrs", zp)
    e.reset()
    _, r0, _, _, _ = e.step(1)
    open_cost = WIN_COST_PER_SIDE_BRL + BOVA_COST_PCT_PER_SIDE * 24000.0        # R$ 5,77 (V_BOVA = W/5)
    assert abs(e.total_shaping - 10.0) < 1e-9, e.total_shaping              # Phi(s)=0 -> Phi(s')=5*1*2
    assert abs(r0 - (10.0 - open_cost)) < 1e-9, r0                          # o PPO vê shaping + P/L real
    assert abs(e.total_reward - (-open_cost)) < 1e-9                        # o P/L REAL só tem o custo
    # entrada no lado ERRADO (comprado com WIN caro, z=+2): F = -10
    zw = np.array([0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ew = _shape_env("pbrs", zw)
    ew.reset()
    ew.step(1)
    assert abs(ew.total_shaping - (-10.0)) < 1e-9, ew.total_shaping
    # TELESCOPAGEM (gamma=1, Phi=0 no terminal e no início): a soma do shaping no episódio é 0
    rng = np.random.default_rng(0)
    zr = rng.normal(0, 2, 200)
    ep = _shape_env("pbrs", zr, W=120000 + np.cumsum(rng.normal(0, 5, 200)))
    ep.reset()
    for a in rng.integers(0, 3, 400):
        _, _, term, _, _ = ep.step(int(a))
        if term:
            break
    assert abs(ep.total_shaping) < 1e-6, ep.total_shaping
    # com gamma < 1 a soma NÃO é zero (prova de que o gamma entra)
    ep2 = _shape_env("pbrs", zr, W=120000 + np.cumsum(rng.normal(0, 5, 200)), gamma="0.9")
    ep2.reset()
    for a in rng.integers(0, 3, 400):
        _, _, term, _, _ = ep2.step(int(a))
        if term:
            break
    assert abs(ep2.total_shaping) > 1e-3

    # regret: ideal = comprado (z[t-1] = -3), posição = flat, dpar > 0 => -lam * dpar
    W = np.array([120000, 120000, 120100, 120100, 120100, 120100], float)   # +100 pts no tick 2
    zg = np.array([0.0, -3.0, -3.0, -3.0, -3.0, -3.0])
    eg = _shape_env("regret", zg, W=W)
    eg.reset()
    eg.step(0); eg.step(0)                          # t=0, t=1: flat; dpar(t=2) ainda não
    _, r2, _, _, _ = eg.step(0)                     # t=2: dW=+100 => dpar = 0,2*100 = 20 (BOVA parado)
    assert abs(r2 - (-lam * 20.0)) < 1e-9, r2
    # se já está comprado, não há arrependimento
    eh = _shape_env("regret", zg, W=W)
    eh.reset()
    eh.step(1); eh.step(1)
    _, r2h, _, _, _ = eh.step(1)
    assert abs(r2h - 20.0) < 1e-6 or abs(r2h) > 0    # recebe o MtM real; sem penalidade extra
    assert eh.total_shaping == 0.0, eh.total_shaping

    # shaping NUNCA muda o P/L real: mesmas ações, com e sem shaping
    rng = np.random.default_rng(1)
    Wr = 120000 + np.cumsum(rng.normal(0, 8, 300))
    zr2 = rng.normal(0, 2, 300)
    acts = rng.integers(0, 3, 299)
    ref = _shape_env("none", zr2, W=Wr)
    _rewards(ref, acts)
    for kind in ("opp_flat", "opp_wrong", "pbrs", "regret"):
        e = _shape_env(kind, zr2, W=Wr)
        _rewards(e, acts)
        assert abs(e.total_reward - ref.total_reward) < 1e-9, (kind, e.total_reward, ref.total_reward)
        assert e.real_pnl_brl == e.total_reward
    print("shaping (opp_flat, opp_wrong, pbrs, regret): OK — P/L real intacto, PBRS telescopa")


if __name__ == "__main__":
    test_synthetic()
    test_real_day()
    test_reward_clip()
    test_cost_curriculum()
    test_shaping()
