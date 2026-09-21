"""
Testes do estado RAW (só preços + features genéricas) e do gerador sintético.

  1) features raw: dimensão/nomes, finitas, CAUSAIS (as primeiras m linhas não
     mudam se o futuro é alterado), ablações RAW_LEVELS e RAW_DOY;
  2) bench: rotula os negócios em `trade_details` sem entrar no estado;
  3) sintético: desvio e meia-vida do spread, deriva e rollover entre dias,
     cotações com tick, cointegração vs controle nulo;
  4) referência: a regra ótima causal tem P/L positivo no 'coint' e não no 'null'.

Uso (de src/):  python smoke_test_raw_synthetic.py
"""

import os

import numpy as np
import pandas as pd

os.environ.setdefault("SYNTH_KIND", "coint")

from synthetic_pair import SYNTH_BASE, synthetic_day, synth_config, N_DAY
import rl_trading_pipeline as P


def test_raw_features():
    df = synthetic_day(SYNTH_BASE + 3)
    os.environ["STATE_KIND"] = "raw"
    X = P.build_raw_features(df)
    names = P.raw_feature_names()
    assert X.shape == (len(df), len(names)), (X.shape, len(names))
    assert np.all(np.isfinite(X))
    m = 5000
    X2 = P.build_raw_features(df.iloc[:m].reset_index(drop=True))
    assert np.allclose(X[:m], X2, atol=1e-12), "features NÃO são causais"
    df3 = df.copy()
    df3.loc[m:, ["bid", "ask", "bbid", "bask"]] *= 1.05                      # altera só o futuro
    X3 = P.build_raw_features(df3)
    assert np.allclose(X[:m], X3[:m], atol=1e-12), "features de t dependem do futuro"
    assert P.build_raw_features(df, levels=False).shape[1] == X.shape[1] - 4
    assert P.build_raw_features(df, doy=True).shape[1] == X.shape[1] + 2
    assert P.build_features(df).shape == X.shape and P.state_kind() == "raw"
    # nenhuma coluna direta do par: ret_h_win e ret_h_bova são SEPARADOS
    assert "ret100_win" in names and "ret100_bova" in names
    assert not any(k in "".join(names) for k in ("razao", "spread_par", "justo", "resid"))
    os.environ["STATE_KIND"] = "spread"
    print(f"raw features: {len(names)} colunas, causais, ablações ok")


def test_bench_and_env():
    df = synthetic_day(SYNTH_BASE + 5)
    os.environ["STATE_KIND"] = "raw"
    X = P.build_raw_features(df)
    mean, std = X.mean(0), X.std(0) + 1e-9
    feat = (X - mean) / std
    bench = P.build_bench_features(df)
    env = P.HedgedPairEnv(df, feat, reward_scale=1.0, track_details=True, bench=bench)
    obs, _ = env.reset()
    assert obs.shape == (X.shape[1] + 2,)
    rng = np.random.default_rng(0)
    acts = np.repeat(rng.integers(0, 3, size=300), 90)
    for a in acts:
        _, _, term, _, _ = env.step(int(a))
        if term:
            break
    assert env.n_trades_closed > 3
    d = env.trade_details[0]
    o = int(d[1])
    assert abs(d[4] - bench[o, 0]) < 1e-9 and abs(d[5] - bench[o, 1]) < 1e-9        # entry_z_* = bench, não o estado
    os.environ["STATE_KIND"] = "spread"
    print("env com bench: ok")


def test_synthetic_stats():
    cfg = synth_config()
    xs, acfs = [], []
    for d in range(6):
        df = synthetic_day(SYNTH_BASE + d)
        assert len(df) == N_DAY
        assert (df.ask > df.bid).all() and (df.bask > df.bbid).all()
        assert np.allclose((df.ask - df.bid), 5.0) and np.allclose(df.bask - df.bbid, 0.01)
        x = df.true_x.to_numpy()
        xs.append(x.std())
        h = int(cfg["half_life"])
        acfs.append(np.corrcoef(x[:-h], x[h:])[0, 1])
    assert abs(np.mean(xs) / cfg["spread_std"] - 1) < 0.35, np.mean(xs)
    assert abs(np.mean(acfs) - 0.5) < 0.2, np.mean(acfs)
    # deriva entre pregões e rollover: razão do dia c(d) = média(W - 1000*B)
    c = []
    for d in range(70):
        df = synthetic_day(SYNTH_BASE + d)
        c.append(np.mean(0.5 * (df.bid + df.ask) - 1000.0 * 0.5 * (df.bbid + df.bask)))
    c = np.array(c)
    dc = np.diff(c)
    assert abs(np.median(dc) - cfg["drift"]) < 25, np.median(dc)
    assert (dc > 0.5 * cfg["roll_pts"]).sum() >= 1, "sem rollover"
    # cotação do BOVA11 inalterada em boa parte dos ticks
    df = synthetic_day(SYNTH_BASE + 1)
    same = (np.diff(0.5 * (df.bbid + df.bask)) == 0).mean()
    assert 0.3 < same < 0.95, same
    print(f"sintético coint: desvio de X {np.mean(xs):.1f} pts, autocorr(meia-vida) {np.mean(acfs):.2f}, "
          f"deriva {np.median(dc):.0f} pts/pregão, rollover ok, BOVA inalterado em {same:.0%} dos ticks")


def test_reference_rule():
    import synthetic_benchmark as SB
    cfg = synth_config()
    orders = [SYNTH_BASE + i for i in range(3)]
    pnl = [SB.run_rule_day(o, 2.0 * cfg["spread_std"])[0] for o in orders]
    n_tr = [SB.run_rule_day(o, 2.0 * cfg["spread_std"])[1] for o in orders]
    print(f"regra 2σ no sintético {cfg['kind']}: {np.mean(pnl):+.1f} R$/dia, {np.mean(n_tr):.1f} negócios/dia")
    return float(np.mean(pnl))


if __name__ == "__main__":
    test_raw_features()
    test_bench_and_env()
    test_synthetic_stats()
    p_coint = test_reference_rule()
    assert p_coint > 0, "a regra ótima deveria ganhar no sintético cointegrado (ajuste SYNTH_SPREAD_STD)"
    os.environ["SYNTH_KIND"] = "null"
    P._DAY_MEMO.clear()
    p_null = test_reference_rule()
    assert p_null < p_coint * 0.2, (p_null, p_coint)
    print("TESTES OK")
