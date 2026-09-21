"""
Previsibilidade das oportunidades com features CAUSAIS mais ricas -- antes de
qualquer RL.

Pergunta: dado o que se vê no tick t (só passado/presente), dá para escolher
entradas do par hedgeado (WIN + BOVA11) que rendam P/L LÍQUIDO positivo depois de
custos, saindo depois de um horizonte FIXO (saída causal, operável)?

Para cada horizonte h e cada direção (comprado/vendido no par) o alvo é o P/L
líquido em R$ de entrar em t e sair em t+h, com a MESMA contabilidade do
HedgedPairEnv (N_BOVA fixo na abertura, custo 0,25 + 0,000230*V_BOVA por lado;
no modo `realistic`, mais o meio-spread bid/ask das duas pernas em cada lado, isto
é, comprar no ask e vender no bid). Um regressor (gradient boosting) por
(horizonte, direção, conjunto de features) é treinado em dias de treino; em
validação escolhe-se o limiar de entrada e em teste avalia-se a estratégia
sequencial: entra quando a previsão máxima passa do limiar, fica h ticks, não
reentra durante a posição.

Conjuntos de features (todas causais):
  base       -- spreads no tick, largura das cotações, hora, variações em 10/100 ticks
  rich       -- base + idade da cotação (WIN e BOVA11), variações em 1..500 ticks,
                estatísticas móveis (100/1000 ticks), "quem moveu" (WIN x preço
                justo), tempo em episódio extremo, atividade das cotações
  rich_noage -- rich sem as features de idade/atividade da cotação (testa a
                hipótese de que o edge vem de cotação defasada)
Não há volume nas colunas cacheadas dos pregões; ficam de fora.

Uso (de src/; no WSL/venv com sklearn):
    python learnability_rich.py --selftest                 # confere o P/L contra o HedgedPairEnv
    python learnability_rich.py                             # análise completa (~15 min)
    python learnability_rich.py --train 1-60 --val 362-376 --test 392-406   # versão rápida
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from config import POINT_VALUE_BRL as PV
from rl_trading_pipeline import (
    process_day_cached, HEDGE_FACTOR, WIN_COST_PER_SIDE_BRL, BOVA_COST_PCT_PER_SIDE, _mp_context,
)

HORIZONS = (50, 200, 600)
CAP = 5000
TAIL_X = 0.4          # |spread|/custo a partir do qual o tick é sempre mantido no treino


# --------------------------------------------------------------------------- dados por dia
def day_arrays(order):
    df = process_day_cached(order)
    a = {c: df[c].to_numpy(dtype=float) for c in ("bid", "ask", "bbid", "bask", "Wbjusto", "Wajusto")}
    a["W"] = (a["bid"] + a["ask"]) / 2
    a["B"] = (a["bbid"] + a["bask"]) / 2
    a["fair"] = (a["Wbjusto"] + a["Wajusto"]) / 2
    dt = pd.to_datetime(df["datahora"])
    a["tod"] = (dt.dt.hour * 60 + dt.dt.minute).to_numpy(dtype=float)
    return a


def net_pnl(a, h, realistic):
    """P/L líquido (R$) de ENTRAR em t e SAIR em t+h, para comprado e vendido no par.
    Retorna (net_long, net_short) com NaN nos últimos h ticks."""
    W, B = a["W"], a["B"]
    n = len(W)
    N = W / (HEDGE_FACTOR * B)                       # ações de BOVA11 abertas em t (fixas)
    t = np.arange(n - h)
    e = t + h
    term1 = PV * (W[e] - W[t]) - N[t] * (B[e] - B[t])          # ganho do par comprado (long WIN / short BOVA)
    co = WIN_COST_PER_SIDE_BRL + BOVA_COST_PCT_PER_SIDE * (N[t] * B[t])
    cc = WIN_COST_PER_SIDE_BRL + BOVA_COST_PCT_PER_SIDE * (N[t] * B[e])
    if realistic:
        half_w = lambda i: 0.5 * (a["ask"][i] - a["bid"][i]) * PV
        half_b = lambda i: 0.5 * (a["bask"][i] - a["bbid"][i]) * N[t]
        co = co + half_w(t) + half_b(t)
        cc = cc + half_w(e) + half_b(e)
    nl = np.full(n, np.nan); ns = np.full(n, np.nan)
    nl[t] = term1 - co - cc
    ns[t] = -term1 - co - cc
    return nl, ns


# --------------------------------------------------------------------------- features
def _age(x):
    """Ticks desde a última mudança de x (0 quando acabou de mudar), com teto CAP."""
    ch = np.r_[True, np.diff(x) != 0]
    idx = np.where(ch, np.arange(len(x)), 0)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(x)) - last, CAP).astype(float)


def _lag_diff(x, k):
    d = np.zeros_like(x)
    d[k:] = x[k:] - x[:-k]
    return d


def features(a):
    W = a["W"]
    cost_pts = (2 * WIN_COST_PER_SIDE_BRL + 2 * BOVA_COST_PCT_PER_SIDE * W / HEDGE_FACTOR) / PV
    s_buy, s_sell = a["ask"] - a["Wbjusto"], a["bid"] - a["Wajusto"]
    f = {"s_buy": s_buy, "s_sell": s_sell, "s_buy_c": s_buy / cost_pts, "s_sell_c": s_sell / cost_pts,
         "win_width": a["ask"] - a["bid"], "bova_width": a["bask"] - a["bbid"], "tod": a["tod"]}
    for k in (10, 100):
        f[f"s_buy_d{k}"] = _lag_diff(s_buy, k)
        f[f"s_sell_d{k}"] = _lag_diff(s_sell, k)
    base = list(f)

    # ---- rico: dinâmica do spread
    for k in (1, 5, 20, 50, 200, 500):
        f[f"s_buy_d{k}"] = _lag_diff(s_buy, k)
        f[f"s_sell_d{k}"] = _lag_diff(s_sell, k)
    mid_s = (s_buy + s_sell) / 2                                   # spread central (sem o efeito largura)
    ser = pd.Series(mid_s)
    for w in (100, 1000):
        r = ser.rolling(w, min_periods=1)
        mu, sd = r.mean().to_numpy(), r.std().fillna(0).to_numpy()
        f[f"mid_minus_mean{w}"] = mid_s - mu
        f[f"mid_z{w}"] = (mid_s - mu) / np.maximum(sd, 1e-6)
        f[f"std{w}"] = sd
        f[f"mid_minus_max{w}"] = mid_s - r.max().to_numpy()
        f[f"mid_minus_min{w}"] = mid_s - r.min().to_numpy()
    # ---- rico: quem moveu (WIN x preço justo)
    for k in (10, 50, 200):
        dw, dfair = _lag_diff(W, k), _lag_diff(a["fair"], k)
        f[f"dWIN{k}"], f[f"dFAIR{k}"], f[f"who{k}"] = dw, dfair, dfair - dw
    # ---- rico: tempo em episódio extremo
    extreme = (np.abs(mid_s) > 0.5 * cost_pts).astype(float)
    ep = np.zeros(len(W))
    run = 0.0
    for i in range(len(W)):
        run = run + 1 if extreme[i] else 0.0
        ep[i] = min(run, CAP)
    f["ticks_in_extreme"] = ep
    f["extreme_frac100"] = pd.Series(extreme).rolling(100, min_periods=1).mean().to_numpy()
    rich = [k for k in f if k not in base]
    # ---- idade/atividade da cotação (candidatas a "cotação defasada")
    age_keys = []
    for name, x in (("bova", a["B"]), ("win", W)):
        f[f"{name}_age"] = _age(x)
        chg = np.r_[0.0, (np.diff(x) != 0).astype(float)]
        f[f"{name}_act100"] = pd.Series(chg).rolling(100, min_periods=1).sum().to_numpy()
        age_keys += [f"{name}_age", f"{name}_act100"]
    sets = {"base": base, "rich": rich + base + age_keys, "rich_noage": rich + base}
    return pd.DataFrame(f).astype(np.float32), sets


def day_pack(args):
    order, horizons, realistic, stride, keep_all = args
    a = day_arrays(order)
    X, sets = features(a)
    out = {"order": order, "sets": sets, "n": len(X)}
    if keep_all:
        idx = np.arange(len(X))
        out["w"] = np.ones(len(idx), dtype=np.float32)
    else:
        # subamostra de 1/stride dos ticks, MAS mantém todos os ticks com sinal forte
        # (x >= TAIL_X: os únicos com chance real de P/L positivo, ~1-2% dos ticks) --
        # com pesos para não distorcer a média. Sem isso a cauda some do treino.
        x = np.maximum(-X["s_buy_c"].to_numpy(), X["s_sell_c"].to_numpy())
        tail = x >= TAIL_X
        base_idx = np.arange(order % stride, len(X), stride)
        idx = np.union1d(base_idx, np.flatnonzero(tail))
        w = np.where(tail[idx], 1.0, float(stride))
        # ticks de cauda que também caem na subamostra base já valem 1 (não somam duas vezes)
        out["w"] = w.astype(np.float32)
    out["idx"] = idx
    out["X"] = X.iloc[idx].reset_index(drop=True)
    for h in horizons:
        nl, ns = net_pnl(a, h, realistic)
        out[f"nl{h}"], out[f"ns{h}"] = nl[idx], ns[idx]
    return out


def load_days(orders, horizons, realistic, stride, keep_all, workers):
    jobs = [(o, horizons, realistic, stride, keep_all) for o in orders]
    if workers > 1:
        with _mp_context().Pool(workers) as pool:
            return pool.map(day_pack, jobs, chunksize=1)
    return [day_pack(j) for j in jobs]


# --------------------------------------------------------------------------- estratégia sequencial
def run_strategy(pred_l, pred_s, net_l, net_s, h, thr, n_days_ticks):
    """Entra quando max(previsão) > thr; sai em h ticks; não reentra durante a posição.
    pred_*/net_* são por tick (contíguos, um dia). Retorna lista de P/L realizados."""
    best = np.maximum(pred_l, pred_s)
    cand = np.flatnonzero((best > thr) & ~np.isnan(net_l))
    pnls, free_at = [], 0
    for i in cand:
        if i < free_at:
            continue
        pnls.append(net_l[i] if pred_l[i] >= pred_s[i] else net_s[i])
        free_at = i + h
    return pnls


def predict_days(models, packs, sets, name, h):
    """Previsões (comprado, vendido) por dia para o conjunto de features `name`."""
    cols = sets[name]
    out = []
    for p in packs:
        X = p["X"][cols]
        out.append((models[(name, h, "l")].predict(X), models[(name, h, "s")].predict(X),
                    p[f"nl{h}"], p[f"ns{h}"]))
    return out


def grid(per_day, h, thr_grid):
    """P/L da estratégia sequencial para cada limiar de entrada."""
    rows = []
    for thr in thr_grid:
        allp = [run_strategy(pl, ps, nl, ns, h, thr, len(pl)) for pl, ps, nl, ns in per_day]
        flat = np.array([x for d in allp for x in d]) if any(len(d) for d in allp) else np.array([])
        rows.append({"limiar_R$": float(thr), "negocios_dia": len(flat) / len(per_day),
                     "pl_dia_R$": float(flat.sum() / len(per_day)) if len(flat) else 0.0,
                     "pl_medio_negocio": float(flat.mean()) if len(flat) else np.nan,
                     "acerto": float((flat > 0).mean()) if len(flat) else np.nan})
    return pd.DataFrame(rows)


def rank_quality(per_day):
    """Correlação de postos entre a previsão máxima e o P/L realizado (mesma direção)."""
    from scipy.stats import spearmanr
    best_pred = np.concatenate([np.maximum(d[0], d[1]) for d in per_day])
    real = np.concatenate([np.where(d[0] >= d[1], d[2], d[3]) for d in per_day])
    ok = ~np.isnan(real)
    return float(spearmanr(best_pred[ok][::20], real[ok][::20]).correlation)


def parse_range(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


# --------------------------------------------------------------------------- autoteste
def _env_pair_pnl(df, t, h, action, realistic):
    """Recompensa do HedgedPairEnv para: ficar flat até t, abrir em t (ação 1=comprado, 2=vendido)
    e fechar em t+h."""
    from rl_trading_pipeline import HedgedPairEnv
    env = HedgedPairEnv(df, np.zeros((len(df), 2)), reward_scale=1.0, include_spread_cost=realistic)
    env.reset()
    for _ in range(t):
        env.step(0)
    total = env.step(action)[1]                 # abre no tick t
    for _ in range(h - 1):
        total += env.step(action)[1]            # mantém: MtM de t até t+h-1
    total += env.step(0)[1]                     # MtM até t+h e fecha em t+h
    return total


def selftest():
    """O P/L de entrar em t e sair em t+h calculado aqui == recompensa do HedgedPairEnv."""
    order, h = 1, 200
    df = process_day_cached(order)
    a = day_arrays(order)
    rng = np.random.default_rng(0)
    for realistic in (False, True):
        nl, ns = net_pnl(a, h, realistic)
        for t in rng.integers(0, len(df) - h - 5, size=6):
            for action, ref in ((1, nl), (2, ns)):
                got = _env_pair_pnl(df, int(t), h, action, realistic)
                assert abs(got - ref[t]) < 1e-6, (realistic, action, t, got, ref[t])
    print("autoteste OK: P/L de entrada+saída em horizonte fixo == HedgedPairEnv (com e sem meio-spread)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", default="1-150")
    ap.add_argument("--val", default="362-391")
    ap.add_argument("--test", default="392-421")
    ap.add_argument("--stride", type=int, default=10, help="subamostra dos ticks de treino")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "3")))
    ap.add_argument("--out", default=os.path.join(os.environ.get("RUNS_DIR", "runs"), "learnability"))
    ap.add_argument("--costs", default="realistic,spec")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    from sklearn.ensemble import HistGradientBoostingRegressor
    os.makedirs(a.out, exist_ok=True)
    tr_o, va_o, te_o = parse_range(a.train), parse_range(a.val), parse_range(a.test)
    all_rows = []
    for cost in a.costs.split(","):
        realistic = cost == "realistic"
        t0 = time.time()
        print(f"\n########## custos: {cost} ##########", flush=True)
        tr = load_days(tr_o, HORIZONS, realistic, a.stride, False, a.workers)
        va = load_days(va_o, HORIZONS, realistic, 1, True, a.workers)
        te = load_days(te_o, HORIZONS, realistic, 1, True, a.workers)
        sets = tr[0]["sets"]
        Xtr = pd.concat([p["X"] for p in tr], ignore_index=True)
        print(f"treino: {len(tr)} dias, {len(Xtr):,} linhas; val {len(va)} dias; teste {len(te)} dias "
              f"({time.time()-t0:.0f}s)", flush=True)
        models = {}
        for h in HORIZONS:
            ytr = {d: np.concatenate([p[f"n{d}{h}"] for p in tr]) for d in ("l", "s")}
            wtr = np.concatenate([p["w"] for p in tr])
            ok = ~np.isnan(ytr["l"])
            for name, cols in sets.items():
                for d in ("l", "s"):
                    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                                                      min_samples_leaf=40, l2_regularization=1.0, random_state=0)
                    m.fit(Xtr.loc[ok, cols], np.clip(ytr[d][ok], -40, 80), sample_weight=wtr[ok])
                    models[(name, h, d)] = m
            print(f"  h={h}: modelos treinados ({time.time()-t0:.0f}s)", flush=True)
        for h in HORIZONS:
            for name in sets:
                pv, pt = predict_days(models, va, sets, name, h), predict_days(models, te, sets, name, h)
                # limiares = percentis das previsões NA VALIDAÇÃO (o modelo quase nunca prevê P/L > 0)
                best_v = np.concatenate([np.maximum(d[0], d[1]) for d in pv])
                qs = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995)
                thr_grid = sorted(set(float(np.quantile(best_v, q)) for q in qs))
                gv, gt = grid(pv, h, thr_grid), grid(pt, h, thr_grid)
                gt["percentil_val"] = [np.mean(best_v <= t) for t in thr_grid]
                k = gv["pl_dia_R$"].idxmax()                            # limiar escolhido NA VALIDAÇÃO
                t_at = gt.iloc[k]
                row = {"custos": cost, "horizonte": h, "features": name,
                       "limiar_escolhido": gv.loc[k, "limiar_R$"], "percentil": t_at["percentil_val"],
                       "val_negocios_dia": gv.loc[k, "negocios_dia"], "val_pl_dia": gv.loc[k, "pl_dia_R$"],
                       "teste_negocios_dia": t_at["negocios_dia"], "teste_pl_dia": t_at["pl_dia_R$"],
                       "teste_pl_negocio": t_at["pl_medio_negocio"], "teste_acerto": t_at["acerto"],
                       "teste_melhor_da_grade (otimista)": gt["pl_dia_R$"].max(),
                       "pred_max_val": float(best_v.max()), "spearman_teste": rank_quality(pt)}
                all_rows.append(row)
                gt.assign(custos=cost, horizonte=h, features=name).to_csv(
                    os.path.join(a.out, f"grade_teste_{cost}_h{h}_{name}.csv"), index=False)
        # referência: P/L médio por tick de entrar SEMPRE (sem modelo), teste
        for h in HORIZONS:
            allv = np.concatenate([np.maximum(p[f"nl{h}"], p[f"ns{h}"])[~np.isnan(p[f"nl{h}"])] for p in te])
            print(f"  ref. custos={cost} h={h}: melhor direção em hindsight, média por entrada = {allv.mean():+.2f} R$", flush=True)
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(a.out, "resumo.csv"), index=False)
    pd.set_option("display.width", 220)
    with pd.option_context("display.float_format", lambda x: f"{x:,.2f}"):
        print("\n=== RESUMO (limiar escolhido na validação; P/L em R$/dia por contrato, teste) ===")
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
