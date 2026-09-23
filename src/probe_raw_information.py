"""
O estado "só preços" (STATE_KIND=raw) contém a informação do spread de forma ACESSÍVEL?

Teste sem RL, em pregões sintéticos cointegrados (spread verdadeiro X_t conhecido):
regride X_t (e a variação futura de X) nas features raw, fora da amostra. Se nem uma
regressão consegue recuperar X_t, o RL não tem o que aprender; se a regressão consegue
(linear ou não linear) mas o RL não, o problema está no treino/crédito, não na observação.

Também decompõe a variância da razão log(WIN/BOVA11) entre dias e dentro do dia, e mede
quanto de X_t cada feature/combinação explica sozinha.

Uso (de src/):  python probe_raw_information.py [--train-days 40] [--val-days 10]
"""

import argparse
import os

import numpy as np
import pandas as pd

os.environ.setdefault("SYNTH_KIND", "coint")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from synthetic_pair import SYNTH_BASE, synthetic_day
import rl_trading_pipeline as P


def build(days, stride, levels=True, h_fut=300):
    X, x_true, fut, extra = [], [], [], []
    for d in days:
        df = synthetic_day(SYNTH_BASE + d)
        F = P.build_raw_features(df, levels=levels, doy=False)
        x = df["true_x"].to_numpy()
        f = np.full(len(x), np.nan)
        f[:-h_fut] = x[h_fut:] - x[:-h_fut]
        w = 0.5 * (df["bid"].to_numpy() + df["ask"].to_numpy())
        b = 0.5 * (df["bbid"].to_numpy() + df["bask"].to_numpy())
        lr = np.log(w) - np.log(b)
        names = P.raw_feature_names(levels=levels, doy=False)
        ex = {"lr": lr, "day": np.full(len(x), d)}
        for h in P.RAW_HORIZONS:
            ex[f"dret{h}"] = F[:, names.index(f"ret{h}_win")] - F[:, names.index(f"ret{h}_bova")]
        sl = slice(0, None, stride)
        X.append(F[sl]); x_true.append(x[sl]); fut.append(f[sl])
        extra.append(pd.DataFrame({k: v[sl] for k, v in ex.items()}))
    return np.concatenate(X), np.concatenate(x_true), np.concatenate(fut), pd.concat(extra, ignore_index=True)


def r2_fit(Xtr, ytr, Xva, yva, model):
    m = np.isfinite(ytr).ravel(); mv = np.isfinite(yva).ravel()
    model.fit(Xtr[m], ytr[m])
    return r2_score(yva[mv], model.predict(Xva[mv]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=40)
    ap.add_argument("--val-days", type=int, default=10)
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    tr = list(range(a.train_days))
    va = list(range(300, 300 + a.val_days))

    print(f"Sintético coint: spread_std={os.environ.get('SYNTH_SPREAD_STD', '40')} pts; treino {len(tr)} dias, validação {len(va)} dias\n")
    for levels in (True, False):
        Xtr, xtr, ftr, etr = build(tr, a.stride, levels=levels)
        Xva, xva, fva, eva = build(va, a.stride, levels=levels)
        mean, std = Xtr.mean(0), Xtr.std(0) + 1e-12
        Ztr, Zva = (Xtr - mean) / std, (Xva - mean) / std
        tag = "COM níveis de preço" if levels else "SEM níveis de preço"
        print(f"=== features raw {tag} ({Xtr.shape[1]} colunas) ===")
        print(f"  R² fora da amostra recuperando o spread X_t:   OLS = {r2_fit(Ztr, xtr, Zva, xva, LinearRegression()):+.3f}   "
              f"| GradientBoosting = {r2_fit(Ztr, xtr, Zva, xva, HistGradientBoostingRegressor(max_iter=200, random_state=0)):+.3f}")
        print(f"  R² prevendo a variação futura X(t+300)-X(t):   OLS = {r2_fit(Ztr, ftr, Zva, fva, LinearRegression()):+.3f}   "
              f"| GradientBoosting = {r2_fit(Ztr, ftr, Zva, fva, HistGradientBoostingRegressor(max_iter=200, random_state=0)):+.3f}")
        if levels:
            # referência: o quanto a MELHOR reversão possível (E[dX] = -0,5 X) explicaria
            print(f"  (teto teórico do R² de X(t+300)-X(t) se X_t fosse conhecido: {0.25 * 1 / (0.25 + 2 * 0.5 * 0.5):.2f}~)")

    # razão log(WIN/BOVA): variância entre dias x dentro do dia, em pts de WIN
    Xtr, xtr, ftr, etr = build(tr, a.stride, levels=True)
    w0 = 120000.0
    lr_pts = (etr["lr"] - etr["lr"].mean()) * w0
    between = etr.assign(v=lr_pts).groupby("day")["v"].mean().var()
    within = etr.assign(v=lr_pts).groupby("day")["v"].var().mean()
    print(f"\nRazão log(WIN/BOVA), em pts de WIN: variância ENTRE dias = {between:,.0f} (desvio {np.sqrt(between):.0f}); "
          f"DENTRO do dia = {within:,.0f} (desvio {np.sqrt(within):.0f}); spread verdadeiro X: desvio {xtr.std():.0f}")
    print("=> o nível da razão é dominado pela deriva/rollover entre dias; o sinal útil (X_t) é a parte intradia")

    # informação por combinação simples (sem RL)
    print("\nCorrelação com X_t (val), por feature/combinação isolada:")
    Xva, xva, fva, eva = build(va, a.stride, levels=True)
    names = P.raw_feature_names(levels=True, doy=False)
    rows = [(n, np.corrcoef(Xva[:, i], xva)[0, 1]) for i, n in enumerate(names)]
    rows += [(f"[combinação] ret{h}_win - ret{h}_bova", np.corrcoef(eva[f"dret{h}"], xva)[0, 1]) for h in P.RAW_HORIZONS]
    lr_intra = eva["lr"] - eva.groupby("day")["lr"].transform("mean")
    rows += [("[oráculo NÃO causal] razão - média do dia", np.corrcoef(lr_intra, xva)[0, 1])]
    for n, c in sorted(rows, key=lambda r: -abs(r[1]))[:10]:
        print(f"  {n:<48s} corr = {c:+.3f}")


if __name__ == "__main__":
    main()
