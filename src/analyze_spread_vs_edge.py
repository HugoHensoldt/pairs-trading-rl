"""O spread bid-ask explica o P/L negativo da heurística? Decomposição por sigma.

Para cada negociação da heurística (mesmas entradas usadas pela LSTM) mede, nos
dados de tick:

  spread        ask - bid do WIN no tick da entrada (pontos)
  d             |Saida - Entrada| = distância até o alvo (pontos); SG = Re*d, SL = Ri*d
  pl_bidask     lucro idealizado da heurística atual (±SG / -SL; fechamento
                compulsório a mercado): é o `pl` do estágio `--stage pl`.
                A entrada é no ask (compra) / bid (venda) e o gatilho de saída
                usa o lado oposto do book, então o custo de cruzar o spread
                já está embutido na PROBABILIDADE de atingir o alvo.
  pl_mid        a MESMA entrada (mesmo tick e lado) executada a preço MEDIO
                (entrada e gatilho de saída no mid, mesmos SG/SL): o que a
                regra faria SEM pagar o spread.
  pl_real       lucro real em pontos, cruzando o book na entrada e na saída
                (compra: bid_saida - ask_entrada; venda: bid_entrada - ask_saida).

Se a regra tem edge de reversão, pl_mid > 0; e pl_mid - pl_bidask mede o custo
do spread. Sem custos de corretagem/emolumentos (só o spread).

Uso: python src/analyze_spread_vs_edge.py --every 6 --workers 4
Saída: tabela no terminal e docs/lstm_spread_vs_edge.csv
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_pipeline as L  # noqa: E402

N_TICKS = L.N_TICKS
# combinações analisadas: sigma varrido (1.4-1.8) e maiores; (Re, Ri) representativos
SIGMAS = [1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0]
REI = [(re, ri) for re in (0.5, 0.75, 0.9) for ri in (0.5, 1.0, 1.5)]
# pregões com cotações impossíveis (saltos de 7.000 a 112.000 pts, bid > ask): ver src/scan_bad_sessions.py
BAD_ORDERS = {168, 304, 336, 371}


def day_trades(args):
    order, combos = args
    base = L.load_day_base(order)
    out = []
    for sigma, Re, Ri in combos:
        df = L.apply_heuristic(base, sigma=sigma, Re=Re, Ri=Ri)
        pos = df['posicao'].to_numpy()
        ask, bid = df['ask'].to_numpy(float), df['bid'].to_numpy(float)
        SG, SL = df['SG'].to_numpy(float), df['SL'].to_numpy(float)
        lucro = df['lucro'].to_numpy(float)
        mid = (ask + bid) / 2
        zeros = np.flatnonzero(pos == 0)
        prev = np.r_[np.nan, pos[:-1]]
        for i in np.flatnonzero((prev == 0) & (pos != 0)):
            if i < N_TICKS - 1:
                continue
            k = np.searchsorted(zeros, i, side='right')
            if k >= len(zeros):
                continue                      # abriu no último tick: sem fechamento
            j = zeros[k]
            side, sg, sl = pos[i], SG[i], SL[i]
            m0 = mid[i]
            fut = mid[i + 1:]
            up = np.flatnonzero(fut >= m0 + (sg if side == 1 else sl))
            dn = np.flatnonzero(fut <= m0 - (sl if side == 1 else sg))
            hit_up, hit_dn = (up[0] if len(up) else 10**9), (dn[0] if len(dn) else 10**9)
            if hit_up == hit_dn == 10**9:
                pl_mid = (mid[-1] - m0) * side          # fechamento compulsório a mercado
            elif (hit_up < hit_dn) == (side == 1):
                pl_mid = sg                             # alvo
            else:
                pl_mid = -sl                            # stop
            real = (bid[j] - ask[i]) if side == 1 else (bid[i] - ask[j])
            out.append((order, sigma, Re, Ri, int(side), ask[i] - bid[i], sg / Re, sg, sl,
                        lucro[j], pl_mid, real, int(j - i)))
    return out


COLS = ['order', 'sigma', 'Re', 'Ri', 'side', 'spread', 'd', 'SG', 'SL', 'pl_bidask', 'pl_mid', 'pl_real', 'dur_ticks']

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=6, help="usa 1 a cada N pregões de L.ALL_ORDERS")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="docs/lstm_spread_vs_edge.csv")
    a = ap.parse_args()

    combos = [(s, re, ri) for s in SIGMAS for re, ri in REI]
    orders = [o for o in L.ALL_ORDERS if o not in BAD_ORDERS][::a.every]
    print(f"{len(orders)} pregões x {len(combos)} combinações", flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(day_trades, [(o, combos) for o in orders]), 1):
            rows += r
            if n % 10 == 0:
                print(f"  {n}/{len(orders)}", flush=True)
    T = pd.DataFrame(rows, columns=COLS)
    T = T[T.spread > 0]                     # descarta entradas com cotação cruzada

    def summ(g):
        return pd.Series({
            'n': len(g), 'spread_med': g.spread.median(), 'd_med': g.d.median(), 'SG_med': g.SG.median(),
            'SL_med': g.SL.median(), 'SG/spread_med': (g.SG / g.spread.clip(lower=1)).median(),
            'win_bidask': (g.pl_bidask > 0).mean(), 'win_mid': (g.pl_mid > 0).mean(),
            'pl_bidask': g.pl_bidask.mean(), 'pl_mid': g.pl_mid.mean(), 'pl_real': g.pl_real.mean(),
            'custo_spread(mid-bidask)': g.pl_mid.mean() - g.pl_bidask.mean(),
            'spread_medio': g.spread.mean()})

    test_start = L.split_orders()[1][0]
    T['periodo'] = np.where(T.order >= test_start, 'teste', 'dev')
    by = T.groupby(['sigma', 'Re', 'Ri']).apply(summ).reset_index()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    by.to_csv(a.out, index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)

    def by_day(g):
        """média e erro padrão (entre pregões) do P/L médio por negociação, em pontos."""
        dm = g.groupby('order')[['pl_bidask', 'pl_mid']].mean()
        n = len(dm)
        return pd.Series({'n_trades': len(g), 'n_dias': n,
                          'pl_bidask': g.pl_bidask.mean(), 'pl_mid': g.pl_mid.mean(), 'pl_real': g.pl_real.mean(),
                          'se_bidask(dias)': dm.pl_bidask.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                          'SG_med': g.SG.median(), 'spread_med': g.spread.median(),
                          'spread/SG_%': 100 * (g.spread / g.SG).median(), 'win_bidask': (g.pl_bidask > 0).mean()})

    print("\n== por sigma x periodo (todas as combinações Re/Ri juntas) ==")
    print(T.groupby(['periodo', 'sigma']).apply(by_day).round(2).to_string())
    print("\n== por sigma x Re x Ri, periodo=teste ==")
    print(T[T.periodo == 'teste'].groupby(['sigma', 'Re', 'Ri']).apply(by_day).round(2).to_string())
    print("\n== por combinação (todos os pregões) ==")
    print(by.round(3).to_string(index=False))
    print("\n== por faixa de SG/spread (todas as combinações) ==")
    T['faixa'] = pd.cut(T.SG / T.spread.clip(lower=1), [0, 2, 5, 10, 20, 50, 1e9])
    print(T.groupby('faixa').apply(summ)[['n', 'pl_bidask', 'pl_mid', 'pl_real', 'win_bidask', 'win_mid']].round(3).to_string())
    T.to_csv(str(a.out).replace('.csv', '_trades.csv'), index=False)
