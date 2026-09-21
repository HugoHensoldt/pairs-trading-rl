"""Varre os pregões 1..N procurando anomalias nos ticks de WIN/BOVA11.

Critérios (dentro do horário usado pela heurística, 10:20-16:30):
  - datas diferentes entre os arquivos WIN e BOVA11 do mesmo pregão
  - cotações cruzadas no WIN (bid > ask), que geram spreads negativos
  - spread do WIN > 100 pts em algum tick
  - amplitude diária do WIN fora do normal (> LIM_AMPLITUDE pts entre min e max do bid)
  - salto entre ticks consecutivos do bid do WIN > LIM_SALTO pts

Uso: python src/scan_bad_sessions.py [--last 488] [--workers 6]
Saída: tabela dos pregões suspeitos e docs/lstm_pregoes_suspeitos.csv (todos).
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("TICK_DATA_DIR", "C:/Users/HugoV/tick_data"))
LIM_AMPLITUDE, LIM_SALTO = 4000, 600


def _read(name):
    with open(DATA / name, "r", encoding="UTF-16 LE") as f:
        return pd.read_json(f)


def scan(order):
    try:
        w, b = _read(f"{order}WINM21.json"), _read(f"{order}BOVA11.json")
    except Exception as e:  # arquivo ausente/corrompido
        return dict(order=order, erro=str(e)[:60])
    dw, db = w.datahora.iloc[0][:10], b.datahora.iloc[0][:10]
    t = pd.to_datetime(w.datahora)
    sel = ((t.dt.hour * 60 + t.dt.minute) >= 10 * 60 + 20) & ((t.dt.hour * 60 + t.dt.minute) <= 16 * 60 + 30)
    ws = w[sel.to_numpy()]
    bid = ws.bid.to_numpy(float)
    return dict(
        order=order, data_win=dw, data_bova=db, data_diferente=(dw != db), ticks=len(ws),
        cruzados=int((ws.bid > ws.ask).sum()), spread_max=float((ws.ask - ws.bid).max()),
        amplitude=float(bid.max() - bid.min()), salto_max=float(np.abs(np.diff(bid)).max()) if len(bid) > 1 else 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=488)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="docs/lstm_pregoes_suspeitos.csv")
    a = ap.parse_args()
    with Pool(a.workers) as p:
        rows = p.map(scan, range(1, a.last + 1))
    D = pd.DataFrame(rows)
    D["suspeito"] = (D.get("data_diferente", False) == True) | (D.cruzados > 0) | (D.spread_max > 100) \
        | (D.amplitude > LIM_AMPLITUDE) | (D.salto_max > LIM_SALTO) | D.get("erro", pd.Series(dtype=object)).notna()
    D.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(f"{len(D)} pregões, {int(D.suspeito.sum())} suspeitos")
    print(D[D.suspeito].to_string(index=False))
    print("\nmediana amplitude:", D.amplitude.median(), "| p99 amplitude:", D.amplitude.quantile(.99),
          "| mediana salto_max:", D.salto_max.median())
