"""
Pares sintéticos WIN x BOVA11 com VERDADE CONHECIDA, para testar se o agente de
RL consegue aprender arbitragem quando ela existe (e não aprende nada quando não
existe), independente dos ruídos dos dados reais.

Cada pregão sintético é um DataFrame com as mesmas colunas do process_day_cached
(bid, ask, bbid, bask, Wbjusto, Wajusto, datahora) mais `true_x` (o spread
verdadeiro em pontos de WIN). Ordens >= SYNTH_BASE são sintéticas: o pipeline
chama synthetic_day(order) em vez de ler os arquivos de tick (ver
process_day_cached em rl_trading_pipeline.py).

Modelo (por pregão, 30.000 ticks; os 3.000 primeiros são descartados como no
process_day real):
  BOVA11 latente : passeio aleatório geométrico, vol diária SYNTH_B_DAILY_VOL.
                   Cotações com tick de R$ 0,01 e spread de 1 tick (cotação
                   inalterada em boa parte dos ticks, como no dado real).
  WIN, SYNTH_KIND=coint : W_mid = 1000 * B_lat + c(d) + X_t, com X_t um processo
                   Ornstein-Uhlenbeck (média 0, desvio estacionário
                   SYNTH_SPREAD_STD pts, meia-vida SYNTH_HALF_LIFE ticks) e c(d)
                   a "razão" do dia: deriva de SYNTH_DRIFT pts por pregão e salto
                   de SYNTH_ROLL pts a cada SYNTH_ROLL_EVERY pregões (rollover),
                   como medido nos dados reais (~-44 pts/pregão, +945 no rollover).
                   Cotações com tick de 5 pts e spread de 1 tick.
  WIN, SYNTH_KIND=null  : passeio aleatório independente do BOVA11 (mesma vol);
                   NÃO existe arbitragem (controle negativo).

O spread real do par tem desvio de ~16 pts contra um custo de ida e volta de
~58 pts, então com o desvio real quase nunca há lucro após custos. O padrão
sintético usa desvio de 40 pts para que a arbitragem exista (a regra ótima
causal, ver synthetic_benchmark.py, tem P/L positivo); mude SYNTH_SPREAD_STD para
outros cenários.
"""

import os

import numpy as np
import pandas as pd

SYNTH_BASE = 1_000_000
N_WARMUP = 3000
N_DAY = 27_000
RATIO = 1000.0            # W ~ 1000 * B (WIN ~ 120.000, BOVA11 ~ 120)


def synth_config():
    e = os.environ.get
    return {
        "kind": e("SYNTH_KIND", "coint"),
        "spread_std": float(e("SYNTH_SPREAD_STD", "40")),
        "half_life": float(e("SYNTH_HALF_LIFE", "300")),
        "drift": float(e("SYNTH_DRIFT", "-30")),
        "roll_every": int(e("SYNTH_ROLL_EVERY", "32")),
        "roll_pts": float(e("SYNTH_ROLL", "960")),
        "b_daily_vol": float(e("SYNTH_B_DAILY_VOL", "0.01")),
        "day_gap_vol": float(e("SYNTH_DAY_GAP_VOL", "0.006")),
    }


_MASTER = {}


def _master_walk(n=4096, seed=777, gap_vol=0.006):
    """Log-nível inicial de cada dia (passeio entre dias), determinístico."""
    key = (n, seed, gap_vol)
    if key not in _MASTER:
        rng = np.random.default_rng(seed)
        _MASTER[key] = np.log(120.0) + np.cumsum(rng.normal(0.0, gap_vol, n))
    return _MASTER[key]


def ratio_offset(d, cfg):
    """c(d): razão do dia em pts de WIN (sawtooth de carry + rollover)."""
    k, j = divmod(d, cfg["roll_every"])
    return cfg["drift"] * j                         # volta a 0 no rollover (salto = -drift*roll_every)


def synthetic_day(order, cfg=None):
    cfg = cfg or synth_config()
    d = int(order - SYNTH_BASE)
    rng = np.random.default_rng(12_345 + d)
    n = N_WARMUP + N_DAY

    # --- BOVA11 latente ---------------------------------------------------
    b0 = float(np.exp(_master_walk(gap_vol=cfg["day_gap_vol"])[d]))
    sig_tick = cfg["b_daily_vol"] / np.sqrt(N_DAY)
    b_lat = b0 * np.exp(np.cumsum(rng.normal(0.0, sig_tick, n)))

    # --- WIN ----------------------------------------------------------------
    if cfg["kind"] == "coint":
        rho = 0.5 ** (1.0 / cfg["half_life"])
        s_x = cfg["spread_std"]
        eps = rng.normal(0.0, 1.0, n)
        x = np.empty(n)
        x[0] = rng.normal(0.0, s_x)
        inn = s_x * np.sqrt(1.0 - rho ** 2)
        for t in range(1, n):                        # AR(1) exato (OU discretizado)
            x[t] = rho * x[t - 1] + inn * eps[t]
        w_mid = RATIO * b_lat + ratio_offset(d, cfg) + x
    elif cfg["kind"] == "null":
        w0 = RATIO * b0 * np.exp(rng.normal(0.0, 0.003))
        w_lat = w0 * np.exp(np.cumsum(rng.normal(0.0, sig_tick, n)))
        x = np.zeros(n)
        w_mid = w_lat + 0.0
    else:
        raise ValueError(f"SYNTH_KIND desconhecido: {cfg['kind']!r} (coint | null)")

    # --- cotações -------------------------------------------------------------
    b_bid = np.floor(b_lat / 0.01) * 0.01
    b_ask = b_bid + 0.01
    w_bid = np.floor(w_mid / 5.0) * 5.0
    w_ask = w_bid + 5.0

    # --- "preço justo" clássico (mesma fórmula do process_day real) ---------
    ds = pd.DataFrame({"bid": w_bid, "ask": w_ask, "bbid": b_bid, "bask": b_ask})
    c_ask = (ds["ask"] / ds["bask"]).rolling(window=N_WARMUP).mean()
    c_bid = (ds["bid"] / ds["bbid"]).rolling(window=N_WARMUP).mean()
    ds["Wbjusto"] = ds["bbid"] * c_bid
    ds["Wajusto"] = ds["bask"] * c_ask

    # --- carimbos de tempo: 10:20-16:30, em dias úteis consecutivos ---------
    day = pd.bdate_range("2023-01-02", periods=d + 1)[-1]
    secs = np.linspace(0, 370 * 60, n)
    ds["datahora"] = (day + pd.Timedelta(hours=10, minutes=20) + pd.to_timedelta(secs, unit="s")).to_numpy()
    ds["true_x"] = x
    out = ds.iloc[N_WARMUP:].reset_index(drop=True)
    return out
