"""
O agente treinado aprendeu ARBITRAGEM? Diagnóstico de uma execução (checkpoint) em
pregões de validação/teste. Objetivo: medir a capacidade de o RL descobrir a relação
WIN x BOVA11 a partir de preços brutos, NÃO o resultado financeiro.

Testes (cada um grava CSV em <run-dir>/diag/ e entra no resumo diag/summary.md):

 A) ALINHAMENTO com métodos clássicos. Preferência do agente "se estivesse flat":
      pref_t = P(comprar par) - P(vender par)
    (rede avaliada com posição = 0 e P/L = 0, sem realimentação da própria
    trajetória) contra o desvio de cada método clássico, orientado para que
    POSITIVO = WIN caro em relação ao BOVA11 (o arbitrador vende o par):
      ratio_ma3000 : mid do WIN - preço justo do pipeline (razão média de 3.000 ticks)
      ols3000      : resíduo do OLS rolante logWIN ~ logBOVA (janela 3.000)
      kalman       : inovação do filtro de Kalman [intercepto, hedge ratio]
      copula       : índice de desalinhamento acumulado da cópula gaussiana
                     (retornos de 100 ticks; marginais empíricas do treino)
      truth        : spread verdadeiro (só nos dados sintéticos)
    Métricas: Spearman(pref, -z) por pregão (média e IC bootstrap sobre pregões),
    concordância de sinal nos extremos, R² da regressão de pref nos sinais e a
    curva pref x z.
 B) RESPOSTA A IMPULSO (stateless). Degraus artificiais nos preços: WIN sobe/desce,
    BOVA11 sobe, ambos sobem. Um arbitrador reage à DIFERENÇA: pref cai quando só o
    WIN sobe, sobe quando só o BOVA11 sobe e quase não muda quando os dois sobem.
 C) PLACEBO DE EMPARELHAMENTO. O agente enxerga o BOVA11 adulterado (atrasado 10/100/
    1000 ticks, de outro pregão, congelado) mas o P/L é calculado com os preços
    verdadeiros. Se o lucro colapsa, ele depende da relação entre os ativos.
    Ressalva: entradas fora da distribuição também derrubam o lucro; leia junto de B.
 D) EVENTOS. O desvio de cada método clássico nas entradas do agente, orientado a
    favor da posição (positivo = o método também via a oportunidade), na entrada, 100
    ticks depois e na saída.

Uso (de src/; precisa dos dados de tick ou de SYNTHETIC=... como no treino):
    python diagnose_agent.py --run-dir runs/<tag>/fold1 --sets val --max-days 10
    python diagnose_agent.py --run-dir runs/<tag>/fold1 --tests A,B --plot
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

SYNTH_KEYS = {"kind": "SYNTH_KIND", "spread_std": "SYNTH_SPREAD_STD", "half_life": "SYNTH_HALF_LIFE",
              "drift": "SYNTH_DRIFT", "roll_every": "SYNTH_ROLL_EVERY", "roll_pts": "SYNTH_ROLL",
              "b_daily_vol": "SYNTH_B_DAILY_VOL", "day_gap_vol": "SYNTH_DAY_GAP_VOL"}

CLASSIC = ["ratio_ma3000", "ols3000", "kalman", "copula"]
_G = {}          # estado dos workers


# --------------------------------------------------------------------------- carga
def apply_metadata_env(meta):
    """Reproduz a configuração do treino (estado, env, sintético) via variáveis de ambiente."""
    os.environ["STATE_KIND"] = meta.get("state_kind", "spread")
    os.environ["ENV_KIND"] = meta.get("env_kind", "hedged")
    os.environ["RAW_LEVELS"] = str(meta.get("raw_levels", "1"))
    os.environ["RAW_DOY"] = str(meta.get("raw_doy", "0"))
    if meta.get("synthetic"):
        os.environ["SYNTHETIC"] = meta["synthetic"]
        for k, v in (meta.get("synth_cfg") or {}).items():
            os.environ[SYNTH_KEYS[k]] = str(v)


def orders_of(meta, which):
    if which == "val":
        return list(meta["val_orders"])
    if which == "test":
        return list(meta["test_orders"])
    tr = meta["train_orders"]
    ex = {int(x) for x in str(meta.get("exclude_orders", "")).split(",") if x.strip()}
    return [o for o in range(tr["first"], tr["last"] + 1) if o not in ex]


# --------------------------------------------------------------------------- sinais clássicos
def rolling_ols_resid(lw, lb, w, L=3000, min_n=500):
    n = len(lw)
    cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])
    Sx, Sy, Sxx, Sxy = cs(lb), cs(lw), cs(lb * lb), cs(lb * lw)
    t = np.arange(1, n + 1)
    s = np.maximum(t - L, 0)
    cnt = (t - s).astype(float)
    mx, my = (Sx[t] - Sx[s]) / cnt, (Sy[t] - Sy[s]) / cnt
    vxx = (Sxx[t] - Sxx[s]) / cnt - mx * mx
    cxy = (Sxy[t] - Sxy[s]) / cnt - mx * my
    beta = np.where(vxx > 1e-18, cxy / np.maximum(vxx, 1e-18), 1.0)
    alpha = my - beta * mx
    res = (lw - alpha - beta * lb) * w
    res[cnt < min_n] = np.nan
    return res


def kalman_resid(lw, lb, w, R=1e-8, qa=1e-15, qb=1e-12, burn=300):
    """Inovação do filtro de Kalman y=logW = a + b*(logB - logB0), [a, b] passeio aleatório."""
    n = len(lw)
    x = lb - lb[0]
    a, b = lw[0], 1.0
    p00, p01, p11 = 1e-8, 0.0, 1e-2
    out = np.empty(n)
    for t in range(n):
        p00 += qa
        p11 += qb
        e = lw[t] - (a + b * x[t])
        ph0 = p00 + p01 * x[t]
        ph1 = p01 + p11 * x[t]
        S = ph0 + x[t] * ph1 + R
        k0, k1 = ph0 / S, ph1 / S
        a += k0 * e
        b += k1 * e
        p00, p01, p11 = p00 - k0 * ph0, p01 - k0 * ph1, p11 - k1 * ph1
        out[t] = e * w[t]
    out[:burn] = np.nan
    return out


def fit_copula(dfs, h=100):
    rw, rb = [], []
    for df in dfs:
        lw = np.log(0.5 * (df["bid"].to_numpy(float) + df["ask"].to_numpy(float)))
        lb = np.log(0.5 * (df["bbid"].to_numpy(float) + df["bask"].to_numpy(float)))
        rw.append(lw[h::h] - lw[:-h:h])
        rb.append(lb[h::h] - lb[:-h:h])
    rw, rb = np.concatenate(rw), np.concatenate(rb)
    tau, _ = stats.kendalltau(rw, rb)
    rho = float(np.clip(np.sin(np.pi * tau / 2.0), 0.05, 0.99))
    return {"sw": np.sort(rw), "sb": np.sort(rb), "rho": rho, "h": h, "tau": float(tau)}


def _ecdf(sorted_ref, v):
    lo = np.searchsorted(sorted_ref, v, side="left")
    hi = np.searchsorted(sorted_ref, v, side="right")
    return np.clip((0.5 * (lo + hi) + 0.5) / (len(sorted_ref) + 1.0), 1e-4, 1 - 1e-4)


def copula_signal(df, cop, window=1000):
    """Soma móvel do índice de desalinhamento MI = P(U_W <= u_w | U_B = u_b) - 0.5 (Liew-Wu)."""
    h = cop["h"]
    lw = np.log(0.5 * (df["bid"].to_numpy(float) + df["ask"].to_numpy(float)))
    lb = np.log(0.5 * (df["bbid"].to_numpy(float) + df["bask"].to_numpy(float)))
    idx = np.arange(len(lw))
    j = np.maximum(idx - h, 0)
    zw = stats.norm.ppf(_ecdf(cop["sw"], lw - lw[j]))
    zb = stats.norm.ppf(_ecdf(cop["sb"], lb - lb[j]))
    r = cop["rho"]
    mi = stats.norm.cdf((zw - r * zb) / np.sqrt(1.0 - r * r)) - 0.5
    return pd.Series(mi).rolling(window, min_periods=200).sum().to_numpy()


def classical_signals(df, cop):
    """Sinais por tick, orientados: POSITIVO = WIN caro em relação ao BOVA11."""
    w = 0.5 * (df["bid"].to_numpy(float) + df["ask"].to_numpy(float))
    b = 0.5 * (df["bbid"].to_numpy(float) + df["bask"].to_numpy(float))
    lw, lb = np.log(w), np.log(b)
    sig = {"ratio_ma3000": w - 0.5 * (df["Wajusto"].to_numpy(float) + df["Wbjusto"].to_numpy(float)),
           "ols3000": rolling_ols_resid(lw, lb, w),
           "kalman": kalman_resid(lw, lb, w),
           "copula": copula_signal(df, cop)}
    if "true_x" in df.columns:
        sig["truth"] = df["true_x"].to_numpy(float)
    return sig


# --------------------------------------------------------------------------- política
def policy_probs(model, obs):
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), 8192):
            t = torch.as_tensor(obs[i:i + 8192], dtype=torch.float32)
            out.append(model.policy.get_distribution(t).distribution.probs.numpy())
    return np.concatenate(out)


def flat_obs(feat_scaled):
    """Linhas de observação com posição 0 e P/L 0."""
    z = np.zeros((len(feat_scaled), 2), dtype=np.float32)
    return np.concatenate([feat_scaled.astype(np.float32), z], axis=1)


def pref_of(probs):
    return probs[:, 1] - probs[:, 2]          # P(comprar par) - P(vender par)


def init_worker(run_dir, model_name, cop, fee):
    import rl_trading_pipeline as P
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    mean, std = np.load(os.path.join(run_dir, "scaler.npz")).values() if False else (None, None)
    z = np.load(os.path.join(run_dir, "scaler.npz"))
    _G.update(P=P, mean=z["mean"], std=z["std"], cop=cop, fee=fee,
              model=PPO.load(os.path.join(run_dir, f"ppo_{model_name}.zip"), device="cpu"))


def scaled(df):
    P = _G["P"]
    return P.apply_scaler(P.build_features(df), _G["mean"], _G["std"])


# --------------------------------------------------------------------------- A) alinhamento
def worker_align(order):
    P = _G["P"]
    df = P.process_day_cached(order)
    pref = pref_of(policy_probs(_G["model"], flat_obs(scaled(df))))
    sig = classical_signals(df, _G["cop"])
    return {"order": order, "pref": pref.astype(np.float32),
            "sig": {k: v.astype(np.float32) for k, v in sig.items()}}


def boot_mean_ci(x, n=4000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x))).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def align_report(days, out_dir, plot):
    names = [k for k in days[0]["sig"].keys()]
    rows, curve_rows = [], []
    per_day = {k: [] for k in names}
    for d in days:
        for k in names:
            m = np.isfinite(d["sig"][k])
            if m.sum() > 500 and np.std(d["pref"][m]) > 1e-9:
                per_day[k].append(stats.spearmanr(d["pref"][m], -d["sig"][k][m])[0])
            else:
                per_day[k].append(np.nan)
    pooled_pref = np.concatenate([d["pref"] for d in days])
    for k in names:
        z = np.concatenate([d["sig"][k] for d in days])
        m = np.isfinite(z)
        mu, lo, hi = boot_mean_ci(per_day[k])
        thr = np.nanstd(z)
        ext = m & (np.abs(z) > thr)
        conf = np.abs(pooled_pref) > 0.2
        agree = float(np.mean(np.sign(pooled_pref[ext & conf]) == -np.sign(z[ext & conf]))) if (ext & conf).any() else np.nan
        r2 = float(np.corrcoef(pooled_pref[m], -z[m])[0, 1] ** 2) if np.std(pooled_pref[m]) > 0 else np.nan
        rows.append({"sinal": k, "spearman_medio_por_pregao": mu, "ic95_inf": lo, "ic95_sup": hi,
                     "pregoes_positivos": f"{int(np.nansum(np.array(per_day[k]) > 0))}/{len(days)}",
                     "concordancia_de_sinal_nos_extremos": agree, "r2_linear": r2,
                     "pct_ticks_com_preferencia_forte": 100 * float(np.mean(conf))})
        # curva pref x z (quantis)
        q = pd.qcut(pd.Series(z[m]), 15, duplicates="drop")
        g = pd.DataFrame({"z": z[m], "pref": pooled_pref[m], "q": q.values}).groupby("q", observed=True).mean()
        for zi, pi in zip(g["z"], g["pref"]):
            curve_rows.append({"sinal": k, "z_medio_pts": zi, "pref_media": pi})
    # correlação entre os próprios sinais (se são muito parecidos, o alinhamento não distingue os métodos)
    cm = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            vals = []
            for d in days:
                m = np.isfinite(d["sig"][a]) & np.isfinite(d["sig"][b])
                if m.sum() > 500:
                    vals.append(stats.spearmanr(d["sig"][a][m], d["sig"][b][m])[0])
            cm.loc[a, b] = np.nanmean(vals) if vals else np.nan
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "alignment.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(out_dir, "alignment_curve.csv"), index=False)
    cm.to_csv(os.path.join(out_dir, "signal_correlation.csv"))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = pd.DataFrame(curve_rows)
        fig, axs = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.2), sharey=True)
        for ax, k in zip(np.atleast_1d(axs), names):
            s = c[c.sinal == k]
            ax.plot(s.z_medio_pts, s.pref_media, "o-", color="#2a78d6", ms=3)
            ax.axhline(0, color="#52514e", lw=0.8); ax.axvline(0, color="#52514e", lw=0.8)
            ax.set_title(k); ax.set_xlabel("desvio (pts; + = WIN caro)")
        np.atleast_1d(axs)[0].set_ylabel("preferência do agente (comprar − vender par)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "alignment_curve.png"), dpi=130); plt.close(fig)
    return df, cm


# --------------------------------------------------------------------------- B) impulso
def worker_impulse(args):
    order, deltas, n_t0, win, seed = args
    P = _G["P"]
    df = P.process_day_cached(order)
    n = len(df)
    base = pref_of(policy_probs(_G["model"], flat_obs(scaled(df))))
    rng = np.random.default_rng(seed + order)
    t0s = rng.integers(1500, n - win - 5, size=n_t0)
    wmid = 0.5 * (df["bid"].to_numpy(float) + df["ask"].to_numpy(float))
    out = []
    for t0 in t0s:
        for delta in deltas:
            for scen in ("win_up", "win_down", "bova_up", "ambos_sobem"):
                mult = 1.0 + delta / wmid[t0]
                d2 = df.copy()
                sl = slice(int(t0), None)
                if scen == "win_up":
                    d2.loc[sl, ["bid", "ask"]] *= mult
                elif scen == "win_down":
                    d2.loc[sl, ["bid", "ask"]] /= mult
                elif scen == "bova_up":
                    d2.loc[sl, ["bbid", "bask"]] *= mult
                else:
                    d2.loc[sl, ["bid", "ask", "bbid", "bask"]] *= mult
                f2 = scaled(d2)[t0 + 1:t0 + 1 + win]
                p2 = pref_of(policy_probs(_G["model"], flat_obs(f2)))
                out.append({"order": order, "t0": int(t0), "delta_pts": delta, "cenario": scen,
                            "delta_pref": float(p2.mean() - base[t0 + 1:t0 + 1 + win].mean())})
    return out


def impulse_report(rows, out_dir):
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out_dir, "impulse_raw.csv"), index=False)
    res = []
    for (delta, scen), g in d.groupby(["delta_pts", "cenario"]):
        by_day = g.groupby("order")["delta_pref"].mean()
        mu, lo, hi = boot_mean_ci(by_day.to_numpy())
        res.append({"delta_pts": delta, "cenario": scen, "delta_pref_medio": mu, "ic95_inf": lo, "ic95_sup": hi,
                    "esperado_arbitragem": {"win_up": "negativo", "win_down": "positivo", "bova_up": "positivo",
                                            "ambos_sobem": "~0"}[scen]})
    out = pd.DataFrame(res)
    out.to_csv(os.path.join(out_dir, "impulse.csv"), index=False)
    return out


# --------------------------------------------------------------------------- C/D) placebo e eventos
def perturb(df, kind, other=None):
    d = df.copy()
    n = len(d)
    idx = np.arange(n)
    if kind == "original":
        return d
    if kind.startswith("bova_lag"):
        j = np.maximum(idx - int(kind.replace("bova_lag", "")), 0)
        for c in ("bbid", "bask"):
            d[c] = df[c].to_numpy()[j]
    elif kind.startswith("win_lag"):
        j = np.maximum(idx - int(kind.replace("win_lag", "")), 0)
        for c in ("bid", "ask"):
            d[c] = df[c].to_numpy()[j]
    elif kind == "bova_other_day":
        m = np.minimum((idx * len(other) / n).astype(int), len(other) - 1)
        sc = float(df["bbid"].iloc[0]) / float(other["bbid"].iloc[0])
        for c in ("bbid", "bask"):
            d[c] = other[c].to_numpy()[m] * sc
    elif kind == "bova_frozen":
        for c in ("bbid", "bask"):
            d[c] = float(df[c].iloc[0])
    else:
        raise ValueError(kind)
    return d


def worker_placebo(args):
    order, kind, other_order = args
    P = _G["P"]
    df = P.process_day_cached(order)
    other = P.process_day_cached(other_order) if kind == "bova_other_day" else None
    d_obs = perturb(df, kind, other)
    feat = P.apply_scaler(P.build_features(d_obs), _G["mean"], _G["std"])
    bench = P.build_bench_features(df) if P.state_kind() == "raw" else None
    env = P._env_class()(df, feat, transaction_fee=_G["fee"], track_details=(kind == "original"), bench=bench)
    model = _G["model"]
    obs, _ = env.reset()
    done = False
    while not done:
        action = int(model.predict(obs, deterministic=True)[0])
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    res = {"order": order, "kind": kind, "pnl": float(env.total_reward), "trades": int(env.n_trades_closed),
           "gross": float(env.gross_mtm)}
    if kind == "original":
        sig = classical_signals(df, _G["cop"])
        ev = []
        for tr in env.trade_details:
            direction, o, c = int(tr[0]), int(tr[1]), int(tr[2])
            row = {"order": order, "direction": direction, "entry": o, "exit": c, "pnl": float(tr[3])}
            for k, z in sig.items():
                f = -direction * z                        # + = o método também via a oportunidade
                row[f"{k}_entrada"] = f[o]
                row[f"{k}_mais100"] = f[min(o + 100, len(f) - 1)]
                row[f"{k}_saida"] = f[min(c, len(f) - 1)]
            ev.append(row)
        res["events"] = ev
    return res


def placebo_report(results, out_dir):
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "events"} for r in results])
    g = d.groupby("kind").agg(pnl_total=("pnl", "sum"), pnl_medio_dia=("pnl", "mean"), negocios_dia=("trades", "mean"),
                              bruto_total=("gross", "sum")).reset_index()
    base = float(g.loc[g.kind == "original", "pnl_total"].iloc[0])
    g["pnl_relativo_ao_original"] = g["pnl_total"] / base if base != 0 else np.nan
    g.to_csv(os.path.join(out_dir, "placebo.csv"), index=False)
    ev = pd.DataFrame([e for r in results if "events" in r for e in r["events"]])
    erows = []
    if len(ev):
        ev.to_csv(os.path.join(out_dir, "events_raw.csv"), index=False)
        for k in [c[:-8] for c in ev.columns if c.endswith("_entrada")]:
            e0, e1, e2 = ev[f"{k}_entrada"], ev[f"{k}_mais100"], ev[f"{k}_saida"]
            erows.append({"sinal": k, "negocios": len(ev), "desvio_a_favor_na_entrada_pts": float(e0.mean()),
                          "pct_entradas_a_favor": 100 * float((e0 > 0).mean()),
                          "desvio_a_favor_100_ticks_depois": float(e1.mean()), "desvio_a_favor_na_saida": float(e2.mean()),
                          "corr_entrada_x_pnl": float(np.corrcoef(e0.fillna(0), ev["pnl"])[0, 1])})
        pd.DataFrame(erows).to_csv(os.path.join(out_dir, "events.csv"), index=False)
    return g, pd.DataFrame(erows)


# --------------------------------------------------------------------------- main
def fmt(df, nd=3):
    return df.to_string(index=False, float_format=lambda v: f"{v:.{nd}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="best_val", choices=["best_val", "last"])
    ap.add_argument("--sets", default="val", help="val, test ou val,test")
    ap.add_argument("--max-days", type=int, default=20)
    ap.add_argument("--tests", default="A,B,C,D")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "3")))
    ap.add_argument("--copula-days", type=int, default=30)
    ap.add_argument("--impulse-t0", type=int, default=3)
    ap.add_argument("--impulse-deltas", default="30,60")
    ap.add_argument("--placebos", default="original,bova_lag10,bova_lag100,bova_lag1000,bova_other_day,bova_frozen,win_lag100")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.run_dir, "metadata.json")))
    apply_metadata_env(meta)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rl_trading_pipeline as P

    out_dir = os.path.join(a.run_dir, "diag")
    os.makedirs(out_dir, exist_ok=True)
    tests = set(a.tests.split(","))
    fee = float(meta.get("transaction_fee", 1.0))
    orders = []
    for s in a.sets.split(","):
        o = orders_of(meta, s.strip())
        orders += o[:a.max_days] if a.max_days else o
    print(f"Diagnóstico de {a.run_dir} ({a.model}); estado={P.state_kind()}, env={os.environ['ENV_KIND']}, "
          f"{len(orders)} pregões ({a.sets}); testes {sorted(tests)}")

    tr = orders_of(meta, "train")
    step = max(1, len(tr) // a.copula_days)
    cop = fit_copula([P.process_day_cached(o) for o in tr[::step][:a.copula_days]])
    print(f"cópula gaussiana: tau de Kendall {cop['tau']:.3f} -> rho {cop['rho']:.3f} (retornos de {cop['h']} ticks)")

    n_w = max(1, min(a.workers, len(orders)))
    ctx = P._mp_context()
    summary = [f"# Diagnóstico de arbitragem — {a.run_dir} ({a.model})\n",
               f"Pregões: {len(orders)} ({a.sets}); estado `{P.state_kind()}`; cópula rho={cop['rho']:.3f}\n"]

    def pool():
        return ctx.Pool(n_w, initializer=init_worker, initargs=(a.run_dir, a.model, cop, fee))

    if "A" in tests:
        with pool() as p:
            days = p.map(worker_align, orders, chunksize=1)
        al, cm = align_report(days, out_dir, a.plot)
        print("\n=== A) Alinhamento da preferência do agente com métodos clássicos ===")
        print("(Spearman entre pref = P(comprar par) − P(vender par) e −desvio; +1 = arbitragem perfeita, 0 = sem relação)")
        print(fmt(al))
        print("\nCorrelação (Spearman) entre os próprios sinais:")
        print(cm.round(2).to_string())
        summary += ["\n## A) Alinhamento\n", fmt(al)]
    if "B" in tests:
        deltas = [float(x) for x in a.impulse_deltas.split(",")]
        with pool() as p:
            rows = [r for lst in p.map(worker_impulse, [(o, deltas, a.impulse_t0, 300, 0) for o in orders], chunksize=1) for r in lst]
        imp = impulse_report(rows, out_dir)
        print("\n=== B) Resposta a impulso (variação da preferência com o degrau; média nos 300 ticks seguintes) ===")
        print(fmt(imp))
        summary += ["\n## B) Impulso\n", fmt(imp)]
    if "C" in tests or "D" in tests:
        kinds = [k for k in a.placebos.split(",") if k == "original" or "C" in tests]
        rng = np.random.default_rng(0)
        jobs = []
        for i, o in enumerate(orders):
            other = orders[(i + max(1, len(orders) // 2)) % len(orders)] if len(orders) > 1 else o      # outro pregão
            for k in kinds:
                jobs.append((o, k, other))
        with pool() as p:
            res = p.map(worker_placebo, jobs, chunksize=1)
        g, ev = placebo_report(res, out_dir)
        if "C" in tests:
            print("\n=== C) Placebo de emparelhamento (o agente vê o BOVA11 adulterado; o P/L usa preços verdadeiros) ===")
            print(fmt(g, 2))
            summary += ["\n## C) Placebo\n", fmt(g, 2)]
        if len(ev):
            print("\n=== D) Desvio dos métodos clássicos nas entradas do agente (orientado a favor da posição) ===")
            print(fmt(ev))
            summary += ["\n## D) Eventos\n", fmt(ev)]
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(summary) + "\n")
    print(f"\nArquivos em {out_dir}/")


if __name__ == "__main__":
    main()
