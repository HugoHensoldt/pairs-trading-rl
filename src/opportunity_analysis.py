"""
Análise de OPORTUNIDADES no dataset, antes de treinar qualquer agente.

Pergunta: dados os custos do par hedgeado (WIN + BOVA11), existem momentos em
que o movimento do par supera os custos? Quantos, e dá para prevê-los com o
que o agente enxerga?

Três partes (todas usam a MESMA contabilidade do HedgedPairEnv: N_BOVA fixo na
abertura, custos 0,25 + 0,000230*V_BOVA por lado, P/L do par em R$):

1) ORÁCULO (em retrospecto, usa o futuro -- NÃO é regra operável):
   - por tick de entrada o: melhor P/L líquido saindo em qualquer c em (o, o+H];
   - programação dinâmica: melhor conjunto de negócios NÃO sobrepostos do dia
     (teto de P/L com previsão perfeita), horas e duração dos negócios.
   Para o par, P/L(o,c) = |0,20*(W_c-W_o) - N_o*(B_c-B_o)| - custos(o,c): a
   direção ótima é a do sinal do movimento, e os custos não dependem dela.

2) REGRA CAUSAL "spread menos custos": entra quando o spread supera o custo de
   ida e volta em pontos de WIN (custo_pts = custo_R$ / 0,20, ~58 pts), sai
   quando o spread volta a 0 ou no timeout. Só usa informação do tick atual.

3) APRENDIZAGEM: um classificador (gradient boosting) com features CAUSAIS
   tenta prever "existe oportunidade líquida > 0 nos próximos H ticks". AUC ~0,5
   => o que o agente vê não permite antecipar as oportunidades.

Rótulos por tick (best_net, best_dir, oracle_pos) são gravados em
<out>/labels_day<N>.npz. ATENÇÃO: usam o futuro. Servem de ALVO auxiliar /
imitação / análise, nunca como feature de entrada (vazamento).

Uso (de src/):
    python opportunity_analysis.py --first 1 --last 60 --train-days 40
    python opportunity_analysis.py --first 1 --last 60 --spread-cost   # custos + meio-spread
"""

import argparse
import os

import numpy as np
import pandas as pd

from config import POINT_VALUE_BRL as PV
from rl_trading_pipeline import (
    process_day_cached, HEDGE_FACTOR, WIN_COST_PER_SIDE_BRL, BOVA_COST_PCT_PER_SIDE,
    _mp_context,
)


def day_arrays(order):
    df = process_day_cached(order)
    a = {c: df[c].to_numpy(dtype=float) for c in ("bid", "ask", "bbid", "bask", "Wbjusto", "Wajusto")}
    a["W"] = (a["bid"] + a["ask"]) / 2
    a["B"] = (a["bbid"] + a["bask"]) / 2
    a["hour"] = pd.to_datetime(df["datahora"]).dt.hour.to_numpy()
    a["order"] = order
    return a


def _open_cost(a, N, spread_cost):
    co = WIN_COST_PER_SIDE_BRL + BOVA_COST_PCT_PER_SIDE * (N * a["B"])
    if spread_cost:
        co = co + 0.5 * (a["ask"] - a["bid"]) * PV + 0.5 * (a["bask"] - a["bbid"]) * N
    return co


def _close_cost(a, N_o, idx, spread_cost):
    """Custo de fechar em ticks `idx` um par aberto com N_o ações de BOVA."""
    cc = WIN_COST_PER_SIDE_BRL + BOVA_COST_PCT_PER_SIDE * (N_o * a["B"][idx])
    if spread_cost:
        cc = cc + (0.5 * (a["ask"][idx] - a["bid"][idx]) * PV
                   + 0.5 * (a["bask"][idx] - a["bbid"][idx]) * N_o)
    return cc


def oracle_day(a, H, spread_cost):
    """Melhor saída por entrada (best_net/best_dir) e DP de negócios não
    sobrepostos (oracle_pos, trades)."""
    W, B = a["W"], a["B"]
    n = len(W)
    N = W / (HEDGE_FACTOR * B)
    co = _open_cost(a, N, spread_cost)

    best_net = np.full(n, np.nan)
    best_dir = np.zeros(n, dtype=np.int8)
    for o in range(n - 1):
        idx = np.arange(o + 1, min(n, o + H + 1))
        term1 = PV * (W[idx] - W[o]) - N[o] * (B[idx] - B[o])
        net = np.abs(term1) - co[o] - _close_cost(a, N[o], idx, spread_cost)
        k = int(np.argmax(net))
        best_net[o] = net[k]
        best_dir[o] = 1 if term1[k] > 0 else -1

    # DP: best[c] = melhor P/L acumulado até c com o par FLAT em c
    best = np.zeros(n)
    choice = np.full(n, -1, dtype=np.int64)      # -1 = carrega best[c-1]; senão, entrada o
    sign = np.zeros(n, dtype=np.int8)
    for c in range(1, n):
        best[c] = best[c - 1]
        lo = max(0, c - H)
        o = np.arange(lo, c)
        term1 = PV * (W[c] - W[o]) - N[o] * (B[c] - B[o])
        net = np.abs(term1) - co[o] - _close_cost_vec(a, N[o], c, spread_cost)
        val = best[o] + net
        k = int(np.argmax(val))
        if val[k] > best[c]:
            best[c] = val[k]
            choice[c] = o[k]
            sign[c] = 1 if term1[k] > 0 else -1

    oracle_pos = np.zeros(n, dtype=np.int8)
    trades = []
    c = n - 1
    while c > 0:
        if choice[c] == -1:
            c -= 1
            continue
        o = int(choice[c])
        oracle_pos[o:c] = sign[c]
        trades.append((o, c, int(sign[c]), float(best[c] - best[o]), a["hour"][o]))
        c = o
    trades.reverse()
    return best_net, best_dir, oracle_pos, trades, float(best[-1])


def _close_cost_vec(a, N_o_vec, c, spread_cost):
    """Custo de fechar no tick c pares abertos em vários o (N_o vetor)."""
    cc = WIN_COST_PER_SIDE_BRL + BOVA_COST_PCT_PER_SIDE * (N_o_vec * a["B"][c])
    if spread_cost:
        cc = cc + (0.5 * (a["ask"][c] - a["bid"][c]) * PV
                   + 0.5 * (a["bask"][c] - a["bbid"][c]) * N_o_vec)
    return cc


def cost_points(a):
    """Custo de ida e volta do par (sem meio-spread) em PONTOS de WIN, por tick."""
    return (2 * WIN_COST_PER_SIDE_BRL + 2 * BOVA_COST_PCT_PER_SIDE * a["W"] / HEDGE_FACTOR) / PV


def causal_rule_day(a, margin, timeout, spread_cost):
    """Regra causal 'spread menos custos'. Retorna lista de P/L líquidos (R$)."""
    W, B = a["W"], a["B"]
    n = len(W)
    s_buy = a["ask"] - a["Wbjusto"]          # comprar WIN (e vender BOVA)
    s_sell = a["bid"] - a["Wajusto"]         # vender WIN (e comprar BOVA)
    thr = margin * cost_points(a)
    N = W / (HEDGE_FACTOR * B)
    co = _open_cost(a, N, spread_cost)
    pnls, pos, o = [], 0, 0
    for t in range(n):
        if pos == 0:
            if s_buy[t] < -thr[t]:
                pos, o = 1, t
            elif s_sell[t] > thr[t]:
                pos, o = -1, t
        else:
            revert = (s_buy[t] >= 0) if pos == 1 else (s_sell[t] <= 0)
            if revert or t - o >= timeout or t == n - 1:
                term1 = PV * (W[t] - W[o]) - N[o] * (B[t] - B[o])
                cc = _close_cost(a, N[o], np.array([t]), spread_cost)[0]
                pnls.append(pos * term1 - co[o] - cc)
                pos = 0
    return pnls


def causal_features(a):
    """Features CAUSAIS por tick (só passado/presente)."""
    s_buy = a["ask"] - a["Wbjusto"]
    s_sell = a["bid"] - a["Wajusto"]
    thr = cost_points(a)
    cols = {
        "s_buy": s_buy, "s_sell": s_sell,
        "s_buy_over_cost": s_buy / thr, "s_sell_over_cost": s_sell / thr,
        "win_width": a["ask"] - a["bid"], "bova_width": a["bask"] - a["bbid"],
        "hour": a["hour"].astype(float),
    }
    for lag in (10, 100):
        for name, s in (("s_buy", s_buy), ("s_sell", s_sell)):
            d = np.zeros_like(s)
            d[lag:] = s[lag:] - s[:-lag]
            cols[f"{name}_d{lag}"] = d
    return pd.DataFrame(cols)


def analyze_day(args):
    order, H, spread_cost, margins, timeout, out_dir = args
    a = day_arrays(order)
    best_net, best_dir, oracle_pos, trades, oracle_pnl = oracle_day(a, H, spread_cost)
    ok = ~np.isnan(best_net)
    thr = cost_points(a)
    rules = {}
    for m in margins:
        p = np.array(causal_rule_day(a, m, timeout, spread_cost))
        rules[m] = {"n": len(p), "net": float(p.sum()) if len(p) else 0.0,
                    "win": float((p > 0).mean()) if len(p) else np.nan}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, f"labels_day{order}.npz"),
                            best_net=best_net.astype(np.float32), best_dir=best_dir,
                            oracle_pos=oracle_pos)
    feats = causal_features(a)
    feats["best_net"] = best_net
    feats["order"] = order
    return {
        "order": order, "n_ticks": len(a["W"]),
        "oracle_pnl": oracle_pnl, "oracle_trades": len(trades),
        "oracle_trade_nets": [t[3] for t in trades],
        "oracle_durations": [t[1] - t[0] for t in trades],
        "oracle_hours": [t[4] for t in trades],
        "frac_entry_gt0": float((best_net[ok] > 0).mean()),
        "frac_entry_gt2": float((best_net[ok] > 2).mean()),
        "frac_entry_gt5": float((best_net[ok] > 5).mean()),
        "best_net_mean": float(best_net[ok].mean()),
        "cost_pts_mean": float(thr.mean()),
        "frac_buy_signal": float(((a["ask"] - a["Wbjusto"]) < -thr).mean()),
        "frac_sell_signal": float(((a["bid"] - a["Wajusto"]) > thr).mean()),
        "abs_spread_p50": float(np.median(np.abs(a["ask"] - a["Wbjusto"]))),
        "abs_spread_p99": float(np.percentile(np.abs(a["ask"] - a["Wbjusto"]), 99)),
        "rules": rules, "feats": feats,
    }


def learnability(results, train_days, H):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    feats = pd.concat([r["feats"] for r in results], ignore_index=True).dropna(subset=["best_net"])
    orders = sorted(feats["order"].unique())
    tr_orders, te_orders = orders[:train_days], orders[train_days:]
    tr = feats[feats.order.isin(tr_orders)].iloc[::5]           # subamostra: ticks vizinhos são quase iguais
    te = feats[feats.order.isin(te_orders)]
    cols = [c for c in feats.columns if c not in ("best_net", "order")]
    out = []
    for margin in (0.0, 5.0):                                   # oportunidade > 0 e > R$ 5
        ytr, yte = (tr.best_net > margin).astype(int), (te.best_net > margin).astype(int)
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0)
        clf.fit(tr[cols], ytr)
        p = clf.predict_proba(te[cols])[:, 1]
        auc = roc_auc_score(yte, p)
        top = te.assign(p=p).nlargest(max(1, len(te) // 20), "p")      # 5% de ticks com maior score
        out.append({"limiar_R$": margin, "base_rate": float(yte.mean()),
                    "AUC_teste": float(auc),
                    "P(oport.) no top5%": float((top.best_net > margin).mean()),
                    "best_net médio geral": float(te.best_net.mean()),
                    "best_net médio top5%": float(top.best_net.mean())})
    return pd.DataFrame(out), len(tr_orders), len(te_orders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=2000, help="máx. ticks de holding (~40 min)")
    ap.add_argument("--train-days", type=int, default=40)
    ap.add_argument("--spread-cost", action="store_true", help="soma meio-spread bid/ask das duas pernas")
    ap.add_argument("--timeout", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.environ.get("RUNS_DIR", "runs"), "opportunities"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "3")))
    args = ap.parse_args()

    orders = list(range(args.first, args.last + 1))
    margins = (1.0, 1.5, 2.0)
    jobs = [(o, args.horizon, args.spread_cost, margins, args.timeout, args.out) for o in orders]
    if args.workers > 1:
        with _mp_context().Pool(args.workers) as pool:
            results = pool.map(analyze_day, jobs, chunksize=1)
    else:
        results = [analyze_day(j) for j in jobs]

    days = len(results)
    cst = "COM meio-spread bid/ask" if args.spread_cost else "só as taxas da especificação"
    print(f"\n=== Oportunidades, pregões {args.first}..{args.last} ({days} dias), custos: {cst}, "
          f"horizonte {args.horizon} ticks ===")
    print(f"custo de ida e volta ≈ {np.mean([r['cost_pts_mean'] for r in results]):.1f} pontos de WIN "
          f"(≈ R$ {np.mean([r['cost_pts_mean'] for r in results]) * PV:.2f})")
    print(f"|spread| (ask_WIN - Wbjusto): mediana {np.mean([r['abs_spread_p50'] for r in results]):.1f} pts, "
          f"p99 {np.mean([r['abs_spread_p99'] for r in results]):.1f} pts")
    print(f"ticks com sinal 'spread < -custo' (compra): {np.mean([r['frac_buy_signal'] for r in results]):.3%}  |  "
          f"'spread > +custo' (venda): {np.mean([r['frac_sell_signal'] for r in results]):.3%}")

    print("\n-- 1) ORÁCULO (previsão perfeita, NÃO operável) --")
    print(f"ticks de entrada com alguma saída líquida > 0 / > R$2 / > R$5 (dentro de {args.horizon} ticks): "
          f"{np.mean([r['frac_entry_gt0'] for r in results]):.1%} / "
          f"{np.mean([r['frac_entry_gt2'] for r in results]):.1%} / "
          f"{np.mean([r['frac_entry_gt5'] for r in results]):.1%}")
    op = np.array([r["oracle_pnl"] for r in results])
    nets = np.array([x for r in results for x in r["oracle_trade_nets"]])
    dur = np.array([x for r in results for x in r["oracle_durations"]])
    print(f"teto de P/L por dia com previsão perfeita: média R$ {op.mean():.1f} "
          f"(mín {op.min():.1f}, máx {op.max():.1f}); negócios/dia: "
          f"{np.mean([r['oracle_trades'] for r in results]):.1f}; "
          f"líquido médio por negócio R$ {nets.mean() if len(nets) else 0:.2f}; "
          f"duração mediana {np.median(dur) if len(dur) else 0:.0f} ticks")
    hrs = pd.Series([h for r in results for h in r["oracle_hours"]]).value_counts(normalize=True).sort_index()
    print("horas dos negócios do oráculo: " + "  ".join(f"{h}h {v:.0%}" for h, v in hrs.items()))

    print("\n-- 2) REGRA CAUSAL 'spread - custos' (entra se |spread| > margem x custo; sai ao voltar a 0 / timeout) --")
    for m in margins:
        n = sum(r["rules"][m]["n"] for r in results)
        net = sum(r["rules"][m]["net"] for r in results)
        wins = [r["rules"][m]["win"] for r in results if r["rules"][m]["n"]]
        print(f"margem {m:.1f}x: {n/days:6.1f} negócios/dia | P/L líquido total R$ {net:9.1f} "
              f"({net/days:7.1f}/dia) | acerto médio {np.mean(wins) if wins else float('nan'):.1%}")

    print("\n-- 3) APRENDIZAGEM: as features causais antecipam as oportunidades? --")
    if days > args.train_days + 2:
        tab, ntr, nte = learnability(results, args.train_days, args.horizon)
        print(f"treino em {ntr} dias, teste em {nte} dias seguintes")
        with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:.3f}"):
            print(tab.to_string(index=False))
    else:
        print("poucos dias para separar treino/teste (aumente --last)")

    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ("feats", "rules", "oracle_trade_nets",
                                                                     "oracle_durations", "oracle_hours")}
                            for r in results])
    os.makedirs(args.out, exist_ok=True)
    suffix = "_spreadcost" if args.spread_cost else ""
    summary.to_csv(os.path.join(args.out, f"summary{suffix}.csv"), index=False)
    print(f"\nrótulos por tick em {args.out}/labels_day<N>.npz; resumo em summary{suffix}.csv")


if __name__ == "__main__":
    main()
