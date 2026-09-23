"""
Avaliação das variantes da LSTM agrupada com as métricas que importam (P/L por janela e AUC DENTRO
da combinação), não a AUC global:

  auc_intra    AUC média dentro de cada combinação (sigma, Re, Ri), ponderada por n; 0,5 = acaso
  uplift_q     P/L médio das oportunidades no topo q% de probabilidade DENTRO de cada combinação menos o
               P/L médio de aceitar tudo (o mesmo mix de combinações nos dois lados): mede só o valor
               do modelo dentro da combinação, sem o atalho "combinação -> taxa" nem o regime da janela
  regra        aceitar se p_adj >= Ri/(Re+Ri) + m, com m escolhida SÓ na validação; P/L bruto e líquido
  corr_y/corr_pl  correlação (média entre combinações) da probabilidade com o acerto e com o |P/L|:
               se corr_pl >> corr_y, o modelo prefere trades GRANDES, não trades que acertam mais

p_adj = p bruta, ou, quando o modelo foi treinado com pesos 50/50 por combinação (predictions.npz traz
`combo_rate_train`), a probabilidade corrigida do deslocamento de prior: logit(p_adj) = logit(p) +
logit(taxa_de_lucro_da_combinação_no_treino) -- só isso devolve uma probabilidade comparável ao
equilíbrio Ri/(Re+Ri).

As janelas são as duas metades do teste (as mesmas das análises anteriores). Só faz sentido para
tags treinadas com o split atual (pregões/teste definidos por lstm_pipeline.split_orders()).

Uso:
  python src/analyze_lstm_variant.py --runs-dir sdumont_backup_lstm/lstm_runs \
      --pl-dir sdumont_backup_lstm/lstm_pl_cache --tags lstm_v2_base lstm_v3_balnosl --out-prefix docs/lstm_v3
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pl_oportunidades import PLReader  # noqa: E402
import lstm_pipeline as L  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MARGINS = np.round(np.arange(0.0, 0.1501, 0.01), 2)
MIN_ACEITAS_VAL = 150
QS = (0.2, 0.5)


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(y) > 1 and len(np.unique(y)) == 2 else float('nan')


def intra_auc(y, p, combo, mask=None):
    num = den = 0.0
    for c in np.unique(combo):
        m = combo == c
        if mask is not None:
            m = m & mask
        a = _auc(y[m], p[m])
        if not np.isnan(a):
            num += a * m.sum()
            den += m.sum()
    return num / den if den else float('nan')


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def adjust(p, combo, rate):
    """p_adj com correção de prior (só quando há combo_rate_train)."""
    if rate is None:
        return p
    r = np.clip(rate[combo], 1e-3, 1 - 1e-3)
    return 1.0 / (1.0 + np.exp(-(logit(p) + logit(r))))


def top_q_mask(score, combo, q, mask):
    """Seleciona, DENTRO de cada combinação (e da janela), a fração q de maior score."""
    sel = np.zeros(len(score), bool)
    for c in np.unique(combo):
        idx = np.flatnonzero((combo == c) & mask)
        if len(idx) == 0:
            continue
        k = max(int(round(q * len(idx))), 1)
        sel[idx[np.argsort(-score[idx], kind='stable')[:k]]] = True
    return sel


def within_corr(p, y, pl, combo, min_n=100):
    cy, cp = [], []
    for c in np.unique(combo):
        m = combo == c
        if m.sum() < min_n or p[m].std() == 0:
            continue
        if y[m].std() > 0:
            cy.append(np.corrcoef(p[m], y[m])[0, 1])
        if np.abs(pl[m]).std() > 0:
            cp.append(np.corrcoef(p[m], np.abs(pl[m]))[0, 1])
    return (float(np.mean(cy)) if cy else np.nan), (float(np.mean(cp)) if cp else np.nan)


def choose_margin(score, combo, pl, pstar, cost):
    best = None
    for m in MARGINS:
        a = score >= pstar[combo] + m
        if a.sum() < MIN_ACEITAS_VAL:
            continue
        v = float((pl[a] - cost).mean())
        if best is None or v > best[1] + 1e-12:
            best = (float(m), v)
    return best[0] if best else float('nan')


def analyze_task(reader, runs_dir, tag, fold, side, cost):
    z = np.load(Path(runs_dir) / tag / f"fold{fold}_{side}" / "predictions.npz")
    grid = [tuple(map(float, c)) for c in z['grid']]
    dev, test = L.split_orders()
    _, va = L.fold_orders(dev, fold)
    pstar = np.array([ri / (re + ri) for _, re, ri in grid])
    rate = z['combo_rate_train'] if 'combo_rate_train' in z.files else None
    data = {}
    for split, orders in (('val', va), ('test', test)):
        pl, ords, cmbs = reader.assemble(orders, grid, side)
        y = z[f'y_{split}']
        if not (len(pl) == len(y) and np.array_equal((pl > 0).astype(np.float32), y.astype(np.float32))
                and np.array_equal(ords, z[f'order_{split}']) and np.array_equal(cmbs, z[f'combo_{split}'])):
            raise RuntimeError(f"P/L desalinhado de predictions.npz: {tag} fold{fold} {side} {split}")
        data[split] = dict(y=y, p=z[f'p_{split}'], pl=pl, ord=ords, cmb=cmbs)
    v, t = data['val'], data['test']
    v['padj'], t['padj'] = adjust(v['p'], v['cmb'], rate), adjust(t['p'], t['cmb'], rate)
    marg = choose_margin(v['padj'], v['cmb'], v['pl'], pstar, cost)
    u = np.unique(t['ord'])
    half = len(u) // 2
    windows = [('teste inteiro', np.ones(len(t['ord']), bool)),
               (f'{u[0]}–{u[half - 1]}', t['ord'] <= u[half - 1]),
               (f'{u[half]}–{u[-1]}', t['ord'] >= u[half])]
    corr_y, corr_pl = within_corr(t['p'], t['y'], t['pl'], t['cmb'])
    rows = []
    for wname, w in windows:
        base_pl = t['pl'][w]
        row = dict(tag=tag, fold=fold, side=side, janela=wname, n=int(w.sum()),
                   auc_intra=intra_auc(t['y'], t['p'], t['cmb'], w),
                   auc_global=_auc(t['y'][w], t['p'][w]),
                   pl_tudo=float(base_pl.mean()), corr_y=corr_y, corr_pl=corr_pl,
                   balanceado=rate is not None, margem=marg)
        for q in QS:
            sel = top_q_mask(t['p'], t['cmb'], q, w)
            row[f'pl_top{int(q * 100)}'] = float(t['pl'][sel].mean())
            row[f'uplift_top{int(q * 100)}'] = float(t['pl'][sel].mean() - base_pl.mean())
        if not np.isnan(marg):
            a = (t['padj'] >= pstar[t['cmb']] + marg) & w
            row.update(n_regra=int(a.sum()), pl_regra=float(t['pl'][a].mean()) if a.any() else np.nan,
                       pl_regra_liq=float(t['pl'][a].mean() - cost) if a.any() else np.nan,
                       acerto_regra=float((t['pl'][a] > 0).mean()) if a.any() else np.nan)
        rows.append(row)
    return rows


def md_table(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("—" if isinstance(v, float) and np.isnan(v) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="sdumont_backup_lstm/lstm_runs")
    ap.add_argument("--pl-dir", default="sdumont_backup_lstm/lstm_pl_cache")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--custo", type=float, default=2.5, help="custo por trade (pts) no P/L líquido")
    ap.add_argument("--out-prefix", default="docs/lstm_v3")
    a = ap.parse_args()

    reader = PLReader(a.pl_dir)
    rows = []
    for tag in a.tags:
        for fold in range(1, L.N_SPLITS + 1):
            for side in L.SIDES:
                if not (Path(a.runs_dir) / tag / f"fold{fold}_{side}" / "predictions.npz").exists():
                    print(f"  (sem predictions.npz: {tag} fold{fold} {side})")
                    continue
                rows += analyze_task(reader, a.runs_dir, tag, fold, side, a.custo)
        print(f"{tag}: ok", flush=True)
    df = pd.DataFrame(rows)
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{a.out_prefix}_por_tarefa.csv", index=False)

    agg = {c: (c, 'mean') for c in ('n', 'auc_intra', 'auc_global', 'pl_tudo', 'pl_top20', 'uplift_top20', 'pl_top50',
                                    'uplift_top50', 'corr_y', 'corr_pl', 'n_regra', 'pl_regra', 'pl_regra_liq') if c in df.columns}
    agg['folds_uplift20_pos'] = ('uplift_top20', lambda s: int((s > 0).sum()))
    agg['n_folds'] = ('fold', 'nunique')
    res = df.groupby(['side', 'janela', 'tag']).agg(**agg).reset_index()
    res['side'] = res['side'].map({'buy': 'Compra', 'sell': 'Venda'})
    res.to_csv(f"{a.out_prefix}_resumo.csv", index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print("\n=== por lado, janela e experimento (média entre folds) ===")
    print(res.round(3).to_string(index=False))
    show = res[['side', 'janela', 'tag', 'auc_intra', 'uplift_top20', 'uplift_top50', 'corr_y', 'corr_pl',
                'n_regra', 'pl_regra', 'pl_regra_liq', 'pl_tudo', 'folds_uplift20_pos', 'n_folds']]
    Path(f"{a.out_prefix}_resumo.md").write_text(
        "Métricas por lado, janela do teste e experimento (média entre os folds). `uplift_top20/50`: P/L médio do "
        "topo 20%/50% de probabilidade dentro de cada combinação menos o de aceitar tudo (pts/trade, bruto). "
        f"`regra`: p_adj ≥ Ri/(Re+Ri)+m, m escolhida na validação; líquido de {a.custo:g} pts.\n\n" + md_table(show, "{:.2f}") + "\n",
        encoding="utf-8")
    print(f"\nSalvo em {a.out_prefix}_por_tarefa.csv, {a.out_prefix}_resumo.csv e {a.out_prefix}_resumo.md")


if __name__ == "__main__":
    main()
