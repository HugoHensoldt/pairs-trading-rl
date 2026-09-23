"""
Relatório dos modelos por combinação (gradient boosting + logística; compra e venda separados),
a partir dos artefatos de src/combo_models.py (COMBO_RUN_DIR/<tag>/<tarefa>/{predictions.npz,
metrics.json,state.json}). Mesmo estilo do report_lstm.py (reaproveita seus estilos/utilitários).

Regra de decisão avaliada: aceitar a oportunidade se p >= Ri/(Re+Ri) + m, onde Ri/(Re+Ri) é a taxa
de equilíbrio da combinação e a margem m é escolhida SÓ na validação (pregões anteriores ao teste)
e aplicada ao teste. P/L em pontos, do cache lateral (lucro idealizado ±SG/-SL; MTM no fechamento
forçado), bruto e líquido de um custo fixo por trade (--custo-pts, padrão 2,5 = corretagem).

Protocolos: A = split das campanhas LSTM (5 folds expanding em desenvolvimento; teste fixo
dividido em duas janelas) e B = 5 blocos móveis (os dois últimos = as duas janelas do teste).

Uso:
  python src/report_combo.py --run-dir combo_runs --tag combo_v1 [--notes docs/combo_leitura.md]
Comparação opcional com a LSTM v2 (grade sigma 2,2-2,6): --lstm-runs sdumont_backup_lstm/lstm_runs
--lstm-pl-dir sdumont_backup_lstm/lstm_pl_cache --lstm-chosen docs/lstm_pl_oportunidades_v2_limiar_validacao.csv
Saída: docs/relatorio_combo_<tag>.md e docs/img/combo_<tag>_fig*.png
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_lstm as R  # noqa: E402

MARGINS = np.round(np.arange(0.0, 0.1501, 0.01), 2)
MODELS = (('hgb', 'p', 'boosting'), ('logit', 'p_logit', 'logística'), ('base', 'p_base', 'só-combo (taxa de treino)'))
MIN_ACEITAS_VAL = 150     # mínimo de negociações aceitas na validação para escolher a margem


# --------------------------------------------------------------------------
# leitura
# --------------------------------------------------------------------------

def load_tasks(run_dir):
    tasks = {}
    for d in sorted(Path(run_dir).glob("[AB]_*")):
        if not ((d / "metrics.json").exists() and (d / "predictions.npz").exists()):
            continue
        t = {'m': json.load(open(d / "metrics.json", encoding="utf-8"))}
        t['s'] = json.load(open(d / "state.json", encoding="utf-8")) if (d / "state.json").exists() else {}
        with np.load(d / "predictions.npz") as z:
            t['p'] = {k: z[k] for k in z.files}
        tasks[d.name] = t
    return tasks


def pstar(grid):
    grid = np.asarray(grid, dtype=float)
    return grid[:, 2] / (grid[:, 1] + grid[:, 2])       # Ri / (Re + Ri)


# --------------------------------------------------------------------------
# regra de decisão e P/L
# --------------------------------------------------------------------------

def accept(score, combo, grid, margin):
    return score >= pstar(grid)[combo] + margin


def choose_margin(score, combo, pl, grid, cost):
    best = None
    for m in MARGINS:
        a = accept(score, combo, grid, m)
        if a.sum() < MIN_ACEITAS_VAL:
            continue
        v = float((pl[a] - cost).mean())
        if best is None or v > best[1] + 1e-12:
            best = (float(m), v)
    return best[0] if best else float('nan')


def windows_for(scheme, order_test):
    u = np.unique(order_test)
    if scheme == 'A':
        half = len(u) // 2
        return [('teste inteiro', np.ones(len(order_test), bool)),
                (f'{u[0]}–{u[half - 1]}', order_test <= u[half - 1]),
                (f'{u[half]}–{u[-1]}', order_test >= u[half])]
    return [(f'{u[0]}–{u[-1]}', np.ones(len(order_test), bool))]


def evaluate(tasks, cost):
    """Uma linha por (tarefa, janela, modelo/regra): P/L de aceitar tudo e da regra p >= p*+m."""
    rows = []
    for name, t in tasks.items():
        m, p = t['m'], t['p']
        grid, side, scheme, k = np.asarray(p['grid']), m['side'], m['scheme'], m['k']
        for wname, wmask in windows_for(scheme, p['order_test']):
            pl_w = p['pl_test'][wmask]
            rows.append(dict(tarefa=name, scheme=scheme, k=k, side=side, janela=wname, modelo='aceitar tudo',
                             margem=np.nan, n_todas=int(wmask.sum()), n_aceitas=int(wmask.sum()),
                             pl_bruto=float(pl_w.mean()), pl_liq=float(pl_w.mean() - cost),
                             acerto=float((pl_w > 0).mean())))
            for key, pfx, label in MODELS:
                marg = choose_margin(p[f'{pfx}_val'], p['combo_val'], p['pl_val'], grid, cost)
                if np.isnan(marg):
                    continue
                a = accept(p[f'{pfx}_test'], p['combo_test'], grid, marg) & wmask
                if a.sum() == 0:
                    continue
                rows.append(dict(tarefa=name, scheme=scheme, k=k, side=side, janela=wname, modelo=label,
                                 margem=marg, n_todas=int(wmask.sum()), n_aceitas=int(a.sum()),
                                 pl_bruto=float(p['pl_test'][a].mean()), pl_liq=float(p['pl_test'][a].mean() - cost),
                                 acerto=float((p['pl_test'][a] > 0).mean())))
    return pd.DataFrame(rows)


def boot_pl_ci(pl, orders, n_boot=300, seed=0):
    """IC95% da média do P/L por bootstrap sobre os PREGÕES (trades do mesmo dia são correlacionados)."""
    if len(pl) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    days, inv = np.unique(orders, return_inverse=True)
    s = np.bincount(inv, weights=pl, minlength=len(days))
    c = np.bincount(inv, minlength=len(days)).astype(float)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(days), len(days))
        means.append(s[pick].sum() / max(c[pick].sum(), 1))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test(aucs):
    """Fração de combinações com AUC > 0,5 e p-valor unilateral exato de um teste de sinal."""
    a = np.asarray([x for x in aucs if not np.isnan(x)])
    n, k = len(a), int((a > 0.5).sum())
    comb = getattr(math, 'comb', None) or (lambda n_, i_: math.factorial(n_) // (math.factorial(i_) * math.factorial(n_ - i_)))
    pval = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else float('nan')
    return k, n, pval


# --------------------------------------------------------------------------
# figuras
# --------------------------------------------------------------------------

def fig_auc_tasks(tasks, figs):
    sides = [s for s in ('buy', 'sell') if any(t['m']['side'] == s for t in tasks.values())]
    fig, axes = plt.subplots(1, len(sides), figsize=(5.2 * len(sides), 3.8), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        labels, x = [], 0
        for scheme, nome in (('A', 'A: folds'), ('B', 'B: blocos')):
            ks = sorted(t['m']['k'] for t in tasks.values() if t['m']['scheme'] == scheme and t['m']['side'] == side)
            for k in ks:
                t = tasks[f"{scheme}_{'fold' if scheme == 'A' else 'block'}{k}_{side}"]
                for key, col, lab in (('auc_intra_hgb', R.C_TEST, 'boosting'), ('auc_intra_logit', R.C_VAL, 'logística')):
                    ax.plot(x, t['m']['test'][key], 'o', color=col, ms=6, label=lab if x == 0 else None)
                labels.append(f"{scheme}{k}")
                x += 1
            x += 1
            labels.append("")
        ax.axhline(0.5, color=R.AXIS, ls='--', lw=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(f"{R.SIDE_PT[side]} · AUC intra-combinação no teste")
        ax.set_ylabel("AUC (só compara oportunidades da mesma combinação)")
        ax.legend(loc='best')
    fig.tight_layout()
    return figs.save(fig, "auc_intra_por_tarefa")


def fig_combo_heat(tasks, side, figs):
    rows = {}
    for t in tasks.values():
        if t['m']['side'] != side or t['m']['scheme'] != 'A':
            continue
        for c in t['m']['per_combo']:
            r = rows.setdefault(c['combo'], dict(sigma=c['sigma'], Re=c['Re'], Ri=c['Ri'], v=[], n=c['n_test']))
            r['v'].append(c['auc_test_hgb'])
    df = pd.DataFrame([dict(sigma=r['sigma'], Re=r['Re'], Ri=r['Ri'], auc=float(np.nanmean(r['v'])), n=r['n']) for r in rows.values()])
    if df.empty:
        return None
    return R.heat_panels(df, 'auc', '{:.2f}', R.DIV_CMAP, 0.5, 0.1, figs, f'auc_combo_{side}',
                         f'AUC do boosting no teste por combinação (média dos 5 folds) — {R.SIDE_PT[side]}', 'AUC')


def fig_pl_janelas(df, custo, figs):
    a = df[(df.scheme == 'A') & (df.janela != 'teste inteiro')]
    if a.empty:
        return None
    ordem = ['aceitar tudo', 'só-combo (taxa de treino)', 'logística', 'boosting']
    cores = {'aceitar tudo': R.MUTED, 'só-combo (taxa de treino)': R.C_TRAIN, 'logística': R.C_VAL, 'boosting': R.C_TEST}
    sides = [s for s in ('buy', 'sell') if (a.side == s).any()]
    fig, axes = plt.subplots(1, len(sides), figsize=(5.6 * len(sides), 4.0), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        d = a[a.side == side].groupby(['janela', 'modelo']).pl_liq.mean().unstack('modelo')
        janelas = list(d.index)
        h = 0.18
        for i, mod in enumerate(ordem):
            if mod not in d.columns:
                continue
            ys = np.arange(len(janelas)) + (i - 1.5) * h
            ax.barh(ys, d[mod].values, height=h * 0.9, color=cores[mod], label=mod, zorder=3)
            for y, v in zip(ys, d[mod].values):
                ax.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:+.1f}", va='center', ha='left' if v >= 0 else 'right', fontsize=7, color=R.INK2)
        ax.set_yticks(range(len(janelas)))
        ax.set_yticklabels(janelas)
        ax.axvline(0, color=R.AXIS, lw=1, zorder=2)
        ax.set_title(f"{R.SIDE_PT[side]} · janelas do teste")
        ax.set_xlabel(f"P/L médio por trade, pontos (líquido de {custo:g} pts; média dos folds)")
        ax.legend(loc='best', fontsize=7)
    fig.tight_layout()
    return figs.save(fig, "pl_janelas_protocolo_A")


def fig_pl_blocos(df, custo, figs):
    b = df[df.scheme == 'B']
    if b.empty:
        return None
    sides = [s for s in ('buy', 'sell') if (b.side == s).any()]
    fig, axes = plt.subplots(1, len(sides), figsize=(5.6 * len(sides), 3.8), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        d = b[b.side == side]
        for mod, col in (('aceitar tudo', R.MUTED), ('só-combo (taxa de treino)', R.C_TRAIN), ('logística', R.C_VAL), ('boosting', R.C_TEST)):
            x = d[d.modelo == mod].sort_values('k')
            if len(x):
                ax.plot(x.k, x.pl_liq, marker='o', ms=5, color=col, label=mod)
        ax.axhline(0, color=R.AXIS, lw=1)
        ks = sorted(d.k.unique())
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{k}\n{d[(d.k == k)].janela.iloc[0]}" for k in ks], fontsize=7)
        ax.set_title(f"{R.SIDE_PT[side]} · blocos móveis (cada um testado com modelo do passado)")
        ax.set_ylabel(f"P/L médio por trade (pts, líq. de {custo:g})")
        ax.legend(loc='best', fontsize=7)
    fig.tight_layout()
    return figs.save(fig, "pl_blocos_protocolo_B")


def fig_calibracao(tasks, figs):
    sides = [s for s in ('buy', 'sell') if any(t['m']['side'] == s for t in tasks.values())]
    fig, axes = plt.subplots(1, len(sides), figsize=(4.6 * len(sides), 4.0), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        bl = [t['p'] for t in tasks.values() if t['m']['side'] == side and t['m']['scheme'] == 'B']
        if not bl:
            continue
        y = np.concatenate([p['y_test'] for p in bl])
        ax.plot([0, 1], [0, 1], color=R.AXIS, ls='--', lw=1)
        for key, col, lab in (('p', R.C_TEST, 'boosting'), ('p_logit', R.C_VAL, 'logística')):
            s = np.concatenate([p[f'{key}_test'] for p in bl])
            edges = np.unique(np.quantile(s, np.linspace(0, 1, 11)))
            b = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
            ax.plot([s[b == i].mean() for i in range(len(edges) - 1)], [y[b == i].mean() for i in range(len(edges) - 1)],
                    marker='o', ms=5, color=col, label=lab)
        ax.set_title(f"{R.SIDE_PT[side]} · calibração (blocos B juntos, decis)")
        ax.set_xlabel("probabilidade prevista")
        ax.set_ylabel("taxa de lucro observada")
        ax.legend(loc='upper left')
    fig.tight_layout()
    return figs.save(fig, "calibracao")


def fig_curvas(tasks, figs):
    sides = [s for s in ('buy', 'sell') if any(t['m']['side'] == s for t in tasks.values())]
    folds = sorted({t['m']['k'] for t in tasks.values() if t['m']['scheme'] == 'A'})
    if not folds:
        return None
    fig, axes = plt.subplots(len(sides), len(folds), figsize=(2.6 * len(folds) + 0.6, 2.4 * len(sides) + 0.5), squeeze=False, sharey='row')
    for i, side in enumerate(sides):
        for j, k in enumerate(folds):
            ax = axes[i][j]
            t = tasks.get(f"A_fold{k}_{side}")
            if t and t['s'].get('history'):
                h = pd.DataFrame(t['s']['history'])
                ax.plot(h.epoch, h.loss, color=R.C_TRAIN, label='treino')
                ax.plot(h.epoch, h.val_loss, color=R.C_VAL, label='validação')
                ax.axhline(1.0, color=R.AXIS, lw=0.8)
            ax.set_title(f"{R.SIDE_PT[side]} · fold {k}")
            if j == 0:
                ax.set_ylabel("log-loss / taxa base")
            if i == len(sides) - 1:
                ax.set_xlabel("iteração do boosting")
    axes[0][0].legend(loc='best')
    fig.tight_layout()
    return figs.save(fig, "curvas_boosting")


# --------------------------------------------------------------------------
# comparação opcional com a LSTM v2
# --------------------------------------------------------------------------

def lstm_v2_windows(lstm_runs, pl_dir, chosen_csv, cost):
    """P/L por janela das oportunidades aceitas pela LSTM v2 (limiar escolhido na validação, igual a
    analyze_pl_oportunidades.py), média entre folds e entre os experimentos base/seed2."""
    import analyze_pl_oportunidades as AP
    L = AP.L
    reader = AP.PLReader(pl_dir)
    dev, test = L.split_orders()
    chosen = pd.read_csv(chosen_csv)
    half = len(test) // 2
    wins = {f'{test[0]}–{test[half - 1]}': (test[0], test[half - 1]), f'{test[half]}–{test[-1]}': (test[half], test[-1])}
    rows = []
    for side in L.SIDES:
        for tag in ('lstm_v2_base', 'lstm_v2_seed2'):
            for fold in range(1, L.N_SPLITS + 1):
                f = Path(lstm_runs) / tag / f"fold{fold}_{side}" / "predictions.npz"
                sel = chosen[(chosen.tag == tag) & (chosen.fold == fold) & (chosen.side == side)]
                if not f.exists() or sel.empty:
                    continue
                z = np.load(f)
                grid = [tuple(map(float, c)) for c in z['grid']]
                pl, ords, _ = reader.assemble(test, grid, side)
                acc = z['p_test'] >= float(sel.thr_val.iloc[0])
                for wn, (lo, hi) in wins.items():
                    m = acc & (ords >= lo) & (ords <= hi)
                    if m.sum():
                        rows.append(dict(side=side, janela=wn, tag=tag, fold=fold, n=int(m.sum()),
                                         pl_bruto=float(pl[m].mean()), pl_liq=float(pl[m].mean() - cost)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# relatório
# --------------------------------------------------------------------------

def build(run_root, tag, out_dir, custo, n_boot, notes, lstm_args):
    tasks = load_tasks(Path(run_root) / tag)
    if not tasks:
        raise SystemExit(f"Nenhuma tarefa concluída em {Path(run_root) / tag}")
    figs = R.Figs(Path(out_dir) / "img", f"combo_{tag}")
    cfg = next(iter(tasks.values()))['m']['config']
    sides = [s for s in ('buy', 'sell') if any(t['m']['side'] == s for t in tasks.values())]
    df0, dfc = evaluate(tasks, 0.0), evaluate(tasks, custo)
    L_ = []

    # ---------- resumo (só fatos) ----------
    L_ += [f"# Relatório — modelos por combinação, experimento `{tag}`", "",
           "> Gerado por `src/report_combo.py` a partir dos artefatos de `src/combo_models.py`. Tudo aqui é medido; "
           "nenhuma interpretação é gerada automaticamente.", "", "## 0. Resumo", ""]
    for side in sides:
        for scheme, nome in (('A', 'protocolo A (split atual)'), ('B', 'blocos móveis B')):
            ts = [t for t in tasks.values() if t['m']['side'] == side and t['m']['scheme'] == scheme]
            if not ts:
                continue
            ah = np.mean([t['m']['test']['auc_intra_hgb'] for t in ts])
            al = np.mean([t['m']['test']['auc_intra_logit'] for t in ts])
            sk = np.mean([t['m']['test']['skill_logloss_hgb'] for t in ts])
            g = np.mean([t['m']['test']['auc_global_hgb'] for t in ts])
            gb = np.mean([t['m']['test']['auc_global_base'] for t in ts])
            L_.append(f"- **{R.SIDE_PT[side]}, {nome}** ({len(ts)} tarefas): AUC intra-combinação = **{ah:.3f}** (boosting) e "
                      f"{al:.3f} (logística); skill de log-loss do boosting sobre a taxa da combinação = {sk:+.4f}; "
                      f"AUC global do boosting {g:.3f} contra {gb:.3f} de um score só-combinação.")
    L_ += ["", "AUC intra-combinação = 0,5 é o acaso. Skill de log-loss > 0 significa prever melhor que a taxa de lucro da própria "
           "combinação; ≤ 0, que o modelo não acrescenta nada além dela.", ""]
    if notes:
        L_ += ["---", "", Path(notes).read_text(encoding="utf-8").strip(), ""]

    # ---------- configuração ----------
    g = np.array(cfg['grid'])
    L_ += ["---", "", "## 1. Configuração", "",
           R.md_table(pd.DataFrame([
               ["Modelos", "um por (combinação, lado, tarefa): HistGradientBoosting raso (profundidade 3, taxa 0,05, até "
                           f"{cfg['n_iter_max']} iterações, nº escolhido na validação temporal) e regressão logística L2"],
               ["Combinações", f"{len(g)}: sigma {sorted(set(map(float, g[:, 0])))}, Re {sorted(set(map(float, g[:, 1])))}, Ri {sorted(set(map(float, g[:, 2])))}"],
               ["Features", f"{cfg['n_features']} estacionárias resumidas da janela de 120 ticks (mispricing e sua dinâmica, momentum do WIN e do justo, bid-ask, volumes, hora/dia)"],
               ["Regra", "aceitar se p ≥ Ri/(Re+Ri) + m; m ∈ [0; 0,15] escolhida na validação (≥ %d trades aceitos)" % MIN_ACEITAS_VAL],
               ["Custo", f"P/L bruto e líquido de {custo:g} pt/trade (corretagem); o spread já está embutido no lucro do cache"],
               ["Protocolo A", "5 folds expanding em desenvolvimento; teste fixo dividido em duas janelas"],
               ["Protocolo B", "5 blocos móveis de mesmo tamanho terminando no fim dos dados; cada um testado com modelo treinado só no passado"],
           ], columns=["Item", "Valor"])), ""]
    jr = []
    for name, t in tasks.items():
        if t['m']['side'] == 'buy':
            c = t['m']['config']
            jr.append([name.replace('_buy', ''), f"{c['train_orders'][0]}–{c['train_orders'][1]} ({c['n_train_days']})",
                       f"{c['val_orders'][0]}–{c['val_orders'][1]} ({c['n_val_days']})", f"{c['test_orders'][0]}–{c['test_orders'][1]} ({c['n_test_days']})"])
    L_ += ["**Janelas** (pregões; dias entre parênteses):", "", R.md_table(pd.DataFrame(jr, columns=["tarefa", "treino", "validação", "teste"])), ""]

    # ---------- métricas de discriminação ----------
    L_ += ["---", "", "## 2. Discriminação dentro de cada combinação", "",
           f"![AUC intra]({fig_auc_tasks(tasks, figs)})", ""]
    mt = []
    for name, t in sorted(tasks.items()):
        m = t['m']['test']
        pc = t['m']['per_combo']
        k, n, pv = sign_test([c['auc_test_hgb'] for c in pc])
        mt.append({'tarefa': name, 'n teste': m['n'], 'AUC intra boosting': m['auc_intra_hgb'], 'AUC intra logística': m['auc_intra_logit'],
                   'skill logloss boosting': m['skill_logloss_hgb'], 'skill logloss logística': m['skill_logloss_logit'],
                   'AUC global só-combo': m['auc_global_base'], 'combos AUC>0,5': f"{k}/{n}", 'p (sinal)': pv})
    L_ += [R.md_table(pd.DataFrame(mt), "{:.3f}"), "",
           "`combos AUC>0,5` conta, das combinações com as duas classes no teste, quantas têm AUC do boosting acima de 0,5; `p (sinal)` é o "
           "p-valor unilateral de um teste binomial contra 50/50 (as combinações compartilham entradas parecidas, então o valor é apenas indicativo).", ""]
    for side in sides:
        p = fig_combo_heat(tasks, side, figs)
        if p:
            L_ += [f"![AUC por combinação {side}]({p})", ""]
    p = fig_curvas(tasks, figs)
    if p:
        L_ += ["### Curvas do boosting", "", f"![Curvas]({p})", "",
               "Log-loss dividida pela da taxa base (1,0 = não melhora sobre a taxa da combinação); média entre combinações, por iteração.", ""]
    L_ += ["### Calibração", "", f"![Calibração]({fig_calibracao(tasks, figs)})", ""]

    # ---------- P/L ----------
    def tabela(df, cost_label):
        a = df[df.scheme == 'A'].groupby(['side', 'janela', 'modelo']).agg(
            n_aceitas=('n_aceitas', 'mean'), margem=('margem', 'mean'), acerto=('acerto', 'mean'),
            pl_bruto=('pl_bruto', 'mean'), pl_liq=('pl_liq', 'mean'), folds_pos=('pl_liq', lambda s: int((s > 0).sum())),
            n_folds=('pl_liq', 'size')).reset_index()
        a['side'] = a['side'].map(R.SIDE_PT)
        return a
    L_ += ["---", "", "## 3. P/L das oportunidades aceitas (pontos por trade)", "",
           "Regra `p ≥ Ri/(Re+Ri) + m` (m escolhida na validação). `aceitar tudo` é a heurística sem filtro; `só-combo` usa como score a taxa de "
           "lucro da combinação no treino (o que se ganha só por escolher parâmetros, sem olhar o mercado).", "",
           "### Protocolo A — janelas do teste (média entre os 5 folds)", "",
           f"![P/L janelas]({fig_pl_janelas(dfc, custo, figs)})", "",
           f"**Bruto (custo 0):**", "", R.md_table(tabela(df0[df0.janela != 'teste inteiro'], 'bruto'), "{:.2f}"), "",
           f"**Líquido ({custo:g} pts/trade):**", "", R.md_table(tabela(dfc[dfc.janela != 'teste inteiro'], 'liq'), "{:.2f}"), ""]

    L_ += ["### Protocolo B — blocos móveis", "", f"![P/L blocos]({fig_pl_blocos(dfc, custo, figs)})", ""]
    bt = dfc[dfc.scheme == 'B'].copy()
    ci_lo, ci_hi = [], []
    for _, r in bt.iterrows():
        t = tasks[r['tarefa']]
        p = t['p']
        grid = np.asarray(p['grid'])
        if r['modelo'] == 'aceitar tudo':
            m = np.ones(len(p['pl_test']), bool)
        else:
            pfx = {lab: pf for _, pf, lab in MODELS}[r['modelo']]
            m = accept(p[f'{pfx}_test'], p['combo_test'], grid, r['margem'])
        lo, hi = boot_pl_ci(p['pl_test'][m] - custo, p['order_test'][m], n_boot=n_boot)
        ci_lo.append(lo)
        ci_hi.append(hi)
    bt['IC95% pl_liq'] = [f"[{lo:.1f}; {hi:.1f}]" for lo, hi in zip(ci_lo, ci_hi)]
    bt['lado'] = bt['side'].map(R.SIDE_PT)
    L_ += [R.md_table(bt[['lado', 'k', 'janela', 'modelo', 'margem', 'n_aceitas', 'acerto', 'pl_bruto', 'pl_liq', 'IC95% pl_liq']]
                      .sort_values(['lado', 'k', 'modelo']).rename(columns={'k': 'bloco'}), "{:.2f}"), "",
           "IC95% por bootstrap sobre os pregões do bloco (média do P/L líquido das oportunidades aceitas).", ""]

    # ---------- comparação com a LSTM v2 ----------
    la = lstm_args
    if la.get('runs') and Path(la['runs']).exists() and Path(la['pl_dir']).exists() and Path(la['chosen']).exists():
        lw = lstm_v2_windows(la['runs'], la['pl_dir'], la['chosen'], custo)
        if not lw.empty:
            lt = lw.groupby(['side', 'janela']).agg(n_aceitas=('n', 'mean'), pl_bruto=('pl_bruto', 'mean'), pl_liq=('pl_liq', 'mean')).reset_index()
            lt['modelo'] = 'LSTM v2 (27 combos, agrupada)'
            hb = dfc[(dfc.scheme == 'A') & (dfc.modelo == 'boosting') & (dfc.janela != 'teste inteiro')].groupby(['side', 'janela']).agg(
                n_aceitas=('n_aceitas', 'mean'), pl_liq=('pl_liq', 'mean'), pl_bruto=('pl_bruto', 'mean')).reset_index()
            hb['modelo'] = 'boosting por combinação (54 combos)'
            ac = dfc[(dfc.scheme == 'A') & (dfc.modelo == 'aceitar tudo') & (dfc.janela != 'teste inteiro')].groupby(['side', 'janela']).agg(
                n_aceitas=('n_aceitas', 'mean'), pl_liq=('pl_liq', 'mean'), pl_bruto=('pl_bruto', 'mean')).reset_index()
            ac['modelo'] = 'aceitar tudo (54 combos)'
            comp = pd.concat([ac, hb, lt], ignore_index=True)
            comp['lado'] = comp['side'].map(R.SIDE_PT)
            L_ += ["### Comparação com a LSTM v2 (mesmas janelas do teste)", "",
                   R.md_table(comp[['lado', 'janela', 'modelo', 'n_aceitas', 'pl_bruto', 'pl_liq']].sort_values(['lado', 'janela', 'modelo']), "{:.2f}"), "",
                   "Atenção: as grades diferem (54 combinações contra 27, e conjuntos de trades diferentes), então a comparação é de ordem de grandeza, não de igualdade de amostra.", ""]

    L_ += ["---", "", "## 4. Limitações e ressalvas", "",
           "- **Poucos dados por modelo.** Cada (combinação, lado) treina com centenas de amostras nos primeiros folds/blocos (até ~2–4 mil nos últimos); os resultados individuais por combinação são ruidosos, por isso a leitura é agregada.",
           "- **Seleção entre 108 modelos.** Uma combinação isolada com bom resultado pode ser sorte; use os agregados, o teste de sinal e a consistência entre blocos.",
           "- **P/L idealizado.** É o lucro do cache (±SG/−SL, fechamento forçado a mercado), sem corretagem além do custo fixo informado; só o WIN, sem a perna de BOVA11.",
           "- **Regimes.** O P/L bruto da própria heurística muda muito entre janelas; compare sempre com `aceitar tudo` na mesma janela.",
           "", "## Apêndice: reprodução", "",
           "```", f"python src/report_combo.py --run-dir {run_root} --tag {tag} --custo-pts {custo:g}", "```", ""]
    out = Path(out_dir) / f"relatorio_combo_{tag}.md"
    out.write_text("\n".join(L_), encoding="utf-8")
    pd.concat([df0.assign(custo=0.0), dfc.assign(custo=custo)]).to_csv(Path(out_dir) / f"combo_{tag}_pl_por_tarefa.csv", index=False)
    print(f"Relatório: {out}  ({figs.n} figuras em {figs.dir})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="combo_runs")
    ap.add_argument("--tag", default="combo_v1")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--custo-pts", type=float, default=2.5)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--lstm-runs", default=None)
    ap.add_argument("--lstm-pl-dir", default="sdumont_backup_lstm/lstm_pl_cache")
    ap.add_argument("--lstm-chosen", default="docs/lstm_pl_oportunidades_v2_limiar_validacao.csv")
    a = ap.parse_args()
    build(a.run_dir, a.tag, a.out_dir, a.custo_pts, a.boot, a.notes,
          dict(runs=a.lstm_runs, pl_dir=a.lstm_pl_dir, chosen=a.lstm_chosen))
