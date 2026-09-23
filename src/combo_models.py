"""
Modelos POR COMBINAÇÃO (sigma, Re, Ri) e por lado (compra/venda): gradient boosting
+ regressão logística sobre features resumidas da janela de 120 ticks.

Motivação (ver docs/lstm_leitura_v2.md): a LSTM agrupada só aprendia a taxa de lucro
da combinação (AUC intra-combinação ~0,5). Aqui cada (combinação, lado) tem o seu
modelo, então esse atalho não existe; e como cada modelo vê só centenas a poucos
milhares de amostras, usa-se um modelo eficiente em dados (árvores rasas + logística L2)
sobre ~45 features estacionárias em vez de uma rede recorrente sobre 120x14 valores brutos.

Reaproveita lstm_pipeline.py (grade, cache de amostras e de P/L, pregões, splits). A grade
vem das mesmas variáveis de ambiente (ex.: LSTM_SIGMA_GRID="1.4,1.6,1.8,2.2,2.4,2.6").

Estágios (python src/combo_models.py --stage ...):
  features   lê as janelas X do cache de amostras e grava features pequenas por (pregão,
             combinação) em COMBO_FEAT_DIR, junto com y, forçado e lucro em pontos (P/L,
             conferido contra y). Paralelo por pregão.
  train      treina as tarefas pendentes (ou --task NOME) e grava, por tarefa, em
             COMBO_RUN_DIR/<tag>/<tarefa>/: predictions.npz (esquema do LSTM + p_logit_*,
             pl_*), metrics.json, state.json.
  pending    lista tarefas ainda não concluídas.
  aggregate  tabela-resumo de todas as tarefas (CSV em COMBO_RUN_DIR/<tag>/).

Tarefas (2 protocolos x 2 lados):
  A_fold<k>_<lado>   split atual: L.fold_orders (folds expanding em desenvolvimento) e teste
                     fixo = últimos TEST_FRAC dos pregões (mesmo teste das campanhas LSTM).
  B_block<j>_<lado>  blocos móveis: COMBO_B_BLOCKS blocos de COMBO_B_SIZE pregões (padrão:
                     metade do teste = 72) terminando no fim dos dados; os dois últimos são
                     exatamente as duas janelas do teste. Cada bloco é testado com modelo
                     treinado só no passado; os COMBO_B_VAL pregões (padrão 56) imediatamente
                     anteriores ao bloco são a validação (iterações do boosting e margem).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_pipeline as L  # noqa: E402

SCRIPT_START = time.time()

FEAT_DIR = Path(os.environ.get("COMBO_FEAT_DIR", "combo_feats"))
RUN_TAG = os.environ.get("COMBO_RUN_TAG", "combo_v1")
RUN_ROOT = Path(os.environ.get("COMBO_RUN_DIR", "combo_runs")) / RUN_TAG
B_BLOCKS = int(os.environ.get("COMBO_B_BLOCKS", "5"))
B_VAL = int(os.environ.get("COMBO_B_VAL", "56"))
B_SIZE = int(os.environ.get("COMBO_B_SIZE", "0"))       # 0 = metade do teste
N_ITER = int(os.environ.get("COMBO_N_ITER", "300"))     # iterações máximas do boosting
SIDES = L.SIDES
COL = {name: i for i, name in enumerate(L.FEATURE_COLS)}
T = L.N_TICKS


# --------------------------------------------------------------------------
# 1) FEATURES (a partir das janelas do cache; nada de níveis de preço absolutos)
# --------------------------------------------------------------------------

FEATURE_NAMES = None   # preenchido na 1ª chamada de engineer_features


def engineer_features(X):
    """(n, T, n_features) -> (n, k) float32 + nomes. Só usa DIFERENÇAS e razões (mispricing,
    bid-ask, variações em lags, volumes), portanto é invariante a deslocar o nível de preço.
    O último tick da janela é o tick da entrada: nenhuma informação do futuro."""
    global FEATURE_NAMES
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    bid, ask = X[:, :, COL['bid']], X[:, :, COL['ask']]
    fair = (X[:, :, COL['Wbjusto']] + X[:, :, COL['Wajusto']]) / 2
    mid = (bid + ask) / 2
    spread = X[:, :, COL['spread']]
    ba = ask - bid
    vol = np.maximum(X[:, -1, COL['volatilidade']], 1.0)
    sg, sl = X[:, -1, COL['SG']], X[:, -1, COL['SL']]
    cols, names = [], []

    def add(name, v):
        names.append(name)
        cols.append(np.asarray(v, dtype=np.float64))

    # estado no tick da entrada
    add('spread', spread[:, -1])
    add('spread_z', spread[:, -1] / vol)
    add('volatilidade', X[:, -1, COL['volatilidade']])
    add('bidask', ba[:, -1])
    add('SG', sg)
    add('SL', sl)
    add('SG_bidask', sg / np.maximum(ba[:, -1], 1.0))
    add('SG_vol', sg / vol)
    # dinâmica do mispricing
    for lag in (5, 10, 30, 60, 119):
        add(f'dspread_{lag}', spread[:, -1] - spread[:, -1 - lag])
    for w in (30, 120):
        seg = spread[:, -w:]
        lo, hi = seg.min(1), seg.max(1)
        add(f'sp_menos_media_{w}', spread[:, -1] - seg.mean(1))
        add(f'sp_std_{w}', seg.std(1))
        add(f'sp_pos_{w}', (spread[:, -1] - lo) / np.maximum(hi - lo, 1e-6))
    add('sp_max_menos_atual_120', spread.max(1) - spread[:, -1])
    add('sp_atual_menos_min_120', spread[:, -1] - spread.min(1))
    sgn = np.sign(spread)
    rev = (sgn == sgn[:, -1:])[:, ::-1]
    add('sp_seq_mesmo_sinal', np.where(rev.all(1), T, np.argmin(rev, axis=1)))
    # momentum do WIN e do preço justo (BOVA11)
    for lag in (5, 10, 30, 119):
        add(f'dmid_{lag}', mid[:, -1] - mid[:, -1 - lag])
        add(f'dfair_{lag}', fair[:, -1] - fair[:, -1 - lag])
    add('dmid_30_z', (mid[:, -1] - mid[:, -31]) / vol)
    # bid-ask da janela
    add('bidask_media', ba.mean(1))
    add('bidask_rel', ba[:, -1] / np.maximum(ba.mean(1), 1.0))
    # volumes (WIN e BOVA11)
    for nome, c in (('volwin', COL['volume']), ('volbova', COL['bvolume'])):
        v = X[:, :, c]
        s10, s30, s120 = v[:, -10:].sum(1), v[:, -30:].sum(1), v.sum(1)
        add(f'{nome}_10', np.log1p(np.maximum(s10, 0)))
        add(f'{nome}_30', np.log1p(np.maximum(s30, 0)))
        add(f'{nome}_120', np.log1p(np.maximum(s120, 0)))
        add(f'{nome}_ratio', s30 / (s120 / 4 + 1.0))
    # tempo (decodifica o seno/cosseno do tick da entrada)
    ts, tc = X[:, -1, COL['time_sin']], X[:, -1, COL['time_cos']]
    seconds = (np.arctan2(ts, tc) % (2 * np.pi)) / (2 * np.pi) * 86400
    add('minuto_do_dia', seconds / 60)
    add('time_sin', ts)
    add('time_cos', tc)
    ds, dc = X[:, -1, COL['day_sin']], X[:, -1, COL['day_cos']]
    add('dia_semana', np.round((np.arctan2(ds, dc) % (2 * np.pi)) / (2 * np.pi) * 7))
    F = np.stack(cols, axis=1)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    FEATURE_NAMES = names
    return F


def feature_names():
    if FEATURE_NAMES is None:
        engineer_features(np.zeros((1, T, len(L.FEATURE_COLS)), dtype=np.float32) + 1.0)
    return list(FEATURE_NAMES)


def _feat_file(order, combo):
    return FEAT_DIR / ("f_" + L._cache_file(order, *combo, 3000, 2400).name)


def _atomic_savez(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def features_for_day(order, grid=None):
    """Gera (ou reaproveita) os arquivos de features de UM pregão para todas as combinações.
    Confere que o P/L lateral está alinhado (mesmo y e forçado) com o cache de amostras."""
    grid = grid or L.PARAM_GRID
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    for combo in grid:
        out = _feat_file(order, combo)
        if out.exists():
            continue
        src = L._cache_file(order, *combo, 3000, 2400)
        pls = L._pl_file(order, *combo, 3000, 2400)
        if not src.exists():
            raise FileNotFoundError(f"cache de amostras ausente: {src}")
        if not pls.exists():
            raise FileNotFoundError(f"cache de P/L ausente: {pls} (rode --stage pl do lstm_pipeline)")
        arrays = {}
        with np.load(src) as z, np.load(pls) as p:
            for side in SIDES:
                y = z[f'{side}_y']
                if not (np.array_equal(y, p[f'{side}_y']) and np.array_equal(z[f'{side}_forced'], p[f'{side}_forced'])):
                    raise RuntimeError(f"P/L desalinhado: pregão {order}, combinação {combo}, lado {side}")
                F = engineer_features(z[f'{side}_X']) if len(y) else np.empty((0, len(feature_names())), np.float32)
                arrays[f'{side}_F'] = F
                arrays[f'{side}_y'] = y.astype(np.float32)
                arrays[f'{side}_pl'] = p[f'{side}_pl'].astype(np.float32)
                arrays[f'{side}_forced'] = z[f'{side}_forced']
        _atomic_savez(out, **arrays)


def _features_worker(order):
    try:
        features_for_day(order)
        return order, None
    except Exception as e:  # um pregão ruim não derruba os outros
        return order, f"{type(e).__name__}: {e}"


def run_features(workers):
    from multiprocessing import Pool
    orders = L.ALL_ORDERS
    failed = []
    with Pool(workers) as pool:
        for i, (order, err) in enumerate(pool.imap_unordered(_features_worker, orders), 1):
            if err:
                failed.append((order, err))
            if i % 25 == 0 or err:
                print(f"[{i}/{len(orders)}] pregão {order}" + (f"  ERRO: {err}" if err else ""), flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} pregões com erro: {failed[:5]}")
    print(f"features prontas em {FEAT_DIR} ({len(orders)} pregões x {len(L.PARAM_GRID)} combinações)")


# --------------------------------------------------------------------------
# 2) DADOS DE UMA TAREFA
# --------------------------------------------------------------------------

def load_feats(orders, side, grid=None):
    """(F, y, pl, ordem, combinação) na MESMA ordem de L.load_samples: pregões e, dentro
    de cada um, combinações da grade."""
    grid = grid or L.PARAM_GRID
    jobs = [(o, ci, _feat_file(o, c)) for o in orders for ci, c in enumerate(grid)]

    def _read(job):
        o, ci, path = job
        with np.load(path) as z:
            y = z[f'{side}_y']
            n = len(y)
            return (z[f'{side}_F'], y, z[f'{side}_pl'],
                    np.full(n, o, np.int32), np.full(n, ci, np.int16))

    with ThreadPoolExecutor(max_workers=int(os.environ.get("LSTM_LOAD_THREADS", "12"))) as ex:
        parts = list(ex.map(_read, jobs))
    k = len(feature_names())
    if not parts:
        return (np.empty((0, k), np.float32), np.empty(0, np.float32), np.empty(0, np.float32),
                np.empty(0, np.int32), np.empty(0, np.int16))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(5))


_ALL = {}   # lado -> (F, y, pl, ordem, combinação) de TODOS os pregões; carregado uma vez por processo


def get_feats(orders, side):
    """Fatia, de um cache em memória de todos os pregões, as amostras de `orders` (ascendentes),
    preservando a ordem de load_feats (pregões e, dentro deles, combinações)."""
    if side not in _ALL:
        t0 = time.time()
        _ALL[side] = load_feats(L.ALL_ORDERS, side)
        print(f"  [features {side}] {len(_ALL[side][1])} amostras de {len(L.ALL_ORDERS)} pregões em memória "
              f"({time.time() - t0:.0f}s)", flush=True)
    F, y, pl, o, c = _ALL[side]
    m = np.isin(o, np.asarray(orders, dtype=np.int32))
    return F[m], y[m], pl[m], o[m], c[m]


def task_names():
    return [f"{sch}_{kind}{k}_{side}"
            for sch, kind, n in (('A', 'fold', L.N_SPLITS), ('B', 'block', B_BLOCKS))
            for k in range(1, n + 1) for side in SIDES]


def resolve_task(name):
    """nome -> (scheme, k, side, train_orders, val_orders, test_orders)."""
    scheme, rest, side = name.split('_')
    k = int(rest.replace('fold', '').replace('block', ''))
    orders = list(L.ALL_ORDERS)
    dev, test = L.split_orders()
    if scheme == 'A':
        tr, va = L.fold_orders(dev, k)
        return scheme, k, side, tr, va, test
    size = B_SIZE or max(len(test) // 2, 1)
    start = len(orders) - (B_BLOCKS - k + 1) * size
    if start - B_VAL < 5:
        raise ValueError(f"bloco {k}: histórico insuficiente (início {start}, validação {B_VAL})")
    return scheme, k, side, orders[:start - B_VAL], orders[start - B_VAL:start], orders[start:start + size]


# --------------------------------------------------------------------------
# 3) TREINO DE UMA COMBINAÇÃO
# --------------------------------------------------------------------------

def _logloss(y, p):
    from sklearn.metrics import log_loss
    return float(log_loss(y, np.clip(p, 1e-4, 1 - 1e-4), labels=[0, 1]))


def _fit_combo(Ftr, ytr, Fva, yva, Fte):
    """Treina, para UMA combinação e lado, o boosting e a logística. Retorna previsões de
    val/teste, nº de iterações escolhido e as curvas de log-loss por iteração."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    base = float(ytr.mean()) if len(ytr) else 0.5
    res = dict(base=base, best_iter=0, p_hgb_val=np.full(len(Fva), base, np.float32),
               p_hgb_test=np.full(len(Fte), base, np.float32),
               p_lr_val=np.full(len(Fva), base, np.float32), p_lr_test=np.full(len(Fte), base, np.float32),
               curve_train=None, curve_val=None, status='base_rate')
    if len(ytr) < 60 or len(np.unique(ytr)) < 2:
        return res
    with threadpool_limits(1):
        # logística L2 (referência linear)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
        lr.fit(Ftr, ytr)
        res['p_lr_val'] = lr.predict_proba(Fva)[:, 1].astype(np.float32) if len(Fva) else res['p_lr_val']
        res['p_lr_test'] = lr.predict_proba(Fte)[:, 1].astype(np.float32) if len(Fte) else res['p_lr_test']
        # boosting raso; nº de iterações escolhido na validação TEMPORAL (o early stopping
        # interno do sklearn sortearia a validação e vazaria tempo)
        hgb = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=N_ITER, l2_regularization=1.0,
            min_samples_leaf=30, max_bins=64, early_stopping=False, random_state=0)
        hgb.fit(Ftr, ytr)
        tr_curve = np.array([_logloss(ytr, s[:, 1]) for s in hgb.staged_predict_proba(Ftr)])
        if len(Fva) >= 20 and len(np.unique(yva)) == 2:
            va_curve = np.array([_logloss(yva, s[:, 1]) for s in hgb.staged_predict_proba(Fva)])
            best = int(np.argmin(va_curve)) + 1
        else:
            va_curve, best = None, min(100, N_ITER)
        pick = lambda F: (next(islice(hgb.staged_predict_proba(F), best - 1, best))[:, 1].astype(np.float32)
                          if len(F) else np.empty(0, np.float32))
        res.update(p_hgb_val=pick(Fva), p_hgb_test=pick(Fte), best_iter=best,
                   curve_train=tr_curve, curve_val=va_curve, status='ok')
    return res


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(y) > 1 and len(np.unique(y)) == 2 else float('nan')


def intra_combo_auc(y, p, combo):
    """AUC média dentro de cada combinação, ponderada por n (só combina amostras da mesma combinação)."""
    num = den = 0.0
    for c in np.unique(combo):
        m = combo == c
        a = _auc(y[m], p[m])
        if not np.isnan(a):
            num += a * m.sum()
            den += m.sum()
    return num / den if den else float('nan')


def pooled_metrics(y, p_hgb, p_lr, p_base, combo):
    ll = lambda p: _logloss(y, p) if len(y) else float('nan')
    ll_base = ll(p_base)
    return {
        'n': int(len(y)), 'taxa_lucro': float(y.mean()) if len(y) else float('nan'),
        'auc_intra_hgb': intra_combo_auc(y, p_hgb, combo), 'auc_intra_logit': intra_combo_auc(y, p_lr, combo),
        'auc_global_hgb': _auc(y, p_hgb), 'auc_global_logit': _auc(y, p_lr), 'auc_global_base': _auc(y, p_base),
        'skill_logloss_hgb': 1 - ll(p_hgb) / ll_base if ll_base else float('nan'),
        'skill_logloss_logit': 1 - ll(p_lr) / ll_base if ll_base else float('nan'),
    }


# --------------------------------------------------------------------------
# 4) UMA TAREFA (protocolo, fold/bloco, lado) = 54 combinações x 2 modelos
# --------------------------------------------------------------------------

def _write_json(path, obj):
    L._write_json_atomic(path, obj)


def task_dir(name):
    return RUN_ROOT / name


def task_finished(name):
    return (task_dir(name) / "metrics.json").exists()


def pending_tasks():
    return [t for t in task_names() if not task_finished(t)]


def run_task(name, workers):
    from joblib import Parallel, delayed
    t0 = time.time()
    scheme, k, side, tr_o, va_o, te_o = resolve_task(name)
    grid = L.PARAM_GRID
    print(f"[{name}] treino {tr_o[0]}..{tr_o[-1]} ({len(tr_o)}d) | val {va_o[0]}..{va_o[-1]} ({len(va_o)}d) | "
          f"teste {te_o[0]}..{te_o[-1]} ({len(te_o)}d) | {len(grid)} combinações", flush=True)
    Ftr, ytr, pltr, otr, ctr = get_feats(tr_o, side)
    Fva, yva, plva, ova, cva = get_feats(va_o, side)
    Fte, yte, plte, ote, cte = get_feats(te_o, side)
    print(f"[{name}] amostras treino/val/teste: {len(ytr)}/{len(yva)}/{len(yte)} (carga {time.time() - t0:.0f}s)", flush=True)

    idx = lambda c, arr: np.flatnonzero(arr == c)
    jobs = []
    for ci in range(len(grid)):
        a, b, c = idx(ci, ctr), idx(ci, cva), idx(ci, cte)
        jobs.append((a, b, c))
    # backend 'multiprocessing' (e não o 'loky' padrão): funciona também no Python 3.7 do Windows
    results = Parallel(n_jobs=workers, backend='multiprocessing')(
        delayed(_fit_combo)(Ftr[a], ytr[a], Fva[b], yva[b], Fte[c]) for a, b, c in jobs)

    p = {key: {'val': np.zeros(len(yva), np.float32), 'test': np.zeros(len(yte), np.float32)}
         for key in ('hgb', 'lr', 'base')}
    per_combo = []
    curves_tr, curves_va = [], []
    for ci, ((a, b, c), r) in enumerate(zip(jobs, results)):
        p['hgb']['val'][b], p['hgb']['test'][c] = r['p_hgb_val'], r['p_hgb_test']
        p['lr']['val'][b], p['lr']['test'][c] = r['p_lr_val'], r['p_lr_test']
        p['base']['val'][b] = p['base']['test'][c] = r['base']
        if r['curve_train'] is not None:
            curves_tr.append(r['curve_train'] / max(_logloss(ytr[a], np.full(len(a), r['base'])), 1e-9))
        if r['curve_val'] is not None:
            curves_va.append(r['curve_val'] / max(_logloss(yva[b], np.full(len(b), r['base'])), 1e-9))
        s, re, ri = grid[ci]
        per_combo.append(dict(combo=ci, sigma=s, Re=re, Ri=ri, n_train=int(len(a)), n_val=int(len(b)), n_test=int(len(c)),
                              base_rate=r['base'], best_iter=r['best_iter'], status=r['status'],
                              auc_test_hgb=_auc(yte[c], p['hgb']['test'][c]), auc_test_logit=_auc(yte[c], p['lr']['test'][c]),
                              auc_val_hgb=_auc(yva[b], p['hgb']['val'][b])))

    d = task_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        d / "predictions.npz",
        y_val=yva, p_val=p['hgb']['val'], p_logit_val=p['lr']['val'], p_base_val=p['base']['val'],
        order_val=ova, combo_val=cva, pl_val=plva,
        y_test=yte, p_test=p['hgb']['test'], p_logit_test=p['lr']['test'], p_base_test=p['base']['test'],
        order_test=ote, combo_test=cte, pl_test=plte,
        grid=np.asarray(grid, dtype=float))
    # curva média (entre combinações) de log-loss RELATIVA à taxa base, por iteração de boosting
    hist = []
    if curves_va and curves_tr:
        n_it = min(min(len(c) for c in curves_va), min(len(c) for c in curves_tr))
        mtr = np.mean([c[:n_it] for c in curves_tr], axis=0)
        mva = np.mean([c[:n_it] for c in curves_va], axis=0)
        hist = [dict(epoch=i + 1, loss=float(mtr[i]), val_loss=float(mva[i]), acc=float('nan'), val_acc=float('nan'), sec=0.0)
                for i in range(n_it)]
    _write_json(d / "state.json", dict(name=name, scheme=scheme, k=k, side=side, done=True, finished=True, history=hist))
    metrics = dict(
        name=name, scheme=scheme, k=k, side=side, seconds=round(time.time() - t0, 1),
        val=pooled_metrics(yva, p['hgb']['val'], p['lr']['val'], p['base']['val'], cva),
        test=pooled_metrics(yte, p['hgb']['test'], p['lr']['test'], p['base']['test'], cte),
        per_combo=per_combo,
        config=dict(model='hgb+logit', n_iter_max=N_ITER, features=feature_names(), n_features=len(feature_names()),
                    grid=[list(map(float, c)) for c in grid],
                    train_orders=[int(tr_o[0]), int(tr_o[-1])], val_orders=[int(va_o[0]), int(va_o[-1])],
                    test_orders=[int(te_o[0]), int(te_o[-1])], n_train_days=len(tr_o), n_val_days=len(va_o),
                    n_test_days=len(te_o), tag=RUN_TAG))
    _write_json(d / "metrics.json", metrics)   # por último: presença = tarefa concluída
    m = metrics['test']
    print(f"[{name}] CONCLUÍDO em {metrics['seconds']}s | teste: AUC intra hgb {m['auc_intra_hgb']:.3f} / logit "
          f"{m['auc_intra_logit']:.3f} | skill logloss hgb {m['skill_logloss_hgb']:+.4f}", flush=True)
    return metrics


def run_train(workers, task=None, max_seconds=0):
    tasks = [task] if task else pending_tasks()
    for name in tasks:
        if task_finished(name):
            print(f"[{name}] já concluído.")
            continue
        if max_seconds and time.time() - SCRIPT_START > max_seconds * 0.8 and not task:
            print("orçamento de tempo quase esgotado; as tarefas restantes ficam para o próximo job.")
            break
        run_task(name, workers)


def aggregate():
    rows = []
    for name in task_names():
        p = task_dir(name) / "metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p, encoding="utf-8"))
        for split in ('val', 'test'):
            rows.append({'tarefa': name, 'split': split, **m[split]})
    if not rows:
        print("Nenhuma tarefa concluída ainda.")
        return None
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(df.round(4).to_string(index=False))
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUN_ROOT / "combo_summary.csv", index=False)
    print(f"\nSalvo em {RUN_ROOT / 'combo_summary.csv'}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["features", "train", "pending", "aggregate"])
    ap.add_argument("--task", help="(train) nome de uma tarefa, ex.: A_fold1_buy ou B_block5_sell")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() or 1, 1))
    args = ap.parse_args()
    if args.stage == "features":
        run_features(args.workers)
    elif args.stage == "train":
        run_train(args.workers, task=args.task,
                  max_seconds=int(os.environ.get("COMBO_MAX_SECONDS", "0")))
    elif args.stage == "pending":
        for t in pending_tasks():
            print(t)
    else:
        aggregate()
