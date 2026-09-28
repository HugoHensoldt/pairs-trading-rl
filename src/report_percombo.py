"""
Relatório do modo "um modelo por combinação" da LSTM (src/lstm_pipeline.py --stage train, com
LSTM_PER_COMBO=1): responde se, para uma estratégia (sigma, Re, Ri) FIXA, uma LSTM treinada só
com as oportunidades daquela combinação separa operações boas de ruins melhor que a taxa de
acerto da própria estratégia.

Fontes (todos os números vêm de arquivos gravados; nada é inventado):
  --run-dir/--tag        RUN_DIR/<tag>/lstm_percombo_results.csv (gerado por --stage aggregate)
                         e RUN_DIR/<tag>/fold<f>_<lado>_c<combo>/metrics.json (por tarefa)
  --lstm-runs/--lstm-tag um experimento da LSTM AGREGADA (predictions.npz por fold/lado), para a
                         referência "AUC intra-combinação do modelo agregado" (recalculada aqui,
                         não copiada de outro relatório)
  --combo-run-dir/--combo-tag  um experimento de src/combo_models.py (boosting por combinação),
                         para a referência "AUC por combinação do boosting"

Uso:
  python src/report_percombo.py --run-dir lstm_runs --tag lstm_v3_percombo \
      --lstm-runs sdumont_backup_lstm/lstm_runs --lstm-tag lstm_v2_base \
      --combo-run-dir sdumont_backup_combo/combo_runs --combo-tag combo_v1 \
      [--notes docs/percombo_leitura.md]
Saída: docs/relatorio_percombo_<tag>.md e docs/img/percombo_<tag>_fig*.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_lstm as R  # noqa: E402  (estilo, md_table, Figs, within_combo_auc, ...)

SIDE_PT = R.SIDE_PT


# --------------------------------------------------------------------------
# leitura
# --------------------------------------------------------------------------

def load_percombo(run_dir, tag):
    csv = Path(run_dir) / tag / "lstm_percombo_results.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} não existe (rode --stage aggregate com LSTM_PER_COMBO=1).")
    df = pd.read_csv(csv)
    n_splits, n_sides = df.fold.nunique(), df.side.nunique()
    n_combos = df.combo.nunique()
    return df, n_splits, n_sides, n_combos


def aggregated_intra_auc(lstm_runs, lstm_tag):
    """Recalcula (não copia de outro relatório) a AUC intra-combinação do modelo AGREGADO, a
    partir dos predictions.npz de cada fold/lado -- mesma definição de report_lstm.within_combo_auc."""
    rows = []
    root = Path(lstm_runs) / lstm_tag
    if not root.exists():
        return None
    for d in sorted(root.glob("fold*_*")):
        if not (d / "predictions.npz").exists():
            continue
        side = d.name.rsplit("_", 1)[1]
        with np.load(d / "predictions.npz") as z:
            rows.append(dict(side=side, fold=int(d.name.split("_")[0].replace("fold", "")),
                             auc_intra=R.within_combo_auc({'y_test': z['y_test'], 'p_test': z['p_test'],
                                                           'combo_test': z['combo_test']})))
    return pd.DataFrame(rows) if rows else None


def combo_boosting_auc(combo_run_dir, combo_tag):
    """AUC por combinação do boosting (src/combo_models.py), lido de per_combo em cada
    A_fold<k>_<lado>/metrics.json -- só o protocolo A (mesmo split que o modo por combinação)."""
    root = Path(combo_run_dir) / combo_tag
    if not root.exists():
        return None
    rows = []
    for d in sorted(root.glob("A_fold*_*")):
        p = d / "metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p, encoding="utf-8"))
        for c in m.get("per_combo", []):
            rows.append(dict(fold=m["k"], side=m["side"], combo=c["combo"], sigma=c["sigma"], Re=c["Re"], Ri=c["Ri"],
                             auc_boosting=c["auc_test_hgb"], n_test=c["n_test"]))
    return pd.DataFrame(rows) if rows else None


# --------------------------------------------------------------------------
# figuras
# --------------------------------------------------------------------------

def fig_auc_by_combo(df, figs):
    ok = df[df.status == "ok"]
    sides = [s for s in ("buy", "sell") if (ok.side == s).any()]
    fig, axes = plt.subplots(1, len(sides), figsize=(6.2 * len(sides), 4.2), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        g = ok[ok.side == side].groupby("combo").agg(
            auc=("auc_test", "mean"), lo=("auc_ci_lo", "mean"), hi=("auc_ci_hi", "mean"),
            n=("n_test", "mean"), folds=("fold", "size")).sort_values("auc")
        y = np.arange(len(g))
        ax.errorbar(g.auc, y, xerr=[g.auc - g.lo, g.hi - g.auc], fmt="o", ms=4, color=R.C_TEST,
                    ecolor=R.AXIS, capsize=2, elinewidth=1)
        ax.axvline(0.5, color=R.AXIS, ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"c{c}  (n={int(n)}, {int(f)} folds)" for c, n, f in zip(g.index, g.n, g.folds)], fontsize=7)
        ax.set_title(f"{SIDE_PT[side]} · AUC de teste por combinação (média entre folds, IC95% médio)")
        ax.set_xlabel("AUC")
    fig.tight_layout()
    return figs.save(fig, "auc_por_combinacao")


def fig_n_vs_auc(df, figs):
    ok = df[df.status == "ok"]
    sides = [s for s in ("buy", "sell") if (ok.side == s).any()]
    fig, axes = plt.subplots(1, len(sides), figsize=(4.6 * len(sides), 4.0), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        d = ok[ok.side == side]
        ax.scatter(d.n_train, d.auc_test, s=16, color=R.C_TEST, alpha=0.6)
        ax.axhline(0.5, color=R.AXIS, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"{SIDE_PT[side]} · nº de amostras de treino x AUC de teste")
        ax.set_xlabel("amostras de treino (log)")
        ax.set_ylabel("AUC de teste")
    fig.tight_layout()
    return figs.save(fig, "n_treino_vs_auc")


def fig_compare(df, agg_auc, boost_auc, figs):
    ok = df[df.status == "ok"]
    sides = [s for s in ("buy", "sell") if (ok.side == s).any()]
    fig, axes = plt.subplots(1, len(sides), figsize=(4.8 * len(sides), 3.6), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        labels, means, stds = [], [], []
        d = ok[ok.side == side].auc_test
        labels.append("LSTM por combinação"); means.append(d.mean()); stds.append(d.std(ddof=1) if len(d) > 1 else 0)
        if boost_auc is not None:
            b = boost_auc[boost_auc.side == side].auc_boosting
            if len(b):
                labels.append("boosting por combinação"); means.append(b.mean()); stds.append(b.std(ddof=1) if len(b) > 1 else 0)
        if agg_auc is not None:
            a = agg_auc[agg_auc.side == side].auc_intra
            if len(a):
                labels.append("LSTM agregada\n(AUC intra-combinação)"); means.append(a.mean()); stds.append(a.std(ddof=1) if len(a) > 1 else 0)
        y = np.arange(len(labels))
        ax.errorbar(means, y, xerr=stds, fmt="o", ms=7, color=R.C_TEST, ecolor=R.AXIS, capsize=4)
        for yi, m in zip(y, means):
            ax.text(m, yi + 0.15, f"{m:.3f}", ha="center", fontsize=8, color=R.INK2)
        ax.axvline(0.5, color=R.AXIS, ls="--", lw=1)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.set_title(f"{SIDE_PT[side]}")
        ax.set_xlabel("AUC (média ± desvio entre combinações/folds)")
    fig.tight_layout()
    return figs.save(fig, "comparacao_modelos")


def fig_pl_windows(df, figs):
    ok = df[df.status == "ok"]
    sides = [s for s in ("buy", "sell") if (ok.side == s).any()]
    fig, axes = plt.subplots(1, len(sides), figsize=(5.2 * len(sides), 3.8), squeeze=False)
    for j, side in enumerate(sides):
        ax = axes[0][j]
        d = ok[ok.side == side]
        rows = []
        for wi in (1, 2):
            for kind, col in (("sem filtro", f"janela{wi}_pl_sem_filtro"), ("aceitas (LSTM)", f"janela{wi}_pl_aceitas")):
                if col in d.columns:
                    rows.append((f"janela {wi}", kind, d[col].mean(), d[col].std(ddof=1) if d[col].notna().sum() > 1 else 0))
        p = pd.DataFrame(rows, columns=["janela", "tipo", "media", "desvio"])
        cores = {"sem filtro": R.MUTED, "aceitas (LSTM)": R.C_TEST}
        janelas = sorted(p.janela.unique())
        h = 0.32
        for i, tipo in enumerate(("sem filtro", "aceitas (LSTM)")):
            pp = p[p.tipo == tipo].set_index("janela").reindex(janelas)
            ys = np.arange(len(janelas)) + (i - 0.5) * h
            ax.barh(ys, pp.media.fillna(0), xerr=pp.desvio.fillna(0), height=h * 0.9, color=cores[tipo], label=tipo)
        ax.set_yticks(range(len(janelas)))
        ax.set_yticklabels(janelas)
        ax.axvline(0, color=R.AXIS, lw=1)
        ax.set_title(f"{SIDE_PT[side]} · P/L médio por trade (pontos, bruto; média entre combos/folds)")
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    return figs.save(fig, "pl_por_janela")


# --------------------------------------------------------------------------
# relatório
# --------------------------------------------------------------------------

def build(run_dir, tag, out_dir, lstm_runs, lstm_tag, combo_run_dir, combo_tag, notes):
    df, n_splits, n_sides, n_combos_seen = load_percombo(run_dir, tag)
    figs = R.Figs(Path(out_dir) / "img", f"percombo_{tag}")
    ok, skipped = df[df.status == "ok"], df[df.status == "skipped_low_n"]
    total_possivel = n_splits * n_sides * n_combos_seen
    L = []

    L += [f"# Relatório — LSTM por combinação, experimento `{tag}`", "",
          "> Gerado por `src/report_percombo.py` a partir de `lstm_percombo_results.csv` e dos "
          "`metrics.json` por tarefa. Tudo aqui é medido; nenhuma interpretação é gerada automaticamente.",
          "", "## 0. Resumo", "",
          f"- **Combinações treinadas:** {len(ok)} de {total_possivel} possíveis "
          f"({n_splits} folds × {n_sides} lados × {n_combos_seen} combinações vistas no CSV).",
          f"- **Puladas por poucas amostras:** {len(skipped)} (mínimo exigido: {int(skipped.n_train.max()) if len(skipped) else '—'} "
          f"não atingido; ver seção 2 para os limites configurados).",
          f"- **Pendentes (nem treinadas nem puladas):** {total_possivel - len(ok) - len(skipped)}.", ""]
    for side in [s for s in ("buy", "sell") if (ok.side == s).any()]:
        d = ok[ok.side == side]
        n_ci_pos = int((d.auc_ci_lo > 0.5).sum())
        L.append(f"- **{SIDE_PT[side]}**: AUC de teste = **{d.auc_test.mean():.3f}** "
                 f"(desvio entre combinação×fold {d.auc_test.std(ddof=1):.3f}; mín {d.auc_test.min():.3f}, máx {d.auc_test.max():.3f}); "
                 f"skill de log-loss médio {d.skill_logloss.mean():+.4f}; "
                 f"**{n_ci_pos} de {len(d)}** (combinação, fold) têm o IC95% da AUC inteiramente acima de 0,5.")
    L += ["", "Skill de log-loss > 0 significa prever melhor que a taxa de acerto da própria combinação no treino; "
          "≤ 0, que o modelo não acrescenta nada além dela. IC95% por bootstrap sobre os pregões do teste.", ""]

    if notes:
        L += ["---", "", Path(notes).read_text(encoding="utf-8").strip(), ""]

    # ---------------- configuração ----------------
    cfg = None
    if len(ok):
        m = json.load(open(Path(run_dir) / tag / f"fold{int(ok.iloc[0].fold)}_{ok.iloc[0].side}_c{int(ok.iloc[0].combo)}" / "metrics.json", encoding="utf-8"))
        cfg = m["config"]
    L += ["---", "", "## 1. Configuração", ""]
    if cfg:
        L += [R.md_table(pd.DataFrame([
            ["Unidade de tarefa", "uma LSTM por (fold, lado, combinação sigma/Re/Ri); só as oportunidades daquela combinação"],
            ["Rede", f"2× LSTM({cfg['units']}) + Dense(1, sigmoid)" + (f", dropout {cfg['dropout']}" if cfg["dropout"] else "")],
            ["Treino", f"Adam, binary_crossentropy, batch {cfg['batch_size']}, até {cfg['max_epochs']} épocas, "
                       f"early stopping (paciência {cfg['patience']}, melhor val_loss), class_weight (desbalanceamento da própria combinação)"],
            ["Mínimo de amostras", f"{cfg['min_train']} no treino, {cfg['min_val']} na validação; abaixo disso a combinação é pulada"],
            ["Validação", "mesmo TimeSeriesSplit expanding das demais campanhas; scaler e class_weight ajustados só no treino de cada combinação"],
            ["P/L", "regra: aceitar se p ≥ limiar escolhido NA VALIDAÇÃO (maximiza P/L médio bruto); comparado com a heurística sem filtro, nas duas metades do teste"],
        ], columns=["Item", "Valor"])), ""]

    # ---------------- puladas ----------------
    L += ["---", "", "## 2. Combinações puladas por poucas amostras", ""]
    if len(skipped):
        t = skipped.groupby(["side"]).agg(n=("combo", "size"), n_train_min=("n_train", "min"),
                                          n_train_max=("n_train", "max"), n_val_min=("n_val", "min")).reset_index()
        t["side"] = t["side"].map(SIDE_PT)
        L += [R.md_table(t, "{:.0f}"), ""]
    else:
        L += ["Nenhuma combinação pulada neste experimento.", ""]

    # ---------------- AUC por combinação ----------------
    L += ["---", "", "## 3. AUC de teste por combinação", "", f"![AUC por combinação]({fig_auc_by_combo(df, figs)})", "",
          f"![nº de amostras x AUC]({fig_n_vs_auc(df, figs)})", "",
          "Se a AUC não sobe com o nº de amostras de treino, a hipótese de \"poucas operações por combinação\" "
          "não é a explicação principal; se sobe, sugere que faltam dados, não que a tarefa seja impossível.", ""]

    # ---------------- comparação ----------------
    agg_auc = aggregated_intra_auc(lstm_runs, lstm_tag) if lstm_runs and lstm_tag else None
    boost_auc = combo_boosting_auc(combo_run_dir, combo_tag) if combo_run_dir and combo_tag else None
    if agg_auc is not None or boost_auc is not None:
        L += ["---", "", "## 4. Comparação com o modelo agregado e com o boosting por combinação", "",
              f"![Comparação]({fig_compare(df, agg_auc, boost_auc, figs)})", ""]
        if agg_auc is not None:
            L += [f"Referência recalculada de `{lstm_tag}` (AUC intra-combinação do modelo agregado, "
                  "mesma definição da seção 5 dos relatórios da LSTM agregada).", ""]
        if boost_auc is not None:
            L += [f"Referência recalculada de `{combo_tag}` (protocolo A, boosting raso por combinação).", ""]

    # ---------------- P/L ----------------
    L += ["---", "", "## 5. P/L por metade do teste", "",
          "Cada combinação tem sua própria divisão do teste em duas metades cronológicas (os dias em que ela "
          "de fato gerou trades); por isso as janelas de combinações diferentes não são idênticas dia a dia.", "",
          f"![P/L por janela]({fig_pl_windows(df, figs)})", ""]
    tj = ok.groupby("side").agg(
        j1_sem=("janela1_pl_sem_filtro", "mean"), j1_lstm=("janela1_pl_aceitas", "mean"), j1_n=("janela1_n_aceitas", "mean"),
        j2_sem=("janela2_pl_sem_filtro", "mean"), j2_lstm=("janela2_pl_aceitas", "mean"), j2_n=("janela2_n_aceitas", "mean")).reset_index()
    tj["side"] = tj["side"].map(SIDE_PT)
    L += [R.md_table(tj, "{:.2f}"), ""]

    L += ["---", "", "## 6. Limitações e ressalvas", "",
          "- **Poucas amostras por combinação** é a limitação central deste modo (por construção): mesmo as combinações "
          "treinadas usam ordens de grandeza menos dados que o modelo agregado, então os resultados individuais são ruidosos.",
          "- **Janelas de P/L não alinhadas entre combinações** (ver seção 5): a comparação é de ordem de grandeza.",
          "- **P/L bruto**, sem custos; o cache lateral já embute o spread na probabilidade de atingir o alvo (não cruza o book duas vezes).",
          "- **Seleção entre muitas combinações**: com dezenas de (combinação, fold), olhe os agregados por lado, não uma linha isolada.",
          "", "## Apêndice: reprodução", "", "```",
          f"python src/report_percombo.py --run-dir {run_dir} --tag {tag}" +
          (f" --lstm-runs {lstm_runs} --lstm-tag {lstm_tag}" if lstm_runs else "") +
          (f" --combo-run-dir {combo_run_dir} --combo-tag {combo_tag}" if combo_run_dir else ""),
          "```", ""]

    out = Path(out_dir) / f"relatorio_percombo_{tag}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Relatório: {out}  ({figs.n} figuras em {figs.dir})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="lstm_runs")
    ap.add_argument("--tag", default="lstm_v3_percombo")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--lstm-runs", default=None, help="pasta de experimentos da LSTM agregada (p/ AUC intra-combinação de referência)")
    ap.add_argument("--lstm-tag", default=None)
    ap.add_argument("--combo-run-dir", default=None, help="pasta de experimentos de src/combo_models.py (p/ AUC do boosting de referência)")
    ap.add_argument("--combo-tag", default=None)
    ap.add_argument("--notes", default=None)
    a = ap.parse_args()
    build(a.run_dir, a.tag, a.out_dir, a.lstm_runs, a.lstm_tag, a.combo_run_dir, a.combo_tag, a.notes)
