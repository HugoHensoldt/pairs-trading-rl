"""
Analisa os resultados do sweep de TARGET_N_PASSADAS (ver
slurm/submit_sweep.sh e docs/relatorio_sweep_passadas.md).

Roda LOCALMENTE (não no cluster), depois de trazer de volta do Santos
Dumont:
  - os arquivos .out de cada job do sweep (job-name
    "pairs-rl-fold1-p<N>", logo arquivo "pairs-rl-fold1-p<N>-<jobid>.out"
    -- ver slurm/submit_all_folds.sh)
  - a árvore de logs CSV gerada pelo SB3
    (logs/p<N>/fold1/chunk<K>/progress.csv -- ver train_ppo() em
    rl_trading_pipeline.py)

Ambos default para procurar recursivamente a partir de --logs-root
(default: raiz do repo, cobre tanto sdumont_backup_sweep/pairs-rl-logs/
quanto src/logs/ se o usuário rodar localmente).

Uso:
    python src/analyze_sweep.py [--logs-root DIR] [--out-dir DIR]

Gera em --out-dir (default: docs/):
  - sweep_resultados.csv        -- tabela final por (passadas, fold, conjunto)
  - sweep_resultados.md         -- mesma tabela em Markdown
  - img/sweep_p<N>.png          -- ep_rew_mean (treino) vs eval/mean_reward
                                    (validação) no mesmo eixo, 1 figura por
                                    valor de target_n_passadas
  - img/sweep_overfitting_gap.png -- gráfico-resumo: gap de overfitting
                                    (ep_rew_mean final - eval/mean_reward
                                    final) vs target_n_passadas
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1) Parsing dos .out -- tabela final de lucro/negócios/taxa de acerto por
#    (target_n_passadas, fold, conjunto), reaproveitando o formato de print
#    de evaluate_policy() (rl_trading_pipeline.py:829-853), o mesmo já lido
#    manualmente pra escrever docs/relatorio_resultados_v2.md.
# --------------------------------------------------------------------------

FILENAME_RE = re.compile(r"pairs-rl-fold(?P<fold>\d+)(?:-(?P<tag>[A-Za-z0-9.]+))?-\d+\.out$")
RUN_TAG_LINE_RE = re.compile(r"^RUN_TAG=(\S*)")
PASSADAS_LINE_RE = re.compile(r"Passadas-alvo/dia\s*:\s*([\d.]+)")
EVAL_HEADER_RE = re.compile(r"===\s*Fold\s+(\d+)\s+--\s+Avaliação em\s+(VALIDAÇÃO|TESTE)\s*===")
LUCRO_RE = re.compile(r"Lucro total agregado:\s*(-?[\d.]+)\s*pontos\s*\(R\$\s*(-?[\d.]+)\)")
NEGOCIOS_RE = re.compile(r"Negócios fechados:\s*(\d+)")
TAXA_RE = re.compile(r"Taxa de acerto:\s*([\d.]+)%")


def _passadas_from_tag(tag):
    """Extrai o valor numérico de target_n_passadas de um RUN_TAG "p<N>"
    (ex.: "p10" -> 10.0). Retorna None se o tag não seguir esse padrão."""
    if not tag:
        return None
    m = re.fullmatch(r"p([\d.]+)", tag)
    return float(m.group(1)) if m else None


def parse_out_file(path: Path):
    """Extrai registros de avaliação (um por bloco VALIDAÇÃO/TESTE
    encontrado) e o target_n_passadas efetivo desse arquivo."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    fname_match = FILENAME_RE.search(path.name)
    tag_from_name = fname_match.group("tag") if fname_match else None

    run_tag = None
    for line in lines:
        m = RUN_TAG_LINE_RE.match(line)
        if m:
            run_tag = m.group(1) or None
            break
    if run_tag is None:
        run_tag = tag_from_name

    passadas_logged = None
    for line in lines:
        m = PASSADAS_LINE_RE.search(line)
        if m:
            passadas_logged = float(m.group(1))
            break

    passadas_from_tag = _passadas_from_tag(run_tag)
    if passadas_from_tag is not None and passadas_logged is not None:
        if abs(passadas_from_tag - passadas_logged) > 1e-6:
            print(f"[aviso] {path.name}: RUN_TAG sugere target_n_passadas="
                  f"{passadas_from_tag}, mas o log imprime {passadas_logged} "
                  f"-- usando o valor do log (mais confiável).")
    target_n_passadas = passadas_logged if passadas_logged is not None else passadas_from_tag

    records = []
    n = len(lines)
    for i, line in enumerate(lines):
        m = EVAL_HEADER_RE.search(line)
        if not m:
            continue
        fold_num, label = int(m.group(1)), m.group(2)

        lucro_pts = lucro_brl = negocios = taxa_acerto = None
        for j in range(i + 1, min(i + 60, n)):
            if lucro_pts is None:
                lm = LUCRO_RE.search(lines[j])
                if lm:
                    lucro_pts, lucro_brl = float(lm.group(1)), float(lm.group(2))
            if negocios is None:
                nm = NEGOCIOS_RE.search(lines[j])
                if nm:
                    negocios = int(nm.group(1))
            if taxa_acerto is None:
                tm = TAXA_RE.search(lines[j])
                if tm:
                    taxa_acerto = float(tm.group(1))
            if lucro_pts is not None and negocios is not None and taxa_acerto is not None:
                break

        if lucro_pts is None:
            # bloco de avaliação sem resultado completo (ex.: pulado por
            # falta de tempo, "AVALIAÇÃO ... PULADA") -- ignora
            continue

        records.append({
            "run_tag": run_tag,
            "target_n_passadas": target_n_passadas,
            "fold": fold_num,
            "conjunto": label,
            "lucro_total_pontos": lucro_pts,
            "lucro_total_brl": lucro_brl,
            "negocios": negocios,
            "taxa_acerto_pct": taxa_acerto,
            "lucro_medio_negocio": (lucro_pts / negocios) if negocios else np.nan,
            "source_file": path.name,
        })

    return records


def build_results_table(logs_root: Path) -> pd.DataFrame:
    all_records = []
    for out_path in sorted(logs_root.rglob("*.out")):
        all_records.extend(parse_out_file(out_path))

    if not all_records:
        print(f"[aviso] nenhum bloco de avaliação encontrado em {logs_root} (*.out)")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # entre múltiplos arquivos pro mesmo (run_tag, fold, conjunto) -- ex.:
    # um job final + um job de recuperação EVAL_ONLY_FOLD -- fica só o
    # último parseado (ordem de rglob é alfabética/por caminho, não
    # temporal; ok pra esse caso raro, não esperado no sweep normal).
    df = df.drop_duplicates(subset=["run_tag", "fold", "conjunto"], keep="last")
    conjunto_order = {"VALIDAÇÃO": 0, "TESTE": 1}
    df["_ordem_conjunto"] = df["conjunto"].map(conjunto_order)
    df = df.sort_values(["target_n_passadas", "fold", "_ordem_conjunto"]).drop(columns="_ordem_conjunto")
    return df.reset_index(drop=True)


def write_results_table(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_resultados.csv"
    df.to_csv(csv_path, index=False)

    md_lines = [
        "| target_n_passadas | Fold | Conjunto | Lucro total | Taxa de acerto | Negócios | Lucro médio/negócio |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['target_n_passadas']:.0f} | {row['fold']} | {row['conjunto']} | "
            f"{row['lucro_total_pontos']:.1f} pts (R$ {row['lucro_total_brl']:.2f}) | "
            f"{row['taxa_acerto_pct']:.1f}% | {row['negocios']} | {row['lucro_medio_negocio']:.2f} |"
        )
    md_path = out_dir / "sweep_resultados.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Tabela de resultados: {csv_path}  |  {md_path}")


# --------------------------------------------------------------------------
# 2) Curvas de treino vs. validação (progress.csv, logger CSV do SB3) --
#    ep_rew_mean (treino) e eval/mean_reward (validação) no mesmo eixo de
#    time/total_timesteps. Ver train_ppo() em rl_trading_pipeline.py.
# --------------------------------------------------------------------------

def load_progress_csv(logs_root: Path, run_tag: str, fold: int) -> pd.DataFrame:
    """Concatena todos os chunks (logs/<run_tag>/fold<N>/chunk*/progress.csv)
    em ordem de time/total_timesteps -- não precisa de offset manual porque
    PPO.load() restaura num_timesteps entre chunks (ver train_ppo)."""
    pattern = f"**/{run_tag}/fold{fold}/chunk*/progress.csv"
    paths = sorted(logs_root.glob(pattern))
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    if "time/total_timesteps" not in df.columns:
        return pd.DataFrame()
    return df.sort_values("time/total_timesteps").reset_index(drop=True)


def plot_train_vs_val(df: pd.DataFrame, run_tag: str, target_n_passadas, out_path: Path):
    import matplotlib.pyplot as plt

    # reindex (não indexação direta): se o treino foi curto demais pra
    # completar 1 rollout, a coluna "rollout/ep_rew_mean" nem chega a
    # existir no CSV (não é só NaN) -- reindex preenche com NaN em vez
    # de estourar KeyError, e o dropna() abaixo já trata esse caso.
    train = df.reindex(columns=["time/total_timesteps", "rollout/ep_rew_mean"]).dropna()
    val = df.reindex(columns=["time/total_timesteps", "eval/mean_reward"]).dropna()
    if train.empty and val.empty:
        print(f"[aviso] {run_tag}: progress.csv sem rollout/ep_rew_mean nem "
              f"eval/mean_reward -- nada pra plotar (treino provavelmente "
              f"muito curto, ver ressalva de updates PPO insuficientes).")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if not train.empty:
        ax.plot(train["time/total_timesteps"], train["rollout/ep_rew_mean"],
                marker="o", markersize=3, label="ep_rew_mean (treino)", color="#1f77b4")
    if not val.empty:
        ax.plot(val["time/total_timesteps"], val["eval/mean_reward"],
                marker="s", markersize=4, label="eval/mean_reward (validação)", color="#d62728")
    ax.set_xlabel("total_timesteps")
    ax.set_ylabel("recompensa média por episódio (pontos)")
    label = f"{target_n_passadas:.0f}" if target_n_passadas is not None else run_tag
    ax.set_title(f"Treino vs. validação -- target_n_passadas={label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Figura: {out_path}")


def compute_overfitting_gap(df: pd.DataFrame, n_last=5):
    """gap = ep_rew_mean (treino, ponto de rollout mais recente até ali) -
    eval/mean_reward (validação), média dos últimos `n_last` pontos de
    avaliação -- métrica-resumo pedida: distância entre as duas curvas."""
    train = df.reindex(columns=["time/total_timesteps", "rollout/ep_rew_mean"]).dropna()
    val = df.reindex(columns=["time/total_timesteps", "eval/mean_reward"]).dropna()
    if train.empty or val.empty:
        return np.nan
    merged = pd.merge_asof(
        val.sort_values("time/total_timesteps"),
        train.sort_values("time/total_timesteps"),
        on="time/total_timesteps", direction="backward",
    ).dropna()
    if merged.empty:
        return np.nan
    gap = merged["rollout/ep_rew_mean"] - merged["eval/mean_reward"]
    return gap.tail(n_last).mean()


def plot_overfitting_gap_summary(gaps: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt

    gaps = gaps.dropna(subset=["target_n_passadas", "gap"]).sort_values("target_n_passadas")
    if gaps.empty:
        print("[aviso] sem pontos suficientes pra gráfico-resumo de overfitting gap")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gaps["target_n_passadas"], gaps["gap"], marker="o", color="#2ca02c")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("target_n_passadas")
    ax.set_ylabel("gap de overfitting (ep_rew_mean treino - eval/mean_reward), pontos")
    ax.set_title("Gap de overfitting vs. target_n_passadas (fold 1)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Figura-resumo: {out_path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("."),
                         help="raiz onde procurar *.out e logs/<tag>/fold<N>/chunk*/progress.csv (default: diretório atual)")
    parser.add_argument("--out-dir", type=Path, default=Path("docs"),
                         help="onde salvar sweep_resultados.{csv,md} e img/ (default: docs/)")
    args = parser.parse_args()

    results_df = build_results_table(args.logs_root)
    write_results_table(results_df, args.out_dir)

    run_tags = sorted({
        (row["run_tag"], row["target_n_passadas"], row["fold"])
        for _, row in results_df.iterrows()
    }) if not results_df.empty else []

    if not run_tags:
        print("Nenhum run_tag identificado nos .out -- pulando curvas de treino/validação.")
        return

    gap_rows = []
    for run_tag, target_n_passadas, fold in run_tags:
        progress_df = load_progress_csv(args.logs_root, run_tag, fold)
        if progress_df.empty:
            print(f"[aviso] sem progress.csv para run_tag={run_tag} fold={fold} "
                  f"(logs/{run_tag}/fold{fold}/chunk*/progress.csv) -- pulando curva")
            continue
        img_path = args.out_dir / "img" / f"sweep_{run_tag}.png"
        plot_train_vs_val(progress_df, run_tag, target_n_passadas, img_path)
        gap_rows.append({
            "run_tag": run_tag,
            "target_n_passadas": target_n_passadas,
            "fold": fold,
            "gap": compute_overfitting_gap(progress_df),
        })

    if gap_rows:
        gaps_df = pd.DataFrame(gap_rows)
        gaps_df.to_csv(args.out_dir / "sweep_overfitting_gap.csv", index=False)
        plot_overfitting_gap_summary(gaps_df, args.out_dir / "img" / "sweep_overfitting_gap.png")


if __name__ == "__main__":
    main()
