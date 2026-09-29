# Pairs Trading BOVA11 × WINM21 — Aprendizado por Reforço

Código e experimentos por trás do paper **"Reinforcement Learning for
Intraday Pairs Trading between an Index Future and an ETF"** (ICAIF 2026;
fonte em [`docs/tex/ICAIF26_pairs.tex`](docs/tex/ICAIF26_pairs.tex) / PDF em
[`docs/tex/ICAIF26_pairs.pdf`](docs/tex/ICAIF26_pairs.pdf); slides em
[`docs/tex/Apresentação_GA036_Pairs_Reforço_Beamer/`](docs/tex/Apresentação_GA036_Pairs_Reforço_Beamer/)).

Se você chegou aqui a partir do paper, **comece pelos relatórios em
`docs/`, não pelo código** — eles têm os números, tabelas e figuras citados
no texto, com o método de cada rodada documentado. O código serve pra
reproduzir esses números, não é o ponto de entrada.

## Onde estão os resultados (nesta ordem de leitura)

1. **[`docs/relatorio_resultados_v5.md`](docs/relatorio_resultados_v5.md)** —
   o resultado principal do agente PPO com a recompensa hedgeada (par WIN +
   BOVA11, P/L real menos custos). Lucra em validação e teste nas 4 seeds,
   mas o lucro não sobrevive a meio-spread nem a 1 tick de latência.
2. **[`docs/relatorio_diagnostico_sintetico.md`](docs/relatorio_diagnostico_sintetico.md)**
   — por que um agente com só preços (sem o spread pronto) falha num par
   **sintético** onde a arbitragem existe por construção: colapso de
   entropia em ~3,4M timesteps, dominado pelo custo de transação da política
   inicial. Inclui a comparação de 5 formatos de *reward shaping*
   (`none`/`opp_flat`/`opp_wrong`/`pbrs`/`regret`) — `pbrs` (shaping baseado
   em potencial, Ng et al. 1999) é o que resolve o colapso.
3. **[`docs/relatorio_walkforward.md`](docs/relatorio_walkforward.md)** — o
   mesmo agente da v5 avaliado em **3 janelas de treino** (não só uma) nos
   dados reais, com e sem o shaping `pbrs` que funcionou no sintético. Testa
   diretamente se o achado do item 2 se sustenta fora do par sintético.
4. Caminho alternativo, abandonado: uma LSTM supervisionada filtrando
   oportunidades de um backtest de bandas fixas (relatórios
   `docs/relatorio_lstm_*.md`, `docs/relatorio_combo_*.md`) — não superou o
   acaso de forma consistente; ver a "Lição" em cada relatório para o porquê.

Os demais `docs/relatorio_*.md` são passos intermediários (v2/v4, sweep de
hiperparâmetros) mantidos por rastreabilidade, não são citados no paper.

## Estrutura do repositório

- **`src/`** — todo o código. `rl_trading_pipeline.py` é o núcleo (ambiente
  `HedgedPairEnv`, treino PPO, avaliação); os demais `analyze_*.py`/
  `report_*.py` leem os artefatos de uma rodada e geram as tabelas/figuras
  dos relatórios. `synthetic_pair.py` gera o par cointegrado sintético.
  `lstm_pipeline.py`/`combo_models.py` são o caminho supervisionado
  abandonado.
- **`slurm/`** — drivers de submissão pro cluster Santos Dumont (um
  `submit_*.sh` por rodada/campanha; ver o cabeçalho de cada um).
- **`docs/`** — os relatórios (`relatorio_*.md`), tabelas resumidas
  (`docs/analysis_*/*.csv`) e figuras (`docs/img/`) — **isso é o que é
  versionado do lado de resultados**. O `docs/tex/` tem o paper e os slides.
- **`sdumont_backup*/`** (local, **não versionado**) — saída bruta de cada
  rodada do cluster (checkpoints, `eval/*.csv` por dia, logs), copiada de
  volta pra análise local. Cada relatório em `docs/` diz de qual backup ele
  veio; os CSVs/figuras que importam já estão extraídos em `docs/`, então
  você não precisa desses backups brutos pra ler os resultados — só pra
  reproduzir uma análise do zero.
- **`runs/`, `src/runs/`, `best_model/`** (local, não versionados) — saída
  de treino/avaliação local.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dados

Os scripts esperam arquivos `{numero}BOVA11.json` e `{numero}WINM21.json`
(tick a tick, um par por pregão) numa pasta apontada por `TICK_DATA_DIR`
(ou `DATA_DIR` em `src/config.py`). Essa pasta fica **fora** do projeto e
não é versionada — os dados de tick não são de distribuição pública.

## Reproduzindo uma rodada

`src/rl_trading_pipeline.py` é configurado inteiramente por variáveis de
ambiente (`ENV_KIND`, `STATE_KIND`, `SHAPING_KIND`, `TRAIN_SIZES`, etc. — ver
o `__main__` do arquivo para a lista completa). Exemplo, treino da v5:

```bash
cd src
ENV_KIND=hedged STATE_KIND=spread TRAIN_SIZES=360 N_VAL=30 N_TEST=30 \
TOTAL_TIMESTEPS=100000000 python rl_trading_pipeline.py
```

Os drivers em `slurm/` (ex.: `submit_walkforward.sh`,
`submit_walkforward_pbrs.sh`) mostram a configuração exata de cada rodada
citada nos relatórios, incluindo os valores de todas as variáveis de
ambiente. Depois de uma rodada terminar, o `analyze_*.py`/`report_*.py`
correspondente (ex.: `python src/analyze_walkforward.py`) regenera a tabela
e a figura do relatório a partir dos artefatos gravados pelo pipeline.
