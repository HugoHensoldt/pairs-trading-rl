# Pairs Trading BOVA11 × WINM21 — Aprendizado por Reforço

Investigação de aprendizado por reforço (PPO) pra arbitragem estatística
intraday entre o futuro WIN e o ETF BOVA11.

**Comece por [`docs/README.md`](docs/README.md)** — guia de leitura dos
relatórios com os números, tabelas e figuras de cada rodada. O código serve
pra reproduzir esses números, não é o ponto de entrada.

## Estrutura do repositório

- **`docs/`** — relatórios (`relatorio_*.md`), tabelas resumidas
  (`analysis_*/*.csv`) e figuras (`img/`); ver [`docs/README.md`](docs/README.md).
- **`src/`** — todo o código. `rl_trading_pipeline.py` é o núcleo (ambiente
  `HedgedPairEnv`, treino PPO, avaliação); os demais `analyze_*.py`/
  `report_*.py` leem os artefatos de uma rodada e geram as tabelas/figuras
  dos relatórios. `synthetic_pair.py` gera o par cointegrado sintético.
  `lstm_pipeline.py`/`combo_models.py` são o caminho supervisionado
  abandonado.
- **`slurm/`** — drivers de submissão pro cluster Santos Dumont (um
  `submit_*.sh` por rodada/campanha; ver o cabeçalho de cada um).
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
