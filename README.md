# Pairs Trading BOVA11 x WINM21

Backtest de arbitragem estatística entre BOVA11 e WINM21, com dois
caminhos de aprendizado de máquina para otimizar as decisões de entrada
e saída:

- **`src/backtest_verify.py`** — replica o backtest original (bandas
  fixas de Bollinger + stops fixos `Re`/`Ri`), usado como referência para
  comparar qualquer abordagem nova.
- **`src/lstm_pipeline.py`** — abordagem supervisionada: uma LSTM
  classifica, a partir dos 120 ticks anteriores, se uma oportunidade de
  arbitragem sinalizada pelo backtest tende a dar Lucro ou Prejuízo.
- **`src/rl_trading_pipeline.py`** — abordagem por aprendizado por
  reforço: um agente PPO controla a posição tick a tick (entrar, segurar,
  sair), usando como estado uma janela do spread de mispricing entre o
  WIN e o preço justo implícito pelo BOVA11.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dados

Os scripts esperam arquivos `{numero}BOVA11.json` e `{numero}WINM21.json`
(tick a tick, um par por pregão) na pasta configurada em
`src/config.py` (`DATA_DIR`), atualmente `C:\Users\HugoV\tick_data`.
Essa pasta fica **fora** do projeto de propósito e não é versionada —
se você mudar de pasta, ajuste só o `DATA_DIR` em `src/config.py`.

## Uso

```bash
# Confirma que o backtest bate com a lógica original
python src/backtest_verify.py

# Treina e avalia a LSTM supervisionada
python src/lstm_pipeline.py

# Treina e avalia o agente de RL
python src/rl_trading_pipeline.py
```
