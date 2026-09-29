# Guia dos relatórios

Ordem de leitura sugerida — cada relatório documenta o método da rodada e
tem os números, tabelas e figuras que ele cita:

1. **[`relatorio_resultados_v5.md`](relatorio_resultados_v5.md)** — o
   resultado principal do agente PPO com a recompensa hedgeada (par WIN +
   BOVA11, P/L real menos custos). Lucra em validação e teste nas 4 seeds,
   mas o lucro não sobrevive a meio-spread nem a 1 tick de latência.
2. **[`relatorio_diagnostico_sintetico.md`](relatorio_diagnostico_sintetico.md)**
   — por que um agente com só preços (sem o spread pronto) falha num par
   **sintético** onde a arbitragem existe por construção: colapso de
   entropia em ~3,4M timesteps, dominado pelo custo de transação da política
   inicial. Inclui a comparação de 5 formatos de *reward shaping*
   (`none`/`opp_flat`/`opp_wrong`/`pbrs`/`regret`) — `pbrs` (shaping baseado
   em potencial, Ng et al. 1999) é o que resolve o colapso.
3. **[`relatorio_walkforward.md`](relatorio_walkforward.md)** — o mesmo
   agente da v5 avaliado em **3 janelas de treino** (não só uma) nos dados
   reais, com e sem o shaping `pbrs` que funcionou no sintético. Testa
   diretamente se o achado do item 2 se sustenta fora do par sintético.
4. Caminho alternativo, abandonado: uma LSTM supervisionada filtrando
   oportunidades de um backtest de bandas fixas (`relatorio_lstm_*.md`,
   `relatorio_combo_*.md`) — não superou o acaso de forma consistente; ver a
   "Lição" em cada relatório para o porquê.

Os demais `relatorio_*.md` são passos intermediários (v2/v4, sweep de
hiperparâmetros) mantidos por rastreabilidade.

## O que mais tem nesta pasta

- **`analysis_*/`** — tabelas resumidas (CSV) por trás de cada relatório.
- **`img/`** — figuras, geradas pelos `analyze_*.py`/`report_*.py`
  correspondentes de `src/`.
