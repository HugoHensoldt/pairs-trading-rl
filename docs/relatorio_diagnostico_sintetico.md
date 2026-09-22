# Relatório — diagnóstico estrutural com par sintético cointegrado (estado "só preços")

> Experimento no Santos Dumont em 21/09/2026 (`syn_coint_s0` e `syn_null_s0`, 60M timesteps cada, seed 0), disparado pelo driver autônomo `slurm/submit_synthetic.sh`. Objetivo: separar "os dados não têm arbitragem capturável" de "o RL não consegue aprender arbitragem", usando um par WIN×BOVA11 **sintético** com cointegração conhecida (`src/synthetic_pair.py`) e o estado "só preços" pedido pelo usuário (`STATE_KIND=raw`, sem preço justo/razão/resíduo prontos — só os 4 preços atuais e features genéricas de mercado). Toda a análise vem de arquivos gravados pelo pipeline, copiados em `sdumont_backup_sintetico/`; as figuras são geradas por `src/analyze_synthetic.py`.

## 0. Resumo

1. **Existe arbitragem real e capturável no sintético, e uma regra causal simples a captura.** Com o spread verdadeiro (conhecido só porque os dados são sintéticos), uma regra de limiar (entra a 2σ, sai ao cruzar 0) rende **+62,6 R$/dia** no teste, com 10,7 negócios/dia, positiva em 30 de 30 dias.
2. **O agente de RL, com o estado "só preços", captura ~0% disso.** O checkpoint escolhido pela validação faz **0,067 negócios/dia** (2 negócios em 30 dias) na validação e **zero negócios** no teste — empata exatamente com o baseline flat (R$ 0,00). Nenhum checkpoint da curva de avaliação chegou perto da regra ótima (Fig. 2).
3. **A causa raiz é um colapso de entropia muito rápido, que trava a política num atrator de "quase não operar" antes de ela explorar o regime lucrativo.** Em **35 de 611 atualizações de PPO (5,7% do orçamento, 3,4M de 60M timesteps)**, a entropia já caiu de ~1,0 para <0,01 nats e a variância explicada do crítico já passa de 0,74 (Fig. 1). Isso acontece antes de 3 episódios completos por ambiente (~129 de ~2.222 episódios do treino inteiro).
4. **O mecanismo é o custo de transação dominando o gradiente inicial.** A política aleatória do início faz ~8.395 negócios por episódio, custando (sem a escala de recompensa ×5) ≈ −83 mil R$/episódio — um castigo grande, imediato e presente em qualquer estado. "Parar de operar" é um gradiente fácil e universal; "operar só quando o spread passa de 2σ" é um sinal raro (~10 vezes em 27 mil ticks) que precisa de uma combinação específica dos preços. O primeiro vence de longe antes do segundo aparecer.
5. **Quando a política treinada opera mais, ela perde dinheiro — não fica perto do lucro da regra ótima.** Nos checkpoints intermediários com mais negócios (1,1–2,4/dia, ver `eval/checkpoint_curve.csv`), o P/L é negativo (−22 a −332 R$ em 30 dias). O agente nunca aprendeu a cronometrar as entradas; só aprendeu, cedo, que operar pouco dói menos que operar muito.
6. **Os testes de diagnóstico são consistentes com essa leitura, não a contradizem.** A resposta a impulsos de preço é **~0,000** (a política não reage a choques, sinal de logits saturados). O alinhamento com o spread verdadeiro e com métodos clássicos (razão com média móvel, OLS, Kalman, cópula) é positivo mas fraco (Spearman 0,05–0,10) — um resíduo do gradiente correto captado nas ~35 atualizações antes do colapso, fraco demais para virar ação. No controle negativo (`syn_null_s0`, sem arbitragem por construção), o agente opera mais (0,35/dia) e tem alinhamento **negativo**, mostrando que o padrão do `coint` não é um artefato do driver — é específico do embate custo-inicial × sinal-raro.
7. **Sem `VecNormalize` nem qualquer normalização de recompensa no pipeline.** O choque de custo inicial entra cru no cálculo de vantagem do PPO; só há um multiplicador fixo (`HEDGE_REWARD_SCALE=5,0`).

**Leitura geral:** a falha não é falta de sinal nos dados — aqui o sinal existe por construção e uma regra causal simples o captura. É um problema estrutural do treino de RL: o PPO, do jeito que está configurado (recompensa não normalizada, `ent_coef` fixo baixo, episódios de 27 mil ticks com rollouts de 2.048 por ambiente), converge para "não operar" muito antes de ter chance de descobrir a estratégia lucrativa. Isso confirma, num cenário controlado, o mesmo padrão de colapso de entropia já visto na campanha com dados reais (`docs/relatorio_resultados_v5.md`, §2.6: entropia < 0,05 em ~1,7M passos nas 4 seeds).

---

## 1. Configuração

| Item | Valor |
|---|---|
| Estado (22 dim.) | 4 preços atuais em log (bid/ask WIN e BOVA11), retornos de cada ativo em 10/100/1.000/4.000 ticks, spread bid-ask relativo, volatilidade de 100 ticks, hora e dia da semana cíclicos, posição, P/L não realizado — **sem preço justo, razão, resíduo ou spread entre os ativos** (`STATE_KIND=raw`, `src/rl_trading_pipeline.py::build_raw_features`) |
| Par sintético | `src/synthetic_pair.py`: BOVA11 = passeio aleatório geométrico; WIN = 1000×BOVA11 + deriva do dia + spread Ornstein-Uhlenbeck (`SYNTH_KIND=coint`: desvio 40 pts, meia-vida 300 ticks, deriva −30 pts/pregão, rollover a cada 32 pregões — parâmetros medidos nos dados reais); `SYNTH_KIND=null`: WIN é um passeio aleatório independente (controle negativo, sem arbitragem) |
| Recompensa | `HedgedPairEnv`: par hedgeado, mesma contabilidade da campanha v5 (WIN + `N_BOVA = P_WIN/(5·P_BOVA)` ações de BOVA11 em sentido oposto; custo R$ 0,25/lado do WIN + 0,023%×valor da perna BOVA11; preços mid, sem meio-spread; recompensa ×5 só para o PPO) |
| PPO | lr 3e-4, `n_steps` 2048 × 48 envs (98.304 timesteps/atualização, 611 atualizações no total), batch 4096, 10 épocas, γ = 0,999999, λ = 0,95, `ent_coef` = 0,01, MLP [64, 64], `DummyVecEnv`, sem `VecNormalize` |
| Orçamento | 60M timesteps por execução (4 chunks de ~20 min) |
| Janelas | 300 pregões sintéticos de treino, 30 de validação, 30 de teste (mesma proporção da v5) |
| Seleção | melhor checkpoint só pela validação (P/L real do par), a cada 5M timesteps |

---

## 2. Referência: o que existe para ser capturado

`src/synthetic_benchmark.py` escolhe, nos 300 dias de treino, o melhor limiar (múltiplo do desvio do spread) para uma regra causal simples — entra quando o spread verdadeiro passa do limiar, sai quando cruza 0 — e mede o resultado nos 30 dias de teste:

| Limiar (× desvio do spread) | 1,0σ | 1,5σ | 2,0σ (escolhido) | 2,5σ | 3,0σ | 3,5σ |
|---|---|---|---|---|---|---|
| P/L médio/dia no treino (R$) | −92,5 | +32,0 | **+55,1** | +33,3 | +15,5 | +4,7 |

| | Valor |
|---|---|
| **Regra ótima causal no teste** | **+62,6 R$/dia** (+1.878 R$ em 30 dias), 10,7 negócios/dia, **30/30 dias positivos** |
| Mesma regra com meio-spread bid/ask nas duas pernas | +30,6 R$/dia |
| Teto do oráculo (previsão perfeita, 10 dias) | +692,9 R$/dia |

A regra usa só informação causal (o spread no tick atual) e nunca vê o futuro. O teto do oráculo mostra que ainda sobra bastante espaço acima da regra simples — mas o agente treinado não chegou nem perto da regra, muito menos do teto.

---

## 3. Resultado do agente

![Colapso de entropia](img/v6_fig1_colapso_entropia.png)

**Fig. 1.** Dinâmica de treino de `syn_coint_s0`: recompensa por episódio (symlog), entropia (symlog), variância explicada do crítico e negócios por episódio, todos contra timesteps. A linha tracejada marca a atualização 35/611 (3,4M timesteps), onde a entropia já caiu abaixo de 0,01 nats.

| Atualização de PPO | Timesteps | Entropia (nats) | Variância explicada do crítico |
|---|---|---|---|
| 14 | 1,4M | −0,51 (ainda alta) | ~0 |
| 26 | 2,6M | −0,077 | 0,03 |
| **35** | **3,4M** | **−0,007** | **0,74** |
| 38 | 3,7M | −0,006 | 0,88 |

Com 48 ambientes e episódios de 27 mil ticks, um episódio completo leva ~13 atualizações; o colapso ocorre antes de 3 episódios completos por ambiente (~129 de ~2.222 episódios do treino inteiro, ~6%).

![Checkpoints contra a regra ótima](img/v6_fig2_checkpoints_vs_regra.png)

**Fig. 2.** P/L real médio por dia de cada checkpoint avaliado (validação e teste), contra a regra ótima causal (+62,6 R$/dia). Nenhum checkpoint chega perto.

| Timesteps (M) | Val, negócios/dia | Val, P/L (30 dias) | Teste, negócios/dia | Teste, P/L (30 dias) |
|---|---|---|---|---|
| 5,0 | 0,03 | −13,7 | 0,00 | 0,0 |
| 10,0 – 15,0 | 0,00 | 0,0 | 0,00 | 0,0 |
| 27,6 | 1,10 | **−147,8** | 0,43 | −83,3 |
| 32,6 | 1,43 | −73,6 | 1,07 | −22,2 |
| **40,0 (melhor de validação)** | **0,07** | **+12,1** | **0,00** | **0,0** |
| 51,3 | 2,43 | −287,0 | 2,13 | −332,5 |
| 56,3 | 1,27 | −295,1 | 1,03 | −253,8 |

**Nos 6 checkpoints em que o agente operou mais de 1 vez/dia, o P/L foi negativo em todos.** O checkpoint escolhido (40M) é o que mais se parece com "não operar" (0,07 negócios/dia na validação, 0 no teste) — a validação escolheu o mais próximo do flat porque toda alternativa que ele descobriu perde dinheiro, não porque encontrou a estratégia lucrativa.

| | best_val (checkpoint 40M) | last (checkpoint 60M) | baseline flat |
|---|---|---|---|
| P/L real, validação (30 dias) | +12,1 R$ | −5,3 R$ | 0,0 R$ |
| P/L real, teste (30 dias) | **0,0 R$** | 0,0 R$ | 0,0 R$ |
| Negócios/dia, validação | 0,067 | 0,067 | 0 |
| Negócios/dia, teste | **0,000** | 0,000 | 0 |
| **Fração da regra ótima capturada (teste)** | **0%** | 0% | — |

---

## 4. Diagnóstico de arbitragem (`src/diagnose_agent.py`)

### A) Alinhamento com métodos clássicos (Spearman entre a preferência do agente e −desvio; 60 pregões)

| Sinal | Spearman médio/pregão | IC95% | Pregões positivos |
|---|---|---|---|
| razão com média móvel de 3.000 ticks | 0,096 | [0,071; 0,120] | 51/60 |
| OLS rolante (3.000 ticks) | 0,087 | [0,061; 0,112] | 49/60 |
| Kalman | 0,095 | [0,067; 0,124] | 47/60 |
| Cópula gaussiana | 0,082 | [0,064; 0,100] | 51/60 |
| **spread verdadeiro (`truth`)** | **0,051** | **[0,016; 0,085]** | 43/60 |

Positivo e estatisticamente distinto de 0 nos 5 sinais, mas fraco (0,05–0,10, numa escala de −1 a 1) — e **`pct_ticks_com_preferencia_forte` é 0,005%**: a política quase nunca assume uma preferência clara. É um resíduo de aprendizado correto, não uma política operante.

### B) Resposta a impulso (degrau de +30/+60 pts nos preços; política avaliada com posição flat)

| Δ (pts) | Cenário | Δ preferência | Esperado se fosse arbitragem |
|---|---|---|---|
| 60 | WIN sobe | 0,004 (IC cruza 0) | negativo |
| 60 | WIN cai | 0,006 | positivo |
| 60 | BOVA11 sobe | 0,000 | positivo |
| 60 | os dois sobem | 0,004 | ~0 |

**Todos os valores são ~0,000 (IC95% cruzando 0 ou quase).** A política não responde a nenhum dos choques testados — nem na direção certa, nem na errada. Isso é evidência direta de logits saturados: o colapso de entropia não deixou a política "confiante e correta às vezes", deixou-a **constante**, indiferente ao estado.

### C) Placebo de emparelhamento (BOVA11 adulterado; P/L calculado com preços verdadeiros)

| BOVA11 visto pelo agente | P/L total (30 dias) | Negócios/dia |
|---|---|---|
| original | +12,1 R$ | 0,03 |
| atrasado 10/100/1.000 ticks | +16,9 / −6,5 / −79,0 R$ | 0,08 / 0,15 / 0,47 |
| congelado | +3,6 R$ | 0,05 |
| de outro pregão | −63,7 R$ | 0,27 |

Com tão poucos negócios (2 no total, ver D), este teste não tem poder estatístico — está aqui por completude, não como evidência.

### D) Eventos: desvio dos métodos clássicos nas 2 entradas do agente

| Sinal | Desvio a favor na entrada (pts) | 100 ticks depois | Na saída |
|---|---|---|---|
| razão com média móvel | −13,0 | +18,8 | −82,1 |
| spread verdadeiro | +16,0 | +48,9 | −60,6 |

Amostra de 2 negócios — não dá para tirar conclusão, só ilustra o quão raramente o agente opera.

### Controle negativo (`syn_null_s0`, sem arbitragem por construção)

| | `coint` | `null` |
|---|---|---|
| Negócios/dia (best_val, val) | 0,067 | 0,35 |
| Alinhamento com métodos clássicos (Spearman) | +0,05 a +0,10 | **−0,23 a −0,32** |
| P/L do "original" no placebo (30 dias) | +12,1 R$ | +241,7 R$ (ruído; 0,35 negócio/dia) |

No `null` o agente opera mais e tem alinhamento **negativo** com os mesmos métodos clássicos (que, sem cointegração real, não deveriam prever nada). Isso mostra que o padrão do `coint` — quase não operar, alinhamento fraco mas positivo — não é um artefato fixo do pipeline; é a assinatura específica do colapso ocorrendo bem no início do treino, antes de explorar o sinal real que só existe no `coint`.

---

## 5. Limitações

- **1 seed por cenário.** Não dá para saber se outra inicialização escaparia do atrator "não operar" por acaso.
- **O placebo (C) e os eventos (D) têm baixo poder estatístico** porque o agente quase não opera (2 negócios no `coint`).
- **Os parâmetros do Kalman e da cópula (A) são escolhas razoáveis, não otimizadas** — servem de referência de alinhamento, não de estratégias calibradas.
- **O `ent_coef` (0,01) e a escala de recompensa (×5) foram herdados da campanha v5** (dados reais), não recalibrados para o sintético.

---

## 6. Próximos passos (hipóteses a testar, nenhuma executada)

Decorrem diretamente do mecanismo identificado (§0.4, §3):

1. **`ent_coef` maior ou com cronograma decrescente**, para adiar o colapso além das ~35 atualizações atuais.
2. **Normalizar a recompensa** (ex.: `VecNormalize` do próprio `stable_baselines3`, ou recorte/normalização manual), para o choque inicial de custo (~−83 mil R$/episódio sem escala) não dominar sozinho o gradiente antes de qualquer sinal de lucro aparecer.
3. **Aquecer com a política de arbitragem** (pergunta original do usuário que motivou este experimento): agora há evidência concreta de que o problema é justamente não escapar do atrator "não operar" — começar de uma política que já opera evita esse atrator por construção. Testável primeiro no sintético `coint`, onde a fração capturada da regra ótima é a métrica de sucesso.
4. **Reduzir o atraso de atribuição de crédito**: aumentar `n_steps` (hoje cobre só ~7,6% de um episódio) ou truncar o episódio.
5. **Curva de custo crescente**: começar o treino com custo de transação baixo (ou zero) e subir gradualmente até o valor real, para o agente aprender a cronometrar entradas antes de aprender a temer o custo.

---

## Apêndice: arquivos e reprodução

- **Resultados brutos:** `sdumont_backup_sintetico/` (`sintetico_progresso.md` com o log completo da campanha; `pairs-trading-rl/src/runs/syn_{coint,null}_s0/fold1/` com `state.json`, `val_curve.csv`, `logs/chunk*/progress.csv`, `eval/*`, `diag/*`).
- **Figuras:** `python src/analyze_synthetic.py` (no WSL/venv com matplotlib).
- **Como foi gerado:** `slurm/submit_synthetic.sh` (driver autônomo) + `slurm/submit_diag.sbatch`; código em `src/synthetic_pair.py`, `src/synthetic_benchmark.py`, `src/diagnose_agent.py`, `src/rl_trading_pipeline.py` (`build_raw_features`, `STATE_KIND=raw`).
- **Branch/commit:** `estado-simplificado-v4`, commit `ba3c7f5` (código que gerou esta rodada).
