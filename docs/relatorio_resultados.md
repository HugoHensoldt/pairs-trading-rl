# Relatório de Resultados (versão anterior — política treinada com custo de 0,5 pontos; avaliação corrigida para 2,5 pontos) — RL (PPO) para Pairs Trading BOVA11 x WINM21

Walk-forward com 3 folds, treinado no Santos Dumont (LNCC).

> Ver [relatorio_resultados_v2.md](relatorio_resultados_v2.md) para a
> rodada seguinte, com o custo de transação corrigido **desde o
> treino** (não só na avaliação, como aqui — ver nota da seção 2).

## 1. Metodologia

O agente opera **tick a tick** em um único pregão por episódio
(`ArbitrageTradingEnv`), decidindo a cada instante se fica flat,
comprado ou vendido no par BOVA11/WINM21. O treino usa PPO
(stable-baselines3) com paralelização de 48 ambientes (`SubprocVecEnv`,
um processo por núcleo).

A validação segue **walk-forward com janela expansiva**
(`generate_walk_forward_folds`, via `sklearn.TimeSeriesSplit`): a cada
fold, a janela de treino cresce (usa todos os pregões anteriores),
enquanto validação e teste são sempre os próximos pregões em ordem
cronológica — sem embaralhar dados no tempo.

| Fold | Dias de treino | Dias de validação | Dias de teste |
|---|---|---|---|
| 1 | 7 | 10 | 9 |
| 2 | 26 | 10 | 9 |
| 3 | 45 | 10 | 9 |

Cada fold recebe o mesmo orçamento de treino: **25.254.000 timesteps**
(orçamento calibrado por throughput medido, não por número fixo de
passadas pelos dados).

## 2. Configurações

**Espaço de estado (observação):** janela das últimas `n_ticks=120`
observações de 5 features por tick — spread de mispricing, spread
bid-ask, atribuição de movimento (BOVA vs. WIN), momentum de curto e de
longo prazo — achatada em vetor, concatenada com a posição atual
(one-hot, 3 valores) e o P&L não-realizado normalizado.

**Ações:** `Discrete(3)` — posição-alvo a assumir no tick: `0` = flat,
`1` = comprado, `2` = vendido.

**Função de recompensa:** a cada tick, a recompensa é a variação de
marcação a mercado (mid-price) da posição já aberta desde o tick
anterior, **menos** o custo de execução (metade do spread bid-ask)
sempre que uma perna abre ou fecha, **menos** uma taxa fixa de
corretora cobrada inteiramente na abertura da posição (não dividida no
fechamento — de propósito, para não penalizar adicionalmente a decisão
de realizar uma posição perdedora). No fim do pregão, qualquer posição
aberta é fechada automaticamente.

> **Nota sobre unidades e custo de transação (atualizada):** toda a
> recompensa, P&L e "Lucro total" deste relatório estão em **PONTOS do
> WIN**, não em R$ — o ambiente (`ArbitrageTradingEnv`) opera
> inteiramente em pontos brutos, sem nenhuma conversão para R$
> internamente. Para converter qualquer valor deste relatório para R$,
> multiplique por **0,20** (1 ponto do WIN = R$0,20).
>
> A política avaliada nas seções 4 e 5 foi **treinada** com a taxa fixa
> de corretora antiga e incorreta, de **0,5 pontos** (== R$0,10) por
> negociação — 5x menor que o valor real de **2,5 pontos** (==
> R$0,50). Os números das seções 4 e 5, porém, já foram **corrigidos
> retroativamente** para refletir a taxa correta: como a taxa é
> cobrada como um valor fixo por negociação completa (uma vez na
> abertura, não dividida no fechamento — ver "Função de recompensa"
> acima), a correção equivale a subtrair **2,0 pontos adicionais por
> negociação** (a diferença entre 2,5 e 0,5) de cada negócio.
>
> Essa correção é **exata** para "Lucro total", "Lucro médio/negócio" e
> para os percentis P5/P50 (mediana)/P95 da distribuição de P&L por
> negócio — um deslocamento constante desloca a soma, a média e
> qualquer percentil pelo mesmo valor, não importa o formato da
> distribuição. Ela **não é exata** para "Taxa de acerto", "Ganho
> médio" e "Perda média" (marcados com `*` abaixo): o deslocamento pode
> empurrar negócios que antes eram pequenos ganhos brutos (entre 0 e
> 2,0 pontos) para o lado de perda, e recalcular esses três campos
> corretamente exigiria os dados de P&L por negociação individual, que
> não foram retidos desta rodada. Eles permanecem com os valores
> **originais (custo de 0,5), não corrigidos** — na prática, a taxa de
> acerto real após a correção é igual ou **menor** que a mostrada, nunca
> maior.
>
> Importante: esta é uma correção da **contabilidade da avaliação**,
> não um retreino — a política em si ainda foi treinada sob o custo
> antigo (mais barato) e aprendeu um comportamento de negociação
> calibrado pra custos 5x menores que os reais. Os números abaixo
> mostram quanto essa mesma política (não recalibrada) teria
> ganho/perdido sob o custo real, e não o resultado de uma política
> treinada já com o custo real — isso é o que a rodada v2 faz (ver
> [relatorio_resultados_v2.md](relatorio_resultados_v2.md)).

**PPO — hiperparâmetros:**

| Parâmetro | Valor |
|---|---|
| Política | `MlpPolicy` |
| Taxa de aprendizado (`learning_rate`) | 3e-4 |
| `n_steps` (por ambiente) | 8192 |
| `batch_size` | 1024 |
| `gamma` | 0.999 |
| Ambientes paralelos (`n_train_envs`) | 48 |

## 3. Curvas de treino

![Curva de recompensa durante o treino](img/reward_curve.png)

A recompensa média por episódio (`ep_rew_mean`, medida durante o
treino, não durante avaliação) **não cresce de forma monotônica** em
nenhum dos 3 folds — o padrão observado é um formato de "U": a
recompensa piora na primeira metade do treino (o agente parece
aumentar a atividade de negociação, pagando mais custos de
execução/spread antes de aprender a filtrar melhor as entradas), e
melhora na segunda metade, terminando mais próxima (mas ainda abaixo)
do valor inicial em todos os folds. Nenhum dos 3 folds chegou a superar
seu próprio valor inicial de `ep_rew_mean` dentro do orçamento de
timesteps usado — sinal de que o treino provavelmente se beneficiaria
de mais timesteps/tempo de treino do que o orçamento atual permite.

> **Nota:** diferente das tabelas da seção 4, esta curva **não** foi
> (nem pode ser) corrigida para o custo de 2,5 pontos — ela é telemetria
> real do treino (`ep_rew_mean` agregado por episódio, logado pelo
> stable-baselines3), não uma contagem de negócios por ponto, então o
> deslocamento constante de -2,0/negócio usado na seção 4 não se aplica
> aqui de forma exata. A política que gerou esta curva foi de fato
> treinada sob o custo antigo (0,5).

## 4. Resultados de avaliação

*(valores em **pontos do WIN**, não R$ — ver nota de unidades na seção 2;
multiplique por 0,20 para converter para R$. Custo de transação
corrigido retroativamente para 2,5 pontos/negócio — ver nota da seção
2 sobre o que é exato e o que não é nesta correção)*

| Fold | Conjunto | Lucro total | Taxa de acerto* | Negócios | Lucro médio/negócio |
|---|---|---|---|---|---|
| 1 | Validação | 1.135,0 | 56,1%* | 578 | 1,96 |
| 1 | Teste | -1.945,0 | 45,2%* | 496 | -3,92 |
| 2 | Validação | -762,5 | 53,6%* | 649 | -1,17 |
| 2 | Teste | -400,0 | 51,3%* | 838 | -0,48 |
| 3 | Validação | 237,5 | 47,7%* | 1.657 | 0,14 |
| 3 | Teste | 4.205,0 | 49,4%* | 1.196 | 3,52 |

*\* Taxa de acerto não recalculada — valor original com custo de 0,5,
ver nota da seção 2 (o valor real após a correção é igual ou menor).*

### Distribuição de P&L por negócio (líquida de custos)

| Fold | Conjunto | Ganho médio* | Perda média* | Perda/Ganho* | P5 | P50 (mediana) | P95 |
|---|---|---|---|---|---|---|---|
| 1 | Validação | 28,73 | -27,63 | 0,96x | -62,50 | 5,00 | 52,50 |
| 1 | Teste | 27,67 | -26,35 | 0,95x | -62,50 | -2,50 | 47,50 |
| 2 | Validação | 22,04 | -23,71 | 1,08x | -62,50 | 2,50 | 42,50 |
| 2 | Teste | 23,38 | -21,52 | 0,92x | -58,25 | 2,50 | 42,50 |
| 3 | Validação | 26,33 | -19,95 | 0,76x | -52,50 | -2,50 | 48,50 |
| 3 | Teste | 27,85 | -16,30 | 0,59x | -42,50 | -2,50 | 52,50 |

*\* Ganho médio, Perda média e Perda/Ganho não recalculados — valores
originais com custo de 0,5, ver nota da seção 2. P5/P50/P95 já estão
corrigidos (deslocamento de -2,0 pontos, exato).*

## 5. Observações

- **Com o custo corrigido (2,5 pontos), a imagem muda bastante.** Com o
  custo antigo (0,5), 5 dos 6 pares fold/conjunto eram positivos; após
  a correção, só 3 continuam positivos (Fold 1 validação: +1.135,0;
  Fold 3 validação: +237,5, praticamente zero; Fold 3 teste: +4.205,0)
  e 3 passam a negativos (Fold 1 teste: -1.945,0; Fold 2 validação:
  -762,5; Fold 2 teste: -400,0).
- **Fold 2 inverte de sinal nos dois conjuntos.** Antes parecia positivo
  tanto em validação (535,5) quanto em teste (1.276,0); depois da
  correção, os dois ficam negativos (-762,5 e -400,0) — o volume de
  negócios do fold 2 (649 + 838 = 1.487) era alto o bastante para que
  os 2,0 pontos extras por negociação superassem o lucro bruto.
- **Fold 3 teste continua o resultado mais forte** mesmo após a
  correção (+4.205,0; +3,52/negócio) — também era, com folga, o
  conjunto de melhor lucro médio bruto por negócio antes da correção
  (5,52), então tinha mais margem para absorver o custo extra.
- **O volume de negócios, não só o lucro bruto por negócio, determina o
  tamanho da correção** — folds/conjuntos com muitas negociações
  pequenas são desproporcionalmente penalizados pelo custo real. Esse
  padrão se confirma de forma ainda mais extrema na v2, onde o
  overfitting gerado por `target_n_passadas` alto levou a volumes de
  milhares de negócios/fold e perdas líquidas muito maiores (ver
  [relatorio_resultados_v2.md](relatorio_resultados_v2.md)).
- **Taxa de acerto, ganho médio e perda média não foram recalculados
  com exatidão** (ver nota da seção 2) — os valores mostrados ainda são
  os originais (custo de 0,5) e tendem a estar levemente otimistas: a
  taxa de acerto real após a correção é igual ou menor que a mostrada.
- **A curva de treino em "U" sem recuperação total** (seção 3, não
  corrigida — telemetria real do treino sob o custo antigo) sugere que
  o orçamento de timesteps atual pode estar interrompendo o treino
  antes da política convergir — um próximo passo natural seria testar
  com mais timesteps/tempo de treino por fold.
