# Sweep de `target_n_passadas` (fold 1) — RL (PPO) para Pairs Trading BOVA11 x WINM21

> **Status: aguardando execução no Santos Dumont.** Este documento é um
> esqueleto — a infraestrutura (código + scripts Slurm) já está pronta e
> validada localmente (ver seção 1), mas os números das seções 3 e 4 só
> existem depois que `slurm/submit_sweep.sh` rodar no cluster e
> `src/analyze_sweep.py` processar os logs trazidos de volta.

## 0. Motivação

A [v2](relatorio_resultados_v2.md) treinou com `target_n_passadas=20`
fixo nos 3 folds e encontrou overfitting severo: mais chunks/mais treino
(folds 2 e 3) correlacionou com prejuízo muito maior em validação/teste
do que o fold 1 (só 1 chunk, resultado quase neutro) — apesar da
recompensa média de treino (`ep_rew_mean`) convergir suavemente para
perto de zero em todos os folds, o que **mascarou** o overfitting em vez
de sinalizá-lo. A conclusão explícita da v2 foi testar
`target_n_passadas` mais baixo.

Este documento cobre um sweep controlado de `target_n_passadas` ∈
{1, 3, 5, 10, 20}, com **tudo mais fixo**: mesma função de folds
walk-forward (`generate_walk_forward_folds`), mesmo `MAX_PREGOES=80`,
mesmo custo de transação (2,5 pontos, já correto desde o treino como na
v2), mesmos hiperparâmetros PPO (`learning_rate=3e-4`, `n_steps=8192`,
`batch_size=1024`, `gamma=0.999`, `n_train_envs=48`). O objetivo central
é medir overfitting **explicitamente**: plotar `ep_rew_mean` de treino
contra `eval/mean_reward` de validação no mesmo eixo de timesteps — a
distância entre as duas curvas é a métrica de overfitting.

## 1. Metodologia e escopo

**Ajuste de escopo (decisão explícita, não um efeito colateral):** rodar
os 5 valores do sweep nos 3 folds da v2 ficaria em torno de 8h de Santos
Dumont (a conta `ppg-lncc` tem `MaxSubmit=1`, então as rodadas são
sequenciais por construção, não só por escolha). Para caber em até ~2h,
**o sweep roda só no fold 1** — 8 dias de treino / 12 validação / 12
teste, exatamente os mesmos limites do fold 1 da v2 (mesma
`generate_walk_forward_folds`, mesmo `MAX_PREGOES=80` — só não treinamos
os folds 2 e 3 desta vez).

Verificado localmente antes de subir pro cluster (`PRINT_FOLD_PLAN=1`,
`MAX_PREGOES=80`, `FOLDS=1`, ver comando em
`slurm/submit_all_folds.sh`/`src/analyze_sweep.py`):

| `target_n_passadas` | `total_timesteps` (fold 1) | Chunks (jobs) | ~ Atualizações PPO* |
|---|---|---|---|
| 1  | 221.296    | 1 | 0,6  |
| 3  | 663.888    | 1 | 1,7  |
| 5  | 1.106.480  | 1 | 2,8  |
| 10 | 2.212.960  | 1 | 5,6  |
| 20 | 4.425.920  | 1 | 11,3 |

*\* Atualizações PPO ≈ `total_timesteps / (n_steps × n_train_envs)` =
`total_timesteps / (8192 × 48)` = `total_timesteps / 393.216`.*

**Ressalva metodológica importante:** com fold 1 (8 dias de treino) e o
tamanho de rollout fixo (`n_steps=8192 × n_train_envs=48 = 393.216`
timesteps por atualização), `target_n_passadas=1` gera menos de 1
atualização PPO completa — a política nesse ponto está muito perto de
uma rede recém-inicializada, não de um "treino leve real". `passadas=3`
também é bem abaixo de 2 atualizações. Os pontos 1 e 3, portanto, devem
ser lidos como uma referência de baseline "quase não-treinado", não como
pontos plenamente comparáveis aos de 5/10/20 em termos de "quanto a
política aprendeu". Essa é uma limitação direta da redução de escopo pra
caber em ~2h — o sweep completo nos 3 folds (fold 2/3 têm bem mais dias
de treino, logo mais atualizações por `target_n_passadas`) não teria essa
ressalva, mas custaria ~8h.

Infraestrutura nova/alterada para viabilizar o sweep (código já
implementado e com sanity check local, ver commit):

- `RUN_TAG` (env var) em `src/rl_trading_pipeline.py`: namespacea
  checkpoint (`ppo_arbitrage_fold{N}_{TAG}.zip`), `best_model` e log CSV
  por valor do sweep, sem colidir artefatos. Vazio por padrão (reproduz
  os caminhos de v1/v2 sem mudança).
- Log estruturado (`progress.csv`, formato CSV nativo do SB3) com
  `rollout/ep_rew_mean` (treino) e `eval/mean_reward` (validação) na
  mesma linha/eixo de `time/total_timesteps` — não existia antes (só
  stdout); é o que alimenta o gráfico de overfitting da seção 4.
- `FOLDS` (env var) em `slurm/submit_all_folds.sh`: permite pedir só
  `FOLDS="1"` sem duplicar a lógica de planejamento de chunks/resume já
  validada.
- `slurm/submit_sweep.sh`: itera `TARGET_N_PASSADAS` ∈ {1,3,5,10,20},
  chamando `submit_all_folds.sh` com `RUN_TAG=p<N>` `FOLDS=1` por
  iteração.
- `src/analyze_sweep.py`: parseia os `.out` (tabela final de
  lucro/negócios/taxa de acerto) e os `progress.csv` (curvas), gera as
  tabelas/figuras deste documento.

## 2. Configurações

Idêntico à v2 em tudo que não é `target_n_passadas`/fold — ver
[relatorio_resultados_v2.md §1-2](relatorio_resultados_v2.md) e
[relatorio_resultados.md §2](relatorio_resultados.md#2-configurações)
para o detalhamento completo (espaço de estado, ações, recompensa, custo
de transação, hiperparâmetros PPO).

## 3. Resultados de avaliação

*(preencher com `docs/sweep_resultados.md`, gerado por
`src/analyze_sweep.py` depois da rodada no cluster)*

## 4. Curvas de treino vs. validação — gap de overfitting

*(preencher com `docs/img/sweep_p<N>.png` por valor de
`target_n_passadas`, e o gráfico-resumo
`docs/img/sweep_overfitting_gap.png` — gap = `ep_rew_mean` de treino
menos `eval/mean_reward` de validação, média dos últimos 5 pontos de
avaliação, no eixo `target_n_passadas`)*

## 5. Observações

*(preencher depois dos resultados — em particular: o gap de overfitting
cresce com `target_n_passadas`, como a hipótese da v2 sugere? Os pontos
1/3 (ressalva da seção 1) se comportam como esperado de uma política
quase não-treinada, ou surpreendem? Vale a pena rodar a versão completa
nos 3 folds depois, com mais orçamento de tempo?)*
