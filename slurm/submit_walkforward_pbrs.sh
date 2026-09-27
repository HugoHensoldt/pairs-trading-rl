#!/bin/bash
# Walk-forward da v5 COM shaping pbrs (ENV_KIND=hedged, STATE_KIND=spread,
# SHAPING_KIND=pbrs, SHAPING_SIGNAL=features) nas MESMAS 3 janelas de
# slurm/submit_walkforward.sh (TRAIN_SIZES="300,360,420", N_VAL=N_TEST=30,
# MAX_ORDER=488, EXCLUDE_ORDERS=338,463) -- objetivo: saber se o pbrs (que no
# sintético capturou 204% da regra ótima causal, docs/relatorio_diagnostico_sintetico.md
# §8, rodada A) também ajuda no dado REAL, e se isso se sustenta nas 3 janelas.
#
# SHAPING_SIGNAL=features usa só as próprias spread_compra/spread_venda que a v5
# já observa -- nenhuma informação privilegiada (isso só existia na rodada B do
# sintético, que usava o spread verdadeiro). Por isso é aplicável ao dado real.
#
# Diferente de submit_walkforward.sh: aqui NENHUM fold é reaproveitado -- não existe
# execução pbrs prévia em dado real em nenhuma das 3 janelas (nem a 360). Treina os
# 3 folds x seeds 0,1 = 6 execuções. Reaproveita submit_all_folds.sh sem alterá-lo.
# SHAPING_K/K0/LAMBDA/C e GAMMA usam os mesmos valores da rodada sintética vencedora
# (defaults do próprio rl_trading_pipeline.py; fixados aqui só para constar no log
# e no metadata.json de cada execução).
#
# MEMÓRIA: mesma configuração de HedgedPairEnv da v5 (o shaping não muda o tamanho
# do estado nem dos buffers) -- usa a mesma estimativa/checagem de submit_walkforward.sh.
#
# Autônomo (login node; sobrevive a logout com setsid+nohup). Não repete execuções
# cujo state.json já está "done": true; uma execução que falha NÃO é repetida.
#
# Uso (na raiz do repo):
#   DRY_RUN=1 bash slurm/submit_walkforward_pbrs.sh
#   setsid nohup bash slurm/submit_walkforward_pbrs.sh > /scratch/ppg-lncc/$USER/walkforward_pbrs.log 2>&1 < /dev/null &
# Progresso: /scratch/ppg-lncc/$USER/walkforward_pbrs_progresso.md
set -uo pipefail

DRY_RUN="${DRY_RUN:-0}"
PROGRESS="${PROGRESS:-/scratch/ppg-lncc/$USER/walkforward_pbrs_progresso.md}"
CHECK_MEM_FOLD="${CHECK_MEM_FOLD:-3}"
MEM_LIMIT_GB="${MEM_LIMIT_GB:-330}"
cd "$(dirname "$0")/.."

# configuração comum: IDÊNTICA à v5, exceto o shaping e as janelas (TRAIN_SIZES tem
# os 3 tamanhos; cada submissão usa FOLDS=<um deles>)
export ENV_KIND=hedged STATE_KIND=spread
export SHAPING_KIND=pbrs SHAPING_SIGNAL=features
export SHAPING_K="${SHAPING_K:-2.0}" SHAPING_K0="${SHAPING_K0:-0.25}"
export SHAPING_LAMBDA="${SHAPING_LAMBDA:-0.05}" SHAPING_C="${SHAPING_C:-5.0}" GAMMA="${GAMMA:-0.999999}"
export TRAIN_SIZES="300,360,420" N_VAL="${N_VAL:-30}" N_TEST="${N_TEST:-30}"
export MAX_ORDER="${MAX_ORDER:-488}" EXCLUDE_ORDERS="${EXCLUDE_ORDERS:-338,463}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}" ENT_COEF="${ENT_COEF:-0.01}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-5000000}" CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-5000000}"
export VEC_ENV="${VEC_ENV:-dummy}" TORCH_THREADS="${TORCH_THREADS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4096}" N_STEPS="${N_STEPS:-2048}"
export N_TRAIN_ENVS="${N_TRAIN_ENVS:-48}"
export THROUGHPUT_STEPS_PER_SEC="${THROUGHPUT_STEPS_PER_SEC:-27000}" AVG_TICKS_PER_DAY="${AVG_TICKS_PER_DAY:-27662}"

module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export DAY_CACHE_DIR="/scratch/ppg-lncc/$USER/day_cache"

mkdir -p "$(dirname "$PROGRESS")"
log() { echo "$@" | tee -a "$PROGRESS"; }

log "##### WALK-FORWARD PBRS: 3 folds (300/360/420), seeds 0-1, ${TOTAL_TIMESTEPS} timesteps cada — $(date) #####"
log "SHAPING_KIND=pbrs SHAPING_SIGNAL=features SHAPING_K=$SHAPING_K SHAPING_LAMBDA=$SHAPING_LAMBDA SHAPING_C=$SHAPING_C GAMMA=$GAMMA"

echo "== plano dos 3 folds (confirmação; janelas devem bater com submit_walkforward.sh)"
(cd src && PRINT_FOLD_PLAN=1 python3 rl_trading_pipeline.py 2>&1) | tee -a "$PROGRESS"

for fold_seed in "1:0" "1:1" "2:0" "2:1" "3:0" "3:1"; do
    IFS=: read -r fold seed <<< "$fold_seed"
    size=$(echo "$TRAIN_SIZES" | cut -d, -f"$fold")
    tag="wfp_${size}_s${seed}"
    state="src/runs/${tag}/fold${fold}/state.json"

    if [ -f "$state" ] && grep -q '"done": true' "$state"; then
        log "[$(date)] $tag (fold $fold) já concluída; pulando o treino"
    else
        log "##### [$(date)] INÍCIO treino $tag (fold $fold = treino de $size dias, seed $seed, pbrs) #####"
        if [ "$DRY_RUN" = "1" ]; then
            echo "DRY_RUN: RUN_TAG=$tag SEED=$seed FOLDS=$fold bash slurm/submit_all_folds.sh"
        elif ! RUN_TAG="$tag" SEED="$seed" FOLDS="$fold" bash slurm/submit_all_folds.sh; then
            log "##### [$(date)] FALHOU $tag (não será repetida; segue para a próxima) #####"
            continue
        fi
        log "##### [$(date)] FIM treino+avaliação $tag #####"
    fi

    if [ "$fold" = "$CHECK_MEM_FOLD" ] && [ "$DRY_RUN" != "1" ]; then
        jobid=$(sacct -X --name="pairs-rl-fold${fold}-${tag}" --noheader --format=JobID,End \
                | sort -k2 | tail -1 | awk '{print $1}')
        if [ -n "$jobid" ]; then
            memkb=$(sacct -j "$jobid" --noheader --format=MaxRSS | tr -dc '0-9')
            if [ -n "$memkb" ]; then
                memgb=$(( memkb / 1024 / 1024 ))
                log "[$(date)] MaxRSS do job de treino $tag (job $jobid): ${memgb} GiB"
                if [ "$memgb" -gt "$MEM_LIMIT_GB" ]; then
                    log "AVISO: MaxRSS (${memgb} GiB) > MEM_LIMIT_GB (${MEM_LIMIT_GB}); considere N_TRAIN_ENVS menor nas próximas execuções deste fold."
                fi
            fi
        fi
    fi
    {
        echo; echo "### Resumo de $tag (fold $fold) — $(date)"
        s="src/runs/${tag}/fold${fold}/eval/summary.csv"
        [ -f "$s" ] && { echo "--- summary.csv (best_val, val e teste)"; grep -E "^label|^best_val_(val|test)_fee" "$s"; }
    } >> "$PROGRESS"
done

log "##### WALK-FORWARD PBRS concluído — $(date) #####"
log "Próximo passo: python src/analyze_walkforward.py --pbrs-backup <backup_desta_rodada> (compara pbrs x v5-sem-shaping nas 3 janelas)."
