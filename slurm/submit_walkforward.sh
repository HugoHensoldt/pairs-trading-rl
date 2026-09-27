#!/bin/bash
# Avaliação walk-forward da v5 (ENV_KIND=hedged, STATE_KIND=spread) em 3 janelas de
# treino, para responder à crítica de "uma janela de teste só" (docs/relatorio_resultados_v5.md).
# Reaproveita generate_fixed_window_folds/submit_all_folds.sh -- NENHUM split novo.
#
# TRAIN_SIZES="300,360,420" (constante, os 3 juntos) numera os folds 1/2/3 = treino
# 300/360/420. O fold 2 (360) é IDÊNTICO ao da v5 (mesmas janelas, ent_coef, gamma,
# total_timesteps -- conferido contra sdumont_backup_campanha/.../hedged_s{0..3}/
# metadata.json) e NÃO é retreinado aqui: reaproveite hedged_s0..s3. Este driver só
# treina os folds NOVOS -- 300 e 420 -- com as seeds 0 e 1 (FOLDS=1 e FOLDS=3).
#
# Para cada (fold, seed): treina (submit_all_folds.sh, chunks de ~20 min) e roda os
# MESMOS 2 jobs de avaliação da v5 (modelos + baselines flat/segurar-o-par +
# sensibilidade a custo/meio-spread + latência de 1 tick + curva de checkpoints --
# já embutidos em submit_all_folds.sh/run_eval_job, nada novo aqui).
#
# MEMÓRIA (ver docs/relatorio_walkforward.md): a v5 mediu ~185 GiB de pico no job de
# treino do fold 360 (360 dias x 48 envs). Fold 420 (420/360 = 1,17x mais dias):
# estimativa LINEAR ~216 GiB -- folga confortável sob os 384 GB do nó e abaixo do teto
# de ~330 GiB. Este driver confere o MaxRSS (via sacct) da execução wf_420_s0 ANTES de
# submeter wf_420_s1 -- não trava um chunk em andamento (o job roda até o fim antes do
# driver conseguir olhar), mas dá uma chance real de ajustar N_TRAIN_ENVS (ex.: 32,
# reduz memória quase linearmente sem mudar o resultado esperado, só o throughput)
# (só o throughput).
#
# Autônomo (login node; sobrevive a logout com setsid+nohup). Não repete execuções
# cujo state.json já está "done": true; uma execução que falha NÃO é repetida.
#
# Uso (na raiz do repo):
#   DRY_RUN=1 bash slurm/submit_walkforward.sh
#   setsid nohup bash slurm/submit_walkforward.sh > /scratch/ppg-lncc/$USER/walkforward.log 2>&1 < /dev/null &
# Progresso: /scratch/ppg-lncc/$USER/walkforward_progresso.md
set -uo pipefail

DRY_RUN="${DRY_RUN:-0}"
PROGRESS="${PROGRESS:-/scratch/ppg-lncc/$USER/walkforward_progresso.md}"
CHECK_MEM_FOLD="${CHECK_MEM_FOLD:-3}"       # fold cujo 1º chunk tem o MaxRSS conferido antes de prosseguir
MEM_LIMIT_GB="${MEM_LIMIT_GB:-330}"
cd "$(dirname "$0")/.."

# configuração comum: IDÊNTICA à v5 (docs/relatorio_resultados_v5.md, §1), exceto as
# janelas (TRAIN_SIZES agora tem os 3 tamanhos; cada submissão usa FOLDS=<um deles>)
export ENV_KIND=hedged STATE_KIND=spread
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
wait_queue() { while squeue --me --noheader 2>/dev/null | grep -q .; do sleep 10; done; }

log "##### WALK-FORWARD v5: folds 300 e 420, seeds 0-1, ${TOTAL_TIMESTEPS} timesteps cada — $(date) #####"
log "Fold 360 (=v5) NÃO é retreinado aqui: reaproveita sdumont_backup_campanha/.../hedged_s{0..3}."

echo "== plano dos 3 folds (confirmação; fold 2 = 360 deve bater com a v5)"
(cd src && PRINT_FOLD_PLAN=1 python3 rl_trading_pipeline.py 2>&1) | tee -a "$PROGRESS"

for fold_seed in "1:0" "1:1" "3:0" "3:1"; do
    IFS=: read -r fold seed <<< "$fold_seed"
    size=$(echo "$TRAIN_SIZES" | cut -d, -f"$fold")
    tag="wf_${size}_s${seed}"
    state="src/runs/${tag}/fold${fold}/state.json"

    if [ -f "$state" ] && grep -q '"done": true' "$state"; then
        log "[$(date)] $tag (fold $fold) já concluída; pulando o treino"
    else
        log "##### [$(date)] INÍCIO treino $tag (fold $fold = treino de $size dias, seed $seed) #####"
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

log "##### WALK-FORWARD concluído — $(date) #####"
log "Próximo passo: python src/analyze_walkforward.py (gera docs/relatorio_walkforward.md e a figura)."
