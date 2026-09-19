#!/bin/bash
# Campanha ADAPTATIVA da v4, autônoma (roda no login node, sem depender de um
# agente/PC ligado). Executa uma execução por vez com slurm/submit_all_folds.sh e
# escolhe a próxima com src/campaign_decide.py (regras objetivas, ver o docstring
# dele): hedged seed 0 (100M) -> pilotos de spread/spreadx (30M) -> hedged seeds
# 1 e 2 -> extensões condicionais. Termina sozinha quando não há mais nada a
# rodar segundo as regras, ou quando a próxima execução não cabe na
# disponibilidade (MAX_HOURS, soft: é limite de disponibilidade, não meta).
#
# Uso (na raiz do repo, sobrevive a logout com setsid+nohup):
#   setsid nohup bash slurm/submit_adaptive.sh > /scratch/ppg-lncc/$USER/adaptativa.log 2>&1 < /dev/null &
#   DRY_RUN=1 bash slurm/submit_adaptive.sh     # só imprime a sequência, não submete nada
#
# Progresso: $PROGRESS (default /scratch/ppg-lncc/$USER/campanha_progresso.md),
# atualizado ao fim de cada execução. Execuções tentadas: $ATTEMPTED.
set -uo pipefail

MAX_HOURS="${MAX_HOURS:-18}"
PROGRESS="${PROGRESS:-/scratch/ppg-lncc/$USER/campanha_progresso.md}"
ATTEMPTED="${ATTEMPTED:-/scratch/ppg-lncc/$USER/campanha_tentadas.txt}"
DRY_RUN="${DRY_RUN:-0}"

cd "$(dirname "$0")/.."

# configuração comum (o que já estiver exportado no ambiente tem precedência)
export TRAIN_SIZES="${TRAIN_SIZES:-360}" N_VAL="${N_VAL:-30}" N_TEST="${N_TEST:-30}"
export ENT_COEF="${ENT_COEF:-0.01}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-5000000}" CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-5000000}"
export VEC_ENV="${VEC_ENV:-dummy}" TORCH_THREADS="${TORCH_THREADS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4096}" N_STEPS="${N_STEPS:-2048}"
export THROUGHPUT_STEPS_PER_SEC="${THROUGHPUT_STEPS_PER_SEC:-27000}"

module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export DAY_CACHE_DIR="/scratch/ppg-lncc/$USER/day_cache"

mkdir -p "$(dirname "$PROGRESS")"
start=$(date +%s)
n_main=0; sum_main=0; n_pilot=0; sum_pilot=0
echo "##### ADAPTATIVA: início $(date), disponibilidade ${MAX_HOURS}h (soft) #####" | tee -a "$PROGRESS"

while true; do
    line=$(python3 src/campaign_decide.py next --runs-dir src/runs --attempted "$ATTEMPTED")
    if [ "$line" = "NONE" ]; then
        echo "##### Sem mais execuções segundo as regras — $(date) #####" | tee -a "$PROGRESS"
        break
    fi
    read -r variant seed steps <<< "$line"
    tag="${variant}_s${seed}"

    # estimativa de duração: média das anteriores da mesma classe, senão 2,5 h (100M) / 1,25 h (30M)
    if [ "$steps" -ge 100000000 ]; then
        est=9000; if [ "$n_main" -gt 0 ]; then est=$(( sum_main / n_main )); fi
    else
        est=4500; if [ "$n_pilot" -gt 0 ]; then est=$(( sum_pilot / n_pilot )); fi
    fi
    elapsed=$(( $(date +%s) - start ))
    if [ $(( elapsed + est )) -gt $(( MAX_HOURS * 3600 )) ]; then
        echo "##### DISPONIBILIDADE: decorridos $((elapsed/60)) min + estimativa $((est/60)) min de $tag > ${MAX_HOURS}h; encerrando sem iniciá-la #####" | tee -a "$PROGRESS"
        break
    fi

    echo "$tag" >> "$ATTEMPTED"
    echo "##### [$(date)] INÍCIO $tag ($steps timesteps; decorridos $((elapsed/60)) min) #####" | tee -a "$PROGRESS"
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN: ENV_KIND=$variant RUN_TAG=$tag SEED=$seed TOTAL_TIMESTEPS=$steps bash slurm/submit_all_folds.sh"
        continue
    fi
    t0=$(date +%s)
    if ENV_KIND="$variant" RUN_TAG="$tag" SEED="$seed" TOTAL_TIMESTEPS="$steps" bash slurm/submit_all_folds.sh; then
        dur=$(( $(date +%s) - t0 ))
        if [ "$steps" -ge 100000000 ]; then n_main=$((n_main+1)); sum_main=$((sum_main+dur));
        else n_pilot=$((n_pilot+1)); sum_pilot=$((sum_pilot+dur)); fi
        echo "##### [$(date)] FIM $tag em $((dur/60)) min #####" | tee -a "$PROGRESS"
    else
        echo "##### [$(date)] FALHOU $tag (não será repetida; a campanha segue) #####" | tee -a "$PROGRESS" >&2
    fi
    {
        echo; echo "### Situação após $tag ($(date))"
        python3 src/campaign_decide.py report --runs-dir src/runs
    } >> "$PROGRESS"
done

{
    echo; echo "### Resumo final ($(date), decorridos $(( ($(date +%s) - start) / 60 )) min)"
    python3 src/campaign_decide.py report --runs-dir src/runs
} | tee -a "$PROGRESS"
