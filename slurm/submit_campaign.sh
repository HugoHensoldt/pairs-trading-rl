#!/bin/bash
# Campanha: variantes de recompensa x seeds, com ORÇAMENTO DE TEMPO.
#
# Para cada seed (loop externo) e cada variante (loop interno) roda
# slurm/submit_all_folds.sh com RUN_TAG=<variante>_s<seed> e ENV_KIND=<variante>.
# A ordem é seed-major de propósito: se o tempo acabar, todas as variantes já
# têm as mesmas seeds completas (comparação justa).
#
# Antes de iniciar cada execução, estima a duração pela MÉDIA das anteriores e
# só começa se ainda couber em MAX_HOURS; caso contrário para de forma limpa
# (nunca deixa uma execução pela metade por falta de tempo).
#
# Uso (as env vars do pipeline são repassadas: TOTAL_TIMESTEPS, TRAIN_SIZES,
# ENT_COEF, VEC_ENV, BATCH_SIZE, EVAL_EVERY_STEPS...):
#   nohup bash slurm/submit_campaign.sh > /scratch/ppg-lncc/$USER/campanha.log 2>&1 &
#   VARIANTS="hedged spread" SEEDS="0 1 2" MAX_HOURS=17 bash slurm/submit_campaign.sh
set -uo pipefail

VARIANTS="${VARIANTS:-hedged spread}"     # hedged | spread | win_only
SEEDS="${SEEDS:-0 1 2}"
MAX_HOURS="${MAX_HOURS:-17}"
cd "$(dirname "$0")/.."

start=$(date +%s)
n_done=0
total_dur=0
falhas=""
echo "##### CAMPANHA: variantes=[$VARIANTS] seeds=[$SEEDS] orçamento=${MAX_HOURS}h — $(date) #####"

for seed in $SEEDS; do
    for variant in $VARIANTS; do
        tag="${variant}_s${seed}"
        elapsed=$(( $(date +%s) - start ))
        avg=0
        if [ "$n_done" -gt 0 ]; then avg=$(( total_dur / n_done )); fi
        if [ $(( elapsed + avg )) -gt $(( MAX_HOURS * 3600 )) ]; then
            echo "##### ORÇAMENTO: decorridos $((elapsed/60)) min + média por execução $((avg/60)) min > ${MAX_HOURS}h; parando antes de $tag #####"
            break 2
        fi
        echo "##### [$(date)] INÍCIO $tag (decorridos $((elapsed/60)) min; execuções concluídas: $n_done) #####"
        t0=$(date +%s)
        if ENV_KIND="$variant" RUN_TAG="$tag" SEED="$seed" bash slurm/submit_all_folds.sh; then
            dur=$(( $(date +%s) - t0 ))
            total_dur=$(( total_dur + dur ))
            n_done=$(( n_done + 1 ))
            echo "##### [$(date)] FIM $tag em $((dur/60)) min #####"
        else
            echo "##### [$(date)] FALHOU $tag #####" >&2
            falhas="$falhas $tag"
        fi
    done
done

echo "##### CAMPANHA encerrada — $(date): $n_done execuções concluídas, decorridos $(( ($(date +%s) - start) / 60 )) min #####"
if [ -n "$falhas" ]; then
    echo "Execuções com falha:$falhas" >&2
    exit 1
fi
