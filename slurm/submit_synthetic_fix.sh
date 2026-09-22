#!/bin/bash
# Valida as correções candidatas ao colapso de entropia (ver
# docs/relatorio_diagnostico_sintetico.md e o commit a8f38a4) no MESMO cenário
# sintético cointegrado (syn_coint) usado no diagnóstico, para comparação direta
# com o baseline `syn_coint_s0` (regra ótima: +62,6 R$/dia no teste; baseline
# capturou 0%). Autônomo (roda no login node, sobrevive a logout com
# setsid+nohup). Para cada variante:
#   1) treina (submit_all_folds.sh, chunks de ~20 min);
#   2) diagnostica + compara com a regra ótima (submit_diag.sbatch, que já
#      chama synthetic_benchmark.py --run-dir ... para DIAG_KIND=coint).
# Não repete execuções cujo state.json já está "done": true. Uma execução que
# falha NÃO é repetida; o driver segue para a próxima.
#
# Variantes (mesmos TRAIN_SIZES/N_VAL/N_TEST/TOTAL_TIMESTEPS do baseline, para
# os dias sintéticos ficarem idênticos -- synthetic_day() é determinístico por
# índice):
#   entcoef : ENT_COEF=0.05 (só isso)
#   costramp: HEDGE_REWARD_CLIP=50, COST_RAMP_STEPS=5.000.000 (0 -> custo cheio)
#   combo   : os dois juntos
#
# Uso (na raiz do repo):
#   DRY_RUN=1 bash slurm/submit_synthetic_fix.sh      # só imprime o que faria
#   setsid nohup bash slurm/submit_synthetic_fix.sh > /scratch/ppg-lncc/$USER/sintetico_fix.log 2>&1 < /dev/null &
# Progresso: /scratch/ppg-lncc/$USER/sintetico_fix_progresso.md
set -uo pipefail

SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
PROGRESS="${PROGRESS:-/scratch/ppg-lncc/$USER/sintetico_fix_progresso.md}"
cd "$(dirname "$0")/.."

# configuração comum -- IGUAL ao baseline (syn_coint_s0), exceto as variáveis
# que cada config abaixo sobrescreve
export STATE_KIND=raw ENV_KIND=hedged SYNTHETIC=coint SYNTH_KIND=coint
export TRAIN_SIZES="${TRAIN_SIZES:-300}" N_VAL="${N_VAL:-30}" N_TEST="${N_TEST:-30}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000000}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-5000000}" CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-5000000}"
export VEC_ENV="${VEC_ENV:-dummy}" TORCH_THREADS="${TORCH_THREADS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4096}" N_STEPS="${N_STEPS:-2048}"
export THROUGHPUT_STEPS_PER_SEC="${THROUGHPUT_STEPS_PER_SEC:-27000}" AVG_TICKS_PER_DAY=27000
export SYNTH_DAYS="${SYNTH_DAYS:-450}"

module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export DAY_CACHE_DIR="/scratch/ppg-lncc/$USER/day_cache"

mkdir -p "$(dirname "$PROGRESS")"
log() { echo "$@" | tee -a "$PROGRESS"; }
wait_queue() { while squeue --me --noheader 2>/dev/null | grep -q .; do sleep 10; done; }
run() { if [ "$DRY_RUN" = "1" ]; then echo "DRY_RUN: $*"; else "$@"; fi; }

log "##### SINTÉTICO (correções) — $(date) #####"

# nome:ent_coef:reward_clip:ramp_steps:ramp_start
CONFIGS=(
    "entcoef:0.05:0:0:0.0"
    "costramp:0.01:50:5000000:0.0"
    "combo:0.05:50:5000000:0.0"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=: read -r name ent clip ramp_steps ramp_start <<< "$cfg"
    tag="syn_coint_fix_${name}_s${SEED}"
    state="src/runs/${tag}/fold1/state.json"
    export ENT_COEF="$ent" HEDGE_REWARD_CLIP="$clip" COST_RAMP_STEPS="$ramp_steps" COST_RAMP_START="$ramp_start"

    if [ -f "$state" ] && grep -q '"done": true' "$state"; then
        log "[$(date)] $tag já concluída; pulando o treino"
    else
        log "##### [$(date)] INÍCIO treino $tag (ent_coef=$ent reward_clip=$clip ramp=$ramp_steps/$ramp_start) #####"
        if ! run env RUN_TAG="$tag" SEED="$SEED" bash slurm/submit_all_folds.sh; then
            log "##### [$(date)] FALHOU $tag (não será repetida; segue para a próxima) #####"
            continue
        fi
        log "##### [$(date)] FIM treino $tag #####"
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN: sbatch --wait diag de $tag"; continue
    fi
    wait_queue
    log "##### [$(date)] INÍCIO diagnóstico $tag #####"
    sbatch --wait --export=ALL,DIAG_RUN_DIR="runs/${tag}/fold1",DIAG_KIND=coint slurm/submit_diag.sbatch \
        || log "AVISO: o job de diagnóstico de $tag terminou com erro/timeout"
    log "##### [$(date)] FIM diagnóstico $tag #####"
    {
        echo; echo "### Resumo de $tag ($(date))"
        f="src/runs/$tag/fold1/diag/benchmark_stdout.txt"
        [ -f "$f" ] && { echo "--- $f"; cat "$f"; }
    } >> "$PROGRESS"
done

log "##### SINTÉTICO (correções) concluído — $(date) #####"
log "Compare com o baseline em src/runs/syn_coint_s0/fold1/diag/benchmark_stdout.txt (0% capturado)."
