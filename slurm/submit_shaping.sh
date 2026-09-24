#!/bin/bash
# Rodada de VARIAÇÕES DE RECOMPENSA (shaping) nos dados sintéticos cointegrados, versão
# hedgeada (HedgedPairEnv). Duas rodadas, 5 formatos de shaping cada (SHAPING_KIND):
#   A) estado da versão hedgeada (STATE_KIND=spread: spread_compra, spread_venda, posição, P/L);
#      sinal do shaping = as próprias features (SHAPING_SIGNAL=features)
#   B) estado SÓ de preços + tempo cíclico, SEM spread do par (STATE_KIND=raw, RAW_LEVELS=0);
#      sinal do shaping = spread verdadeiro do sintético (SHAPING_SIGNAL=truth, informação
#      privilegiada usada só como PROFESSOR na recompensa, não entra na observação)
# Formatos: none (controle) | pbrs | opp_wrong | opp_flat | regret  (ver
# HedgedPairEnv._shaping). O shaping só altera o que o PPO vê; o P/L real reportado é intacto.
#
# Referência de comparação: regra ótima causal do sintético (+62,6 R$/dia no teste) e o
# baseline syn_coint_s0 (estado raw com níveis, sem shaping: 0% capturado).
#
# Autônomo (login node; sobrevive a logout com setsid+nohup). Para cada execução:
#   1) treina (submit_all_folds.sh, chunks de ~20 min);
#   2) diagnostica e compara com a regra ótima (submit_diag.sbatch, DIAG_KIND=coint).
# Não repete execuções cujo state.json já está "done": true; uma execução que falha NÃO é
# repetida e o driver segue para a próxima. Ordem = prioridade (as mais informativas primeiro).
#
# Uso (na raiz do repo):
#   DRY_RUN=1 bash slurm/submit_shaping.sh
#   setsid nohup bash slurm/submit_shaping.sh > /scratch/ppg-lncc/$USER/shaping.log 2>&1 < /dev/null &
# Progresso: /scratch/ppg-lncc/$USER/shaping_progresso.md
set -uo pipefail

SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
PROGRESS="${PROGRESS:-/scratch/ppg-lncc/$USER/shaping_progresso.md}"
cd "$(dirname "$0")/.."

# configuração comum (mesma do baseline syn_coint_s0, exceto os 30M timesteps)
export ENV_KIND=hedged SYNTHETIC=coint SYNTH_KIND=coint
export TRAIN_SIZES="${TRAIN_SIZES:-300}" N_VAL="${N_VAL:-30}" N_TEST="${N_TEST:-30}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000000}" ENT_COEF="${ENT_COEF:-0.01}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-5000000}" CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-5000000}"
export VEC_ENV="${VEC_ENV:-dummy}" TORCH_THREADS="${TORCH_THREADS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4096}" N_STEPS="${N_STEPS:-2048}"
export THROUGHPUT_STEPS_PER_SEC="${THROUGHPUT_STEPS_PER_SEC:-27000}" AVG_TICKS_PER_DAY=27000
export SYNTH_DAYS="${SYNTH_DAYS:-450}"
# parâmetros do shaping (gravados no metadata.json de cada execução)
export SHAPING_K="${SHAPING_K:-2.0}" SHAPING_K0="${SHAPING_K0:-0.25}"
export SHAPING_LAMBDA="${SHAPING_LAMBDA:-0.05}" SHAPING_C="${SHAPING_C:-5.0}" GAMMA="${GAMMA:-0.999999}"

module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-rl/bin/activate"
export TICK_DATA_DIR="/scratch/ppg-lncc/$USER/tick_data"
export DAY_CACHE_DIR="/scratch/ppg-lncc/$USER/day_cache"

mkdir -p "$(dirname "$PROGRESS")"
log() { echo "$@" | tee -a "$PROGRESS"; }
wait_queue() { while squeue --me --noheader 2>/dev/null | grep -q .; do sleep 10; done; }
run() { if [ "$DRY_RUN" = "1" ]; then echo "DRY_RUN: $*"; else "$@"; fi; }

# nome:estado:raw_levels:formato:sinal
CONFIGS=(
    "A_none:spread:1:none:features"
    "A_pbrs:spread:1:pbrs:features"
    "A_opp_wrong:spread:1:opp_wrong:features"
    "B_none:raw:0:none:truth"
    "B_pbrs:raw:0:pbrs:truth"
    "B_opp_wrong:raw:0:opp_wrong:truth"
    "A_opp_flat:spread:1:opp_flat:features"
    "A_regret:spread:1:regret:features"
    "B_opp_flat:raw:0:opp_flat:truth"
    "B_regret:raw:0:regret:truth"
)

log "##### SHAPING sintético (hedged): ${#CONFIGS[@]} execuções, ${TOTAL_TIMESTEPS} timesteps cada — $(date) #####"

for cfg in "${CONFIGS[@]}"; do
    IFS=: read -r name skind levels shaping signal <<< "$cfg"
    tag="syn_shape_${name}_s${SEED}"
    state="src/runs/${tag}/fold1/state.json"
    export STATE_KIND="$skind" RAW_LEVELS="$levels" SHAPING_KIND="$shaping" SHAPING_SIGNAL="$signal"

    if [ -f "$state" ] && grep -q '"done": true' "$state"; then
        log "[$(date)] $tag já concluída; pulando o treino"
    else
        log "##### [$(date)] INÍCIO treino $tag (estado=$skind niveis=$levels shaping=$shaping sinal=$signal) #####"
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
        v="src/runs/$tag/fold1/val_curve.csv"
        [ -f "$v" ] && { echo "--- val_curve.csv (timesteps, val_pnl_total, val_trades_per_day, val_pct_flat)"; cut -d, -f1,5,7,9 "$v"; }
    } >> "$PROGRESS"
done

log "##### SHAPING sintético concluído — $(date) #####"
log "Compare com src/runs/syn_coint_s0/fold1/diag/benchmark_stdout.txt (baseline: 0% da regra ótima)."
