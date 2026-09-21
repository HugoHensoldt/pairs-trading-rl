#!/bin/bash
# Roda o estágio de P/L (slurm/submit_lstm_pl.sbatch) até concluir, com até 3
# tentativas encadeadas (a conta ppg-lncc tem MaxSubmit=1: um job por vez, sem
# --array). É retomável: o que já está em lstm_pl_cache é pulado.
#
# Uso (o driver precisa continuar vivo -> nohup/setsid):
#   nohup setsid bash slurm/submit_lstm_pl.sh > lstm_pl.log 2>&1 < /dev/null &
set -euo pipefail

cd "$(dirname "$0")/.."

for attempt in 1 2 3; do
    while squeue --me --noheader --format="%j" 2>/dev/null | grep -q '^lstm-'; do
        echo "Aguardando fila liberar (ainda há job lstm-* ativo)..."
        sleep 5
    done
    echo "=== pl: tentativa $attempt ($(date)) ==="
    if sbatch --wait slurm/submit_lstm_pl.sbatch; then
        echo "=== pl concluído ($(date)) ==="
        exit 0
    fi
    echo "AVISO: job pl terminou com erro/timeout (tentativa $attempt); o cache lateral é retomável."
done
echo "ERRO: P/L não concluiu em 3 tentativas. Veja /scratch/ppg-lncc/$USER/lstm-logs/lstm-pl-*.err" >&2
exit 1
