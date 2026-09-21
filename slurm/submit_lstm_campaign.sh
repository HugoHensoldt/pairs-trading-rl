#!/bin/bash
# Campanha de experimentos da LSTM para deixar rodando (ex.: a noite toda).
# Cada experimento é uma execução COMPLETA de slurm/submit_lstm.sh com o seu
# próprio LSTM_RUN_TAG (resultados em /scratch/.../lstm_runs/<tag>/) e
# variáveis de ambiente próprias. O cache de amostras é COMPARTILHADO: combinações
# (sigma, Re, Ri) já geradas por um experimento são reaproveitadas pelos outros.
#
# Os experimentos mudam UMA coisa por vez em relação ao `base` (ablação), para
# dar para atribuir qualquer diferença de resultado a uma causa só:
#   base       grade de 27 combinações + rede da tese (2x LSTM(50)).
#   grid_wide  grade mais larga e mais densa (120 combinações), mesma rede.
#   net_big    grade base, rede 2x LSTM(128) + dropout 0.2.
#   seed2      repete o base com outra semente: mede o ruído entre execuções.
# A ORDEM é a de prioridade: se o tempo/fila acabar, o que ficou para trás é o
# menos importante. Cada experimento é independente: se um falhar, a campanha
# segue para o próximo.
#
# Uso (o driver precisa continuar vivo -> nohup/tmux):
#   nohup bash slurm/submit_lstm_campaign.sh > lstm_campaign.log 2>&1 &
#   EXPERIMENTS_ONLY="base net_big" bash slurm/submit_lstm_campaign.sh   # subconjunto
#
# Retomar após queda: rode o mesmo comando; o que já terminou é pulado
# (tarefas concluídas ficam com finished=true) e o resto continua dos checkpoints.
set -uo pipefail   # sem -e: falha de um experimento não pode abortar a campanha

cd "$(dirname "$0")/.."

# Prefixo dos experimentos (tags = <prefixo>_<experimento>). Use um prefixo NOVO a cada campanha com
# grade/teste diferentes (ex.: CAMPAIGN_PREFIX=lstm_v2): tarefas já concluídas de uma tag são puladas.
PREFIX="${CAMPAIGN_PREFIX:-lstm}"

# "tag|VAR=valor VAR=valor ..."  (vazio = configuração padrão do código)
EXPERIMENTS=(
    "base|"
    "grid_wide|LSTM_SIGMA_GRID=1.0,1.2,1.4,1.6,1.8,2.0 LSTM_RE_GRID=0.5,0.6,0.75,0.9 LSTM_RI_GRID=0.5,0.75,1.0,1.25,1.5"
    "net_big|LSTM_UNITS=128 LSTM_DROPOUT=0.2"
    "seed2|LSTM_SEED=2"
)

for exp in "${EXPERIMENTS[@]}"; do
    tag="${exp%%|*}"
    envs="${exp#*|}"
    if [ -n "${EXPERIMENTS_ONLY:-}" ] && [[ " $EXPERIMENTS_ONLY " != *" $tag "* ]]; then
        continue
    fi
    echo ""
    echo "################ experimento: $tag  [$envs]  ($(date)) ################"
    # shellcheck disable=SC2086
    if ! env $envs LSTM_RUN_TAG="${PREFIX}_$tag" bash slurm/submit_lstm.sh; then
        echo "AVISO: experimento $tag terminou com erro; seguindo para o próximo."
    fi
done

echo ""
echo "################ comparação entre experimentos ($(date)) ################"
module load python/3.10.16_sequana
source "/scratch/ppg-lncc/$USER/envs/pairs-lstm/bin/activate"
export LSTM_RUN_DIR="${LSTM_RUN_DIR:-/scratch/ppg-lncc/$USER/lstm_runs}"
export LSTM_RUN_TAG="${PREFIX}_base"
export LSTM_COMPARE_PREFIX="$PREFIX"
(cd src && CUDA_VISIBLE_DEVICES="" python lstm_pipeline.py --stage compare)
