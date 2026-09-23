"""
Pipeline: do backtest de arbitragem estatística (BOVA11 x WINM21) até
o treinamento de duas LSTMs (compra e venda) que classificam se uma
oportunidade de arbitragem tende a ser Lucro ou Prejuízo.

Estrutura:
  1) process_day(order)      -> replica o backtest e devolve o df completo
                                 com sinais e resultado por negociação
  2) extract_trade_windows() -> para cada negociação, recorta a janela de
                                 N ticks anteriores como features (X) e o
                                 resultado (lucro>0) como label (y)
  3) build_datasets()        -> percorre vários dias, separa em
                                 treino/validação/teste por DIA (nunca por
                                 tick), separado para compra e venda
  4) scale_datasets()        -> normaliza com StandardScaler ajustado
                                 apenas no treino
  5) build_lstm()            -> monta a arquitetura (2x LSTM(50) + Dense)
  6) train_and_evaluate()    -> treina e reporta acurácia/precisão

Modificações desta versão (marcadas no código como [MUDANÇA 1..7]):
  [MUDANÇA 1] TimeSeriesSplit (expanding window, por PREGÃO) no lugar do
              split fixo treino/validação. O teste final (últimos 15% dos
              pregões) continua o mesmo de antes.
  [MUDANÇA 2] 4 features cíclicas (dia da semana e hora do dia, em
              seno/cosseno) somadas às 7 features da Tabela 1.
  [MUDANÇA 3] Limiares reduzidos (sigma, Re e Ri) para gerar mais
              negociações, inclusive perdedoras.
  [MUDANÇA 4] Varredura (sweep) sobre combinações de (sigma, Re, Ri); todas
              as amostras geradas são usadas juntas, e há um relatório de
              balanceamento de labels por combinação.
  [MUDANÇA 5] (i) posições abertas no fim do pregão são FECHADAS
              compulsoriamente (antes eram descartadas); (ii) o loop da
              heurística usa arrays numpy em vez de iterrows (mesma lógica).
  [MUDANÇA 6] +3 features: spread, volatilidade e Take Profit (SG), além do
              Stop Loss (SL) que já existia.
  [MUDANÇA 9] Rede (LSTM_UNITS, LSTM_DROPOUT) e semente (LSTM_SEED) configuráveis
              por ambiente para ablações; padrão = tese (2x LSTM(50)). Estágio
              `--stage compare` compara experimentos.
  [MUDANÇA 7] Execução no Santos Dumont: estágios (--stage), tarefas
              (fold, lado) com checkpoint/retomada e geração em paralelo.

Parâmetros da heurística: `sigma` = largura da banda de Bollinger;
`Re` = realização (fração do caminho até o alvo que queremos lucrar);
`Ri` = risco (fração do caminho até o alvo que aceitamos perder).

Ajuste os parâmetros no bloco `if __name__ == "__main__":` ao final.
Uso:
  python src/lstm_pipeline.py                        # fluxo serial original (tudo num processo)
  python src/lstm_pipeline.py --stage report         # só gera amostras + balanceamento (escolher a grade)
  python src/lstm_pipeline.py --stage generate --workers 48
  python src/lstm_pipeline.py --stage train --fold 1 --side buy   # treina/retoma 1 tarefa
  python src/lstm_pipeline.py --stage pending        # tarefas ainda não concluídas
  python src/lstm_pipeline.py --stage aggregate      # tabela final de métricas
No Santos Dumont use slurm/submit_lstm.sh (encadeia os jobs).
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

# Marca o início do processo: os jobs do Slurm têm limite de 20 min e o
# orçamento de tempo do treino é contado a partir daqui (ver
# LSTM_TRAIN_MAX_SECONDS), incluindo carga/normalização dos dados.
SCRIPT_START = time.time()

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)

# TensorFlow é importado de forma tolerante: os estágios de geração de
# amostras e de relatório (--stage generate/report/pending/aggregate) só usam
# pandas/numpy e devem poder rodar num nó sem TensorFlow (ex.: sequana_cpu_dev).
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:  # pragma: no cover
    keras = layers = None

from config import data_path

# --------------------------------------------------------------------------
# 1) REPLICA O BACKTEST PARA UM DIA E RETORNA O DF PROCESSADO
# --------------------------------------------------------------------------

# [MUDANÇA 4] O antigo process_day foi dividido em duas etapas para o sweep
# não repetir a parte cara (leitura dos JSON + médias móveis) a cada
# combinação de parâmetros:
#   load_day_base()   -> depende só do pregão (Periodo/Amostra)
#   apply_heuristic() -> depende de (sigma, Re, Ri)
# process_day() continua existindo com a mesma assinatura e o mesmo
# resultado de antes.

def add_cyclical_features(df):
    """[MUDANÇA 2] Codifica dia da semana e hora do dia como seno/cosseno,
    calculados a partir do timestamp real ('datahora') de CADA tick.

    Sem a codificação seno/cosseno, segunda=0 e sexta=4 ficariam "longe" e
    23:59 -> 00:00 daria um salto; com o par (sen, cos) o dado vira um
    ponto num círculo e a distância entre valores vizinhos é contínua.
    """
    ts = pd.to_datetime(df['datahora'])
    weekday = ts.dt.dayofweek.to_numpy()  # segunda=0 ... domingo=6
    seconds = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy()

    df['day_sin'] = np.sin(2 * np.pi * weekday / 7)
    df['day_cos'] = np.cos(2 * np.pi * weekday / 7)
    df['time_sin'] = np.sin(2 * np.pi * seconds / 86400)
    df['time_cos'] = np.cos(2 * np.pi * seconds / 86400)
    return df


def load_day_base(order, Periodo=3000, Amostra=2400):
    """Parte do backtest que não depende de sigma/Re/Ri: leitura, merge,
    médias móveis e recorte do pregão."""
    with open(data_path(str(order) + 'BOVA11.json'), 'r', encoding="UTF-16 LE") as myfile:
        dfb = pd.read_json(myfile)
    with open(data_path(str(order) + 'WINM21.json'), 'r', encoding="UTF-16 LE") as WI:
        dfi = pd.read_json(WI)

    dfb.index = dfb["miliseconds"]
    dfb = dfb.rename(columns={'miliseconds': 'ms', 'ask': 'bask', 'bid': 'bbid',
                               'last': 'blast', 'volume': 'bvolume', 'flags': 'bflags'})
    dfi.index = dfi["miliseconds"]
    dfi = dfi.rename(columns={'miliseconds': 'ms'})

    ds = pd.merge_ordered(dfb, dfi, fill_method="ffill")
    ds['Cask_3k'] = (ds['ask'] / ds['bask']).rolling(window=Periodo).mean()
    ds['Cbid_3k'] = (ds['bid'] / ds['bbid']).rolling(window=Periodo).mean()
    ds['Wbjusto'] = ds['bbid'] * ds['Cbid_3k']
    ds['Wajusto'] = ds['bask'] * ds['Cask_3k']

    dj = ds.copy()
    dj.index = pd.to_datetime(dj["datahora"])
    dj = dj.between_time('10:20', '16:30')
    dj.index = dj["ms"]
    dj = dj[Periodo:len(dj)]

    # [MUDANÇA 6] features novas: spread e volatilidade. Calculadas AQUI (antes
    # de cortar as primeiras `Amostra` linhas) para que o rolling de
    # VOL_WINDOW ticks já esteja "aquecido" em toda janela de treino.
    #   spread       = mid do WIN - mid do preço justo implícito pelo BOVA11
    #                  (mesma definição do spread_mispricing do rl_trading_pipeline)
    #   volatilidade = desvio-padrão do spread nos últimos VOL_WINDOW ticks
    fair_mid = (dj['Wajusto'] + dj['Wbjusto']) / 2
    dj['spread'] = (dj['ask'] + dj['bid']) / 2 - fair_mid
    dj['volatilidade'] = dj['spread'].rolling(window=VOL_WINDOW).std()

    dj['miWa'] = dj['Wajusto'].rolling(window=Amostra).mean()
    dj['miWb'] = dj['Wbjusto'].rolling(window=Amostra).mean()
    dj['StDa'] = dj['Wajusto'].rolling(window=Amostra).std()
    dj['StDb'] = dj['Wbjusto'].rolling(window=Amostra).std()

    df = dj.copy()
    df = df[Amostra:len(df)].reset_index(drop=True)

    # [MUDANÇA 2] as features cíclicas são independentes dos parâmetros do
    # sweep, então são calculadas uma única vez por pregão, aqui.
    df = add_cyclical_features(df)
    return df


def apply_heuristic(base, sigma=0.2, Re=0.75, Ri=50):  # sigma=0.2 == antigo 2 (antes do /10 sair)
    """Bandas de Bollinger + regra de entrada/saída do backtest original,
    aplicadas sobre o df base de um pregão. Lógica idêntica à original."""
    df = base.copy()

    # [MUDANÇA 8] A antiga divisão por 10 foi removida: `sigma` agora é o
    # multiplicador REAL do desvio-padrão (banda = média +/- sigma * desvio).
    # O sigma antigo x equivale ao novo x/10 (ex.: antigo 2 == novo 0.2).
    df['BOLUC'] = df['miWa'] + (sigma * df['StDa'])
    df['BOLDV'] = df['miWb'] - (sigma * df['StDb'])

    df['p'] = np.where((df['ask'] - df['BOLDV']) <= 0, df['ask'],
                        np.where((df['bid'] - df['BOLUC']) >= 0, -df['bid'], 0))

    # [MUDANÇA 5] O loop agora roda sobre arrays numpy em vez de
    # df.iterrows()/df.loc (~100x mais rápido). A lógica de entrada/saída é
    # exatamente a mesma do loop original.
    n = len(df)
    ask = df['ask'].to_numpy(dtype=float)
    bid = df['bid'].to_numpy(dtype=float)
    p = df['p'].to_numpy(dtype=float)
    miWa = df['miWa'].to_numpy(dtype=float)
    miWb = df['miWb'].to_numpy(dtype=float)

    posicao = np.zeros(n)
    entrada = np.zeros(n)
    SG = np.zeros(n)
    SL = np.zeros(n)

    Entrada = 0
    Saida = 0
    Poze = 0
    for i in range(n):
        if Poze == 0:
            if p[i] != 0:
                Poze = 1
                Entrada = ask[i]
                Saida = miWb[i]
                if p[i] < 0:
                    Poze = -1
                    Entrada = bid[i]
                    Saida = miWa[i]
        else:
            if Poze == 1:
                if bid[i] >= Entrada + (Re * abs(Saida - Entrada)):
                    Poze = 0
                elif bid[i] <= Entrada - (Ri * abs(Saida - Entrada)):
                    Poze = 0
            else:
                if ask[i] >= Entrada + (Ri * abs(Saida - Entrada)):
                    Poze = 0
                elif ask[i] <= Entrada - (Re * abs(Saida - Entrada)):
                    Poze = 0

        posicao[i] = Poze
        entrada[i] = Entrada
        SG[i] = Re * abs(Saida - Entrada)
        SL[i] = Ri * abs(Saida - Entrada)

    # [MUDANÇA 5] FECHAMENTO COMPULSÓRIO NO FIM DO PREGÃO. Antes, uma posição
    # ainda aberta no último tick era descartada (a negociação sumia do
    # dataset, enviesando os labels). Agora ela é zerada no último tick, a
    # mercado: comprado sai no bid, vendido sai no ask. Se a posição foi
    # aberta justamente no último tick não há como fechar (não existe tick
    # seguinte) e ela continua sendo descartada.
    forcado = np.zeros(n, dtype=bool)
    if n >= 2 and posicao[-1] != 0 and posicao[-2] == posicao[-1]:
        forcado[-1] = True
        lado_forcado = posicao[-1]
        posicao[-1] = 0

    df['posicao'] = posicao
    df['entrada'] = entrada
    df['SG'] = SG
    df['SL'] = SL

    df['lucro'] = np.where(
        (df['posicao'].shift(1) == 1) & (df['posicao'] == 0),
        np.where(df['bid'] <= (df['entrada'] - df['SL']), -df['SL'], df['SG']),
        np.where(
            (df['posicao'].shift(1) == -1) & (df['posicao'] == 0),
            np.where(df['ask'] >= (df['entrada'] + df['SL']), -df['SL'], df['SG']),
            0,
        ),
    )

    # [MUDANÇA 5] lucro do fechamento compulsório = resultado real a mercado
    # (em pontos, mesma unidade de SG/SL), e não SG/SL.
    df['fechamento_forcado'] = forcado
    if forcado.any():
        lucro_forcado = (bid[-1] - entrada[-1]) if lado_forcado == 1 else (entrada[-1] - ask[-1])
        df.iloc[-1, df.columns.get_loc('lucro')] = lucro_forcado
    return df


def process_day(order, Periodo=3000, Amostra=2400, sigma=0.2, Re=0.75, Ri=50):
    """Replica exatamente a lógica do backtest original para um único
    pregão (order) e devolve o df já com sinais, posicao e lucro por tick.
    """
    df = apply_heuristic(load_day_base(order, Periodo, Amostra), sigma=sigma, Re=Re, Ri=Ri)
    df['order'] = order
    return df


# --------------------------------------------------------------------------
# 2) EXTRAI AS JANELAS DE TREINAMENTO (X, y) PARA CADA NEGOCIAÇÃO
# --------------------------------------------------------------------------

# As 7 features conforme a Tabela 1 da tese.
# 'SL' já é a coluna 'DA' da tese: Ri*abs(Saida-Entrada), calculada pelo
# próprio backtest. Fica em 0 antes de qualquer entrada (Entrada/Saida
# iniciam em 0) e só assume valor quando a posição realmente abre.
#
# [MUDANÇA 6] +3 features: 'spread', 'volatilidade' e 'SG' (= Take Profit,
# Re*|Saida-Entrada|, o par do Stop Loss 'SL'). Como o SL, o TP fica em 0
# até a primeira entrada do dia e depois carrega o valor da última
# negociação até a próxima abertura; no tick de entrada (último da janela)
# já é o TP/SL da negociação que vai ser classificada -- ambos são
# conhecidos no momento da decisão, então não há vazamento.
BASE_FEATURE_COLS = ['bid', 'ask', 'Wbjusto', 'Wajusto', 'volume', 'bvolume', 'SL',
                     'spread', 'volatilidade', 'SG']
VOL_WINDOW = 120  # ticks usados no desvio-padrão do spread ('volatilidade')

# [MUDANÇA 2] 4 features cíclicas, acrescentadas AO FINAL das 7 originais.
# A ordem importa: scale_datasets() normaliza só as primeiras
# len(BASE_FEATURE_COLS) colunas, pois seno/cosseno já estão em [-1, 1].
CYCLICAL_COLS = ['day_sin', 'day_cos', 'time_sin', 'time_cos']
FEATURE_COLS = BASE_FEATURE_COLS + CYCLICAL_COLS   # 11 features no total
N_TICKS = 120  # janela de ticks anteriores à confirmação do sinal

# [MUDANÇA 11] Ablação de features: LSTM_DROP_FEATURES="SL,SG" tira essas colunas da ENTRADA da
# rede (o cache de amostras continua com as 14 colunas; a seleção é feita na leitura, em
# load_samples). Motivo: SL = Ri*d e SG = Re*d, então a razão SL/SG = Ri/Re revela a combinação
# (sigma, Re, Ri), e a taxa de lucro depende sobretudo de Ri -- a rede aprendia a taxa da
# combinação em vez de ler o mercado (docs/lstm_leitura_base.md). Padrão: nada é removido.
_DROP = [c.strip() for c in os.environ.get("LSTM_DROP_FEATURES", "").split(",") if c.strip()]
_bad = [c for c in _DROP if c not in FEATURE_COLS]
if _bad:
    raise ValueError(f"LSTM_DROP_FEATURES com nomes desconhecidos: {_bad}; válidos: {FEATURE_COLS}")
KEEP_NAMES = [c for c in FEATURE_COLS if c not in _DROP]     # ordem preservada (cíclicas no fim)
KEEP_IDX = [FEATURE_COLS.index(c) for c in KEEP_NAMES] if _DROP else None


def extract_trade_windows(df, n_ticks=N_TICKS, return_forced=False, return_pl=False):
    """Percorre o df de um dia e, a cada TRANSIÇÃO real de posição
    (flat -> comprado ou flat -> vendido), recorta a janela
    [i-n_ticks+1 : i] como features e usa o 'lucro' realizado dessa
    negociação como label.

    Importante: usar 'posicao' (transição 0->1 ou 0->-1) em vez de 'p',
    porque 'p' pode continuar diferente de zero em vários ticks seguidos
    mesmo com a posição já aberta -- usar 'p' geraria janelas duplicadas
    para o mesmo negócio.

    Retorna duas listas de (X, y): uma para compras, outra para vendas.
    Com return_forced=True devolve também, para cada lado, a lista de
    flags "esta negociação foi fechada compulsoriamente no fim do pregão"
    ([MUDANÇA 5]). Com return_pl=True devolve, como 4º elemento, o lucro
    realizado de cada negociação em PONTOS (mesma ordem das amostras).
    """
    df = df.copy()

    buy_entry = (df['posicao'].shift(1) == 0) & (df['posicao'] == 1)
    sell_entry = (df['posicao'].shift(1) == 0) & (df['posicao'] == -1)

    def process_entries(entry_mask, is_buy):
        X_list, y_list, forced_list, pl_list = [], [], [], []
        for idx in df.index[entry_mask]:
            pos = df.index.get_loc(idx)
            if pos < n_ticks - 1:
                continue  # não há histórico suficiente ainda

            window = df.iloc[pos - n_ticks + 1: pos + 1]

            # [MUDANÇA 2] FEATURE_COLS agora inclui as colunas cíclicas,
            # então cada tick da janela carrega o seu próprio dia/hora.
            X = window[FEATURE_COLS].to_numpy(dtype=float)
            if np.isnan(X).any():
                continue  # descarta janelas com NaN (ex: início do rolling)

            # label: resultado realizado da negociação aberta neste tick
            trade_close_idx = df.index[(df.index > idx) & (df['posicao'] == 0)]
            if len(trade_close_idx) == 0:
                # [MUDANÇA 5] só chega aqui se a posição abriu no último
                # tick do pregão (as demais são fechadas em apply_heuristic).
                continue
            close_idx = trade_close_idx[0]
            lucro = df.loc[close_idx, 'lucro']
            y = 1.0 if lucro > 0 else 0.0

            X_list.append(X)
            y_list.append(y)
            forced_list.append(bool(df.loc[close_idx, 'fechamento_forcado']))
            pl_list.append(float(lucro))
        return X_list, y_list, forced_list, pl_list

    buy_X, buy_y, buy_f, buy_pl = process_entries(buy_entry, is_buy=True)
    sell_X, sell_y, sell_f, sell_pl = process_entries(sell_entry, is_buy=False)

    if return_pl:
        return (buy_X, buy_y), (sell_X, sell_y), (buy_f, sell_f), (buy_pl, sell_pl)
    if return_forced:
        return (buy_X, buy_y), (sell_X, sell_y), (buy_f, sell_f)
    return (buy_X, buy_y), (sell_X, sell_y)


# --------------------------------------------------------------------------
# 3) MONTA OS DATASETS PARA VÁRIOS DIAS, COM SPLIT POR DIA
# --------------------------------------------------------------------------

def build_datasets(train_orders, val_orders, test_orders):
    """[LEGADO] Caminho original de um único par de parâmetros, com split
    fixo treino/val/teste. Não é mais usado pelo __main__ (substituído por
    generate_day_samples + run_time_series_cv); mantido só por
    compatibilidade.

    Retorna dicionários com X/y de treino, validação e teste,
    separados para compra e venda. O split é feito por PREGÃO INTEIRO
    para não vazar informação entre janelas sobrepostas do mesmo dia.
    """

    n_features = len(FEATURE_COLS)

    def to_array(X_list):
        """Garante shape (0, N_TICKS, n_features) mesmo quando a lista
        está vazia -- np.array([]) sozinho vira shape (0,), que quebra o
        Keras ao tentar montar o batch (ver erro 'Expected shape (None,
        120, 7)... incompatible shape (32,)')."""
        if len(X_list) == 0:
            return np.empty((0, N_TICKS, n_features), dtype=float)
        return np.stack(X_list).astype(float)

    def collect(orders):
        buy_X_all, buy_y_all = [], []
        sell_X_all, sell_y_all = [], []
        for order in orders:
            df = process_day(order)
            (bX, by), (sX, sy) = extract_trade_windows(df)
            buy_X_all += bX
            buy_y_all += by
            sell_X_all += sX
            sell_y_all += sy
        return (to_array(buy_X_all), np.array(buy_y_all, dtype=float),
                to_array(sell_X_all), np.array(sell_y_all, dtype=float))

    train_buy_X, train_buy_y, train_sell_X, train_sell_y = collect(train_orders)
    val_buy_X, val_buy_y, val_sell_X, val_sell_y = collect(val_orders)
    test_buy_X, test_buy_y, test_sell_X, test_sell_y = collect(test_orders)

    # Diagnóstico: sem isso, dataset vazio só aparece como um crash do
    # TensorFlow sem contexto nenhum. Sempre olhe esses números antes de
    # treinar -- se treino tiver poucas dezenas de amostras, a rede vai
    # decorar em vez de aprender.
    print("Contagem de amostras por split/lado:")
    print(f"  compra  -> treino: {len(train_buy_y)}  val: {len(val_buy_y)}  teste: {len(test_buy_y)}")
    print(f"  venda   -> treino: {len(train_sell_y)}  val: {len(val_sell_y)}  teste: {len(test_sell_y)}")

    return {
        'buy': {
            'train': (train_buy_X, train_buy_y),
            'val': (val_buy_X, val_buy_y),
            'test': (test_buy_X, test_buy_y),
        },
        'sell': {
            'train': (train_sell_X, train_sell_y),
            'val': (val_sell_X, val_sell_y),
            'test': (test_sell_X, test_sell_y),
        },
    }


# --------------------------------------------------------------------------
# 3b) [MUDANÇAS 3 e 4] SWEEP DE (sigma, Re, Ri) COM CACHE EM DISCO POR PREGÃO
# --------------------------------------------------------------------------

# Parâmetros da heurística (nomes do código):
#   sigma -> largura da banda de Bollinger, em desvios-padrão (entrada)
#   Re    -> realização: fração do caminho até o alvo (a banda/média) que
#            queremos lucrar    (SG = Re * |Saida - Entrada|)
#   Ri    -> risco: fração do caminho até o alvo que aceitamos perder
#            (SL = Ri * |Saida - Entrada|)
#
# [MUDANÇA 3] Limiares REDUZIDOS de propósito. Valores anteriores:
# sigma=2, Re=0.75 e Ri=50. Com Ri=50 o stop ficava 50x mais longe que o
# alvo e praticamente nunca disparava (daí ~30 lucros para cada prejuízo).
#   sigma menor -> bandas mais estreitas -> entra muito mais vezes;
#   Ri menor    -> stop próximo do alvo  -> muito mais negociações perdedoras;
#   Re menor    -> alvo mais curto       -> fecha mais rápido, mais amostras.
# Não importa se a heurística deixa de ser lucrativa: o objetivo aqui é
# gerar exemplos bons E ruins para a LSTM classificar.
# Este é um PONTO DE PARTIDA: ajuste a grade olhando o relatório de
# balanceamento (--stage report) antes de treinar.
#
# ESCALA: Re e Ri são FRAÇÕES do caminho até o alvo, não percentuais:
# 0.5 = 50%, 1.0 = 100%. (O antigo Ri=50 era, portanto, 5000%.)
#
# A grade pode ser sobrescrita sem editar o código, por variável de ambiente
# (listas separadas por vírgula), ex.: LSTM_RI_GRID="0.5,1,2".
def _grid_from_env(name, default):
    raw = os.environ.get(name)
    return [float(v) for v in raw.split(",")] if raw else default


# sigma 2.2-2.6 (antes 1.4-1.8): a triagem da heurística (src/analyze_spread_vs_edge.py) mostrou que o
# ganho bruto por trade cresce com sigma, enquanto o spread bid-ask custa ~5 pts fixos por trade.
SIGMA_GRID = _grid_from_env("LSTM_SIGMA_GRID", [2.2, 2.4, 2.6])
RE_GRID = _grid_from_env("LSTM_RE_GRID", [0.5, 0.75, 0.9])
RI_GRID = _grid_from_env("LSTM_RI_GRID", [0.5, 1.0, 1.5])

# [MUDANÇA 4] Todas as combinações (sigma, Re, Ri) da grade.
PARAM_GRID = [(s, re, ri) for s in SIGMA_GRID for re in RE_GRID for ri in RI_GRID]

# Cada (pregão, combinação) vira um .npz; a execução pode ser interrompida
# e retomada, e dá para acrescentar combinações à grade sem refazer as
# antigas. A chave do arquivo inclui Periodo/Amostra/N_TICKS/n_features e a
# versão da heurística (C2 = fechamento compulsório no fim do pregão + bandas sem
# a divisão por 10; caches C1 não são reaproveitados) para
# não reaproveitar cache gerado com outra configuração.
CACHE_DIR = Path(os.environ.get("LSTM_CACHE_DIR", "lstm_sample_cache"))


def _cache_file(order, sigma, Re, Ri, Periodo, Amostra):
    return CACHE_DIR / (f"o{order}_s{sigma:g}_Re{Re:g}_Ri{Ri:g}_P{Periodo}_A{Amostra}"
                        f"_N{N_TICKS}_F{len(FEATURE_COLS)}_C2.npz")


def _stack(X_list):
    """Lista de janelas -> array (n, N_TICKS, n_features) em float32, mesmo
    quando vazia (ver comentário em build_datasets sobre shape (0,))."""
    if len(X_list) == 0:
        return np.empty((0, N_TICKS, len(FEATURE_COLS)), dtype=np.float32)
    return np.stack(X_list).astype(np.float32)


def generate_day_samples(order, param_grid=PARAM_GRID, Periodo=3000, Amostra=2400):
    """[MUDANÇA 4] Roda a heurística de UM pregão para TODAS as combinações
    (sigma, Re, Ri) da grade e grava as amostras (compra/venda) no cache.
    Só processa o que ainda não está em cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pending = [c for c in param_grid
               if not _cache_file(order, *c, Periodo, Amostra).exists()]
    if not pending:
        return

    base = load_day_base(order, Periodo, Amostra)  # parte cara: 1x por pregão
    for sigma, Re, Ri in pending:
        df = apply_heuristic(base, sigma=sigma, Re=Re, Ri=Ri)
        (bX, by), (sX, sy), (bf, sf) = extract_trade_windows(df, return_forced=True)
        np.savez_compressed(
            _cache_file(order, sigma, Re, Ri, Periodo, Amostra),
            buy_X=_stack(bX), buy_y=np.asarray(by, dtype=np.float32),
            buy_forced=np.asarray(bf, dtype=bool),
            sell_X=_stack(sX), sell_y=np.asarray(sy, dtype=np.float32),
            sell_forced=np.asarray(sf, dtype=bool),
        )


# Cache LATERAL de P/L por negociação, em pontos (não altera o cache de amostras
# nem os treinos). Um .npz por (pregão, combinação), com buy_pl/sell_pl na MESMA
# ordem de buy_y/sell_y do cache de amostras; buy_y/sell_y vão junto só para
# conferência. Estágio: --stage pl (slurm/submit_lstm_pl.sbatch).
PL_DIR = Path(os.environ.get("LSTM_PL_DIR", "lstm_pl_cache"))


def _pl_file(order, sigma, Re, Ri, Periodo, Amostra):
    return PL_DIR / _cache_file(order, sigma, Re, Ri, Periodo, Amostra).name


def generate_day_pl(order, param_grid=PARAM_GRID, Periodo=3000, Amostra=2400):
    """Regera a heurística de UM pregão para todas as combinações e grava o
    lucro em pontos de cada negociação. Se a amostra correspondente já existe no
    cache principal, confere que o alinhamento bate (mesmo nº de negociações e
    mesmo sinal do lucro); qualquer divergência é erro, nunca dado desalinhado."""
    PL_DIR.mkdir(parents=True, exist_ok=True)
    pending = [c for c in param_grid if not _pl_file(order, *c, Periodo, Amostra).exists()]
    if not pending:
        return

    base = load_day_base(order, Periodo, Amostra)
    for sigma, Re, Ri in pending:
        df = apply_heuristic(base, sigma=sigma, Re=Re, Ri=Ri)
        (_, by), (_, sy), (bf, sf), (bpl, spl) = extract_trade_windows(
            df, return_forced=True, return_pl=True)
        out = dict(buy_pl=np.asarray(bpl, dtype=np.float32), buy_y=np.asarray(by, dtype=np.float32),
                   buy_forced=np.asarray(bf, dtype=bool),
                   sell_pl=np.asarray(spl, dtype=np.float32), sell_y=np.asarray(sy, dtype=np.float32),
                   sell_forced=np.asarray(sf, dtype=bool))
        ref = _cache_file(order, sigma, Re, Ri, Periodo, Amostra)
        if ref.exists():
            with np.load(ref) as z:   # npz é lazy: só lê y e forced, não o X
                for side in SIDES:
                    if not (np.array_equal(z[f'{side}_y'], out[f'{side}_y']) and
                            np.array_equal(z[f'{side}_forced'], out[f'{side}_forced'])):
                        raise RuntimeError(f"P/L desalinhado do cache de amostras: pregão {order}, "
                                           f"(sigma, Re, Ri)=({sigma:g}, {Re:g}, {Ri:g}), lado {side}")
        final = _pl_file(order, sigma, Re, Ri, Periodo, Amostra)
        tmp = final.with_name(final.stem + ".tmp.npz")
        np.savez_compressed(tmp, **out)
        os.replace(tmp, final)   # atômico: retomada nunca vê arquivo pela metade


def load_sample_pl(orders, param_grid=PARAM_GRID, Periodo=3000, Amostra=2400):
    """Lucro em pontos de cada amostra, na MESMA ordem de load_samples(orders,
    param_grid) e portanto de y/p em predictions.npz. Retorna {'buy': array,
    'sell': array} em float32."""
    out = {}
    for side in SIDES:
        parts = []
        for order in orders:
            for combo in param_grid:
                with np.load(_pl_file(order, *combo, Periodo, Amostra)) as z:
                    parts.append(z[f'{side}_pl'])
        out[side] = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    return out


def load_samples(orders, param_grid=PARAM_GRID, Periodo=3000, Amostra=2400):
    """[MUDANÇA 4] Junta as amostras de TODAS as combinações da grade para
    os pregões dados. Retorna {'buy': (X, y), 'sell': (X, y)}.

    Como o split é por pregão e cada pregão traz TODAS as combinações, o
    mesmo pregão nunca aparece em dois splits diferentes -- isso mantém o
    split consistente entre todos os parâmetros do sweep (e evita
    vazamento: amostras de combinações diferentes no mesmo dia são quase
    cópias umas das outras)."""
    out = {}
    files = [_cache_file(order, *combo, Periodo, Amostra)
             for order in orders for combo in param_grid]
    for side in ('buy', 'sell'):
        def _read(path, side=side):
            with np.load(path) as z:
                y = z[f'{side}_y']
                if not len(y):
                    return None
                X = z[f'{side}_X']
                return (X[:, :, KEEP_IDX] if KEEP_IDX is not None else X, y)
        # leitura em threads (I/O + descompressão): milhares de .npz pequenos
        # levavam >10 min em série; map() preserva a ordem
        with ThreadPoolExecutor(max_workers=int(os.environ.get("LSTM_LOAD_THREADS", "12"))) as ex:
            parts = [p for p in ex.map(_read, files) if p is not None]
        Xs = [p[0] for p in parts]
        ys = [p[1] for p in parts]
        X = np.concatenate(Xs) if Xs else np.empty((0, N_TICKS, len(KEEP_NAMES)), dtype=np.float32)
        y = np.concatenate(ys) if ys else np.empty((0,), dtype=np.float32)
        out[side] = (X, y)
    return out


def load_sample_meta(orders, param_grid=PARAM_GRID, Periodo=3000, Amostra=2400):
    """[MUDANÇA 10] Para cada amostra devolvida por load_samples (MESMA ordem),
    o pregão de origem e o índice da combinação (sigma, Re, Ri) na grade. Serve
    para o relatório quebrar as métricas por combinação e por dia."""
    out = {}
    jobs = [(order, ci, _cache_file(order, *combo, Periodo, Amostra))
            for order in orders for ci, combo in enumerate(param_grid)]
    for side in ('buy', 'sell'):
        def _len(job, side=side):
            with np.load(job[2]) as z:
                return len(z[f'{side}_y'])
        with ThreadPoolExecutor(max_workers=int(os.environ.get("LSTM_LOAD_THREADS", "12"))) as ex:
            lens = list(ex.map(_len, jobs))          # map() preserva a ordem
        ords = [np.full(n, j[0], dtype=np.int32) for j, n in zip(jobs, lens)]
        combos = [np.full(n, j[1], dtype=np.int16) for j, n in zip(jobs, lens)]
        out[side] = (np.concatenate(ords) if ords else np.empty(0, np.int32),
                     np.concatenate(combos) if combos else np.empty(0, np.int16))
    return out


# [MUDANÇA 11] Balanceamento POR COMBINAÇÃO: pesos de amostra tais que, dentro de cada
# combinação (sigma, Re, Ri), lucro e prejuízo pesem igual (50/50). O peso total de cada
# combinação continua proporcional ao seu nº de amostras. Assim a taxa de lucro da combinação
# deixa de ser aprendível (o melhor palpite constante passa a ser 0,5 em TODA combinação) e
# sobra, como sinal, só o que distingue oportunidades DENTRO da combinação. Usado também na
# validação, para o early stopping medir esse sinal e não o atalho.
COMBO_BALANCE = os.environ.get("LSTM_COMBO_BALANCE", "0") not in ("", "0", "false", "False")


def combo_balance_weights(y, combo):
    y = np.asarray(y, dtype=np.float32)
    w = np.ones(len(y), dtype=np.float32)
    for c in np.unique(combo):
        m = combo == c
        n1 = float(y[m].sum())
        n0 = float(m.sum()) - n1
        if n1 > 0 and n0 > 0:
            w[m & (y > 0.5)] = m.sum() / (2.0 * n1)
            w[m & (y <= 0.5)] = m.sum() / (2.0 * n0)
    return w * (len(w) / w.sum())                   # média 1


def label_balance_report(orders, param_grid=PARAM_GRID, Periodo=3000, Amostra=2400,
                         csv_path="lstm_sweep_label_balance.csv"):
    """[MUDANÇA 4] Para cada combinação (sigma, Re, Ri) e lado, conta
    negociações, lucros/prejuízos, quantas foram fechadas compulsoriamente
    no fim do pregão e um índice de balanceamento:

        balanco = 1 - |2*taxa_lucro - 1|      (1.0 = 50/50, 0.0 = só uma classe)

    Use SOMENTE pregões de desenvolvimento (treino/validação) para escolher
    a grade -- não olhe o teste para decidir parâmetros."""
    rows = []
    for sigma, Re, Ri in param_grid:
        for side in ('buy', 'sell'):
            ys, fs = [], []
            for order in orders:
                with np.load(_cache_file(order, sigma, Re, Ri, Periodo, Amostra)) as z:
                    ys.append(z[f'{side}_y'])  # npz é lazy: não carrega o X
                    fs.append(z[f'{side}_forced'])
            y = np.concatenate(ys) if ys else np.empty((0,))
            n, wins = len(y), int(y.sum())
            rate = wins / n if n else float('nan')
            rows.append({'sigma': sigma, 'Re': Re, 'Ri': Ri, 'lado': side, 'n': n,
                         'lucros': wins, 'prejuizos': n - wins,
                         'fechados_forcado': int(np.concatenate(fs).sum()) if fs else 0,
                         'taxa_lucro': rate,
                         'balanco': 1 - abs(2 * rate - 1) if n else 0.0})
    report = pd.DataFrame(rows)
    report.to_csv(csv_path, index=False)

    print("\n=== Balanceamento de labels por combinação (pregões de desenvolvimento) ===")
    print(report.sort_values(['lado', 'balanco'], ascending=[True, False]).to_string(index=False))
    tot = report.groupby('lado')[['n', 'lucros', 'prejuizos', 'fechados_forcado']].sum()
    print("\nTotal (todas as combinações somadas):")
    print(tot.to_string())
    print(f"\nRelatório salvo em {csv_path}")
    return report


# --------------------------------------------------------------------------
# 4) NORMALIZAÇÃO — SCALER AJUSTADO SOMENTE NO TREINO
# --------------------------------------------------------------------------

def scale_datasets(splits):
    """Ajusta um StandardScaler por feature, usando SOMENTE os dados de
    treino, e aplica o mesmo scaler em treino/val/teste. Achata as
    janelas (n_amostras*n_ticks, n_features) para ajustar o scaler, depois
    devolve ao formato 3D exigido pelo Keras.

    [MUDANÇA 2] Só as 7 features originais são normalizadas; as 4
    cíclicas (últimas colunas) já estão em [-1, 1] e ficam como estão.
    """
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    if len(X_train) == 0:
        # Não há como ajustar o scaler sem dados de treino; devolve tudo
        # como está e deixa train_and_evaluate decidir pular este lado.
        return {'train': (X_train, y_train), 'val': (X_val, y_val),
                'test': (X_test, y_test), 'scaler': None}

    # colunas base mantidas (as cíclicas ficam sempre no fim e sem normalizar)
    n_scale = sum(1 for c in KEEP_NAMES if c in BASE_FEATURE_COLS)

    scaler = StandardScaler()
    scaler.fit(X_train[:, :, :n_scale].reshape(-1, n_scale))

    def apply_scaler(X):
        if len(X) == 0:
            return X
        X = X.copy()
        X[:, :, :n_scale] = scaler.transform(
            X[:, :, :n_scale].reshape(-1, n_scale)).reshape(X.shape[0], X.shape[1], n_scale)
        return X

    return {
        'train': (apply_scaler(X_train), y_train),
        'val': (apply_scaler(X_val), y_val),
        'test': (apply_scaler(X_test), y_test),
        'scaler': scaler,
    }


# --------------------------------------------------------------------------
# 5) ARQUITETURA DA LSTM (conforme especificado na tese)
# --------------------------------------------------------------------------

# [MUDANÇA 9] Tamanho da rede e dropout configuráveis por variável de ambiente,
# para experimentos de ablação. O PADRÃO continua sendo o da tese: 2x LSTM(50),
# sem dropout. Dropout é aplicado com camadas Dropout separadas (e não no
# argumento `dropout` da LSTM) para não tirar a LSTM do kernel cuDNN da GPU.
LSTM_UNITS = int(os.environ.get("LSTM_UNITS", "50"))
LSTM_DROPOUT = float(os.environ.get("LSTM_DROPOUT", "0.0"))


def build_lstm(n_ticks, n_features, units=None, dropout=None):
    units = LSTM_UNITS if units is None else units
    dropout = LSTM_DROPOUT if dropout is None else dropout

    def drop():
        return [layers.Dropout(dropout)] if dropout > 0 else []

    model = keras.Sequential([
        layers.Input(shape=(n_ticks, n_features)),
        layers.LSTM(units, return_sequences=True),
        *drop(),
        layers.LSTM(units),
        *drop(),
        layers.Dense(1, activation='sigmoid'),  # sigmoid: saída em [0,1] (prob. de Lucro)
    ])
    # A tese cita MSE como loss; para classificação binária, binary_crossentropy
    # costuma treinar melhor e mais rápido. Deixo MSE comentado como alternativa
    # fiel ao texto original.
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='binary_crossentropy',   # ou 'mse', se quiser seguir a tese à risca
        metrics=['accuracy'],
    )
    return model


# --------------------------------------------------------------------------
# 6) TREINO + AVALIAÇÃO
# --------------------------------------------------------------------------

def oversample_minority(X, y, random_state=42):
    """Duplica (com reposição) as amostras da classe minoritária até
    igualar a contagem da classe majoritária.

    IMPORTANTE: só deve ser chamada com o conjunto de TREINO, depois de
    já ter passado pelo scaler. Nunca aplique isso em validação/teste --
    duplicar exemplos nesses conjuntos infla artificialmente as métricas
    (o modelo "acerta" o mesmo exemplo várias vezes) e você perde a
    capacidade de medir o desempenho real em dados não vistos.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)

    if len(classes) < 2:
        return X, y  # só uma classe presente, nada a balancear

    majority_count = counts.max()
    X_parts, y_parts = [X], [y]

    for cls, cnt in zip(classes, counts):
        if cnt == majority_count:
            continue
        idx_cls = np.where(y == cls)[0]
        n_to_add = majority_count - cnt
        extra_idx = rng.choice(idx_cls, size=n_to_add, replace=True)
        X_parts.append(X[extra_idx])
        y_parts.append(y[extra_idx])

    X_bal = np.concatenate(X_parts, axis=0)
    y_bal = np.concatenate(y_parts, axis=0)

    # embaralha para não deixar todas as duplicatas agrupadas no final
    # (facilita a vida de qualquer split manual futuro e evita vieses de
    # ordem durante o treinamento)
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


MIN_SAMPLES = 30  # mínimo arbitrário para sequer tentar treinar uma LSTM


def train_and_evaluate(scaled, label, epochs=50, batch_size=32, use_oversampling=True):
    X_train, y_train = scaled['train']
    X_val, y_val = scaled['val']
    X_test, y_test = scaled['test']

    if len(X_train) < MIN_SAMPLES or len(X_val) == 0 or len(X_test) == 0:
        print(f"\n[{label}] Dados insuficientes para treinar "
              f"(treino={len(X_train)}, val={len(X_val)}, teste={len(X_test)}). "
              f"Pulei este lado -- aumente o período de dias ou revise os "
              f"parâmetros do backtest para gerar mais negociações.")
        return None, None

    if use_oversampling:
        n_before = len(y_train)
        pos_before = int(y_train.sum())
        X_train, y_train = oversample_minority(X_train, y_train)
        print(f"[{label}] Oversampling: treino foi de {n_before} "
              f"({pos_before} lucro / {n_before - pos_before} prejuízo) "
              f"para {len(y_train)} ({int(y_train.sum())} lucro / "
              f"{len(y_train) - int(y_train.sum())} prejuízo).")

    n_ticks, n_features = X_train.shape[1], X_train.shape[2]
    model = build_lstm(n_ticks, n_features)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )

    # class_weight é uma alternativa ao oversampling para lidar com
    # desbalanceamento -- usar os dois ao mesmo tempo corrige o problema
    # duas vezes e tende a enviesar o modelo para o lado oposto.
    if use_oversampling:
        class_weight = None
    else:
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)} if n_pos > 0 else None

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print(f"\n=== Resultados no teste ({label}) ===")
    print("Acurácia :", accuracy_score(y_test, y_pred))
    print("Precisão :", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

    return model, history


# --------------------------------------------------------------------------
# 7) [MUDANÇA 1] VALIDAÇÃO TEMPORAL COM TimeSeriesSplit
# --------------------------------------------------------------------------

def _split_metrics(model, X, y):
    """Métricas de classificação de um split. Com classes desbalanceadas a
    acurácia sozinha engana, então reporta também acurácia balanceada e
    AUC (que não depende do limiar de 0.5)."""
    return _metrics_from_prob(y, model.predict(X, batch_size=1024, verbose=0).flatten())


def _metrics_from_prob(y, prob):
    pred = (prob >= 0.5).astype(int)
    both = len(np.unique(y)) == 2
    return {
        'n': len(y), 'taxa_lucro': float(np.mean(y)),
        'acc': accuracy_score(y, pred),
        'bal_acc': balanced_accuracy_score(y, pred),
        'prec': precision_score(y, pred, zero_division=0),
        'rec': recall_score(y, pred, zero_division=0),
        'auc': roc_auc_score(y, prob) if both else float('nan'),
    }


def run_time_series_cv(dev_orders, test_orders, param_grid=PARAM_GRID, n_splits=5,
                       use_oversampling=False, epochs=50, batch_size=32,
                       csv_path="lstm_cv_results.csv"):
    """[MUDANÇA 1] Validação cruzada temporal (expanding window):

        fold 1: treino [B1]              | val [B2]
        fold 2: treino [B1 B2]           | val [B3]
        ...
        fold k: treino [B1 ... Bk]       | val [Bk+1]      teste final: [T]

    - O split é feito sobre a lista de PREGÕES (em ordem cronológica), não
      sobre amostras: todas as amostras de um mesmo dia (de todas as
      combinações do sweep) ficam juntas no mesmo lado do corte.
    - Cada fold tem o seu próprio scaler e o seu próprio oversampling,
      ajustados só com o treino daquele fold.
    - `test_orders` (últimos pregões) nunca entra em nenhum fold; cada
      modelo é avaliado nele também, então dá para ver a estabilidade
      entre folds.

    Retorna {'buy': DataFrame, 'sell': DataFrame} com as métricas por fold
    e os modelos do último fold (o que viu mais dados) em `models`.
    """
    dev = np.asarray(dev_orders)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    test = load_samples(test_orders, param_grid)

    rows = {'buy': [], 'sell': []}
    models = {}

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(dev), start=1):
        tr_orders, va_orders = dev[tr_idx].tolist(), dev[va_idx].tolist()
        # Sanidade: treino estritamente antes da validação e sem pregões em comum.
        assert tr_idx.max() < va_idx.min() and not set(tr_orders) & set(va_orders)
        print(f"\n################ Fold {fold}/{n_splits} ################")
        print(f"  treino: {len(tr_orders)} pregões ({tr_orders[0]}..{tr_orders[-1]}) | "
              f"val: {len(va_orders)} pregões ({va_orders[0]}..{va_orders[-1]}) | "
              f"teste: {len(test_orders)} pregões")

        train = load_samples(tr_orders, param_grid)
        val = load_samples(va_orders, param_grid)

        for side in ('buy', 'sell'):
            print(f"\n---------- Lado: {side} | fold {fold} ----------")
            scaled = scale_datasets({'train': train[side], 'val': val[side], 'test': test[side]})
            model, _ = train_and_evaluate(scaled, label=f"{side} fold {fold}",
                                          epochs=epochs, batch_size=batch_size,
                                          use_oversampling=use_oversampling)
            if model is None:
                continue
            for split_name in ('val', 'test'):
                Xs, ys = scaled[split_name]
                rows[side].append({'fold': fold, 'split': split_name,
                                   **_split_metrics(model, Xs, ys)})
            models[side] = model  # sobrescreve: fica o do último fold
            if fold < n_splits:
                keras.backend.clear_session()  # libera memória entre folds

    results = {}
    for side in ('buy', 'sell'):
        df = pd.DataFrame(rows[side])
        results[side] = df
        if df.empty:
            continue
        print(f"\n=== Validação temporal ({side}): métricas por fold ===")
        print(df.to_string(index=False))
        print(f"\n--- média/desvio entre folds ({side}) ---")
        print(df.groupby('split')[['acc', 'bal_acc', 'prec', 'rec', 'auc']]
                .agg(['mean', 'std']).round(4).to_string())
    pd.concat({s: d for s, d in results.items() if not d.empty},
              names=['lado']).to_csv(csv_path)
    print(f"\nMétricas por fold salvas em {csv_path}")
    return results, models


# --------------------------------------------------------------------------
# 8) [MUDANÇA 7] EXECUÇÃO NO SANTOS DUMONT: 1 TAREFA = (fold, lado), COM CHECKPOINT
# --------------------------------------------------------------------------
# Jobs do Slurm duram no máximo 20 min e a conta permite 1 job por vez, então
# cada (fold, lado) é uma TAREFA que treina em "pedaços" de até
# LSTM_TRAIN_MAX_SECONDS e retoma de onde parou. Estado em disco, por tarefa,
# em RUN_ROOT/fold<k>_<lado>/:
#   state.json        época, melhor val_loss, paciência, histórico, flags
#   last.keras        modelo + estado do otimizador (regravado a cada época)
#   best.weights.h5   melhores pesos na validação (early stopping)
#   metrics.json      métricas val/teste do melhor modelo (tarefa concluída)
#   model.keras / scaler.npz   modelo final e média/desvio das features

RUN_TAG = os.environ.get("LSTM_RUN_TAG", "lstm_v1")
RUN_ROOT = Path(os.environ.get("LSTM_RUN_DIR", "lstm_runs")) / RUN_TAG
N_SPLITS = int(os.environ.get("LSTM_N_SPLITS", "5"))
SIDES = ('buy', 'sell')

# Pregões 1..488 em ordem cronológica. Os pregões 338 e 463 estão ruins e ficam de
# fora por padrão (restam 486). O teste final = últimos 15% (~72 pregões).
# Também ficam de fora, por padrão, 168, 304, 336 e 371: cotações impossíveis (bid > ask, saltos de
# 7.000 a 112.000 pts) achadas por src/scan_bad_sessions.py. Para mudar sem editar o código:
# LSTM_EXCLUDE_ORDERS="338,463,<outro>" (ou "" p/ nenhum).
_EXCLUDED = {int(v) for v in os.environ.get("LSTM_EXCLUDE_ORDERS", "168,304,336,338,371,463").split(",") if v}
LAST_ORDER = int(os.environ.get("LSTM_LAST_ORDER", "488"))   # último pregão usado (os dados vão de 1 a 488)
ALL_ORDERS = [o for o in range(1, LAST_ORDER + 1) if o not in _EXCLUDED]


TEST_FRAC = float(os.environ.get("LSTM_TEST_FRAC", "0.30"))   # antes 0.15; dobrado a pedido


def split_orders(orders=None, test_frac=None):
    """[MUDANÇA 1] Últimos TEST_FRAC (padrão 30%) = teste final; o restante = base do TimeSeriesSplit."""
    orders = list(ALL_ORDERS if orders is None else orders)
    test_frac = TEST_FRAC if test_frac is None else test_frac
    n_dev = len(orders) - int(len(orders) * test_frac)
    return orders[:n_dev], orders[n_dev:]


def fold_orders(dev_orders, fold, n_splits=N_SPLITS):
    """Pregões de treino e validação do fold `fold` (1..n_splits) do
    TimeSeriesSplit -- o MESMO split para todas as combinações do sweep."""
    dev = np.asarray(dev_orders)
    for k, (tr_idx, va_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(dev), start=1):
        if k == fold:
            assert tr_idx.max() < va_idx.min()
            return dev[tr_idx].tolist(), dev[va_idx].tolist()
    raise ValueError(f"fold {fold} fora de 1..{n_splits}")


def task_dir(fold, side):
    return RUN_ROOT / f"fold{fold}_{side}"


def _write_json_atomic(path, obj):
    """Grava em arquivo temporário e renomeia: um job morto pelo Slurm no
    meio da escrita nunca deixa um JSON/checkpoint pela metade."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _new_state(fold, side):
    return {'fold': fold, 'side': side, 'epochs_done': 0, 'best_val_loss': float('inf'),
            'wait': 0, 'done': False, 'finished': False, 'chunks': 0, 'history': []}


def read_state(fold, side):
    p = task_dir(fold, side) / "state.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return _new_state(fold, side)


def pending_tasks(n_splits=N_SPLITS):
    """Tarefas (fold, lado) ainda não concluídas, na ordem fold 1 buy, fold 1 sell, ..."""
    return [(f, s) for f in range(1, n_splits + 1) for s in SIDES
            if not read_state(f, s)['finished']]


def _make_checkpoint_callback(tdir, state, patience, max_epochs, deadline):
    """Early stopping + checkpoint que sobrevive a reinício do processo (o
    EarlyStopping do Keras perde o contador de paciência ao reiniciar)."""

    class _Checkpoint(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            dur = time.time() - self._t0
            val_loss = float(logs['val_loss'])
            state['epochs_done'] = epoch + 1
            state['history'].append({
                'epoch': epoch + 1, 'sec': round(dur, 1), 'loss': float(logs['loss']),
                'val_loss': val_loss, 'acc': float(logs.get('accuracy', float('nan'))),
                'val_acc': float(logs.get('val_accuracy', float('nan')))})

            if val_loss < state['best_val_loss']:
                state['best_val_loss'], state['wait'] = val_loss, 0
                tmp = tdir / "best.tmp.weights.h5"
                self.model.save_weights(tmp)
                os.replace(tmp, tdir / "best.weights.h5")
            else:
                state['wait'] += 1

            tmp = tdir / "last.tmp.keras"
            self.model.save(tmp)                      # pesos + otimizador
            os.replace(tmp, tdir / "last.keras")

            if state['wait'] >= patience or state['epochs_done'] >= max_epochs:
                state['done'] = True                  # treino terminou de fato
                self.model.stop_training = True
            elif deadline and time.time() + 1.15 * dur > deadline:
                self.model.stop_training = True       # sem tempo p/ mais uma época: próximo job retoma
            _write_json_atomic(tdir / "state.json", state)

    return _Checkpoint()


def train_task(fold, side, param_grid=PARAM_GRID, batch_size=256, max_epochs=50,
               patience=8, use_oversampling=False, max_seconds=0):
    """Treina (ou RETOMA) a LSTM de um (fold, lado) até acabar o orçamento de
    tempo `max_seconds` (contado desde o início do processo; 0 = sem limite)
    ou o early stopping. Quando o treino termina, avalia o melhor modelo em
    validação e teste e grava metrics.json."""
    if keras is None:
        raise RuntimeError("TensorFlow não está instalado neste ambiente.")
    tdir = task_dir(fold, side)
    tdir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("LSTM_SEED"):   # [MUDANÇA 9] semente opcional (repetições p/ medir ruído)
        keras.utils.set_random_seed(int(os.environ["LSTM_SEED"]))
    state = read_state(fold, side)
    if state['finished']:
        print(f"[fold{fold} {side}] já concluído.")
        return state

    dev_orders, test_orders = split_orders()
    tr_orders, va_orders = fold_orders(dev_orders, fold)
    print(f"[fold{fold} {side}] treino {tr_orders[0]}..{tr_orders[-1]} ({len(tr_orders)} pregões) | "
          f"val {va_orders[0]}..{va_orders[-1]} ({len(va_orders)}) | teste {len(test_orders)} pregões | "
          f"época {state['epochs_done']} | pedaço {state['chunks'] + 1}", flush=True)

    t0 = time.time()
    # Arrays já normalizados ficam num arquivo único no diretório da tarefa: a
    # retomada (próximo job) lê 1 arquivo em vez de milhares de .npz. A chave
    # cobre pregões, grade e lado; só é gravado quando a carga foi lenta.
    import hashlib
    cache_key = hashlib.md5(repr((
        [int(o) for o in tr_orders], [int(o) for o in va_orders], [int(o) for o in test_orders],
        [tuple(map(float, c)) for c in param_grid], side, N_TICKS, list(KEEP_NAMES))).encode()).hexdigest()
    scaled_path = tdir / "scaled_data.npz"
    scaled = None
    if scaled_path.exists():
        try:
            with np.load(scaled_path) as z:
                if str(z['key']) == cache_key:
                    scaled = {'train': (z['X_train'], z['y_train']), 'val': (z['X_val'], z['y_val']),
                              'test': (z['X_test'], z['y_test']),
                              'scaler': SimpleNamespace(mean_=z['mean'], scale_=z['scale'])}
                    print(f"[fold{fold} {side}] dados normalizados lidos de {scaled_path.name}", flush=True)
        except Exception as e:   # arquivo truncado por timeout etc.: recarrega
            print(f"[fold{fold} {side}] cache normalizado ignorado ({e!r})", flush=True)
            scaled = None
    if scaled is None:
        train = load_samples(tr_orders, param_grid)[side]
        val = load_samples(va_orders, param_grid)[side]
        test = load_samples(test_orders, param_grid)[side]
        scaled = scale_datasets({'train': train, 'val': val, 'test': test})
        del train, val, test
        if scaled['scaler'] is not None and \
                time.time() - t0 > float(os.environ.get("LSTM_SCALED_CACHE_MIN_SECONDS", "120")):
            tmp_path = tdir / "scaled_data.tmp.npz"
            np.savez(tmp_path, key=cache_key,
                     X_train=scaled['train'][0], y_train=scaled['train'][1],
                     X_val=scaled['val'][0], y_val=scaled['val'][1],
                     X_test=scaled['test'][0], y_test=scaled['test'][1],
                     mean=scaled['scaler'].mean_, scale=scaled['scaler'].scale_)
            os.replace(tmp_path, scaled_path)
            print(f"[fold{fold} {side}] dados normalizados gravados em {scaled_path.name} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    X_train, y_train = scaled['train']
    X_val, y_val = scaled['val']
    X_test, y_test = scaled['test']
    print(f"[fold{fold} {side}] amostras treino/val/teste: {len(y_train)}/{len(y_val)}/{len(y_test)} "
          f"(lucro no treino: {y_train.mean() if len(y_train) else float('nan'):.3f}); "
          f"carga+normalização: {time.time() - t0:.0f}s", flush=True)

    if len(X_train) < MIN_SAMPLES or len(X_val) == 0 or len(X_test) == 0:
        print(f"[fold{fold} {side}] dados insuficientes -- tarefa pulada.")
        state.update(done=True, finished=True, skipped=True)
        _write_json_atomic(tdir / "state.json", state)
        return state

    class_weight = None
    sw_train = sw_val = combo_rate_train = None
    if COMBO_BALANCE:
        # [MUDANÇA 11] cada combinação 50/50 (treino e validação); substitui class_weight/oversampling
        c_tr = load_sample_meta(tr_orders, param_grid)[side][1]
        c_va = load_sample_meta(va_orders, param_grid)[side][1]
        if len(c_tr) != len(y_train) or len(c_va) != len(y_val):
            raise RuntimeError("meta das combinações desalinhada dos dados normalizados")
        sw_train, sw_val = combo_balance_weights(y_train, c_tr), combo_balance_weights(y_val, c_va)
        combo_rate_train = np.full(len(param_grid), np.nan, dtype=np.float32)
        for ci in np.unique(c_tr):
            combo_rate_train[ci] = float(y_train[c_tr == ci].mean())
        print(f"[fold{fold} {side}] balanceamento por combinação: {len(np.unique(c_tr))} combinações; "
              f"taxa de lucro no treino por combinação: {np.nanmin(combo_rate_train):.2f}-{np.nanmax(combo_rate_train):.2f} "
              f"(pesos 50/50)", flush=True)
    elif use_oversampling:
        X_train, y_train = oversample_minority(X_train, y_train)   # seed fixa -> mesmo em todo pedaço
    else:
        n_pos = float(y_train.sum())
        class_weight = {0: 1.0, 1: (len(y_train) - n_pos) / n_pos} if n_pos > 0 else None

    last = tdir / "last.keras"
    if state['epochs_done'] > 0 and last.exists():
        model = keras.models.load_model(last)
        print(f"[fold{fold} {side}] retomado de {last} (época {state['epochs_done']}).", flush=True)
    else:
        state = _new_state(fold, side)
        model = build_lstm(X_train.shape[1], X_train.shape[2])
    state['chunks'] += 1

    if state['epochs_done'] >= max_epochs:
        state['done'] = True
    if not state['done']:
        deadline = SCRIPT_START + max_seconds - 60 if max_seconds else 0  # 60s p/ avaliação final
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val) if sw_val is None else (X_val, y_val, sw_val),
                  epochs=max_epochs, initial_epoch=state['epochs_done'],
                  batch_size=batch_size, class_weight=class_weight, sample_weight=sw_train,
                  callbacks=[_make_checkpoint_callback(tdir, state, patience, max_epochs, deadline)],
                  verbose=2)
    _write_json_atomic(tdir / "state.json", state)

    if not state['done']:
        print(f"[fold{fold} {side}] orçamento de tempo esgotado na época {state['epochs_done']}; "
              f"o próximo job retoma.", flush=True)
        return state

    # --- treino terminou: avalia o MELHOR modelo (early stopping) ---
    if (tdir / "best.weights.h5").exists():
        model.load_weights(tdir / "best.weights.h5")
    # [MUDANÇA 10] Guarda as PREDIÇÕES (não só métricas escalares) e a
    # configuração, para o relatório (src/report_lstm.py) poder desenhar ROC,
    # matriz de confusão, calibração e quebras por combinação/dia depois, sem
    # o modelo nem os dados. Treino: subamostra de até 50 mil (antes do
    # class_weight, que não altera os X), p/ comparar treino x val x teste.
    p_val = model.predict(X_val, batch_size=1024, verbose=0).flatten()
    p_test = model.predict(X_test, batch_size=1024, verbose=0).flatten()
    rng = np.random.default_rng(0)
    tr_idx = np.sort(rng.choice(len(y_train), size=min(50000, len(y_train)), replace=False))
    p_train = model.predict(X_train[tr_idx], batch_size=1024, verbose=0).flatten()
    meta_val = load_sample_meta(va_orders, param_grid)[side]
    meta_test = load_sample_meta(test_orders, param_grid)[side]
    np.savez_compressed(
        tdir / "predictions.npz",
        y_val=y_val, p_val=p_val, order_val=meta_val[0], combo_val=meta_val[1],
        y_test=y_test, p_test=p_test, order_test=meta_test[0], combo_test=meta_test[1],
        y_train=y_train[tr_idx], p_train=p_train,
        grid=np.asarray(param_grid, dtype=float),
        **({'combo_rate_train': combo_rate_train} if combo_rate_train is not None else {}))
    metrics = {'fold': fold, 'side': side, 'epochs': state['epochs_done'],
               'best_val_loss': state['best_val_loss'], 'n_train': int(len(y_train)),
               'val': _metrics_from_prob(y_val, p_val),
               'test': _metrics_from_prob(y_test, p_test),
               'config': {'tag': RUN_TAG, 'grid': [list(map(float, c)) for c in param_grid],
                          'features': KEEP_NAMES, 'n_ticks': N_TICKS,
                          'combo_balance': bool(COMBO_BALANCE), 'drop_features': _DROP,
                          'units': LSTM_UNITS, 'dropout': LSTM_DROPOUT,
                          'batch_size': batch_size, 'max_epochs': max_epochs, 'patience': patience,
                          'class_weight': (class_weight is not None), 'oversampling': use_oversampling,
                          'seed': os.environ.get("LSTM_SEED"),
                          'train_orders': [int(tr_orders[0]), int(tr_orders[-1])],
                          'val_orders': [int(va_orders[0]), int(va_orders[-1])],
                          'test_orders': [int(test_orders[0]), int(test_orders[-1])],
                          'n_train_days': len(tr_orders), 'n_val_days': len(va_orders),
                          'n_test_days': len(test_orders)}}
    _write_json_atomic(tdir / "metrics.json", metrics)
    model.save(tdir / "model.keras")
    if scaled['scaler'] is not None:
        np.savez(tdir / "scaler.npz", mean=scaled['scaler'].mean_, scale=scaled['scaler'].scale_)
    state['finished'] = True
    _write_json_atomic(tdir / "state.json", state)
    print(f"[fold{fold} {side}] CONCLUÍDO: val {metrics['val']} | teste {metrics['test']}", flush=True)
    return state


def aggregate_results(n_splits=N_SPLITS, csv_path=None):
    """Reúne os metrics.json de todas as tarefas concluídas."""
    csv_path = csv_path or RUN_ROOT / "lstm_cv_results.csv"
    rows = []
    for fold in range(1, n_splits + 1):
        for side in SIDES:
            p = task_dir(fold, side) / "metrics.json"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            for split_name in ('val', 'test'):
                rows.append({'lado': side, 'fold': fold, 'split': split_name,
                             'epochs': m['epochs'], **m[split_name]})
    if not rows:
        print("Nenhuma tarefa concluída ainda.")
        return None
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n--- média/desvio entre folds ---")
    print(df.groupby(['lado', 'split'])[['acc', 'bal_acc', 'prec', 'rec', 'auc']]
            .agg(['mean', 'std']).round(4).to_string())
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSalvo em {csv_path}")
    return df


def compare_runs(csv_path=None):
    """[MUDANÇA 9] Compara os experimentos (um RUN_TAG cada) lado a lado, usando
    os lstm_cv_results.csv já gerados em LSTM_RUN_DIR/<tag>/."""
    root = RUN_ROOT.parent
    frames = []
    # LSTM_COMPARE_PREFIX restringe a comparação a uma campanha (ex.: "lstm_v2"), já que experimentos
    # com grades/testes diferentes não são comparáveis.
    prefix = os.environ.get("LSTM_COMPARE_PREFIX", "")
    for csv in sorted(root.glob(prefix + "*/lstm_cv_results.csv")):
        d = pd.read_csv(csv)
        d.insert(0, 'experimento', csv.parent.name)
        frames.append(d)
    if not frames:
        print(f"Nenhum lstm_cv_results.csv em {root}/*/ (rode --stage aggregate por experimento).")
        return None
    df = pd.concat(frames)
    tab = (df.groupby(['lado', 'split', 'experimento'])[['auc', 'bal_acc', 'prec', 'rec']]
             .agg(['mean', 'std']).round(4))
    print(tab.to_string())
    out = csv_path or root / "comparacao_experimentos.csv"
    tab.to_csv(out)
    print(f"\nSalvo em {out}")
    return tab


def _generate_worker(order):
    try:
        generate_day_samples(order)
        return order, None
    except Exception as e:  # um pregão ruim não pode derrubar os outros
        return order, f"{type(e).__name__}: {e}"


def _generate_pl_worker(order):
    try:
        generate_day_pl(order)
        return order, None
    except Exception as e:
        return order, f"{type(e).__name__}: {e}"


def generate_all(orders, workers=1, worker=_generate_worker):
    """Gera (ou reaproveita do cache) as amostras de todos os pregões, em
    paralelo por pregão. Falha ao final, listando os pregões com erro."""
    failed = []
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            results = pool.imap_unordered(worker, orders)
            for i, (order, err) in enumerate(results, start=1):
                print(f"[{i}/{len(orders)}] pregão {order}" + (f"  ERRO: {err}" if err else ""), flush=True)
                if err:
                    failed.append(order)
    else:
        for i, order in enumerate(orders, start=1):
            _, err = worker(order)
            print(f"[{i}/{len(orders)}] pregão {order}" + (f"  ERRO: {err}" if err else ""), flush=True)
            if err:
                failed.append(order)
    if failed:
        raise SystemExit(f"Pregões com erro: {sorted(failed)}. Exclua-os com "
                         f'LSTM_EXCLUDE_ORDERS="{",".join(map(str, sorted(failed)))}" e rode de novo.')


# --------------------------------------------------------------------------
# EXECUÇÃO
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all",
                        choices=["all", "generate", "report", "train", "pending", "aggregate", "compare", "pl"],
                        help="all = fluxo serial original (gera, relatório e treina os 5 folds "
                             "num único processo); os demais são os estágios do Santos Dumont")
    parser.add_argument("--report-only", action="store_true",
                        help="atalho: gera as amostras + relatório de balanceamento e para")
    parser.add_argument("--fold", type=int, help="(train) fold 1..N_SPLITS")
    parser.add_argument("--side", choices=SIDES, help="(train) lado")
    parser.add_argument("--workers", type=int, default=1, help="(generate/all) processos em paralelo")
    parser.add_argument("--limit", type=int, default=0, help="(pending) no máximo N tarefas")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="default: 256 em --stage train (dataset grande, GPU), 32 em --stage all")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--oversampling", action="store_true",
                        help="usa oversampling da classe minoritária em vez de class_weight (padrão)")
    args = parser.parse_args()
    if args.report_only:
        args.stage = "report"

    dev_orders, test_orders = split_orders()
    use_oversampling = args.oversampling  # padrão: class_weight

    if args.stage == "pending":
        for fold, side in pending_tasks()[: args.limit or None]:
            print(fold, side)
    elif args.stage == "aggregate":
        aggregate_results()
    elif args.stage == "compare":
        compare_runs()
    elif args.stage == "pl":
        print(f"P/L por negociação: {len(PARAM_GRID)} combinações x {len(ALL_ORDERS)} pregões -> {PL_DIR}")
        generate_all(ALL_ORDERS, workers=args.workers, worker=_generate_pl_worker)
    elif args.stage == "train":
        if not args.fold or not args.side:
            parser.error("--stage train exige --fold e --side")
        train_task(args.fold, args.side, batch_size=args.batch_size or 256,
                   max_epochs=args.max_epochs, patience=args.patience,
                   use_oversampling=use_oversampling,
                   max_seconds=int(os.environ.get("LSTM_TRAIN_MAX_SECONDS", "0")))
    else:  # all | generate | report
        print("Desenvolvimento (TimeSeriesSplit):", dev_orders[0], "..", dev_orders[-1])
        print("Teste:", test_orders)
        print(f"Sweep: {len(PARAM_GRID)} combinações (sigma, Re, Ri) = {PARAM_GRID}")

        # [MUDANÇA 4] Gera (ou reaproveita do cache) as amostras de todos os
        # pregões para todas as combinações da grade.
        generate_all(ALL_ORDERS, workers=args.workers)
        if args.stage in ("all", "report"):
            RUN_ROOT.mkdir(parents=True, exist_ok=True)   # um relatório por experimento
            label_balance_report(dev_orders, csv_path=RUN_ROOT / "label_balance.csv")

        if args.stage == "all":
            results, models = run_time_series_cv(dev_orders, test_orders, n_splits=N_SPLITS,
                                                 batch_size=args.batch_size or 32,
                                                 use_oversampling=use_oversampling)
            for side, model in models.items():
                model.save(f"lstm_{side}.keras")
