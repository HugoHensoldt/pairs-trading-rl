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

Ajuste os parâmetros no bloco `if __name__ == "__main__":` ao final.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers

from config import data_path

# --------------------------------------------------------------------------
# 1) REPLICA O BACKTEST PARA UM DIA E RETORNA O DF PROCESSADO
# --------------------------------------------------------------------------

def process_day(order, Periodo=3000, Amostra=2400, sigma=2, Re=0.75, Ri=50):
    """Replica exatamente a lógica do backtest original para um único
    pregão (order) e devolve o df já com sinais, posicao e lucro por tick.
    """
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

    dj['miWa'] = dj['Wajusto'].rolling(window=Amostra).mean()
    dj['miWb'] = dj['Wbjusto'].rolling(window=Amostra).mean()
    dj['StDa'] = dj['Wajusto'].rolling(window=Amostra).std()
    dj['StDb'] = dj['Wbjusto'].rolling(window=Amostra).std()

    df = dj.copy()
    df = df[Amostra:len(df)].reset_index(drop=True)

    df['BOLUC'] = df['miWa'] + (sigma * df['StDa'] / 10)
    df['BOLDV'] = df['miWb'] - (sigma * df['StDb'] / 10)

    df['p'] = np.where((df['ask'] - df['BOLDV']) <= 0, df['ask'],
                        np.where((df['bid'] - df['BOLUC']) >= 0, -df['bid'], 0))

    Entrada = 0
    Saida = 0
    Poze = 0
    for index, row in df.iterrows():
        if Poze == 0:
            if row['p'] != 0:
                Poze = 1
                Entrada = row['ask']
                Saida = row['miWb']
                if row['p'] < 0:
                    Poze = -1
                    Entrada = row['bid']
                    Saida = row['miWa']
        else:
            if Poze == 1:
                if row['bid'] >= Entrada + (Re * abs(Saida - Entrada)):
                    Poze = 0
                elif row['bid'] <= Entrada - (Ri * abs(Saida - Entrada)):
                    Poze = 0
            else:
                if row['ask'] >= Entrada + (Ri * abs(Saida - Entrada)):
                    Poze = 0
                elif row['ask'] <= Entrada - (Re * abs(Saida - Entrada)):
                    Poze = 0

        df.loc[index, 'posicao'] = Poze
        df.loc[index, 'entrada'] = Entrada
        df.loc[index, 'SG'] = Re * abs(Saida - Entrada)
        df.loc[index, 'SL'] = Ri * abs(Saida - Entrada)

    df['lucro'] = np.where(
        (df['posicao'].shift(1) == 1) & (df['posicao'] == 0),
        np.where(df['bid'] <= (df['entrada'] - df['SL']), -df['SL'], df['SG']),
        np.where(
            (df['posicao'].shift(1) == -1) & (df['posicao'] == 0),
            np.where(df['ask'] >= (df['entrada'] + df['SL']), -df['SL'], df['SG']),
            0,
        ),
    )

    df['order'] = order
    return df


# --------------------------------------------------------------------------
# 2) EXTRAI AS JANELAS DE TREINAMENTO (X, y) PARA CADA NEGOCIAÇÃO
# --------------------------------------------------------------------------

# As 7 features conforme a Tabela 1 da tese.
# 'SL' já é a coluna 'DA' da tese: Ri*abs(Saida-Entrada), calculada pelo
# próprio backtest. Fica em 0 antes de qualquer entrada (Entrada/Saida
# iniciam em 0) e só assume valor quando a posição realmente abre.
FEATURE_COLS = ['bid', 'ask', 'Wbjusto', 'Wajusto', 'volume', 'bvolume', 'SL']
N_TICKS = 120  # janela de ticks anteriores à confirmação do sinal


def extract_trade_windows(df, n_ticks=N_TICKS):
    """Percorre o df de um dia e, a cada TRANSIÇÃO real de posição
    (flat -> comprado ou flat -> vendido), recorta a janela
    [i-n_ticks+1 : i] como features e usa o 'lucro' realizado dessa
    negociação como label.

    Importante: usar 'posicao' (transição 0->1 ou 0->-1) em vez de 'p',
    porque 'p' pode continuar diferente de zero em vários ticks seguidos
    mesmo com a posição já aberta -- usar 'p' geraria janelas duplicadas
    para o mesmo negócio.

    Retorna duas listas de (X, y): uma para compras, outra para vendas.
    """
    df = df.copy()

    buy_entry = (df['posicao'].shift(1) == 0) & (df['posicao'] == 1)
    sell_entry = (df['posicao'].shift(1) == 0) & (df['posicao'] == -1)

    buy_X, buy_y = [], []
    sell_X, sell_y = [], []

    def process_entries(entry_mask, is_buy):
        X_list, y_list = [], []
        for idx in df.index[entry_mask]:
            pos = df.index.get_loc(idx)
            if pos < n_ticks - 1:
                continue  # não há histórico suficiente ainda

            window = df.iloc[pos - n_ticks + 1: pos + 1]

            X = window[FEATURE_COLS].to_numpy(dtype=float)
            if np.isnan(X).any():
                continue  # descarta janelas com NaN (ex: início do rolling)

            # label: resultado realizado da negociação aberta neste tick
            trade_close_idx = df.index[(df.index > idx) & (df['posicao'] == 0)]
            if len(trade_close_idx) == 0:
                continue  # negociação não fechou até o fim do pregão, descarta
            close_idx = trade_close_idx[0]
            lucro = df.loc[close_idx, 'lucro']
            y = 1.0 if lucro > 0 else 0.0

            X_list.append(X)
            y_list.append(y)
        return X_list, y_list

    buy_X, buy_y = process_entries(buy_entry, is_buy=True)
    sell_X, sell_y = process_entries(sell_entry, is_buy=False)

    return (buy_X, buy_y), (sell_X, sell_y)


# --------------------------------------------------------------------------
# 3) MONTA OS DATASETS PARA VÁRIOS DIAS, COM SPLIT POR DIA
# --------------------------------------------------------------------------

def build_datasets(train_orders, val_orders, test_orders):
    """Retorna dicionários com X/y de treino, validação e teste,
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
# 4) NORMALIZAÇÃO — SCALER AJUSTADO SOMENTE NO TREINO
# --------------------------------------------------------------------------

def scale_datasets(splits):
    """Ajusta um StandardScaler por feature, usando SOMENTE os dados de
    treino, e aplica o mesmo scaler em treino/val/teste. Achata as
    janelas (n_amostras*n_ticks, n_features) para ajustar o scaler, depois
    devolve ao formato 3D exigido pelo Keras.
    """
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    if len(X_train) == 0:
        # Não há como ajustar o scaler sem dados de treino; devolve tudo
        # como está e deixa train_and_evaluate decidir pular este lado.
        return {'train': (X_train, y_train), 'val': (X_val, y_val),
                'test': (X_test, y_test), 'scaler': None}

    n_ticks, n_features = X_train.shape[1], X_train.shape[2]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def apply_scaler(X):
        if len(X) == 0:
            return X
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_features)).reshape(shape)

    return {
        'train': (apply_scaler(X_train), y_train),
        'val': (apply_scaler(X_val), y_val),
        'test': (apply_scaler(X_test), y_test),
        'scaler': scaler,
    }


# --------------------------------------------------------------------------
# 5) ARQUITETURA DA LSTM (conforme especificado na tese)
# --------------------------------------------------------------------------

def build_lstm(n_ticks, n_features):
    model = keras.Sequential([
        layers.Input(shape=(n_ticks, n_features)),
        layers.LSTM(50, return_sequences=True),
        layers.LSTM(50),
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
# EXECUÇÃO
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Dynamic split for train, validation, and test sets
    all_orders = list(range(1, 339)) # Example: adjust this range as needed
    total_days = len(all_orders)

    train_split = int(total_days * 0.70)
    val_split = int(total_days * 0.15)

    train_orders = all_orders[:train_split]
    val_orders = all_orders[train_split:train_split + val_split]
    test_orders = all_orders[train_split + val_split:]
    print("Treino:", train_orders)
    print("Validação:", val_orders)
    print("Teste:", test_orders)

    datasets = build_datasets(train_orders, val_orders, test_orders)

    for side in ['buy', 'sell']:
        print(f"\n########## Lado: {side} ##########")
        scaled = scale_datasets(datasets[side])
        model, history = train_and_evaluate(scaled, label=side)
        if model is not None:
            model.save(f"lstm_{side}.keras")
