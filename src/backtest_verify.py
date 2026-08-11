"""
Roda a MESMA lógica de backtest do seu script original, isolada em uma
função (process_day), e imprime as mesmas métricas de saída (Total,
contagem de negócios, contagem de negócios lucrativos, média da razão
Cask/Cbid) para comparação direta, linha a linha, com o output do seu
código original.

Se os números baterem, a reimplementação usada no pipeline de LSTM não
alterou o resultado do backtest.
"""

import pandas as pd
import numpy as np

from config import data_path


def process_day(order, Periodo=7200, Amostra=1400, sigma=50, Re=0.75, Ri=0.9):
    """Réplica fiel do backtest original, apenas organizada em função."""
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
    df = df[Amostra:len(df)]  # <- mantém o índice original (ms), sem reset

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

    df['lu'] = np.where(
        df['posicao'] >= 1, df['bid'] - df['entrada'],
        np.where(df['posicao'] <= -1, (-df['ask'] + df['entrada']), 0)
    )
    df['lucro'] = np.where(
        (df['posicao'].shift(1) == 1) & (df['posicao'] == 0),
        np.where(df['bid'] <= (df['entrada'] - df['SL']), -df['SL'], df['SG']),
        np.where(
            (df['posicao'].shift(1) == -1) & (df['posicao'] == 0),
            np.where(df['ask'] >= (df['entrada'] + df['SL']), -df['SL'], df['SG']),
            0,
        ),
    )

    return df


if __name__ == "__main__":
    for order in range(318, 338):
        df = process_day(order)

        Total = df['lucro'].sum() - (3 * (df.lucro[df.lucro != 0].count())) + df['lu'].iloc[-1]
        n_negocios = df.lucro[df.lucro != 0].count()
        n_lucrativos = df.lucro[df.lucro > 0].count()
        media_razao = ((df['Cask_3k'] + df['Cbid_3k']) / 2).mean()
        data_primeiro_tick = df['datahora'].iloc[0]

        # Mesma ordem de informação do print original, mas sem o corte
        # frágil de string por posição -- usa a data já formatada.
        print(f"{data_primeiro_tick}  {media_razao}  {Total}  {n_negocios}  {n_lucrativos}")
