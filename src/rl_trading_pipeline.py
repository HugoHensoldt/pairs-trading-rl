"""
Reinforcement Learning para negociação intradiária BOVA11 x WINM21 -- v4.

Estado SIMPLIFICADO (4 dimensões, só o tick atual, sem janela):
    [spread_compra, spread_venda, posição (-1/0/+1), P/L não realizado]
  spread_compra = ask_WIN - Wbjusto   (comprar WIN / vender BOVA no bid)
  spread_venda  = bid_WIN - Wajusto   (vender WIN / comprar BOVA no ask)
(ver build_state_features; as versões v1-v3 usavam uma janela de 120 ticks
x 5 features + posição one-hot + P/L = 604 dimensões).

Diferente da abordagem supervisionada (LSTM prevendo Lucro/Prejuízo por
negociação com stops fixos Re/Ri), aqui o agente controla a posição TICK A
TICK: decide entrar, segurar ou sair a cada instante, aprendendo sua
própria política de entrada/saída.

Requer:
    pip install gymnasium stable-baselines3

Estrutura:
  1) process_day()/process_day_cached() -> preço-justo (Wajusto/Wbjusto) de
                                um pregão (o cached grava .npz em disco)
  2) build_state_features() -> os 2 spreads do estado v4 por tick
     (build_feature_matrix() e RL_FEATURE_NAMES são da v1-v3 e continuam
     aqui porque rl_exit_only.py importa deles)
  3) fit_feature_scaler()   -> normaliza usando estatística SOMENTE do
                                treino (salvo em runs/<tag>/fold<N>/scaler.npz)
  4) ArbitrageTradingEnv    -> ambiente Gymnasium: 1 episódio = 1 pregão
  5) MultiDayEnv            -> alterna entre vários pregões pré-carregados
  6) generate_fixed_window_folds() -> folds de janela expansiva com
                                tamanhos de treino fixos (100/150/200)
  7) train_chunk()          -> treina PPO até TOTAL_TIMESTEPS ou até o
                                deadline do job; checkpoint/resume entre
                                jobs Slurm; state.json marca `done`
  8) run_eval_job()         -> avaliação de val/teste (melhor-de-validação
                                e último), baselines, sensibilidade a custo
                                e curva de checkpoints, tudo em CSV
Saídas em runs/<RUN_TAG>/fold<N>/ (ver slurm/submit_all_folds.sh).
"""

import math
import os

# Evita oversubscription de threads BLAS/OMP: em cluster (Santos Dumont),
# n_train_envs sobe para ~48 processos paralelos (um por núcleo físico do
# nó); se cada um também abrir suas próprias threads BLAS por conta
# própria, o total de threads estoura muito além dos núcleos disponíveis
# e o throughput CAI em vez de subir. Precisa ser setado ANTES de
# importar numpy/torch (as libs de BLAS leem essas env vars só na
# inicialização). setdefault() para não sobrescrever se o .sbatch já
# tiver setado explicitamente.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time

# Marca o início real do processo Python -- usado pra calcular quanto
# tempo de parede (não só tempo de model.learn()) já passou desde que o
# script começou. Isso importa porque, sob Slurm, o overhead de carregar
# dados e subir os processos do SubprocVecEnv (~180s observados num
# smoke test com 48 processos) acontece ANTES do model.learn() comecar
# --  medir só a partir do learn() deixa esse overhead de fora da conta
# e comeu a margem de segurança inteira num treino real (ver
# TimeLimitCallback abaixo).
SCRIPT_START = time.time()

from functools import partial

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.model_selection import TimeSeriesSplit

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import configure as configure_logger

from config import data_path, POINT_VALUE_BRL

# mesma lógica do OMP_NUM_THREADS acima, mas para o processo PRINCIPAL
# (que roda o forward/backward da rede via torch, concorrendo por CPU com
# os processos do SubprocVecEnv) -- por padrão o torch abre 1 thread por
# núcleo físico, o que sozinho já tomaria o nó inteiro.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))


def _detect_n_train_envs():
    """Detecta quantos processos de ambiente paralelos usar.

    Em execução local, usa os.cpu_count(). Sob Slurm, os.cpu_count() pode
    reportar o total de CPUs do nó físico em vez do que foi de fato
    alocado ao job (depende de cgroups); então preferimos as env vars que
    o próprio Slurm exporta com a alocação real.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            # SLURM_JOB_CPUS_PER_NODE pode vir como "48" ou "48(x2)" em
            # alocações multi-nó; pegamos só o primeiro número.
            digits = val.split("(")[0].split(",")[0].strip()
            if digits.isdigit():
                return int(digits)
    return os.cpu_count() or 1


class TimeLimitCallback(BaseCallback):
    """Para o treino com margem de segurança antes do MaxTime do job Slurm.

    O Slurm mata o job na marca exata do --time, sem aviso -- se o
    model.save() ainda não tiver terminado, o checkpoint fica corrompido
    ou incompleto. Este callback interrompe o `model.learn()` mais cedo.

    O corte é por DEADLINE absoluto (SCRIPT_START + max_seconds), não por
    tempo desde o início do learn() -- um smoke test real mostrou ~180s
    de overhead (carregar dados, subir os 48 processos do
    SubprocVecEnv) ANTES do learn() começar; contar só a partir do
    learn() deixava esse overhead de fora da margem e comeu o buffer de
    segurança inteiro.
    """

    def __init__(self, deadline, verbose=0):
        super().__init__(verbose)
        self.deadline = deadline

    def _on_step(self):
        now = time.time()
        if now > self.deadline:
            if self.verbose:
                print(f"[TimeLimitCallback] deadline atingido "
                      f"({now - self.deadline:.0f}s além do limite) -- "
                      f"parando treino com margem de segurança.")
            return False
        return True


# --------------------------------------------------------------------------
# 1) PREÇO-JUSTO (sem bandas, sem máquina de estados de posição)
# --------------------------------------------------------------------------

def process_day(order, Periodo=3000):
    """Gera preço-justo (Wajusto/Wbjusto) para um pregão. Não calcula mais
    bandas de Bollinger (miWa/miWb/StDa/StDb/BOLUC/BOLDV) -- a única
    feature usada agora é o spread de mispricing, derivado direto de
    Wajusto/Wbjusto.
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

    df = dj.dropna().reset_index(drop=True)
    return df


def _day_cache_dir():
    d = os.environ.get("DAY_CACHE_DIR") or os.path.join(os.environ.get("RUNS_DIR", "runs"), "_day_cache")
    return d


_DAY_MEMO = {}


def process_day_cached(order, Periodo=3000):
    """process_day() com cache em .npz das colunas usadas na v4 (bid, ask,
    Wbjusto, Wajusto, datahora). Ler/parsear os JSONs UTF-16 e calcular as
    médias móveis é o que dominava o startup de cada chunk (cada um dos ~48
    processos do SubprocVecEnv carrega TODOS os dias de treino); com o
    cache o startup vira leitura de arrays. O arquivo é gravado de forma
    atômica (tmp + replace) porque vários processos podem preencher o mesmo
    dia ao mesmo tempo."""
    cols = ('bid', 'ask', 'Wbjusto', 'Wajusto')
    # memo em memória: com DummyVecEnv, dezenas de envs no MESMO processo
    # pedem os mesmos dias -- carrega cada um uma vez só (o df não é
    # modificado por ninguém, só lido)
    if (order, Periodo) in _DAY_MEMO:
        return _DAY_MEMO[(order, Periodo)]
    path = os.path.join(_day_cache_dir(), f"day{order}_p{Periodo}.npz")
    if os.path.exists(path):
        try:
            z = np.load(path)
            out = {c: z[c] for c in cols}
            out['datahora'] = z['datahora']
            _DAY_MEMO[(order, Periodo)] = pd.DataFrame(out)
            return _DAY_MEMO[(order, Periodo)]
        except Exception:
            pass  # cache corrompido/parcial: reprocessa abaixo
    df = process_day(order, Periodo)
    slim = pd.DataFrame({c: df[c].to_numpy(dtype=float) for c in cols})
    slim['datahora'] = pd.to_datetime(df['datahora']).to_numpy()
    try:
        os.makedirs(_day_cache_dir(), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **{c: slim[c].to_numpy() for c in slim.columns})
        os.replace(tmp, path)
    except OSError:
        pass  # cache é só otimização
    _DAY_MEMO[(order, Periodo)] = slim
    return slim


# --------------------------------------------------------------------------
# 2) FEATURES: MISPRICING + SPREAD BID-ASK + QUEM MOVEU + MOMENTUM
# --------------------------------------------------------------------------

RL_FEATURE_NAMES = [
    'spread_mispricing', 'bid_ask_spread', 'who_moved_bova_vs_win',
    'momentum_short', 'momentum_long',
]

# janelas curtas usadas nas features de atribuição/momentum (bem menores
# que N_TICKS=120, que é a janela de OBSERVAÇÃO -- essas são só o
# intervalo usado para medir a variação recente de cada série)
WHO_MOVED_LAG = 10     # ticks usados para atribuir o movimento a BOVA/WIN
MOMENTUM_SHORT_LAG = 30
MOMENTUM_LONG_LAG = 120


def build_feature_matrix(df):
    """Constrói a matriz de features (n_ticks, 5) usada como observação.

    Feature 1 -- spread de mispricing: ponto médio do preço atual do WIN
    menos o ponto médio do preço justo implícito pelo BOVA11.
        spread > 0  -> WIN "caro" em relação ao justo (viés de venda)
        spread < 0  -> WIN "barato" em relação ao justo (viés de compra)

    Feature 2 -- spread bid-ask do WIN (ask - bid): custo de execução
    (cruzar o book), NÃO tem relação com o BOVA11. Sem isso o agente não
    tem como perceber "operar agora está mais caro que o normal".

    Feature 3 -- quem moveu (BOVA vs WIN): diferença entre a variação
    recente do preço-justo implícito pelo BOVA (proxy do movimento do
    BOVA, já na escala do WIN) e a variação recente do mid-price do
    próprio WIN, nos últimos WHO_MOVED_LAG ticks.
        > 0 -> o BOVA moveu mais que o WIN recentemente (WIN "atrasado",
               mais chance de correr atrás no próximo tick)
        < 0 -> o WIN moveu mais que o BOVA recentemente (mais chance de
               ser ruído/sobre-reação que se autocorrige)
    Isso NÃO existia antes: spread_mispricing só mostra o NÍVEL do
    desalinhamento, não QUEM o causou.

    Features 4/5 -- momentum do WIN em duas janelas (curta/longa):
    variação do mid-price do WIN nos últimos MOMENTUM_SHORT_LAG /
    MOMENTUM_LONG_LAG ticks. Sem isso o agente não tinha nenhuma
    informação de tendência do mercado -- spread_mispricing é uma
    DIFERENÇA entre WIN e BOVA, que pode ficar estável mesmo com os dois
    subindo/descendo juntos.

    Todas as janelas curtas (features 3-5) usam .diff(...).fillna(0.0):
    os primeiros ticks do dia (sem histórico suficiente) viram 0 --
    "sem sinal de tendência ainda" -- em vez de NaN, que quebraria a
    normalização e a janela de observação inicial do ambiente.

    O agente recebe uma JANELA de N_TICKS valores de cada feature (não só
    o valor instantâneo). As 5 colunas são normalizadas separadamente
    (mean/std por coluna, só com dados de treino -- ver
    fit_feature_scaler/apply_scaler, já genéricos para N colunas).
    """
    price_mid = (df['ask'] + df['bid']) / 2
    fair_mid = (df['Wajusto'] + df['Wbjusto']) / 2

    spread_mispricing = (price_mid - fair_mid).to_numpy(dtype=float)
    bid_ask_spread = (df['ask'] - df['bid']).to_numpy(dtype=float)

    delta_fair = fair_mid.diff(WHO_MOVED_LAG).fillna(0.0)
    delta_win = price_mid.diff(WHO_MOVED_LAG).fillna(0.0)
    who_moved_bova_vs_win = (delta_fair - delta_win).to_numpy(dtype=float)

    momentum_short = price_mid.diff(MOMENTUM_SHORT_LAG).fillna(0.0).to_numpy(dtype=float)
    momentum_long = price_mid.diff(MOMENTUM_LONG_LAG).fillna(0.0).to_numpy(dtype=float)

    return np.stack([
        spread_mispricing, bid_ask_spread, who_moved_bova_vs_win,
        momentum_short, momentum_long,
    ], axis=1)


STATE_FEATURE_NAMES = ['spread_compra', 'spread_venda']


def build_state_features(df):
    """Features de mercado do estado v4 (n, 2) -- só o tick atual, sem janela.

    Colunas (ver STATE_FEATURE_NAMES):
      spread_compra = ask_WIN - Wbjusto
          quanto se PAGA a mais para comprar o WIN em relação ao que se
          recebe vendendo o BOVA11 (comprar WIN no ask / vender BOVA no bid).
      spread_venda  = bid_WIN - Wajusto
          quanto se RECEBE a mais vendendo o WIN em relação ao que se paga
          comprando o BOVA11 (vender WIN no bid / comprar BOVA no ask).

    Diferente do mispricing de mid (build_feature_matrix), cada spread já
    embute o custo de cruzar o book do WIN e do preço justo -- o agente
    enxerga o desalinhamento LÍQUIDO de cada lado. Posição atual e P/L não
    realizado são anexados por ArbitrageTradingEnv._get_obs (não passam
    pelo scaler).
    """
    spread_compra = (df['ask'] - df['Wbjusto']).to_numpy(dtype=float)
    spread_venda = (df['bid'] - df['Wajusto']).to_numpy(dtype=float)
    return np.stack([spread_compra, spread_venda], axis=1)


def fit_feature_scaler(feature_matrices):
    """Ajusta média/desvio-padrão usando SOMENTE as matrizes de treino
    (lista de arrays, uma por dia). Retorna (mean, std) para normalizar
    qualquer outra matriz depois.
    """
    stacked = np.concatenate(feature_matrices, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std == 0] = 1.0  # evita divisão por zero em colunas constantes
    return mean, std


def apply_scaler(feature_matrix, mean, std):
    return (feature_matrix - mean) / std


# --------------------------------------------------------------------------
# 3) AMBIENTE GYMNASIUM -- 1 EPISÓDIO = 1 PREGÃO
# --------------------------------------------------------------------------

class ArbitrageTradingEnv(gym.Env):
    """Ambiente de RL para um único pregão.

    Ação (Discrete(3)): posição-ALVO a assumir neste tick.
        0 = flat
        1 = comprado
        2 = vendido  (internamente convertido para -1)

    Observação (Box, 4 dimensões, só o tick atual -- sem janela):
    [spread_compra, spread_venda (normalizados, ver build_state_features),
    posição atual (-1/0/+1), PnL não-realizado normalizado pelo spread
    bid-ask do tick].

    Recompensa: variação de marcação a mercado (mid-price) da posição já
    aberta desde o tick anterior, MENOS custo de execução (metade do
    spread) sempre que uma perna abre/fecha, MENOS taxa fixa cobrada
    INTEIRAMENTE na abertura (2.5 PONTOS -- equivalente a R$0,50 de custo
    real de corretora por negociação completa, ao câmbio de R$0,20/ponto
    do WIN; ver POINT_VALUE_BRL em config.py. `bid`/`ask`/reward aqui
    ficam em pontos brutos o tempo todo -- não há conversão para R$
    dentro do ambiente, só na exibição de resultados). A taxa fixa é
    concentrada na abertura -- e não dividida ou cobrada no fechamento --
    de propósito: isso evita que a ação de fechar concentre custo extra
    além do spread, o que poderia reforçar relutância do agente em
    realizar posições perdedoras (efeito parecido com disposition
    effect).
    """

    metadata = {"render_modes": []}

    def __init__(self, df, feature_matrix, transaction_fee=2.5,
                 reward_scale=1.0, max_loss_per_position=None,
                 record_equity=False, track_details=False):
        super().__init__()
        assert len(df) == len(feature_matrix)
        # só o tamanho do pregão é guardado (não o DataFrame): cada um dos
        # ~48 processos do SubprocVecEnv carrega TODOS os dias de treino, e
        # com 100-200 dias o df completo (~20 colunas) custaria GBs por
        # processo sem ser usado depois de extraídos bid/ask.
        self.n_day_ticks = len(df)
        self.record_equity = record_equity
        # track_details (só na avaliação): decompõe o P/L em convergência do
        # spread vs componente direcional (preço justo) e guarda os
        # detalhes de cada negócio -- fica desligado no treino pra não
        # custar nada por step.
        self.track_details = track_details and 'Wajusto' in df.columns
        if self.track_details:
            self.fair = ((df['Wajusto'] + df['Wbjusto']) / 2).to_numpy(dtype=float)
            self.mid = (df['bid'] + df['ask']).to_numpy(dtype=float) / 2
        # bid/ask pré-extraídos como arrays numpy: evita .iloc (indexação
        # pandas) tick a tick dentro de step()/_get_obs(), que é bem mais
        # lento que indexação numpy direta.
        self.bid = df['bid'].to_numpy(dtype=float)
        self.ask = df['ask'].to_numpy(dtype=float)
        self.feature_matrix = feature_matrix.astype(np.float32)
        self.transaction_fee = transaction_fee
        self.reward_scale = reward_scale
        # stop-loss por posição: regra FIXA (não aprendida), desligada por
        # padrão (None). Se setado, força o fechamento da posição aberta
        # assim que a perda não-realizada ultrapassar esse limiar (em
        # reais) -- ver aplicação em step(). Não é usado na configuração
        # de treino padrão -- ver train_ppo()/build_day_envs().
        self.max_loss_per_position = max_loss_per_position
        self.n_features = feature_matrix.shape[1]

        obs_dim = self.n_features + 1 + 1      # features + posição + P/L
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _reset_state(self):
        self.t = 0                         # sem janela: todo tick já tem estado completo
        self.position = 0                  # -1, 0, 1
        self.entry_price = 0.0             # preço de execução da posição aberta
        self.n_trades_closed = 0
        self.n_trades_won = 0
        self.trade_pnls = []               # P&L realizado de CADA negócio fechado
        self.n_stop_loss_triggers = 0      # quantas vezes o stop-loss forçou fechamento
        # --- contadores para o relatório (avaliação/diagnóstico) ---------
        self.ticks_long = 0
        self.ticks_short = 0
        self.gross_mtm = 0.0               # P/L de marcação a mercado, antes de custos
        self.total_reward = 0.0            # reward acumulado (líquido de custos)
        self.trade_durations = []          # ticks entre abrir e fechar cada negócio
        self.trade_close_ticks = []        # índice do tick de fechamento de cada negócio
        self._open_tick = 0
        self.equity = [] if self.record_equity else None
        # decomposição: gross_mtm == mtm_spread + mtm_fair (só com track_details)
        self.mtm_spread = 0.0              # posição x variação do mispricing (mid WIN - justo)
        self.mtm_fair = 0.0                # posição x variação do preço justo (exposição direcional)
        self.trade_details = []            # um dict por negócio fechado

    def _get_obs(self):
        bid_t = self.bid[self.t]
        ask_t = self.ask[self.t]
        if self.position == 1:
            unrealized = bid_t - self.entry_price
        elif self.position == -1:
            unrealized = self.entry_price - ask_t
        else:
            unrealized = 0.0
        # normalização pelo spread bid-ask do tick atual (não é média)
        # para manter escala razoável
        unrealized_norm = unrealized / max(ask_t - bid_t, 1e-6)

        obs = np.empty(self.n_features + 2, dtype=np.float32)
        obs[:self.n_features] = self.feature_matrix[self.t]
        obs[self.n_features] = self.position
        obs[self.n_features + 1] = unrealized_norm
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _close_leg(self, bid_t, ask_t):
        """Fecha a posição aberta no tick atual, retornando (reward_da_saida,
        lucro_realizado). Aplica só o meio-spread de execução (a taxa fixa
        já foi cobrada na abertura -- ver _open_leg), e registra o P&L
        líquido (já descontado do spread e da taxa de abertura) em
        self.trade_pnls para diagnóstico posterior.

        `realized` (exec_price - entry_price) JÁ embute os dois
        meios-spreads (entrada E saída) implicitamente, porque
        entry_price/exec_price usam ask/bid em vez do mid-price -- por
        isso net_realized NÃO subtrai spread_cost de novo aqui (bug
        corrigido: a versão anterior subtraía o meio-spread de saída
        duas vezes, deixando o diagnóstico de P&L por negócio mais
        pessimista que a realidade; não afetava `reward`/lucro_total,
        que usam contabilidade tick-a-tick separada e sempre estiveram
        corretos).
        """
        if self.position == 1:
            exec_price = bid_t             # vende no bid para fechar compra
            realized = exec_price - self.entry_price
            spread_cost = (ask_t - bid_t) / 2
        elif self.position == -1:
            exec_price = ask_t             # compra no ask para fechar venda
            realized = self.entry_price - exec_price
            spread_cost = (ask_t - bid_t) / 2
        else:
            return 0.0, 0.0

        reward = -spread_cost
        net_realized = realized - self.transaction_fee
        self.n_trades_closed += 1
        self.trade_pnls.append(net_realized)
        self.trade_durations.append(self.t - self._open_tick)
        self.trade_close_ticks.append(self.t)
        if self.track_details:
            o, c = self._open_tick, self.t
            self.trade_details.append((
                self.position, o, c, net_realized,
                self.feature_matrix[o, 0], self.feature_matrix[o, 1],
                self.feature_matrix[c, 0], self.feature_matrix[c, 1],
                self.mid[o] - self.fair[o], self.mid[c] - self.fair[c],
                self.fair[o], self.fair[c]))
        if net_realized > 0:
            self.n_trades_won += 1
        return reward, realized

    def _open_leg(self, target_position, bid_t, ask_t):
        """Abre uma nova posição no tick atual. Retorna a recompensa
        (custo de execução de meio-spread + taxa fixa de corretora, cobrada
        inteira aqui na abertura -- ver docstring da classe para o motivo)."""
        if target_position == 1:
            self.entry_price = ask_t       # compra no ask
            spread_cost = (ask_t - bid_t) / 2
        elif target_position == -1:
            self.entry_price = bid_t       # vende no bid
            spread_cost = (ask_t - bid_t) / 2
        else:
            self.entry_price = 0.0
            self.position = target_position
            return 0.0  # ir para flat não é abrir negócio: sem custo
        self.position = target_position
        self._open_tick = self.t
        return -spread_cost - self.transaction_fee

    def step(self, action):
        target_position = {0: 0, 1: 1, 2: -1}[int(action)]

        bid_t = self.bid[self.t]
        ask_t = self.ask[self.t]
        mid_t = (bid_t + ask_t) / 2

        reward = 0.0

        # 1) marcação a mercado da posição já aberta ANTES desta ação,
        #    desde o mid-price do tick anterior até agora
        if self.t > 0:
            bid_prev = self.bid[self.t - 1]
            ask_prev = self.ask[self.t - 1]
            mid_prev = (bid_prev + ask_prev) / 2
            mtm = self.position * (mid_t - mid_prev)
            reward += mtm
            self.gross_mtm += mtm
            if self.track_details:
                fair_t, fair_p = self.fair[self.t], self.fair[self.t - 1]
                self.mtm_fair += self.position * (fair_t - fair_p)
                self.mtm_spread += self.position * ((mid_t - fair_t) - (mid_prev - fair_p))

        # 1.5) stop-loss por posição (regra FIXA, não aprendida -- ver
        #    __init__): se configurado e a perda não-realizada da posição
        #    aberta ultrapassar o limiar, força o fechamento agora,
        #    sobrescrevendo a ação escolhida pelo agente nesta tick.
        if self.max_loss_per_position is not None and self.position != 0:
            if self.position == 1:
                unrealized = bid_t - self.entry_price
            else:  # self.position == -1
                unrealized = self.entry_price - ask_t
            if unrealized < -self.max_loss_per_position:
                target_position = 0
                self.n_stop_loss_triggers += 1

        # 2) troca de posição: fecha a perna atual (se houver) e abre a nova
        #    (se houver) -- cobre flat->long, flat->short, long->short, etc.
        if target_position != self.position:
            close_reward, _ = self._close_leg(bid_t, ask_t)
            reward += close_reward
            reward += self._open_leg(target_position, bid_t, ask_t)

        if self.position == 1:
            self.ticks_long += 1
        elif self.position == -1:
            self.ticks_short += 1

        self.t += 1
        terminated = self.t >= self.n_day_ticks - 1
        truncated = False

        # 3) fim do pregão: força o fechamento de qualquer posição aberta
        if terminated and self.position != 0:
            bid_last = self.bid[self.t]
            ask_last = self.ask[self.t]
            close_reward, _ = self._close_leg(bid_last, ask_last)
            reward += close_reward
            self.position = 0

        self.total_reward += reward
        if self.equity is not None:
            self.equity.append(self.total_reward)

        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32)

        info = {
            "n_trades_closed": self.n_trades_closed,
            "n_trades_won": self.n_trades_won,
            "n_stop_loss_triggers": self.n_stop_loss_triggers,
        }
        return obs, reward * self.reward_scale, terminated, truncated, info


# --------------------------------------------------------------------------
# 4) AMBIENTE MULTI-DIA -- alterna entre pregões pré-carregados a cada reset
# --------------------------------------------------------------------------

class MultiDayEnv(gym.Env):
    """Envolve vários ArbitrageTradingEnv (um por pregão) e sorteia um
    pregão diferente a cada reset(), para o agente treinar em vários dias
    sem recarregar/reprocessar os JSONs a cada episódio.
    """

    metadata = {"render_modes": []}

    def __init__(self, day_envs, seed=None):
        super().__init__()
        assert len(day_envs) > 0
        self.day_envs = day_envs
        self.observation_space = day_envs[0].observation_space
        self.action_space = day_envs[0].action_space
        self.rng = np.random.default_rng(seed)
        self.current_env = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_env = self.day_envs[self.rng.integers(len(self.day_envs))]
        return self.current_env.reset()

    def step(self, action):
        return self.current_env.step(action)


def build_day_envs(orders, mean, std, transaction_fee=2.5,
                    max_loss_per_position=None):
    """Carrega e processa cada pregão em `orders` UMA VEZ, retornando uma
    lista de ArbitrageTradingEnv prontos para uso -- evita reler os JSONs
    a cada episódio de treino.

    `max_loss_per_position=None` (default) mantém o stop-loss desligado
    -- ver ArbitrageTradingEnv.
    """
    envs = []
    for order in orders:
        df = process_day_cached(order)
        feat = build_state_features(df)
        feat = apply_scaler(feat, mean, std)
        envs.append(ArbitrageTradingEnv(df, feat,
                                         transaction_fee=transaction_fee,
                                         max_loss_per_position=max_loss_per_position))
    return envs


def _make_multiday_env(orders, mean, std, transaction_fee=2.5,
                        max_loss_per_position=None, seed=None):
    """Função construtora usada como env_fn do SubprocVecEnv.

    Recebe só identificadores leves (lista de números de pregão + mean/std
    da normalização) e faz o trabalho pesado (ler JSONs, processar,
    montar features) DENTRO do processo em que é chamada. Isso evita
    serializar/enviar pelo pipe DataFrames e arrays já processados do
    processo pai para cada processo-filho -- só os `orders` (ints) e
    mean/std (arrays pequenos) atravessam o pickle.
    """
    day_envs = build_day_envs(orders, mean, std,
                               transaction_fee=transaction_fee,
                               max_loss_per_position=max_loss_per_position)
    return Monitor(MultiDayEnv(day_envs, seed=seed))


# --------------------------------------------------------------------------
# 4.5) SPLIT WALK-FORWARD (TimeSeriesSplit) -- treino/validação/teste por fold
# --------------------------------------------------------------------------

def generate_walk_forward_folds(
    all_orders,
    n_folds=3,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    target_n_passadas=20.0,
    throughput_steps_per_sec=4209.0,
    chunk_seconds=1140.0,
    avg_ticks_per_day=None,
    verbose=True,
):
    """Gera folds walk-forward (janela expansiva) sobre `all_orders`
    (pregões em ordem cronológica), usando
    sklearn.model_selection.TimeSeriesSplit como base: cada fold usa um
    bloco CRESCENTE de dias passados como treino e um bloco contíguo
    seguinte (no futuro) como validação+teste -- sem embaralhar, sem
    sobreposição.

    Dentro do bloco de validação+teste de cada fold, a divisão é
    cronológica: a parte de validação vem antes, a de teste depois
    (ambas ainda no futuro em relação ao treino daquele fold).

    Como o treino cresce a cada fold, a proporção `train_frac`/
    `val_frac`/`test_frac` só é atingida (aproximadamente) no ÚLTIMO
    fold, que é o maior -- folds iniciais têm proporcionalmente menos
    dias de treino. Isso é esperado em walk-forward CV com janela
    expansiva.

    `target_n_passadas` é o número de vezes, em média, que cada dia de
    treino deve ser visitado pelo MultiDayEnv -- e é o MESMO para todos
    os folds (ao contrário de uma versão anterior que fixava um
    orçamento de TEMPO total igual pra todos os folds, o que fazia
    folds pequenos serem revisitados dezenas de vezes a mais que o
    fold grande e overfitarem: um teste real no Santos Dumont mostrou o
    fold 1, com 7 dias de treino, chegando a ~128 passadas/dia contra
    ~20 do fold 3 -- justamente o fold 1 foi o único com teste negativo
    apesar de validação ótima, sinal clássico de overfitting). Com
    `target_n_passadas` fixo, `total_timesteps` de cada fold cresce
    proporcionalmente ao seu número de dias de treino, em vez de ser
    igual para todos.

    `throughput_steps_per_sec` (fps real de treino PPO medido no
    SubprocVecEnv -- RECALIBRAR ao trocar de máquina, ex.: o node do
    Santos Dumont tem fps diferente do PC doméstico onde o default
    abaixo foi medido) e `chunk_seconds` (tempo real de treino por job
    Slurm encadeado -- ver TRAIN_MAX_SECONDS em submit_fold.sbatch para
    o chunk intermediário) convertem `total_timesteps` em
    `n_chunks_needed`: quantos jobs de ~20min encadeados (checkpoint+
    resume) são necessários pra completar aquele fold.

    Retorna uma lista de dicts, um por fold, com "fold", "train_orders",
    "val_orders", "test_orders", "total_timesteps" (calibrado para dar
    `target_n_passadas` passadas médias naquele fold), "n_passadas_medias"
    (== target_n_passadas, constante entre folds por construção) e
    "n_chunks_needed" (quantos jobs Slurm de `chunk_seconds` encadear).
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9
    assert n_folds >= 2, "TimeSeriesSplit exige n_folds >= 2"

    if avg_ticks_per_day is None:
        sample_orders = all_orders[:2]
        avg_ticks_per_day = float(np.mean([len(process_day(o)) for o in sample_orders]))

    # Sem orçamento de tempo total a respeitar (isso só fazia sentido pro
    # PC doméstico): usa todos os pregões disponíveis, calibrando só o
    # tamanho do bloco val+teste pela proporção desejada. Para o ÚLTIMO
    # fold (o maior), TimeSeriesSplit dá train ≈ N - test_size, então
    # test_size = N × (val_frac+test_frac) já deixa o último fold próximo
    # de train_frac/val_frac/test_frac sem precisar calcular um
    # "max_train_days" auxiliar.
    n_samples_total = len(all_orders)
    test_size = max(1, int(round(n_samples_total * (val_frac + test_frac))))
    min_required = n_folds * test_size + 1
    if n_samples_total < min_required:
        n_samples_total = min(min_required, len(all_orders))

    orders_used = all_orders[:n_samples_total]

    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=test_size)

    folds = []
    for i, (train_idx, val_test_idx) in enumerate(tscv.split(orders_used), start=1):
        train_orders = [orders_used[j] for j in train_idx]

        n_val = max(1, int(round(len(val_test_idx) * val_frac / (val_frac + test_frac))))
        n_val = min(n_val, len(val_test_idx) - 1)  # garante ao menos 1 dia de teste
        val_idx = val_test_idx[:n_val]
        test_idx = val_test_idx[n_val:]

        val_orders = [orders_used[j] for j in val_idx]
        test_orders = [orders_used[j] for j in test_idx]

        # total_timesteps do fold = target_n_passadas × dias_treino ×
        # ticks_por_dia -- CRESCE com o tamanho do fold (ao contrário da
        # versão anterior, que fixava total_timesteps e deixava
        # n_passadas_medias variar). n_chunks_needed traduz isso em
        # quantos jobs Slurm de chunk_seconds encadear pra esse fold.
        dias_treino = len(train_orders)
        total_timesteps_fold = int(round(target_n_passadas * dias_treino * avg_ticks_per_day))
        timesteps_per_chunk = throughput_steps_per_sec * chunk_seconds
        n_chunks_needed = max(1, math.ceil(total_timesteps_fold / timesteps_per_chunk))

        folds.append({
            "fold": i,
            "train_orders": train_orders,
            "val_orders": val_orders,
            "test_orders": test_orders,
            "total_timesteps": total_timesteps_fold,
            "n_passadas_medias": target_n_passadas,
            "n_chunks_needed": n_chunks_needed,
        })

    if verbose:
        print(f"=== Folds walk-forward (TimeSeriesSplit, n_folds={n_folds}) ===")
        print(f"Throughput assumido : {throughput_steps_per_sec:,.0f} steps/s  "
              f"(ajuste para a máquina onde for treinar)")
        print(f"Passadas-alvo/dia   : {target_n_passadas:.1f} (igual para todos os folds)")
        print(f"Chunk (job Slurm)   : {chunk_seconds:.0f}s de treino real por job encadeado")
        print(f"Ticks médios/pregão : {avg_ticks_per_day:,.0f}")
        print(f"Pregões disponíveis : {len(all_orders)}  |  pregões usados: {len(orders_used)}")
        for f in folds:
            tr, va, te = f["train_orders"], f["val_orders"], f["test_orders"]
            assert tr[-1] < va[0] <= va[-1] < te[0] <= te[-1], \
                "sobreposição/ordem cronológica violada entre treino/validação/teste"
            print(f"  Fold {f['fold']}: treino=[{tr[0]:>3}..{tr[-1]:>3}] ({len(tr):>3} dias)  "
                  f"validação=[{va[0]:>3}..{va[-1]:>3}] ({len(va):>3} dias)  "
                  f"teste=[{te[0]:>3}..{te[-1]:>3}] ({len(te):>3} dias)  "
                  f"total_timesteps={f['total_timesteps']:,} "
                  f"(~{f['n_passadas_medias']:.1f} passadas/dia de treino, "
                  f"{f['n_chunks_needed']} chunk(s) de {chunk_seconds:.0f}s)")

    return folds



# --------------------------------------------------------------------------
# 4.6) FOLDS DE JANELA EXPANSIVA COM TAMANHOS FIXOS DE TREINO (v4)
# --------------------------------------------------------------------------

def generate_fixed_window_folds(
    all_orders,
    train_sizes=(100, 150, 200),
    n_val=15,
    n_test=15,
    total_timesteps=30_000_000,
    throughput_steps_per_sec=5500.0,
    chunk_seconds=1050.0,
    avg_ticks_per_day=None,
    verbose=True,
):
    """Folds walk-forward de janela expansiva com tamanhos de treino FIXOS.

    Fold i treina nos primeiros `train_sizes[i]` pregões e valida/testa nos
    `n_val` + `n_test` pregões seguintes (cronológicos, no futuro do
    treino daquele fold). Diferente de generate_walk_forward_folds (que usa
    TimeSeriesSplit e não permite escolher os tamanhos), aqui o usuário
    escolhe, ex.: 100/150/200 dias de treino.

    O orçamento é em TIMESTEPS TOTAIS por fold (`total_timesteps`), não em
    "passadas por dia": o que importa pro PPO é o número de atualizações
    (o sweep mostrou política degenerada abaixo de ~10 atualizações),
    independente de quantos dias há no treino. `n_passadas_medias` é só
    informativo (total / (dias_treino × ticks_por_dia)).

    `n_chunks_needed` é uma ESTIMATIVA (throughput × chunk_seconds); o
    controle real é o `state.json` de cada fold, que marca `done` quando
    `total_timesteps` é atingido -- ver slurm/submit_all_folds.sh.
    """
    needed = max(train_sizes) + n_val + n_test
    assert len(all_orders) >= needed, (
        f"são necessários {needed} pregões (maior treino {max(train_sizes)} + "
        f"{n_val} val + {n_test} teste), só há {len(all_orders)}")

    if avg_ticks_per_day is None:
        env_avg = os.environ.get("AVG_TICKS_PER_DAY")
        if env_avg:
            avg_ticks_per_day = float(env_avg)
        else:
            avg_ticks_per_day = float(np.mean([len(process_day_cached(o)) for o in all_orders[:2]]))

    timesteps_per_chunk = throughput_steps_per_sec * chunk_seconds
    n_chunks_needed = max(1, math.ceil(total_timesteps / timesteps_per_chunk))

    folds = []
    for i, size in enumerate(train_sizes, start=1):
        train_orders = list(all_orders[:size])
        val_orders = list(all_orders[size:size + n_val])
        test_orders = list(all_orders[size + n_val:size + n_val + n_test])
        assert train_orders[-1] < val_orders[0] <= val_orders[-1] < test_orders[0], \
            "ordem cronológica violada entre treino/validação/teste"
        folds.append({
            "fold": i,
            "train_orders": train_orders,
            "val_orders": val_orders,
            "test_orders": test_orders,
            "total_timesteps": int(total_timesteps),
            "n_passadas_medias": total_timesteps / (size * avg_ticks_per_day),
            "n_chunks_needed": n_chunks_needed,
        })

    if verbose:
        print(f"=== Folds de janela fixa (v4, {len(folds)} folds) ===")
        print(f"Throughput assumido : {throughput_steps_per_sec:,.0f} steps/s")
        print(f"Chunk (job Slurm)   : {chunk_seconds:.0f}s de treino efetivo por job")
        print(f"Ticks médios/pregão : {avg_ticks_per_day:,.0f}")
        print(f"Timesteps por fold  : {total_timesteps:,}")
        for f in folds:
            tr, va, te = f["train_orders"], f["val_orders"], f["test_orders"]
            print(f"  Fold {f['fold']}: treino=[{tr[0]:>3}..{tr[-1]:>3}] ({len(tr):>3} dias)  "
                  f"validação=[{va[0]:>3}..{va[-1]:>3}] ({len(va):>3} dias)  "
                  f"teste=[{te[0]:>3}..{te[-1]:>3}] ({len(te):>3} dias)  "
                  f"~{f['n_passadas_medias']:.1f} passadas/dia, "
                  f"~{f['n_chunks_needed']} chunk(s)")
    return folds


# --------------------------------------------------------------------------
# 5) AVALIAÇÃO PARALELA POR DIA + REGISTRO DETALHADO
# --------------------------------------------------------------------------
# Cada pregão é um episódio independente, então a avaliação é paralelizada
# por dia (um processo por dia). Sequencial, val+teste de 30 dias levaria
# ~10 min por modelo (27k ticks × predict por dia) -- inviável no job de
# 20 min do Santos Dumont quando se quer avaliar vários modelos/baselines.

def _n_pool_workers():
    """Processos para avaliação/pré-processamento paralelos por dia
    (EVAL_WORKERS; default = CPUs alocadas). Cada worker importa torch, então
    em máquinas com pouca RAM convém limitar."""
    return int(os.environ.get("EVAL_WORKERS") or _detect_n_train_envs())


def _mp_context():
    import multiprocessing as mp
    return mp.get_context("spawn" if os.name == "nt" else "fork")


_EVAL = {}

# offsets (em ticks) do event study: variação do mispricing em relação à
# ENTRADA, na direção da posição (positivo = spread convergiu a favor)
EVENT_OFFSETS = (-200, -100, -50, 0, 50, 100, 200, 400, 800)

TRADE_COLS = ["direction", "entry_tick", "exit_tick", "pnl",
              "entry_z_compra", "entry_z_venda", "exit_z_compra", "exit_z_venda",
              "entry_misp", "exit_misp", "entry_fair", "exit_fair"]


def _eval_init(spec, mean, std, fee, latency=0):
    torch.set_num_threads(1)
    _EVAL["spec"] = spec
    _EVAL["mean"] = mean
    _EVAL["std"] = std
    _EVAL["fee"] = fee
    _EVAL["latency"] = latency
    _EVAL["model"] = PPO.load(spec[1], device="cpu") if spec[0] == "model" else None


def _make_policy(spec, model, order):
    kind = spec[0]
    if kind == "model":
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    if kind == "flat":
        return lambda obs: 0
    if kind == "random":
        rng = np.random.default_rng(int(spec[1]) * 100003 + int(order))
        return lambda obs: int(rng.integers(3))
    if kind == "threshold":
        # regra de reversão simples nos spreads JÁ normalizados (z-score):
        # compra se WIN está "barato" (spread_compra << 0), vende se está
        # "caro" (spread_venda >> 0), zera quando o spread volta a 0.
        th = float(spec[1])

        def rule(obs):
            pos = obs[2]
            if obs[0] < -th:
                return 1
            if obs[1] > th:
                return 2
            if pos > 0 and obs[0] < 0:
                return 1   # segura a compra até o spread voltar a >= 0
            if pos < 0 and obs[1] > 0:
                return 2   # segura a venda até o spread voltar a <= 0
            return 0
        return rule
    raise ValueError(f"policy spec desconhecida: {spec}")


def _run_day(order, spec, model, mean, std, fee, record_equity, latency=0):
    """Roda 1 pregão e devolve as estatísticas do dia + colunas por negócio.

    `latency` = nº de ticks entre a decisão do agente e a execução da ordem
    (0 = executa no mesmo tick, como no treino). Serve de teste de robustez:
    uma arbitragem real precisa sobreviver a algum atraso de execução."""
    from collections import deque

    df = process_day_cached(order)
    feat = apply_scaler(build_state_features(df), mean, std)
    env = ArbitrageTradingEnv(df, feat, transaction_fee=fee, record_equity=record_equity,
                              track_details=True)
    policy = _make_policy(spec, model, order)
    pending = deque([0] * latency)

    obs, _ = env.reset()
    done = False
    while not done:
        action = policy(obs)
        if latency:
            pending.append(action)
            action = pending.popleft()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    n_ticks = max(env.t, 1)
    dt = pd.to_datetime(df['datahora'])
    hours = dt.dt.hour.to_numpy()
    mid = env.mid
    misp = mid - env.fair

    # --- colunas por negócio ------------------------------------------------
    if env.trade_details:
        arr = np.array(env.trade_details, dtype=float)
    else:
        arr = np.empty((0, len(TRADE_COLS)))
    trades = {name: arr[:, k] for k, name in enumerate(TRADE_COLS)}
    entry = trades["entry_tick"].astype(int)
    direction = trades["direction"]
    trades["duration"] = trades["exit_tick"] - trades["entry_tick"]
    trades["hour_entry"] = hours[entry] if len(entry) else np.empty(0)
    trades["hour_exit"] = hours[np.minimum(trades["exit_tick"].astype(int), len(df) - 1)] if len(entry) else np.empty(0)
    # ganho de convergência (pts, na direção da posição) e componente do preço justo
    trades["d_misp_dir"] = direction * (trades["exit_misp"] - trades["entry_misp"])
    trades["d_fair_dir"] = direction * (trades["exit_fair"] - trades["entry_fair"])
    # event study: direction x (misp[entry+off] - misp[entry])
    offs = np.array(EVENT_OFFSETS)
    idx = entry[:, None] + offs[None, :]
    valid = (idx >= 0) & (idx < len(misp))
    path = np.where(valid, misp[np.clip(idx, 0, len(misp) - 1)] - misp[entry][:, None], np.nan)
    ev = direction[:, None] * path
    for j, off in enumerate(EVENT_OFFSETS):
        trades[f"ev_{off}"] = ev[:, j]

    equity = None
    max_dd = float('nan')
    if env.equity is not None:
        eq = np.array(env.equity)
        max_dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
        equity = eq[::50].tolist()
    pct_long = env.ticks_long / n_ticks
    pct_short = env.ticks_short / n_ticks
    return {
        "order": int(order),
        "date": str(dt.iloc[0].date()),
        "pnl": float(env.total_reward),
        "gross_mtm": float(env.gross_mtm),
        "pnl_spread": float(env.mtm_spread),   # ganho por convergência do mispricing
        "pnl_fair": float(env.mtm_fair),       # ganho por exposição ao preço justo (direcional)
        "cost": float(env.gross_mtm - env.total_reward),
        "win_move": float(mid[-1] - mid[0]),   # variação do WIN no dia (pts)
        "net_exposure": pct_long - pct_short,
        "n_trades": int(env.n_trades_closed),
        "n_wins": int(env.n_trades_won),
        "pct_long": pct_long,
        "pct_short": pct_short,
        "pct_flat": 1.0 - pct_long - pct_short,
        "mean_duration": float(np.mean(env.trade_durations)) if env.trade_durations else 0.0,
        "max_drawdown": max_dd,
        "n_ticks": int(n_ticks),
        "trades": trades,
        "equity": equity,
    }


def _eval_day_worker(args):
    order, record_equity = args
    return _run_day(order, _EVAL["spec"], _EVAL["model"], _EVAL["mean"],
                    _EVAL["std"], _EVAL["fee"], record_equity, _EVAL["latency"])


def run_policy_on_days(spec, orders, mean, std, fee=2.5, n_workers=None,
                       record_equity=False, latency=0):
    """Roda uma política ('model', path) / ('flat',) / ('random', seed) /
    ('threshold', z) em cada pregão de `orders`, em paralelo. Retorna uma
    lista de dicts (um por dia, na ordem de `orders`)."""
    orders = list(orders)
    n_workers = min(n_workers or _n_pool_workers(), len(orders))
    if n_workers <= 1:
        _eval_init(spec, mean, std, fee, latency)
        return [_eval_day_worker((o, record_equity)) for o in orders]
    with _mp_context().Pool(n_workers, initializer=_eval_init,
                            initargs=(spec, mean, std, fee, latency)) as pool:
        return pool.map(_eval_day_worker, [(o, record_equity) for o in orders], chunksize=1)


def _all_trades_df(day_results):
    """DataFrame com TODOS os negócios de todos os dias (colunas por negócio)."""
    frames = []
    for d in day_results:
        t = pd.DataFrame(d["trades"])
        t.insert(0, "order", d["order"])
        t.insert(1, "date", d["date"])
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["order", "date"] + TRADE_COLS)


def summarize_days(day_results):
    """Resumo agregado (dict) de uma lista de resultados por dia."""
    pnl = np.array([d["pnl"] for d in day_results])
    n_trades = int(sum(d["n_trades"] for d in day_results))
    n_wins = int(sum(d["n_wins"] for d in day_results))
    daily_std = float(pnl.std(ddof=1)) if len(pnl) > 1 else float('nan')
    sharpe = float(pnl.mean() / daily_std * np.sqrt(252)) if daily_std and daily_std > 0 else float('nan')
    cum = np.cumsum(pnl)
    max_dd_days = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0

    # --- diagnóstico de arbitragem ------------------------------------------
    win_move = np.array([d["win_move"] for d in day_results])
    pnl_gross = np.array([d["gross_mtm"] for d in day_results])
    corr = beta = float('nan')
    if len(pnl) > 2 and pnl_gross.std() > 0 and win_move.std() > 0:
        corr = float(np.corrcoef(pnl_gross, win_move)[0, 1])
        beta = float(np.polyfit(win_move, pnl_gross, 1)[0])
    d_misp = np.concatenate([d["trades"]["d_misp_dir"] for d in day_results] or [np.empty(0)])
    pnl_spread = float(sum(d["pnl_spread"] for d in day_results))
    pnl_fair = float(sum(d["pnl_fair"] for d in day_results))
    return {
        "n_days": len(day_results),
        "pnl_total": float(pnl.sum()),
        "pnl_total_brl": float(pnl.sum() * POINT_VALUE_BRL),
        "pnl_mean_day": float(pnl.mean()),
        "pnl_std_day": daily_std,
        "sharpe_daily_annualized": sharpe,
        "max_drawdown_daily_cum": max_dd_days,
        "gross_mtm_total": float(sum(d["gross_mtm"] for d in day_results)),
        "cost_total": float(sum(d["cost"] for d in day_results)),
        "pnl_spread_component": pnl_spread,
        "pnl_fair_component": pnl_fair,
        "spread_share_of_gross": pnl_spread / (pnl_spread + pnl_fair) if (pnl_spread + pnl_fair) != 0 else float('nan'),
        "corr_gross_vs_win_move": corr,
        "beta_gross_vs_win_move": beta,
        "mean_net_exposure": float(np.mean([d["net_exposure"] for d in day_results])),
        "mean_d_misp_per_trade": float(d_misp.mean()) if len(d_misp) else float('nan'),
        "pct_trades_converged": float((d_misp > 0).mean()) if len(d_misp) else float('nan'),
        "n_trades": n_trades,
        "trades_per_day": n_trades / max(len(day_results), 1),
        "win_rate": n_wins / max(n_trades, 1),
        "pct_long": float(np.mean([d["pct_long"] for d in day_results])),
        "pct_short": float(np.mean([d["pct_short"] for d in day_results])),
        "pct_flat": float(np.mean([d["pct_flat"] for d in day_results])),
        "mean_trade_duration_ticks": float(np.mean(
            np.concatenate([d["trades"]["duration"] for d in day_results] or [np.zeros(1)]))),
    }


def _signal_z(trades):
    """z do sinal na ENTRADA, orientado: positivo = sinal 'forte' (compra com
    spread_compra baixo, venda com spread_venda alto)."""
    return np.where(trades["direction"] > 0, -trades["entry_z_compra"], trades["entry_z_venda"])


def evaluate_and_log(label, spec, orders, mean, std, fee=2.5, out_dir=None,
                     n_workers=None, verbose=True, latency=0):
    """Avalia uma política em `orders`, imprime o resumo e (se `out_dir`)
    grava CSVs: `<label>_days.csv`, `_trades.csv` (com spreads de entrada/
    saída e ganho de convergência), `_equity.csv`, `_eventstudy.csv` e
    `_entry_buckets.csv`. Retorna o dict de resumo."""
    days = run_policy_on_days(spec, orders, mean, std, fee=fee, n_workers=n_workers,
                              record_equity=out_dir is not None, latency=latency)
    summ = summarize_days(days)

    if verbose:
        print(f"\n--- {label} (fee={fee}, latência={latency} ticks) ---")
        print(f"Dias: {summ['n_days']}  |  P/L total: {summ['pnl_total']:.1f} pts "
              f"(R$ {summ['pnl_total_brl']:.2f})  |  média/dia: {summ['pnl_mean_day']:.1f} "
              f"(desvio {summ['pnl_std_day']:.1f})  |  Sharpe diário anualizado: "
              f"{summ['sharpe_daily_annualized']:.2f}")
        print(f"Negócios: {summ['n_trades']} ({summ['trades_per_day']:.1f}/dia)  |  "
              f"taxa de acerto: {summ['win_rate']:.1%}  |  P/L bruto: "
              f"{summ['gross_mtm_total']:.1f}  |  custos: {summ['cost_total']:.1f}")
        print(f"Tempo comprado/vendido/flat: {summ['pct_long']:.1%} / "
              f"{summ['pct_short']:.1%} / {summ['pct_flat']:.1%}  |  duração média "
              f"do negócio: {summ['mean_trade_duration_ticks']:.0f} ticks")
        print(f"ARBITRAGEM? P/L bruto = convergência do spread {summ['pnl_spread_component']:.1f} "
              f"+ direcional (preço justo) {summ['pnl_fair_component']:.1f}  "
              f"(spread = {summ['spread_share_of_gross']:.0%} do bruto)")
        print(f"   corr(P/L bruto dia, movimento do WIN) = {summ['corr_gross_vs_win_move']:.2f}  |  "
              f"exposição líquida média = {summ['mean_net_exposure']:+.1%}  |  "
              f"negócios em que o spread convergiu: {summ['pct_trades_converged']:.1%} "
              f"(média {summ['mean_d_misp_per_trade']:+.2f} pts)")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        day_cols = ["order", "date", "pnl", "gross_mtm", "pnl_spread", "pnl_fair", "cost",
                    "win_move", "net_exposure", "n_trades", "n_wins", "pct_long",
                    "pct_short", "pct_flat", "mean_duration", "max_drawdown", "n_ticks"]
        pd.DataFrame([{k: d[k] for k in day_cols} for d in days]).to_csv(
            os.path.join(out_dir, f"{label}_days.csv"), index=False)

        trades = _all_trades_df(days)
        if len(trades) > 50_000:  # ex.: baseline aleatório faz ~13k negócios/dia
            trades = trades.sample(50_000, random_state=0).sort_index()
        trades.to_csv(os.path.join(out_dir, f"{label}_trades.csv"), index=False)

        eq_rows, base = [], 0.0
        for d in days:
            for k, v in enumerate(d["equity"] or []):
                eq_rows.append({"order": d["order"], "tick": k * 50, "equity": base + v})
            base += d["pnl"]
        pd.DataFrame(eq_rows, columns=["order", "tick", "equity"]).to_csv(
            os.path.join(out_dir, f"{label}_equity.csv"), index=False)

        full = _all_trades_df(days)
        if len(full):
            # event study: variação média do mispricing após a entrada, na
            # direção da posição, por direção (long/short) e geral
            rows = []
            for name, sub in (("all", full), ("long", full[full.direction > 0]),
                              ("short", full[full.direction < 0])):
                for off in EVENT_OFFSETS:
                    v = sub[f"ev_{off}"].dropna()
                    rows.append({"group": name, "offset_ticks": off,
                                 "mean_directed_d_misp": v.mean() if len(v) else np.nan,
                                 "n": len(v)})
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{label}_eventstudy.csv"), index=False)

            # por força do sinal na entrada: o agente ganha mais quando o
            # spread está mais extremo? (arbitragem => sim)
            z = _signal_z(full)
            bins = [-np.inf, 0.0, 1.0, 2.0, np.inf]
            names = ["z<0", "0<=z<1", "1<=z<2", "z>=2"]
            bucket = pd.cut(pd.Series(z), bins=bins, labels=names, right=False)
            g = full.assign(bucket=bucket.values).groupby("bucket", observed=False).agg(
                n=("pnl", "size"), pnl_mean=("pnl", "mean"),
                win_rate=("pnl", lambda x: (x > 0).mean()),
                d_misp_mean=("d_misp_dir", "mean"), duration_mean=("duration", "mean"))
            g.to_csv(os.path.join(out_dir, f"{label}_entry_buckets.csv"))
            if verbose:
                print("   por força do sinal na entrada (z orientado):")
                for b, r in g.iterrows():
                    if r["n"]:
                        print(f"     {b:>7}: n={int(r['n']):>6}  P/L médio {r['pnl_mean']:+7.2f}  "
                              f"acerto {r['win_rate']:.0%}  convergência média {r['d_misp_mean']:+6.2f}")
                ev_all = {off: full[f'ev_{off}'].mean() for off in (50, 200, 800)}
                print("   event study (Δ mispricing médio pós-entrada, a favor): "
                      + "  ".join(f"+{o} ticks: {v:+.2f}" for o, v in ev_all.items()))
            pnls = full["pnl"].to_numpy()
            wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
            print(f"   P/L por negócio: ganho médio {wins.mean() if len(wins) else 0:.2f} | "
                  f"perda média {losses.mean() if len(losses) else 0:.2f} | "
                  + " ".join(f"p{q}={np.percentile(pnls, q):.2f}" for q in (1, 5, 50, 95, 99)))
    return summ


# --------------------------------------------------------------------------
# 6) TREINO (PPO) EM CHUNKS -- estado/checkpoints/logs em runs/<tag>/fold<N>/
# --------------------------------------------------------------------------

RUNS_DIR = os.environ.get("RUNS_DIR", "runs")


def fold_run_dir(run_tag, fold_num):
    d = os.path.join(RUNS_DIR, run_tag, f"fold{fold_num}")
    os.makedirs(d, exist_ok=True)
    return d


def _new_state():
    return {"num_timesteps": 0, "done": False, "best_val_score": None,
            "best_val_steps": None, "chunks": []}


def read_state(run_dir):
    import json
    path = os.path.join(run_dir, "state.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return _new_state()


def write_state(run_dir, state):
    import json
    tmp = os.path.join(run_dir, "state.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, os.path.join(run_dir, "state.json"))


def ppo_hparams_from_env():
    """Hiperparâmetros PPO, sobrescrevíveis por env var (ficam gravados em
    metadata.json). gamma=0.999999: com episódio de ~27,7k ticks, desconta
    só ~2,7% até o fim do dia (0.999 zerava o fim do pregão)."""
    return {
        "learning_rate": float(os.environ.get("LEARNING_RATE", "3e-4")),
        "n_steps": int(os.environ.get("N_STEPS", "2048")),
        "batch_size": int(os.environ.get("BATCH_SIZE", "1024")),
        "n_epochs": int(os.environ.get("N_EPOCHS", "10")),
        "gamma": float(os.environ.get("GAMMA", "0.999999")),
        "gae_lambda": float(os.environ.get("GAE_LAMBDA", "0.95")),
        "ent_coef": float(os.environ.get("ENT_COEF", "0.0")),
    }


def write_metadata(run_dir, fold, hparams, extra):
    import json
    import platform
    import subprocess
    import stable_baselines3
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_hash = None
    meta = {
        "git_hash": git_hash,
        "hparams": hparams,
        "fold": fold["fold"],
        "train_orders": {"first": fold["train_orders"][0], "last": fold["train_orders"][-1],
                         "n": len(fold["train_orders"])},
        "val_orders": fold["val_orders"],
        "test_orders": fold["test_orders"],
        "total_timesteps": fold["total_timesteps"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__,
                     "gymnasium": gym.__version__},
        "state_features": STATE_FEATURE_NAMES + ["posicao", "pl_nao_realizado_norm"],
        **extra,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def _day_features(order):
    return build_state_features(process_day_cached(order))


def load_or_fit_scaler(run_dir, train_orders, n_workers):
    """Scaler (mean/std) só do treino, salvo em scaler.npz -- os chunks
    seguintes e o job de avaliação reusam o MESMO arquivo em vez de
    reprocessar todos os dias de treino (que custava ~1-3 min por chunk)."""
    path = os.path.join(run_dir, "scaler.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["mean"], z["std"]
    n_workers = max(1, min(n_workers, len(train_orders)))
    if n_workers == 1:
        feats = [_day_features(o) for o in train_orders]
    else:
        with _mp_context().Pool(n_workers) as pool:
            feats = pool.map(_day_features, train_orders, chunksize=1)
    mean, std = fit_feature_scaler(feats)
    np.savez(path, mean=mean, std=std)
    return mean, std


class ValidationCallback(BaseCallback):
    """Avalia a política DETERMINÍSTICA em validação (e num subconjunto fixo
    de dias de treino, pra curva treino-vs-validação comparável) a cada
    `eval_every_steps` timesteps, grava val_curve.csv e mantém o MELHOR
    checkpoint de validação entre chunks (ppo_best_val.zip + state.json).

    Seleção de checkpoint usa SÓ validação; o teste não é tocado aqui.
    """

    def __init__(self, run_dir, val_orders, train_subset, mean, std, fee,
                 eval_every_steps, deadline=None, verbose=1):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.val_orders = val_orders
        self.train_subset = train_subset
        self.mean, self.std, self.fee = mean, std, fee
        self.eval_every = eval_every_steps
        self.deadline = deadline
        self.next_eval = None
        self.total_eval_seconds = 0.0

    def _on_training_start(self):
        self.next_eval = (self.num_timesteps // self.eval_every + 1) * self.eval_every

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        if self.deadline is not None and self.deadline - time.time() < 240:
            return True  # sem tempo seguro: adia pra o próximo chunk
        self.next_eval += self.eval_every
        self._evaluate()
        return True

    def _evaluate(self):
        tmp = os.path.join(self.run_dir, "_eval_tmp.zip")
        self.model.save(tmp)
        t0 = time.time()
        val = summarize_days(run_policy_on_days(("model", tmp), self.val_orders,
                                                self.mean, self.std, self.fee))
        trn = summarize_days(run_policy_on_days(("model", tmp), self.train_subset,
                                                self.mean, self.std, self.fee))
        row = {"timesteps": self.num_timesteps,
               "val_pnl_total": val["pnl_total"], "val_pnl_mean_day": val["pnl_mean_day"],
               "val_trades_per_day": val["trades_per_day"], "val_win_rate": val["win_rate"],
               "val_pct_flat": val["pct_flat"],
               "train_subset_pnl_mean_day": trn["pnl_mean_day"],
               "train_subset_trades_per_day": trn["trades_per_day"],
               "eval_seconds": time.time() - t0}
        self.total_eval_seconds += row["eval_seconds"]
        path = os.path.join(self.run_dir, "val_curve.csv")
        pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        if self.verbose:
            print(f"[Validação @ {self.num_timesteps:,}] P/L val {val['pnl_total']:.1f} pts "
                  f"({val['trades_per_day']:.1f} negócios/dia) | treino(sub) "
                  f"{trn['pnl_mean_day']:.1f}/dia | {row['eval_seconds']:.0f}s")

        state = read_state(self.run_dir)
        if state["best_val_score"] is None or val["pnl_total"] > state["best_val_score"]:
            os.replace(tmp, os.path.join(self.run_dir, "ppo_best_val.zip"))
            state["best_val_score"] = val["pnl_total"]
            state["best_val_steps"] = int(self.num_timesteps)
            write_state(self.run_dir, state)
            if self.verbose:
                print("  -> novo melhor checkpoint de validação")
        elif os.path.exists(tmp):
            os.remove(tmp)


class TradeStatsCallback(BaseCallback):
    """Registra no logger do SB3 quantos negócios por episódio o agente faz
    durante o TREINO (sinal de degeneração: a política ruim do sweep fazia
    ~7 mil negócios/dia)."""

    def __init__(self):
        super().__init__(0)
        self._trades, self._wins = [], []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self._trades.append(info.get("n_trades_closed", 0))
                self._wins.append(info.get("n_trades_won", 0))
        return True

    def _on_rollout_end(self):
        if self._trades:
            self.logger.record("train/trades_per_episode", float(np.mean(self._trades)))
            self.logger.record("train/win_rate_episode",
                               float(np.sum(self._wins) / max(np.sum(self._trades), 1)))
            self._trades, self._wins = [], []


def train_chunk(fold, run_dir, hparams, seed=0, transaction_fee=2.5, max_seconds=None,
                resume=False, chunk_index=1, eval_every_steps=4_000_000,
                ckpt_every_steps=3_000_000):
    """Treina até `fold['total_timesteps']` OU até o deadline de tempo,
    o que vier primeiro, e atualiza state.json (`done` quando o alvo foi
    atingido). Chamado uma vez por job Slurm; o próximo chunk retoma de
    ppo_last.zip."""
    from stable_baselines3.common.callbacks import CheckpointCallback

    n_train_envs = int(os.environ.get("N_TRAIN_ENVS") or _detect_n_train_envs())
    mean, std = load_or_fit_scaler(run_dir, fold["train_orders"], _n_pool_workers())

    model_path = os.path.join(run_dir, "ppo_last.zip")
    resume = resume and os.path.exists(model_path)
    if not resume:
        write_state(run_dir, _new_state())
        write_metadata(run_dir, fold, hparams, {"seed": seed, "transaction_fee": transaction_fee,
                                                 "n_train_envs": n_train_envs,
                                                 "vec_env": os.environ.get("VEC_ENV", "dummy"),
                                                 "torch_threads": torch.get_num_threads()})

    train_env_fns = [
        partial(_make_multiday_env, fold["train_orders"], mean, std,
                transaction_fee, None, seed * 1000 + i)
        for i in range(n_train_envs)
    ]
    # DummyVecEnv (tudo num processo) x SubprocVecEnv (1 processo por env):
    # com o env de 4 dimensões o step custa ~4 µs, então o IPC do
    # SubprocVecEnv domina e o DummyVecEnv foi ~2x mais rápido no teste
    # local. Confirmar no nó com slurm/benchmark_throughput.sbatch
    # (VEC_ENV=subproc|dummy).
    vec_kind = os.environ.get("VEC_ENV", "dummy")
    vec_env = (DummyVecEnv if vec_kind == "dummy" else SubprocVecEnv)(train_env_fns)
    print(f"VecEnv: {vec_kind} com {n_train_envs} envs | torch threads: {torch.get_num_threads()}")

    if resume:
        print(f"Retomando treino a partir de '{model_path}'.")
        model = PPO.load(model_path, env=vec_env)
    else:
        model = PPO("MlpPolicy", vec_env, seed=seed, verbose=1, **hparams)
    model.set_logger(configure_logger(os.path.join(run_dir, "logs", f"chunk{chunk_index}"),
                                      ["stdout", "csv"]))

    n_start = int(model.num_timesteps)
    remaining = max(int(fold["total_timesteps"]) - n_start, 0)
    deadline = (SCRIPT_START + max_seconds) if max_seconds is not None else None

    # subconjunto FIXO de dias de treino, avaliado com a mesma política
    # determinística usada em validação (curva treino-vs-validação
    # comparável; ep_rew_mean do treino é estocástico e não é comparável)
    step = max(1, len(fold["train_orders"]) // 10)
    train_subset = fold["train_orders"][::step][:10]

    val_cb = ValidationCallback(run_dir, fold["val_orders"], train_subset, mean, std,
                                transaction_fee, eval_every_steps, deadline)
    callbacks = [
        val_cb,
        TradeStatsCallback(),
        CheckpointCallback(save_freq=max(1, ckpt_every_steps // n_train_envs),
                           save_path=os.path.join(run_dir, "checkpoints"),
                           name_prefix="ckpt"),
    ]
    if deadline is not None:
        callbacks.append(TimeLimitCallback(deadline, verbose=1))

    t_learn = time.time()
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=CallbackList(callbacks),
                    reset_num_timesteps=not resume)
    learn_seconds = time.time() - t_learn
    train_seconds = max(learn_seconds - val_cb.total_eval_seconds, 1e-9)
    model.save(model_path)
    vec_env.close()

    state = read_state(run_dir)  # ValidationCallback pode ter atualizado o best
    n_end = int(model.num_timesteps)
    state["num_timesteps"] = n_end
    state["done"] = n_end >= int(fold["total_timesteps"])
    state["chunks"].append({
        "chunk": chunk_index, "start_steps": n_start, "end_steps": n_end,
        "learn_seconds": round(learn_seconds, 1),
        "eval_seconds": round(val_cb.total_eval_seconds, 1),
        "startup_seconds": round(t_learn - SCRIPT_START, 1),
        # steps/s SEM contar o tempo das validações periódicas
        "steps_per_sec": round((n_end - n_start) / train_seconds, 1),
    })
    write_state(run_dir, state)
    print(f"\n=== Chunk {chunk_index}: {n_start:,} -> {n_end:,} / "
          f"{fold['total_timesteps']:,} timesteps | done={state['done']} | "
          f"{state['chunks'][-1]['steps_per_sec']:.0f} steps/s ===")
    return state


# --------------------------------------------------------------------------
# 7) JOB DE AVALIAÇÃO (val+teste do melhor-de-validação e do último modelo,
#    baselines, sensibilidade a custo, curva de checkpoints)
# --------------------------------------------------------------------------

def run_eval_job(fold, run_dir, fee=2.5, job_hard_seconds=None, part=1):
    """part=1: modelos principais, baselines e sensibilidade a custo.
    part=2: teste de latência e curva de checkpoints. Divididos em dois
    jobs porque juntos passam de 20 min."""
    mean, std = load_or_fit_scaler(run_dir, fold["train_orders"], _n_pool_workers())
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    deadline = (SCRIPT_START + job_hard_seconds) if job_hard_seconds else None
    summary_path = os.path.join(out_dir, "summary.csv")
    if part == 1 and os.path.exists(summary_path):
        os.remove(summary_path)

    def has_time(needed):
        return deadline is None or deadline - time.time() > needed

    def record(label, model_name, set_name, fee_used, summ):
        row = {"label": label, "model": model_name, "set": set_name, "fee": fee_used, **summ}
        pd.DataFrame([row]).to_csv(summary_path, mode="a",
                                   header=not os.path.exists(summary_path), index=False)

    sets = {"val": fold["val_orders"], "test": fold["test_orders"]}
    models = {}
    for name, fname in (("best_val", "ppo_best_val.zip"), ("last", "ppo_last.zip")):
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            models[name] = path

    ref_name = "best_val" if "best_val" in models else ("last" if "last" in models else None)

    if part == 1:
        _eval_part1(models, ref_name, sets, mean, std, fee, out_dir, has_time, record)
    else:
        _eval_part2(models, ref_name, sets, mean, std, fee, out_dir, has_time, record, run_dir)


def _eval_part1(models, ref_name, sets, mean, std, fee, out_dir, has_time, record):
    # 1) modelos principais (fee de treino)
    for mname, mpath in models.items():
        for sname, orders in sets.items():
            if not has_time(120):
                print(f"[eval] sem tempo para {mname}/{sname}; pulando")
                continue
            label = f"{mname}_{sname}_fee{fee}"
            record(label, mname, sname, fee,
                   evaluate_and_log(label, ("model", mpath), orders, mean, std,
                                    fee=fee, out_dir=out_dir))

    # 2) baselines
    for bname, spec in (("flat", ("flat",)), ("threshold1.0", ("threshold", 1.0)),
                        ("random", ("random", 0))):
        for sname, orders in sets.items():
            if not has_time(120):
                continue
            label = f"baseline_{bname}_{sname}_fee{fee}"
            record(label, f"baseline_{bname}", sname, fee,
                   evaluate_and_log(label, spec, orders, mean, std, fee=fee, out_dir=out_dir))

    # 3) sensibilidade a custo do melhor-de-validação
    if ref_name:
        for fee_alt in (0.0, 0.5):
            for sname, orders in sets.items():
                if not has_time(120):
                    continue
                label = f"cost_sens_{sname}_fee{fee_alt}"
                record(label, ref_name, sname, fee_alt,
                       evaluate_and_log(label, ("model", models[ref_name]), orders, mean, std,
                                        fee=fee_alt, out_dir=out_dir))



def _eval_part2(models, ref_name, sets, mean, std, fee, out_dir, has_time, record, run_dir):
    import glob
    import re
    # 3.5) teste de latência: a decisão só é executada `k` ticks depois (uma
    # arbitragem real precisa sobreviver a algum atraso de execução)
    if ref_name:
        for lat in (1, 5, 20):
            for sname, orders in sets.items():
                if not has_time(120):
                    continue
                label = f"latency{lat}_{sname}_fee{fee}"
                record(label, ref_name, sname, fee,
                       evaluate_and_log(label, ("model", models[ref_name]), orders, mean, std,
                                        fee=fee, out_dir=out_dir, latency=lat))

    # 4) curva de checkpoints em val e teste (SÓ para relatório, não seleciona nada)
    curve_path = os.path.join(out_dir, "checkpoint_curve.csv")
    if os.path.exists(curve_path):
        os.remove(curve_path)
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_*_steps.zip")),
                   key=lambda p: int(re.search(r"ckpt_(\d+)_steps", p).group(1)))
    for ck in ckpts:
        if not has_time(150):
            print("[eval] sem tempo para o resto da curva de checkpoints")
            break
        steps = int(re.search(r"ckpt_(\d+)_steps", ck).group(1))
        row = {"timesteps": steps}
        for sname, orders in sets.items():
            summ = summarize_days(run_policy_on_days(("model", ck), orders, mean, std, fee))
            row.update({f"{sname}_pnl_total": summ["pnl_total"],
                        f"{sname}_trades_per_day": summ["trades_per_day"],
                        f"{sname}_win_rate": summ["win_rate"]})
        pd.DataFrame([row]).to_csv(curve_path, mode="a",
                                   header=not os.path.exists(curve_path), index=False)
        print(f"[curva] {steps:,}: val {row['val_pnl_total']:.1f} | teste {row['test_pnl_total']:.1f}")


# --------------------------------------------------------------------------
# EXECUÇÃO
# --------------------------------------------------------------------------

if __name__ == "__main__":
    all_orders = list(range(1, 481))  # 480 pregões disponíveis, em ordem cronológica
    max_pregoes = os.environ.get("MAX_PREGOES")
    if max_pregoes:
        all_orders = all_orders[:int(max_pregoes)]

    run_tag = os.environ.get("RUN_TAG", "v4")
    seed = int(os.environ.get("SEED", "0"))
    fee = float(os.environ.get("TRANSACTION_FEE", "2.5"))
    train_sizes = tuple(int(x) for x in os.environ.get("TRAIN_SIZES", "100,150,200").split(","))

    folds = generate_fixed_window_folds(
        all_orders,
        train_sizes=train_sizes,
        n_val=int(os.environ.get("N_VAL", "15")),
        n_test=int(os.environ.get("N_TEST", "15")),
        total_timesteps=int(float(os.environ.get("TOTAL_TIMESTEPS", "30000000"))),
        throughput_steps_per_sec=float(os.environ.get("THROUGHPUT_STEPS_PER_SEC", "5500")),
        chunk_seconds=float(os.environ.get("CHUNK_SECONDS", "1050")),
    )

    # slurm/submit_all_folds.sh usa isto pra estimar quantos chunks encadear
    if os.environ.get("PRINT_FOLD_PLAN") == "1":
        for f in folds:
            print(f"{f['fold']} {f['n_chunks_needed']}")
        raise SystemExit(0)

    fold_sel = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("FOLD")
    folds_to_run = [f for f in folds if f["fold"] == int(fold_sel)] if fold_sel else folds

    max_seconds = os.environ.get("TRAIN_MAX_SECONDS")
    max_seconds = float(max_seconds) if max_seconds else None
    job_hard = os.environ.get("JOB_HARD_SECONDS")
    job_hard = float(job_hard) if job_hard else None
    hparams = ppo_hparams_from_env()
    print(f"Hiperparâmetros PPO: {hparams}")

    for f in folds_to_run:
        print(f"\n########## FOLD {f['fold']}/{len(folds)} (RUN_TAG={run_tag}) ##########")
        run_dir = fold_run_dir(run_tag, f["fold"])

        if os.environ.get("RUN_EVAL") in ("1", "2"):
            run_eval_job(f, run_dir, fee=fee, job_hard_seconds=job_hard,
                         part=int(os.environ["RUN_EVAL"]))
            continue

        state = train_chunk(
            f, run_dir, hparams, seed=seed, transaction_fee=fee,
            max_seconds=max_seconds,
            resume=os.environ.get("RESUME_TRAINING", "0") == "1",
            chunk_index=int(os.environ.get("CHUNK_INDEX", "1")),
            eval_every_steps=int(float(os.environ.get("EVAL_EVERY_STEPS", "4000000"))),
            ckpt_every_steps=int(float(os.environ.get("CKPT_EVERY_STEPS", "3000000"))),
        )
        # execução local (sem limite de job): avalia direto ao terminar
        if state["done"] and job_hard is None:
            run_eval_job(f, run_dir, fee=fee, part=1)
            run_eval_job(f, run_dir, fee=fee, part=2)
