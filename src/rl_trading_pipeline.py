"""
Reinforcement Learning para negociação intradiária BOVA11 x WINM21 --
5 features: spread de mispricing, spread bid-ask, atribuição de
movimento (BOVA vs WIN) e momentum (curto e longo).

Diferente da abordagem supervisionada (LSTM prevendo Lucro/Prejuízo por
negociação com stops fixos Re/Ri), aqui o agente controla a posição TICK A
TICK: decide entrar, segurar ou sair a cada instante, aprendendo sua
própria política de entrada/saída.

O estado do agente é composto por uma JANELA de N_TICKS valores de cada
uma das 5 features (ver build_feature_matrix para detalhes de cada uma)
-- sem bandas de Bollinger, sem volume, sem retornos separados.

Requer:
    pip install gymnasium stable-baselines3

Estrutura:
  1) process_day()          -> gera o preço-justo (Wajusto/Wbjusto) de um
                                pregão, sem bandas de Bollinger
  2) build_feature_matrix() -> calcula as 5 features (RL_FEATURE_NAMES)
                                por tick
  3) fit_feature_scaler()   -> normaliza usando estatística SOMENTE do
                                treino
  4) ArbitrageTradingEnv    -> ambiente Gymnasium: 1 episódio = 1 pregão
  5) MultiDayEnv            -> alterna entre vários pregões pré-carregados
                                a cada reset, para treinar em múltiplos dias
  6) train_ppo()            -> treina um agente PPO (stable-baselines3)
  7) evaluate_policy()      -> roda a política treinada nos dias de
                                teste/validação e reporta lucro total,
                                número de negociações e taxa de acerto
"""

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

from config import data_path, POINT_VALUE_BRL

# mesma lógica do OMP_NUM_THREADS acima, mas para o processo PRINCIPAL
# (que roda o forward/backward da rede via torch, concorrendo por CPU com
# os processos do SubprocVecEnv) -- por padrão o torch abre 1 thread por
# núcleo físico, o que sozinho já tomaria o nó inteiro.
torch.set_num_threads(1)


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

    Observação (Box): janela de N_TICKS ticks de features relativas,
    achatada em um vetor, concatenada com [posição atual (one-hot, 3),
    PnL não-realizado normalizado (1)].

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

    def __init__(self, df, feature_matrix, n_ticks=120, transaction_fee=2.5,
                 reward_scale=1.0, max_loss_per_position=None):
        super().__init__()
        assert len(df) == len(feature_matrix)
        self.df = df
        # bid/ask pré-extraídos como arrays numpy: evita .iloc (indexação
        # pandas) tick a tick dentro de step()/_get_obs(), que é bem mais
        # lento que indexação numpy direta.
        self.bid = df['bid'].to_numpy(dtype=float)
        self.ask = df['ask'].to_numpy(dtype=float)
        self.feature_matrix = feature_matrix
        self.n_ticks = n_ticks
        self.transaction_fee = transaction_fee
        self.reward_scale = reward_scale
        # stop-loss por posição: regra FIXA (não aprendida), desligada por
        # padrão (None). Se setado, força o fechamento da posição aberta
        # assim que a perda não-realizada ultrapassar esse limiar (em
        # reais) -- ver aplicação em step(). Não é usado na configuração
        # de treino padrão -- ver train_ppo()/build_day_envs().
        self.max_loss_per_position = max_loss_per_position
        self.n_features = feature_matrix.shape[1]

        obs_dim = self.n_ticks * self.n_features + 3 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _reset_state(self):
        self.t = self.n_ticks - 1          # primeiro tick com histórico completo
        self.position = 0                  # -1, 0, 1
        self.entry_price = 0.0             # preço de execução da posição aberta
        self.n_trades_closed = 0
        self.n_trades_won = 0
        self.trade_pnls = []               # P&L realizado de CADA negócio fechado
        self.n_stop_loss_triggers = 0      # quantas vezes o stop-loss forçou fechamento

    def _get_obs(self):
        window = self.feature_matrix[self.t - self.n_ticks + 1: self.t + 1]
        window_flat = window.flatten()

        pos_onehot = np.zeros(3, dtype=np.float32)
        pos_onehot[self.position + 1] = 1.0  # posição -1,0,1 -> índice 0,1,2

        bid_t = self.bid[self.t]
        ask_t = self.ask[self.t]
        if self.position == 1:
            unrealized = bid_t - self.entry_price
        elif self.position == -1:
            unrealized = self.entry_price - ask_t
        else:
            unrealized = 0.0
        # normalização grosseira pelo spread médio para manter escala razoável
        unrealized_norm = np.array([unrealized / max(ask_t - bid_t, 1e-6)],
                                    dtype=np.float32)

        return np.concatenate([window_flat, pos_onehot, unrealized_norm]).astype(np.float32)

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
        return -spread_cost - self.transaction_fee

    def step(self, action):
        target_position = {0: 0, 1: 1, 2: -1}[int(action)]

        bid_t = self.bid[self.t]
        ask_t = self.ask[self.t]
        mid_t = (bid_t + ask_t) / 2

        reward = 0.0

        # 1) marcação a mercado da posição já aberta ANTES desta ação,
        #    desde o mid-price do tick anterior até agora
        if self.t > self.n_ticks - 1:
            bid_prev = self.bid[self.t - 1]
            ask_prev = self.ask[self.t - 1]
            mid_prev = (bid_prev + ask_prev) / 2
            reward += self.position * (mid_t - mid_prev)

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

        self.t += 1
        terminated = self.t >= len(self.df) - 1
        truncated = False

        # 3) fim do pregão: força o fechamento de qualquer posição aberta
        if terminated and self.position != 0:
            bid_last = self.bid[self.t]
            ask_last = self.ask[self.t]
            close_reward, _ = self._close_leg(bid_last, ask_last)
            reward += close_reward
            self.position = 0

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


def build_day_envs(orders, mean, std, n_ticks=120, transaction_fee=2.5,
                    max_loss_per_position=None):
    """Carrega e processa cada pregão em `orders` UMA VEZ, retornando uma
    lista de ArbitrageTradingEnv prontos para uso -- evita reler os JSONs
    a cada episódio de treino.

    `max_loss_per_position=None` (default) mantém o stop-loss desligado
    -- ver ArbitrageTradingEnv.
    """
    envs = []
    for order in orders:
        df = process_day(order)
        feat = build_feature_matrix(df)
        feat = apply_scaler(feat, mean, std)
        envs.append(ArbitrageTradingEnv(df, feat, n_ticks=n_ticks,
                                         transaction_fee=transaction_fee,
                                         max_loss_per_position=max_loss_per_position))
    return envs


def _make_multiday_env(orders, mean, std, n_ticks=120, transaction_fee=2.5,
                        max_loss_per_position=None):
    """Função construtora usada como env_fn do SubprocVecEnv.

    Recebe só identificadores leves (lista de números de pregão + mean/std
    da normalização) e faz o trabalho pesado (ler JSONs, processar,
    montar features) DENTRO do processo em que é chamada. Isso evita
    serializar/enviar pelo pipe DataFrames e arrays já processados do
    processo pai para cada processo-filho -- só os `orders` (ints) e
    mean/std (arrays pequenos) atravessam o pickle.
    """
    day_envs = build_day_envs(orders, mean, std, n_ticks=n_ticks,
                               transaction_fee=transaction_fee,
                               max_loss_per_position=max_loss_per_position)
    return Monitor(MultiDayEnv(day_envs))


# --------------------------------------------------------------------------
# 4.5) SPLIT WALK-FORWARD (TimeSeriesSplit) -- treino/validação/teste por fold
# --------------------------------------------------------------------------

def generate_walk_forward_folds(
    all_orders,
    n_folds=3,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    time_budget_hours=5.0,
    throughput_steps_per_sec=4209.0,
    min_visits_per_train_day=20,
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

    `time_budget_hours` é o orçamento TOTAL somado entre todos os folds
    (não por fold) -- é dividido igualmente por `n_folds` para chegar
    no orçamento de tempo/timesteps de cada fold individual. Ex.:
    time_budget_hours=5.0 com n_folds=3 -> ~1h40 (~5h/3) de treino por
    fold, ~5h no total rodando os 3 folds em sequência.

    O tamanho do último fold (o maior) é limitado por esse orçamento:
    dado o throughput medido (`throughput_steps_per_sec` -- default é o
    fps real de treino PPO medido no smoke test do SubprocVecEnv da
    tarefa 3 NESTA máquina; ajuste este valor ao rodar em outra
    máquina), calcula-se quantos timesteps cabem por fold, o que por
    sua vez limita quantos dias de treino cabem no último fold, de
    forma que cada dia, em média, seja visitado pelo menos
    `min_visits_per_train_day` vezes (com poucos timesteps e muitos
    dias, cada dia seria raramente visitado pelo MultiDayEnv -- melhor
    usar menos dias e visitar cada um o suficiente).

    Retorna uma lista de dicts, um por fold, com "fold", "train_orders",
    "val_orders", "test_orders", "total_timesteps" (orçamento de
    timesteps calibrado para aquele fold -- ver comentário na conta
    logo abaixo) e "n_passadas_medias" (quantas vezes, em média, cada
    dia de treino daquele fold é visitado dentro desse orçamento).
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9
    assert n_folds >= 2, "TimeSeriesSplit exige n_folds >= 2"

    if avg_ticks_per_day is None:
        sample_orders = all_orders[:2]
        avg_ticks_per_day = float(np.mean([len(process_day(o)) for o in sample_orders]))

    # --- Cálculo do orçamento de timesteps por fold ---
    # time_budget_hours é o orçamento TOTAL entre todos os folds (decisão
    # confirmada com o usuário), então cada fold recebe uma fração igual:
    budget_seconds_per_fold = (time_budget_hours * 3600.0) / n_folds
    # throughput_steps_per_sec (medido empiricamente, tarefa 3) converte
    # esse tempo em quantos timesteps cabem no orçamento daquele fold:
    timesteps_budget_per_fold = throughput_steps_per_sec * budget_seconds_per_fold

    # total_timesteps de um fold = dias_treino × ticks_por_dia × n_passadas
    # (n_passadas = quantas vezes, em média, o MultiDayEnv visita cada dia
    # de treino dentro do orçamento). Aqui usamos essa mesma equação ao
    # contrário para dimensionar o fold: fixamos n_passadas em
    # min_visits_per_train_day (queremos garantir esse mínimo de
    # cobertura por dia) e resolvemos para dias_treino, dado o orçamento
    # de timesteps já calculado acima:
    #   dias_treino = timesteps_budget_per_fold / (ticks_por_dia × n_passadas)
    max_train_days = int(timesteps_budget_per_fold / (avg_ticks_per_day * min_visits_per_train_day))
    max_train_days = max(max_train_days, n_folds)  # ao menos 1 dia de treino por fold

    # tamanho do bloco val+teste calibrado para que o ÚLTIMO fold (o
    # maior) fique próximo de train_frac/val_frac/test_frac
    last_fold_window = int(round(max_train_days / train_frac))
    test_size = max(1, int(round(last_fold_window * (val_frac + test_frac))))

    n_samples_total = min(max_train_days + test_size, len(all_orders))
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

        # total_timesteps do fold = orçamento de timesteps do fold (fixo,
        # dividido igualmente entre os folds -- ver acima). Como
        # dias_treino VARIA por fold (janela expansiva: cresce a cada
        # fold), o número de passadas médias por dia de treino que esse
        # mesmo orçamento compra também varia -- é isso que
        # n_passadas_medias documenta abaixo:
        #   n_passadas_medias = total_timesteps / (dias_treino × ticks_por_dia)
        dias_treino = len(train_orders)
        total_timesteps_fold = int(timesteps_budget_per_fold)
        n_passadas_medias = total_timesteps_fold / (dias_treino * avg_ticks_per_day)

        folds.append({
            "fold": i,
            "train_orders": train_orders,
            "val_orders": val_orders,
            "test_orders": test_orders,
            "total_timesteps": total_timesteps_fold,
            "n_passadas_medias": n_passadas_medias,
        })

    if verbose:
        print(f"=== Folds walk-forward (TimeSeriesSplit, n_folds={n_folds}) ===")
        print(f"Throughput assumido : {throughput_steps_per_sec:,.0f} steps/s  "
              f"(ajuste para a máquina onde for treinar)")
        print(f"Orçamento total     : {time_budget_hours:.1f}h somadas entre os {n_folds} folds  ->  "
              f"{timesteps_budget_per_fold:,.0f} timesteps/fold "
              f"(~{time_budget_hours / n_folds:.2f}h/fold)")
        print(f"Ticks médios/pregão : {avg_ticks_per_day:,.0f}")
        print(f"Pregões disponíveis : {len(all_orders)}  |  pregões usados: {len(orders_used)} "
              f"(limitado pelo orçamento de tempo)")
        for f in folds:
            tr, va, te = f["train_orders"], f["val_orders"], f["test_orders"]
            assert tr[-1] < va[0] <= va[-1] < te[0] <= te[-1], \
                "sobreposição/ordem cronológica violada entre treino/validação/teste"
            print(f"  Fold {f['fold']}: treino=[{tr[0]:>3}..{tr[-1]:>3}] ({len(tr):>3} dias)  "
                  f"validação=[{va[0]:>3}..{va[-1]:>3}] ({len(va):>3} dias)  "
                  f"teste=[{te[0]:>3}..{te[-1]:>3}] ({len(te):>3} dias)  "
                  f"total_timesteps={f['total_timesteps']:,} "
                  f"(~{f['n_passadas_medias']:.1f} passadas/dia de treino)")

    return folds


# --------------------------------------------------------------------------
# 5) TREINO (PPO)
# --------------------------------------------------------------------------

def train_ppo(train_orders, val_orders, n_ticks=120, total_timesteps=5000, #300_000,
              model_path="ppo_arbitrage.zip", best_model_dir="./best_model",
              max_seconds=None, resume_from=None):
    # a normalização é ajustada SOMENTE com os dias de treino
    train_dfs_feats = []
    for order in train_orders:
        df = process_day(order)
        train_dfs_feats.append(build_feature_matrix(df))
    mean, std = fit_feature_scaler(train_dfs_feats)

    # treino: um processo por núcleo lógico, cada um reprocessando os dias
    # de treino a partir dos identificadores (orders) -- só orders/mean/std
    # (leves) atravessam o pickle para os processos-filho, não os
    # DataFrames/arrays já montados.
    n_train_envs = _detect_n_train_envs()
    train_env_fns = [
        partial(_make_multiday_env, train_orders, mean, std, n_ticks)
        for _ in range(n_train_envs)
    ]
    vec_train_env = SubprocVecEnv(train_env_fns)

    # validação: continua single-processo (EvalCallback roda com pouca
    # frequência e precisa de n_eval_episodes == len(val_orders))
    val_envs = build_day_envs(val_orders, mean, std, n_ticks=n_ticks)

    def make_val_env():
        return Monitor(MultiDayEnv(val_envs))

    vec_val_env = DummyVecEnv([make_val_env])

    # n_steps: cada episódio é 1 pregão inteiro (~27.700 ticks medidos nos
    # pregões de amostra da tarefa 4, na mesma ordem de grandeza dos
    # ~35.000 ticks/pregão típicos). Com n_steps=2048 (valor anterior), a
    # cada atualização o agente via só 2048/27700 ≈ 7% de um pregão por
    # env -- fatia pequena demais para o rollout capturar trechos
    # representativos de um pregão inteiro (ex.: dinâmica de fechamento
    # forçado no fim do dia). Subindo para n_steps=8192, cada rollout
    # cobre 8192/27700 ≈ 30% de um pregão por env -- trecho bem mais
    # representativo, sem ir até o episódio inteiro (o que multiplicaria
    # por ~13x o tamanho do buffer e o tempo por atualização).
    #
    # batch_size: precisa dividir (n_steps × n_train_envs) -- mas
    # n_train_envs = os.cpu_count() varia por máquina (tarefa 3), então
    # escolhemos batch_size como divisor do PRÓPRIO n_steps (8192 = 2^13):
    # qualquer divisor de n_steps também divide n_steps × n_train_envs
    # para QUALQUER número de processos, sem depender da máquina.
    # batch_size=1024 (8192/1024=8) dá um número de minibatches por época
    # razoável (ex.: 8 processos -> buffer=65536 -> 64 minibatches/época).
    n_steps = 8192
    batch_size = 1024
    assert n_steps % batch_size == 0, "batch_size deve dividir n_steps (e portanto n_steps × n_train_envs)"

    # resume_from: continua o treino de um checkpoint já salvo (de um job
    # anterior, ver TREINO EM CHUNKS abaixo) em vez de começar do zero --
    # os hiperparâmetros (learning_rate, n_steps, etc.) já vêm salvos no
    # checkpoint, não são reaplicados aqui.
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Retomando treino a partir de '{resume_from}'.")
        model = PPO.load(resume_from, env=vec_train_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_train_env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=0.999,        # horizonte longo: episódio = um pregão inteiro
            verbose=1,
        )

    # eval_freq: no EvalCallback do SB3, `eval_freq` é contado em chamadas
    # de callback (uma por env.step() do VecEnv), NÃO em timesteps totais
    # -- cada chamada avança n_train_envs timesteps de uma vez (todos os
    # processos em paralelo dão 1 passo por chamada). Ou seja, o número
    # real de timesteps entre avaliações é eval_freq × n_train_envs.
    # Queremos ~15-20 avaliações ao longo do treino inteiro (nem mais,
    # que gasta tempo repetindo avaliação sem ganho, nem menos, que
    # deixa a escolha de "melhor modelo" grosseira). Isolando eval_freq:
    #   total_timesteps ≈ eval_freq × n_train_envs × n_avaliações
    #   eval_freq ≈ total_timesteps / (n_avaliações × n_train_envs)
    n_avaliacoes_alvo = 18  # meio do intervalo 15-20 pedido
    eval_freq = max(1, int(total_timesteps / (n_avaliacoes_alvo * n_train_envs)))

    # Nota (treino em chunks/resume): best_mean_reward começa em -inf a
    # cada chamada desta função, então best_model_dir pode ser
    # sobrescrito por um modelo deste chunk pior que o melhor de um
    # chunk anterior. Por isso o resume_from acima sempre usa model_path
    # (checkpoint "cru" salvo ao fim de cada chunk), nunca best_model_dir.
    eval_callback = EvalCallback(
        vec_val_env, best_model_save_path=best_model_dir,
        eval_freq=eval_freq,
        n_eval_episodes=len(val_orders),  # 1 episódio por dia de validação do fold atual
        deterministic=True,
    )

    callbacks = [eval_callback]
    if max_seconds is not None:
        # max_seconds é contado a partir de SCRIPT_START (início do
        # processo Python), não do início deste learn() -- ver docstring
        # de TimeLimitCallback.
        callbacks.append(TimeLimitCallback(SCRIPT_START + max_seconds, verbose=1))

    # reset_num_timesteps=False ao retomar: NÃO zera o contador interno de
    # timesteps do SB3, então total_timesteps aqui passa a significar
    # "treinar total_timesteps A MAIS a partir de onde parou" (semântica
    # do próprio SB3 quando reset_num_timesteps=False). Cada chunk só
    # consegue treinar uma fração pequena disso no tempo real disponível
    # (ver TimeLimitCallback) -- não precisa descontar steps já feitos
    # manualmente, o corte real é por tempo de parede, não por contagem.
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks),
                reset_num_timesteps=(resume_from is None))
    model.save(model_path)

    return model, (mean, std)


# --------------------------------------------------------------------------
# 6) AVALIAÇÃO -- roda a política determinística nos dias de teste
# --------------------------------------------------------------------------

def evaluate_policy(model, orders, mean, std, n_ticks=120):
    results = []
    all_trade_pnls = []  # P&L de TODOS os negócios, de todos os dias, para o diagnóstico

    for order in orders:
        df = process_day(order)
        feat = apply_scaler(build_feature_matrix(df), mean, std)
        env = ArbitrageTradingEnv(df, feat, n_ticks=n_ticks)

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        results.append({
            "order": order,
            "lucro_total": total_reward,
            "negocios_fechados": info["n_trades_closed"],
            "negocios_vencedores": info["n_trades_won"],
        })
        all_trade_pnls.extend(env.trade_pnls)

    results_df = pd.DataFrame(results)
    print(results_df)
    lucro_total_pontos = results_df['lucro_total'].sum()
    print(f"\nLucro total agregado: {lucro_total_pontos:.2f} pontos "
          f"(R$ {lucro_total_pontos * POINT_VALUE_BRL:.2f})")
    print(f"Negócios fechados: {results_df['negocios_fechados'].sum()}")
    print(f"Taxa de acerto: "
          f"{results_df['negocios_vencedores'].sum() / max(results_df['negocios_fechados'].sum(), 1):.2%}")

    # --- Diagnóstico de distribuição de P&L por negócio (já líquido de custos) ---
    pnls = np.array(all_trade_pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    print("\n=== Distribuição de P&L por negócio (líquida de custos) ===")
    print(f"Total de negócios: {len(pnls)}  |  vencedores: {len(wins)}  |  perdedores: {len(losses)}")
    if len(wins) > 0:
        print(f"Ganho médio       : {wins.mean():.2f}   |  mediana: {np.median(wins):.2f}   |  máximo: {wins.max():.2f}")
    if len(losses) > 0:
        print(f"Perda média       : {losses.mean():.2f}   |  mediana: {np.median(losses):.2f}   |  mínimo: {losses.min():.2f}")
    if len(wins) > 0 and len(losses) > 0:
        razao = abs(losses.mean()) / wins.mean()
        print(f"Perda média / Ganho médio: {razao:.2f}x")
    # percentis extremos ajudam a enxergar a "cauda" de perdas grandes
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  percentil {p:>2}: {np.percentile(pnls, p):.2f}")

    return results_df, pnls


# --------------------------------------------------------------------------
# EXECUÇÃO
# --------------------------------------------------------------------------

if __name__ == "__main__":
    all_orders = list(range(1, 481))  # 480 pregões disponíveis, em ordem cronológica

    N_FOLDS = 3  # configurável: quantos folds walk-forward gerar

    folds = generate_walk_forward_folds(all_orders, n_folds=N_FOLDS)

    # Sob Slurm job array (--array=1-N_FOLDS), cada task treina só o fold
    # correspondente ao seu índice, em vez de rodar os N_FOLDS em
    # sequência no mesmo job -- assim os folds treinam em paralelo, um
    # por nó. Em execução local (sem Slurm), a env var não existe e o
    # comportamento antigo (todos os folds em sequência) é preservado.
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    folds_to_run = [f for f in folds if f["fold"] == int(array_task_id)] if array_task_id else folds

    # Margem de segurança sob o --time do job Slurm (ver TimeLimitCallback);
    # setado pelo .sbatch. Sem essa env var (execução local), sem limite.
    max_seconds = os.environ.get("TRAIN_MAX_SECONDS")
    max_seconds = float(max_seconds) if max_seconds else None

    # Deadline duro do job inteiro (contado a partir de SCRIPT_START,
    # igual ao TimeLimitCallback). O evaluate_policy() de validação/teste
    # roda DEPOIS do model.learn() já ter parado e salvo o modelo -- sem
    # essa checagem, um smoke test real mostrou que o Slurm mata o job no
    # meio dessa avaliação (o modelo fica salvo, mas o relatório de
    # validação/teste se perde). Se não sobrar reserva suficiente, PULA a
    # avaliação em vez de arriscar ser matado no meio -- o checkpoint já
    # está salvo, então dá pra rodar evaluate_policy() depois carregando
    # ele, sem precisar retreinar.
    job_hard_seconds = os.environ.get("JOB_HARD_SECONDS")
    job_hard_deadline = (SCRIPT_START + float(job_hard_seconds)) if job_hard_seconds else None
    EVAL_RESERVE_SECONDS = 150  # reserva conservadora por chamada de evaluate_policy

    def run_eval_if_time_allows(fold_num, label, orders, model, mean, std):
        if job_hard_deadline is not None:
            remaining = job_hard_deadline - time.time()
            if remaining < EVAL_RESERVE_SECONDS:
                print(f"\n=== Fold {fold_num} -- Avaliação em {label} PULADA "
                      f"({remaining:.0f}s restantes, menos que a reserva de "
                      f"{EVAL_RESERVE_SECONDS}s) -- o checkpoint já está "
                      f"salvo; rode evaluate_policy() depois carregando-o. ===")
                return
        print(f"\n=== Fold {fold_num} -- Avaliação em {label} ===")
        evaluate_policy(model, orders, mean, std)

    # Treino em chunks: RESUME_TRAINING=1 retoma do checkpoint do fold
    # (job anterior na mesma cadeia) em vez de treinar do zero;
    # FINAL_CHUNK=0 pula a avaliação de val/teste (ainda vai treinar mais
    # depois). Sem essas env vars (execução local ou job único), o
    # default preserva o comportamento de sempre: treina do zero e avalia
    # ao final -- ver slurm/submit_all_folds.sh pra como isso é orquestrado.
    resume_training = os.environ.get("RESUME_TRAINING", "0") == "1"
    final_chunk = os.environ.get("FINAL_CHUNK", "1") == "1"

    for f in folds_to_run:
        print(f"\n########## FOLD {f['fold']}/{len(folds)} ##########")
        model_path = f"ppo_arbitrage_fold{f['fold']}.zip"
        model, (mean, std) = train_ppo(
            f["train_orders"], f["val_orders"],
            total_timesteps=f["total_timesteps"],
            model_path=model_path,
            best_model_dir=f"./best_model_fold{f['fold']}",
            max_seconds=max_seconds,
            resume_from=(model_path if resume_training else None),
        )

        if final_chunk:
            run_eval_if_time_allows(f["fold"], "VALIDAÇÃO", f["val_orders"], model, mean, std)
            run_eval_if_time_allows(f["fold"], "TESTE", f["test_orders"], model, mean, std)
        else:
            print(f"\n=== Fold {f['fold']} -- chunk intermediário, avaliação "
                  f"fica pro chunk final (FINAL_CHUNK=1) ===")
