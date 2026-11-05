"""根号棋 AlphaZero 自我对弈系统（简化版）。

核心要素（对应 AlphaZero 论文）：

* 策略/价值网络（numpy 手写前向与反向）：alphazero.network.AZNetwork
* PUCT 蒙特卡洛树搜索：alphazero.mcts.MCTS
* 自我对弈生成 (局面, pi, z) 数据：alphazero.selfplay
* 训练循环 + 门控评估 + 与「尽可能多吃子」基线的锚点对战：alphazero.trainer
* 统一选手接口（可给引擎/GUI 用）：alphazero.players

    python -m alphazero train --iterations 8 --games 40 --sims 60 --workers 8 --out runs/az1
"""

from .arena import MatchResult, play_match
from .encoding import PLANES, encode
from .mcts import MCTS, Node
from .network import AZNetwork
from .players import AlphaZeroPlayer, CapturePlayer, MCTSPlayer, load_player
from .selfplay import GameData, Sample, play_baseline_game, play_game
from .trainer import Trainer, TrainConfig

__all__ = ["AZNetwork", "encode", "PLANES", "MCTS", "Node", "Sample", "GameData",
           "play_game", "play_baseline_game", "Trainer", "TrainConfig",
           "play_match", "MatchResult", "MCTSPlayer", "AlphaZeroPlayer",
           "CapturePlayer", "load_player"]
