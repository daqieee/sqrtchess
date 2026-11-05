"""神经引导的 Alpha-Beta：保留全宽搜索框架，把叶子评估换成「网络价值 + 手工评估」。

motivation：这个棋种战术密度高，「全宽深度 2 + 吃子静态搜索」明显强于
「网络 + 少量模拟的窄采样 MCTS」。于是把两者的长处合起来：

* **搜索**用 Alpha-Beta（迭代加深、置换表、杀手/历史、SEE 剪枝）——保证战术不漏；
* **叶子评估**用「手工评估 + net_scale × 网络价值」——保证局面判断来自学习。

网络是逐步变强的：训练初期网络接近随机（价值≈0），此时搜索完全由手工评估驱动，
等价于原来的 Alpha-Beta；随着训练，网络价值权重逐渐体现出来。
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Optional, Tuple

from sqrtchess import Game, Move
from sqrtchess.alphabeta import MATE_SCORE, Searcher, evaluate

from .network import AZNetwork

__all__ = ["NeuralSearcher", "HybridPlayer", "NET_SCALE"]

#: 网络价值 ±1 相当于多少「子力分」
NET_SCALE = 30.0


class NeuralSearcher(Searcher):
    """叶子评估 = 手工评估 + net_scale × 网络价值。"""

    def __init__(self, network: AZNetwork, depth: int = 2, time_limit: Optional[float] = None,
                 net_scale: float = NET_SCALE, node_limit: Optional[int] = None,
                 tt_size: int = 300000, seed: Optional[int] = None, use_see: bool = True):
        super().__init__(depth=depth, time_limit=time_limit, node_limit=node_limit,
                         tt_size=tt_size, seed=seed, use_see=use_see)
        self.network = network
        self.net_scale = float(net_scale)
        self.net_calls = 0

    def _evaluate(self, game: Game, color: Optional[int] = None) -> float:
        side = game.side if color is None else color
        base = evaluate(game, side)
        if self.net_scale == 0.0 or game.is_over:
            return base
        # 网络价值是「行棋方视角」；当 color 与行棋方不同时取反
        _, value = self.network.evaluate(game)
        self.net_calls += 1
        if side != game.side:
            value = -value
        return base + self.net_scale * value

    def score_moves(self, game: Game, depth: int = 1) -> List[Tuple[float, Move]]:
        """给全部合法着法打分（用于策略软目标）。"""
        self.net_calls = 0
        return super().score_moves(game, depth=depth)

    def search(self, game: Game, depth: Optional[int] = None,
               time_limit: Optional[float] = None):
        self.net_calls = 0
        return super().search(game, depth=depth, time_limit=time_limit)


class HybridPlayer:
    """把「神经引导 Alpha-Beta」包装成统一的选手接口（供竞技场/界面使用）。"""

    name = "神经Alpha-Beta"

    def __init__(self, network: AZNetwork, depth: int = 2, net_scale: float = NET_SCALE,
                 time_limit: Optional[float] = None, seed: Optional[int] = None,
                 temperature: float = 0.0):
        self.network = network
        self.depth = int(depth)
        self.net_scale = float(net_scale)
        self.time_limit = time_limit
        self.temperature = float(temperature)
        self.searcher = NeuralSearcher(network, depth=depth, net_scale=net_scale,
                                       time_limit=time_limit, seed=seed)
        self.random = random.Random(seed)
        self.last = None

    def choose(self, game: Game) -> Optional[Move]:
        if game.is_over:
            return None
        result = self.searcher.search(game, depth=self.depth, time_limit=self.time_limit)
        self.last = result
        return result.move

    def rank_moves(self, game: Game, depth: Optional[int] = None):
        return self.searcher.score_moves(game, depth=depth or max(1, self.depth - 1))

    def describe(self) -> str:
        extra = "，限时 %.2fs" % self.time_limit if self.time_limit else ""
        return "%s（深度 %d，网络权重 %.0f%s）" % (self.name, self.depth, self.net_scale, extra)

    def __repr__(self) -> str:
        return "<HybridPlayer depth=%d>" % self.depth

def play_hybrid_game(network: AZNetwork, depth: int = 1, net_scale: float = NET_SCALE,
                     max_plies: int = 400, temp_plies: int = 16, temperature: float = 8.0,
                     limit=None, seed: Optional[int] = None, keep_record: bool = False,
                     draw_no_moves_value: Optional[float] = None):
    """用「神经引导 Alpha-Beta」自我对弈，生成 (局面, pi, z) 训练数据。

    pi 来自根节点全部着法的搜索分值（softmax，前 temp_plies 步带温度采样，之后取最优一步）。
    深度 1 时每步只要约 0.015 秒，可以放心地跑很多局。
    """
    from sqrtchess import DEFAULT_NO_CAPTURE_LIMIT, Recorder, default_headers

    from .selfplay import Sample, finalize
    from .values import DRAW_NO_MOVES_VALUE

    limit = DEFAULT_NO_CAPTURE_LIMIT if limit is None else limit
    draw_value = DRAW_NO_MOVES_VALUE if draw_no_moves_value is None else draw_no_moves_value
    started = time.time()
    game = Game.new(no_capture_limit=limit)
    recorder = Recorder(game, default_headers(white="HybridAB", black="HybridAB",
                                              event="神经 Alpha-Beta 自我对弈"))
    searcher = NeuralSearcher(network, depth=depth, net_scale=net_scale, seed=seed)
    rng = random.Random(seed)
    samples: List = []

    while not game.is_over and game.ply < max_plies:
        scored = searcher.score_moves(game, depth=depth)
        if not scored:
            break
        best_score = scored[0][0]
        if game.ply < temp_plies:
            tau = max(1e-6, temperature)
            weights = [math.exp((score - best_score) / tau) for score, _ in scored]
            total = sum(weights) or 1.0
            probs = [weight / total for weight in weights]
        else:
            probs = [1.0] + [0.0] * (len(scored) - 1)
        entries = []
        for (score, move), probability in zip(scored, probs):
            if probability > 1e-9:
                entries.append((game.move_dict(move)["notation"], probability))
        samples.append(Sample(fen=game.fen(), policy=entries,
                              side=game.status(False)["turn"], ply=game.ply))
        threshold = rng.random()
        acc = 0.0
        chosen = scored[0][1]
        for (score, move), probability in zip(scored, probs):
            acc += probability
            if acc >= threshold:
                chosen = move
                break
        recorder.push(chosen)

    return finalize(game, samples, recorder, started, keep_record=keep_record,
                    draw_no_moves_value=draw_value)
