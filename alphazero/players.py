"""统一的对弈接口：把网络、基线算法、内置 AI 都包装成 choose(game) -> Move。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqrtchess import BaseAI, CaptureAI, Game, Move, make_ai

from .mcts import MCTS
from .network import AZNetwork

__all__ = ["MCTSPlayer", "AlphaZeroPlayer", "CapturePlayer", "load_player", "PLAYER_HELP"]


class MCTSPlayer:
    """用「网络 + PUCT 搜索」下棋的选手。"""

    name = "AlphaZero"

    def __init__(self, network: AZNetwork, sims: int = 100, temperature: float = 0.0,
                 c_puct: float = 1.5, seed: Optional[int] = None, noise: bool = False,
                 material_weight: float = 0.45, leaf_depth: int = 1):
        self.network = network
        self.sims = int(sims)
        self.temperature = float(temperature)
        self.noise = bool(noise)
        self.material_weight = float(material_weight)
        self.leaf_depth = int(leaf_depth)
        self.search = MCTS(network, sims=sims, c_puct=c_puct, seed=seed,
                           material_weight=material_weight, leaf_depth=leaf_depth)

    def choose(self, game: Game) -> Optional[Move]:
        if game.is_over:
            return None
        return self.search.choose(game, temperature=self.temperature)

    def describe(self) -> str:
        return "%s（%d 次模拟%s）" % (self.name, self.sims,
                                    "，带探索噪声" if self.noise else "")

    def __repr__(self) -> str:
        return "<MCTSPlayer sims=%d>" % self.sims


class AlphaZeroPlayer(MCTSPlayer):
    """从文件加载权重的 AlphaZero 选手。"""

    def __init__(self, model_path, sims: int = 100, temperature: float = 0.0,
                 c_puct: float = 1.5, seed: Optional[int] = None, noise: bool = False,
                 material_weight: float = 0.45, leaf_depth: int = 1):
        self.model_path = str(model_path)
        network = AZNetwork.load(self.model_path)
        super().__init__(network, sims=sims, temperature=temperature, c_puct=c_puct,
                         seed=seed, noise=noise, material_weight=material_weight,
                         leaf_depth=leaf_depth)
        self.name = "AlphaZero(%s)" % Path(self.model_path).stem


class CapturePlayer(CaptureAI):
    """「尽可能多吃子」基线选手（就是引擎里的 CaptureAI）。"""

    def describe(self) -> str:
        return self.name + "（深度 %d）" % self.depth


PLAYER_HELP = ("选手写法：abN（Alpha-Beta 引擎，深度 N，如 ab2 / ab4）、"
               "baseline/capture（吃子基线）、random、greedy、search2、search3、"
               "或一个 .json 网络权重文件路径")


def load_player(spec, sims: int = 100, temperature: float = 0.0, seed: Optional[int] = None,
                noise: bool = False, depth: int = 4, material_weight: float = 0.45,
                leaf_depth: int = 1, search_kind: str = "mcts", ab_depth: int = 2,
                net_scale: float = 30.0, time_limit: Optional[float] = None):
    """按名字或权重文件创建选手。"""
    if spec is None:
        return None
    if isinstance(spec, (BaseAI, MCTSPlayer)):
        return spec
    text = str(spec).strip()
    if text in ("capture", "baseline", "吃子", "基线"):
        return CapturePlayer(depth=depth, seed=seed)
    path = Path(text)
    if path.exists():
        if search_kind == "hybrid":
            from .hybrid import HybridPlayer

            return HybridPlayer(AZNetwork.load(path), depth=ab_depth, net_scale=net_scale,
                                time_limit=time_limit, seed=seed)
        return AlphaZeroPlayer(path, sims=sims, temperature=temperature, seed=seed,
                               noise=noise, material_weight=material_weight,
                               leaf_depth=leaf_depth)
    # 其余交给引擎的 make_ai：random / greedy / search2 / search3 / ab / abN
    try:
        return make_ai(text.lower(), seed=seed, depth=depth)
    except ValueError:
        raise ValueError("无法识别的选手 %r。%s" % (spec, PLAYER_HELP))
