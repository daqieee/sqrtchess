"""AlphaZero 的 PUCT 蒙特卡洛树搜索。

保留论文里的核心要素：

* 每个节点保存先验概率 P、访问次数 N、累计价值 W；
* 选择用 PUCT：a = argmax(-Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)))；
* 叶子节点用神经网络评估，得到的价值反向传播（逐层变号）；
* 自我对弈时在根节点加 Dirichlet 噪声做探索；
* 搜索结束后用访问次数分布 N^(1/T) 作为策略目标 pi。

为节省时间，子节点是「按需创建」的：扩展时只记录着法与先验，真正被选中时才复制局面。
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sqrtchess import BLACK, WHITE, Game, Move

from .network import AZNetwork
from .values import DRAW_NO_MOVES_VALUE, terminal_value

__all__ = ["Node", "MCTS", "material_term", "material_score", "quiesce"]

#: 物质项归一化尺度：大小和之差除以该值后过 tanh
MATERIAL_SCALE = 40.0
#: 吃子静态搜索里「终局」的分值（会被 tanh 压到 ±1）
WIN_SCORE = 10000.0


def material_score(game: Game, color=None) -> float:
    """物质分（默认当前行棋方视角）：大小和之差 + 3 × 棋子数之差。"""
    if color is None:
        color = game.side
    white_count, white_size = game.piece_stats(WHITE)
    black_count, black_size = game.piece_stats(BLACK)
    if color > 0:
        return (white_size - black_size) + 3.0 * (white_count - black_count)
    return (black_size - white_size) + 3.0 * (black_count - white_count)


def _capture_moves(game: Game):
    """按被吃子大小从大到小排序的吃子着法。"""
    found = []
    for move in game.legal_moves():
        value = game.cells[move.frm]
        target = game.cells[move.to]
        if target != 0 and (target > 0) != (value > 0):
            found.append((abs(target), move))
    found.sort(key=lambda item: -item[0])
    return [move for _, move in found]


def quiesce(game: Game, depth: int, alpha: float = -float("inf"),
            beta: float = float("inf")) -> float:
    """吃子静态搜索（alpha-beta），返回行棋方视角的物质分。

    这是「搜索深度 +1~2 层」的实现：MCTS 到达叶子时，先用它把接下来的吃子交换算清楚，
    再叠加网络价值，从而补上网络看不到的战术。
    """
    color = game.side
    if game.is_over:
        result = game.result
        if result is None or result.winner is None:
            return 0.0
        return WIN_SCORE if result.winner == color else -WIN_SCORE
    stand = material_score(game, color)
    if depth <= 0:
        return stand
    captures = _capture_moves(game)
    if not captures:
        return stand
    best = stand
    for move in captures:
        game.push(move)
        score = -quiesce(game, depth - 1, -beta, -alpha)
        game.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def material_term(game: Game) -> float:
    """局面物质差（当前行棋方视角），归一化到 [-1, 1]。

    这个游戏里「棋子数量与大小总和」既决定谁能吃光对方，也直接决定规则 7 的判定，
    所以把物质差作为搜索叶子评价的先验项，等于把「尽可能多吃子」的基线直觉
    注入到 AlphaZero 的搜索里；网络价值仍然负责长期判断。
    """
    white_count, white_size = game.piece_stats(WHITE)
    black_count, black_size = game.piece_stats(BLACK)
    if game.side > 0:
        diff = (white_size - black_size) + 3.0 * (white_count - black_count)
    else:
        diff = (black_size - white_size) + 3.0 * (black_count - white_count)
    return math.tanh(diff / MATERIAL_SCALE)


class Node:
    """搜索树节点。"""

    __slots__ = ("game", "parent", "move", "prior", "children", "priors", "N", "W",
                 "expanded")

    def __init__(self, game: Game, parent: Optional["Node"] = None,
                 move: Optional[Move] = None, prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: Dict[Move, "Node"] = {}
        self.priors: Dict[Move, float] = {}
        self.N = 0
        self.W = 0.0
        self.expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

    def terminal_value(self, draw_no_moves_value: float = DRAW_NO_MOVES_VALUE) -> float:
        """终局价值（当前行棋方视角，与训练标签同一套口径）。"""
        return terminal_value(self.game, draw_no_moves_value)

    def expand(self, moves: Sequence[Move], priors: Sequence[float]) -> None:
        self.priors = {move: float(p) for move, p in zip(moves, priors)}
        self.expanded = True

    def child(self, move: Move) -> "Node":
        """取（必要时创建）某个着法对应的子节点。"""
        node = self.children.get(move)
        if node is None:
            game = self.game.copy()
            game.push(move)
            node = Node(game, parent=self, move=move, prior=self.priors.get(move, 0.0))
            self.children[move] = node
        return node

    def visit_counts(self) -> Dict[Move, int]:
        return {move: node.N for move, node in self.children.items()}

    def __repr__(self) -> str:
        return "<Node N=%d Q=%.3f children=%d>" % (self.N, self.Q, len(self.children))


class MCTS:
    """PUCT 搜索器。"""

    def __init__(self, network: AZNetwork, sims: int = 60, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 seed: Optional[int] = None, material_weight: float = 0.45,
                 draw_no_moves_value: float = DRAW_NO_MOVES_VALUE,
                 leaf_depth: int = 1):
        self.network = network
        #: 叶子价值里「物质项」的权重（0 表示纯网络价值）
        self.material_weight = float(material_weight)
        #: 叶子评估前额外做的吃子搜索层数（0 = 只用静态物质差，1~3 = 多算 1~3 层吃子交换）
        self.leaf_depth = max(0, int(leaf_depth))
        #: 「一方无法动子」判和时的分值（与训练标签保持一致）
        self.draw_no_moves_value = float(draw_no_moves_value)
        self.sims = int(sims)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.random = random.Random(seed)
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0

    # ------------------------------------------------------------- 内部

    def _evaluate(self, node: Node) -> float:
        """用网络评估叶子节点，写入先验并返回值（行棋方视角）。"""
        self.evaluations += 1
        game = node.game
        moves = game.legal_moves()
        if not moves:
            return node.terminal_value()
        priors, value = self.network.evaluate(game)
        if self.material_weight:
            if self.leaf_depth > 0:
                raw = quiesce(game, self.leaf_depth)
            else:
                raw = material_score(game)
            value += self.material_weight * math.tanh(raw / MATERIAL_SCALE)
            value = max(-1.0, min(1.0, value))
        node.expand(moves, priors)
        return value

    def _select(self, node: Node) -> Node:
        sqrt_parent = math.sqrt(max(1, node.N))
        best_score = -float("inf")
        best_move: Optional[Move] = None
        for move, prior in node.priors.items():
            child = node.children.get(move)
            q = child.Q if child is not None else 0.0
            n = child.N if child is not None else 0
            score = -q + self.c_puct * prior * sqrt_parent / (1.0 + n)
            if score > best_score:
                best_score = score
                best_move = move
        if best_move is None:
            best_move = next(iter(node.priors))
        return node.child(best_move)

    def _simulate(self, root: Node) -> None:
        node = root
        path: List[Node] = [node]
        while True:
            if node.game.is_over:
                value = node.terminal_value(self.draw_no_moves_value)
                break
            if not node.expanded:
                value = self._evaluate(node)
                break
            node = self._select(node)
            path.append(node)
        for item in reversed(path):
            item.N += 1
            item.W += value
            value = -value

    def add_dirichlet_noise(self, root: Node, alpha: Optional[float] = None,
                            eps: Optional[float] = None) -> None:
        moves = list(root.priors)
        if not moves:
            return
        alpha = self.dirichlet_alpha if alpha is None else alpha
        eps = self.dirichlet_eps if eps is None else eps
        noise = self.rng.dirichlet([alpha] * len(moves))
        for move, sample in zip(moves, noise):
            root.priors[move] = (1.0 - eps) * root.priors[move] + eps * float(sample)

    # ------------------------------------------------------------- 接口

    def search(self, game: Game, add_noise: bool = False, sims: Optional[int] = None) -> Node:
        """搜索并返回根节点。"""
        self.evaluations = 0
        root = Node(game.copy())
        if root.game.is_over:
            return root
        self._evaluate(root)
        if add_noise:
            self.add_dirichlet_noise(root)
        for _ in range(int(sims if sims is not None else self.sims)):
            self._simulate(root)
        return root

    @staticmethod
    def visit_policy(root: Node, temperature: float = 1.0) -> Dict[Move, float]:
        """把访问次数转成策略分布（温度 <= 0 时取访问最多的一步）。"""
        counts = {}
        for move in root.priors:
            child = root.children.get(move)
            counts[move] = child.N if child is not None else 0
        if not counts:
            return {}
        if temperature is None or temperature <= 1e-3:
            best = max(counts.values())
            winners = [move for move, count in counts.items() if count == best]
            chosen = winners[0]
            return {move: (1.0 if move == chosen else 0.0) for move in counts}
        weights = {move: (count ** (1.0 / temperature)) for move, count in counts.items()}
        total = sum(weights.values())
        if total <= 0:
            uniform = 1.0 / len(counts)
            return {move: uniform for move in counts}
        return {move: weight / total for move, weight in weights.items()}

    def policy(self, game: Game, temperature: float = 0.0, add_noise: bool = False,
               sims: Optional[int] = None) -> Tuple[Dict[Move, float], Node]:
        """便捷入口：搜索并返回 (着法 -> 概率, 根节点)。"""
        root = self.search(game, add_noise=add_noise, sims=sims)
        return self.visit_policy(root, temperature), root

    def choose(self, game: Game, temperature: float = 0.0,
               sims: Optional[int] = None) -> Optional[Move]:
        """按访问次数选一步棋。"""
        policy, _ = self.policy(game, temperature=temperature, sims=sims)
        if not policy:
            return None
        moves = list(policy)
        if temperature and temperature > 1e-3:
            total = sum(policy[move] for move in moves)
            if total <= 0:
                return self.random.choice(moves)
            threshold = self.random.random() * total
            acc = 0.0
            for move in moves:
                acc += policy[move]
                if acc >= threshold:
                    return move
            return moves[-1]
        best = max(policy.values())
        winners = [move for move in moves if policy[move] == best]
        return self.random.choice(winners)
