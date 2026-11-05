"""根号棋的简易 AI：随机、贪心与浅层搜索。

这些 AI 既是给人类玩家的对手，也是外部 AI 的参考实现（见 docs/PROTOCOL.md
中的 AI 应答协议，任何程序只要按该协议输出着法就能接入引擎和图形界面）。
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

from .core import BLACK, Game, Move, WHITE

__all__ = ["BaseAI", "RandomAI", "GreedyAI", "SearchAI", "CaptureAI", "make_ai", "choose_move",
           "AI_LEVELS", "AI_LEVEL_NAMES", "evaluate"]

#: 供界面选择的标准 AI 档位
AI_LEVELS: Tuple[Tuple[str, str], ...] = (
    ("ab", "Alpha-Beta 引擎（深度 4）"),
    ("capture", "吃子基线（尽可能多吃子）"),
    ("random", "随机走子"),
    ("greedy", "贪心（吃大子、避免被吃）"),
    ("search2", "搜索 2 层"),
    ("search3", "搜索 3 层"),
)
AI_LEVEL_NAMES: Dict[str, str] = dict(AI_LEVELS)

WIN_SCORE = 100000.0


class BaseAI:
    """AI 基类。"""

    name = "AI"

    def choose(self, game: Game) -> Optional[Move]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "<%s>" % self.__class__.__name__


class RandomAI(BaseAI):
    """随机走子。"""

    name = "随机"

    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)

    def choose(self, game: Game) -> Optional[Move]:
        moves = game.legal_moves()
        if not moves:
            return None
        return self.random.choice(moves)


class GreedyAI(BaseAI):
    """贪心：优先吃大子，同时避免把自己的棋子送到对方口里。"""

    name = "贪心"

    def __init__(self, seed: Optional[int] = None, noise: float = 0.6):
        self.random = random.Random(seed)
        self.noise = noise

    @staticmethod
    def threat_after(game: Game, move: Move, mover: int) -> int:
        """走完这一步之后，对方能吃回该落点的最大棋子大小。

        注意：调用时该着法必须已经走在棋盘上，因此需要显式传入走子方。
        """
        worst = 0
        for sq, other in enumerate(game.cells):
            if other == 0 or (other > 0) == (mover > 0):
                continue  # 跳过自己的棋子
            for reply in game.moves_from(sq):
                if reply.to == move.to:
                    worst = max(worst, abs(other))
                    break
        return worst

    def score(self, game: Game, move: Move) -> float:
        value = game.cells[move.frm]
        size = abs(value)
        target = game.cells[move.to]
        captured = abs(target) if (target != 0 and (target > 0) != (value > 0)) else 0
        score = captured * 10.0 + (size + captured) * 0.4
        mover = WHITE if value > 0 else BLACK
        game.push(move)
        try:
            score -= GreedyAI.threat_after(game, move, mover) * 3.0
            # 走完这一步可能直接结束对局（吃光、无子可动或步数上限）
            if game.result is not None:
                if game.result.winner is None:
                    score -= 2.0
                elif game.result.winner == mover:
                    score += WIN_SCORE
                else:
                    score -= WIN_SCORE
        finally:
            game.pop()
        return score

    def choose(self, game: Game) -> Optional[Move]:
        moves = game.legal_moves()
        if not moves:
            return None
        scored = [(self.score(game, move), move) for move in moves]
        best = max(score for score, _ in scored)
        pool = [move for score, move in scored if score >= best - 1e-9]
        if self.noise:
            jittered = [(score + self.random.uniform(-self.noise, self.noise), move)
                        for score, move in scored]
            jittered.sort(key=lambda item: item[0], reverse=True)
            top = jittered[0][0]
            pool = [move for score, move in jittered if score >= top - 1e-9] or pool
        return self.random.choice(pool)


def evaluate(game: Game, color: int) -> float:
    """局面评估（站在 color 的角度）：大小总和差 + 棋子数量差。"""
    white_count, white_size = game.piece_stats(WHITE)
    black_count, black_size = game.piece_stats(BLACK)
    if color == WHITE:
        return (white_size - black_size) + 3.0 * (white_count - black_count)
    return (black_size - white_size) + 3.0 * (black_count - white_count)


class SearchAI(BaseAI):
    """带 alpha-beta 剪枝的浅层搜索。"""

    def __init__(self, depth: int = 2, seed: Optional[int] = None,
                 time_limit: float = 3.0, node_limit: int = 40000):
        self.depth = max(1, int(depth))
        self.random = random.Random(seed)
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.nodes = 0
        self.deadline = 0.0

    @property
    def name(self) -> str:
        return "搜索%d层" % self.depth

    # -------------------------------------------------- 搜索

    def _order(self, game: Game, moves: List[Move]) -> List[Move]:
        def key(move: Move):
            target = game.cells[move.to]
            value = game.cells[move.frm]
            captured = abs(target) if (target != 0 and (target > 0) != (value > 0)) else 0
            return (-captured, -abs(value))
        return sorted(moves, key=key)

    def _negamax(self, game: Game, depth: int, alpha: float, beta: float, color: int) -> float:
        self.nodes += 1
        if self.nodes > self.node_limit or (self.deadline and time.time() > self.deadline):
            return evaluate(game, color)
        if game.is_over:
            result = game.result
            if result is None or result.winner is None:
                return 0.0
            if result.winner == color:
                return WIN_SCORE - (10 - depth)
            return -WIN_SCORE + (10 - depth)
        if depth <= 0:
            return evaluate(game, color)
        moves = game.legal_moves()
        if not moves:
            return evaluate(game, color)
        best = -float("inf")
        for move in self._order(game, moves):
            game.push(move)
            value = -self._negamax(game, depth - 1, -beta, -alpha, -color)
            game.pop()
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def choose(self, game: Game) -> Optional[Move]:
        moves = game.legal_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        self.nodes = 0
        self.deadline = time.time() + self.time_limit if self.time_limit else 0.0
        color = game.side
        alpha = -float("inf")
        best_moves: List[Move] = []
        best = -float("inf")
        for move in self._order(game, moves):
            game.push(move)
            value = -self._negamax(game, self.depth - 1, -float("inf"), -alpha, -color)
            game.pop()
            if value > best + 1e-9:
                best = value
                best_moves = [move]
                if value > alpha:
                    alpha = value
            elif abs(value - best) <= 1e-9:
                best_moves.append(move)
        return self.random.choice(best_moves) if best_moves else moves[0]


def make_ai(spec="greedy", **kwargs) -> Optional[BaseAI]:
    """按名称创建 AI。

    接受 random / greedy / search / search3 / 中文名，或直接传入 BaseAI 实例。
    """
    if spec is None:
        return None
    if isinstance(spec, BaseAI):
        return spec
    text = str(spec).strip().lower()
    if text in ("none", "human", "人类", "无"):
        return None
    seed = kwargs.get("seed")
    time_limit = kwargs.get("time_limit") or 3.0
    depth = kwargs.get("depth")
    noise = kwargs.get("noise")
    if text in ("ab", "alphabeta", "alpha-beta", "negamax", "搜索"):
        from .alphabeta import AlphaBetaAI

        return AlphaBetaAI(depth=depth or 4, seed=seed,
                           time_limit=kwargs.get("time_limit"),
                           node_limit=kwargs.get("node_limit"))
    if text.startswith("ab") and text[2:].isdigit():
        from .alphabeta import AlphaBetaAI

        return AlphaBetaAI(depth=int(text[2:]), seed=seed,
                           time_limit=kwargs.get("time_limit"))
    if text in ("capture", "eat", "吃子", "吃子基线"):
        return CaptureAI(depth=depth or 4, seed=seed, noise=noise or 0.0)
    if text in ("random", "rand", "随机"):
        return RandomAI(seed=seed)
    if text in ("greedy", "greedy1", "贪心"):
        return GreedyAI(seed=seed, noise=0.6 if noise is None else noise)
    if text in ("search", "search2", "搜索", "搜索2"):
        return SearchAI(depth=depth or 2, seed=seed, time_limit=time_limit)
    if text in ("search3", "搜索3"):
        return SearchAI(depth=depth or 3, seed=seed, time_limit=time_limit)
    if text.startswith("search") and text[6:].isdigit():
        return SearchAI(depth=int(text[6:]), seed=seed, time_limit=time_limit)
    raise ValueError("未知的 AI 档位: %r（可用: random / greedy / search2 / search3）" % (spec,))


def choose_move(game: Game, level="greedy", **kwargs) -> Optional[Move]:
    """便捷函数：让指定档位的 AI 选一步棋。"""
    engine = make_ai(level, **kwargs)
    if engine is None:
        return None
    return engine.choose(game)

class CaptureAI(BaseAI):
    """基线算法：尽可能多吃子。

    以「棋子大小总和之差」为分值，做吃子静态搜索（quiescence search）：

    * 每一步都优先吃掉对方最大的棋子；
    * 搜索对方能吃回什么、自己再能吃回什么……直到没有吃子为止（可配深度）；
    * 因此在「白吃一个大子」和「吃一个小子但会被吃回」之间能做出正确取舍。

    它同时是 AlphaZero 训练的基线算法、评估锚点，也可以在界面里直接作为对手。
    """

    name = "吃子基线"

    def __init__(self, depth: int = 4, seed: Optional[int] = None, noise: float = 0.0,
                 mobility_weight: float = 0.05):
        self.depth = max(1, int(depth))
        self.random = random.Random(seed)
        self.noise = noise
        self.mobility_weight = mobility_weight
        self.nodes = 0

    # ------------------------------------------------------------- 评估

    @staticmethod
    def material(game: Game, color: int) -> float:
        """以 color 的视角看「棋子大小总和」之差。"""
        white_count, white_size = game.piece_stats(WHITE)
        black_count, black_size = game.piece_stats(BLACK)
        if color == WHITE:
            return float(white_size - black_size)
        return float(black_size - white_size)

    def _terminal_value(self, game: Game, color: int) -> Optional[float]:
        if not game.is_over:
            return None
        result = game.result
        if result is None or result.winner is None:
            return 0.0
        return WIN_SCORE if result.winner == color else -WIN_SCORE

    def _captures(self, game: Game) -> List[Tuple[int, Move]]:
        found = []
        for move in game.legal_moves():
            value = game.cells[move.frm]
            target = game.cells[move.to]
            if target != 0 and (target > 0) != (value > 0):
                found.append((abs(target), move))
        found.sort(key=lambda item: -item[0])
        return found

    def _quiesce(self, game: Game, depth: int, alpha: float, beta: float) -> float:
        """只考虑吃子的静态搜索，返回行棋方视角的分值。"""
        self.nodes += 1
        color = game.side
        terminal = self._terminal_value(game, color)
        if terminal is not None:
            return terminal
        stand = self.material(game, color)
        if depth <= 0:
            return stand
        captures = self._captures(game)
        if not captures:
            return stand + self.mobility_weight * len(game.legal_moves())
        best = stand
        for _, move in captures:
            game.push(move)
            score = -self._quiesce(game, depth - 1, -beta, -alpha)
            game.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    # ------------------------------------------------------------- 接口

    def rank_moves(self, game: Game) -> List[Tuple[float, Move]]:
        """给全部合法着法打分（分值 = 走完后经吃子搜索评估的物质差），从高到低排序。"""
        moves = game.legal_moves()
        if not moves:
            return []
        self.nodes = 0
        scored: List[Tuple[float, Move]] = []
        for move in moves:
            game.push(move)
            score = -self._quiesce(game, self.depth - 1, -float("inf"), float("inf"))
            game.pop()
            scored.append((score, move))
        scored.sort(key=lambda item: -item[0])
        return scored

    def best_moves(self, game: Game) -> List[Move]:
        """返回并列最优的着法集合（用于做「并列最优」的软目标）。"""
        scored = self.rank_moves(game)
        if not scored:
            return []
        best = scored[0][0]
        return [move for score, move in scored if score >= best - 1e-9]

    def choose(self, game: Game) -> Optional[Move]:
        scored = self.rank_moves(game)
        if not scored:
            return None
        best = scored[0][0]
        pool = [move for score, move in scored if score >= best - 1e-9]
        if self.noise:
            jittered = [(score + self.random.uniform(-self.noise, self.noise), move)
                        for score, move in scored]
            jittered.sort(key=lambda item: -item[0])
            top = jittered[0][0]
            pool = [move for score, move in jittered if score >= top - 1e-9] or pool
        return self.random.choice(pool)

    def describe(self) -> str:
        return "吃子基线（尽可能多吃子，吃子搜索深度 %d）" % self.depth
