"""根号棋 Alpha-Beta 引擎。

实现的要素（对应需求里的可选项，这里全部落地）：

* **Negamax + Alpha-Beta**：默认深度 4（@BT@depth@BT@ 可调，从深度 1 开始迭代加深）；
* **迭代加深**：逐层加深，可以给时间上限，时间到就用上一层的结果；
* **着法排序**：置换表着法 → 吃子（按被吃子大小 + 攻击方大小）→ 杀手着法 → 历史启发；
* **置换表**：Zobrist 哈希 + 深度/上下界标志/分值/最佳着法；
* **静态交换评估 SEE**：判断「吃这一格到底赚不赚」，并在静态搜索里剪掉亏本吃子；
* **攻防分析** @BT@analyze_exchange()@BT@：回答「这一格攻防结束后归谁、那个子还能不能动、
  双方各损失多少、再补一个 1 或 2 能不能吃回并复活」——这是根号棋特有、也是需求里最看重的部分；
* **评估函数**：物质（按「能否移动」加权）+ 子数 + 机动性 + 对敌方死子的威胁。

关于「能否移动」：一个棋子能否移动只取决于它的大小能否表示成两个整数平方和
（3、6、7、11…… 这类大小是永远动不了的「死子」）。死子仍然计入规则 7 的大小和，
但既不能进攻也不能逃跑，所以评估里给它打折，并对「能吃到的敌方死子」额外加分——
这正是需求里说的：用 1 或 2 这种小子去取回对方动不了的大子，等于白赚。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .core import (BLACK, DEFAULT_NO_CAPTURE_LIMIT, NUM_SQUARES, WHITE, Game,
                   IllegalMove, Move, MoveInfo, move_offsets, opponent,
                   square_name)
from .ai import BaseAI

__all__ = ["AlphaBetaAI", "Searcher", "SearchResult", "ExchangeReport",
           "TranspositionTable", "Zobrist", "evaluate", "see", "analyze_exchange",
           "is_playable_size", "EXACT", "LOWER", "UPPER", "MATE_SCORE"]

EXACT, LOWER, UPPER = 0, 1, 2
MATE_SCORE = 100000.0
INFINITY = float("inf")

#: 死子（大小无法移动）在评估里的折价
STUCK_WEIGHT = 0.65
#: 棋子数量在评估里的权重（规则 7 与「吃光对方」都看子数）
COUNT_WEIGHT = 3.0
#: 每多一个合法着法的加分
MOBILITY_WEIGHT = 0.15
#: 能吃到的敌方死子的加分系数
STUCK_THREAT_WEIGHT = 0.6
#: 能立刻吃到的普通棋子的加分系数（鼓励进攻，但远小于实际吃子收益）
HANGING_WEIGHT = 0.4

_PLAYABLE_CACHE: Dict[int, bool] = {}


def is_playable_size(size: int) -> bool:
    """该大小能否移动（能否表示成两个整数平方和）。"""
    size = int(size)
    cached = _PLAYABLE_CACHE.get(size)
    if cached is None:
        cached = bool(move_offsets(size))
        _PLAYABLE_CACHE[size] = cached
    return cached


# --------------------------------------------------------------------------- Zobrist


class Zobrist:
    """Zobrist 哈希：每个 (格点, 带符号大小) 一个随机数。"""

    __slots__ = ("_piece", "_side", "_nocap", "_rng", "_bucket")

    def __init__(self, seed: int = 0x5F3759DF):
        self._rng = random.Random(seed)
        self._piece: Dict[Tuple[int, int], int] = {}
        self._side = self._rng.getrandbits(64)
        self._bucket = 16
        self._nocap = [self._rng.getrandbits(64) for _ in range(self._bucket)]

    def piece(self, square: int, value: int) -> int:
        key = (square, value)
        found = self._piece.get(key)
        if found is None:
            found = self._rng.getrandbits(64)
            self._piece[key] = found
        return found

    def game_hash(self, game: Game) -> int:
        h = 0
        cells = game.cells
        piece = self._piece
        for square in range(NUM_SQUARES):
            value = cells[square]
            if value:
                key = (square, value)
                found = piece.get(key)
                if found is None:
                    found = self._rng.getrandbits(64)
                    piece[key] = found
                h ^= found
        if game.side < 0:
            h ^= self._side
        limit = game.no_capture_limit or DEFAULT_NO_CAPTURE_LIMIT
        progress = game.no_capture * self._bucket // max(1, limit)
        h ^= self._nocap[min(self._bucket - 1, progress)]
        return h


class TranspositionTable:
    """置换表（简单的整体清空淘汰策略，够用且省内存）。"""

    def __init__(self, max_entries: int = 300000):
        self.max_entries = int(max_entries)
        self.table: Dict[int, Tuple[int, float, int, Optional[Move]]] = {}
        self.hits = 0
        self.stores = 0

    def clear(self) -> None:
        self.table.clear()

    def get(self, key: int):
        entry = self.table.get(key)
        if entry is not None:
            self.hits += 1
        return entry

    def put(self, key: int, depth: int, score: float, flag: int, move: Optional[Move]) -> None:
        if len(self.table) >= self.max_entries:
            self.table.clear()
        self.table[key] = (depth, score, flag, move)
        self.stores += 1


# --------------------------------------------------------------------------- 评估


def evaluate(game: Game, color: Optional[int] = None) -> float:
    """局面评估（color 视角）：物质 + 子数 + 机动性 + 死子威胁。"""
    if color is None:
        color = game.side
    score = 0.0
    own_count = 0
    enemy_count = 0
    for _, who, size in game.pieces():
        if is_playable_size(size):
            value = float(size)
        else:
            value = size * STUCK_WEIGHT
        if who == color:
            score += value
            own_count += 1
        else:
            score -= value
            enemy_count += 1

    # 我方所有着法：用来算机动性与「能吃到什么」
    attacks: Dict[int, int] = {}
    mobility = 0
    for square, who, size in game.pieces(color):
        for move in game.moves_from(square):
            mobility += 1
            target = game.cells[move.to]
            if target:
                # 记录「用多大的子能吃到这里、能吃多大的子」
                gain = abs(target) - abs(game.cells[square]) * 0.35
                if gain > attacks.get(move.to, -1e9):
                    attacks[move.to] = gain
    score += MOBILITY_WEIGHT * mobility
    for square, who, size in game.pieces():
        if who == color:
            continue
        if square in attacks:
            if not is_playable_size(size):
                score += STUCK_THREAT_WEIGHT * size      # 能吃到的死子：基本白赚
            else:
                score += HANGING_WEIGHT * size           # 能吃到但对方大概会还手
    score += COUNT_WEIGHT * (own_count - enemy_count)
    return score


# --------------------------------------------------------------------------- SEE


def _terminal_score(game: Game, ply: int = 0) -> Optional[float]:
    """对局已结束时的分值（当前行棋方视角）。"""
    if not game.is_over:
        return None
    result = game.result
    if result is None or result.winner is None:
        return 0.0
    if result.winner == game.side:
        return MATE_SCORE - ply
    return -(MATE_SCORE - ply)


def _capture_to(game: Game, square: int, color: int) -> List[Tuple[int, Move]]:
    """某一方所有能吃到 square 的着法，按「攻击方大小」从小到大排序。"""
    found = []
    cells = game.cells
    for from_square, who, size in game.pieces(color):
        for move in game.moves_from(from_square):
            if move.to == square:
                found.append((size, move))
    found.sort(key=lambda item: item[0])
    return found


def _see_recapture(game: Game, square: int, depth: int) -> float:
    """继续吃这一格的一方的最佳净收益（0 表示可以选择不吃）。"""
    if depth <= 0:
        return 0.0
    terminal = _terminal_score(game)
    if terminal is not None:
        return terminal
    color = game.side
    best = 0.0
    for _, move in _capture_to(game, square, color):
        captured = abs(game.cells[square])
        values = _push(game, move)
        if values is None:
            continue
        try:
            gain = captured - _see_recapture(game, square, depth - 1)
        finally:
            game.pop()
        if gain > best:
            best = gain
    return best


def _push(game: Game, move: Move) -> Optional[MoveInfo]:
    try:
        return game.push(move)
    except (IllegalMove, RuntimeError):
        return None


def see(game: Game, move: Move, depth: int = 6) -> float:
    """静态交换评估：走子方吃这一格的净收益（>0 表示赚）。"""
    value = game.cells[move.frm]
    target = game.cells[move.to]
    if value == 0 or target == 0 or (target > 0) == (value > 0):
        return 0.0
    captured = abs(target)
    if _push(game, move) is None:
        return 0.0
    try:
        # 注意：_terminal_score 是「走完这一步之后行棋方（对手）视角」的分值，
        # 要取反才是走子方自己的收益。原先漏了取反，导致「一步吃光对手/把对手卡死」
        # 这种赢棋着法被当成巨亏而在静态搜索里被剪掉。
        terminal = _terminal_score(game)
        gain = (-terminal if terminal is not None
                else captured - _see_recapture(game, move.to, depth - 1))
    finally:
        game.pop()
    return gain


# --------------------------------------------------------------------------- 攻防分析


@dataclass
class ExchangeReport:
    """一次「攻防」的分析结果。"""

    square: str
    attacker: str = ""              # 先动手的一方 w/b
    owner_after: str = "-"          # 攻防结束后这一格归谁
    final_size: int = 0
    final_mobile: bool = False
    gain_attacker: float = 0.0      # 先手方在这次攻防里的净物质收益
    gain_defender: float = 0.0
    sequence: List[str] = field(default_factory=list)
    recapture: List[str] = field(default_factory=list)   # 再补哪些子能吃回并复活
    material_swing: float = 0.0     # 这条线里双方实际吃到的子力差（先手方视角）
    ends_game: bool = False         # 这条线走完对局是否直接结束
    outcome: str = ""               # 若结束，对局结果
    attacker_stuck: bool = False    # 先手方走完是否变成「无子可动」
    notes: str = ""

    def text(self) -> str:
        lines = ["攻防焦点：%s（先手 %s）" % (self.square, self.attacker or "-"),
                 "  结束后归属：%s，该子大小 %d，%s"
                 % (self.owner_after, self.final_size,
                    "可以移动" if self.final_mobile else "无法移动（死子）"),
                 "  交换序列：%s" % (" ".join(self.sequence) if self.sequence else "（不吃）"),
                 "  这条线的净收益（含终局）：先手方 %+.1f，后手方 %+.1f"
                 % (self.gain_attacker, self.gain_defender),
                 "  实际吃到的子力差：%+.1f（先手方视角）" % self.material_swing,
                 "  继续吃回的选择：%s" % ("、".join(self.recapture) if self.recapture else "无")]
        if self.ends_game:
            lines.append("  ⚠ 这条线走完对局直接结束：%s" % self.outcome)
        if self.attacker_stuck:
            lines.append("  ⚠ 先手方走完变成「无子可动」，按规则 5/7 直接进入平局判定")
        if self.notes:
            lines.append("  说明：" + self.notes)
        return "\n".join(lines)


def analyze_exchange(game: Game, square, depth: int = 6) -> ExchangeReport:
    """分析某一格的攻防：谁先动手、最后归谁、那个子能不能动、双方损失、补 1/2 能否复活。

    这对应需求里最看重的那条：**抢到一个大子但把自己变成死子，未必划算**；
    反过来，用 1 或 2 去取回对方动不了的大子，等于白赚（大小并进来还变回可动）。
    """
    if isinstance(square, str):
        from .core import parse_square

        square = parse_square(square)
    side = game.side
    report = ExchangeReport(square=square_name(square),
                            attacker="w" if side == WHITE else "b")
    if game.is_over:
        report.notes = "对局已经结束"
        return report

    candidates = _capture_to(game, square, side)
    if not candidates:
        report.owner_after = "w" if game.cells[square] > 0 else ("b" if game.cells[square] < 0 else "-")
        report.final_size = abs(game.cells[square])
        report.final_mobile = is_playable_size(report.final_size) if report.final_size else False
        report.notes = "当前没有棋子能吃到这一格"
        return report

    # 用 SEE 递归找出先手方最赚的一条线，并记录序列
    best_gain = -INFINITY
    best_move: Optional[Move] = None
    best_sequence: List[str] = []
    for _, move in candidates:
        captured = abs(game.cells[square])
        info = _push(game, move)
        if info is None:
            continue
        try:
            continuation, line = _best_line(game, square, depth - 1)
            gain = captured - continuation
        finally:
            game.pop()
        if gain > best_gain:
            best_gain = gain
            best_move = move
            best_sequence = [info.notation] + line
    if not best_sequence:
        report.notes = "没有可用的吃子着法"
        return report

    # 走一遍这条线，看最终归属、终局结果与「谁被卡死」
    applied: List[MoveInfo] = []
    try:
        for token in best_sequence:
            info = _push(game, token)
            if info is None:
                break
            applied.append(info)
        owner = game.cells[square]
        report.sequence = list(best_sequence)
        report.owner_after = "w" if owner > 0 else ("b" if owner < 0 else "-")
        report.final_size = abs(owner)
        report.final_mobile = is_playable_size(report.final_size) if report.final_size else False
        # 净收益以 SEE 的递归结果为准（它会把「走完直接终局」也算进去）
        report.gain_attacker = best_gain
        report.gain_defender = -best_gain
        swing = 0.0
        for info in applied:
            swing += info.captured if info.mover == side else -info.captured
        report.material_swing = swing
        if game.is_over:
            report.ends_game = True
            report.outcome = game.result.text() if game.result else "对局结束"
        report.attacker_stuck = not game.has_moves(side)

        # 再补一个 1~5 能不能吃回，以及吃回之后这个子能不能动
        loser = opponent(side) if (applied and applied[-1].mover == side) else side
        for size in (1, 2, 3, 4, 5):
            for from_square, who, piece_size in game.pieces(loser):
                if piece_size != size:
                    continue
                for move in game.moves_from(from_square):
                    if move.to != square:
                        continue
                    merged = size + report.final_size
                    tag = "可动" if is_playable_size(merged) else "仍是死子"
                    report.recapture.append(
                        "%d%sx%s→%d（%s）" % (size, square_name(from_square),
                                             report.square, merged, tag))
        if report.ends_game:
            report.notes = "这条线会把对局直接走完：" + report.outcome
        elif report.attacker_stuck:
            report.notes = ("先手方虽然吃到子，但走完自己已经无子可动，"
                            "按规则 5/7 会直接进入平局判定，未必划算。")
        elif report.final_size and not report.final_mobile and report.recapture:
            report.notes = ("抢下这一格的一方留下的是死子：既不能再进攻也不能逃，"
                            "对方可以用小子吃回并把大小并过去，通常并不划算。")
        elif report.final_mobile and report.gain_attacker > 0:
            report.notes = "先手方赚到子力，且留下的子可以继续行动。"
    finally:
        for _ in applied:
            game.pop()
    return report


def _best_line(game: Game, square: int, depth: int) -> Tuple[float, List[str]]:
    """继续吃这一格的最佳净收益与着法序列（0 表示停手）。"""
    if depth <= 0:
        return 0.0, []
    terminal = _terminal_score(game)
    if terminal is not None:
        return terminal, []
    color = game.side
    best = 0.0
    best_line: List[str] = []
    for _, move in _capture_to(game, square, color):
        captured = abs(game.cells[square])
        info = _push(game, move)
        if info is None:
            continue
        try:
            continuation, line = _best_line(game, square, depth - 1)
            gain = captured - continuation
        finally:
            game.pop()
        if gain > best:
            best = gain
            best_line = [info.notation] + line
    return best, best_line

# --------------------------------------------------------------------------- 搜索


@dataclass
class SearchResult:
    """一次搜索的结果。"""

    move: Optional[Move] = None
    score: float = 0.0
    depth: int = 0
    nodes: int = 0
    seconds: float = 0.0
    pv: List[str] = field(default_factory=list)
    stopped: bool = False

    def text(self) -> str:
        return ("深度 %d，分值 %+.1f，节点 %d，用时 %.2fs%s"
                % (self.depth, self.score, self.nodes, self.seconds,
                   "（超时中断）" if self.stopped else ""))


class Searcher:
    """Negamax + Alpha-Beta 搜索器。"""

    def __init__(self, depth: int = 4, time_limit: Optional[float] = None,
                 node_limit: Optional[int] = None, tt_size: int = 300000,
                 seed: Optional[int] = None, use_see: bool = True):
        self.depth = max(1, int(depth))
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.zobrist = Zobrist(seed if seed is not None else 0x5F3759DF)
        self.tt = TranspositionTable(tt_size)
        self.use_see = bool(use_see)
        self.nodes = 0
        self.stopped = False
        self.deadline: Optional[float] = None
        self.killers: List[List[Optional[Move]]] = []
        self.history: Dict[Move, float] = {}

    # ------------------------------------------------------------- 工具

    def _out_of_time(self) -> bool:
        if self.stopped:
            return True
        if self.node_limit is not None and self.nodes >= self.node_limit:
            self.stopped = True
            return True
        if self.deadline is not None and (self.nodes & 1023) == 0 and time.time() > self.deadline:
            self.stopped = True
            return True
        return False

    def _evaluate(self, game: Game, color: Optional[int] = None) -> float:
        """叶子/静态评估（当前行棋方视角）。子类可以换成「网络价值 + 手工评估」。"""
        return evaluate(game, game.side if color is None else color)

    def _order(self, game: Game, tt_move: Optional[Move], ply: int) -> List[Move]:
        """着法排序：置换表着法 → 吃子（吃大子优先）→ 杀手 → 历史。"""
        moves = game.legal_moves()
        killers = self.killers[ply] if ply < len(self.killers) else (None, None)
        history = self.history
        cells = game.cells
        scored = []
        for move in moves:
            value = cells[move.frm]
            target = cells[move.to]
            if move == tt_move:
                score = 1e12
            elif target != 0 and (target > 0) != (value > 0):
                score = 1e9 + abs(target) * 1000 - abs(value)
            elif move == killers[0]:
                score = 1e8
            elif move == killers[1]:
                score = 9e7
            else:
                score = history.get(move, 0.0)
            scored.append((score, move))
        scored.sort(key=lambda item: -item[0])
        return [move for _, move in scored]

    # ------------------------------------------------------------- 静态搜索

    def _quiesce(self, game: Game, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1
        if self._out_of_time():
            return alpha
        terminal = _terminal_score(game, ply)
        if terminal is not None:
            return terminal
        stand = self._evaluate(game)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        captures = []
        cells = game.cells
        for move in game.legal_moves():
            value = cells[move.frm]
            target = cells[move.to]
            if target != 0 and (target > 0) != (value > 0):
                captures.append((abs(target), move))
        captures.sort(key=lambda item: -item[0])
        for _, move in captures:
            if self.use_see and see(game, move, depth=4) < 0:
                continue                      # 亏本的吃子不搜（交换分析已经算清楚了）
            if _push(game, move) is None:
                continue
            try:
                score = -self._quiesce(game, -beta, -alpha, ply + 1)
            finally:
                game.pop()
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return alpha

    # ------------------------------------------------------------- 主搜索

    def _negamax(self, game: Game, depth: int, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1
        if self._out_of_time():
            return alpha
        terminal = _terminal_score(game, ply)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self._quiesce(game, alpha, beta, ply)

        key = self.zobrist.game_hash(game)
        alpha_origin = alpha
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            entry_depth, entry_score, flag, entry_move = entry
            tt_move = entry_move
            if entry_depth >= depth:
                if flag == EXACT:
                    return entry_score
                if flag == LOWER and entry_score > alpha:
                    alpha = entry_score
                elif flag == UPPER and entry_score < beta:
                    beta = entry_score
                if alpha >= beta:
                    return entry_score

        best = -INFINITY
        best_move: Optional[Move] = None
        flag = UPPER
        for move in self._order(game, tt_move, ply):
            info = _push(game, move)
            if info is None:
                continue
            try:
                score = -self._negamax(game, depth - 1, -beta, -alpha, ply + 1)
            finally:
                game.pop()
            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
                flag = EXACT
            if alpha >= beta:
                flag = LOWER
                if info.captured == 0:
                    slot = self.killers[ply]
                    if slot[0] != move:
                        slot[1] = slot[0]
                        slot[0] = move
                    self.history[move] = self.history.get(move, 0.0) + depth * depth
                break
        self.tt.put(key, depth, best, flag, best_move)
        return best

    def _root(self, game: Game, depth: int, previous_best: Optional[Move]):
        alpha, beta = -INFINITY, INFINITY
        best_score = -INFINITY
        best_move: Optional[Move] = None
        best_line: List[str] = []
        for move in self._order(game, previous_best, 0):
            info = _push(game, move)
            if info is None:
                continue
            try:
                score = -self._negamax(game, depth - 1, -beta, -alpha, 1)
            finally:
                game.pop()
            if self.stopped:
                break
            if score > best_score:
                best_score = score
                best_move = move
                best_line = [info.notation]
            if score > alpha:
                alpha = score
        return best_score, best_move, best_line

    # ------------------------------------------------------------- 接口

    def score_moves(self, game: Game, depth: int = 1) -> List[Tuple[float, Move]]:
        """给全部合法着法打分（固定深度、从走子方视角），用于软目标与诊断。"""
        scored: List[Tuple[float, Move]] = []
        if game.is_over:
            return scored
        for move in game.legal_moves():
            info = _push(game, move)
            if info is None:
                continue
            try:
                score = -self._negamax(game, max(0, depth - 1), -INFINITY, INFINITY, 1)
            finally:
                game.pop()
            scored.append((score, move))
        scored.sort(key=lambda item: -item[0])
        return scored

    def search(self, game: Game, depth: Optional[int] = None,
               time_limit: Optional[float] = None) -> SearchResult:
        """迭代加深搜索，返回最佳着法。"""
        if game.is_over:
            return SearchResult()
        max_depth = max(1, int(depth if depth is not None else self.depth))
        limit = time_limit if time_limit is not None else self.time_limit
        started = time.time()
        self.nodes = 0
        self.stopped = False
        self.deadline = (started + limit) if limit else None
        result = SearchResult()
        previous: Optional[Move] = None
        for current in range(1, max_depth + 1):
            self.killers = [[None, None] for _ in range(current + 2)]
            score, move, line = self._root(game, current, previous)
            if self.stopped and current > 1:
                break
            if move is not None:
                result = SearchResult(move=move, score=score, depth=current,
                                      nodes=self.nodes, seconds=time.time() - started,
                                      pv=line, stopped=self.stopped)
                previous = move
            if abs(score) > MATE_SCORE - 5000 or self.stopped:
                break
        if result.move is None:
            moves = game.legal_moves()
            if moves:
                fallback = moves[0]
                result = SearchResult(move=fallback, score=0.0, depth=0,
                                      nodes=self.nodes, seconds=time.time() - started)
        return result


# --------------------------------------------------------------------------- AI 接口


class AlphaBetaAI(BaseAI):
    """Alpha-Beta 引擎（Negamax + 迭代加深 + 置换表 + 杀手/历史 + SEE）。"""

    name = "Alpha-Beta"

    def __init__(self, depth: int = 4, time_limit: Optional[float] = None,
                 node_limit: Optional[int] = None, seed: Optional[int] = None,
                 tt_size: int = 300000, use_see: bool = True):
        self.depth = max(1, int(depth))
        self.time_limit = time_limit
        self.searcher = Searcher(depth=self.depth, time_limit=time_limit,
                                 node_limit=node_limit, tt_size=tt_size, seed=seed,
                                 use_see=use_see)
        self.random = random.Random(seed)
        self.last: Optional[SearchResult] = None

    def choose(self, game: Game) -> Optional[Move]:
        self.last = self.searcher.search(game, depth=self.depth, time_limit=self.time_limit)
        return self.last.move

    def describe(self) -> str:
        extra = "，限时 %.2fs" % self.time_limit if self.time_limit else ""
        return "Alpha-Beta（深度 %d%s）" % (self.depth, extra)

    def analyze(self, game: Game, square, depth: int = 6) -> ExchangeReport:
        """分析某一格的攻防（归属 / 能否移动 / 双方损失 / 补小子能否复活）。"""
        return analyze_exchange(game, square, depth=depth)

    def rank_moves(self, game: Game, depth: Optional[int] = None) -> List[Tuple[float, Move]]:
        """全部着法的评分（模仿学习等场景用的软目标来源）。"""
        return self.searcher.score_moves(game, depth=depth or max(1, self.depth - 1))

    def __repr__(self) -> str:
        return "<AlphaBetaAI depth=%d>" % self.depth
