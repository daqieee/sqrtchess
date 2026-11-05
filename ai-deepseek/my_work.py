"""根号棋强力 AI（纯 Python 标准库）。

设计要点
--------
1. 自建轻量搜索棋盘，避免在搜索树中反复创建 Game 对象；
2. 迭代加深 + Negamax + Alpha-Beta/PVS + 置换表；
3. 吃子静态搜索（quiescence search），避免只搜索一半吃子交换；
4. 评估函数综合物质、棋子数、可动性、卡死子、攻击与受攻威胁，
   并特别考虑根号棋独有的吃子后尺寸变化导致棋子卡死和无子可动立即裁定；
5. 着法排序综合置换表着法、吃子、Killer、History 与落点机动性。

该模块既可直接作为 AI 应答协议程序运行：

    python ai-deepseek/my_work.py

也可以被 Python 直接调用：

    import sys
    sys.path.insert(0, "ai-deepseek")
    from my_work import SqrtChessAI, choose_move
    move = choose_move(game, time_limit=2.0)
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 允许从项目根目录或任意工作目录运行本文件。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqrtchess import BLACK, WHITE, Game, Move, move_offsets  # noqa: E402

# ---------------------------------------------------------------------------
# 基础常量
# ---------------------------------------------------------------------------

BOARD = 11
N_SQUARES = BOARD * BOARD
MOVE_BASE = N_SQUARES

INF = float("inf")
WIN_SCORE = 10 ** 9
DEFAULT_TIME_LIMIT = 1.8       # 秒；外部协议未给出时限时使用
MAX_SEARCH_DEPTH = 40
MAX_QDEPTH = 8

# analyze() 返回元组的字段位置
A_MOVES = 0
A_CAPTURES = 1
A_MOB = 2
A_GEO = 3
A_STUCK = 4
A_CAPSUM = 5
A_CAPMAX = 6

# ---------------------------------------------------------------------------
# 走法几何缓存
# ---------------------------------------------------------------------------

#: size -> tuple(每个格点上的目标格点元组)
_TARGETS: Dict[int, Tuple[Tuple[int, ...], ...]] = {}
_TARGET_MASKS: Dict[int, Tuple[int, ...]] = {}
_TARGET_COUNTS: Dict[int, Tuple[int, ...]] = {}
_POPCOUNT = getattr(int, "bit_count", None)
if _POPCOUNT is not None:
    _popcount = _POPCOUNT
else:
    def _popcount(value: int) -> int:  # Python 3.8/3.9 兼容
        return bin(value).count("1")


def _target_table(size: int) -> Tuple[Tuple[int, ...], ...]:
    """返回某个棋子尺寸在 121 个格点上的全部几何落点。"""
    table = _TARGETS.get(size)
    if table is not None:
        return table
    offsets = move_offsets(size)
    rows: List[Tuple[int, ...]] = []
    for sq in range(N_SQUARES):
        x = sq % BOARD
        y = sq // BOARD
        dests = []
        for dx, dy in offsets:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < BOARD and 0 <= ny < BOARD:
                dests.append(ny * BOARD + nx)
        rows.append(tuple(dests))
    table = tuple(rows)
    _TARGETS[size] = table
    _TARGET_MASKS[size] = tuple(
        sum(1 << dest for dest in dests) for dests in table
    )
    _TARGET_COUNTS[size] = tuple(len(dests) for dests in table)
    return table


# ---------------------------------------------------------------------------
# Zobrist 散列
# ---------------------------------------------------------------------------

_RNG = random.Random(0x5EED5EEDC0FFEE)
_PIECE_KEYS: Dict[Tuple[int, int], int] = {}
_CAPTURE_KEYS: Dict[int, int] = {}
_SIDE_KEY = _RNG.getrandbits(64)


def _piece_key(sq: int, signed_value: int) -> int:
    key = (sq, signed_value)
    value = _PIECE_KEYS.get(key)
    if value is None:
        value = _RNG.getrandbits(64)
        _PIECE_KEYS[key] = value
    return value


def _capture_key(count: int) -> int:
    value = _CAPTURE_KEYS.get(count)
    if value is None:
        value = _RNG.getrandbits(64)
        _CAPTURE_KEYS[count] = value
    return value


# ---------------------------------------------------------------------------
# 轻量搜索局面
# ---------------------------------------------------------------------------


class _State:
    """搜索专用局面。只做增删棋子与尺寸更新，不保存完整历史。"""

    __slots__ = ("cells", "side", "no_capture", "limit", "counts", "sums", "hash", "occ", "color_occ")

    def __init__(self, game: Game):
        self.cells = list(game.cells)
        self.side = int(game.side)
        self.no_capture = int(game.no_capture)
        self.limit = None if game.no_capture_limit is None else int(game.no_capture_limit)
        wc, ws = game.piece_stats(WHITE)
        bc, bs = game.piece_stats(BLACK)
        self.counts = [wc, bc]
        self.sums = [ws, bs]

        h = 0
        occ = 0
        white_occ = 0
        black_occ = 0
        for sq, value in enumerate(self.cells):
            if value:
                h ^= _piece_key(sq, value)
                bit = 1 << sq
                occ |= bit
                if value > 0:
                    white_occ |= bit
                else:
                    black_occ |= bit
        self.occ = occ
        self.color_occ = [white_occ, black_occ]
        if self.side == BLACK:
            h ^= _SIDE_KEY
        h ^= _capture_key(self.no_capture)
        self.hash = h

    def do_move(self, encoded: int) -> Tuple[int, int, int, int, int, int]:
        """走一步。返回 undo 信息 (frm, to, captured, old_no_capture, old_hash, old_side)。"""
        frm, to = divmod(encoded, MOVE_BASE)
        cells = self.cells
        value = cells[frm]
        target = cells[to]
        sign = 1 if value > 0 else -1
        captured = abs(target) if (target and (target > 0) != (value > 0)) else 0
        own = 0 if sign > 0 else 1
        opp = 1 - own

        old_no_capture = self.no_capture
        old_hash = self.hash
        old_side = self.side

        h = old_hash
        h ^= _capture_key(old_no_capture)
        h ^= _piece_key(frm, value)
        if target:
            h ^= _piece_key(to, target)

        new_value = sign * (abs(value) + captured)
        cells[frm] = 0
        cells[to] = new_value
        # 更新占用位图：起点清空，终点置位。
        from_bit = 1 << frm
        to_bit = 1 << to
        self.occ = (self.occ & ~from_bit) | to_bit
        self.color_occ[own] = (self.color_occ[own] & ~from_bit) | to_bit
        if captured:
            self.color_occ[opp] &= ~to_bit
        h ^= _piece_key(to, new_value)

        if captured:
            self.counts[opp] -= 1
            self.sums[opp] -= captured
            self.sums[own] += captured
            self.no_capture = 0
        else:
            self.no_capture = old_no_capture + 1

        # 行棋方翻转；_SIDE_KEY 始终 xor，因为 1 与 -1 互换。
        h ^= _SIDE_KEY
        h ^= _capture_key(self.no_capture)
        self.hash = h
        self.side = -self.side
        return frm, to, captured, old_no_capture, old_hash, old_side

    def undo_move(self, undo: Tuple[int, int, int, int, int, int]) -> None:
        frm, to, captured, old_no_capture, old_hash, old_side = undo
        cells = self.cells
        value = cells[to]
        sign = 1 if value > 0 else -1
        new_size = abs(value)
        cells[frm] = sign * (new_size - captured)
        cells[to] = (-sign * captured) if captured else 0
        from_bit = 1 << frm
        to_bit = 1 << to
        self.occ = (self.occ & ~to_bit) | from_bit
        own = 0 if sign > 0 else 1
        opp = 1 - own
        self.color_occ[own] = (self.color_occ[own] & ~to_bit) | from_bit
        if captured:
            self.occ |= to_bit
            self.color_occ[opp] |= to_bit
            self.counts[opp] += 1
            self.sums[opp] += captured
            self.sums[own] -= captured
        self.no_capture = old_no_capture
        self.hash = old_hash
        self.side = old_side

    def adjudicate_winner(self) -> Optional[int]:
        """按规则 7 返回胜方；None 表示和棋。"""
        wc, bc = self.counts
        ws, bs = self.sums
        if wc > bc and ws > bs:
            return WHITE
        if bc > wc and bs > ws:
            return BLACK
        return None

    def result_value(self, winner: Optional[int], ply: int) -> float:
        """把裁定结果转换成当前行棋方视角的搜索分值。"""
        if winner is None:
            return 0.0
        if winner == self.side:
            return float(WIN_SCORE - ply)
        return float(-WIN_SCORE + ply)

    def analyze(self, mode: str = "all") -> Tuple[Any, ...]:
        """一次扫描同时得到走法、双方可动性、几何机动性、卡死子和攻击压力。

        使用位图快速计算合法落点数，只有当前行棋方的全部着法以及双方的
        吃子目标需要逐格枚举，从而减少静态搜索中的 Python 循环开销。

        mode:
          * "all"      : 生成行棋方全部着法，并生成行棋方吃子着法；
          * "captures" : 只生成行棋方吃子着法（仍计算双方全部机动性）；
          * "none"     : 不生成着法，只做静态分析。
        """
        cells = self.cells
        side = self.side
        side_index = 0 if side == WHITE else 1
        moves: List[int] = []
        captures: List[int] = []
        mob = [0, 0]
        geo = [0, 0]
        stuck = [0, 0]
        capsum = [0, 0]
        capmax = [0, 0]
        attacked_mask = [0, 0]
        mask_cache = _TARGET_MASKS
        count_cache = _TARGET_COUNTS
        popcount = _popcount
        all_occupied = self.occ
        occupied_by_color = self.color_occ

        occupied = all_occupied
        while occupied:
            piece_bit = occupied & -occupied
            occupied ^= piece_bit
            sq = piece_bit.bit_length() - 1
            value = cells[sq]
            color_index = 0 if value > 0 else 1
            size = value if value > 0 else -value
            size_masks = mask_cache.get(size)
            if size_masks is None:
                _target_table(size)
                size_masks = mask_cache[size]
            piece_mask = size_masks[sq]
            geometric_count = count_cache[size][sq]
            own_mask = occupied_by_color[color_index]
            enemy_mask = occupied_by_color[1 - color_index]

            legal_count = geometric_count - popcount(piece_mask & own_mask)
            geo[color_index] += geometric_count
            mob[color_index] += legal_count
            if legal_count == 0:
                stuck[color_index] += 1

            # 吃子目标：同时用于攻击压力和当前方着法生成。
            enemy_targets = piece_mask & enemy_mask
            while enemy_targets:
                to_bit = enemy_targets & -enemy_targets
                enemy_targets ^= to_bit
                to = to_bit.bit_length() - 1
                if not (attacked_mask[color_index] & to_bit):
                    attacked_mask[color_index] |= to_bit
                    target_size = abs(cells[to])
                    capsum[color_index] += target_size
                    if target_size > capmax[color_index]:
                        capmax[color_index] = target_size
                if color_index == side_index:
                    encoded = sq * MOVE_BASE + to
                    captures.append(encoded)
                    if mode == "all":
                        moves.append(encoded)

            # 空点着法只需要当前行棋方，而且只在主搜索节点生成。
            if mode == "all" and color_index == side_index:
                empty_targets = piece_mask & ~all_occupied
                while empty_targets:
                    to_bit = empty_targets & -empty_targets
                    empty_targets ^= to_bit
                    to = to_bit.bit_length() - 1
                    moves.append(sq * MOVE_BASE + to)

        return moves, captures, mob, geo, stuck, capsum, capmax

    def evaluate(self, info: Tuple[Any, ...]) -> float:
        """静态评估，返回当前行棋方视角的分数。"""
        side_index = 0 if self.side == WHITE else 1
        opp_index = 1 - side_index
        mob = info[A_MOB]
        geo = info[A_GEO]
        stuck = info[A_STUCK]
        capsum = info[A_CAPSUM]
        capmax = info[A_CAPMAX]

        material_diff = self.sums[side_index] - self.sums[opp_index]
        count_diff = self.counts[side_index] - self.counts[opp_index]
        mobility_diff = mob[side_index] - mob[opp_index]
        geometry_diff = geo[side_index] - geo[opp_index]
        stuck_diff = stuck[side_index] - stuck[opp_index]

        score = 10.0 * material_diff + 28.0 * count_diff
        score += 1.25 * mobility_diff + 0.28 * geometry_diff
        score -= 17.0 * stuck_diff

        # 当前行棋方拥有先手，因此自己的吃子机会权重更高；
        # 对手的攻击机会代表下一手威胁。
        score += 0.72 * capsum[side_index] + 1.9 * capmax[side_index]
        score -= 0.48 * capsum[opp_index] + 1.35 * capmax[opp_index]

        # 接近无吃子上限时，双方都会关心规则 7 的裁定。
        if self.limit is not None:
            remaining = self.limit - self.no_capture
            if remaining <= 16:
                urgency = float(17 - max(0, remaining))
                if count_diff > 0 and material_diff > 0:
                    score += 2.1 * urgency
                elif count_diff < 0 and material_diff < 0:
                    score -= 2.1 * urgency

        # 只剩一枚可动子时局面脆弱，给一点额外的风险提示。
        if mob[side_index] == 1:
            score -= 4.0
        if mob[opp_index] == 1:
            score += 4.0
        return score

# ---------------------------------------------------------------------------
# 搜索器
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    pass


class _Searcher:
    def __init__(self, state: _State, time_limit: float):
        self.state = state
        self.time_limit = max(0.02, float(time_limit))
        self.deadline = time.perf_counter() + self.time_limit
        self.nodes = 0
        self.tt: Dict[int, Tuple[int, float, int, int]] = {}
        self.qtt: Dict[int, Tuple[int, float, int, int]] = {}
        # TT flag: 0 exact, 1 lower bound, 2 upper bound
        self.killers: List[List[int]] = [[-1, -1] for _ in range(MAX_SEARCH_DEPTH + MAX_QDEPTH + 8)]
        self.history = [0] * (MOVE_BASE * MOVE_BASE)
        self.completed_depth = 0
        self.root_score = 0.0
        self.pv: List[int] = []

    # -------------------------------------------------------------- 工具

    def _tick(self) -> None:
        self.nodes += 1
        if (self.nodes & 255) == 0:
            if time.perf_counter() >= self.deadline:
                raise _Timeout

    def _cheap_terminal(self, ply: int) -> Optional[float]:
        """不生成着法即可判断的终止条件：吃光或无吃子上限。"""
        if self.state.counts[0] == 0 or self.state.counts[1] == 0:
            if self.state.counts[0] == 0 and self.state.counts[1] == 0:
                return 0.0
            winner = BLACK if self.state.counts[0] == 0 else WHITE
            return self.state.result_value(winner, ply)
        if self.state.limit is not None and self.state.no_capture >= self.state.limit:
            return self.state.result_value(self.state.adjudicate_winner(), ply)
        return None

    def _tt_probe(self, key: int, depth: int, alpha: float, beta: float):
        entry = self.tt.get(key)
        if entry is None:
            return None, alpha, beta
        entry_depth, value, flag, best_move = entry
        if entry_depth >= depth:
            if flag == 0:
                return value, alpha, beta
            if flag == 1 and value > alpha:
                alpha = value
            elif flag == 2 and value < beta:
                beta = value
            if alpha >= beta:
                return value, alpha, beta
        return None, alpha, beta

    def _tt_store(self, key: int, depth: int, value: float,
                  flag: int, best_move: int) -> None:
        old = self.tt.get(key)
        if old is None or depth >= old[0]:
            # 限制表大小，避免极端长局内存无限增长。
            if len(self.tt) > 260000 and old is None:
                self.tt.clear()
            self.tt[key] = (depth, value, flag, best_move)

    def _order_moves(self, moves: List[int], info: Tuple[Any, ...],
                     tt_move: int, ply: int) -> List[int]:
        cells = self.state.cells
        history = self.history
        killer0 = -1
        killer1 = -1
        if 0 <= ply < len(self.killers):
            killer0, killer1 = self.killers[ply]
        opp_index = 1 if self.state.side == WHITE else 0
        opp_count = self.state.counts[opp_index]

        scored: List[Tuple[float, int]] = []
        target_cache = _TARGETS
        for encoded in moves:
            frm, to = divmod(encoded, MOVE_BASE)
            value = cells[frm]
            target = cells[to]
            if target and (target > 0) != (value > 0):
                captured = abs(target)
                new_size = abs(value) + captured
                score = 20_000_000.0 + captured * 1200.0
                if opp_count == 1:
                    score += 100_000_000.0  # 直接吃光对方
                size_table = target_cache.get(new_size)
                if size_table is None:
                    size_table = _target_table(new_size)
                dest_targets = size_table[to]
                score += len(dest_targets) * 180.0
                if not dest_targets:
                    score -= 3_000_000.0
                # 吃大子后若自己变成卡死尺寸，通常仍可接受，但降低优先级。
            else:
                size = abs(value)
                size_table = target_cache.get(size)
                if size_table is None:
                    size_table = _target_table(size)
                dest_targets = size_table[to]
                score = len(dest_targets) * 18.0
                score += history[encoded] * 0.001
                if encoded == killer0:
                    score += 900_000.0
                elif encoded == killer1:
                    score += 450_000.0
            if encoded == tt_move:
                score += 500_000_000.0
            scored.append((score, encoded))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [encoded for _, encoded in scored]

    def _order_captures(self, moves: List[int], tt_move: int = -1) -> List[int]:
        cells = self.state.cells
        scored: List[Tuple[float, int]] = []
        target_cache = _TARGETS
        for encoded in moves:
            frm, to = divmod(encoded, MOVE_BASE)
            value = cells[frm]
            target = cells[to]
            captured = abs(target) if target else 0
            new_size = abs(value) + captured
            size_table = target_cache.get(new_size)
            if size_table is None:
                size_table = _target_table(new_size)
            dest_targets = size_table[to]
            score = captured * 1000.0 + len(dest_targets) * 50.0
            if not dest_targets:
                score -= 5000.0
            if encoded == tt_move:
                score += 1_000_000_000.0
            scored.append((score, encoded))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [encoded for _, encoded in scored]

    def _record_cutoff(self, encoded: int, ply: int, depth: int) -> None:
        frm, to = divmod(encoded, MOVE_BASE)
        target = self.state.cells[to]
        mover = self.state.cells[frm]
        if target == 0 and mover != 0:
            if 0 <= ply < len(self.killers):
                killers = self.killers[ply]
                if killers[0] != encoded:
                    killers[1] = killers[0]
                    killers[0] = encoded
            self.history[encoded] += depth * depth + 1

    # -------------------------------------------------------------- 静态搜索

    def _qsearch(self, alpha: float, beta: float, ply: int, qdepth: int) -> float:
        self._tick()
        terminal = self._cheap_terminal(ply)
        if terminal is not None:
            return terminal

        # 吃子静态搜索也保存置换表。键与主搜索相同，但使用独立表，
        # 避免浅层静态结果与主搜索深度条目互相污染。
        key = self.state.hash
        original_alpha = alpha
        original_beta = beta
        tt_move = -1
        qentry = self.qtt.get(key)
        if qentry is not None:
            entry_qdepth, entry_value, entry_flag, entry_move = qentry
            if entry_qdepth >= qdepth:
                tt_move = entry_move
                if entry_flag == 0:
                    return entry_value
                if entry_flag == 1 and entry_value > alpha:
                    alpha = entry_value
                elif entry_flag == 2 and entry_value < beta:
                    beta = entry_value
                if alpha >= beta:
                    return entry_value

        info = self.state.analyze("captures")
        # 任何一方无子可动都会立即进入规则 7 裁定。
        if info[A_MOB][0] == 0 or info[A_MOB][1] == 0:
            return self.state.result_value(self.state.adjudicate_winner(), ply)

        stand_pat = self.state.evaluate(info)
        best = stand_pat
        best_move = -1
        captures = info[A_CAPTURES]

        if qdepth <= 0:
            value = stand_pat
        elif stand_pat >= beta:
            value = stand_pat
        else:
            if stand_pat > alpha:
                alpha = stand_pat
            if not captures:
                value = stand_pat
            else:
                for encoded in self._order_captures(captures, tt_move):
                    undo = self.state.do_move(encoded)
                    try:
                        score = -self._qsearch(-beta, -alpha, ply + 1, qdepth - 1)
                    finally:
                        self.state.undo_move(undo)
                    if score > best:
                        best = score
                        best_move = encoded
                    if best > alpha:
                        alpha = best
                    if alpha >= beta:
                        break
                value = best

        if value <= original_alpha:
            flag = 2
        elif value >= original_beta:
            flag = 1
        else:
            flag = 0
        if len(self.qtt) > 260000 and qentry is None:
            self.qtt.clear()
        self.qtt[key] = (qdepth, value, flag, best_move)
        return value

    # -------------------------------------------------------------- 主搜索

    def negamax(self, depth: int, ply: int, alpha: float, beta: float) -> float:
        self._tick()
        terminal = self._cheap_terminal(ply)
        if terminal is not None:
            return terminal

        key = self.state.hash
        original_alpha = alpha
        original_beta = beta
        tt_value, alpha, beta = self._tt_probe(key, depth, alpha, beta)
        if tt_value is not None:
            return tt_value
        entry = self.tt.get(key)
        tt_move = entry[3] if entry is not None else -1

        if depth <= 0:
            return self._qsearch(alpha, beta, ply, MAX_QDEPTH)

        info = self.state.analyze("all")
        if info[A_MOB][0] == 0 or info[A_MOB][1] == 0:
            value = self.state.result_value(self.state.adjudicate_winner(), ply)
            self._tt_store(key, depth, value, 0, -1)
            return value

        moves = info[A_MOVES]
        if not moves:
            # analyze 已覆盖：正常情况下不会到达这里，保留兜底。
            value = self.state.result_value(self.state.adjudicate_winner(), ply)
            return value

        ordered = self._order_moves(moves, info, tt_move, ply)
        best = -INF
        best_move = ordered[0]
        first = True
        move_index = 0

        for encoded in ordered:
            moved_target = self.state.cells[encoded % MOVE_BASE]
            is_quiet = moved_target == 0
            undo = self.state.do_move(encoded)
            try:
                if first:
                    score = -self.negamax(depth - 1, ply + 1, -beta, -alpha)
                    first = False
                else:
                    # 迟着缩减：只对靠后的静着做浅一层验证，命中后再完整重搜。
                    reduction = 0
                    if (depth >= 4 and move_index >= 5 and is_quiet
                            and encoded != tt_move
                            and encoded != self.killers[ply][0]
                            and encoded != self.killers[ply][1]):
                        reduction = 1
                    score = -self.negamax(depth - 1 - reduction, ply + 1,
                                          -alpha - 1.0, -alpha)
                    if alpha < score < beta or (reduction and score > alpha):
                        score = -self.negamax(depth - 1, ply + 1, -alpha - 1.0, -alpha)
                        if alpha < score < beta:
                            score = -self.negamax(depth - 1, ply + 1, -beta, -alpha)
            finally:
                self.state.undo_move(undo)
            move_index += 1

            if score > best:
                best = score
                best_move = encoded
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self._record_cutoff(encoded, ply, depth)
                break

        if best <= original_alpha:
            flag = 2
        elif best >= original_beta:
            flag = 1
        else:
            flag = 0
        self._tt_store(key, depth, best, flag, best_move)
        return best

    def choose(self, root_moves: List[int]) -> Optional[int]:
        if not root_moves:
            return None
        if len(root_moves) == 1:
            self.completed_depth = 1
            return root_moves[0]

        root_info = self.state.analyze("all")
        ordered = self._order_moves(root_moves, root_info, -1, 0)
        best_move = ordered[0]

        for depth in range(1, MAX_SEARCH_DEPTH + 1):
            alpha = -INF
            iteration_best = ordered[0]
            iteration_score = -INF
            first = True
            try:
                for encoded in ordered:
                    if time.perf_counter() >= self.deadline:
                        raise _Timeout
                    undo = self.state.do_move(encoded)
                    try:
                        if first:
                            score = -self.negamax(depth - 1, 1, -INF, -alpha)
                            first = False
                        else:
                            score = -self.negamax(depth - 1, 1, -alpha - 1.0, -alpha)
                            if alpha < score < INF:
                                score = -self.negamax(depth - 1, 1, -INF, -alpha)
                    finally:
                        self.state.undo_move(undo)

                    if score > iteration_score:
                        iteration_score = score
                        iteration_best = encoded
                    if score > alpha:
                        alpha = score
            except _Timeout:
                break

            # 只有完整跑完的迭代才提交，避免半途结果污染最终选择。
            best_move = iteration_best
            self.root_score = iteration_score
            self.completed_depth = depth
            # 下一轮优先搜索本轮最佳着法。
            ordered.remove(best_move)
            ordered.insert(0, best_move)
            if abs(iteration_score) > WIN_SCORE * 0.5:
                break
            if time.perf_counter() >= self.deadline:
                break

        return best_move

# ---------------------------------------------------------------------------
# 对外 AI 类
# ---------------------------------------------------------------------------


class SqrtChessAI:
    """根号棋迭代加深 AI。"""

    name = "DeepSquareRoot"
    version = "2.0"

    def __init__(self, time_limit: float = DEFAULT_TIME_LIMIT, seed: Optional[int] = None):
        self.time_limit = max(0.03, float(time_limit))
        self.random = random.Random(seed)
        self.nodes = 0
        self.depth = 0
        self.score = 0.0
        self.last_info: Dict[str, Any] = {}

    def choose(self, game: Game) -> Optional[Move]:
        # 对局已结束时不应走棋。
        if game.is_over:
            return None
        legal = game.legal_moves()
        if not legal:
            return None
        if len(legal) == 1:
            self.last_info = {"depth": 1, "nodes": 0, "score": 0.0, "time_ms": 0.0}
            return legal[0]

        state = _State(game)
        root_info = state.analyze("all")
        if root_info[A_MOB][0] == 0 or root_info[A_MOB][1] == 0:
            return None
        root_moves = root_info[A_MOVES]
        # 防御性过滤：确保只搜索合法着法。
        legal_set = {move.frm * MOVE_BASE + move.to for move in legal}
        root_moves = [mv for mv in root_moves if mv in legal_set]
        if not root_moves:
            return None

        started = time.perf_counter()
        searcher = _Searcher(state, self.time_limit)
        chosen = searcher.choose(root_moves)
        elapsed = time.perf_counter() - started

        self.nodes = searcher.nodes
        self.depth = searcher.completed_depth
        self.score = searcher.root_score
        self.last_info = {
            "depth": self.depth,
            "nodes": self.nodes,
            "score": round(float(self.score), 2),
            "time_ms": round(elapsed * 1000.0, 1),
            "tt_entries": len(searcher.tt),
        }
        if chosen is None:
            return None
        candidate = Move(chosen // MOVE_BASE, chosen % MOVE_BASE)
        return candidate if game.is_legal(candidate) else legal[0]


# ---------------------------------------------------------------------------
# 兼容函数与 AI 应答协议
# ---------------------------------------------------------------------------

#: 常用别名，便于外部按不同命名习惯导入。
AI = SqrtChessAI
DeepSquareRootAI = SqrtChessAI


def choose_move(game: Game, time_limit: float = DEFAULT_TIME_LIMIT,
                seed: Optional[int] = None) -> Optional[Move]:
    """便捷函数：为给定 Game 选择一个合法着法。"""
    return SqrtChessAI(time_limit=time_limit, seed=seed).choose(game)


def _request_time_limit(request: Dict[str, Any]) -> float:
    raw = None
    for key in ("time_ms", "think_time_ms", "time_limit_ms"):
        if request.get(key) is not None:
            raw = request.get(key)
            break
    if raw is None:
        for key in ("time_limit", "seconds"):
            if request.get(key) is not None:
                try:
                    return max(0.03, float(request[key]))
                except (TypeError, ValueError):
                    pass
        return DEFAULT_TIME_LIMIT
    try:
        seconds = float(raw) / 1000.0
    except (TypeError, ValueError):
        return DEFAULT_TIME_LIMIT
    # 太短的时限也至少给 30ms，避免完全不搜索。
    return max(0.03, min(seconds, 30.0))


def answer_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """处理一条 AI 应答协议请求。"""
    kind = str(request.get("request") or request.get("cmd") or "move").strip().lower()
    if kind in ("ping", "info", "hello", "id"):
        return {
            "ok": True,
            "name": SqrtChessAI.name,
            "version": SqrtChessAI.version,
            "kind": "ai",
            "protocol": 1,
            "algorithm": "iterative-deepening alpha-beta/PVS + quiescence + custom evaluation",
        }
    if kind not in ("move", "bestmove", "go", "think"):
        return {"ok": False, "error": "未知请求: %s（支持 ping / move）" % kind}

    fen = request.get("fen") or request.get("position") or "startpos"
    game = Game.from_fen(str(fen))
    if game.is_over or not game.legal_moves():
        result = game.result.as_dict() if game.result else None
        return {"ok": True, "move": None, "reason": "当前局面没有合法着法",
                "game_over": game.is_over, "result": result}

    limit = _request_time_limit(request)
    started = time.perf_counter()
    ai = SqrtChessAI(time_limit=limit, seed=request.get("seed"))
    move = ai.choose(game)
    elapsed = (time.perf_counter() - started) * 1000.0
    if move is None:
        return {"ok": True, "move": None, "reason": "当前局面没有合法着法",
                "think_ms": round(elapsed, 1)}
    detail = game.move_dict(move)
    response = {
        "ok": True,
        "move": detail["move"],
        "notation": detail["notation"],
        "detail": detail,
        "name": SqrtChessAI.name,
        "think_ms": round(elapsed, 1),
    }
    response.update(ai.last_info)
    return response


def main(argv: Optional[List[str]] = None) -> int:
    """stdin/stdout JSON 行协议主循环。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                # 与官方示例一致：非 JSON 单行按 FEN 处理。
                request = {"request": "move", "fen": line}
            if isinstance(request, str):
                request = {"request": "move", "fen": request}
            if not isinstance(request, dict):
                raise ValueError("请求必须是 JSON 对象")
            response = answer_request(request)
        except Exception as exc:  # noqa: BLE001
            response = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
