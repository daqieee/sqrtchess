"""根号棋（Square Root Chess）核心规则引擎。

本模块实现了根号棋的全部规则，只依赖 Python 标准库。

坐标约定
--------
* 棋盘为 11x11 的格点，x = 0..10 对应棋谱列 a..k，y = 0..10 对应第 1..11 行。
* 白方（先手）在下方（第 1、2 行），黑方在上方（第 10、11 行），双方镜像布局。
* 格点索引 sq = y * 11 + x，范围 0..120。
* 棋盘数组元素：0 空点；正数 = 白方棋子且数值为棋子大小；负数 = 黑方棋子。

规则摘要（详见 docs/RULES.md）
------------------------------
1. 布局：每方 18 枚棋子，靠棋盘边缘一行是 4 8 5 2 9 2 5 8 4，靠中心一行是 9 枚 1。
2. 走子：大小为 n 的棋子每步必须恰好移动 sqrt(n) 格，即位移 (dx, dy) 满足
   dx*dx + dy*dy == n，且落点必须在棋盘内。走子无视路径上的棋子，只看落点：
   空点可走、己方棋子不可走、敌方棋子则吃子。
3. 吃子：吃子后自身大小加上被吃棋子的大小。
4. 胜负：一方棋子被全部吃光则判负。
5. 无着可动：任意一步之后双方都还有棋子，但有一方的所有棋子都没有合法落点，进入平局判定。
6. 无吃子：双方合计连续若干步（默认 100 步，半回合）没有吃子，进入平局判定。
7. 平局判定：比较双方的棋子数量与棋子大小总和；若一方两项都严格大于另一方则该方获胜，
   否则平局。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isqrt
from typing import Dict, List, Optional, Tuple

__all__ = [
    "BOARD_SIZE", "NUM_SQUARES", "FILES", "WHITE", "BLACK", "DEFAULT_NO_CAPTURE_LIMIT",
    "INITIAL_EDGE_ROW", "INITIAL_FRONT_ROW", "SQUARE_NAMES", "REASON_TEXT",
    "IllegalMove", "GameOver", "Move", "MoveInfo", "GameResult", "Game",
    "initial_cells", "move_offsets", "is_legal_displacement", "square_name",
    "parse_square", "format_move_text", "parse_move_text", "color_name",
    "color_letter", "color_from_letter", "opponent", "perft", "perft_divide",
]

# --------------------------------------------------------------------------- 常量

BOARD_SIZE = 11
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
FILES = "abcdefghijk"
RANKS = tuple(str(i) for i in range(1, BOARD_SIZE + 1))

WHITE = 1
BLACK = -1

#: 规则 6 的默认上限：双方合计连续多少步（半回合）没有吃子就进入平局判定。
#: （11x11 棋盘较大，经对局验证默认取 100 步更合适，可用 no_capture_limit 调整。）
DEFAULT_NO_CAPTURE_LIMIT = 100

#: 初始布局中靠棋盘边缘的一行（白方第 1 行 / 黑方第 11 行）。
INITIAL_EDGE_ROW: Tuple[int, ...] = (4, 8, 5, 2, 9, 2, 5, 8, 4)
#: 初始布局中靠棋盘中心的一行（白方第 2 行 / 黑方第 10 行）。
INITIAL_FRONT_ROW: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1, 1)

SQUARE_NAMES: Tuple[str, ...] = tuple(
    FILES[sq % BOARD_SIZE] + str(sq // BOARD_SIZE + 1) for sq in range(NUM_SQUARES)
)
SQUARE_INDEX: Dict[str, int] = {name: sq for sq, name in enumerate(SQUARE_NAMES)}

#: 结束原因 -> 中文说明
REASON_TEXT: Dict[str, str] = {
    "all-captured": "吃光对方全部棋子",
    "no-moves": "有一方无子可动，进入平局判定",
    "no-capture-limit": "连续多步无吃子，进入平局判定",
    "resign": "认输",
    "agreement": "双方同意和棋",
    "unknown": "未知原因",
}


# --------------------------------------------------------------------------- 工具


def color_name(color: Optional[int]) -> str:
    """返回 1/-1 对应的中文名称。"""
    return "白方" if color == WHITE else ("黑方" if color == BLACK else "无")


def color_letter(color: Optional[int]) -> str:
    """返回 1/-1 对应的 FEN 字母（w/b）。"""
    return "w" if color == WHITE else ("b" if color == BLACK else "-")


def color_from_letter(letter) -> int:
    """把 w/b（大小写均可，也接受 white/black）转成 WHITE/BLACK。"""
    text = str(letter).strip().lower()
    if text in ("w", "white", "白", "白方"):
        return WHITE
    if text in ("b", "black", "黑", "黑方"):
        return BLACK
    raise ValueError("无法识别的行棋方: %r" % (letter,))


def opponent(color: int) -> int:
    """返回对手颜色。"""
    if color == WHITE:
        return BLACK
    if color == BLACK:
        return WHITE
    raise ValueError("颜色必须为 WHITE(1) 或 BLACK(-1)：" + repr(color))


def square_name(sq: int) -> str:
    """把格点索引转成棋谱坐标，例如 0 -> 'a1'。"""
    if not 0 <= sq < NUM_SQUARES:
        raise ValueError("格点索引越界: %r" % (sq,))
    return SQUARE_NAMES[sq]


def parse_square(text) -> int:
    """把棋谱坐标转成格点索引，例如 'a1' -> 0。"""
    name = str(text).strip().lower()
    try:
        return SQUARE_INDEX[name]
    except KeyError:
        raise ValueError("非法格点: %r" % (text,))


def initial_cells() -> List[int]:
    """构造初始局面的棋盘数组（长度 121 的 int 列表）。"""
    cells = [0] * NUM_SQUARES
    top = BOARD_SIZE - 1
    for index, size in enumerate(INITIAL_EDGE_ROW):
        x = index + 1
        cells[x] = WHITE * size                      # 白方第 1 行（靠边）
        cells[top * BOARD_SIZE + x] = BLACK * size   # 黑方第 11 行（靠边）
    for index, size in enumerate(INITIAL_FRONT_ROW):
        x = index + 1
        cells[BOARD_SIZE + x] = WHITE * size                  # 白方第 2 行
        cells[(top - 1) * BOARD_SIZE + x] = BLACK * size      # 黑方第 10 行
    return cells


_OFFSET_CACHE: Dict[int, Tuple[Tuple[int, int], ...]] = {}


def move_offsets(size: int) -> Tuple[Tuple[int, int], ...]:
    """返回大小为 size 的棋子所有可能的位移向量。

    棋子每步必须恰好移动 sqrt(size) 格，因此位移 (dx, dy) 必须满足
    dx*dx + dy*dy == size（整数解），例如：

    * 大小 1  -> 上下左右 1 格（4 种）
    * 大小 2  -> 斜向 1 格（4 种）
    * 大小 4  -> 上下左右 2 格（4 种）
    * 大小 5  -> 马步（8 种）
    * 大小 8  -> 斜向 2 格（4 种）
    * 大小 9  -> 上下左右 3 格（4 种）

    若 size 不能表示成两个整数平方和（例如 3、6、7、11），则该棋子完全没有走法。
    """
    size = int(size)
    cached = _OFFSET_CACHE.get(size)
    if cached is not None:
        return cached
    found = set()
    if size > 0:
        for dx in range(0, isqrt(size) + 1):
            rest = size - dx * dx
            dy = isqrt(rest)
            if dy * dy != rest:
                continue
            xs = (0,) if dx == 0 else (dx, -dx)
            ys = (0,) if dy == 0 else (dy, -dy)
            for sx in xs:
                for sy in ys:
                    if sx == 0 and sy == 0:
                        continue
                    found.add((sx, sy))
    result = tuple(sorted(found))
    _OFFSET_CACHE[size] = result
    return result


def is_legal_displacement(size: int, dx: int, dy: int) -> bool:
    """判断 (dx, dy) 是否为大小 size 棋子的合法位移。"""
    return dx * dx + dy * dy == size and (dx != 0 or dy != 0)


_SQUARE_PAT = r"[a-kA-K](?:11|10|[1-9])"
_MOVE_RE = re.compile(
    r"^\s*(?:(\d+)\s*)?(" + _SQUARE_PAT + r")\s*([-xX:>]?)\s*(" + _SQUARE_PAT + r")\s*$"
)


def format_move_text(frm: int, to: int, capture: bool = False, size: Optional[int] = None) -> str:
    """把着法格式化成文本。

    size 不为 None 时会在最前面带上走子前的大小（自校验记谱），例如 5b2xc3。
    """
    sep = "x" if capture else "-"
    body = square_name(frm) + sep + square_name(to)
    return ("%d%s" % (size, body)) if size else body


def parse_move_text(text) -> Tuple[int, int, Optional[int]]:
    """解析着法文本，返回 (起点, 终点, 声明的大小或 None)。

    接受 b2-b3、b2xb3、b2b3、5b2xb3 等写法（大小写、空格均可）。
    """
    match = _MOVE_RE.match(str(text))
    if not match:
        raise ValueError("无法解析的着法: %r" % (text,))
    declared = int(match.group(1)) if match.group(1) else None
    return parse_square(match.group(2)), parse_square(match.group(4)), declared


# --------------------------------------------------------------------------- 异常


class IllegalMove(ValueError):
    """非法着法。"""


class GameOver(RuntimeError):
    """对局已经结束，不能再走子。"""


# --------------------------------------------------------------------------- 着法


@dataclass(frozen=True)
class Move:
    """一步棋：从 frm 走到 to（格点索引）。"""

    frm: int
    to: int

    @property
    def dx(self) -> int:
        return self.to % BOARD_SIZE - self.frm % BOARD_SIZE

    @property
    def dy(self) -> int:
        return self.to // BOARD_SIZE - self.frm // BOARD_SIZE

    def text(self, capture: bool = False) -> str:
        """坐标记法，例如 b2-b3 或 b2xc3。"""
        return format_move_text(self.frm, self.to, capture=capture)

    def as_dict(self) -> Dict[str, object]:
        return {
            "from": square_name(self.frm),
            "to": square_name(self.to),
            "move": self.text(False),
            "dx": self.dx,
            "dy": self.dy,
        }

    def __str__(self) -> str:
        return self.text(False)


@dataclass(frozen=True)
class MoveInfo:
    """已经走出的着法记录（含吃子信息），用于棋谱、界面显示与协议输出。"""

    move: Move
    mover: int          # WHITE / BLACK
    size: int           # 走子前的大小
    captured: int       # 吃掉的敌方棋子大小，0 表示没吃子
    new_size: int       # 走子后的大小
    ply: int            # 该着法之后的半回合数（从 1 开始）
    no_capture: int     # 该着法之后的无吃子计数

    @property
    def is_capture(self) -> bool:
        return self.captured > 0

    @property
    def notation(self) -> str:
        """带大小的自校验记谱，例如 5b2xc3。"""
        return format_move_text(self.move.frm, self.move.to, self.is_capture, self.size)

    @property
    def text(self) -> str:
        """不带大小的坐标记谱，例如 b2xc3。"""
        return format_move_text(self.move.frm, self.move.to, self.is_capture)

    def describe(self) -> str:
        """中文描述，例如 白方 5b2xc3 吃 3 -> 8。"""
        head = "%s %d%s%s" % (color_name(self.mover), self.size,
                              square_name(self.move.frm), "x" if self.is_capture else "-")
        tail = square_name(self.move.to)
        if self.is_capture:
            return "%s%s 吃 %d -> %d" % (head, tail, self.captured, self.new_size)
        return "%s%s" % (head, tail)

    def as_dict(self) -> Dict[str, object]:
        return {
            "move": self.move.text(self.is_capture),
            "notation": self.notation,
            "from": square_name(self.move.frm),
            "to": square_name(self.move.to),
            "color": color_letter(self.mover),
            "size": self.size,
            "captured": self.captured,
            "new_size": self.new_size,
            "is_capture": self.is_capture,
            "ply": self.ply,
            "describe": self.describe(),
        }


# --------------------------------------------------------------------------- 结果


@dataclass(frozen=True)
class GameResult:
    """对局结果。winner 为 None 表示平局。"""

    winner: Optional[int]
    reason: str
    detail: str = ""

    @property
    def score(self) -> str:
        if self.winner == WHITE:
            return "1-0"
        if self.winner == BLACK:
            return "0-1"
        return "1/2-1/2"

    @property
    def is_draw(self) -> bool:
        return self.winner is None

    def text(self) -> str:
        why = REASON_TEXT.get(self.reason, self.reason)
        head = "和棋" if self.winner is None else (color_name(self.winner) + "胜")
        return ("%s（%s）" % (head, why)) + (("：" + self.detail) if self.detail else "")

    def as_dict(self) -> Dict[str, object]:
        return {
            "winner": color_letter(self.winner) if self.winner else None,
            "score": self.score,
            "reason": self.reason,
            "reason_text": REASON_TEXT.get(self.reason, self.reason),
            "detail": self.detail,
            "text": self.text(),
            "is_draw": self.is_draw,
        }


@dataclass
class _Undo:
    """一步棋的撤销信息。"""

    info: MoveInfo
    no_capture_before: int
    move_number_before: int
    result_before: Optional[GameResult]




# --------------------------------------------------------------------------- 局面


class Game:
    """根号棋局面与对局状态机。

    典型用法::

        from sqrtchess import Game
        game = Game.new()
        print(game.fen())
        for move in game.legal_moves():
            print(move.text())
        info = game.push("1b2-b3")
        game.pop()

    * 走子：push()；悔棋：pop()；合法性：legal_moves() / moves_from() / is_legal()
    * 局面文本：fen() / Game.from_fen()（SQFEN，见 docs/NOTATION.md）
    * 结果：result / is_over（规则 4-7 自动判定）
    """

    __slots__ = ("cells", "side", "no_capture", "move_number", "no_capture_limit",
                 "_stack", "_result", "_start_cells", "_start_side", "_start_no_capture",
                 "_start_move_number", "_start_fen")

    def __init__(self, cells=None, side: int = WHITE, no_capture: int = 0,
                 move_number: int = 1, no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT,
                 refresh: bool = True):
        board = initial_cells() if cells is None else [int(v) for v in cells]
        if len(board) != NUM_SQUARES:
            raise ValueError("棋盘长度必须为 %d，收到 %d" % (NUM_SQUARES, len(board)))
        if side not in (WHITE, BLACK):
            raise ValueError("行棋方必须为 WHITE(1) 或 BLACK(-1)")
        self.cells = board
        self.side = side
        self.no_capture = int(no_capture)
        self.move_number = int(move_number)
        self.no_capture_limit = None if no_capture_limit is None else int(no_capture_limit)
        self._stack: List[_Undo] = []
        self._result: Optional[GameResult] = None
        self._start_cells = tuple(board)
        self._start_side = side
        self._start_no_capture = self.no_capture
        self._start_move_number = self.move_number
        self._start_fen: Optional[str] = None
        if refresh:
            self._refresh_result()

    # ------------------------------------------------------------- 构造

    @classmethod
    def new(cls, **kwargs) -> "Game":
        """新建一局标准初始局面。"""
        return cls(**kwargs)

    @classmethod
    def from_fen(cls, fen: str, no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT) -> "Game":
        """从句面文本（SQFEN）载入局面。"""
        from .fen import parse_fen

        data = parse_fen(fen)
        return cls(data.cells, data.side, data.no_capture, data.move_number,
                   no_capture_limit=no_capture_limit)

    def copy(self) -> "Game":
        """复制当前局面（含历史），用于搜索等场景。"""
        clone = Game.__new__(Game)
        clone.cells = list(self.cells)
        clone.side = self.side
        clone.no_capture = self.no_capture
        clone.move_number = self.move_number
        clone.no_capture_limit = self.no_capture_limit
        clone._stack = list(self._stack)
        clone._result = self._result
        clone._start_cells = self._start_cells
        clone._start_side = self._start_side
        clone._start_no_capture = self._start_no_capture
        clone._start_move_number = self._start_move_number
        clone._start_fen = self._start_fen
        return clone

    # ------------------------------------------------------------- 基本查询

    @property
    def ply(self) -> int:
        """已经走过的半回合数。"""
        return len(self._stack)

    @property
    def result(self) -> Optional[GameResult]:
        return self._result

    @property
    def is_over(self) -> bool:
        return self._result is not None

    @property
    def side_to_move(self) -> int:
        return self.side

    @property
    def history(self) -> Tuple[MoveInfo, ...]:
        """从本局开始到当前的全部着法。"""
        return tuple(undo.info for undo in self._stack)

    @property
    def last_move(self) -> Optional[MoveInfo]:
        return self._stack[-1].info if self._stack else None

    @property
    def start_fen(self) -> str:
        """本局起始局面的 FEN（push/pop 不会改变它）。"""
        if self._start_fen is None:
            from .fen import format_fen

            self._start_fen = format_fen(self._start_cells, self._start_side,
                                         self._start_no_capture, self._start_move_number)
        return self._start_fen

    def piece_at(self, sq) -> int:
        """返回格点上的棋子（带符号的大小），0 表示空点。"""
        return self.cells[sq if isinstance(sq, int) else parse_square(sq)]

    def pieces(self, color: Optional[int] = None) -> List[Tuple[int, int, int]]:
        """返回 [(格点索引, 颜色, 大小), ...]。"""
        out = []
        for sq, value in enumerate(self.cells):
            if value:
                who = WHITE if value > 0 else BLACK
                if color is None or who == color:
                    out.append((sq, who, abs(value)))
        return out

    def piece_stats(self, color: int) -> Tuple[int, int]:
        """返回 (棋子数量, 棋子大小总和)。"""
        count = 0
        total = 0
        sign = 1 if color == WHITE else -1
        for value in self.cells:
            if value * sign > 0:
                count += 1
                total += value * sign
        return count, total

    # ------------------------------------------------------------- 走法

    def moves_from(self, sq: int) -> List[Move]:
        """返回某个格点上棋子的全部合法着法（不做行棋方检查）。"""
        value = self.cells[sq]
        if value == 0:
            return []
        x = sq % BOARD_SIZE
        y = sq // BOARD_SIZE
        positive = value > 0
        out = []
        for dx, dy in move_offsets(abs(value)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                to = ny * BOARD_SIZE + nx
                target = self.cells[to]
                if target == 0 or (target > 0) != positive:
                    out.append(Move(sq, to))
        return out

    def legal_moves(self, color: Optional[int] = None) -> List[Move]:
        """返回合法着法列表；默认返回行棋方的着法。对局结束后返回空列表。"""
        if self._result is not None:
            return []
        who = self.side if color is None else int(color)
        sign = 1 if who == WHITE else -1
        out: List[Move] = []
        for sq, value in enumerate(self.cells):
            if value * sign > 0:
                out.extend(self.moves_from(sq))
        return out

    def has_moves(self, color: Optional[int] = None) -> bool:
        """判断某一方是否至少有一个合法着法（只与棋盘有关，与胜负无关）。"""
        who = self.side if color is None else int(color)
        sign = 1 if who == WHITE else -1
        for sq, value in enumerate(self.cells):
            if value * sign > 0 and self.moves_from(sq):
                return True
        return False

    def is_legal(self, move: Move) -> bool:
        """判断着法在当前局面下是否合法（含行棋方检查）。"""
        if self._result is not None or not isinstance(move, Move):
            return False
        value = self.cells[move.frm]
        if value == 0 or (value > 0) != (self.side > 0):
            return False
        return move in self.moves_from(move.frm)

    def find_move(self, frm, to) -> Optional[Move]:
        """把起点/终点（坐标名或索引）转成合法着法，找不到返回 None。"""
        if isinstance(frm, str):
            frm = parse_square(frm)
        if isinstance(to, str):
            to = parse_square(to)
        move = Move(int(frm), int(to))
        return move if self.is_legal(move) else None

    def parse(self, text) -> Move:
        """解析并校验着法文本，返回 Move；不合法时抛出 IllegalMove。"""
        frm, to, declared = parse_move_text(text)
        value = self.cells[frm]
        if value == 0:
            raise IllegalMove("起点没有棋子：%s" % square_name(frm))
        piece_color = WHITE if value > 0 else BLACK
        if piece_color != self.side:
            raise IllegalMove("现在轮到%s走棋，不能移动%s位于 %s 的棋子"
                              % (color_name(self.side), color_name(piece_color), square_name(frm)))
        if declared is not None and declared != abs(value):
            raise IllegalMove("记谱声明的大小 %d 与 %s 上的实际大小 %d 不符"
                              % (declared, square_name(frm), abs(value)))
        move = Move(frm, to)
        if move not in self.moves_from(frm):
            raise IllegalMove("非法着法：%s（大小 %d 的棋子必须恰好移动 sqrt(%d) 格）"
                              % (move.text(), abs(value), abs(value)))
        return move

    # ------------------------------------------------------------- 走子 / 悔棋

    def push(self, move) -> MoveInfo:
        """走一步棋并返回着法记录；非法着法抛 IllegalMove，对局结束抛 GameOver。"""
        if self._result is not None:
            raise GameOver("对局已结束：" + self._result.text())
        if isinstance(move, str):
            move = self.parse(move)
        elif not isinstance(move, Move):
            if isinstance(move, (tuple, list)) and len(move) == 2:
                move = Move(int(move[0]), int(move[1]))
            else:
                raise TypeError("着法必须是 Move、坐标名或 (起点, 终点)：%r" % (move,))
        value = self.cells[move.frm]
        if value == 0:
            raise IllegalMove("起点没有棋子：%s" % square_name(move.frm))
        if (value > 0) != (self.side > 0):
            raise IllegalMove("不能移动对方的棋子：%s" % square_name(move.frm))
        if move not in self.moves_from(move.frm):
            raise IllegalMove("非法着法：%s（大小 %d 的棋子必须恰好移动 sqrt(%d) 格）"
                              % (move.text(), abs(value), abs(value)))
        target = self.cells[move.to]
        captured = abs(target) if (target != 0 and (target > 0) != (value > 0)) else 0
        size = abs(value)
        new_size = size + captured
        no_capture_after = 0 if captured else self.no_capture + 1
        info = MoveInfo(move=move, mover=self.side, size=size, captured=captured,
                        new_size=new_size, ply=len(self._stack) + 1,
                        no_capture=no_capture_after)
        self._stack.append(_Undo(info=info, no_capture_before=self.no_capture,
                                 move_number_before=self.move_number,
                                 result_before=self._result))
        self.cells[move.frm] = 0
        self.cells[move.to] = self.side * new_size
        self.no_capture = no_capture_after
        if self.side == BLACK:
            self.move_number += 1
        self.side = -self.side
        self._refresh_result()
        return info

    def pop(self) -> MoveInfo:
        """撤销最后一步棋并返回被撤销的着法记录。"""
        if not self._stack:
            raise IllegalMove("没有可以撤销的着法")
        undo = self._stack.pop()
        info = undo.info
        self.cells[info.move.frm] = info.mover * info.size
        self.cells[info.move.to] = (-info.mover * info.captured) if info.captured else 0
        self.no_capture = undo.no_capture_before
        self.move_number = undo.move_number_before
        self.side = info.mover
        self._result = undo.result_before
        return info

    def undo(self, plies: int = 1) -> List[MoveInfo]:
        """连续悔棋若干步。"""
        undone = []
        for _ in range(max(0, int(plies))):
            if not self._stack:
                break
            undone.append(self.pop())
        return undone

    def declare_result(self, winner: Optional[int], reason: str = "resign",
                       detail: str = "") -> GameResult:
        """由外部（认输、协议和棋等）宣布对局结果。"""
        if winner not in (None, WHITE, BLACK):
            raise ValueError("winner 必须是 WHITE、BLACK 或 None")
        self._result = GameResult(winner, reason, detail)
        return self._result

    # ------------------------------------------------------------- 结果判定

    def _refresh_result(self) -> Optional[GameResult]:
        """按规则 4/5/6/7 重新判定对局是否结束。"""
        white_count, _ = self.piece_stats(WHITE)
        black_count, _ = self.piece_stats(BLACK)
        if white_count == 0 and black_count == 0:
            self._result = GameResult(None, "all-captured", "双方都没有棋子")
        elif white_count == 0:
            self._result = GameResult(BLACK, "all-captured", "白方棋子被全部吃光")
        elif black_count == 0:
            self._result = GameResult(WHITE, "all-captured", "黑方棋子被全部吃光")
        elif not self.has_moves(WHITE) or not self.has_moves(BLACK):
            self._result = self.adjudicate("no-moves")
        elif self.no_capture_limit is not None and self.no_capture >= self.no_capture_limit:
            self._result = self.adjudicate("no-capture-limit")
        else:
            self._result = None
        return self._result

    def adjudicate(self, reason: str = "no-moves") -> GameResult:
        """按规则 7 进行平局判定（不改变局面，只返回结果）。"""
        white_count, white_size = self.piece_stats(WHITE)
        black_count, black_size = self.piece_stats(BLACK)
        if white_count > black_count and white_size > black_size:
            winner: Optional[int] = WHITE
        elif black_count > white_count and black_size > white_size:
            winner = BLACK
        else:
            winner = None
        detail = "白方 %d 子/大小和 %d，黑方 %d 子/大小和 %d" % (
            white_count, white_size, black_count, black_size)
        if reason == "no-moves":
            stuck = color_name(WHITE if not self.has_moves(WHITE) else BLACK)
            detail = stuck + "无子可动；" + detail
        else:
            detail = "自上次吃子起已连续 %d 步无吃子；%s" % (self.no_capture, detail)
        return GameResult(winner, reason, detail)

    # ------------------------------------------------------------- 输出

    def fen(self) -> str:
        """导出当前局面的 SQFEN 文本。"""
        from .fen import format_fen

        return format_fen(self.cells, self.side, self.no_capture, self.move_number)

    def board_text(self, flip: bool = False, marks: Optional[Dict[int, str]] = None) -> str:
        """返回棋盘的等宽文本图，便于命令行查看与日志记录。

        marks 可以把某些格点替换成自定义文本（例如上一步的 from/to）。
        """
        marks = marks or {}
        order = range(BOARD_SIZE) if flip else range(BOARD_SIZE - 1, -1, -1)
        header = "     " + "".join((" %s " % FILES[x]) for x in range(BOARD_SIZE))
        lines = [header]
        for y in order:
            row = []
            for x in range(BOARD_SIZE):
                sq = y * BOARD_SIZE + x
                value = self.cells[sq]
                if sq in marks:
                    text = str(marks[sq])
                elif value == 0:
                    text = "."
                else:
                    text = ("W" if value > 0 else "B") + str(abs(value))
                row.append(text.center(3))
            lines.append("%3d  %s" % (y + 1, "".join(row)))
        lines.append(header)
        return "\n".join(lines)

    def status(self, include_moves: bool = True) -> Dict[str, object]:
        """返回当前局面的完整状态（可直接 JSON 序列化）。"""
        white_count, white_size = self.piece_stats(WHITE)
        black_count, black_size = self.piece_stats(BLACK)
        data: Dict[str, object] = {
            "fen": self.fen(),
            "turn": color_letter(self.side),
            "turn_name": color_name(self.side),
            "move_number": self.move_number,
            "ply": self.ply,
            "no_capture": self.no_capture,
            "no_capture_limit": self.no_capture_limit,
            "white": {"pieces": white_count, "size_sum": white_size},
            "black": {"pieces": black_count, "size_sum": black_size},
            "game_over": self.is_over,
            "result": self._result.as_dict() if self._result else None,
            "last_move": self.last_move.as_dict() if self.last_move else None,
        }
        if include_moves:
            moves = [self.move_dict(move) for move in self.legal_moves()]
            data["legal_moves"] = moves
            data["legal_move_count"] = len(moves)
        return data

    def move_dict(self, move: Move) -> Dict[str, object]:
        """描述一步合法着法（含吃子信息），用于协议与界面。"""
        value = self.cells[move.frm]
        size = abs(value)
        target = self.cells[move.to]
        captured = abs(target) if (target != 0 and (target > 0) != (value > 0)) else 0
        return {
            "move": move.text(captured > 0),
            "notation": format_move_text(move.frm, move.to, captured > 0, size),
            "from": square_name(move.frm),
            "to": square_name(move.to),
            "color": color_letter(WHITE if value > 0 else BLACK),
            "size": size,
            "captured": captured,
            "new_size": size + captured,
            "is_capture": captured > 0,
        }

    def __repr__(self) -> str:
        white_count, white_size = self.piece_stats(WHITE)
        black_count, black_size = self.piece_stats(BLACK)
        return "<Game ply=%d turn=%s white=%d/%d black=%d/%d%s>" % (
            self.ply, color_letter(self.side), white_count, white_size,
            black_count, black_size, (" over:" + self._result.score) if self._result else "")


# --------------------------------------------------------------------------- perft


def perft(game: Game, depth: int) -> int:
    """统计从当前局面出发 depth 步之内的着法序列总数（走法生成自检用）。"""
    if depth <= 0 or game.is_over:
        return 1
    total = 0
    for move in game.legal_moves():
        game.push(move)
        total += perft(game, depth - 1)
        game.pop()
    return total


def perft_divide(game: Game, depth: int) -> Dict[str, int]:
    """按第一步着法分组的 perft 统计。"""
    result: Dict[str, int] = {}
    if depth <= 0 or game.is_over:
        return result
    for move in game.legal_moves():
        game.push(move)
        result[move.text()] = perft(game, depth - 1)
        game.pop()
    return result
