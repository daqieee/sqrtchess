"""根号棋局面文本（SQFEN，Square-root Chess FEN）。

格式
----
    棋盘 行棋方 无吃子步数 回合数

* 棋盘：11 行（第 11 行在最前，第 1 行在最后）用 / 分隔，每行恰好 11 个格点记号。
  格点记号为 .（空点）或 W<大小> / B<大小>（大小用十进制，W = 白方，B = 黑方）。
* 行棋方：w（白方）或 b（黑方）。
* 无吃子步数：双方合计连续没有吃子的步数（半回合），规则 6 用。
* 回合数：从 1 开始，黑方走完一步后 +1。

初始局面（为避免歧义，此处用 ▼ 标出白方一侧）::

    .........../.........../.........../.........../.........../.........../
    .........../.........../.........../.W1W1W1W1W1W1W1W1W1./.W4W8W5W2W9W2W5W8W4. w 0 1

示例::

    >>> from sqrtchess import START_FEN, Game
    >>> START_FEN.startswith("...........")
    True
    >>> Game.from_fen(START_FEN).fen() == START_FEN
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .core import (BOARD_SIZE, NUM_SQUARES, WHITE, color_from_letter,
                   color_letter, initial_cells)

__all__ = ["FenError", "FenData", "START_FEN", "format_fen", "parse_fen",
           "format_board_field", "parse_board_field", "START_POSITION_ALIASES"]

EMPTY_CHARS = ".-*_"


class FenError(ValueError):
    """局面文本非法。"""


@dataclass(frozen=True)
class FenData:
    """解析后的局面文本。"""

    cells: List[int]
    side: int = WHITE
    no_capture: int = 0
    move_number: int = 1

    def fen(self) -> str:
        return format_fen(self.cells, self.side, self.no_capture, self.move_number)


def format_board_field(cells: Sequence[int]) -> str:
    """把棋盘数组格式化成 SQFEN 的棋盘字段。"""
    if len(cells) != NUM_SQUARES:
        raise FenError("棋盘长度必须为 %d，收到 %d" % (NUM_SQUARES, len(cells)))
    ranks = []
    for y in range(BOARD_SIZE - 1, -1, -1):
        tokens = []
        for x in range(BOARD_SIZE):
            value = int(cells[y * BOARD_SIZE + x])
            if value == 0:
                tokens.append(".")
            else:
                tokens.append(("W" if value > 0 else "B") + str(abs(value)))
        ranks.append("".join(tokens))
    return "/".join(ranks)


def parse_board_field(text: str) -> List[int]:
    """解析 SQFEN 的棋盘字段，返回棋盘数组。"""
    ranks = str(text).strip().split("/")
    if len(ranks) != BOARD_SIZE:
        raise FenError("棋盘必须为 %d 行，收到 %d 行" % (BOARD_SIZE, len(ranks)))
    cells = [0] * NUM_SQUARES
    for index, rank_text in enumerate(ranks):
        y = BOARD_SIZE - 1 - index
        x = 0
        position = 0
        while position < len(rank_text):
            ch = rank_text[position]
            if ch in EMPTY_CHARS:
                x += 1
                position += 1
            elif ch in "WwBb":
                scan = position + 1
                digits = ""
                while scan < len(rank_text) and rank_text[scan].isdigit():
                    digits += rank_text[scan]
                    scan += 1
                if not digits:
                    raise FenError("第 %d 行的 %r 后面缺少棋子大小" % (y + 1, ch))
                size = int(digits)
                if size < 1:
                    raise FenError("第 %d 行出现非法棋子大小 %d" % (y + 1, size))
                if x >= BOARD_SIZE:
                    raise FenError("第 %d 行的格点数量超过 %d" % (y + 1, BOARD_SIZE))
                cells[y * BOARD_SIZE + x] = size if ch in "Ww" else -size
                x += 1
                position = scan
            else:
                raise FenError("第 %d 行出现无法识别的字符 %r" % (y + 1, ch))
        if x != BOARD_SIZE:
            raise FenError("第 %d 行共有 %d 个格点，应为 %d 个" % (y + 1, x, BOARD_SIZE))
    return cells


def format_fen(cells: Sequence[int], side: int = WHITE, no_capture: int = 0,
               move_number: int = 1) -> str:
    """把局面格式化成 SQFEN 文本。"""
    return "%s %s %d %d" % (format_board_field(cells), color_letter(side),
                            int(no_capture), int(move_number))


START_POSITION_ALIASES = ("startpos", "start", "initial", "开局", "初始局面")


def parse_fen(text: str) -> FenData:
    """解析 SQFEN 文本；也接受 startpos 等别名。"""
    if text is None:
        raise FenError("局面文本为空")
    raw = str(text).strip()
    if not raw:
        raise FenError("局面文本为空")
    if raw.lower() in START_POSITION_ALIASES:
        return FenData(initial_cells(), WHITE, 0, 1)
    fields = raw.split()
    if len(fields) < 1:
        raise FenError("局面文本为空")
    cells = parse_board_field(fields[0])
    side = WHITE
    if len(fields) >= 2:
        try:
            side = color_from_letter(fields[1])
        except ValueError as exc:
            raise FenError(str(exc))
    no_capture = 0
    if len(fields) >= 3:
        try:
            no_capture = int(fields[2])
        except ValueError:
            raise FenError("无吃子步数必须是整数：%r" % (fields[2],))
        if no_capture < 0:
            raise FenError("无吃子步数不能为负数：%d" % no_capture)
    move_number = 1
    if len(fields) >= 4:
        try:
            move_number = int(fields[3])
        except ValueError:
            raise FenError("回合数必须是整数：%r" % (fields[3],))
        if move_number < 1:
            raise FenError("回合数必须大于 0：%d" % move_number)
    return FenData(cells, side, no_capture, move_number)


START_FEN = format_fen(initial_cells(), WHITE, 0, 1)
