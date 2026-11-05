"""AlphaZero 的局面编码。

输入张量形状固定为 (PLANES, 11, 11)，每个格点 10 个通道：

===== ============================================
通道   含义
===== ============================================
0      白方棋子（0/1）
1      黑方棋子（0/1）
2      白方棋子大小 / 112
3      黑方棋子大小 / 112
4      该点棋子的合法着法数 / 8
5      该点棋子是否完全无法移动（大小为 3、6、7 之类，0/1）
6      是否轮到白方（全局广播）
7      无吃子进度 no_capture / no_capture_limit（全局广播）
8      白方棋子大小总和 / 112（全局广播）
9      黑方棋子大小总和 / 112（全局广播）
===== ============================================

网络输出的策略是「对当前局面的合法着法列表打分」，所以着法不需要固定动作空间编码，
只要求先验概率与 legal_moves() 的顺序一一对应即可。
"""

from __future__ import annotations

from typing import List

import numpy as np

from sqrtchess import (BOARD_SIZE, DEFAULT_NO_CAPTURE_LIMIT, Game, Move, square_name)

PLANES = 10
MAX_TOTAL = 112.0     # 双方全部棋子大小之和
MAX_MOBILITY = 8.0    # 单点最大着法数（大小 5/10/13 的马步）

__all__ = ["PLANES", "MAX_TOTAL", "MAX_MOBILITY", "encode", "legal_moves", "move_text",
           "moves_from_notation"]


def encode(game: Game, out: np.ndarray = None) -> np.ndarray:
    """把局面编码成 (PLANES, 11, 11) 的 float32 张量。"""
    if out is None:
        out = np.zeros((PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    else:
        out.fill(0.0)

    cells = game.cells
    white_total = 0
    black_total = 0
    for sq in range(BOARD_SIZE * BOARD_SIZE):
        value = cells[sq]
        if value == 0:
            continue
        x = sq % BOARD_SIZE
        y = sq // BOARD_SIZE
        size = abs(value)
        moves = len(game.moves_from(sq))
        if value > 0:
            out[0, y, x] = 1.0
            out[2, y, x] = size / MAX_TOTAL
            white_total += size
        else:
            out[1, y, x] = 1.0
            out[3, y, x] = size / MAX_TOTAL
            black_total += size
        out[4, y, x] = moves / MAX_MOBILITY if moves else 0.0
        out[5, y, x] = 1.0 if moves == 0 else 0.0

    limit = game.no_capture_limit or DEFAULT_NO_CAPTURE_LIMIT
    progress = min(1.0, game.no_capture / float(limit)) if limit else 0.0
    if game.side > 0:
        out[6, :, :] = 1.0
    out[7, :, :] = progress
    out[8, :, :] = white_total / MAX_TOTAL
    out[9, :, :] = black_total / MAX_TOTAL
    return out


def legal_moves(game: Game) -> List[Move]:
    """当前局面的合法着法（与网络策略输出一一对应）。"""
    return game.legal_moves()


def move_text(game: Game, move: Move) -> str:
    """着法的自校验记谱（用于棋谱与策略字典的键）。"""
    return game.move_dict(move)["notation"]


def moves_from_notation(game: Game, notation: str) -> Move:
    """把记谱还原成合法着法。"""
    return game.parse(notation)
