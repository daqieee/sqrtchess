"""对局结果 → 价值目标（训练标签与搜索终局值共用同一套口径）。

* **获胜 +1，战败 -1**（不变）。
* **「一方无法动子」判和 → ±0.1**：把对手逼到无子可动的一方拿一点正分，被逼死的一方拿一点负分。
  这样网络会更看重「限制对手」，而不是只盯着吃子。
* **「100 步无吃子」判和 → 0**：双方消极避战换来的和棋不给任何奖励。
* 其它和棋（含步数上限截断后按规则 7 判为平局）→ 0。

注意：搜索（MCTS）的终局值与训练标签必须用同一套口径，否则搜索优化的目标和
网络学的目标会不一致。
"""

from __future__ import annotations

from sqrtchess import BLACK, WHITE, Game

__all__ = ["DRAW_NO_MOVES_VALUE", "stuck_side", "terminal_value", "sample_value",
           "describe_value"]

#: 「一方无法动子」判和时的分值（对称，保持零和）
DRAW_NO_MOVES_VALUE = 0.1


def stuck_side(game: Game) -> int:
    """返回「所有棋子都动不了」的一方；双方都能动或都动不了时返回 0。"""
    white_stuck = not game.has_moves(WHITE)
    black_stuck = not game.has_moves(BLACK)
    if white_stuck and not black_stuck:
        return WHITE
    if black_stuck and not white_stuck:
        return BLACK
    return 0


def terminal_value(game: Game, draw_no_moves_value: float = DRAW_NO_MOVES_VALUE) -> float:
    """终局价值（当前行棋方视角）。"""
    result = game.result
    if result is None:
        return 0.0
    if result.winner is not None:
        return 1.0 if result.winner == game.side else -1.0
    if result.reason == "no-moves" and draw_no_moves_value:
        stuck = stuck_side(game)
        if stuck:
            return -draw_no_moves_value if stuck == game.side else draw_no_moves_value
    return 0.0


def sample_value(game: Game, side_letter: str, winner, reason: str,
                 draw_no_moves_value: float = DRAW_NO_MOVES_VALUE) -> float:
    """训练样本的价值标签：side_letter 是样本所属一方（w 或 b）。"""
    if winner is not None:
        return 1.0 if (side_letter == "w") == (winner > 0) else -1.0
    if reason == "no-moves" and draw_no_moves_value:
        stuck = stuck_side(game)
        if stuck:
            i_am_stuck = (side_letter == "w") == (stuck > 0)
            return -draw_no_moves_value if i_am_stuck else draw_no_moves_value
    return 0.0


def describe_value(value: float) -> str:
    """把分值写成可读文字（日志用）。"""
    if value >= 0.999:
        return "胜 +1"
    if value <= -0.999:
        return "负 -1"
    if abs(value) < 1e-9:
        return "和 0"
    return "和 %+.2f" % value
