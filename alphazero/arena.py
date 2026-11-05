"""对局评估：让两个选手互相对战，统计胜率与物质差。

AlphaZero 的「门控」就用这里的结果：新网络必须在对战中达到阈值分数才会被采纳为 best。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sqrtchess import BLACK, DEFAULT_NO_CAPTURE_LIMIT, Game, WHITE

__all__ = ["MatchResult", "play_match"]


@dataclass
class MatchResult:
    """一次对战的结果。"""

    games: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    score_a: float = 0.0          # (胜 + 0.5 平) / 局数
    material_a: float = 0.0       # 平均「棋子大小总和之差」（A 视角）
    avg_plies: float = 0.0
    seconds: float = 0.0
    reasons: Dict[str, int] = field(default_factory=dict)
    errors: int = 0

    def summary(self) -> str:
        return ("%d 局: %d 胜 %d 负 %d 和，得分率 %.3f，平均大小差 %+.1f，平均 %d 步，用时 %.1fs"
                % (self.games, self.wins_a, self.wins_b, self.draws, self.score_a,
                   self.material_a, int(self.avg_plies), self.seconds))

    def to_dict(self) -> Dict:
        return {"games": self.games, "wins_a": self.wins_a, "wins_b": self.wins_b,
                "draws": self.draws, "score_a": round(self.score_a, 4),
                "material_a": round(self.material_a, 2),
                "avg_plies": round(self.avg_plies, 1), "seconds": round(self.seconds, 2),
                "reasons": dict(self.reasons), "errors": self.errors}


def play_match(player_a, player_b, games: int = 10, max_plies: int = 400,
               limit=DEFAULT_NO_CAPTURE_LIMIT, seed: int = 0,
               swap_colors: bool = True, adjudicate_truncated: bool = True) -> MatchResult:
    """让 A 与 B 对战若干局（默认交替先后手）。

    adjudicate_truncated=True 时，达到步数上限的对局按棋规第 7 条判定领先方，
    这样评估结果能区分「谁占优」，不会退化成一大堆无信息的和棋。
    """
    result = MatchResult()
    started = time.time()
    total_plies = 0
    for index in range(games):
        a_is_white = (index % 2 == 0) if swap_colors else True
        game = Game.new(no_capture_limit=limit)
        while not game.is_over and game.ply < max_plies:
            player = player_a if (game.side == WHITE) == a_is_white else player_b
            try:
                move = player.choose(game)
            except Exception:  # noqa: BLE001 - 选手出错按判负处理
                result.errors += 1
                move = None
            if move is None or not game.is_legal(move):
                result.errors += 1
                break
            game.push(move)
        total_plies += game.ply
        if game.result is not None:
            winner = game.result.winner
            reason = game.result.reason
        elif adjudicate_truncated:
            # 到达步数上限：按棋规第 7 条判定领先方，避免大量无信息的和棋
            outcome = game.adjudicate("no-capture-limit")
            winner = outcome.winner
            reason = "truncated-adjudicated"
        else:
            winner = None
            reason = "truncated"
        result.reasons[reason] = result.reasons.get(reason, 0) + 1
        white_size = game.piece_stats(WHITE)[1]
        black_size = game.piece_stats(BLACK)[1]
        result.material_a += (white_size - black_size) * (1 if a_is_white else -1)
        if winner is None:
            result.draws += 1
            result.score_a += 0.5
        elif (winner == WHITE) == a_is_white:
            result.wins_a += 1
            result.score_a += 1.0
        else:
            result.wins_b += 1
    result.games = games
    result.seconds = time.time() - started
    result.avg_plies = total_plies / max(1, games)
    result.score_a /= max(1, games)
    result.material_a /= max(1, games)
    return result
