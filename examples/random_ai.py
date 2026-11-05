"""示例：一个自定义根号棋 AI（演示 AI 应答协议）。

用法::

    python examples/random_ai.py            # 作为外部 AI，等待 JSON 行请求

在根号棋界面里把白方或黑方设为“外部程序”，命令行填：

    python examples/random_ai.py

然后它的走子逻辑就会出现在对局里。协议说明见 docs/PROTOCOL.md：
每行一个 JSON 请求 {"request": "move", "fen": "..."}，回答一行
{"ok": true, "move": "5b2xc3"}。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqrtchess import Game  # noqa: E402

RNG = random.Random(20261009)


def choose(game: Game):
    """在这里实现你自己的思路：输入局面，返回一个合法着法（Game 的 Move）。"""
    moves = game.legal_moves()
    if not moves:
        return None
    # 示例策略：能吃子就吃最大的，否则随机走一步。
    captures = [move for move in moves if game.cells[move.to] != 0]
    if captures:
        return max(captures, key=lambda move: abs(game.cells[move.to]))
    return RNG.choice(moves)


def answer(request: dict) -> dict:
    kind = str(request.get("request") or "move").lower()
    if kind in ("ping", "info", "hello"):
        return {"ok": True, "name": "example-random-ai", "kind": "ai", "protocol": 1}
    game = Game.from_fen(request.get("fen") or "startpos")
    move = choose(game)
    if move is None:
        return {"ok": True, "move": None, "reason": "没有合法着法"}
    detail = game.move_dict(move)
    return {"ok": True, "move": detail["move"], "notation": detail["notation"],
            "detail": detail, "name": "example-random-ai"}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            request = {"request": "move", "fen": line}
        try:
            response = answer(request)
        except Exception as exc:  # noqa: BLE001
            response = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
