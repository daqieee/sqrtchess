"""命令行版「斗蛐蛐」：让两个外部 AI 程序轮流执先对打若干局，统计比分。

外部程序需实现本项目的 AI 应答协议（每行一个 JSON）：

    请求  {"request": "move", "fen": "...", "time_ms": 300, "level": "greedy"}
    响应  {"ok": true, "move": "f2-f3", "notation": "1f2-f3"}

用法（在包根目录）：

    python duel.py                       # 默认 ai-deepseek vs ai-gpt，各执先 2 局
    python duel.py --games 6 --time-ms 500
    python duel.py --white "python -B ai-gpt/my_work.py" --black "python -m sqrtchess ai --level search2"
    python duel.py --out duel.sqg        # 把全部对局存成棋谱
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqrtchess import (DEFAULT_NO_CAPTURE_LIMIT, Game, Recorder,  # noqa: E402
                       default_headers)

__all__ = ["ExternalProcess", "play_duel"]


class ExternalProcess:
    """把外部 AI 程序包成「给一个 FEN，返回一个着法」。"""

    def __init__(self, command: str, name: str = "外部AI", timeout: float = 30.0):
        self.command = command
        self.name = name
        self.timeout = timeout
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"]
                                        if env.get("PYTHONPATH") else "")
        self.process = subprocess.Popen(
            command, shell=True, cwd=str(ROOT), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.errors: List[str] = []
        threading.Thread(target=self._pump_out, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()

    def _pump_out(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.queue.put(line)
        self.queue.put("")

    def _pump_err(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.errors.append(line.rstrip())
            del self.errors[:-30]

    def ask(self, fen: str, time_ms: int) -> Tuple[str, Optional[dict]]:
        if self.process.poll() is not None:
            raise RuntimeError("%s 已退出：%s" % (self.name, self.errors[-1] if self.errors else ""))
        assert self.process.stdin is not None
        payload = {"request": "move", "fen": fen, "time_ms": int(time_ms), "level": "greedy"}
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        try:
            line = self.queue.get(timeout=max(self.timeout, time_ms / 1000.0 * 10 + 10))
        except queue.Empty:
            raise RuntimeError("%s 没有响应（超过 %.0f 秒）" % (self.name, self.timeout))
        if not line:
            raise RuntimeError("%s 已退出：%s" % (self.name, self.errors[-1] if self.errors else ""))
        data = json.loads(line)
        if not data.get("ok", True):
            raise RuntimeError("%s 返回错误：%s" % (self.name, data.get("error")))
        move = data.get("notation") or data.get("move") or ""
        return str(move), data

    def close(self) -> None:
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.process.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.process.terminate()
        except Exception:  # noqa: BLE001
            pass


def _label(command: str) -> str:
    low = command.lower()
    if "deepseek" in low:
        return "DeepSeek"
    if "gpt" in low or "rootknight" in low:
        return "GPT"
    if "claude" in low or "strong_ai" in low:
        return "Claude"
    for token in command.replace('"', " ").split():
        if token.lower().endswith(".py"):
            name = os.path.basename(token)[:-3]
            if name:
                return name
    return "外部AI"


def play_duel(white_cmd: str, black_cmd: str, games: int = 4, time_ms: int = 300,
              limit: Optional[int] = DEFAULT_NO_CAPTURE_LIMIT, max_plies: int = 400,
              out_path: Optional[str] = None, seed: int = 1,
              quiet: bool = False) -> Dict[str, float]:
    white_ai = ExternalProcess(white_cmd, _label(white_cmd))
    black_ai = ExternalProcess(black_cmd, _label(black_cmd))
    scores: Dict[str, float] = {white_ai.name: 0.0, black_ai.name: 0.0}
    records: List[str] = []
    started_all = time.time()
    try:
        for index in range(1, games + 1):
            same = (index % 2 == 1)
            first = white_ai if same else black_ai
            second = black_ai if same else white_ai
            game = Game.new(no_capture_limit=limit)
            recorder = Recorder(game, default_headers(
                white=first.name, black=second.name, event="斗蛐蛐 第%d局" % index))
            started = time.time()
            error = ""
            while not game.is_over and game.ply < max_plies:
                side = first if game.side == 1 else second
                try:
                    notation, _ = side.ask(game.fen(), time_ms)
                except Exception as exc:  # noqa: BLE001
                    error = "%s 出错：%s" % (side.name, exc)
                    break
                notation = (notation or "").strip()
                if not notation:
                    error = "%s 没有返回着法" % side.name
                    break
                legal = {game.move_dict(move)["notation"] for move in game.legal_moves()}
                if notation not in legal:
                    shorter = notation.lstrip("0123456789")
                    if shorter in legal:
                        notation = shorter
                    else:
                        error = "%s 返回非法着法 %s" % (side.name, notation)
                        break
                recorder.push(notation)
            elapsed = time.time() - started
            if error:
                print("第 %d 局：%s" % (index, error))
                break
            if game.result is not None:
                winner = game.result.winner
                label = game.result.text()
                reason = game.result.detail
            else:
                outcome = game.adjudicate("truncated")
                winner = outcome.winner
                label = "截断判定：" + outcome.detail
                reason = outcome.detail
                # 把裁定结果写回局面，这样棋谱里 Result 字段不会留成 *（便于事后统计）
                game.declare_result(winner, "truncated-adjudicated", outcome.detail)
            if winner is None or winner == 0:
                first_point = 0.5
            elif winner == (1 if same else -1):
                first_point = 1.0
            else:
                first_point = 0.0
            second_point = 1.0 - first_point
            scores[first.name] = scores.get(first.name, 0.0) + first_point
            scores[second.name] = scores.get(second.name, 0.0) + second_point
            records.append(recorder.finish())
            if not quiet:
                print("第 %d 局（白=%s 黑=%s）：%s｜%d 步｜%.1f 秒"
                      % (index, first.name, second.name, label, game.ply, elapsed))
    finally:
        white_ai.close()
        black_ai.close()

    print()
    parts = ["%s %.1f" % (name, value) for name, value in
             sorted(scores.items(), key=lambda kv: -kv[1])]
    print("比分（%d 局）：%s" % (games, " : ".join(parts)))
    if games > 0:
        print("总用时 %.1f 秒" % (time.time() - started_all))
    if out_path and records:
        with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n\n".join(records))
        print("棋谱已写入：%s" % out_path)
    return scores


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    parser = argparse.ArgumentParser(
        prog="duel.py", description="让两个外部 AI 程序对打若干局（斗蛐蛐）。")
    parser.add_argument("--white", default="python -B \"%s\"" % (ROOT / "ai-deepseek" / "my_work.py"),
                        help="白方外部程序命令行")
    parser.add_argument("--black", default="python -B \"%s\"" % (ROOT / "ai-gpt" / "my_work.py"),
                        help="黑方外部程序命令行")
    parser.add_argument("--games", type=int, default=4, help="总局数（轮流执先，默认 4）")
    parser.add_argument("--time-ms", type=int, default=300, help="每步思考时限（毫秒，默认 300）")
    parser.add_argument("--limit", default=str(DEFAULT_NO_CAPTURE_LIMIT),
                        help="无吃子步数上限，off 表示不限制（默认 100）")
    parser.add_argument("--max-plies", type=int, default=400, help="单局步数上限（默认 400）")
    parser.add_argument("--out", default="", help="把全部对局写入棋谱文件")
    parser.add_argument("--quiet", action="store_true", help="只输出比分")
    args = parser.parse_args(argv)
    limit = None if str(args.limit).lower() in ("off", "none", "不限") else int(args.limit)
    play_duel(args.white, args.black, games=max(1, args.games), time_ms=args.time_ms,
              limit=limit, max_plies=args.max_plies, out_path=args.out or None,
              quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
