"""根号棋命令行入口。

用法::

    python -m sqrtchess fen                     # 打印初始局面 FEN
    python -m sqrtchess board --fen "..."       # 打印棋盘
    python -m sqrtchess moves                   # 列出合法着法
    python -m sqrtchess proto                   # 控制协议（stdin/stdout，JSON 行）
    python -m sqrtchess ai --level greedy       # AI 应答协议（stdin/stdout）
    python -m sqrtchess serve --port 8765       # TCP 服务
    python -m sqrtchess demo --white greedy --black random --out game.sqg
    python -m sqrtchess replay game.sqg         # 回放并校验棋谱
    python -m sqrtchess perft --depth 3         # 走法生成计数
    python -m sqrtchess selfcheck               # 内置自检
    python -m sqrtchess gui                     # 启动图形界面
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from ._version import ENGINE_NAME, RULE_TAG, __version__
from .ai import make_ai
from .core import (BLACK, DEFAULT_NO_CAPTURE_LIMIT, WHITE, Game, color_name,
                   perft, perft_divide, square_name)
from .fen import START_FEN, FenError
from .protocol import dumps, run_ai_stdio, run_stdio, serve_forever
from .record import Recorder, RecordError, default_headers, replay


def _parse_limit(text: str) -> Optional[int]:
    """把 --limit 参数转成整数；0 / off / none 表示不限制。"""
    value = str(text).strip().lower()
    if value in ("0", "off", "none", "no", "无", "不限"):
        return None
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("必须是整数或 off")
    if number < 1:
        raise argparse.ArgumentTypeError("必须大于 0，或使用 off 表示不限制")
    return number


def _fix_console() -> None:
    """避免在 GBK 控制台上因为少数符号而崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _read_text(path: str) -> str:
    if path in ("-", ""):
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


# --------------------------------------------------------------------------- 命令


def cmd_fen(args) -> int:
    if args.fen:
        game = Game.from_fen(args.fen, no_capture_limit=args.limit)
    else:
        game = Game.new(no_capture_limit=args.limit)
    print(game.fen())
    return 0


def cmd_board(args) -> int:
    game = Game.from_fen(args.fen or START_FEN, no_capture_limit=args.limit)
    print(game.board_text(flip=args.flip))
    white_count, white_size = game.piece_stats(WHITE)
    black_count, black_size = game.piece_stats(BLACK)
    print("轮到: %s   白方 %d 子/大小和 %d   黑方 %d 子/大小和 %d   无吃子计数: %d"
          % (color_name(game.side), white_count, white_size, black_count, black_size,
             game.no_capture))
    if game.result:
        print("对局结果: " + game.result.text())
    print("FEN: " + game.fen())
    return 0


def cmd_moves(args) -> int:
    game = Game.from_fen(args.fen or START_FEN, no_capture_limit=args.limit)
    moves = game.legal_moves()
    if args.json:
        print(dumps({"ok": True, "count": len(moves),
                     "moves": [game.move_dict(move) for move in moves]}))
        return 0
    print("%s可走 %d 步：" % (color_name(game.side), len(moves)))
    for index, move in enumerate(moves, 1):
        info = game.move_dict(move)
        print("%3d. %-10s %s" % (index, info["notation"], square_name(move.frm)))
    return 0


def cmd_proto(args) -> int:
    return run_stdio(no_capture_limit=args.limit)


def cmd_ai(args) -> int:
    return run_ai_stdio(default_level=args.level, no_capture_limit=args.limit)


def cmd_serve(args) -> int:
    def ready(server):
        host, port = server.server_address[:2]
        print("根号棋引擎已启动：%s:%s（每行一个 JSON 请求，Ctrl+C 退出）" % (host, port))
        print("控制协议示例: {\"cmd\": \"moves\"}")
        print("AI 协议示例:   {\"request\": \"move\", \"fen\": \"...\"}")
        sys.stdout.flush()

    try:
        serve_forever(args.host, args.port, no_capture_limit=args.limit, on_ready=ready)
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


def cmd_demo(args) -> int:
    game = Game.new(no_capture_limit=args.limit)
    headers = default_headers(white=args.white, black=args.black, event=args.event)
    recorder = Recorder(game, headers)
    white_ai = make_ai(args.white, seed=args.seed)
    black_ai = make_ai(args.black, seed=None if args.seed is None else args.seed + 1)
    plies = 0
    while not game.is_over and plies < args.plies:
        engine = white_ai if game.side == WHITE else black_ai
        move = engine.choose(game)
        if move is None:
            break
        info = recorder.push(move)
        plies += 1
        if not args.quiet:
            print("%4d. %-14s %s" % (info.ply, info.notation, info.describe()))
    text = recorder.finish()
    if not args.quiet:
        print()
        print(game.board_text())
    print(text if not args.quiet else text, end="")
    if game.result:
        print("# " + game.result.text(), file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print("棋谱已保存: %s" % args.out, file=sys.stderr)
    return 0


def cmd_replay(args) -> int:
    try:
        text = _read_text(args.file)
    except OSError as exc:
        print("无法读取棋谱: %s" % exc, file=sys.stderr)
        return 2
    try:
        report = replay(text, no_capture_limit=args.limit)
    except RecordError as exc:
        print("棋谱格式错误: %s" % exc, file=sys.stderr)
        return 2
    if args.json:
        print(dumps({"ok": report.ok, "summary": report.summary(),
                     "moves": [info.as_dict() for info in report.moves],
                     "fens": report.fens, "errors": report.errors,
                     "warnings": report.warnings,
                     "result": report.game.result.as_dict() if report.game.result else None}))
    else:
        print(report.summary())
    if args.list:
        for index, info in enumerate(report.moves, 1):
            print("%4d. %-14s %s" % (index, info.notation, info.describe()))
    if args.board:
        print()
        print(report.game.board_text())
    return 0 if report.ok else 1


def cmd_perft(args) -> int:
    game = Game.from_fen(args.fen or START_FEN, no_capture_limit=args.limit)
    if args.divide:
        table = perft_divide(game, args.depth)
        for key in sorted(table):
            print("%-10s %d" % (key, table[key]))
        print("总计 %d" % sum(table.values()))
        return 0
    import time

    started = time.time()
    nodes = perft(game, args.depth)
    elapsed = time.time() - started
    print("perft(%d) = %d  用时 %.3f 秒 (%.0f 节点/秒)"
          % (args.depth, nodes, elapsed, nodes / elapsed if elapsed else 0))
    return 0


def cmd_selfcheck(args) -> int:
    """不依赖测试框架的快速自检。"""
    import random

    failures: List[str] = []

    def check(name: str, ok: bool, extra: str = "") -> None:
        print("[%s] %s%s" % ("OK  " if ok else "FAIL", name, ("  " + extra) if extra else ""))
        if not ok:
            failures.append(name)

    game = Game.new()
    check("初始局面 FEN 往返", Game.from_fen(game.fen()).fen() == game.fen())
    check("初始着法数 = 22", len(game.legal_moves()) == 22,
          "实际 %d" % len(game.legal_moves()))
    check("双方各 18 子 / 大小和 56",
          game.piece_stats(WHITE) == (18, 56) and game.piece_stats(BLACK) == (18, 56))

    rng = random.Random(args.seed)
    ok_fen = ok_record = True
    for _ in range(args.games):
        trial = Game.new(no_capture_limit=args.limit)
        while not trial.is_over and trial.ply < args.plies:
            moves = trial.legal_moves()
            if not moves:
                break
            trial.push(rng.choice(moves))
            if Game.from_fen(trial.fen()).fen() != trial.fen():
                ok_fen = False
                break
        check("随机对局 %d 步后 FEN 往返" % trial.ply, ok_fen)
        recorder = Recorder(Game.new(no_capture_limit=args.limit))
        trial2 = recorder.game
        while not trial2.is_over and trial2.ply < args.plies:
            moves = trial2.legal_moves()
            if not moves:
                break
            recorder.push(rng.choice(moves))
        text = recorder.finish()
        try:
            report = replay(text, no_capture_limit=args.limit)
            ok_record = report.ok and len(report.moves) == len(recorder.moves)
        except Exception as exc:  # pragma: no cover
            ok_record = False
            print("   回放异常: %s" % exc)
        check("棋谱导出与回放一致", ok_record)
        break

    depth = args.depth
    nodes = perft(Game.new(), depth)
    check("perft(%d) = %d" % (depth, nodes), nodes > 0)

    print()
    if failures:
        print("自检失败 %d 项：%s" % (len(failures), "，".join(failures)))
        return 1
    print("自检全部通过。")
    return 0


def run_gui(extra: Optional[List[str]] = None) -> int:
    """启动图形界面，extra 为透传给界面的参数（如 --selftest、--limit 30）。"""
    import runpy

    path = Path(__file__).resolve().parent.parent / "gui" / "sqrtchess_gui.py"
    if not path.exists():
        print("未找到图形界面文件: %s" % path, file=sys.stderr)
        return 2
    sys.argv = [str(path)] + list(extra or [])
    runpy.run_path(str(path), run_name="__main__")
    return 0


def cmd_gui(args) -> int:
    return run_gui(list(getattr(args, "args", None) or []))


# --------------------------------------------------------------------------- 解析器


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sqrtchess",
        description="根号棋（Square Root Chess）引擎 %s，规则版本 %s。" % (__version__, RULE_TAG),
        epilog="示例: python -m sqrtchess demo --white greedy --black random",
    )
    parser.add_argument("--version", action="version",
                        version="%s %s" % (ENGINE_NAME, __version__))
    subparsers = parser.add_subparsers(dest="command", metavar="命令")

    def add(name: str, help_text: str):
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("--limit", type=_parse_limit, default=DEFAULT_NO_CAPTURE_LIMIT,
                         help="无吃子步数上限（默认 100，off 表示不限制）")
        return sub

    sub = add("fen", "打印/规范化局面文本（SQFEN）")
    sub.add_argument("--fen", help="输入局面文本，默认初始局面")
    sub.set_defaults(func=cmd_fen)

    sub = add("board", "用文本图打印棋盘")
    sub.add_argument("--fen", help="局面文本，默认初始局面")
    sub.add_argument("--flip", action="store_true", help="黑白翻转显示")
    sub.set_defaults(func=cmd_board)

    sub = add("moves", "列出当前局面的全部合法着法")
    sub.add_argument("--fen", help="局面文本，默认初始局面")
    sub.add_argument("--json", action="store_true", help="以 JSON 输出")
    sub.set_defaults(func=cmd_moves)

    sub = add("proto", "控制协议：stdin/stdout 上的 JSON 行接口")
    sub.set_defaults(func=cmd_proto)

    sub = add("ai", "AI 应答协议：每行一个 JSON 请求，返回一步着法")
    sub.add_argument("--level", default="greedy",
                     help="AI 档位: random / greedy / search2 / search3（默认 greedy）")
    sub.set_defaults(func=cmd_ai)

    sub = add("serve", "启动 TCP 服务（同时支持控制协议与 AI 协议）")
    sub.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    sub.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    sub.set_defaults(func=cmd_serve)

    sub = add("demo", "让两个 AI 自对弈并输出棋谱")
    sub.add_argument("--white", default="greedy", help="白方 AI（默认 greedy）")
    sub.add_argument("--black", default="random", help="黑方 AI（默认 random）")
    sub.add_argument("--plies", type=int, default=400, help="最多走多少步（默认 400）")
    sub.add_argument("--seed", type=int, default=None, help="随机种子")
    sub.add_argument("--out", help="把棋谱写入文件")
    sub.add_argument("--event", default="根号棋自对弈", help="棋谱 Event 字段")
    sub.add_argument("--quiet", action="store_true", help="只输出棋谱")
    sub.set_defaults(func=cmd_demo)

    sub = add("replay", "回放并校验棋谱文件")
    sub.add_argument("file", help="棋谱文件路径，- 表示标准输入")
    sub.add_argument("--list", action="store_true", help="列出每一步")
    sub.add_argument("--board", action="store_true", help="打印最终棋盘")
    sub.add_argument("--json", action="store_true", help="以 JSON 输出")
    sub.set_defaults(func=cmd_replay)

    sub = add("perft", "走法生成计数（自检用）")
    sub.add_argument("--depth", type=int, default=3, help="深度（默认 3）")
    sub.add_argument("--fen", help="局面文本，默认初始局面")
    sub.add_argument("--divide", action="store_true", help="按第一步着法分组")
    sub.set_defaults(func=cmd_perft)

    sub = add("selfcheck", "运行内置自检")
    sub.add_argument("--games", type=int, default=5, help="随机对局局数")
    sub.add_argument("--plies", type=int, default=120, help="每局最多步数")
    sub.add_argument("--depth", type=int, default=3, help="perft 深度")
    sub.add_argument("--seed", type=int, default=20261009, help="随机种子")
    sub.set_defaults(func=cmd_selfcheck)

    sub = subparsers.add_parser("gui", help="启动图形界面", description="启动根号棋图形界面")
    sub.add_argument("args", nargs="*", help="传给图形界面的参数（如 --selftest）")
    sub.set_defaults(func=cmd_gui)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _fix_console()
    raw = list(sys.argv[1:] if argv is None else argv)
    # gui 之后的参数（如 --selftest）直接透传给图形界面，不经过 argparse
    if raw and raw[0] == "gui":
        return run_gui(raw[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except (FenError, RecordError, ValueError) as exc:
        print("错误: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
