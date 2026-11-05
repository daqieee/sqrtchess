"""根号棋引擎的机器接口（JSON 行协议）。

本模块让其它软件（包括 AI 程序）可以读取局面、列出着法、走子与导出棋谱。
完整说明见 docs/PROTOCOL.md。两种用法：

1. 控制协议（有状态）：每行一个 JSON 请求，形如 {"cmd": "moves"}。
   命令包括 ping / new / state / fen / moves / move / undo / goto / record /
   load_record / ai / perft 等，由 EngineSession 维护一局棋。
2. AI 应答协议（无状态）：每行一个 JSON 请求，形如
   {"request": "move", "fen": "..."}，返回 {"ok": true, "move": "5b2xc3"}。
   任何程序只要实现这一种请求即可作为根号棋 AI 接入。
"""

from __future__ import annotations

import json
import os
import socketserver
import sys
import time
from typing import Any, Dict, List, Optional, TextIO

from ._version import ENGINE_NAME, RULE_TAG, __version__
from .ai import make_ai
from .core import (DEFAULT_NO_CAPTURE_LIMIT, Game, GameOver, IllegalMove, MoveInfo,
                   format_move_text, parse_square)
from .fen import START_FEN, FenError
from .record import (RESULT_UNKNOWN, Record, RecordError, default_headers,
                     format_record, replay, result_winner)

__all__ = ["PROTOCOL_VERSION", "RULES_SUMMARY", "COMMANDS", "EngineSession",
           "handle_ai_request", "run_stdio", "run_ai_stdio", "serve_forever", "dumps"]

PROTOCOL_VERSION = 1

RULES_SUMMARY = """根号棋规则（sqrtchess-1.0）
1. 11x11 格点棋盘，双方镜像布局；每方 18 枚棋子：靠边一行 4 8 5 2 9 2 5 8 4，靠中心一行 9 枚 1。
2. 大小为 n 的棋子每步必须恰好移动 sqrt(n) 格，即位移 (dx,dy) 满足 dx*dx+dy*dy = n；
   无视路径上的棋子，只看落点：空点可走、己方棋子不可走、敌方棋子则吃子。
3. 吃子后自身大小 += 被吃棋子大小。
4. 一方棋子被全部吃光则判负。
5. 任意一步之后双方都还有棋子，但有一方所有棋子都没有合法落点，进入平局判定。
6. 双方合计连续 100 步（半回合）无吃子，进入平局判定（默认值，可配置）。
7. 平局判定：比较双方棋子数与大小总和，一方两项均严格大于另一方则该方胜，否则平局。"""

#: 控制协议支持的命令
COMMANDS = (
    "ping", "info", "new", "restart", "state", "fen", "board", "moves", "move",
    "undo", "goto", "line", "record", "load_record", "save_record", "adjudicate",
    "ai", "perft", "rules", "help", "quit",
)

_ALIASES = {
    "status": "state",
    "legal_moves": "moves",
    "legalmoves": "moves",
    "play": "move",
    "do": "move",
    "make_move": "move",
    "reset": "new",
    "history": "line",
    "get_record": "record",
    "load": "fen",
    "load_fen": "fen",
    "set_fen": "fen",
    "open_record": "load_record",
    "save": "save_record",
    "exit": "quit",
    "bye": "quit",
}


def dumps(obj: Any) -> str:
    """把响应编码成一行 JSON（中文不转义）。"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- 控制协议


class EngineSession:
    """维护一局根号棋，处理控制协议请求。

    会话同时保存了完整的着法行（line），因此支持 goto 前后跳转、悔棋与棋谱导出。
    """

    def __init__(self, no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT,
                 start_fen: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        self.no_capture_limit = no_capture_limit
        self.headers: Dict[str, str] = dict(headers) if headers else default_headers()
        self.declared: Optional[str] = None
        self.start_fen: str = START_FEN
        self.line: List[str] = []
        self.game: Game = Game.new(no_capture_limit=no_capture_limit)
        self.reset(start_fen)

    # -------------------------------------------------- 内部操作

    def reset(self, fen: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        """开始新对局（可指定起始局面）。"""
        game = Game.from_fen(fen if fen else START_FEN,
                             no_capture_limit=self.no_capture_limit)
        self.game = game
        self.start_fen = game.fen()
        self.line = []
        if headers:
            self.headers = dict(headers)
        self.declared = None

    def restart(self, ply: int = 0) -> None:
        """回到本局起始局面（或指定的第 ply 步）。"""
        self.goto(ply)

    def goto(self, ply: int) -> int:
        """跳转到着法行中的第 ply 步（0 = 起始局面）。"""
        target = max(0, min(int(ply), len(self.line)))
        game = Game.from_fen(self.start_fen, no_capture_limit=self.no_capture_limit)
        for token in self.line[:target]:
            game.push(token)
        self.game = game
        return target

    def play(self, token) -> MoveInfo:
        """走一步棋（自动截断 goto 之后的着法行）。"""
        if len(self.line) > self.game.ply:
            del self.line[self.game.ply:]
        info = self.game.push(token)
        self.line.append(info.notation)
        if not self.game.is_over:
            self.declared = None
        return info

    def _state(self, include_moves: bool = True) -> Dict[str, Any]:
        data = self.game.status(include_moves=include_moves)
        data["line"] = list(self.line)
        data["start_fen"] = self.start_fen
        data["declared_result"] = self.declared
        return data

    def _ok(self, request: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"ok": True}
        payload.update(extra)
        payload["state"] = self._state(include_moves=bool(request.get("moves", True)))
        return payload

    def _http_like_path(self, request: Dict[str, Any]) -> Optional[str]:
        path = request.get("path")
        if path is None:
            return None
        return os.path.abspath(os.path.expanduser(str(path)))

    def build_record(self) -> Record:
        """按当前会话内容生成棋谱。"""
        headers = dict(self.headers)
        if self.game.result is not None:
            headers["Termination"] = self.game.result.reason
            if self.game.result.detail:
                headers["ResultDetail"] = self.game.result.detail
        result = self.declared
        if result is None:
            result = self.game.result.score if self.game.result else RESULT_UNKNOWN
        headers["Result"] = result
        return Record(headers=headers, moves=list(self.line), result=result,
                      start_fen=self.start_fen)

    # -------------------------------------------------- 命令实现

    def cmd_ping(self, request):
        return {"ok": True, "name": ENGINE_NAME, "version": __version__,
                "rule": RULE_TAG, "protocol": PROTOCOL_VERSION, "pong": True}

    def cmd_info(self, request):
        return {"ok": True, "name": ENGINE_NAME, "version": __version__, "rule": RULE_TAG,
                "protocol": PROTOCOL_VERSION, "commands": list(COMMANDS),
                "rules": RULES_SUMMARY}

    def cmd_new(self, request):
        self.reset(request.get("fen") or request.get("start_fen"),
                   request.get("headers"))
        return self._ok(request)

    def cmd_restart(self, request):
        self.goto(int(request.get("ply", 0)))
        return self._ok(request)

    def cmd_state(self, request):
        return self._ok(request)

    def cmd_fen(self, request):
        text = request.get("fen") or request.get("text")
        if text:
            self.reset(str(text))
            return self._ok(request, loaded=True)
        return {"ok": True, "fen": self.game.fen(), "start_fen": self.start_fen,
                "line": list(self.line)}

    def cmd_board(self, request):
        return {"ok": True, "text": self.game.board_text(flip=bool(request.get("flip"))),
                "fen": self.game.fen()}

    def cmd_moves(self, request):
        moves = self.game.legal_moves()
        data = [self.game.move_dict(move) for move in moves]
        return {"ok": True, "count": len(moves), "turn": self.game.status(False)["turn"],
                "moves": data, "text": [item["notation"] for item in data]}

    def cmd_move(self, request):
        token = request.get("move") or request.get("notation")
        if token is None:
            if request.get("from") is None or request.get("to") is None:
                return {"ok": False, "error": "move 命令需要 move 或 from/to 参数"}
            token = format_move_text(parse_square(request["from"]), parse_square(request["to"]))
        info = self.play(str(token))
        return self._ok(request, played=info.as_dict())

    def cmd_undo(self, request):
        plies = int(request.get("plies") or request.get("count") or 1)
        if plies <= 0:
            return {"ok": False, "error": "plies 必须为正整数"}
        undone = []
        for _ in range(plies):
            if self.game.ply <= 0:
                break
            undone.append(self.game.pop().as_dict())
        del self.line[self.game.ply:]
        return self._ok(request, undone=undone)

    def cmd_goto(self, request):
        ply = int(request.get("ply", request.get("n", 0)))
        applied = self.goto(ply)
        return self._ok(request, ply=applied)

    def cmd_line(self, request):
        return {"ok": True, "line": list(self.line), "ply": self.game.ply,
                "start_fen": self.start_fen}

    def cmd_record(self, request):
        record = self.build_record()
        return {"ok": True, "text": format_record(record), "result": record.result,
                "headers": record.headers, "moves": list(self.line)}

    def cmd_load_record(self, request):
        text = request.get("text")
        path = self._http_like_path(request)
        if not text and path:
            with open(path, "r", encoding="utf-8-sig") as handle:
                text = handle.read()
        if not text:
            return {"ok": False, "error": "load_record 需要 text 或 path 参数"}
        report = replay(text, no_capture_limit=self.no_capture_limit)
        if not report.ok:
            return {"ok": False, "error": "棋谱回放失败", "errors": report.errors}
        self.headers = dict(report.headers) or default_headers()
        self.start_fen = report.start_fen
        self.line = [info.notation for info in report.moves]
        self.declared = None if report.declared_result == RESULT_UNKNOWN else report.declared_result
        self.goto(len(self.line))
        termination = report.headers.get("Termination")
        if self.declared and termination in ("resign", "agreement"):
            self.game.declare_result(result_winner(self.declared), termination, "棋谱声明的结果")
        return self._ok(request, summary=report.summary(), warnings=report.warnings)

    def cmd_save_record(self, request):
        path = self._http_like_path(request)
        text = request.get("text") or format_record(self.build_record())
        if not path:
            return {"ok": False, "error": "save_record 需要 path 参数"}
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return {"ok": True, "path": path, "bytes": len(text.encode("utf-8"))}

    def cmd_adjudicate(self, request):
        result = self.game.adjudicate(str(request.get("reason", "no-moves")))
        white = self.game.status(False)["white"]
        black = self.game.status(False)["black"]
        return {"ok": True, "result": result.as_dict(), "white": white, "black": black}

    def cmd_ai(self, request):
        level = request.get("level") or request.get("ai") or "greedy"
        engine = make_ai(level, depth=request.get("depth"), seed=request.get("seed"))
        started = time.time()
        move = engine.choose(self.game)
        elapsed = (time.time() - started) * 1000.0
        payload: Dict[str, Any] = {"ok": True, "level": str(level),
                                   "think_ms": round(elapsed, 1), "move": None}
        if move is not None:
            detail = self.game.move_dict(move)
            payload["move"] = detail["move"]
            payload["notation"] = detail["notation"]
            payload["detail"] = detail
            if request.get("play"):
                payload["played"] = self.play(detail["notation"]).as_dict()
        payload["state"] = self._state()
        return payload

    def cmd_perft(self, request):
        from .core import perft

        depth = int(request.get("depth", 3))
        started = time.time()
        nodes = perft(self.game, depth)
        return {"ok": True, "depth": depth, "nodes": nodes,
                "ms": round((time.time() - started) * 1000.0, 1)}

    def cmd_rules(self, request):
        return {"ok": True, "text": RULES_SUMMARY, "rule": RULE_TAG}

    def cmd_help(self, request):
        return {"ok": True, "commands": list(COMMANDS), "text":
                "控制协议：每行一个 JSON，例如 {\"cmd\": \"moves\"}。"}

    def cmd_quit(self, request):
        return {"ok": True, "quit": True}

    # -------------------------------------------------- 分发

    def handle(self, request: Any) -> Dict[str, Any]:
        """处理一条请求，永远返回一个可 JSON 序列化的响应字典。"""
        try:
            return self._dispatch(request)
        except GameOver as exc:
            return {"ok": False, "error": "对局已结束：%s" % exc}
        except (IllegalMove, FenError, RecordError, ValueError, KeyError, TypeError) as exc:
            return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        except OSError as exc:
            return {"ok": False, "error": "文件操作失败: %s" % exc}
        except Exception as exc:  # pragma: no cover - 兜底
            return {"ok": False, "error": "内部错误: %s: %s" % (type(exc).__name__, exc)}

    def _dispatch(self, request: Any) -> Dict[str, Any]:
        if isinstance(request, str):
            request = {"cmd": request}
        if not isinstance(request, dict):
            return {"ok": False, "error": "请求必须是 JSON 对象"}
        cmd = str(request.get("cmd") or request.get("command") or request.get("op") or "")
        cmd = cmd.strip().lower().replace("-", "_")
        if not cmd:
            return {"ok": False, "error": "缺少 cmd 字段", "commands": list(COMMANDS)}
        cmd = _ALIASES.get(cmd, cmd)
        handler = getattr(self, "cmd_" + cmd, None)
        if handler is None or not callable(handler):
            return {"ok": False, "error": "未知命令: %s" % cmd, "commands": list(COMMANDS)}
        return handler(request)


# --------------------------------------------------------------------------- AI 应答协议


def handle_ai_request(request: Any, default_level: str = "greedy",
                      no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT) -> Dict[str, Any]:
    """处理一条无状态 AI 请求：给出 FEN，返回一步着法。"""
    if isinstance(request, str):
        text = request.strip()
        request = {"request": "move", "fen": text} if text else {}
    if not isinstance(request, dict):
        return {"ok": False, "error": "请求必须是 JSON 对象"}
    kind = str(request.get("request") or request.get("cmd") or "move").strip().lower()
    if kind in ("ping", "info", "hello", "id"):
        return {"ok": True, "name": ENGINE_NAME, "version": __version__, "rule": RULE_TAG,
                "protocol": PROTOCOL_VERSION, "kind": "ai", "level": default_level}
    if kind not in ("move", "bestmove", "go", "think"):
        return {"ok": False, "error": "未知请求: %s（支持 ping / move）" % kind}
    fen = request.get("fen") or request.get("position") or START_FEN
    game = Game.from_fen(str(fen), no_capture_limit=no_capture_limit)
    level = request.get("level") or request.get("ai") or default_level
    engine = make_ai(level, depth=request.get("depth"), seed=request.get("seed"))
    started = time.time()
    move = engine.choose(game)
    elapsed = (time.time() - started) * 1000.0
    if move is None:
        return {"ok": True, "move": None, "reason": "当前局面没有合法着法",
                "game_over": game.is_over,
                "result": game.result.as_dict() if game.result else None,
                "think_ms": round(elapsed, 1)}
    detail = game.move_dict(move)
    return {"ok": True, "move": detail["move"], "notation": detail["notation"],
            "detail": detail, "level": str(level), "think_ms": round(elapsed, 1),
            "game_over": game.is_over}


# --------------------------------------------------------------------------- 行循环


def _force_utf8(stream: Any) -> None:
    """重定向/管道时强制 UTF-8（交互式控制台保持系统默认，避免中文乱码）。"""
    try:
        if not stream.isatty():
            stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _read_request(line: str) -> Any:
    text = line.strip()
    if not text:
        return None
    if text[0] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            return {"__error__": "JSON 解析失败: %s" % exc}
    return text  # 纯文本：控制协议里当作命令名，AI 协议里当作 FEN


def run_stdio(stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None,
              session: Optional[EngineSession] = None,
              no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT) -> int:
    """控制协议的 stdin/stdout 主循环。"""
    inp = stdin if stdin is not None else sys.stdin
    out = stdout if stdout is not None else sys.stdout
    _force_utf8(inp)
    _force_utf8(out)
    engine = session if session is not None else EngineSession(no_capture_limit=no_capture_limit)
    out.write(dumps({"ok": True, "hello": ENGINE_NAME, "version": __version__,
                     "protocol": PROTOCOL_VERSION, "rule": RULE_TAG,
                     "commands": list(COMMANDS), "fen": engine.game.fen()}) + "\n")
    out.flush()
    for raw in inp:
        request = _read_request(raw)
        if request is None:
            continue
        if isinstance(request, str):
            request = {"cmd": request}
        if isinstance(request, dict) and "__error__" in request:
            response: Dict[str, Any] = {"ok": False, "error": request["__error__"]}
        else:
            response = engine.handle(request)
        out.write(dumps(response) + "\n")
        out.flush()
        if response.get("quit"):
            break
    return 0


def run_ai_stdio(stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None,
                 default_level: str = "greedy",
                 no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT) -> int:
    """AI 应答协议的 stdin/stdout 主循环。"""
    inp = stdin if stdin is not None else sys.stdin
    out = stdout if stdout is not None else sys.stdout
    _force_utf8(inp)
    _force_utf8(out)
    for raw in inp:
        request = _read_request(raw)
        if request is None:
            continue
        if isinstance(request, dict) and "__error__" in request:
            response = {"ok": False, "error": request["__error__"]}
        else:
            try:
                response = handle_ai_request(request, default_level=default_level,
                                             no_capture_limit=no_capture_limit)
            except Exception as exc:  # pragma: no cover - 兜底
                response = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        out.write(dumps(response) + "\n")
        out.flush()
    return 0


# --------------------------------------------------------------------------- TCP 服务


def serve_forever(host: str = "127.0.0.1", port: int = 8765,
                  no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT,
                  on_ready=None) -> None:
    """启动 TCP 服务：每个连接一个会话，同时支持控制协议与 AI 应答协议。"""

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            session = EngineSession(no_capture_limit=no_capture_limit)
            self.wfile.write((dumps({"ok": True, "hello": ENGINE_NAME,
                                     "version": __version__, "protocol": PROTOCOL_VERSION,
                                     "fen": session.game.fen()}) + "\n").encode("utf-8"))
            for raw in self.rfile:
                try:
                    line = raw.decode("utf-8", "replace")
                except Exception:  # pragma: no cover
                    continue
                request = _read_request(line)
                if request is None:
                    continue
                if isinstance(request, dict) and "__error__" in request:
                    response: Dict[str, Any] = {"ok": False, "error": request["__error__"]}
                elif isinstance(request, dict) and ("request" in request and "cmd" not in request):
                    response = handle_ai_request(request, no_capture_limit=no_capture_limit)
                else:
                    response = session.handle(request)
                self.wfile.write((dumps(response) + "\n").encode("utf-8"))
                if response.get("quit"):
                    break

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server((host, port), Handler) as server:
        if on_ready is not None:
            on_ready(server)
        server.serve_forever()
