"""根号棋棋谱（.sqg 文本）的读写与回放。

棋谱格式类似国际象棋 PGN：

    [Event "根号棋对局"]
    [Site "sqrtchess"]
    [Date "2026.10.09"]
    [White "人类"]
    [Black "贪心AI"]
    [Result "1-0"]
    [Rule "sqrtchess-1.0"]
    [Termination "all-captured"]

    1. 1b2-b3 1i10-i9
    2. 9f1-f4 4c11-c9
    1-0

* 头部行写成 [键 "值"]，键可以自由增减。
* 着法使用带大小的自校验记谱（例如 5b2xc3），也接受 b2xb3 / b2b3 等形式；
  大小与实际不符会被判为错误。
* { 花括号内为注释 } 与 ; 行尾注释在读取时忽略。
* [FEN] 头（配合 [Setup "1"]）表示非标准起始局面；没有时按初始局面回放。
* 一个文件可以保存多局，用空行分隔。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence

from .core import (BLACK, DEFAULT_NO_CAPTURE_LIMIT, WHITE, Game, GameResult,
                   IllegalMove, MoveInfo, color_name)
from .fen import START_FEN, FenError, parse_fen

__all__ = [
    "Record", "Replay", "Recorder", "RecordError",
    "RESULT_WHITE", "RESULT_BLACK", "RESULT_DRAW", "RESULT_UNKNOWN",
    "parse_record", "parse_records", "format_record", "build_record",
    "replay", "default_headers", "result_winner", "winner_result", "RULES_TAG",
]

RESULT_WHITE = "1-0"
RESULT_BLACK = "0-1"
RESULT_DRAW = "1/2-1/2"
RESULT_UNKNOWN = "*"

#: 规则版本标记，写入棋谱头部
RULES_TAG = "sqrtchess-1.0"

HEADER_ORDER = ("Event", "Site", "Date", "Round", "White", "Black", "Result",
                "Rule", "Termination", "TimeControl", "FEN", "Setup")


class RecordError(ValueError):
    """棋谱文本非法。"""


def result_winner(result: Optional[str]) -> Optional[int]:
    """把 1-0 / 0-1 / 1/2-1/2 转成 WHITE / BLACK / None（* 也视为未定）。"""
    text = (result or "").strip()
    if text == RESULT_WHITE:
        return WHITE
    if text == RESULT_BLACK:
        return BLACK
    return None


def winner_result(winner: Optional[int]) -> str:
    """把 WHITE / BLACK / None 转成棋谱结果串。"""
    if winner == WHITE:
        return RESULT_WHITE
    if winner == BLACK:
        return RESULT_BLACK
    return RESULT_DRAW


def default_headers(white: str = "?", black: str = "?", event: str = "根号棋对局",
                    site: str = "sqrtchess") -> Dict[str, str]:
    """生成一套默认头信息。"""
    return {
        "Event": event,
        "Site": site,
        "Date": date.today().strftime("%Y.%m.%d"),
        "Round": "-",
        "White": white,
        "Black": black,
        "Result": RESULT_UNKNOWN,
        "Rule": RULES_TAG,
    }


# --------------------------------------------------------------------------- 记录


@dataclass
class Record:
    """一局棋的棋谱（头部 + 着法文本 + 结果）。"""

    headers: Dict[str, str] = field(default_factory=dict)
    moves: List[str] = field(default_factory=list)
    result: Optional[str] = None
    start_fen: Optional[str] = None

    @property
    def white(self) -> str:
        return self.headers.get("White", "?")

    @property
    def black(self) -> str:
        return self.headers.get("Black", "?")

    def to_text(self) -> str:
        return format_record(self)

    def __str__(self) -> str:
        return self.to_text()


def _flush_record(headers: Dict[str, str], tokens: List[str]) -> Optional[Record]:
    if not headers and not tokens:
        return None
    result: Optional[str] = None
    moves: List[str] = []
    for token in tokens:
        if token in (RESULT_WHITE, RESULT_BLACK, RESULT_DRAW, RESULT_UNKNOWN):
            result = token
        else:
            moves.append(token)
    record = Record(headers=dict(headers), moves=moves, result=result)
    if headers.get("FEN"):
        record.start_fen = headers["FEN"]
    return record


_HEADER_RE = re.compile(r'^\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)\s+"?(.*?)"?\s*\]\s*$')
_COMMENT_RE = re.compile(r"\{[^}]*\}", re.DOTALL)


def _strip_comments(text: str) -> str:
    text = _COMMENT_RE.sub(" ", text)
    lines = []
    for line in text.split("\n"):
        if ";" in line:
            line = line.split(";", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _clean_line(text: str, headers: Dict[str, str]):
    """把棋谱正文拆成 token 列表（忽略回合号、NAG 等）。"""
    tokens = []
    for raw in text.replace(".", ". ").split():
        token = raw.strip()
        if not token:
            continue
        if token.endswith(".") or token.strip(".") == "":
            continue
        if token.startswith("$"):
            continue
        if token in (RESULT_WHITE, RESULT_BLACK, RESULT_DRAW, RESULT_UNKNOWN):
            tokens.append(token)
            continue
        tokens.append(token)
    return tokens


def parse_records(text: str) -> List[Record]:
    """解析棋谱文本，返回其中的全部对局。"""
    content = _strip_comments(str(text)).replace("\r\n", "\n").replace("\r", "\n")
    records: List[Record] = []
    headers: Dict[str, str] = {}
    tokens: List[str] = []
    for line in content.split("\n"):
        match = _HEADER_RE.match(line)
        if match:
            if tokens:
                done = _flush_record(headers, tokens)
                if done:
                    records.append(done)
                headers, tokens = {}, []
            headers[match.group(1)] = match.group(2)
            continue
        if not line.strip():
            if tokens:
                done = _flush_record(headers, tokens)
                if done:
                    records.append(done)
                headers, tokens = {}, []
            continue
        tokens.extend(_clean_line(line, headers))
    done = _flush_record(headers, tokens)
    if done:
        records.append(done)
    return records


def parse_record(text: str) -> Record:
    """解析棋谱文本中的第一局。"""
    records = parse_records(text)
    if not records:
        raise RecordError("棋谱为空或格式不正确")
    return records[0]


# --------------------------------------------------------------------------- 输出


def format_record(record: Record) -> str:
    """把 Record 渲染成棋谱文本。"""
    headers = dict(record.headers)
    result = record.result or headers.get("Result") or RESULT_UNKNOWN
    headers["Result"] = result
    start_fen = record.start_fen or headers.get("FEN")
    if start_fen and start_fen != START_FEN:
        headers["FEN"] = start_fen
        headers.setdefault("Setup", "1")
    else:
        headers.pop("FEN", None)
        if headers.get("Setup") == "1":
            headers.pop("Setup", None)
    headers.setdefault("Rule", RULES_TAG)

    order = [key for key in HEADER_ORDER if key in headers]
    order += sorted(key for key in headers if key not in HEADER_ORDER)

    lines = ['[%s "%s"]' % (key, headers[key]) for key in order]
    lines.append("")

    start_side = WHITE
    if start_fen:
        try:
            start_side = parse_fen(start_fen).side
        except FenError:
            start_side = WHITE
    offset = 1 if start_side == BLACK else 0
    buffer: List[str] = []
    body: List[str] = []
    for index, token in enumerate(record.moves):
        is_white = (index % 2 == 0) if start_side == WHITE else (index % 2 == 1)
        if index == 0:
            buffer.append("1..." if start_side == BLACK else "1.")
        elif is_white:
            if buffer:
                body.append(" ".join(buffer))
                buffer = []
            buffer.append("%d." % ((index + offset) // 2 + 1))
        buffer.append(token)
    if buffer:
        body.append(" ".join(buffer))
    lines.extend(body)
    lines.append(result)
    return "\n".join(lines) + "\n"


def build_record(headers: Optional[Dict[str, str]] = None,
                 moves: Optional[Sequence[str]] = None,
                 result: Optional[str] = None,
                 start_fen: Optional[str] = None) -> str:
    """快速拼装棋谱文本。"""
    record = Record(headers=dict(headers or {}), moves=list(moves or []),
                    result=result, start_fen=start_fen)
    return format_record(record)


# --------------------------------------------------------------------------- 回放


@dataclass
class Replay:
    """棋谱回放结果。"""

    headers: Dict[str, str]
    start_fen: str
    moves: List[MoveInfo]
    fens: List[str]
    game: Game
    declared_result: Optional[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def computed_result(self) -> Optional[GameResult]:
        return self.game.result

    @property
    def result_matches(self) -> bool:
        """棋谱声明的结果与按规则推演出的结果是否一致。"""
        declared = (self.declared_result or RESULT_UNKNOWN).strip()
        if declared == RESULT_UNKNOWN:
            return True
        if declared == RESULT_DRAW:
            return self.game.result is not None and self.game.result.is_draw
        winner = result_winner(declared)
        return self.game.result is not None and self.game.result.winner == winner

    def summary(self) -> str:
        """返回人可读的回放摘要。"""
        parts = []
        parts.append("起始局面: %s" % ("标准初始局面" if self.start_fen == START_FEN else self.start_fen))
        parts.append("着法数: %d 步" % len(self.moves))
        if self.game.result:
            parts.append("推演结果: %s" % self.game.result.text())
        else:
            parts.append("推演结果: 对局未结束（轮到%s）" % color_name(self.game.side))
        parts.append("棋谱结果: %s" % (self.declared_result or RESULT_UNKNOWN))
        if self.declared_result and not self.result_matches:
            parts.append("注意: 棋谱结果与推演结果不一致")
        if self.warnings:
            parts.append("提示: " + "；".join(self.warnings))
        if self.errors:
            parts.append("错误: " + "；".join(self.errors))
        return "\n".join(parts)


def replay(source, no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT,
           stop_on_error: bool = True) -> Replay:
    """回放棋谱（接受 Record 或文本），返回每一步的局面。"""
    record = source if isinstance(source, Record) else parse_record(source)
    start_fen = record.start_fen or START_FEN
    errors: List[str] = []
    warnings: List[str] = []
    try:
        game = Game.from_fen(start_fen, no_capture_limit=no_capture_limit)
    except FenError as exc:
        game = Game.new(no_capture_limit=no_capture_limit)
        start_fen = game.start_fen
        errors.append("起始局面无法解析（%s），已改用初始局面" % exc)
    moves: List[MoveInfo] = []
    fens = [game.fen()]
    for index, token in enumerate(record.moves):
        try:
            info = game.push(token)
        except (IllegalMove, ValueError) as exc:
            errors.append("第 %d 步 %r 无法执行: %s" % (index + 1, token, exc))
            if stop_on_error:
                break
            continue
        moves.append(info)
        fens.append(game.fen())
    if record.headers.get("Termination") in ("resign", "agreement") and not errors:
        winner = result_winner(record.result)
        game.declare_result(winner, record.headers.get("Termination", "resign"),
                            "棋谱声明的结果")
    if record.headers.get("NoCaptureLimit"):
        try:
            limit = int(record.headers["NoCaptureLimit"])
            if limit != game.no_capture_limit:
                warnings.append("棋谱记录的 NoCaptureLimit=%d 与当前设置 %s 不同"
                                % (limit, game.no_capture_limit))
        except ValueError:
            pass
    return Replay(headers=dict(record.headers), start_fen=start_fen, moves=moves,
                  fens=fens, game=game, declared_result=record.result,
                  errors=errors, warnings=warnings)


# --------------------------------------------------------------------------- 记录器


class Recorder:
    """记录一局正在进行的对局，随时可以导出棋谱文本。"""

    def __init__(self, game: Optional[Game] = None, headers: Optional[Dict[str, str]] = None,
                 no_capture_limit=DEFAULT_NO_CAPTURE_LIMIT):
        self.game = game if game is not None else Game.new(no_capture_limit=no_capture_limit)
        self.headers = dict(headers) if headers else default_headers()
        self.start_fen = self.game.start_fen
        self.moves: List[MoveInfo] = list(self.game.history)
        self.final_result: Optional[str] = None

    # -------------------------------------------------- 操作

    def push(self, move) -> MoveInfo:
        info = self.game.push(move)
        self.moves.append(info)
        return info

    def pop(self) -> MoveInfo:
        info = self.game.pop()
        if self.moves:
            self.moves.pop()
        return info

    def set_result(self, result: Optional[str]) -> None:
        self.final_result = result
        if result:
            self.headers["Result"] = result

    def finish(self, winner: Optional[int] = None, reason: Optional[str] = None) -> str:
        """结束记录：确定结果并写入头部，返回棋谱文本。"""
        if reason:
            self.headers["Termination"] = reason
        elif self.game.result and self.game.result.reason:
            self.headers["Termination"] = self.game.result.reason
        if winner is not None or self.game.result is not None:
            result = winner_result(winner if winner is not None else self.game.result.winner)
        else:
            result = RESULT_DRAW
        self.set_result(result)
        if self.game.result and self.game.result.detail:
            self.headers["ResultDetail"] = self.game.result.detail
        return self.to_text()

    # -------------------------------------------------- 输出

    @property
    def result(self) -> str:
        if self.final_result:
            return self.final_result
        if self.game.result:
            return self.game.result.score
        return RESULT_UNKNOWN

    def to_record(self) -> Record:
        headers = dict(self.headers)
        headers["Result"] = self.result
        return Record(headers=headers,
                      moves=[info.notation for info in self.moves],
                      result=self.result,
                      start_fen=self.start_fen)

    def to_text(self) -> str:
        return format_record(self.to_record())

    def move_pairs_text(self) -> str:
        """返回紧凑的着法列表文本（每行一个回合），用于日志输出。"""
        lines = []
        buffer: List[str] = []
        for index, info in enumerate(self.moves):
            if index % 2 == 0:
                if buffer:
                    lines.append(" ".join(buffer))
                    buffer = []
                buffer.append("%d." % (index // 2 + 1))
            buffer.append(info.notation)
        if buffer:
            lines.append(" ".join(buffer))
        return "\n".join(lines)
