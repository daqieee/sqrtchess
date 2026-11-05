"""根号棋引擎测试套件。

运行方式::

    python -m unittest discover -s tests -v
    python tests/test_sqrtchess.py
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sqrtchess import (AI_LEVELS, BLACK, DEFAULT_NO_CAPTURE_LIMIT, EngineSession,
                       FenError, Game, GameOver, IllegalMove, Recorder,
                       START_FEN, WHITE, build_record, choose_move, default_headers,
                       format_fen, handle_ai_request, initial_cells, make_ai,
                       move_offsets, parse_fen, parse_record,
                       parse_records, perft, replay, square_name, run_stdio)
from sqrtchess.core import NUM_SQUARES, is_legal_displacement, parse_square

_PIECE_RE = re.compile(r"^([WB])(\d+)([a-k](?:11|10|[1-9]))$")


def position(description: str, side: int = WHITE, no_capture: int = 0,
             limit=DEFAULT_NO_CAPTURE_LIMIT) -> Game:
    """按 W1a1 B2b2 这样的描述快速摆出一个局面。"""
    cells = [0] * NUM_SQUARES
    for token in description.split():
        match = _PIECE_RE.match(token)
        if not match:
            raise ValueError("无法解析的棋子描述: %r" % (token,))
        color = WHITE if match.group(1) == "W" else BLACK
        size = int(match.group(2))
        sq = parse_square(match.group(3))
        if cells[sq] != 0:
            raise ValueError("格点重复: %s" % token)
        cells[sq] = color * size
    return Game(cells, side=side, no_capture=no_capture, no_capture_limit=limit)


def squares(game: Game, moves=None) -> set:
    return {square_name(move.to) for move in (moves if moves is not None else game.legal_moves())}


class TestInitialPosition(unittest.TestCase):
    def setUp(self):
        self.game = Game.new()

    def test_piece_counts_and_sizes(self):
        self.assertEqual(self.game.piece_stats(WHITE), (18, 56))
        self.assertEqual(self.game.piece_stats(BLACK), (18, 56))

    def test_layout(self):
        for index, size in enumerate((4, 8, 5, 2, 9, 2, 5, 8, 4)):
            sq = parse_square("abcdefghijk"[index + 1] + "1")
            self.assertEqual(self.game.cells[sq], size, "第 1 行第 %d 个棋子" % (index + 1))
            mirror = parse_square("abcdefghijk"[index + 1] + "11")
            self.assertEqual(self.game.cells[mirror], -size)
        for index in range(9):
            sq = parse_square("abcdefghijk"[index + 1] + "2")
            self.assertEqual(self.game.cells[sq], 1)
            self.assertEqual(self.game.cells[parse_square("abcdefghijk"[index + 1] + "10")], -1)
        for x in (0, 10):
            self.assertEqual(self.game.cells[x], 0)
            self.assertEqual(self.game.cells[10 * 11 + x], 0)

    def test_initial_fen(self):
        self.assertEqual(self.game.fen(), START_FEN)
        self.assertEqual(Game.from_fen(START_FEN).fen(), START_FEN)

    def test_initial_moves(self):
        moves = self.game.legal_moves()
        self.assertEqual(len(moves), 22)
        self.assertEqual(sorted(m.text() for m in moves),
                         sorted(["b2-a2", "b2-b3", "c2-c3", "d2-d3", "e2-e3", "f2-f3",
                                 "g2-g3", "h2-h3", "i2-i3", "j2-j3", "j2-k2",
                                 "b1-b3", "j1-j3", "d1-c3", "d1-e3", "h1-g3", "h1-i3",
                                 "c1-a3", "c1-e3", "i1-g3", "i1-k3", "f1-f4"]))
        self.assertEqual(len(self.game.legal_moves(BLACK)), 22)

    def test_no_moves_from_empty_square(self):
        self.assertEqual(self.game.moves_from(parse_square("a5")), [])


class TestMoveOffsets(unittest.TestCase):
    def test_known_sizes(self):
        self.assertEqual(set(move_offsets(1)), {(-1, 0), (1, 0), (0, -1), (0, 1)})
        self.assertEqual(set(move_offsets(2)), {(-1, -1), (-1, 1), (1, -1), (1, 1)})
        self.assertEqual(len(move_offsets(5)), 8)
        self.assertEqual(set(move_offsets(3)), set())
        for size in (6, 7, 11, 12, 14, 15):
            self.assertEqual(move_offsets(size), (), "大小 %d 不应有走法" % size)

    def test_complete_against_bruteforce(self):
        for size in list(range(1, 60)) + [64, 72, 81, 98, 112]:
            expected = {(dx, dy) for dx in range(-12, 13) for dy in range(-12, 13)
                        if is_legal_displacement(size, dx, dy)}
            self.assertEqual(set(move_offsets(size)), expected, "大小 %d" % size)

    def test_offsets_are_cached_and_stable(self):
        self.assertIs(move_offsets(9), move_offsets(9))


class TestMovementRules(unittest.TestCase):
    def test_size_one_moves_orthogonally(self):
        game = position("W1e5 B1k11")
        self.assertEqual(squares(game), {"e6", "e4", "d5", "f5"})

    def test_size_two_moves_diagonally(self):
        game = position("W2e5 B1k11")
        self.assertEqual(squares(game), {"d4", "f4", "d6", "f6"})

    def test_size_five_moves_like_knight(self):
        game = position("W5e5 B1k11")
        self.assertEqual(squares(game),
                         {"d3", "f3", "c4", "g4", "c6", "g6", "d7", "f7"})

    def test_size_nine_moves_three_straight(self):
        game = position("W9e5 B1k11")
        self.assertEqual(squares(game), {"b5", "h5", "e2", "e8"})

    def test_cannot_leave_board(self):
        game = position("W9a1 B1k11")
        self.assertEqual(squares(game), {"d1", "a4"})

    def test_jumps_over_pieces(self):
        game = position("W4a1 W1b1 B1k11")
        targets = squares(game, game.moves_from(parse_square("a1")))
        self.assertIn("c1", targets)
        self.assertEqual(targets, {"c1", "a3"})

    def test_blocked_by_own_piece(self):
        game = position("W4a1 W1c1 B1k11")
        targets = squares(game, game.moves_from(parse_square("a1")))
        self.assertEqual(targets, {"a3"})

    def test_cannot_capture_for_own_block(self):
        game = position("W4c1 B1a1 B1k11", side=WHITE)
        # c1 上的白 4 可以吃 a1 的黑 1（跳过 b1 空格）
        self.assertIn("a1", squares(game))

    def test_capture_adds_size(self):
        game = position("W4a1 B1c1 B1k11")
        info = game.push("a1xc1")
        self.assertEqual(info.size, 4)
        self.assertEqual(info.captured, 1)
        self.assertEqual(info.new_size, 5)
        self.assertEqual(game.piece_at("c1"), 5)
        self.assertEqual(game.piece_at("a1"), 0)
        self.assertEqual(game.no_capture, 0)

    def test_black_capture_adds_size(self):
        # 回归：黑方吃白子时曾经把 captured 算成负数，导致吃子后反而变小
        game = position("B4a1 W1c1 W1k11", side=BLACK)
        info = game.push("a1xc1")
        self.assertEqual(info.captured, 1)
        self.assertEqual(info.size, 4)
        self.assertEqual(info.new_size, 5)
        self.assertEqual(game.piece_at("c1"), -5)
        self.assertEqual(game.piece_at("a1"), 0)
        game.pop()
        self.assertEqual(game.piece_at("c1"), 1)     # 被吃的白 1 恢复
        self.assertEqual(game.piece_at("a1"), -4)
        self.assertEqual(game.fen(), position("B4a1 W1c1 W1k11", side=BLACK).fen())

    def test_move_dict_capture_signs_for_both_colors(self):
        white = position("W4a1 B1c1 B1k11")
        entry = white.move_dict(white.parse("a1xc1"))
        self.assertEqual(entry["captured"], 1)
        self.assertEqual(entry["new_size"], 5)
        black = position("B4a1 W1c1 W1k11", side=BLACK)
        entry = black.move_dict(black.parse("a1xc1"))
        self.assertEqual(entry["captured"], 1)
        self.assertEqual(entry["new_size"], 5)

    def test_capture_size_reaches_unreachable_class(self):
        # 3 本身无解，但 1+2=3 会让棋子无法移动
        game = position("W1a1 B2a2 B1k11")
        game.push("a1xa2")
        self.assertEqual(game.piece_at("a2"), 3)
        self.assertEqual(game.moves_from(parse_square("a2")), [])

    def test_cannot_move_opponent_or_wrong_side(self):
        game = position("W1a1 B1k11")
        with self.assertRaises(IllegalMove):
            game.push("k11-k10")

    def test_illegal_displacement_rejected(self):
        game = position("W1a1 B1k11")
        with self.assertRaises(IllegalMove):
            game.push("a1-a3")
        with self.assertRaises(IllegalMove):
            game.push("a1-b2")

    def test_notation_size_must_match(self):
        game = position("W5a1 B1k11")
        with self.assertRaises(IllegalMove):
            game.push("4a1-c2")
        game.push("5a1-c2")
        self.assertEqual(game.piece_at("c2"), 5)

    def test_push_accepts_various_forms(self):
        game = position("W1a1 B1k11")
        self.assertEqual(game.push("a1-a2").notation, "1a1-a2")
        game.pop()
        self.assertEqual(game.push("a1a2").notation, "1a1-a2")
        game.pop()
        self.assertEqual(game.push((parse_square("a1"), parse_square("a2"))).notation, "1a1-a2")

    def test_no_capture_counter(self):
        game = position("W1a1 B1k11")
        game.push("a1-a2")
        self.assertEqual(game.no_capture, 1)
        game.push("k11-k10")
        self.assertEqual(game.no_capture, 2)


class TestUndo(unittest.TestCase):
    def test_pop_restores_everything(self):
        rng = random.Random(7)
        game = Game.new()
        snapshots = [game.fen()]
        for _ in range(40):
            moves = game.legal_moves()
            if not moves or game.is_over:
                break
            game.push(rng.choice(moves))
            snapshots.append(game.fen())
        while len(snapshots) > 1:
            game.pop()
            snapshots.pop()
            self.assertEqual(game.fen(), snapshots[-1])
        self.assertEqual(game.fen(), START_FEN)
        self.assertIsNone(game.result)

    def test_pop_without_history(self):
        with self.assertRaises(IllegalMove):
            Game.new().pop()

    def test_push_after_game_over(self):
        game = position("W1a1 B2a2")
        game.push("a1xa2")
        self.assertIsNotNone(game.result)
        with self.assertRaises(GameOver):
            game.push("a2-a3")
        self.assertEqual(game.legal_moves(), [])


class TestEndConditions(unittest.TestCase):
    def test_last_piece_captured(self):
        game = position("W1a1 B2a2")
        info = game.push("a1xa2")
        self.assertIsNotNone(game.result)
        self.assertEqual(game.result.winner, WHITE)
        self.assertEqual(game.result.reason, "all-captured")
        self.assertEqual(game.result.score, "1-0")

    def test_stalemate_like_adjudication_draw(self):
        # 黑方只剩一枚大小 3 的棋子（无任何着法）→ 进入平局判定
        game = position("B3a1 W1k1")
        self.assertIsNotNone(game.result)
        self.assertEqual(game.result.reason, "no-moves")
        self.assertIsNone(game.result.winner)
        self.assertEqual(game.result.score, "1/2-1/2")

    def test_adjudication_win_for_both_greater(self):
        game = position("B3a1 W1a5 W5b5")
        self.assertIsNotNone(game.result)
        self.assertEqual(game.result.reason, "no-moves")
        self.assertEqual(game.result.winner, WHITE)

    def test_adjudication_not_win_when_only_one_greater(self):
        game = position("B3a1 W1a5 W1b5")
        self.assertIsNotNone(game.result)
        self.assertIsNone(game.result.winner)

    def test_no_capture_limit(self):
        game = position("W1a1 B1k11", no_capture=2, limit=3)
        self.assertIsNone(game.result)
        game.push("a1-a2")
        self.assertIsNotNone(game.result)
        self.assertEqual(game.result.reason, "no-capture-limit")
        self.assertIsNone(game.result.winner)
        self.assertEqual(game.no_capture, 3)

    def test_no_capture_limit_can_be_disabled(self):
        game = position("W1a1 B1k11", no_capture=500, limit=None)
        self.assertIsNone(game.result)
        game.push("a1-a2")
        self.assertIsNone(game.result)

    def test_capture_resets_counter(self):
        game = position("W1a1 W1c5 B2a2 B1k11", no_capture=9)
        game.push("a1xa2")
        self.assertEqual(game.no_capture, 0)
        self.assertIsNone(game.result)

    def test_immobile_piece_triggers_adjudication(self):
        # 1 + 2 = 3，而 3 不是两个平方数之和 → 合并后的棋子再也动不了
        game = position("W1a1 B2a2 B1k11")
        game.push("a1xa2")
        self.assertEqual(game.piece_at("a2"), 3)
        self.assertIsNotNone(game.result)
        self.assertEqual(game.result.reason, "no-moves")

    def test_adjudicate_statistics(self):
        # 白方 3 子/大小和 3，黑方 1 子/大小和 9 → 只有一项占优，判平局
        game = position("W1a1 W1b1 W1c1 B9k11")
        result = game.adjudicate("no-moves")
        self.assertIsNone(result.winner)
        self.assertIn("白方 3 子/大小和 3", result.detail)
        self.assertIn("黑方 1 子/大小和 9", result.detail)

    def test_adjudicate_matrix(self):
        game = position("W1a1 W1b1 B1c1")
        result = game.adjudicate("no-moves")
        self.assertEqual(result.winner, WHITE)
        game = position("W1a1 B1c1 B1d1")
        result = game.adjudicate("no-moves")
        self.assertEqual(result.winner, BLACK)
        game = position("W5a1 W5b1 B1c1 B9d1")
        result = game.adjudicate("no-moves")   # 白 2 子/10，黑 2 子/10 → 平局
        self.assertIsNone(result.winner)

    def test_declare_result(self):
        game = Game.new()
        result = game.declare_result(BLACK, "resign", "白方认输")
        self.assertEqual(game.result.score, "0-1")
        self.assertEqual(result.reason, "resign")
        self.assertTrue(game.is_over)

    def test_perft_stops_at_terminal(self):
        self.assertEqual(perft(position("B3a1 W1k1"), 2), 1)


class TestFen(unittest.TestCase):
    def test_roundtrip_random_positions(self):
        rng = random.Random(11)
        for _ in range(20):
            game = Game.new()
            for _ in range(rng.randint(0, 60)):
                moves = game.legal_moves()
                if not moves or game.is_over:
                    break
                game.push(rng.choice(moves))
            self.assertEqual(Game.from_fen(game.fen()).fen(), game.fen())

    def test_fields(self):
        data = parse_fen(START_FEN)
        self.assertEqual(data.side, WHITE)
        self.assertEqual(data.no_capture, 0)
        self.assertEqual(data.move_number, 1)
        self.assertEqual(data.cells, initial_cells())

    def test_alias_startpos(self):
        self.assertEqual(parse_fen("startpos").cells, initial_cells())

    def test_flexible_empty_and_case(self):
        text = START_FEN.replace(".", "-").replace("W", "w").replace("B", "b")
        self.assertEqual(Game.from_fen(text).fen(), START_FEN)

    def test_optional_fields(self):
        board = START_FEN.split()[0]
        self.assertEqual(parse_fen(board).side, WHITE)
        self.assertEqual(parse_fen(board + " b").side, BLACK)
        self.assertEqual(parse_fen(board + " b 7").no_capture, 7)
        self.assertEqual(parse_fen(board + " b 7 12").move_number, 12)

    def test_errors(self):
        board = START_FEN.split()[0]
        with self.assertRaises(FenError):
            parse_fen("/".join(["." * 11] * 10))
        with self.assertRaises(FenError):
            parse_fen("." * 12 + "/" + "/".join(["." * 11] * 10))
        with self.assertRaises(FenError):
            parse_fen("X1" + "." * 9 + "/" + "/".join(["." * 11] * 10))
        with self.assertRaises(FenError):
            parse_fen("W" + "." * 10 + "/" + "/".join(["." * 11] * 10))
        with self.assertRaises(FenError):
            parse_fen(board + " x")
        with self.assertRaises(FenError):
            parse_fen(board + " w -1")

    def test_format_fen_roundtrip(self):
        game = position("W5a1 B2k11", side=BLACK, no_capture=4)
        text = format_fen(game.cells, game.side, game.no_capture, game.move_number)
        again = Game.from_fen(text)
        self.assertEqual(again.fen(), text)
        self.assertEqual(again.side, BLACK)
        self.assertEqual(again.no_capture, 4)


class TestRecords(unittest.TestCase):
    def _random_game(self, plies=60, seed=3):
        rng = random.Random(seed)
        recorder = Recorder(Game.new())
        while not recorder.game.is_over and recorder.game.ply < plies:
            moves = recorder.game.legal_moves()
            if not moves:
                break
            recorder.push(rng.choice(moves))
        return recorder

    def test_record_roundtrip(self):
        recorder = self._random_game()
        text = recorder.finish()
        report = replay(text)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(len(report.moves), len(recorder.moves))
        self.assertEqual(report.game.fen(), recorder.game.fen())

    def test_record_contains_headers_and_result(self):
        recorder = self._random_game(plies=10)
        text = recorder.finish()
        record = parse_record(text)
        self.assertIn("Rule", record.headers)
        self.assertEqual(record.headers["Event"], "根号棋对局")
        self.assertIn(record.result, ("1-0", "0-1", "1/2-1/2", "*"))
        self.assertEqual(len(record.moves), len(recorder.moves))

    def test_size_prefix_is_checked(self):
        text = build_record(default_headers(), ["99b2-b3"], None)
        report = replay(text)
        self.assertFalse(report.ok)
        self.assertTrue(any("大小" in item for item in report.errors))

    def test_comments_and_numbering_ignored(self):
        text = ("[Event \"x\"]\n\n"
                "1. {开局注释} 1b2-b3 1i10-i9 ; 行尾注释\n"
                "2. 1b3-b4 1i9-i8\n*\n")
        record = parse_record(text)
        self.assertEqual(record.moves, ["1b2-b3", "1i10-i9", "1b3-b4", "1i9-i8"])
        report = replay(text)
        self.assertTrue(report.ok, report.errors)

    def test_multiple_games_in_one_file(self):
        first = build_record(default_headers(white="A"), ["1b2-b3"], "1/2-1/2")
        second = build_record(default_headers(white="B"), ["1c2-c3"], "1/2-1/2")
        records = parse_records(first + "\n" + second)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].white, "A")
        self.assertEqual(records[1].white, "B")

    def test_custom_start_fen(self):
        game = position("W1a1 B1k11", side=WHITE)
        recorder = Recorder(game, default_headers())
        recorder.push("1a1-a2")
        text = recorder.finish()
        self.assertIn("[FEN ", text)
        report = replay(text)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.start_fen, game.start_fen)
        self.assertEqual(report.game.fen(), recorder.game.fen())

    def test_resign_record(self):
        game = Game.new()
        recorder = Recorder(game, default_headers())
        recorder.push("1b2-b3")
        text = recorder.finish(winner=BLACK, reason="resign")
        self.assertIn('[Termination "resign"]', text)
        report = replay(text)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.declared_result, "0-1")
        self.assertEqual(report.game.result.winner, BLACK)

    def test_result_mismatch_detected(self):
        text = build_record(default_headers(), ["1b2-b3"], "0-1")
        report = replay(text)
        self.assertFalse(report.result_matches)

    def test_bad_move_reports_error(self):
        text = build_record(default_headers(), ["1b2-b9"], "1/2-1/2")
        report = replay(text)
        self.assertFalse(report.ok)
        self.assertTrue(report.errors)

    def test_recorder_pop(self):
        recorder = Recorder(Game.new())
        recorder.push("1b2-b3")
        recorder.push("1i10-i9")
        recorder.pop()
        self.assertEqual(len(recorder.moves), 1)
        report = replay(recorder.finish())
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(len(report.moves), 1)


class TestProtocol(unittest.TestCase):
    def test_basic_commands(self):
        session = EngineSession()
        self.assertTrue(session.handle({"cmd": "ping"})["pong"])
        self.assertEqual(session.handle({"cmd": "moves"})["count"], 22)
        response = session.handle({"cmd": "move", "move": "1b2-b3"})
        self.assertTrue(response["ok"])
        self.assertEqual(response["played"]["notation"], "1b2-b3")
        self.assertEqual(session.handle({"cmd": "line"})["line"], ["1b2-b3"])
        session.handle({"cmd": "undo"})
        self.assertEqual(session.handle({"cmd": "line"})["line"], [])
        self.assertEqual(session.handle({"cmd": "state"})["state"]["ply"], 0)

    def test_aliases_and_text_commands(self):
        session = EngineSession()
        self.assertTrue(session.handle("legal_moves")["ok"])
        self.assertTrue(session.handle({"cmd": "status"})["ok"])

    def test_unknown_command(self):
        response = EngineSession().handle({"cmd": "frobnicate"})
        self.assertFalse(response["ok"])
        self.assertIn("未知命令", response["error"])

    def test_illegal_move_error(self):
        response = EngineSession().handle({"cmd": "move", "move": "1b2-b5"})
        self.assertFalse(response["ok"])
        self.assertIn("非法着法", response["error"])

    def test_goto_and_record(self):
        session = EngineSession()
        session.handle({"cmd": "move", "move": "1b2-b3"})
        session.handle({"cmd": "move", "move": "1i10-i9"})
        session.handle({"cmd": "move", "move": "1b3-b4"})
        self.assertEqual(session.handle({"cmd": "goto", "ply": 1})["ply"], 1)
        self.assertEqual(session.game.ply, 1)
        self.assertEqual(session.handle({"cmd": "state"})["state"]["turn"], "b")
        # 回到第 1 步后另走一步黑棋，原来的着法行被截断
        response = session.handle({"cmd": "move", "move": "1c10-c9"})
        self.assertTrue(response["ok"], response.get("error"))
        record = session.handle({"cmd": "record"})
        self.assertIn("1c10-c9", record["text"])
        self.assertNotIn("1i10-i9", record["text"])

    def test_load_record(self):
        recorder = Recorder(Game.new())
        recorder.push("1b2-b3")
        recorder.push("1i10-i9")
        text = recorder.finish()
        session = EngineSession()
        response = session.handle({"cmd": "load_record", "text": text})
        self.assertTrue(response["ok"], response.get("error"))
        self.assertEqual(session.game.ply, 2)
        self.assertEqual(session.line, ["1b2-b3", "1i10-i9"])

    def test_fen_command_get_and_set(self):
        session = EngineSession()
        self.assertEqual(session.handle({"cmd": "fen"})["fen"], START_FEN)
        custom = position("W1a1 B1k11").fen()
        session.handle({"cmd": "fen", "fen": custom})
        self.assertEqual(session.handle({"cmd": "fen"})["fen"], custom)
        self.assertEqual(session.handle({"cmd": "state"})["state"]["ply"], 0)

    def test_ai_command(self):
        session = EngineSession()
        response = session.handle({"cmd": "ai", "level": "greedy", "play": True})
        self.assertTrue(response["ok"])
        self.assertIsNotNone(response["move"])
        self.assertEqual(session.game.ply, 1)

    def test_adjudicate_command(self):
        session = EngineSession()
        response = session.handle({"cmd": "adjudicate"})
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["score"], "1/2-1/2")

    def test_stdio_loop(self):
        stdin = io.StringIO('{"cmd":"moves"}\n{"cmd":"quit"}\n')
        stdout = io.StringIO()
        self.assertEqual(run_stdio(stdin, stdout), 0)
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertEqual(lines[1]["count"], 22)
        self.assertTrue(lines[-1]["quit"])

    def test_ai_request_protocol(self):
        for level, _ in AI_LEVELS:
            response = handle_ai_request({"request": "move", "fen": START_FEN, "level": level})
            self.assertTrue(response["ok"], level)
            self.assertIsNotNone(response["move"], level)
            game = Game.from_fen(START_FEN)
            self.assertTrue(game.parse(response["notation"]), level)

    def test_ai_request_ping(self):
        self.assertEqual(handle_ai_request({"request": "ping"})["kind"], "ai")

    def test_example_ai_subprocess(self):
        script = os.path.join(ROOT, "examples", "random_ai.py")
        if not os.path.exists(script):
            self.skipTest("示例 AI 不存在")
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.run(
            [sys.executable, script],
            input='{"request":"move","fen":"startpos"}\n',
            capture_output=True, text=True, encoding="utf-8", cwd=ROOT, env=env, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads(process.stdout.strip().splitlines()[0])
        self.assertTrue(payload["ok"], payload)
        game = Game.from_fen(START_FEN)
        self.assertTrue(game.is_legal(game.parse(payload["notation"])))

    def test_cli_subprocess(self):
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.run(
            [sys.executable, "-m", "sqrtchess", "proto"],
            input='{"cmd":"moves"}\n{"cmd":"quit"}\n',
            capture_output=True, text=True, encoding="utf-8", cwd=ROOT, env=env, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr)
        payloads = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
        self.assertEqual(payloads[1]["count"], 22)


class TestAi(unittest.TestCase):
    def test_all_levels_return_legal_moves(self):
        game = Game.new()
        for level, _ in AI_LEVELS:
            move = choose_move(game, level, seed=1)
            self.assertIsNotNone(move, level)
            self.assertTrue(game.is_legal(move), level)

    def test_ai_prefers_capture(self):
        game = position("W9a1 B1a4 B1k11", side=WHITE)
        move = choose_move(game, "greedy", seed=5)
        self.assertEqual(move.text(True), "a1xa4")

    def test_ai_returns_none_when_over(self):
        game = position("B3a1 W1k1")
        self.assertIsNone(choose_move(game, "random"))

    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            make_ai("nosuchai")


class TestPerft(unittest.TestCase):
    def test_initial_perft(self):
        game = Game.new()
        self.assertEqual(perft(game, 1), 22)
        self.assertEqual(perft(game, 2), 484)
        self.assertEqual(perft(game, 3), 12342)
        self.assertEqual(game.fen(), START_FEN)


class TestFuzz(unittest.TestCase):
    def test_random_games_consistency(self):
        rng = random.Random(2026)
        for game_index in range(12):
            game = Game.new()
            recorder = Recorder(game, default_headers())
            while not game.is_over and game.ply < 150:
                moves = game.legal_moves()
                if not moves:
                    break
                move = rng.choice(moves)
                info = recorder.push(move)
                self.assertEqual(info.notation, game.last_move.notation)
                self.assertEqual(Game.from_fen(game.fen()).fen(), game.fen())
            text = recorder.finish()
            report = replay(text)
            self.assertTrue(report.ok, "第 %d 局回放失败: %s" % (game_index, report.errors))
            self.assertEqual(report.game.fen(), game.fen(), "第 %d 局最终局面不一致" % game_index)

    def test_size_conservation_and_capture_arithmetic(self):
        """任意对局中棋子总大小守恒，且吃子叠加对黑白双方都正确。"""
        rng = random.Random(4242)
        captures = 0
        for _ in range(8):
            game = Game.new()
            total = sum(abs(value) for value in game.cells)
            while not game.is_over and game.ply < 200:
                moves = game.legal_moves()
                if not moves:
                    break
                move = rng.choice(moves)
                value = game.cells[move.frm]
                target = game.cells[move.to]
                expected = abs(target) if target and (target > 0) != (value > 0) else 0
                info = game.push(move)
                self.assertGreaterEqual(info.captured, 0)
                self.assertEqual(info.captured, expected)
                self.assertEqual(info.new_size, abs(value) + expected)
                self.assertEqual(game.piece_at(move.to), info.mover * info.new_size)
                self.assertEqual(sum(abs(v) for v in game.cells), total)
                captures += 1 if expected else 0
        self.assertGreater(captures, 0, "随机对局里应当出现吃子")

    def test_undo_redo_consistency(self):
        rng = random.Random(99)
        game = Game.new()
        history = []
        for _ in range(60):
            moves = game.legal_moves()
            if not moves or game.is_over:
                break
            move = rng.choice(moves)
            game.push(move)
            history.append(game.fen())
        for expected in reversed(history[:-1]):
            game.pop()
            self.assertEqual(game.fen(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
