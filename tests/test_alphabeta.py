"""Alpha-Beta 引擎的测试：搜索、SEE、攻防分析、置换表、实用对战。"""

from __future__ import annotations

import os
import re
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sqrtchess import (BLACK, CaptureAI, Game, NUM_SQUARES, WHITE, make_ai,
                       parse_square, square_name)
from sqrtchess.alphabeta import (AlphaBetaAI, MATE_SCORE, Searcher, Zobrist,
                                 analyze_exchange, evaluate, is_playable_size, see)

_PIECE_RE = re.compile(r"^([WB])(\d+)([a-k](?:11|10|[1-9]))$")


def position(description: str, side: int = WHITE, limit=100) -> Game:
    cells = [0] * NUM_SQUARES
    for token in description.split():
        match = _PIECE_RE.match(token)
        if not match:
            raise ValueError("无法解析: %r" % (token,))
        color = WHITE if match.group(1) == "W" else BLACK
        cells[parse_square(match.group(3))] = color * int(match.group(2))
    return Game(cells, side=side, no_capture_limit=limit)


class TestPlayableSize(unittest.TestCase):
    def test_known_dead_sizes(self):
        for size in (3, 6, 7, 11, 12, 14, 15, 19, 21, 22, 23, 24):
            self.assertFalse(is_playable_size(size), size)
        for size in (1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25):
            self.assertTrue(is_playable_size(size), size)


class TestZobristAndTT(unittest.TestCase):
    def test_hash_is_stable_and_position_sensitive(self):
        zobrist = Zobrist(seed=1)
        game = Game.new()
        first = zobrist.game_hash(game)
        self.assertEqual(first, Zobrist(seed=1).game_hash(Game.new()))
        game.push("1b2-b3")
        self.assertNotEqual(first, zobrist.game_hash(game))
        game.pop()
        self.assertEqual(first, zobrist.game_hash(game))

    def test_hash_sees_side_to_move(self):
        zobrist = Zobrist(seed=2)
        game = position("W1a1 B1k11")
        white_hash = zobrist.game_hash(game)
        game.side = BLACK
        self.assertNotEqual(white_hash, zobrist.game_hash(game))

    def test_transposition_table_does_not_change_best_move(self):
        game = position("W9e5 W1a1 B1e8 B1k11 B5c4")
        with_tt = Searcher(depth=3, seed=3, tt_size=1000)
        without_tt = Searcher(depth=3, seed=3, tt_size=0)
        first = with_tt.search(game)
        second = without_tt.search(game)
        self.assertIsNotNone(first.move)
        self.assertEqual(first.move, second.move)


class TestSEE(unittest.TestCase):
    def test_free_capture_is_positive(self):
        game = position("W9a1 B1a4 B1k11")
        self.assertGreater(see(game, game.parse("a1xa4")), 0)

    def test_losing_capture_is_negative(self):
        # 白 5e5 吃 f7 之后变成死子 6，白方无子可动 → 直接判负
        game = position("W5e5 B1f7 B9f4 B1k11")
        self.assertLess(see(game, game.parse("e5xf7")), 0)

    def test_winning_capture_is_huge(self):
        game = position("W1a1 B2a2")
        self.assertGreater(see(game, game.parse("a1xa2")), MATE_SCORE / 2)


class TestExchangeAnalysis(unittest.TestCase):
    def test_reports_owner_and_mobility(self):
        game = position("W9e5 B1e8 B1k11")
        report = analyze_exchange(game, "e8")
        self.assertEqual(report.owner_after, "w")
        self.assertEqual(report.final_size, 10)     # 9 + 1
        self.assertTrue(report.final_mobile)        # 10 可以移动
        self.assertGreater(report.gain_attacker, 0)

    def test_detects_self_immobilisation(self):
        game = position("W5e5 B1f7 B9f4 B1k11")
        report = analyze_exchange(game, "f7")
        self.assertTrue(report.ends_game)
        self.assertTrue(report.attacker_stuck)
        self.assertLess(report.gain_attacker, 0)
        self.assertIn("无子可动", report.outcome)

    def test_reports_recapture_options(self):
        # 白 1a1 吃 2a2 变成 3（死子），黑方 k11 的 1 太远吃不到
        game = position("W1a1 B2a2 B1k11")
        report = analyze_exchange(game, "a2")
        self.assertEqual(report.final_size, 3)
        self.assertFalse(report.final_mobile)
        self.assertTrue(report.ends_game)

    def test_recapture_with_small_piece_restores_mobility(self):
        """白 1 吃黑 2 → 变死子 3；黑方能用 1 吃回（→4 可动）但算下来不划算，于是停手。

        报告要能告诉使用者：这一格白方占了，留下的是死子 3，
        而黑方其实可以用 1e6 吃回并变成可动的 4。
        """
        game = position("W1e4 W1d5 B2e5 B1e6 B9k11", side=WHITE)
        report = analyze_exchange(game, "e5")
        self.assertEqual(report.owner_after, "w")
        self.assertEqual(report.final_size, 3)
        self.assertFalse(report.final_mobile, "3 是死子")
        self.assertTrue(any("1e6xe5" in item for item in report.recapture),
                        "应给出黑方用 1 吃回的选择：%s" % report.recapture)
        self.assertTrue(any("可动" in item for item in report.recapture),
                        "1 + 3 = 4 是可动的，应标注「可动」：%s" % report.recapture)
        self.assertIn("死子", report.notes)


class TestEvaluation(unittest.TestCase):
    def test_material_dominates(self):
        stronger = position("W9a1 B1k11")
        weaker = position("W1a1 B9k11")
        self.assertGreater(evaluate(stronger, WHITE), 0)
        self.assertLess(evaluate(weaker, WHITE), 0)

    def test_dead_piece_is_discounted(self):
        mobile = position("W5a1 B1k11")
        dead = position("W3a1 B1k11")
        self.assertGreater(evaluate(mobile, WHITE), evaluate(dead, WHITE))

    def test_symmetric_positions_are_equal(self):
        self.assertAlmostEqual(evaluate(Game.new(), WHITE), evaluate(Game.new(), BLACK), places=6)


class TestSearch(unittest.TestCase):
    def test_finds_free_capture(self):
        game = position("W9a1 B1a4 B1k11")
        result = Searcher(depth=4, seed=1).search(game)
        self.assertEqual(result.move.text(True), "a1xa4")

    def test_finds_winning_capture(self):
        game = position("W1a1 B2a2")
        result = Searcher(depth=3, seed=1).search(game)
        self.assertEqual(result.move.text(True), "a1xa2")

    def test_avoids_losing_capture(self):
        game = position("W5e5 B1f7 B9f4 B1k11")
        result = Searcher(depth=4, seed=1).search(game)
        self.assertNotEqual(result.move.to, parse_square("f7"))

    def test_iterative_deepening_reaches_depth(self):
        game = Game.new()
        result = Searcher(depth=3, seed=2).search(game)
        self.assertGreaterEqual(result.depth, 2)
        self.assertGreater(result.nodes, 0)
        self.assertIsNotNone(result.move)

    def test_time_limit_is_respected(self):
        game = Game.new()
        for _ in range(10):
            moves = game.legal_moves()
            game.push(moves[0])
        searcher = Searcher(depth=8, time_limit=0.3, seed=3)
        started = time.time()
        result = searcher.search(game)
        elapsed = time.time() - started
        self.assertIsNotNone(result.move)
        self.assertLess(elapsed, 3.0, "限时 0.3 秒不应该跑这么久：%.2fs" % elapsed)

    def test_terminal_position_returns_nothing(self):
        game = position("W3a1 B1k11")
        self.assertTrue(game.is_over)
        self.assertIsNone(Searcher(depth=3).search(game).move)


class TestAlphaBetaAI(unittest.TestCase):
    def test_registered_in_make_ai(self):
        ai = make_ai("ab", depth=2)
        self.assertIsInstance(ai, AlphaBetaAI)
        self.assertEqual(ai.depth, 2)
        self.assertIsInstance(make_ai("ab6"), AlphaBetaAI)
        self.assertEqual(make_ai("ab6").depth, 6)

    def test_choose_returns_legal_move(self):
        ai = AlphaBetaAI(depth=2, seed=4)
        game = Game.new()
        move = ai.choose(game)
        self.assertTrue(game.is_legal(move))
        self.assertIsNotNone(ai.last)

    def test_analyze_helper(self):
        ai = AlphaBetaAI(depth=2, seed=5)
        report = ai.analyze(position("W9e5 B1e8 B1k11"), "e8")
        self.assertEqual(report.owner_after, "w")

    def test_beats_random(self):
        from alphazero.arena import play_match

        result = play_match(AlphaBetaAI(depth=2, seed=6), make_ai("random", seed=7),
                            games=4, max_plies=120, seed=8)
        self.assertGreaterEqual(result.score_a, 0.75)

    def test_competitive_with_capture_baseline(self):
        from alphazero.arena import play_match

        result = play_match(AlphaBetaAI(depth=2, seed=9), CaptureAI(depth=3, seed=10),
                            games=4, max_plies=120, seed=11)
        self.assertGreaterEqual(result.score_a, 0.4,
                                "Alpha-Beta 至少要和吃子基线互有胜负：%s" % result.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
