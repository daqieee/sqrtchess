"""Rule-oracle differential tests and search/protocol regression tests."""
import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqrtchess import Game, Move, perft
from sqrtchess.core import parse_square
from my_work import (MATE, INF, Position, StrongAI, answer, evaluate,
                             jumps, static_evaluate)


def position(text, side=1, clock=0, limit=100):
    cells = [0] * 121
    for token in text.split():
        split = next(i for i in range(1, len(token)) if token[i].isalpha())
        cells[parse_square(token[split:])] = int(token[1:split]) * (1 if token[0] == 'W' else -1)
    return Game(cells, side, clock, no_capture_limit=limit)


def fast_perft(pos, depth):
    data = pos.analyse()
    if depth <= 0 or pos.terminal(data) is not None:
        return 1
    total = 0
    for move in data[0][0 if pos.side == 1 else 1]:
        undo = pos.push(move)
        total += fast_perft(pos, depth - 1)
        pos.pop(move, undo)
    return total


def oracle_minimax(game, depth, ply=0):
    if game.result:
        winner = game.result.winner or 0
        return winner * game.side * (MATE - ply)
    if depth == 0:
        pos = Position(game)
        return static_evaluate(pos, pos.analyse())
    best = -INF
    for move in game.legal_moves():
        game.push(move)
        try:
            best = max(best, -oracle_minimax(game, depth - 1, ply + 1))
        finally:
            game.pop()
    return best


class RuleTests(unittest.TestCase):
    def compare(self, game):
        pos = Position(game)
        data = pos.analyse()
        terminal = pos.terminal(data)
        self.assertEqual(terminal is not None, game.is_over)
        if game.is_over:
            self.assertEqual(terminal, (game.result.winner or 0) * game.side * MATE)
        for color, index in ((1, 0), (-1, 1)):
            self.assertEqual((pos.count[index], pos.mass[index]), game.piece_stats(color))
            # moves_from ignores adjudication; compare geometric legality too.
            expected = {(m.frm, m.to) for s, v in enumerate(game.cells)
                        if v * color > 0 for m in game.moves_from(s)}
            self.assertEqual(set(data[0][index]), expected)
        if game.is_over:
            return
        before = pos.key(), list(pos.count), list(pos.mass)
        for move in game.legal_moves():
            pair = move.frm, move.to
            undo = pos.push(pair)
            game.push(move)
            self.assertEqual(pos.cells, game.cells)
            self.assertEqual(pos.side, game.side)
            self.assertEqual(pos.clock, game.no_capture)
            value = pos.terminal(pos.analyse())
            expected = None if game.result is None else (game.result.winner or 0) * game.side * MATE
            self.assertEqual(value, expected)
            for color, index in ((1, 0), (-1, 1)):
                self.assertEqual((pos.count[index], pos.mass[index]), game.piece_stats(color))
            pos.pop(pair, undo)
            game.pop()
            self.assertEqual((pos.key(), pos.count, pos.mass), before)

    def test_random_legal_trajectories(self):
        rng = random.Random(87291)
        for limit in (None, 1, 7, 100):
            for _ in range(5):
                game = Game.new(no_capture_limit=limit)
                for ply in range(100):
                    if ply % 5 == 0 or game.is_over:
                        self.compare(game)
                    if game.is_over:
                        break
                    moves = game.legal_moves()
                    captures = [m for m in moves if game.cells[m.to]]
                    game.push(rng.choice(captures or moves))

    def test_arbitrary_positions(self):
        rng = random.Random(901)
        for _ in range(120):
            cells = [0] * 121
            for sq in rng.sample(range(121), rng.randint(0, 20)):
                cells[sq] = rng.choice((-1, 1)) * rng.choice((1, 2, 3, 5, 8, 9, 13, 25, 50, 100, 200, 201, 10000))
            self.compare(Game(cells, rng.choice((-1, 1)), rng.randrange(101)))

    def test_perft(self):
        for depth, expected in ((1, 22), (2, 484), (3, 12342), (4, 314721)):
            self.assertEqual(fast_perft(Position(Game.new()), depth), expected)

    def test_special_endings(self):
        for text, clock, limit in [
            ('W2a1 B1b2', 0, 100),  # eating the last piece wins, even if frozen
            ('W2a1 W3k1 B1b2 B3k11', 0, 100),  # freeze the moving side
            ('W1a1 W3c1 B1a2 B3k11', 0, 100),  # freeze nonmoving side
            ('W1a1 B2k11', 99, 100),
            ('W1a1 W1b1 B1k11', 0, 1),
            ('W1a1 B1k11', 0, 0),
            ('W10000a1 B1k11', 0, None),
            ('', 0, 100),
        ]:
            self.compare(position(text, clock=clock, limit=limit))


class SearchTests(unittest.TestCase):
    def test_does_not_mutate_game(self):
        game = Game.new()
        game.push('f1-f4')
        before = game.fen(), game.history, game.result, list(game.cells)
        ai = StrongAI(time_limit=0.05)
        self.assertTrue(game.is_legal(ai.choose(game)))
        self.assertEqual((game.fen(), game.history, game.result, game.cells), before)

    def test_tiny_budgets_and_node_limits(self):
        for seconds, nodes in ((0, 10), (0.001, 100000), (1, 1), (1, 33)):
            ai = StrongAI(time_limit=seconds, node_limit=nodes)
            self.assertTrue(Game.new().is_legal(ai.choose(Game.new())))
            self.assertLessEqual(ai.stats['nodes'], nodes)
            self.assertLess(ai.stats['think_ms'], 1100)
        complete = StrongAI(time_limit=2, max_depth=1)
        interrupted = StrongAI(time_limit=2, node_limit=50)
        expected = complete.choose(Game.new())
        self.assertEqual(interrupted.choose(Game.new()), expected)
        self.assertEqual(interrupted.stats['depth'], 1)
        self.assertEqual(interrupted.stats['score'], complete.stats['score'])
        self.assertEqual(interrupted.stats['pv'], complete.stats['pv'])

    def test_forced_win_and_freeze_win(self):
        for text in ('W2a1 B1b2', 'W1a1 W3c1 W3e1 B1a2 B3k11'):
            game = position(text)
            ai = StrongAI(time_limit=0.2)
            game.push(ai.choose(game))
            self.assertIsNotNone(game.result)
            self.assertEqual(game.result.winner, 1)
            self.assertEqual(ai.stats['score'], MATE - 1)

    def test_escape_large_capture(self):
        game = position('W5f6 W1a1 B1f5 B1k11')
        ai = StrongAI(time_limit=0.5, max_depth=3)
        move = ai.choose(game)
        self.assertEqual(move.frm, parse_square('f6'))

    def test_quiescence_sees_poisoned_capture(self):
        # c3xc4 looks like a gain, but c5xc4 removes the last mobile white
        # piece; the remaining white piece is frozen, so black wins.
        game = position('W1c3 W3a1 B1c4 B1c5 B3k11')
        ai = StrongAI(time_limit=1, max_depth=1, qdepth=6)
        move = ai.choose(game)
        self.assertNotEqual(move.to, parse_square('c4'))
        self.assertEqual(ai.stats['depth'], 1)

    def test_public_evaluation(self):
        game = position('W3a1 B1k11')
        self.assertEqual(evaluate(game), 0)  # count tied, mass alone cannot win
        game.declare_result(-1, 'resign')
        self.assertEqual(evaluate(game, 1), -MATE)
        self.assertIsNone(StrongAI().choose(game))

    def test_pvs_matches_full_width_oracle(self):
        rng = random.Random(516)
        for _ in range(14):
            cells = [0] * 121
            for i, square in enumerate(rng.sample(range(121), 6)):
                cells[square] = rng.choice((1, 2, 4, 5, 9)) * (1 if i < 3 else -1)
            game = Game(cells, side=rng.choice((-1, 1)), no_capture_limit=None)
            if game.is_over:
                continue
            expected = oracle_minimax(game, 3)
            for tt_size in (0, 10000):
                ai = StrongAI(time_limit=5, max_depth=3, qdepth=0,
                              tt_size=tt_size, use_lmr=False)
                ai.choose(game)
                self.assertEqual(ai.stats['depth'], 3)
                self.assertEqual(ai.stats['score'], expected)

    def test_clock_boundary_quiescence(self):
        game = position('W1a1 B2k11', clock=99)
        ai = StrongAI(time_limit=1, max_depth=1)
        ai.choose(game)
        self.assertEqual(ai.stats['score'], 0)
        # Every noncapturing action loses; stand-pat must not mask this.
        game = position('W1a1 B1j11 B1k11', clock=99)
        ai.choose(game)
        self.assertEqual(ai.stats['score'], -MATE + 1)


class ProtocolTests(unittest.TestCase):
    def test_protocol(self):
        requests = [dict(request='ping'), dict(fen='startpos', time_ms=20),
                    dict(fen=position('W3a1 B1k11').fen()), [],
                    dict(time_ms=-1), dict(request='bad'),
                    dict(fen='startpos', time_ms=0, no_capture_limit=0)]
        process = subprocess.run(
            [sys.executable, '-B', str(ROOT / 'ai-gpt' / 'my_work.py')],
            input='\n'.join(json.dumps(r) for r in requests) + '\n',
            text=True, encoding='utf-8', capture_output=True, cwd=str(ROOT),
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, '')
        responses = [json.loads(s) for s in process.stdout.splitlines()]
        self.assertEqual(len(responses), len(requests))
        self.assertEqual(responses[0]['name'], 'RootKnight')
        self.assertIsNotNone(Game.new().parse(responses[1]['move']))
        self.assertIsNone(responses[2]['move'])
        for index in (3, 4, 5):
            self.assertFalse(responses[index]['ok'])
        self.assertIsNone(responses[6]['move'])

    def test_configurable_clock_and_validation(self):
        self.assertFalse(answer(dict(time_ms=0, no_capture_limit=None))['game_over'])
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                answer(dict(time_ms=value))


if __name__ == '__main__':
    unittest.main()
