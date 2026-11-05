"""Seeded, color-balanced matches against the project's reference AIs.

Usage: python -B ai-gpt/benchmark.py --time-ms 100 --pairs 2
Results/records are saved under ai-gpt/ only. Capped games remain unfinished.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqrtchess import Game
from sqrtchess.ai import CaptureAI, GreedyAI, SearchAI
from my_work import StrongAI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time-ms', type=float, default=100)
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--max-plies', type=int, default=400)
    parser.add_argument('--opponents', nargs='+', default=['greedy', 'capture', 'search2'])
    parser.add_argument('--output', default='benchmark_results.json')
    args = parser.parse_args()
    output = (ROOT / 'ai-gpt' / args.output).resolve()
    if output.parent != (ROOT / 'ai-gpt').resolve():
        raise ValueError('output must be a filename inside ai-gpt')
    report = {'settings': vars(args),
              'implementation_sha256': hashlib.sha256(
                  (ROOT / 'ai-gpt' / 'my_work.py').read_bytes()).hexdigest(),
              'games': []}
    for opponent in args.opponents:
        for pair in range(args.pairs):
            # The same opening is played with both assignments of AI colors.
            game = Game.new()
            rng = random.Random(20261018 + pair)
            opening = []
            for _ in range(6 if pair else 0):
                move = rng.choice(game.legal_moves())
                opening.append(game.move_dict(move)['notation'])
                game.push(move)
            for color in (1, -1):
                current = Game.from_fen(game.fen())
                ours = StrongAI(time_limit=args.time_ms / 1000)
                if opponent == 'capture':
                    theirs = CaptureAI(depth=4, seed=pair)
                elif opponent == 'greedy':
                    theirs = GreedyAI(seed=pair)
                elif opponent == 'search2':
                    theirs = SearchAI(depth=2, seed=pair, time_limit=args.time_ms / 1000)
                else:
                    raise ValueError(opponent)
                record = {'opponent': opponent, 'pair': pair, 'color': color,
                          'opening': opening, 'start_fen': current.fen(), 'moves': [],
                          'our_ms': 0.0, 'opponent_ms': 0.0, 'depths': []}
                start = time.perf_counter()
                for _ in range(args.max_plies):
                    if current.is_over:
                        break
                    side = current.side
                    engine = ours if side == color else theirs
                    before = time.perf_counter()
                    move = engine.choose(current)
                    elapsed = (time.perf_counter() - before) * 1000
                    record['our_ms' if side == color else 'opponent_ms'] += elapsed
                    if side == color:
                        record['depths'].append(ours.stats['depth'])
                    if move is None or not current.is_legal(move):
                        raise RuntimeError('AI returned an illegal move')
                    record['moves'].append(current.push(move).notation)
                result = current.result
                outcome = 'unfinished' if result is None else (
                    'draw' if result.winner is None else ('win' if result.winner == color else 'loss'))
                record.update(outcome=outcome, result=result.as_dict() if result else None,
                              end_fen=current.fen(), seconds=round(time.perf_counter() - start, 3),
                              plies=len(record['moves']))
                report['games'].append(record)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                print('%s pair=%d color=%s %s plies=%d time=%.1fs depth=%.1f' % (
                    opponent, pair, 'white' if color == 1 else 'black', outcome,
                    record['plies'], record['seconds'],
                    sum(record['depths']) / max(1, len(record['depths']))), flush=True)
    summary = {}
    for opponent in args.opponents:
        games = [g for g in report['games'] if g['opponent'] == opponent]
        summary[opponent] = {result: sum(g['outcome'] == result for g in games)
                             for result in ('win', 'draw', 'loss', 'unfinished')}
    report['summary'] = summary
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
