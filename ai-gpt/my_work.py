"""RootKnight: a standard-library-only AI for sqrtchess.

Run ``python -B ai-gpt/my_work.py`` for the JSON-lines AI protocol, or use
``StrongAI(time_limit=2.0).choose(game)``. The supplied Game is never mutated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

# Do not create bytecode files inside the supplied rules engine.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqrtchess import Game, Move  # noqa: E402
from sqrtchess.ai import BaseAI  # noqa: E402

NAME = "RootKnight"
MATE = 1_000_000
INF = 2_000_000
EXACT, LOWER, UPPER = 0, 1, 2
CENTER = tuple(10 - abs(s % 11 - 5) - abs(s // 11 - 5) for s in range(121))


def _make_jumps():
    # Enumerating board displacements also handles arbitrary huge FEN sizes:
    # anything > 200 cannot have an on-board destination.
    table = [[[] for _ in range(121)] for _ in range(201)]
    for a in range(121):
        x, y = a % 11, a // 11
        for b in range(121):
            n = (b % 11 - x) ** 2 + (b // 11 - y) ** 2
            if n:
                table[n][a].append(b)
    return tuple(tuple(tuple(row) for row in size) for size in table)


JUMPS = _make_jumps()
EMPTY_JUMPS = ((),) * 121


def jumps(size):
    return JUMPS[size] if 0 < size <= 200 else EMPTY_JUMPS


@lru_cache(maxsize=64)
def distances(size):
    """Empty-board graph distances, respecting parity/residue components."""
    graph = jumps(size)
    rows = []
    for start in range(121):
        row = [99] * 121
        row[start] = 0
        queue = [start]
        for square in queue:
            for target in graph[square]:
                if row[target] == 99:
                    row[target] = row[square] + 1
                    queue.append(target)
        rows.append(tuple(row))
    return tuple(rows)


class Position:
    """Private search state; public engine remains the rules oracle at the API."""

    def __init__(self, game):
        self.cells = list(game.cells)
        self.side = game.side
        self.clock = game.no_capture
        self.limit = game.no_capture_limit
        self.count = [sum(v > 0 for v in self.cells), sum(v < 0 for v in self.cells)]
        self.mass = [sum(v for v in self.cells if v > 0),
                     -sum(v for v in self.cells if v < 0)]

    def key(self):
        # Exact tuple identity rather than a probabilistic hash signature.
        return (tuple(self.cells), self.side, self.clock)

    def push(self, move):
        a, b = move
        value, victim, old_clock = self.cells[a], self.cells[b], self.clock
        self.cells[a] = 0
        self.cells[b] = value + self.side * abs(victim)
        if victim:
            us = 0 if self.side == 1 else 1
            self.count[1 - us] -= 1
            self.mass[us] += abs(victim)
            self.mass[1 - us] -= abs(victim)
            self.clock = 0
        else:
            self.clock += 1
        self.side = -self.side
        return value, victim, old_clock

    def pop(self, move, undo):
        value, victim, self.clock = undo
        self.side = -self.side
        self.cells[move[0]], self.cells[move[1]] = value, victim
        if victim:
            us = 0 if self.side == 1 else 1
            self.count[1 - us] += 1
            self.mass[us] -= abs(victim)
            self.mass[1 - us] += abs(victim)

    def analyse(self):
        moves = [[], []]
        attacks = [[0] * 121, [0] * 121]
        pieces = [[], []]
        board = self.cells
        for square, value in enumerate(board):
            if not value:
                continue
            color = 0 if value > 0 else 1
            size = abs(value)
            targets = jumps(size)[square]
            legal = []
            for target in targets:
                attacks[color][target] += 1
                if board[target] * value <= 0:
                    legal.append(target)
                    moves[color].append((square, target))
            pieces[color].append((square, size, targets, legal))
        return moves, attacks, pieces

    def terminal(self, data, ply=0):
        if not self.count[0] or not self.count[1]:
            winner = 1 if self.count[0] else (-1 if self.count[1] else 0)
        elif (not data[0][0] or not data[0][1] or
              (self.limit is not None and self.clock >= self.limit)):
            dc, dm = self.count[0] - self.count[1], self.mass[0] - self.mass[1]
            winner = 1 if dc > 0 and dm > 0 else (-1 if dc < 0 and dm < 0 else 0)
        else:
            return None
        return winner * self.side * (MATE - ply) if winner else 0


def static_evaluate(pos, data):
    """Integer score from side-to-move's perspective; 100 = one unit of size."""
    moves, attacks, pieces = data
    dc, dm = pos.count[0] - pos.count[1], pos.mass[0] - pos.mass[1]
    score = 100 * dm + 240 * dc
    # The two STRICT inequalities are both necessary to win adjudication.
    if dc * dm > 0:
        score += (1 if dc > 0 else -1) * min(300, abs(dc) * 100, abs(dm) * 40)
    endgame = sum(pos.count) <= 14
    for color in (0, 1):
        ours, theirs = attacks[color], attacks[1 - color]
        value = 0
        biggest_threat = 0
        for square, size, targets, legal in pieces[color]:
            if not targets:
                # Frozen pieces still count toward adjudication; don't discard
                # their material. Penalize the lost ability to act instead.
                value -= 100 + min(size, 20) * 5
            elif not legal:
                value -= 75  # friendly blockade, potentially reversible
            else:
                safe = sum(theirs[t] == 0 for t in legal)
                value += 18 + 7 * len(legal) + 9 * safe + 4 * CENTER[square]
                if safe == 0:
                    value -= 35
                if endgame:
                    row = distances(size)[square]
                    distance = min((row[p[0]] for p in pieces[1 - color]), default=99)
                    # Inaccessible residue classes are genuinely inaccessible
                    # until a capture changes the moving piece's size.
                    value += max(0, 12 - distance) * 6
            if theirs[square]:
                loss = 240 + 200 * size  # capture transfers size to the enemy
                risk = loss // (12 if ours[square] else 7)
                value -= risk
                biggest_threat = max(biggest_threat, risk)
        # A threat against the side that cannot move first is more immediate.
        if (color == 0) != (pos.side == 1):
            value -= biggest_threat // 2
        score += value if color == 0 else -value
    # As the clock expires, value a likely adjudication more explicitly.
    if pos.limit is not None and pos.limit > 0:
        progress = max(0, 2 * pos.clock - pos.limit)
        if progress:
            if dc * dm > 0:
                score += (1 if dc > 0 else -1) * 1200 * progress // pos.limit
            else:
                score = score * (2 * pos.limit - progress) // (2 * pos.limit)
    return max(-MATE // 4, min(MATE // 4, score * pos.side))


def evaluate(game, color=None):
    """Public evaluation helper (including exact terminal outcomes)."""
    color = game.side if color is None else color
    if color not in (1, -1):
        raise ValueError("color must be 1 or -1")
    if game.result is not None:
        return 0 if game.result.winner is None else MATE * game.result.winner * color
    pos = Position(game)
    data = pos.analyse()
    terminal = pos.terminal(data)
    score = terminal if terminal is not None else static_evaluate(pos, data)
    return score * pos.side * color


class SearchStopped(Exception):
    pass


class StrongAI(BaseAI):
    """Iterative deepening PVS, exact-key TT, quiescence and move ordering."""

    name = NAME

    def __init__(self, time_limit=2.0, max_depth=64, node_limit=500_000,
                 qdepth=6, tt_size=60_000, use_lmr=True):
        self.time_limit = float(time_limit)
        if not math.isfinite(self.time_limit) or self.time_limit < 0:
            raise ValueError("time_limit must be finite and nonnegative")
        self.max_depth = max(1, min(96, int(max_depth)))
        self.node_limit = max(1, int(node_limit))
        self.qdepth = max(0, min(16, int(qdepth)))
        self.tt_size = max(0, int(tt_size))
        self.use_lmr = bool(use_lmr)
        self.stats = {}

    def _visit(self, quiet=False):
        if self.nodes >= self.node_limit or time.perf_counter() >= self.deadline:
            raise SearchStopped
        self.nodes += 1
        if quiet:
            self.qnodes += 1

    @staticmethod
    def _pack(score, ply):
        return score + ply if score > MATE // 2 else (score - ply if score < -MATE // 2 else score)

    @staticmethod
    def _unpack(score, ply):
        return score - ply if score > MATE // 2 else (score + ply if score < -MATE // 2 else score)

    def _ordered(self, pos, moves, data, tt_move, ply):
        board = pos.cells
        us = 0 if pos.side == 1 else 1
        enemy_attacks = data[1][1 - us]
        killers = self.killers.get(ply, ())

        def rank(move):
            if move == tt_move:
                return 100_000_000
            a, b = move
            size, victim = abs(board[a]), abs(board[b])
            if victim:
                score = 200_000 + 200 * victim + 240
                if enemy_attacks[b]:
                    score -= 170 * (size + victim)
                if not jumps(size + victim)[b]:
                    score -= 120
                return score
            score = self.history.get((pos.side, a, b), 0)
            if move in killers:
                score += 100_000
            if enemy_attacks[a]:
                score += 800 + 80 * size
            if enemy_attacks[b]:
                score -= 1000 + 100 * size
            return score + 10 * (CENTER[b] - CENTER[a])

        return sorted(moves, key=rank, reverse=True)

    def _qsearch(self, pos, alpha, beta, left, ply, data=None):
        self._visit(True)
        data = pos.analyse() if data is None else data
        terminal = pos.terminal(data, ply)
        if terminal is not None:
            return terminal
        stand = static_evaluate(pos, data)
        us = 0 if pos.side == 1 else 1
        # At the clock boundary a quiet move is a real terminal action; a
        # fictitious stand-pat would incorrectly hide forced draws/losses.
        boundary = pos.limit is not None and pos.clock + 1 >= pos.limit
        if left <= 0 and not boundary:
            return stand
        if boundary:
            candidates = data[0][us]
            best = -INF
        else:
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
            best = stand
            candidates = [m for m in data[0][us] if pos.cells[m[1]]]
        for move in self._ordered(pos, candidates, data, None, ply):
            # With limit=0 the root is terminal; with limit=1 every quiet
            # move terminates, and captures strictly reduce the piece count.
            undo = pos.push(move)
            try:
                score = -self._qsearch(pos, -beta, -alpha, left - 1, ply + 1)
            finally:
                pos.pop(move, undo)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def _search(self, pos, depth, alpha, beta, ply):
        if depth <= 0:
            return self._qsearch(pos, alpha, beta, self.qdepth, ply)
        self._visit()
        original_alpha = alpha
        key = pos.key()
        entry = self.tt.get(key)
        tt_move = entry[3] if entry else None
        if entry and entry[0] >= depth:
            value = self._unpack(entry[1], ply)
            if entry[2] == EXACT or (entry[2] == LOWER and value >= beta) or (entry[2] == UPPER and value <= alpha):
                self.tt_hits += 1
                return value
        data = pos.analyse()
        terminal = pos.terminal(data, ply)
        if terminal is not None:
            return terminal
        us = 0 if pos.side == 1 else 1
        moves = self._ordered(pos, data[0][us], data, tt_move, ply)
        best, best_move = -INF, moves[0]
        near_limit = pos.limit is not None and pos.clock + depth + 2 >= pos.limit
        for index, move in enumerate(moves):
            quiet = not pos.cells[move[1]]
            # Conservative late quiet-move reduction; all captures, escapes,
            # root moves and clock-critical lines keep their nominal depth.
            reduce = (self.use_lmr and ply > 0 and depth >= 4 and index >= 6
                      and quiet and not near_limit and len(moves) > 10
                      and not data[1][1 - us][move[0]]
                      and move not in self.killers.get(ply, ()))
            undo = pos.push(move)
            try:
                if index == 0:
                    score = -self._search(pos, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self._search(pos, depth - 1 - int(reduce), -alpha - 1, -alpha, ply + 1)
                    if reduce and score > alpha:
                        score = -self._search(pos, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self._search(pos, depth - 1, -beta, -alpha, ply + 1)
            finally:
                pos.pop(move, undo)
            if score > best:
                best, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if quiet:
                    old = self.killers.get(ply, ())
                    self.killers[ply] = (move,) + tuple(m for m in old if m != move)[:1]
                    hkey = (pos.side, move[0], move[1])
                    self.history[hkey] = min(50_000, self.history.get(hkey, 0) + depth * depth)
                break
        flag = UPPER if best <= original_alpha else (LOWER if best >= beta else EXACT)
        if self.tt_size:
            if key in self.tt or len(self.tt) < self.tt_size:
                self.tt[key] = (depth, self._pack(best, ply), flag, best_move)
        if ply == 0:
            self.root_move = best_move
        return best

    def choose(self, game):
        start = time.perf_counter()
        self.deadline = start + self.time_limit
        self.nodes = self.qnodes = self.tt_hits = 0
        self.tt, self.killers, self.history = {}, {}, {}
        self.stats = {"depth": 0, "score": None, "nodes": 0, "qnodes": 0,
                      "tt_hits": 0, "think_ms": 0.0, "pv": []}
        legal = game.legal_moves()
        if not legal:
            self.stats["think_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return None
        pos = Position(game)
        data = pos.analyse()
        root_moves = [(m.frm, m.to) for m in legal]
        best = self._ordered(pos, root_moves, data, None, 0)[0]
        score = 0
        completed = 0
        pv = []
        try:
            for depth in range(1, self.max_depth + 1):
                # Aspiration can fail; only publish a fully re-searched depth.
                lo, hi = (score - 120, score + 120) if depth >= 3 else (-INF, INF)
                value = self._search(pos, depth, lo, hi, 0)
                if value <= lo or value >= hi:
                    value = self._search(pos, depth, -INF, INF, 0)
                best, score, completed = self.root_move, value, depth
                pv = self._pv(game, best, depth)
                if abs(score) >= MATE - 256:
                    break
        except SearchStopped:
            pass
        self.stats.update(depth=completed, score=score if completed else None,
                          nodes=self.nodes, qnodes=self.qnodes, tt_hits=self.tt_hits,
                          think_ms=round((time.perf_counter() - start) * 1000, 2), pv=pv)
        result = Move(*best)
        # Fail closed if a future private-board change ever breaks the contract.
        if result not in legal:
            raise RuntimeError("internal search returned an illegal move")
        return result

    def _pv(self, game, first, depth):
        pos = Position(game)
        move, out = first, []
        for _ in range(depth):
            data = pos.analyse()
            if pos.terminal(data) is not None:
                break
            if move not in data[0][0 if pos.side == 1 else 1]:
                break
            out.append(Move(*move).text(bool(pos.cells[move[1]])))
            pos.push(move)
            entry = self.tt.get(pos.key())
            if not entry:
                break
            move = entry[3]
        return out


def choose(game, time_limit=2.0, **kwargs):
    """Convenience entry point compatible with examples/random_ai.py."""
    return StrongAI(time_limit=time_limit, **kwargs).choose(game)


choose_move = choose
MyAI = StrongAI


def answer(request, default_time_ms=2000, default_depth=64, default_limit=100):
    if isinstance(request, str):
        request = {"fen": request}
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object or FEN string")
    kind = str(request.get("request") or request.get("cmd") or "move").lower()
    if kind in ("ping", "info", "hello", "id"):
        return {"ok": True, "name": NAME, "kind": "ai", "protocol": 1,
                "version": "1.0", "rule": "sqrtchess-1.0"}
    if kind not in ("move", "bestmove", "go", "think"):
        raise ValueError("unknown request: " + kind)
    limit = request.get("no_capture_limit", default_limit)
    if isinstance(limit, str) and limit.lower() in ("off", "none"):
        limit = None
    game = Game.from_fen(request.get("fen") or request.get("position") or "startpos",
                         no_capture_limit=limit)
    budget = float(request.get("time_ms", default_time_ms))
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("time_ms must be finite and nonnegative")
    ai = StrongAI(time_limit=budget / 1000 * 0.95,
                  max_depth=request.get("depth", default_depth),
                  node_limit=request.get("node_limit", 500_000))
    move = ai.choose(game)
    payload = {"ok": True, "name": NAME, "move": None, "game_over": game.is_over,
               "think_ms": ai.stats["think_ms"], "search": ai.stats}
    if move is None:
        payload.update(reason="当前局面没有合法着法",
                       result=game.result.as_dict() if game.result else None)
    else:
        detail = game.move_dict(move)
        payload.update(move=detail["move"], notation=detail["notation"], detail=detail)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-ms", type=float, default=2000)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--limit", default="100", help="no-capture limit, or off")
    args = parser.parse_args()
    limit = None if args.limit.lower() in ("off", "none") else int(args.limit)
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line) if line[0] in '{["' else {"fen": line}
            response = answer(request, args.time_ms, args.depth, limit)
        except Exception as exc:
            response = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
