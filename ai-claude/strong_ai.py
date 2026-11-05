#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根号棋强力 AI：迭代加深 + PVS(alpha-beta) + 置换表 + 静态搜索 + 针对规则 7 的评估。

完全独立（不依赖 sqrtchess 包），自带棋盘/走法生成，只按 RULES.md / NOTATION.md 实现。

用法
----
1. 作为外部程序（AI 应答协议，每行一个 JSON）::

       python strong_ai.py [--time 2.0] [--limit 100]

   GUI 里把白方/黑方设为“外部程序”，命令行填 ``python strong_ai.py``。
   请求字段：fen、time_ms（思考上限提示）、depth（最大深度）、seed、limit（无吃子上限）。

2. 命令行直接分析::

       python strong_ai.py --fen startpos --time 3

3. 自检（走法数、与随机/吃子基线对弈）::

       python strong_ai.py --selftest
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time

# --------------------------------------------------------------------------- #
# 常量与预计算
# --------------------------------------------------------------------------- #
N = 11
NSQ = N * N
MAXSZ = 200                  # 平方和最大 10²+10²=200，更大的棋子必然不能动
INF = 10 ** 9
MATE = 30000                 # 吃光
ADJ = 20000                  # 规则 7 判胜
MATE_T = 19000
QDEPTH = 8
TT_MAX = 800_000
DOM = 250                    # 规则 7 “双项占优” 的评估奖励
START_FEN = ("." + "B4B8B5B2B9B2B5B8B4" + "./." + "B1" * 9 + "./" +
             "/".join(["..........."] * 7) + "/." + "W1" * 9 + "./." +
             "W4W8W5W2W9W2W5W8W4" + ". w 0 1")
START_FEN = ".B4B8B5B2B9B2B5B8B4./.B1B1B1B1B1B1B1B1B1./.........../.........../.........../" \
            ".........../.........../.........../.........../.W1W1W1W1W1W1W1W1W1./" \
            ".W4W8W5W2W9W2W5W8W4. w 0 1"


def _build_targets():
    table = []
    for n in range(MAXSZ + 1):
        vecs = []
        if n > 0:
            for dx in range(-14, 15):
                for dy in range(-14, 15):
                    if dx * dx + dy * dy == n:
                        vecs.append((dx, dy))
        per = []
        for sq in range(NSQ):
            r, c = divmod(sq, N)
            per.append(tuple((r + dy) * N + (c + dx) for dx, dy in vecs
                             if 0 <= r + dy < N and 0 <= c + dx < N))
        table.append(per)
    return table


TARGETS = _build_targets()

ZOFF = 300
_rng = random.Random(0x5EED2026)
ZOB = [[_rng.getrandbits(64) for _ in range(2 * ZOFF + 1)] for _ in range(NSQ)]
for _row in ZOB:
    _row[ZOFF] = 0
ZSIDE = _rng.getrandbits(64)


class TimeUp(Exception):
    pass


# --------------------------------------------------------------------------- #
# SQFEN 解析与着法文本
# --------------------------------------------------------------------------- #
_EMPTY = set(".-*_")


def parse_fen(text):
    t = (text or "startpos").strip()
    if t.lower() in ("startpos", "initial", "开局", ""):
        t = START_FEN
    parts = t.split()
    rows = parts[0].split("/")
    if len(rows) != N:
        raise ValueError("FEN 需要 11 行棋盘")
    cells = [0] * NSQ
    for i, row in enumerate(rows):
        r = N - 1 - i
        c = 0
        k = 0
        while k < len(row):
            ch = row[k]
            if ch in _EMPTY:
                c += 1
                k += 1
            elif ch in "WwBb":
                k += 1
                j = k
                while j < len(row) and row[j].isdigit():
                    j += 1
                if j == k:
                    raise ValueError("棋子缺少大小")
                size = int(row[k:j])
                if size > ZOFF:
                    raise ValueError("棋子过大")
                if c >= N:
                    raise ValueError("FEN 一行超过 11 格")
                cells[r * N + c] = size if ch in "Ww" else -size
                c += 1
                k = j
            else:
                raise ValueError("FEN 含有非法字符 %r" % ch)
        if c != N:
            raise ValueError("FEN 第 %d 行不是 11 格" % (i + 1))
    side = 1 if len(parts) > 1 and parts[1].lower() == "b" else 0
    nocap = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return cells, side, nocap


def sq_name(sq):
    r, c = divmod(sq, N)
    return "%s%d" % (chr(97 + c), r + 1)


def move_text(cells, f, t):
    size = abs(cells[f])
    return "%d%s%s%s" % (size, sq_name(f), "x" if cells[t] else "-", sq_name(t))


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, limit=100, seed=None):
        self.limit = limit
        self.rng = random.Random(seed)
        self.tt = {}
        self.hist = {}
        self.killers = [[None, None] for _ in range(80)]
        self.deadline = 0.0
        self.nodes = 0
        self.load("startpos")

    # ---- 局面 ---------------------------------------------------------- #
    def load(self, fen):
        cells, side, nocap = parse_fen(fen)
        self.cells = cells
        self.stm = side
        self.nocap = nocap
        self.pieces = [set(), set()]
        self.mass = [0, 0]
        h = ZSIDE if side == 1 else 0
        for sq, v in enumerate(cells):
            if v > 0:
                self.pieces[0].add(sq)
                self.mass[0] += v
            elif v < 0:
                self.pieces[1].add(sq)
                self.mass[1] -= v
            if v:
                h ^= ZOB[sq][v + ZOFF]
        self.hash = h

    def make(self, f, t):
        cells = self.cells
        v = cells[f]
        tv = cells[t]
        stm = self.stm
        opp = 1 - stm
        undo = (f, t, v, tv, self.nocap, self.hash)
        nv = v - tv                      # 白正黑负：吃子后大小相加
        cells[f] = 0
        cells[t] = nv
        mine = self.pieces[stm]
        mine.remove(f)
        mine.add(t)
        h = self.hash ^ ZOB[f][v + ZOFF] ^ ZOB[t][tv + ZOFF] ^ ZOB[t][nv + ZOFF] ^ ZSIDE
        if tv:
            self.pieces[opp].remove(t)
            a = tv if tv > 0 else -tv
            self.mass[stm] += a
            self.mass[opp] -= a
            self.nocap = 0
        else:
            self.nocap += 1
        self.hash = h
        self.stm = opp
        return undo

    def unmake(self, undo):
        f, t, v, tv, nocap, h = undo
        mover = 1 - self.stm
        cells = self.cells
        cells[f] = v
        cells[t] = tv
        mine = self.pieces[mover]
        mine.remove(t)
        mine.add(f)
        if tv:
            opp = self.stm
            self.pieces[opp].add(t)
            a = tv if tv > 0 else -tv
            self.mass[mover] -= a
            self.mass[opp] += a
        self.nocap = nocap
        self.hash = h
        self.stm = mover

    # ---- 走法生成 ------------------------------------------------------ #
    def gen(self, side, caps_only=False):
        cells = self.cells
        out = []
        for f in self.pieces[side]:
            v = cells[f]
            n = v if v > 0 else -v
            if n > MAXSZ:
                continue
            for t in TARGETS[n][f]:
                c = cells[t]
                if c == 0:
                    if not caps_only:
                        out.append((f, t))
                elif (c > 0) != (v > 0):
                    out.append((f, t))
        return out

    def has_move(self, side):
        cells = self.cells
        for f in self.pieces[side]:
            v = cells[f]
            n = v if v > 0 else -v
            if n > MAXSZ:
                continue
            for t in TARGETS[n][f]:
                c = cells[t]
                if c == 0 or (c > 0) != (v > 0):
                    return True
        return False

    # ---- 规则 5/6/7 ---------------------------------------------------- #
    def adjudicate(self):
        wc, bc = len(self.pieces[0]), len(self.pieces[1])
        wm, bm = self.mass
        if wc > bc and wm > bm:
            return 1
        if bc > wc and bm > wm:
            return -1
        return 0

    def adj_score(self, ply):
        r = self.adjudicate()
        if r == 0:
            return 0
        v = ADJ - ply
        return v if (r == 1) == (self.stm == 0) else -v

    def terminal(self):
        """对局是否结束：返回 None 或 (白方视角结果 1/-1/0, 原因)。"""
        if not self.pieces[0] and not self.pieces[1]:
            return 0, "all-captured"
        if not self.pieces[1]:
            return 1, "all-captured"
        if not self.pieces[0]:
            return -1, "all-captured"
        if not self.has_move(0) or not self.has_move(1):
            return self.adjudicate(), "no-moves"
        if self.limit and self.nocap >= self.limit:
            return self.adjudicate(), "no-capture-limit"
        return None

    # ---- 评估（白方视角）----------------------------------------------- #
    def evaluate(self):
        cells = self.cells
        P0, P1 = self.pieces
        wc, bc = len(P0), len(P1)
        wm, bm = self.mass
        score = 100 * (wc - bc) + 10 * (wm - bm)
        if wc > bc and wm > bm:
            score += DOM + 3 * self.nocap        # 守住即可靠规则 7 取胜
        elif bc > wc and bm > wm:
            score -= DOM + 3 * self.nocap
        ws = 0
        for f in P0:
            v = cells[f]
            m = a = 0
            if v <= MAXSZ:
                for t in TARGETS[v][f]:
                    c = cells[t]
                    if c <= 0:
                        m += 1
                        if c < 0:
                            a += 3 - c
            ws += 2 * m + a - (10 if m == 0 else 0)
        bs = 0
        for f in P1:
            v = -cells[f]
            m = a = 0
            if v <= MAXSZ:
                for t in TARGETS[v][f]:
                    c = cells[t]
                    if c >= 0:
                        m += 1
                        if c > 0:
                            a += 3 + c
            bs += 2 * m + a - (10 if m == 0 else 0)
        score += ws - bs
        score += 8 if self.stm == 0 else -8
        return score

    # ---- 排序 ---------------------------------------------------------- #
    def order(self, moves, ttmove, ply):
        cells = self.cells
        hist = self.hist
        k0, k1 = self.killers[ply] if ply < 80 else (None, None)
        scored = []
        for m in moves:
            f, t = m
            tv = cells[t]
            if tv:
                s = 1_000_000 + (tv if tv > 0 else -tv) * 16 - abs(cells[f])
            else:
                s = hist.get(f * 128 + t, 0)
                if m == k0:
                    s += 500_000
                elif m == k1:
                    s += 400_000
            if m == ttmove:
                s = 10 ** 8
            scored.append((s, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ---- 静态搜索 ------------------------------------------------------ #
    def qsearch(self, alpha, beta, ply, qd):
        self.nodes += 1
        if not (self.nodes & 1023) and time.time() > self.deadline:
            raise TimeUp
        stm = self.stm
        stand = self.evaluate()
        if stm == 1:
            stand = -stand
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        if qd <= 0:
            return stand
        caps = self.gen(stm, True)
        if not caps:
            return stand
        cells = self.cells
        caps.sort(key=lambda m: abs(cells[m[1]]) * 16 - abs(cells[m[0]]), reverse=True)
        best = stand
        for f, t in caps:
            undo = self.make(f, t)
            if not self.pieces[self.stm]:
                score = MATE - ply
            else:
                score = -self.qsearch(-beta, -alpha, ply + 1, qd - 1)
            self.unmake(undo)
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # ---- 主搜索 -------------------------------------------------------- #
    def search(self, depth, alpha, beta, ply):
        self.nodes += 1
        if not (self.nodes & 1023) and time.time() > self.deadline:
            raise TimeUp
        stm = self.stm
        opp = 1 - stm
        if not self.pieces[stm]:
            return -(MATE - ply)
        if not self.pieces[opp]:
            return MATE - ply
        if (self.limit and self.nocap >= self.limit) or not self.has_move(opp):
            return self.adj_score(ply)
        if depth <= 0:
            return self.qsearch(alpha, beta, ply, QDEPTH)

        alpha0 = alpha
        use_tt = (not self.limit) or (self.nocap + depth + 2 < self.limit)
        h = self.hash
        ttmove = None
        if use_tt:
            e = self.tt.get(h)
            if e is not None:
                ed, flag, sc, ttmove = e
                if ed >= depth:
                    if sc > MATE_T:
                        sc -= ply
                    elif sc < -MATE_T:
                        sc += ply
                    if flag == 0:
                        return sc
                    if flag == 1 and sc >= beta:
                        return sc
                    if flag == 2 and sc <= alpha:
                        return sc
        moves = self.gen(stm)
        if not moves:
            return self.adj_score(ply)
        moves = self.order(moves, ttmove, ply)
        cells = self.cells
        best = -INF
        bestmove = None
        for i, m in enumerate(moves):
            f, t = m
            quiet = cells[t] == 0
            undo = self.make(f, t)
            if not self.pieces[self.stm]:
                score = MATE - ply
            elif i == 0:
                score = -self.search(depth - 1, -beta, -alpha, ply + 1)
            else:
                red = 1 if (depth >= 3 and i >= 5 and quiet) else 0
                score = -self.search(depth - 1 - red, -alpha - 1, -alpha, ply + 1)
                if score > alpha and red:
                    score = -self.search(depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self.search(depth - 1, -beta, -alpha, ply + 1)
            self.unmake(undo)
            if score > best:
                best = score
                bestmove = m
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if quiet and ply < 80:
                            kl = self.killers[ply]
                            if kl[0] != m:
                                kl[1] = kl[0]
                                kl[0] = m
                            key = f * 128 + t
                            self.hist[key] = self.hist.get(key, 0) + depth * depth
                        break
        if use_tt and (len(self.tt) < TT_MAX or h in self.tt):
            flag = 2 if best <= alpha0 else (1 if best >= beta else 0)
            sc = best
            if sc > MATE_T:
                sc += ply
            elif sc < -MATE_T:
                sc -= ply
            self.tt[h] = (depth, flag, sc, bestmove)
        return best

    # ---- 根节点：迭代加深 ---------------------------------------------- #
    def think(self, fen, time_s=2.0, max_depth=40, seed=None):
        self.load(fen)
        if seed is not None:
            self.rng = random.Random(seed)
        root_cells = self.cells[:]
        start = time.time()
        self.deadline = start + max(0.02, time_s)
        self.nodes = 0
        self.hist = {}
        self.killers = [[None, None] for _ in range(80)]
        if len(self.tt) > TT_MAX // 2:
            self.tt.clear()

        stm = self.stm
        moves = self.gen(stm)
        info = {"depth": 0, "score": 0, "nodes": 0}
        if not moves:
            return None, info
        if seed is not None:
            self.rng.shuffle(moves)
        moves = self.order(moves, None, 0)
        best = moves[0]
        best_score = 0
        scores = {}
        if len(moves) > 1:
            for depth in range(1, max_depth + 1):
                cur_best, cur_score = None, -INF
                alpha = -INF
                try:
                    for i, m in enumerate(moves):
                        f, t = m
                        quiet = root_cells[t] == 0
                        undo = self.make(f, t)
                        if not self.pieces[self.stm]:
                            score = MATE
                        elif i == 0:
                            score = -self.search(depth - 1, -INF, -alpha, 1)
                        else:
                            red = 1 if (depth >= 3 and i >= 6 and quiet) else 0
                            score = -self.search(depth - 1 - red, -alpha - 1, -alpha, 1)
                            if score > alpha and red:
                                score = -self.search(depth - 1, -alpha - 1, -alpha, 1)
                            if score > alpha:
                                score = -self.search(depth - 1, -INF, -alpha, 1)
                        self.unmake(undo)
                        scores[m] = score
                        if score > cur_score:
                            cur_best, cur_score = m, score
                        if score > alpha:
                            alpha = score
                except TimeUp:
                    if cur_best is not None:
                        best, best_score = cur_best, cur_score
                    break
                best, best_score = cur_best, cur_score
                info["depth"] = depth
                moves.sort(key=lambda m: (m == best, scores.get(m, -INF)), reverse=True)
                if abs(best_score) > MATE_T and depth >= 2:
                    break
                if time.time() - start > time_s * 0.4:
                    break
        info.update(score=best_score, nodes=self.nodes)
        info["text"] = move_text(root_cells, *best)
        info["from"], info["to"] = sq_name(best[0]), sq_name(best[1])
        info["is_capture"] = root_cells[best[1]] != 0
        info["size"] = abs(root_cells[best[0]])
        info["captured"] = abs(root_cells[best[1]])
        return best, info


# --------------------------------------------------------------------------- #
# AI 应答协议
# --------------------------------------------------------------------------- #
def answer(engine, request, default_time, default_depth):
    kind = str(request.get("request") or "move").lower()
    if kind in ("ping", "info", "hello"):
        return {"ok": True, "name": "sqrt-pvs-ai", "kind": "ai", "protocol": 1}
    if request.get("limit") is not None:
        lim = request["limit"]
        engine.limit = 0 if str(lim).lower() in ("off", "none", "0") else int(lim)
    t = default_time
    if request.get("time_ms"):
        t = min(60.0, max(0.05, float(request["time_ms"]) / 1000.0 * 0.8))
    depth = int(request["depth"]) if request.get("depth") else default_depth
    t0 = time.time()
    move, info = engine.think(request.get("fen") or "startpos", t, depth, request.get("seed"))
    if move is None:
        return {"ok": True, "move": None, "reason": "当前局面没有合法着法"}
    detail = {"move": info["from"] + ("x" if info["is_capture"] else "-") + info["to"],
              "notation": info["text"], "from": info["from"], "to": info["to"],
              "size": info["size"], "captured": info["captured"],
              "is_capture": info["is_capture"]}
    return {"ok": True, "move": info["text"], "notation": info["text"], "detail": detail,
            "level": "sqrt-pvs", "name": "sqrt-pvs-ai",
            "think_ms": round((time.time() - t0) * 1000, 1),
            "depth": info["depth"], "score": info["score"], "nodes": info["nodes"]}


def serve(args):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    engine = Engine(limit=args.limit)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                request = {"request": "move", "fen": line}
        except json.JSONDecodeError:
            request = {"request": "move", "fen": line}
        try:
            response = answer(engine, request, args.time, args.depth)
        except Exception as exc:  # noqa: BLE001
            response = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# 自检：走法数 + 与基线对弈
# --------------------------------------------------------------------------- #
def perft(engine, depth):
    if depth == 0:
        return 1
    total = 0
    for f, t in engine.gen(engine.stm):
        u = engine.make(f, t)
        total += perft(engine, depth - 1)
        engine.unmake(u)
    return total


def baseline_move(engine, rng):
    moves = engine.gen(engine.stm)
    caps = [m for m in moves if engine.cells[m[1]]]
    if caps:
        return max(caps, key=lambda m: abs(engine.cells[m[1]]))
    return rng.choice(moves)


def selftest(args):
    eng = Engine(limit=args.limit)
    eng.load("startpos")
    print("初始合法着法数:", len(eng.gen(0)), "(应为 22)")
    print("perft(2):", perft(eng, 2))
    rng = random.Random(7)
    games = 4
    ai_score = 0.0
    for g in range(games):
        ai_white = g % 2 == 0
        game = Engine(limit=args.limit)
        game.load("startpos")
        ply = 0
        while ply < 600:
            res = game.terminal()
            if res:
                break
            ai_turn = (game.stm == 0) == ai_white
            if ai_turn:
                fen_move, _ = brain.think_from(game, args.time)
            else:
                fen_move = baseline_move(game, rng)
            game.make(*fen_move)
            ply += 1
        res = game.terminal() or (0, "ply-cap")
        r = res[0]
        pts = 0.5 if r == 0 else (1.0 if (r == 1) == ai_white else 0.0)
        ai_score += pts
        print("第%d局 AI执%s: 结果=%s 原因=%s 步数=%d 子数 %d:%d 大小和 %d:%d"
              % (g + 1, "白" if ai_white else "黑", r, res[1], ply,
                 len(game.pieces[0]), len(game.pieces[1]), game.mass[0], game.mass[1]))
    print("AI 得分 %.1f / %d" % (ai_score, games))
    return 0


class _Brain:
    """自检用：对一个正在进行的 Engine 局面做一次思考（不污染对局状态）。"""

    def __init__(self):
        self.engine = Engine()

    def think_from(self, game, time_s):
        self.engine.limit = game.limit
        fen = game_to_fen(game)
        return self.engine.think(fen, time_s, 40)


def game_to_fen(game):
    rows = []
    for r in range(N - 1, -1, -1):
        s = ""
        for c in range(N):
            v = game.cells[r * N + c]
            s += "." if v == 0 else ("W%d" % v if v > 0 else "B%d" % -v)
        rows.append(s)
    return "%s %s %d 1" % ("/".join(rows), "w" if game.stm == 0 else "b", game.nocap)


brain = _Brain()


def main():
    ap = argparse.ArgumentParser(description="根号棋强力 AI")
    ap.add_argument("--time", type=float, default=2.0, help="每步思考秒数（默认 2.0）")
    ap.add_argument("--depth", type=int, default=40, help="最大搜索深度")
    ap.add_argument("--limit", type=int, default=100, help="无吃子步数上限（0=不判定）")
    ap.add_argument("--fen", help="分析指定局面并打印结果")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest(args)
    if args.fen:
        eng = Engine(limit=args.limit)
        print(json.dumps(answer(eng, {"fen": args.fen}, args.time, args.depth),
                         ensure_ascii=False, indent=2))
        return 0
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())
