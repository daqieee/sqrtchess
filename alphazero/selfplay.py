"""自我对弈：用当前网络 + MCTS 生成训练数据。

每条样本记录 (局面 FEN, MCTS 访问分布 pi, 最终胜负 z)：

* pi 是搜索后的着法分布（前 temp_plies 步按温度 1 采样，之后取访问最多的一步）；
* z 是该局面「行棋方」视角的最终结果：胜 +1、负 -1、平局 0。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sqrtchess import BLACK, DEFAULT_NO_CAPTURE_LIMIT, WHITE, Game, Recorder, default_headers

from .mcts import MCTS
from .network import AZNetwork
from .values import DRAW_NO_MOVES_VALUE, sample_value

__all__ = ["Sample", "GameData", "play_game", "play_baseline_game",
           "play_sparring_game", "finalize"]


@dataclass
class Sample:
    """一条训练样本。"""

    fen: str
    policy: List[Tuple[str, float]]
    z: float = 0.0
    side: str = "w"
    ply: int = 0

    def to_dict(self) -> Dict:
        return {"fen": self.fen,
                "p": [[text, round(prob, 5)] for text, prob in self.policy],
                "z": self.z, "side": self.side, "ply": self.ply}

    @classmethod
    def from_dict(cls, data: Dict) -> "Sample":
        return cls(fen=data["fen"],
                   policy=[(item[0], float(item[1])) for item in data["p"]],
                   z=float(data.get("z", 0.0)), side=data.get("side", "w"),
                   ply=int(data.get("ply", 0)))


@dataclass
class GameData:
    """一局自我对弈的结果。"""

    samples: List[Sample] = field(default_factory=list)
    moves: List[str] = field(default_factory=list)
    result: str = "*"
    reason: str = ""
    plies: int = 0
    seconds: float = 0.0
    record: str = ""
    detail: str = ""
    truncated: bool = False


def finalize(game: Game, samples: List[Sample], recorder, started: float,
             keep_record: bool = False,
             draw_no_moves_value: float = DRAW_NO_MOVES_VALUE) -> GameData:
    """对局收尾：算出价值标签、整理成 GameData。"""
    truncated = game.result is None
    if truncated:
        outcome = game.adjudicate("no-capture-limit")
        winner = outcome.winner
        detail = "截断判定：" + outcome.detail
    else:
        winner = game.result.winner
        detail = game.result.detail
    outcome_reason = game.result.reason if game.result else "truncated"
    for sample in samples:
        sample.z = sample_value(game, sample.side, winner, outcome_reason,
                                draw_no_moves_value)
    data = GameData(samples=samples, moves=[item.notation for item in recorder.moves],
                    result=game.result.score if game.result else "*",
                    reason=outcome_reason,
                    plies=game.ply, seconds=time.time() - started,
                    detail=detail, truncated=truncated)
    if keep_record:
        data.record = recorder.finish()
    return data


def _sample_move(policy: Dict, rng: random.Random, temperature: float):
    moves = list(policy)
    if not moves:
        return None
    if temperature is None or temperature <= 1e-3:
        best = max(policy.values())
        winners = [move for move in moves if policy[move] == best]
        return rng.choice(winners)
    total = sum(policy.values())
    if total <= 0:
        return rng.choice(moves)
    threshold = rng.random() * total
    acc = 0.0
    for move in moves:
        acc += policy[move]
        if acc >= threshold:
            return move
    return moves[-1]


def play_game(network: AZNetwork, sims: int = 60, max_plies: int = 400,
              temp_plies: int = 20, c_puct: float = 1.5, seed: Optional[int] = None,
              limit=DEFAULT_NO_CAPTURE_LIMIT, noise: bool = True,
              material_weight: float = 0.45,
              draw_no_moves_value: float = DRAW_NO_MOVES_VALUE,
              leaf_depth: int = 1, keep_record: bool = True) -> GameData:
    """让网络与自己下一整局，返回训练样本与棋谱。"""
    started = time.time()
    game = Game.new(no_capture_limit=limit)
    recorder = Recorder(game, default_headers(white="AlphaZero", black="AlphaZero",
                                              event="AlphaZero 自我对弈"))
    search = MCTS(network, sims=sims, c_puct=c_puct, material_weight=material_weight,
                  draw_no_moves_value=draw_no_moves_value, leaf_depth=leaf_depth,
                  seed=None if seed is None else seed * 7919 + 13)
    samples: List[Sample] = []
    rng = random.Random(seed)

    while not game.is_over and game.ply < max_plies:
        temperature = 1.0 if game.ply < temp_plies else 0.0
        root = search.search(game, add_noise=noise)
        policy = MCTS.visit_policy(root, temperature=temperature if temperature > 0 else 0.0)
        if not policy:
            break
        entries = []
        for move, probability in policy.items():
            if probability > 0:
                entries.append((game.move_dict(move)["notation"], probability))
        samples.append(Sample(fen=game.fen(), policy=entries,
                              side=game.status(False)["turn"], ply=game.ply))
        move = _sample_move(policy, rng, temperature)
        if move is None:
            break
        recorder.push(move)

    truncated = game.result is None
    if truncated:
        # 到达步数上限还没自然结束：用规则 7 的平局判定作为价值目标，
        # 这样每条样本都有明确的学习信号（否则截断对局只能给出「和棋」标签）。
        outcome = game.adjudicate("no-capture-limit")
        winner = outcome.winner
        detail = "截断判定：" + outcome.detail
    else:
        winner = game.result.winner
        detail = game.result.detail
    outcome_reason = game.result.reason if game.result else "truncated"
    for sample in samples:
        sample.z = sample_value(game, sample.side, winner, outcome_reason,
                                draw_no_moves_value)

    data = GameData(samples=samples, moves=[info.notation for info in recorder.moves],
                    result=game.result.score if game.result else "*",
                    reason=game.result.reason if game.result else "truncated",
                    plies=game.ply, seconds=time.time() - started,
                    detail=detail, truncated=truncated)
    if keep_record:
        data.record = recorder.finish()
    return data

def play_baseline_game(engine, seed: Optional[int] = None,
                       limit=DEFAULT_NO_CAPTURE_LIMIT, max_plies: int = 400,
                       epsilon: float = 0.1,
                       draw_no_moves_value: float = DRAW_NO_MOVES_VALUE,
                       keep_record: bool = False) -> GameData:
    """用「尽可能多吃子」基线自我对弈，生成模仿学习数据。

    策略目标是基线所选着法（做 epsilon 平滑），价值目标是该局的真实结果，
    这样训练出来的网络在搜索之前就具备「进攻、抢吃子」的直觉，作为 AlphaZero 的起点。
    """
    started = time.time()
    game = Game.new(no_capture_limit=limit)
    recorder = Recorder(game, default_headers(white="CaptureAI", black="CaptureAI",
                                              event="吃子基线自我对弈"))
    samples: List[Sample] = []
    if seed is not None:
        engine.random.seed(seed)
    while not game.is_over and game.ply < max_plies:
        moves = game.legal_moves()
        if not moves:
            break
        # 关键：基线在「并列最优」的着法之间是随机选的，如果直接做 one-hot 模仿，
        # 目标本身就是噪声（交叉熵只能降到 ln(并列数)）。这里改成「并列最优平分概率」的
        # 软目标，网络学到的才是真正有用的信号：哪些着法在吃子意义上是好棋。
        scored = engine.rank_moves(game)
        if not scored:
            break
        best_score = scored[0][0]
        pool = [move for score, move in scored if score >= best_score - 1e-9]
        rest = len(scored) - len(pool)
        if rest > 0:
            share = (1.0 - epsilon) / len(pool)
            other = epsilon / rest
        else:
            # 所有着法并列最优：概率在它们之间平分，保证总和为 1
            share = 1.0 / len(pool)
            other = 0.0
        entries = []
        for score, move in scored:
            probability = share if score >= best_score - 1e-9 else other
            entries.append((game.move_dict(move)["notation"], probability))
        chosen = engine.random.choice(pool)
        samples.append(Sample(fen=game.fen(), policy=entries,
                              side=game.status(False)["turn"], ply=game.ply))
        recorder.push(chosen)

    winner = game.result.winner if game.result else None
    outcome_reason = game.result.reason if game.result else "truncated"
    for sample in samples:
        sample.z = sample_value(game, sample.side, winner, outcome_reason,
                                draw_no_moves_value)
    data = GameData(samples=samples, moves=[item.notation for item in recorder.moves],
                    result=game.result.score if game.result else "*",
                    reason=game.result.reason if game.result else "",
                    plies=game.ply, seconds=time.time() - started,
                    detail=game.result.detail if game.result else "")
    if keep_record:
        data.record = recorder.finish()
    return data

def play_sparring_game(network: AZNetwork, opponent, sims: int = 60, max_plies: int = 400,
                       temp_plies: int = 20, c_puct: float = 1.5,
                       limit=DEFAULT_NO_CAPTURE_LIMIT, seed: Optional[int] = None,
                       noise: bool = True, net_is_white: bool = True,
                       material_weight: float = 0.45,
                       draw_no_moves_value: float = DRAW_NO_MOVES_VALUE,
                       leaf_depth: int = 1, keep_record: bool = False) -> GameData:
    """网络与外部对手（例如「尽可能多吃子」基线）对局。

    只记录**网络自己**的决策样本（对手的走子不进训练集），价值目标仍是真实胜负。
    这样网络能从「会狠狠惩罚物质损失的强对手」身上学到：丢子就会输。
    """
    started = time.time()
    game = Game.new(no_capture_limit=limit)
    recorder = Recorder(game, default_headers(
        white="AlphaZero" if net_is_white else "吃子基线",
        black="吃子基线" if net_is_white else "AlphaZero",
        event="AlphaZero 与基线陪练"))
    search = MCTS(network, sims=sims, c_puct=c_puct, material_weight=material_weight,
                  draw_no_moves_value=draw_no_moves_value, leaf_depth=leaf_depth,
                  seed=None if seed is None else seed * 7919 + 13)
    samples: List[Sample] = []
    rng = random.Random(seed)
    net_color = WHITE if net_is_white else BLACK

    while not game.is_over and game.ply < max_plies:
        if game.side == net_color:
            temperature = 1.0 if game.ply < temp_plies else 0.0
            root = search.search(game, add_noise=noise)
            policy = MCTS.visit_policy(root, temperature=temperature)
            if not policy:
                break
            entries = []
            for move, probability in policy.items():
                if probability > 0:
                    entries.append((game.move_dict(move)["notation"], probability))
            samples.append(Sample(fen=game.fen(), policy=entries,
                                  side=game.status(False)["turn"], ply=game.ply))
            move = _sample_move(policy, rng, temperature)
        else:
            move = opponent.choose(game)
        if move is None:
            break
        recorder.push(move)

    truncated = game.result is None
    if truncated:
        outcome = game.adjudicate("no-capture-limit")
        winner = outcome.winner
        detail = "截断判定：" + outcome.detail
    else:
        winner = game.result.winner
        detail = game.result.detail
    outcome_reason = game.result.reason if game.result else "truncated"
    for sample in samples:
        sample.z = sample_value(game, sample.side, winner, outcome_reason,
                                draw_no_moves_value)
    data = GameData(samples=samples, moves=[item.notation for item in recorder.moves],
                    result=game.result.score if game.result else "*",
                    reason=game.result.reason if game.result else "truncated",
                    plies=game.ply, seconds=time.time() - started,
                    detail=detail, truncated=truncated)
    if keep_record:
        data.record = recorder.finish()
    return data
