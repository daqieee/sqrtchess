"""AlphaZero 训练循环：自我对弈 → 训练 → 评估门控 → 迭代。

每一轮迭代：

1. **自我对弈**：用当前 best 网络 + PUCT 搜索下 N 局（可多进程并行），
   记录 (局面, MCTS 访问分布 pi, 最终胜负 z)；
2. **训练**：从经验池里采样 mini-batch，用 Adam 最小化
   「策略交叉熵 + 价值均方误差 + L2」；
3. **评估门控**：候选网络与当前 best 对战，得分率达到阈值才被采纳（AlphaZero 的门控）；
4. **基线锚点**：每轮都让 best 与「尽可能多吃子」基线对战，作为贯穿训练的进度标尺；
5. 保存 best / 候选权重、指标 JSONL 与若干自我对弈棋谱。

第一次迭代之前可以先用基线自我对弈生成的数据做模仿学习（--bootstrap-games），
让网络以「尽可能多吃子」的进攻直觉作为起点。
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sqrtchess import DEFAULT_NO_CAPTURE_LIMIT, CaptureAI, Game, color_name

from .arena import MatchResult, play_match
from .encoding import encode
from .network import AZNetwork
from .players import CapturePlayer, MCTSPlayer, load_player
from .selfplay import (GameData, Sample, play_baseline_game, play_game,
                       play_sparring_game)

__all__ = ["TrainConfig", "Trainer"]

_WORKER: Dict[str, object] = {}


def _worker_play(task) -> Dict:
    """子进程任务：跑一局自我对弈（或基线模仿对局）。"""
    kind, weights_json, seed, options = task
    if _WORKER.get("weights") != weights_json:
        _WORKER["net"] = AZNetwork.from_json(weights_json)
        _WORKER["weights"] = weights_json
    common = dict(seed=seed, limit=options["limit"], max_plies=options["max_plies"],
                  keep_record=False)
    draw_value = options.get("draw_no_moves_value", 0.1)
    material_weight = options.get("material_weight", 0.45)
    opponent_name = ""
    if kind == "baseline":
        engine = _make_opponent(options.get("bootstrap_opponent", "capture"), seed,
                                options.get("baseline_depth", 4))
        opponent_name = options.get("bootstrap_opponent", "capture")
        data = play_baseline_game(engine, draw_no_moves_value=draw_value, **common)
    elif kind == "spar":
        # 陪练对手从一个池子里轮换：吃子基线 / 随机 / 贪心 / 搜索2，
        # 避免网络只学会对付某一种棋风。
        specs = options.get("spar_opponents") or ["baseline"]
        opponent_name = specs[seed % len(specs)]
        opponent = _make_opponent(opponent_name, seed, options.get("baseline_depth", 4))
        data = play_sparring_game(_WORKER["net"], opponent, sims=options["sims"],
                                  temp_plies=options["temp_plies"],
                                  c_puct=options["c_puct"],
                                  material_weight=material_weight,
                                  draw_no_moves_value=draw_value,
                                  leaf_depth=options.get("leaf_depth", 1),
                                  net_is_white=(seed % 2 == 0), **common)
    elif options.get("search_kind") == "hybrid":
        from .hybrid import play_hybrid_game

        data = play_hybrid_game(_WORKER["net"], depth=options.get("ab_depth", 1),
                                net_scale=options.get("net_scale", 30.0),
                                temp_plies=options["temp_plies"],
                                draw_no_moves_value=draw_value, **common)
    else:
        data = play_game(_WORKER["net"], sims=options["sims"],
                         temp_plies=options["temp_plies"], c_puct=options["c_puct"],
                         material_weight=material_weight,
                         draw_no_moves_value=draw_value,
                         leaf_depth=options.get("leaf_depth", 1), **common)
    return {"kind": kind, "opponent": opponent_name, "plies": data.plies,
            "result": data.result, "reason": data.reason, "seconds": data.seconds,
            "samples": [sample.to_dict() for sample in data.samples],
            "moves": data.moves}


def _make_opponent(spec: str, seed: int, depth: int):
    """按名字创建陪练对手。"""
    from sqrtchess import make_ai

    if spec in ("baseline", "capture", "吃子基线"):
        return CaptureAI(depth=depth, seed=seed + 1)
    return make_ai(spec, seed=seed + 1)


@dataclass
class TrainConfig:
    """训练配置。"""

    iterations: int = 6
    games_per_iter: int = 40
    sims: int = 60
    temp_plies: int = 20
    max_plies: int = 400
    c_puct: float = 1.5
    limit: int = DEFAULT_NO_CAPTURE_LIMIT
    channels: int = 24
    policy_hidden: int = 32
    value_hidden: int = 32
    batch_size: int = 32
    lr: float = 0.002
    lr_decay: float = 0.9
    train_steps: int = 300
    buffer_samples: int = 20000
    arena_games: int = 8
    arena_sims: int = 30
    arena_max_plies: int = 240
    promote_threshold: float = 0.55
    baseline_games: int = 6
    baseline_depth: int = 4
    spar_fraction: float = 0.3
    material_weight: float = 0.45
    draw_no_moves_value: float = 0.1
    leaf_depth: int = 1
    #: 自我对弈/评估用的搜索：mcts（PUCT）或 hybrid（神经引导 Alpha-Beta）
    search_kind: str = "mcts"
    #: hybrid 自我对弈用的 Alpha-Beta 深度
    ab_depth: int = 1
    #: hybrid 叶子评估里网络价值的权重（±1 价值 = 多少子力分）
    net_scale: float = 30.0
    #: hybrid 评估时每步的时间上限（秒），防止搜索太慢
    arena_time_limit: Optional[float] = 0.35
    #: 每多少轮做一次门控与锚点评估（省时间）
    eval_every: int = 1
    #: 陪练对手池（按局数轮换，避免只针对某一种棋风过拟合）
    spar_opponents: str = "baseline,random,greedy,search2"
    anchor_others_games: int = 3
    init_model: str = ""
    #: 主锚点对手（默认用 Alpha-Beta 深度 2；可换 capture / random / greedy / search2 / abN）
    anchor_opponent: str = "ab2"
    #: 模仿学习（bootstrap）用的对手
    bootstrap_opponent: str = "ab2"
    bootstrap_games: int = 20
    workers: int = 1
    seed: int = 1
    out_dir: str = "runs/az"
    keep_records: int = 4

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainConfig":
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**known)


class Trainer:
    """训练器。"""

    def __init__(self, config: TrainConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "games").mkdir(exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.log_path = self.out_dir / "train.log"
        self.buffer: List[Sample] = []
        self.history: List[Dict] = []
        self.pool: Optional[ProcessPoolExecutor] = None
        self.rng = random.Random(config.seed)
        self._record_saved = 0

    # ------------------------------------------------------------- 基础设施

    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = "[%s] %s" % (stamp, message)
        if self.verbose:
            print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_metrics(self, record: Dict) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------- 自我对弈

    def self_play(self, network: AZNetwork, games: int, kind: str = "az",
                  seeds: Optional[Sequence[int]] = None) -> List[Dict]:
        """并行跑若干局自我对弈，返回每局的摘要（样本已展开）。"""
        if games <= 0:
            return []
        options = {"sims": self.config.sims, "temp_plies": self.config.temp_plies,
                   "c_puct": self.config.c_puct, "limit": self.config.limit,
                   "max_plies": self.config.max_plies,
                   "baseline_depth": self.config.baseline_depth,
                   "material_weight": self.config.material_weight,
                   "draw_no_moves_value": self.config.draw_no_moves_value,
                   "leaf_depth": self.config.leaf_depth,
                   "spar_opponents": [item.strip() for item in
                                      self.config.spar_opponents.split(",") if item.strip()],
                   "bootstrap_opponent": self.config.bootstrap_opponent,
                   "search_kind": self.config.search_kind,
                   "ab_depth": self.config.ab_depth,
                   "net_scale": self.config.net_scale}
        if seeds is None:
            seeds = [self.rng.randrange(1 << 30) for _ in range(games)]
        weights = network.to_json()
        if kind == "az" and self.config.spar_fraction > 0:
            spar = int(round(games * min(1.0, max(0.0, self.config.spar_fraction))))
            kinds = ["spar"] * spar + ["az"] * (games - spar)
        else:
            kinds = [kind] * games
        tasks = [(task_kind, weights, seed, options)
                 for task_kind, seed in zip(kinds, seeds)]
        started = time.time()
        if self.config.workers > 1:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.config.workers)
            results = list(self.pool.map(_worker_play, tasks, chunksize=1))
        else:
            results = [_worker_play(task) for task in tasks]
        elapsed = time.time() - started
        total_plies = sum(item["plies"] for item in results)
        for index, item in enumerate(results):
            if self._record_saved < self.config.keep_records and item.get("moves"):
                self._save_record(item, index)
        label = {"baseline": "基线", "spar": "陪练"}.get(kind, "网络")
        tally: Dict[str, int] = {}
        for item in results:
            if item.get("opponent"):
                tally[item["opponent"]] = tally.get(item["opponent"], 0) + 1
        extra = ("，对手 " + str(tally)) if tally else ""
        self.log("  %s 对弈 %d 局：%d 步，用时 %.1fs（%.1fs/局）%s%s"
                 % (label, len(results), total_plies,
                    elapsed, elapsed / max(1, len(results)), self._result_brief(results), extra))
        return results

    def _result_brief(self, results: Sequence[Dict]) -> str:
        tally: Dict[str, int] = {}
        for item in results:
            tally[item["result"]] = tally.get(item["result"], 0) + 1
        reasons: Dict[str, int] = {}
        for item in results:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        return "结果 %s，结束方式 %s" % (tally, reasons)

    def _save_record(self, item: Dict, index: int) -> None:
        moves = item.get("moves") or []
        if not moves:
            return
        from sqrtchess import Recorder, Game, default_headers

        game = Game.new(no_capture_limit=self.config.limit)
        recorder = Recorder(game, default_headers(white="AlphaZero", black="AlphaZero",
                                                  event="AlphaZero 自我对弈"))
        for token in moves:
            if game.is_over:
                break
            recorder.push(token)
        recorder.finish()
        head = moves[0][:12]
        path = self.out_dir / "games" / ("iter-%02d-%s.sqg" % (self._record_saved + 1, head))
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(recorder.to_text())
        self._record_saved += 1

    # ------------------------------------------------------------- 训练

    def sample_to_tensors(self, sample: Sample):
        """把一条样本转成 (特征, 着法列表, 策略目标, 价值目标)。"""
        from sqrtchess import format_move_text

        game = Game.from_fen(sample.fen, no_capture_limit=self.config.limit)
        moves = game.legal_moves()
        if not moves:
            return None
        index = {}
        for position, move in enumerate(moves):
            value = game.cells[move.frm]
            target = game.cells[move.to]
            captured = target != 0 and (target > 0) != (value > 0)
            index[format_move_text(move.frm, move.to, captured, abs(value))] = position
        pi = np.zeros(len(moves), dtype=np.float32)
        for text, probability in sample.policy:
            slot = index.get(text)
            if slot is not None:
                pi[slot] += float(probability)
        total = float(pi.sum())
        if total <= 0:
            pi[:] = 1.0 / len(moves)
        else:
            pi /= total
        return encode(game), moves, pi, float(sample.z)

    def train_network(self, network: AZNetwork, lr: float) -> Dict[str, float]:
        """在经验池上做若干步梯度下降。"""
        if not self.buffer:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "steps": 0}
        policy_loss = value_loss = 0.0
        steps = 0
        epochs = 0
        started = time.time()
        target_steps = max(1, int(self.config.train_steps))
        batch_size = max(1, int(self.config.batch_size))
        # 按「打乱后的经验池顺序」成批训练：每条样本每个 epoch 只做一次特征编码，
        # 比每次随机抽样的重复编码快得多，也更接近「一轮数据走一遍」的标准做法。
        while steps < target_steps:
            order = list(range(len(self.buffer)))
            self.rng.shuffle(order)
            epochs += 1
            for start in range(0, len(order), batch_size):
                if steps >= target_steps:
                    break
                batch = []
                for index in order[start:start + batch_size]:
                    tensors = self.sample_to_tensors(self.buffer[index])
                    if tensors is not None:
                        batch.append(tensors)
                if not batch:
                    continue
                metrics = network.train_batch(batch, lr=lr)
                policy_loss += metrics["policy_loss"]
                value_loss += metrics["value_loss"]
                steps += 1
        if steps == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "steps": 0}
        return {"loss": (policy_loss + value_loss) / steps,
                "policy_loss": policy_loss / steps,
                "value_loss": value_loss / steps,
                "steps": steps, "epochs": epochs,
                "samples_seen": steps * batch_size,
                "seconds": time.time() - started}

    # ------------------------------------------------------------- 评估

    def net_player(self, network: AZNetwork, seed: int):
        """按 search_kind 造一个「网络选手」。"""
        if self.config.search_kind == "hybrid":
            from .hybrid import HybridPlayer

            return HybridPlayer(network, depth=self.config.ab_depth + 1,
                                net_scale=self.config.net_scale, seed=seed,
                                time_limit=self.config.arena_time_limit)
        return MCTSPlayer(network, sims=self.config.arena_sims, seed=seed,
                          material_weight=self.config.material_weight,
                          leaf_depth=self.config.leaf_depth)

    def arena(self, player_a, player_b, games: int, sims: int, tag: str) -> MatchResult:
        if games <= 0:
            return MatchResult()
        result = play_match(player_a, player_b, games=games,
                            max_plies=self.config.arena_max_plies,
                            limit=self.config.limit,
                            seed=self.rng.randrange(1 << 30))
        self.log("  %s：%s" % (tag, result.summary()))
        return result

    # ------------------------------------------------------------- 主循环

    def run(self) -> Dict:
        config = self.config
        self.log("=" * 78)
        self.log("AlphaZero 训练开始：%d 轮迭代，每轮 %d 局自我对弈，%d 次模拟，%d 个进程"
                 % (config.iterations, config.games_per_iter, config.sims, config.workers))
        self.log("网络 %d 通道，参数 %d；经验池上限 %d 条；输出目录 %s"
                 % (config.channels, AZNetwork(config.channels).parameter_count,
                    config.buffer_samples, self.out_dir))
        with open(self.out_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, ensure_ascii=False, indent=2)

        started = time.time()
        best = AZNetwork(config.channels, config.policy_hidden, config.value_hidden,
                         seed=config.seed)
        best_path = self.out_dir / "best.json"
        if best_path.exists():
            best = AZNetwork.load(best_path)
            self.log("已从 %s 继续训练" % best_path)
        elif config.init_model and Path(config.init_model).exists():
            best = AZNetwork.load(config.init_model)
            best.save(best_path)
            self.log("以已有网络 %s 为起点开始训练" % config.init_model)
        baseline_player = load_player(config.anchor_opponent,
                                            depth=config.baseline_depth, seed=config.seed)

        # 0) 可选的基线模仿学习：让网络一开始就懂「尽可能多吃子」
        if config.bootstrap_games > 0 and not best_path.exists():
            self.log("阶段 0：用基线对手 %s 生成 %d 局模仿数据"
                     % (config.bootstrap_opponent, config.bootstrap_games))
            results = self.self_play(best, config.bootstrap_games, kind="baseline")
            self._extend_buffer(results)
            metrics = self.train_network(best, lr=config.lr)
            self.log("  模仿学习完成：loss %.4f（策略 %.4f 价值 %.4f），经验池 %d 条"
                     % (metrics["loss"], metrics["policy_loss"], metrics["value_loss"],
                        len(self.buffer)))
            best.save(best_path)

        # 持续训练：平时用 trainer_net 自我对弈并继续训练，只有门控通过才更新 gated best
        trainer_net = best
        for iteration in range(1, config.iterations + 1):
            lr = config.lr * (config.lr_decay ** (iteration - 1))
            self.log("-" * 78)
            self.log("第 %d / %d 轮迭代（学习率 %.5f）" % (iteration, config.iterations, lr))
            round_started = time.time()

            # 1) 自我对弈（用当前训练网络）
            results = self.self_play(trainer_net, config.games_per_iter, kind="az")
            added = self._extend_buffer(results)

            # 2) 继续训练（候选承接上一轮的权重，不浪费任何一轮训练）
            candidate = trainer_net.copy()
            metrics = self.train_network(candidate, lr=lr)
            trainer_net = candidate
            self.log("  训练：%d 步，loss %.4f（策略 %.4f 价值 %.4f），用时 %.1fs，经验池 %d 条"
                     % (metrics.get("steps", 0), metrics["loss"], metrics["policy_loss"],
                        metrics["value_loss"], metrics.get("seconds", 0.0), len(self.buffer)))

            # 3) 门控评估：候选 vs 当前 best（可以按 eval_every 跳过以省时间）
            candidate_path = self.out_dir / "candidate.json"
            candidate.save(candidate_path)
            do_eval = (config.eval_every <= 1 or iteration % config.eval_every == 0
                       or iteration == config.iterations)
            gate = MatchResult()
            anchor = MatchResult()
            others: Dict[str, Dict] = {}
            promoted = False
            if do_eval:
                gate = self.arena(self.net_player(candidate, iteration),
                                  self.net_player(best, iteration + 1000),
                                  config.arena_games, config.arena_sims,
                                  "门控（候选 vs best）")
                promoted = gate.games > 0 and gate.score_a >= config.promote_threshold
                if promoted:
                    best = candidate
                    best.save(best_path)
                    self.log("  → 候选网络被采纳为新的 best（得分率 %.3f ≥ %.2f）"
                             % (gate.score_a, config.promote_threshold))
                else:
                    self.log("  → 候选未达阈值（得分率 %.3f < %.2f），best 保持不变"
                             % (gate.score_a, config.promote_threshold))

                # 4) 与固定基线对战：贯穿训练的进度标尺
                anchor = self.arena(self.net_player(best, iteration + 7),
                                    baseline_player, config.baseline_games,
                                    config.arena_sims,
                                    "锚点（best vs %s）" % config.anchor_opponent)

                # 多对手锚点：看它是不是只会对付某一种棋风
                if config.anchor_others_games > 0:
                    for spec in ("capture", "random", "greedy", "search2"):
                        opponent = load_player(spec, seed=iteration + 31)
                        if opponent is None:
                            continue
                        outcome = self.arena(self.net_player(best, iteration + 17),
                                             opponent, config.anchor_others_games,
                                             config.arena_sims,
                                             "锚点（best vs %s）" % spec)
                        others[spec] = outcome.to_dict()
            else:
                self.log("  （本轮跳过评估，eval_every=%d）" % config.eval_every)

            trainer_net.save(self.out_dir / ("net-iter%03d.json" % iteration))
            if not do_eval:
                # 没有门控的轮次也要留一个可用的权重文件，方便随时接手
                trainer_net.save(self.out_dir / "latest.json")
            record = {
                "iteration": iteration, "lr": round(lr, 6),
                "selfplay": {"games": len(results), "samples_added": added,
                             "avg_plies": round(sum(r["plies"] for r in results) / max(1, len(results)), 1),
                             "seconds": round(sum(r["seconds"] for r in results), 1)},
                "train": {key: (round(value, 5) if isinstance(value, float) else value)
                          for key, value in metrics.items()},
                "gate": gate.to_dict(), "promoted": promoted,
                "anchor_baseline": anchor.to_dict(),
                "anchor_opponent": config.anchor_opponent,
                "anchor_others": others,
                "buffer": len(self.buffer),
                "evaluated": do_eval,
                "round_seconds": round(time.time() - round_started, 1),
                "total_seconds": round(time.time() - started, 1),
            }
            self.history.append(record)
            self.write_metrics(record)
            self.log("  本轮用时 %.1fs，累计 %.1fs；best 对吃子基线的得分率 %.3f"
                     % (record["round_seconds"], record["total_seconds"], anchor.score_a))

        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        summary = {"iterations": config.iterations, "best": str(best_path),
                   "buffer": len(self.buffer), "seconds": round(time.time() - started, 1),
                   "history": self.history}
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        self.log("训练结束：best 权重 %s，累计用时 %.1f 分钟"
                 % (best_path, (time.time() - started) / 60.0))
        return summary

    def _extend_buffer(self, results: Sequence[Dict]) -> int:
        added = 0
        for item in results:
            for raw in item["samples"]:
                self.buffer.append(Sample.from_dict(raw))
                added += 1
        if len(self.buffer) > self.config.buffer_samples:
            del self.buffer[:len(self.buffer) - self.config.buffer_samples]
        return added
