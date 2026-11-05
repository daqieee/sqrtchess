"""AlphaZero 命令行：训练、自我对弈、评估、人机对战。

    python -m alphazero train --iterations 8 --games 40 --sims 60 --workers 8 --out runs/az1
    python -m alphazero selfplay --games 4 --sims 60 --out runs/sp
    python -m alphazero arena --a runs/az1/best.json --b baseline --games 20
    python -m alphazero play --model runs/az1/best.json --sims 120
    python -m alphazero eval --model runs/az1/best.json --fen "startpos"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from sqrtchess import (DEFAULT_NO_CAPTURE_LIMIT, Game, START_FEN, WHITE, BLACK,
                       color_name, parse_square, replay, square_name)

from .arena import play_match
from .encoding import encode
from .mcts import MCTS
from .network import AZNetwork
from .players import PLAYER_HELP, AlphaZeroPlayer, CapturePlayer, MCTSPlayer, load_player
from .selfplay import play_game
from .trainer import TrainConfig, Trainer


def _limit(text: str) -> Optional[int]:
    value = str(text).strip().lower()
    if value in ("off", "none", "0", "不限"):
        return None
    return int(value)


# --------------------------------------------------------------------------- 子命令


def cmd_train(args) -> int:
    config = TrainConfig(
        iterations=args.iterations, games_per_iter=args.games, sims=args.sims,
        temp_plies=args.temp_plies, max_plies=args.max_plies, c_puct=args.c_puct,
        limit=args.limit, channels=args.channels, policy_hidden=args.hidden,
        value_hidden=args.hidden, batch_size=args.batch_size, lr=args.lr,
        lr_decay=args.lr_decay, train_steps=args.train_steps,
        buffer_samples=args.buffer, arena_games=args.arena_games,
        arena_sims=args.arena_sims, arena_max_plies=args.arena_max_plies,
        promote_threshold=args.promote_threshold,
        baseline_games=args.baseline_games, baseline_depth=args.baseline_depth,
        spar_fraction=args.spar_fraction, material_weight=args.material_weight,
        spar_opponents=args.spar_opponents,
        draw_no_moves_value=args.draw_no_moves_value, leaf_depth=args.leaf_depth,
        anchor_others_games=args.anchor_others_games, init_model=args.init_model,
        anchor_opponent=args.anchor_opponent, bootstrap_opponent=args.bootstrap_opponent,
        search_kind=args.search_kind, ab_depth=args.ab_depth, net_scale=args.net_scale,
        arena_time_limit=args.arena_time_limit, eval_every=args.eval_every,
        bootstrap_games=args.bootstrap_games, workers=args.workers, seed=args.seed,
        out_dir=args.out, keep_records=args.keep_records)
    trainer = Trainer(config)
    summary = trainer.run()
    print("best 权重：%s" % summary["best"])
    return 0


def cmd_selfplay(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model:
        network = AZNetwork.load(args.model)
        print("使用网络：%s" % args.model)
    else:
        network = AZNetwork(args.channels, seed=args.seed)
        print("使用随机初始化的网络（%d 通道）" % args.channels)
    samples = []
    started = time.time()
    for index in range(args.games):
        data = play_game(network, sims=args.sims, max_plies=args.max_plies,
                         temp_plies=args.temp_plies, seed=args.seed + index,
                         limit=args.limit, keep_record=args.records)
        samples.extend(sample.to_dict() for sample in data.samples)
        if args.records:
            path = out_dir / ("game-%03d.sqg" % (index + 1))
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(data.record)
        print("第 %d 局：%d 步，结果 %s（%s），用时 %.1fs，样本 %d"
              % (index + 1, data.plies, data.result, data.reason, data.seconds,
                 len(data.samples)))
    data_path = out_dir / "samples.jsonl"
    with open(data_path, "w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("共 %d 条样本 -> %s；总用时 %.1fs" % (len(samples), data_path, time.time() - started))
    return 0


def cmd_arena(args) -> int:
    player_a = load_player(args.a, sims=args.sims, seed=args.seed,
                           material_weight=args.material_weight, leaf_depth=args.leaf_depth)
    player_b = load_player(args.b, sims=args.sims, seed=args.seed + 1,
                           material_weight=args.material_weight, leaf_depth=args.leaf_depth)
    print("A = %s\nB = %s" % (args.a, args.b))
    result = play_match(player_a, player_b, games=args.games, max_plies=args.max_plies,
                        limit=args.limit, seed=args.seed)
    print(result.summary())
    print("结束方式分布：%s" % result.reasons)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0


def cmd_eval(args) -> int:
    network = AZNetwork.load(args.model)
    game = Game.from_fen(args.fen or START_FEN, no_capture_limit=args.limit)
    print(game.board_text())
    moves = game.legal_moves()
    priors, value = network.evaluate(game)
    print("价值评估（%s视角）：%+.3f" % (color_name(game.side), value))
    ranked = sorted(zip(moves, priors), key=lambda item: -item[1])
    print("策略先验前 %d：" % args.top)
    for move, probability in ranked[:args.top]:
        print("  %-10s %.3f  %s" % (game.move_dict(move)["notation"], probability,
                                    game.move_dict(move).get("describe", "")))
    if args.search > 0:
        search = MCTS(network, sims=args.search, seed=args.seed)
        policy, root = search.policy(game, temperature=0.0)
        best = max(policy.items(), key=lambda item: item[1])
        print("搜索 %d 次后建议：%s（访问 %d，Q=%+.3f）"
              % (args.search, game.move_dict(best[0])["notation"],
                 root.children[best[0]].N, root.children[best[0]].Q))
    return 0


def cmd_play(args) -> int:
    network = AZNetwork.load(args.model)
    game = Game.new(no_capture_limit=args.limit)
    search = MCTS(network, sims=args.sims, seed=args.seed)
    human = args.human
    print("根号棋：AlphaZero（%s）对局。你执%s。" % (args.model, "白" if human == "white" else
                                              ("黑" if human == "black" else "无（观战）")))
    print("输入着法（如 b2-b3、1b2xc3），命令：board / hint / undo / quit")
    while not game.is_over:
        print()
        print(game.board_text())
        white_count, white_size = game.piece_stats(WHITE)
        black_count, black_size = game.piece_stats(BLACK)
        print("轮到 %s｜白 %d 子/%d｜黑 %d 子/%d｜无吃子 %d/%s"
              % (color_name(game.side), white_count, white_size, black_count, black_size,
                 game.no_capture, game.no_capture_limit))
        side_is_human = ((game.side == WHITE) == (human == "white")) and human != "none"
        if not side_is_human:
            started = time.time()
            policy, root = search.policy(game, temperature=args.temperature)
            if not policy:
                break
            move = max(policy.items(), key=lambda item: item[1])[0]
            info = game.move_dict(move)
            print("AlphaZero 走：%s（价值 %+.3f，用时 %.1fs）"
                  % (info["notation"], root.Q, time.time() - started))
            game.push(move)
            continue
        try:
            text = input("你的着法> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("quit", "exit", "q"):
            break
        if text == "board":
            continue
        if text == "undo":
            if game.ply >= 2:
                game.pop()
                game.pop()
            continue
        if text == "hint":
            policy, root = search.policy(game, temperature=0.0)
            best = max(policy.items(), key=lambda item: item[1])
            print("建议：%s（价值 %+.3f）" % (game.move_dict(best[0])["notation"], root.Q))
            continue
        try:
            info = game.push(text)
            print("你走了：%s" % info.describe())
        except Exception as exc:  # noqa: BLE001
            print("无法走这一步：%s" % exc)
    print()
    print(game.board_text())
    print("对局结束：%s" % (game.result.text() if game.result else "未结束"))
    return 0


def cmd_report(args) -> int:
    """对一个模型跑一轮多对手评估，并写成 Markdown 报告。"""
    import datetime

    from alphazero.players import load_player as _load

    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    rows = []
    started = time.time()
    for spec in opponents:
        try:
            player_a = _load(args.model, sims=args.sims, seed=args.seed,
                             search_kind=args.search_kind, ab_depth=args.ab_depth,
                             net_scale=args.net_scale, time_limit=args.time_limit)
            player_b = _load(spec, sims=args.sims, seed=args.seed + 1, depth=args.baseline_depth)
        except Exception as exc:  # noqa: BLE001
            print("跳过 %s：%s" % (spec, exc))
            continue
        result = play_match(player_a, player_b, games=args.games,
                            max_plies=args.max_plies, limit=args.limit,
                            seed=args.seed + len(rows))
        rows.append((spec, result))
        print("%-28s %s" % (spec, result.summary()))

    lines = ["# 模型评估报告：%s" % args.model, "",
             "生成时间：%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "搜索：%s（ab 深度 %d，限时 %s，sims %d），每对手 %d 局，上限 %d 步"
             % (args.search_kind, args.ab_depth, args.time_limit, args.sims, args.games,
                args.max_plies), "",
             "| 对手 | 得分率 | 胜 | 负 | 和 | 平均大小差 | 平均步数 | 用时(s) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for spec, result in rows:
        lines.append("| %s | **%.3f** | %d | %d | %d | %+.1f | %.0f | %.0f |"
                     % (spec, result.score_a, result.wins_a, result.wins_b, result.draws,
                        result.material_a, result.avg_plies, result.seconds))
    lines += ["", "结束方式分布："]
    for spec, result in rows:
        lines.append("* %s：%s" % (spec, result.reasons))
    lines += ["", "总用时 %.1f 分钟" % ((time.time() - started) / 60.0)]
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print("报告已写入：%s" % args.out)
    else:
        print(text)
    return 0


def cmd_info(args) -> int:
    network = AZNetwork.load(args.model)
    print("网络文件：%s" % args.model)
    print("通道数：%d，策略隐藏层：%d，价值隐藏层：%d"
          % (network.channels, network.policy_hidden, network.value_hidden))
    print("参数量：%d" % network.parameter_count)
    return 0


# --------------------------------------------------------------------------- 入口


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alphazero",
        description="根号棋 AlphaZero 自我对弈系统（简化版：PUCT + 策略/价值网络 + 自我对弈 + 门控迭代）")
    sub = parser.add_subparsers(dest="command", metavar="命令")

    def common(p):
        p.add_argument("--limit", type=_limit, default=DEFAULT_NO_CAPTURE_LIMIT,
                       help="无吃子步数上限（默认 100，off 表示不限制）")
        p.add_argument("--max-plies", type=int, default=400, help="单局最大步数")
        p.add_argument("--seed", type=int, default=1)
        return p

    p = common(sub.add_parser("train", help="训练：自我对弈 → 训练 → 门控评估 → 迭代"))
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--games", type=int, default=40, help="每轮自我对弈局数")
    p.add_argument("--sims", type=int, default=60, help="每步 MCTS 模拟次数")
    p.add_argument("--temp-plies", type=int, default=20, help="前多少步用温度采样")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--channels", type=int, default=24)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--lr-decay", type=float, default=0.9)
    p.add_argument("--train-steps", type=int, default=300, help="每轮梯度步数")
    p.add_argument("--buffer", type=int, default=20000, help="经验池样本上限")
    p.add_argument("--arena-games", type=int, default=8, help="门控对战局数（0 = 跳过）")
    p.add_argument("--arena-sims", type=int, default=30)
    p.add_argument("--arena-max-plies", type=int, default=240,
                   help="评估对局的步数上限（超过则按规则 7 判定）")
    p.add_argument("--promote-threshold", type=float, default=0.55)
    p.add_argument("--baseline-games", type=int, default=6, help="与吃子基线的锚点对战局数")
    p.add_argument("--baseline-depth", type=int, default=4)
    p.add_argument("--spar-fraction", type=float, default=0.3,
                   help="自我对弈里与其它算法陪练的比例（0 = 不陪练）")
    p.add_argument("--spar-opponents", default="baseline,random,greedy,search2",
                   help="陪练对手池，逗号分隔；按局轮换以免只学会对付一种棋风")
    p.add_argument("--material-weight", type=float, default=0.45,
                   help="搜索叶子价值中「物质项」的权重（0 = 纯网络价值，越大越看重棋子）")
    p.add_argument("--search", dest="search_kind", choices=["mcts", "hybrid"],
                   default="mcts",
                   help="自我对弈与评估用的搜索：mcts（PUCT）或 hybrid（神经引导 Alpha-Beta）")
    p.add_argument("--ab-depth", type=int, default=1,
                   help="hybrid 自我对弈用的 Alpha-Beta 深度（评估时自动 +1，并受时间上限约束）")
    p.add_argument("--net-scale", type=float, default=30.0,
                   help="hybrid 叶子评估里网络价值的权重（±1 价值 = 多少子力分）")
    p.add_argument("--arena-time-limit", type=float, default=0.35,
                   help="hybrid 评估时每步时间上限（秒）")
    p.add_argument("--eval-every", type=int, default=1,
                   help="每多少轮做一次门控与锚点评估（1 = 每轮都评）")
    p.add_argument("--leaf-depth", type=int, default=1,
                   help="MCTS 叶子评估前额外做几层吃子静态搜索（0 = 关闭，1~3 = 加深 1~3 层战术）")
    p.add_argument("--draw-no-moves-value", type=float, default=0.1,
                   help="「一方无法动子」判和时的分值（±该值）；100 步无吃子判和固定为 0")
    p.add_argument("--anchor-others-games", type=int, default=3,
                   help="每轮额外与随机/贪心/搜索2 各对战多少局（0 = 跳过）")
    p.add_argument("--anchor-opponent", default="ab2",
                   help="主锚点对手：abN（Alpha-Beta 深度 N）/ capture / random / greedy / search2")
    p.add_argument("--bootstrap-opponent", default="ab2",
                   help="模仿学习用的对手（同上写法）")
    p.add_argument("--init-model", default="",
                   help="从这个权重文件开始训练（输出目录没有 best.json 时生效）")
    p.add_argument("--bootstrap-games", type=int, default=20,
                   help="用吃子基线生成多少局模仿数据（0 = 不启用）")
    p.add_argument("--workers", type=int, default=1, help="自我对弈并行进程数")
    p.add_argument("--out", default="runs/az", help="输出目录")
    p.add_argument("--keep-records", type=int, default=4, help="保留多少局自我对弈棋谱")
    p.set_defaults(func=cmd_train)

    p = common(sub.add_parser("selfplay", help="用网络（或随机网络）自我对弈并导出样本"))
    p.add_argument("--model", help="网络权重文件；省略则用随机初始化网络")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--sims", type=int, default=60)
    p.add_argument("--temp-plies", type=int, default=20)
    p.add_argument("--channels", type=int, default=24)
    p.add_argument("--out", default="runs/selfplay")
    p.add_argument("--records", action="store_true", help="同时保存棋谱")
    p.set_defaults(func=cmd_selfplay)

    p = common(sub.add_parser("arena", help="两个选手对战评估"))
    p.add_argument("--a", default="baseline", help=PLAYER_HELP)
    p.add_argument("--b", default="random", help=PLAYER_HELP)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--sims", type=int, default=60, help="网络选手的模拟次数")
    p.add_argument("--leaf-depth", type=int, default=1, help="叶子吃子搜索层数")
    p.add_argument("--material-weight", type=float, default=0.45, help="物质项权重")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_arena)

    p = common(sub.add_parser("eval", help="查看网络对某个局面的策略与价值"))
    p.add_argument("--model", required=True)
    p.add_argument("--fen", default=START_FEN)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--search", type=int, default=0, help="额外做多少次 MCTS 模拟")
    p.set_defaults(func=cmd_eval)

    p = common(sub.add_parser("play", help="在终端与网络对战（或观战）"))
    p.add_argument("--model", required=True)
    p.add_argument("--sims", type=int, default=120)
    p.add_argument("--human", choices=["white", "black", "none"], default="white")
    p.add_argument("--temperature", type=float, default=0.0)
    p.set_defaults(func=cmd_play)

    p = common(sub.add_parser("report", help="对一个模型跑多对手评估并写报告"))
    p.add_argument("--model", required=True)
    p.add_argument("--opponents", default="ab2,ab3,ab4,capture,random,greedy,search2")
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--search", dest="search_kind", choices=["mcts", "hybrid"], default="hybrid")
    p.add_argument("--ab-depth", type=int, default=2)
    p.add_argument("--net-scale", type=float, default=30.0)
    p.add_argument("--time-limit", type=float, default=0.2)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--baseline-depth", type=int, default=4)
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("info", help="查看网络文件信息")
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
