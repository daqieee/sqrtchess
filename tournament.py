"""多配置「斗蛐蛐」锦标赛：几个外部 AI 两两对战，各配置并行跑，最后汇总比分。

用法（在包根目录）：

    python tournament.py                       # 默认 4 组配置，每组 6 局
    python tournament.py --games 10 --time 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: 内置的三个外部 AI
AI_FILES = {
    "claude": ROOT / "ai-claude" / "strong_ai.py",
    "deepseek": ROOT / "ai-deepseek" / "my_work.py",
    "gpt": ROOT / "ai-gpt" / "my_work.py",
}

#: 名字 -> 显示名
LABELS = {"claude": "Claude", "deepseek": "DeepSeek", "gpt": "GPT"}


def command_for(name: str) -> str:
    path = AI_FILES[name]
    exe = sys.executable or "python"
    if " " in exe:
        exe = '"%s"' % exe
    return '%s -B "%s"' % (exe, path)


def run_config(config, games: int, max_plies: int, quiet: bool):
    name, white, black, time_ms = config
    label = "%s vs %s @%dms" % (LABELS[white], LABELS[black], time_ms)
    started = time.time()
    cmd = [sys.executable, str(ROOT / "duel.py"),
           "--white", command_for(white), "--black", command_for(black),
           "--games", str(games), "--time-ms", str(time_ms),
           "--max-plies", str(max_plies), "--out", str(ROOT / ("duel-%s.sqg" % name))]
    if quiet:
        cmd.append("--quiet")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(ROOT), env=env, timeout=12 * 3600)
    return {
        "name": name, "label": label, "time_ms": time_ms,
        "white": LABELS[white], "black": LABELS[black],
        "seconds": round(time.time() - started, 1),
        "returncode": proc.returncode,
        "output": (proc.stdout or "") + (proc.stderr or ""),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="多配置斗蛐蛐锦标赛")
    parser.add_argument("--games", type=int, default=6, help="每组局数（默认 6）")
    parser.add_argument("--time", type=int, nargs="+", default=[1000, 2000],
                        help="每步时限（毫秒），可以给多个（默认 1000 2000）")
    parser.add_argument("--pairs", nargs="+", default=["claude:deepseek", "claude:gpt"],
                        help="对战组合，例如 claude:deepseek claude:gpt")
    parser.add_argument("--max-plies", type=int, default=300, help="单局步数上限")
    parser.add_argument("--workers", type=int, default=4, help="同时跑几组")
    parser.add_argument("--quiet", action="store_true", help="只输出比分")
    args = parser.parse_args(argv)

    configs = []
    for pair in args.pairs:
        left, right = pair.split(":")
        for ms in args.time:
            configs.append(("%s-vs-%s-%dms" % (left, right, ms), left, right, ms))

    print("共 %d 组配置，每组 %d 局，合计 %d 局；并行 %d 组"
          % (len(configs), args.games, len(configs) * args.games, args.workers))
    for name, white, black, ms in configs:
        print("  - %s vs %s，每步 %d ms" % (LABELS[white], LABELS[black], ms))
    print()

    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_config, cfg, args.games, args.max_plies, args.quiet)
                   for cfg in configs]
        for future in futures:
            item = future.result()
            results.append(item)
            print("=" * 70)
            print(item["label"], "（用时 %.1f 分钟）" % (item["seconds"] / 60.0))
            print(item["output"].strip())

    summary = {"games_per_config": args.games, "total_games": len(configs) * args.games,
               "seconds": round(time.time() - started, 1),
               "configs": [{k: v for k, v in item.items() if k != "output"}
                           for item in results]}
    with open(ROOT / "tournament.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print()
    print("汇总已写入 tournament.json，总用时 %.1f 分钟" % ((time.time() - started) / 60.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
