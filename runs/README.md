# AlphaZero 模型与训练记录

本目录完整收录从原 `chess/runs/` 迁入的训练产物。权重以 JSON 保存，无需 Git LFS。

| 目录 | 内容 |
| --- | --- |
| `archive/az1` ～ `archive/az4` | 前四代模型、配置、迭代快照、指标、训练日志及棋谱 |
| `az5/` | Alpha-Beta 陪练阶段模型及训练记录 |
| `az6/` | 60 轮混合搜索训练；包含 `best.json`、`best_hybrid.json`、`latest.json` 和评估报告 |

`best.json` 为历史门控选择的权重；`latest.json` 与各轮快照不必相同。
本次迁移保留原文件，不重命名或重新挑选权重。历史指标受对手、时间预算、步数截断及机器负载影响。

```sh
python -m alphazero info --model runs/az6/best.json
python -m alphazero eval --model runs/az6/best.json --fen startpos --search 12
```

GUI 可选 MCTS 或混合搜索；az6 的历史报告使用混合搜索，见 [HYBRID.md](../docs/HYBRID.md)。
新实验请输出到 `runs/local/<实验名>/`（已忽略），避免覆盖这些归档。
