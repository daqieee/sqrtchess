# 项目目录与本地迁移

统一工作目录为 `D:\project\sqrtchess`，对应 GitHub 仓库 `daqieee/sqrtchess`。

| 来源 | 主仓库位置 |
| --- | --- |
| 原 chess 的 AlphaZero 代码 | `alphazero/` |
| 原 chess 的全部模型、训练快照和结果 | `runs/`（保留原相对路径） |
| 原 chess 的 Alpha-Beta、CLI 扩展 | `sqrtchess/` |
| 原 chess 的 AlphaZero / Alpha-Beta 测试 | `tests/` |
| 原 chess 的训练文档 | `docs/ALPHAZERO.md`、`ALPHABETA.md`、`HYBRID.md` |
| 原 chess 的个人棋谱 | `examples/matches/personal/` |
| GPT 的 AI、测试、评测脚本与历史结果 | `ai-gpt/` |
| DeepSeek / Claude 的 AI | `ai-deepseek/`、`ai-claude/` |
| 双外部 AI 对战、棋盘缩放与网络选择 | 统一到 `gui/sqrtchess_gui.py` |

重复规则引擎不再作为独立活动项目维护。原始目录收纳到 `.local/legacy/`，保留未采用的旧界面、说明及缓存；
该目录仅用于本机追溯，不上传 GitHub。`.local/migration-manifest.json` 记录原文件的 SHA-256 与归档路径。
本机备份中的老路径与命令可能失效，日常开发、运行和训练均应使用主仓库。

新训练输出放在 `runs/local/`；临时文件与其他本地记录放在 `.local/`。
