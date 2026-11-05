# 历史对局记录

此目录保存整理项目前已有的五份 `.sqg` 棋谱及锦标赛结果。

- `tournament.json`：四组配置、每组 6 局的运行汇总。
- `tournament-output.txt`：对应的完整控制台输出。
- `duel-*.sqg`：可通过 GUI 载入，或用 `python -m sqrtchess replay examples/matches/<文件名>.sqg --list` 回放。

这些是历史样本，时限与机器负载会影响 AI 结果；部分对局触及脚本步数上限，详见原始记录。
新的锦标赛输出仍生成在项目根目录，不会覆盖这里的归档。
