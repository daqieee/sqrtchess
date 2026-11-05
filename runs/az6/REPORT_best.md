# 模型评估报告：runs/az6/best.json

生成时间：2026-10-10 23:16:22
搜索：hybrid（ab 深度 2，限时 0.2，sims 100），每对手 12 局，上限 220 步

| 对手 | 得分率 | 胜 | 负 | 和 | 平均大小差 | 平均步数 | 用时(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ab2 | **0.458** | 2 | 3 | 7 | +5.2 | 206 | 188 |
| ab3 | **0.208** | 0 | 7 | 5 | -18.3 | 202 | 433 |
| capture | **0.833** | 8 | 0 | 4 | +22.5 | 199 | 156 |
| search2 | **0.542** | 4 | 3 | 5 | +0.8 | 205 | 543 |
| random | **0.958** | 11 | 0 | 1 | +91.3 | 203 | 136 |

结束方式分布：
* ab2：{'truncated-adjudicated': 7, 'no-capture-limit': 5}
* ab3：{'truncated-adjudicated': 8, 'no-capture-limit': 4}
* capture：{'truncated-adjudicated': 8, 'no-capture-limit': 4}
* search2：{'truncated-adjudicated': 10, 'no-capture-limit': 2}
* random：{'truncated-adjudicated': 8, 'all-captured': 2, 'no-moves': 1, 'no-capture-limit': 1}

总用时 24.3 分钟
