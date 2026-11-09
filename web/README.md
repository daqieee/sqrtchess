# GitHub Pages 人机对战

纯静态页面，访客可选先手（白）、后手（黑）或随机，对战浏览器内启发式 AI。无需后端、登录、CDN 或构建依赖。

- `engine.js` 按 `sqrtchess/core.py` 移植规则，按 `sqrtchess/ai.py` 的 `GreedyAI` 移植评分及随机扰动。不是 Alpha-Beta 或 AlphaZero。
- `app.js` 管理选子、AI 应手、整回合悔棋、新局确认、规则和对局记录。
- 所有资源使用相对地址，适配 `/sqrtchess/` 项目子路径。

本地运行：在仓库根目录执行 `python -m http.server 8000 --directory web`，访问 `http://localhost:8000`。
测试：`node --test web/*.test.mjs`。规则基线 perft 1–4：22、484、12,342、314,721。

## 部署（仅在设计审阅通过后）

1. 提交 `web/` 和 `.github/workflows/pages.yml` 到 main。
2. 仓库 Settings → Pages → Source 选择 GitHub Actions。
3. 运行 `Deploy SqrtChess Pages` 或在 main 有相关改动时自动触发。
4. 检查 workflow 成功及返回的 Pages 地址，实际访问并试走一回合。

当前对局仅保存在页面内存中，刷新即重开。默认连续 100 半回合无吃子触发判定。仅保留访客对战 AI，无模型下载、外部程序、AI 自对弈或联机功能。
