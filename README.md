# 根号棋（SqrtChess）

根号棋的**规则引擎 + 局面文本（SQFEN）+ 棋谱（.sqg）+ 机器接口**，纯 Python 标准库实现，
不依赖任何第三方包（Python 3.8+）。

## 下载与启动

```sh
git clone https://github.com/daqieee/sqrtchess.git
cd sqrtchess
python gui/sqrtchess_gui.py
```

需要 Python 3.8+。图形界面使用 Tkinter，Windows 官方 Python 安装时需包含 Tcl/Tk；
Linux 如缺少 Tkinter，请通过系统包管理器安装对应的 Python Tk 包。
Windows 也可双击 `start_gui.bat`。命令行引擎不需要图形环境。

## 这个目录里有什么

| 部分 | 位置 | 说明 |
| --- | --- | --- |
| 规则引擎 | `sqrtchess/core.py` | 棋盘、走法、吃子、胜负与平局判定 |
| 局面文本 | `sqrtchess/fen.py` | SQFEN 读写，见 [docs/NOTATION.md](docs/NOTATION.md) |
| 棋谱与回放 | `sqrtchess/record.py` | `.sqg` 文本棋谱、Recorder、replay |
| 机器接口 | `sqrtchess/protocol.py` | JSON 行控制协议、AI 应答协议、TCP 服务，见 [docs/PROTOCOL.md](docs/PROTOCOL.md) |
| 命令行 | `sqrtchess/cli.py` | `python -m sqrtchess` |
| 内置 AI | `sqrtchess/ai.py` | 随机 / 贪心 / 浅层搜索 / 吃子基线（够用即可，不是本项目的重点） |
| 规则文档 | [docs/RULES.md](docs/RULES.md) | 规则原文 + 实现口径 + 边界情况 |
| 示例 | [examples/](examples/) | Node.js 客户端、自定义 AI 示例、示例棋谱 |
| **图形界面** | [gui/sqrtchess_gui.py](gui/sqrtchess_gui.py) | 对局、观战、悔棋、载入棋谱、写入 FEN 查看局面；可让两路外部 AI 对打 |
| **外部 AI 对手** | [ai-claude/](ai-claude/) · [ai-deepseek/](ai-deepseek/) · [ai-gpt/](ai-gpt/) | 三个纯标准库实现的外部 AI（斗蛐蛐用） |
| **对战脚本** | [duel.py](duel.py) | 命令行批量对战并统计比分 |
| 锦标赛 | [tournament.py](tournament.py) | 多组合、多时限并行对战 |
| 历史对局 | [examples/matches/](examples/matches/) | 已归档棋谱、锦标赛汇总与输出 |
| 启动脚本 | [start_gui.bat](start_gui.bat) | Windows 双击启动界面 |
| 测试 | [tests/test_sqrtchess.py](tests/test_sqrtchess.py) | 79 项单元测试 |

## 安装（可选）

从本目录直接运行不需要安装。如果要把引擎当库用，可以：

~~~bat
pip install -e .
sqrtchess fen          :: 安装后会多一个命令行入口
~~~

（构建依赖是 setuptools + wheel，正常 pip 会自动安装。）

## 斗蛐蛐：让两个外部 AI 对打

包里预装了三个外部 AI（都是纯标准库、各自实现 AI 应答协议）：

| 目录 | 名称 | 启动方式 |
| --- | --- | --- |
| [ai-deepseek/](ai-deepseek/) | DeepSeek | `python -B ai-deepseek/my_work.py` |
| [ai-gpt/](ai-gpt/) | GPT（RootKnight） | `python -B ai-gpt/my_work.py` |
| [ai-claude/](ai-claude/) | Claude（strong_ai） | `python -B ai-claude/strong_ai.py` |

**图形界面**：打开“对局设置”，白方/黑方的“外部程序”已预填好命令，下面还有「快速填充」按钮
（Claude vs DeepSeek / Claude vs GPT / DeepSeek vs GPT）一键切换；外部 AI 每步时限默认 300 ms。
直接点主界面的 **「斗蛐蛐（多局 AI 对战）」** 按钮或菜单“对局 → 斗蛐蛐”，输入局数即可。
每局结束自动交换先后手，日志里逐局记录胜负，最后给出总比分。

**命令行**（更快，适合跑几十局）：

~~~bat
python duel.py                                  :: 默认 DeepSeek vs GPT，共 4 局
python duel.py --games 10 --time-ms 200         :: 10 局，每步 200 毫秒
python duel.py --white "python -B ai-gpt/my_work.py" --black "python -m sqrtchess ai --level search2"
python duel.py --out 对局.sqg                    :: 顺便存棋谱

:: 多配置锦标赛：几个 AI 两两对战，各配置并行跑，最后汇总（默认 Claude vs DeepSeek / GPT，1s 与 2s 各 6 局）
python tournament.py --games 6 --time 1000 2000 --pairs claude:deepseek claude:gpt
~~~

只要程序实现 AI 应答协议，任何语言写的 AI 都能接入：

~~~text
请求  {"request": "move", "fen": "<SQFEN>", "time_ms": 300, "level": "greedy"}
响应  {"ok": true, "move": "f2-f3", "notation": "1f2-f3"}
~~~

## 图形界面

~~~bat
start_gui.bat          :: 双击也可以
python gui/sqrtchess_gui.py
~~~

界面里可以把白方/黑方设为：人类 / 吃子基线 AI / 贪心 AI / 随机 AI / 搜索 2 层 / 外部程序，
双方都设 AI 即为实时观战；粘贴 FEN 可查看任意局面；可载入/保存 `.sqg` 棋谱。

**棋盘大小**会自动按屏幕挑一个尽量大的尺寸，也可以手动调整：拖动“棋盘大小”滑块、
点 ＋/－ 按钮，或在棋盘上 Ctrl+滚轮、Ctrl+加号/减号（60% ~ 150%）。

## 快速开始

~~~bat
python -m sqrtchess fen                      :: 初始局面 FEN
python -m sqrtchess board                    :: 文本棋盘
python -m sqrtchess moves                    :: 列出合法着法
python -m sqrtchess demo --white greedy --black random --out 对局.sqg
python -m sqrtchess replay 对局.sqg --list    :: 回放并校验棋谱
python -m sqrtchess perft --depth 3          :: 走法生成计数（12,342）
python -m sqrtchess selfcheck                :: 内置自检
python -m sqrtchess proto                    :: 控制协议（JSON 行）
python -m sqrtchess ai --level greedy        :: AI 应答协议
python -m sqrtchess serve --port 8765        :: TCP 服务
~~~

## 在 Python 里使用

~~~python
from sqrtchess import Game, START_FEN, Recorder, replay

game = Game.new()                     # 也可以用 Game.from_fen(fen)
print(game.fen())                     # 局面文本
print(len(game.legal_moves()), "种走法")
print(game.board_text())              # 文本棋盘

info = game.push("1b2-b3")            # 走一步
print(info.describe())                # 白方 1b2-b3
game.pop()                            # 悔棋

game.push("9f1xf4")                   # 吃子：大小 9 + 4 = 13
print(game.piece_at("f4"), game.result)

recorder = Recorder(Game.new())       # 记录一局并导出棋谱
recorder.push("1b2-b3")
print(recorder.finish())
print(replay(recorder.to_text()).game.fen())   # 从棋谱还原局面
~~~

## 规则要点

完整规则见 [docs/RULES.md](docs/RULES.md)，关键口径：

1. 11x11 格点棋盘，双方镜像布局；每方 18 枚棋子（靠边一行 `4 8 5 2 9 2 5 8 4`，靠中心一行 9 枚 `1`）。
2. 大小为 `n` 的棋子每步必须移动 `sqrt(n)` 格，即位移满足 `dx*dx + dy*dy == n`；
   **无视路径**，只看落点：空点可走、己方棋子不可走、敌方棋子则吃子。
3. 吃子后自身大小 += 被吃棋子大小。
4. 一方棋子被全部吃光即判负。
5. 任意一步之后双方都还有棋子，但有一方所有棋子都没有合法落点 → 平局判定。
6. 双方合计连续 **100 步**（半回合，默认值，可配置）没有吃子 → 平局判定。
7. 平局判定：比较双方棋子数与大小总和，一方两项**都严格更大**则该方胜，否则平局。

## 机器接口

两种 JSON 行协议（详见 [docs/PROTOCOL.md](docs/PROTOCOL.md)）：

~~~bash
echo {"cmd":"moves"} | python -m sqrtchess proto          # 控制协议（有状态）
echo {"request":"move","fen":"startpos"} | python -m sqrtchess ai --level greedy   # AI 应答协议
~~~

任何语言都能接入：[examples/engine_client.js](examples/engine_client.js) 是 Node.js 客户端，
[examples/random_ai.py](examples/random_ai.py) 是一个最小的外部 AI 实现。

## 测试

~~~bat
python -m unittest discover -s tests -v
~~~

perft 基线：深度 1 = 22，2 = 484，3 = 12,342，4 = 314,721。
## 文件与运行产物

- `sqrtchess/`、`gui/`、`ai-*/`：引擎、界面和外部 AI 源码。
- `docs/`、`tests/`、`examples/`：规则与协议、测试、示例及历史对局。
- 锦标赛仍在项目根目录生成 `duel-*.sqg` 和 `tournament.json`；这些文件、日志及 Python 缓存由 `.gitignore` 排除。
- `examples/sample_game.sqg` 使用 50 步无吃子上限，回放时请运行 `python -m sqrtchess replay examples/sample_game.sqg --limit 50`。
- 安装包只包含 `sqrtchess` 引擎；GUI、外部 AI 和对战脚本请从仓库运行。

## 许可证

本项目使用 [MIT 许可证](LICENSE)。
