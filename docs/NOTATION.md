# 记谱格式：FEN、着法与棋谱

## 一、SQFEN（局面文本）

格式与国象 FEN 类似，四个字段用空格分隔：

~~~
<棋盘> <行棋方> <无吃子步数> <回合数>
~~~

* **棋盘**：11 行，第 11 行在最前（黑方底线），第 1 行在最后（白方底线），用 `/` 分隔。
  每行恰好 11 个格点记号，记号之间没有分隔符：

  | 记号 | 含义 |
  | --- | --- |
  | `.` | 空点 |
  | `W<大小>` | 白方棋子，大小为十进制整数 |
  | `B<大小>` | 黑方棋子，大小为十进制整数 |

* **行棋方**：`w`（白方，先手）或 `b`（黑方）。
* **无吃子步数**：双方合计连续没有吃子的半回合数（规则 6 用）。
* **回合数**：从 1 开始，黑方走完一步后 +1。

初始局面：

~~~
.B4B8B5B2B9B2B5B8B4./.B1B1B1B1B1B1B1B1B1./.........../.........../.........../.........../.........../.........../.........../.W1W1W1W1W1W1W1W1W1./.W4W8W5W2W9W2W5W8W4. w 0 1
~~~

第 1 行 `.W4W8W5W2W9W2W5W8W4.` 读作：a1 空，b1..j1 依次是白方 4 8 5 2 9 2 5 8 4，k1 空。

解析时的宽松处：

* 大小写不敏感（`w1` 与 `W1` 等价，`b` 与 `B` 等价）。
* 空格可以用 `-`、`*`、`_` 代替（输出时统一写成 `.`）。
* 后面两个字段可以省略（默认 `w 0 1`），也可以只写一部分。
* `startpos`、`initial`、`开局` 等别名表示初始局面。
* 多余字段会被忽略（方便以后扩展）。

命令行与 Python 里的用法：

~~~bash
python -m sqrtchess fen                       # 打印初始局面
python -m sqrtchess board --fen "startpos"    # 打印棋盘
python -m sqrtchess moves --fen "<FEN>"       # 列出该局面的合法着法
~~~

~~~python
from sqrtchess import Game, START_FEN, parse_fen, format_fen

game = Game.from_fen(START_FEN)
print(game.fen())                             # 导出当前局面
print(parse_fen(game.fen()).side)             # 解析
~~~

## 二、着法记谱（自校验坐标记法）

~~~
[大小]起点[-或x]终点
~~~

* `-` 表示走到空点，`x` 表示吃子。
* 最前面的「大小」是**走子前**该棋子的大小，用它可以在回放时校验局面是否同步；可以省略。
* 例：`1b2-b3` = 大小 1 的棋子从 b2 走到 b3；`9f1xf4` = 大小 9 的棋子从 f1 吃 f4。

解析时接受的写法：`b2-b3`、`b2xb3`、`b2b3`、`1b2-b3`、`1b2xb3`、
大小写混用、以及 `:`、`>` 作为分隔符。命令行与界面输入都可以直接用这些写法。

注意：终点是否存在敌方棋子是由棋盘决定的，`-` 与 `x` 只影响可读性；
若写了大小而它与棋盘上的实际大小不符，引擎会报错（这就是「自校验」）。

## 三、棋谱文件（.sqg）

棋谱是纯文本，类似 PGN：

~~~
[Event "根号棋对局"]
[Site "sqrtchess"]
[Date "2026.10.09"]
[Round "-"]
[White "人类"]
[Black "贪心 AI"]
[Result "1-0"]
[Rule "sqrtchess-1.0"]
[Termination "all-captured"]

1. 8i1-k3 9f11-f8
2. 9f1-f4 9f8-c8
3. 9f4xf11 1-0
~~~

规则：

* 头部行写成 `[键 "值"]`，键可自由增减；常用键：
  `Event/Site/Date/Round/White/Black/Result/Rule/Termination`。
* `Termination` 记录结束方式：`all-captured`、`no-moves`、
  `no-capture-limit`、`resign`、`agreement`。
* 正文是回合号 + 着法；回合号与 `...` 在读取时会被忽略，所以只写着法也能读。
* `{ 花括号 }` 是注释，`;` 之后到行尾也是注释，读取时忽略。
* 最后一行是结果：`1-0` / `0-1` / `1/2-1/2` / `*`（未结束）。
* 非标准起始局面会写成 `[FEN "..."]` + `[Setup "1"]`，回放时从该局面开始。
* 一个文件可以放多局，用空行分隔（`parse_records` 会返回全部对局）。

~~~python
from sqrtchess import Recorder, Game, replay, parse_record

recorder = Recorder(Game.new(), {"White": "A", "Black": "B"})
recorder.push("1b2-b3")
recorder.push("1i10-i9")
text = recorder.finish()          # 结束并导出棋谱文本
report = replay(text)             # 回放
print(report.summary())           # 摘要：着法数、推演结果、是否与棋谱声明一致
print(report.moves[0].describe()) # 白方 1b2-b3
~~~

命令行：

~~~bash
python -m sqrtchess replay 对局.sqg --list --board
python -m sqrtchess demo --white greedy --black random --out 对局.sqg
~~~
