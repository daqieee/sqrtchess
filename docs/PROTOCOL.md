# 引擎机器接口（给其它软件与 AI）

引擎提供三种接入方式，全部基于 JSON，UTF-8 编码，一行一个对象：

1. **控制协议**（有状态）：`python -m sqrtchess proto`，stdin/stdout。
2. **AI 应答协议**（无状态）：`python -m sqrtchess ai`，给 AI 用，也可以用 TCP。
3. **TCP 服务**：`python -m sqrtchess serve --port 8765`，每个连接一个独立会话，
   同一个端口同时识别两种协议（带 `cmd` 的走控制协议，带 `request` 的走 AI 协议）。

另外，Python 程序可以直接 `import sqrtchess` 使用引擎对象（见 README）。

## 一、控制协议

请求：`{"cmd": "命令", ...参数}`；响应：`{"ok": true, ...}` 或 `{"ok": false, "error": "..."}`。
非 JSON 的单行纯文本会被当作 `{"cmd": 该文本}`，方便手敲调试。

### 命令表

| 命令 | 别名 | 参数 | 说明 |
| --- | --- | --- | --- |
| `ping` | | | 连通性测试 |
| `info` | | | 名称、版本、规则版本、支持的命令、规则文本 |
| `new` | `reset` | `fen`, `headers` | 开新局（可指定起始局面） |
| `restart` | | `ply` | 回到起始局面（或第 ply 步） |
| `state` | `status` | `moves`(bool) | 完整状态（默认含合法着法） |
| `fen` | `load`, `load_fen`, `set_fen` | `fen` | 不带参数返回当前 FEN；带参数则载入 |
| `board` | | `flip` | 文本棋盘 |
| `moves` | `legal_moves` | | 全部合法着法（含吃子与结果大小） |
| `move` | `play`, `do` | `move` 或 `from`+`to` | 走一步 |
| `undo` | | `plies` / `count` | 悔棋若干步 |
| `goto` | | `ply` | 跳到着法行的第 ply 步（0 = 起始局面） |
| `line` | `history` | | 着法行（自校验记谱） |
| `record` | `get_record` | | 导出棋谱文本（含头部与结果） |
| `load_record` | `open_record` | `text` 或 `path` | 载入棋谱并回放 |
| `save_record` | `save` | `path`, `text`(可选) | 把棋谱写入文件 |
| `adjudicate` | | `reason` | 按规则 7 做一次平局判定（不改变局面） |
| `ai` | | `level`, `depth`, `seed`, `play` | 让引擎 AI 选一步；`play=true` 时直接走出 |
| `perft` | | `depth` | 走法生成计数（自检用） |
| `rules` | | | 规则文本 |
| `help` | | | 命令列表 |
| `quit` | `exit`, `bye` | | 退出（响应里带 `"quit": true`） |

### 状态对象

`state` 命令以及大多数命令的响应的 `state` 字段：

~~~json
{
  "fen": "…", "turn": "w", "turn_name": "白方", "move_number": 1, "ply": 0,
  "no_capture": 0, "no_capture_limit": 100,
  "white": {"pieces": 18, "size_sum": 56},
  "black": {"pieces": 18, "size_sum": 56},
  "game_over": false, "result": null, "last_move": null,
  "legal_moves": [{"move": "b1-b3", "notation": "4b1-b3", "from": "b1", "to": "b3",
                   "color": "w", "size": 4, "captured": 0, "new_size": 4, "is_capture": false}],
  "legal_move_count": 22,
  "line": ["4b1-b3"], "start_fen": "…", "declared_result": null
}
~~~

结果对象（`game_over` 为 true 时）：

~~~json
{"winner": "w", "score": "1-0", "reason": "all-captured",
 "reason_text": "吃光对方全部棋子", "detail": "黑方棋子被全部吃光",
 "text": "白方胜（吃光对方全部棋子）：黑方棋子被全部吃光", "is_draw": false}
~~~

### 会话示例

~~~text
$ python -m sqrtchess proto
{"ok":true,"hello":"sqrtchess","version":"1.0.0","protocol":1,"rule":"sqrtchess-1.0","commands":[...],"fen":"…"}
{"cmd":"moves"}
{"ok":true,"count":22,"turn":"w","moves":[…],"text":["4b1-b3","4b1-b2",…]}
{"cmd":"move","move":"4b1-b3"}
{"ok":true,"played":{"move":"b1-b3","notation":"4b1-b3",…},"state":{…}}
{"cmd":"goto","ply":0}
{"ok":true,"ply":0,"state":{…}}
{"cmd":"ai","level":"greedy","play":true}
{"ok":true,"level":"greedy","think_ms":1.2,"move":"f11-f8","notation":"9f11-f8","played":{…},"state":{…}}
{"cmd":"record"}
{"ok":true,"text":"[Event \"根号棋对局\"]\n…","result":"*","headers":{…},"moves":[…]}
{"cmd":"quit"}
{"ok":true,"quit":true}
~~~

## 二、AI 应答协议（写 AI 用这个）

无状态：每次请求带上完整局面，AI 回一步棋。

请求字段：

| 字段 | 说明 |
| --- | --- |
| `request` | `"move"`（默认）或 `"ping"` |
| `fen` | 当前局面（SQFEN）；省略表示初始局面 |
| `level` | 内置 AI 档位：`capture`（吃子基线，尽可能多吃子）/ `random` / `greedy` / `search2` / `search3` |
| `depth` / `seed` | 可选 |
| `time_ms` | 可选，思考时间上限（提示用） |

响应：

~~~json
{"ok": true, "move": "9f11xf8", "notation": "9f11xf8", "detail": {…}, "level": "greedy", "think_ms": 1.2}
~~~

如果局面没有合法着法：`{"ok": true, "move": null, "reason": "当前局面没有合法着法"}`。

命令行用法（把 FEN 通过管道交给它）：

~~~bash
echo {"request":"move","fen":"startpos"} | python -m sqrtchess ai --level greedy
echo {"cmd":"moves"} | python -m sqrtchess proto      # 控制协议
~~~

## 三、写一个自己的 AI

只要程序能从 stdin 逐行读 JSON、往 stdout 逐行写 JSON 就行，任何语言都可以。
下面是一个最小的 Python AI（完整版见 `examples/random_ai.py`）：

~~~python
import json, random, sys
sys.path.insert(0, "sqrtchess 所在目录")
from sqrtchess import Game

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    if request.get("request") == "ping":
        print(json.dumps({"ok": True, "name": "my-ai"}), flush=True)
        continue
    game = Game.from_fen(request.get("fen", "startpos"))
    moves = game.legal_moves()
    move = random.choice(moves) if moves else None
    answer = {"ok": True, "move": game.move_dict(move)["move"] if move else None}
    print(json.dumps(answer, ensure_ascii=False), flush=True)
~~~

在图形界面里使用：把「对局设置」里的白方/黑方选为 **外部程序**，命令行填
`python 我的AI.py`（或任意可执行文件）。

命令行测试：

~~~bash
echo {"request":"move","fen":"startpos"} | python 我的AI.py
~~~

## 四、TCP 服务

~~~bash
python -m sqrtchess serve --host 127.0.0.1 --port 8765
~~~

* 每个连接一个独立的控制协议会话，连接建立时会先收到一行 hello（含初始 FEN）。
* 同一连接里，带 `cmd` 的请求走控制协议；带 `request` 而不是 `cmd` 的请求走 AI 协议。
* 适合跨语言、跨进程或局域网接入。

## 五、其它语言的客户端

`examples/engine_client.js` 是一个 Node.js 客户端示例：启动引擎进程、读取局面、
选一步棋、导出棋谱。核心代码：

~~~javascript
const { spawn } = require("child_process");
const engine = spawn("python", ["-m", "sqrtchess", "proto"], { cwd: "…", env: { ...process.env, PYTHONUTF8: "1" } });
let buffer = "";
engine.stdout.on("data", chunk => {
  buffer += chunk.toString("utf8");
  let index;
  while ((index = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, index); buffer = buffer.slice(index + 1);
    if (line.trim()) handle(JSON.parse(line));
  }
});
engine.stdin.write(JSON.stringify({ cmd: "moves" }) + "\n");
~~~

注意：通过管道通信时引擎会强制使用 UTF-8；如果宿主程序的默认编码不是 UTF-8，
请设置环境变量 `PYTHONUTF8=1` 与 `PYTHONIOENCODING=utf-8`（Windows 上尤其重要）。