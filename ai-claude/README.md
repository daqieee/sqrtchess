# Claude 根号棋 AI

`strong_ai.py` 为纯 Python 标准库实现，独立维护棋盘与走法生成，使用迭代加深、PVS / Alpha-Beta 搜索、置换表与静态搜索。

从项目根目录运行（也可填写到 GUI 的“外部程序”）：

```sh
python -B ai-claude/strong_ai.py
```

标准输入和标准输出使用 JSON 行协议，例如：

```json
{"request":"move","fen":"startpos","time_ms":300}
```

分析初始局面：

```sh
python -B ai-claude/strong_ai.py --fen startpos --time 1
```

完整接口见 [协议文档](../docs/PROTOCOL.md)，多 AI 对战见[项目说明](../README.md)。
