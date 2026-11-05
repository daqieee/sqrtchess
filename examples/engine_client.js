// 示例：用 Node.js 驱动根号棋引擎（演示控制协议）。
//
// 运行： node examples/engine_client.js
// 它做四件事：启动引擎进程 -> 读取局面 -> 自己选着法走 20 步 -> 打印棋盘与棋谱。

const { spawn } = require("child_process");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const engine = spawn("python", ["-m", "sqrtchess", "proto"], {
  cwd: ROOT,
  env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
});

let buffer = "";
const pending = [];

engine.stdout.on("data", (chunk) => {
  buffer += chunk.toString("utf8");
  let index;
  while ((index = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, index).trim();
    buffer = buffer.slice(index + 1);
    if (!line) continue;
    const message = JSON.parse(line);
    pending.push(message);
  }
});

engine.stderr.on("data", (chunk) => process.stderr.write("[engine] " + chunk.toString("utf8")));

function send(payload) {
  engine.stdin.write(JSON.stringify(payload) + "\n");
}

function waitFor(predicate, timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const timer = setInterval(() => {
      const index = pending.findIndex(predicate);
      if (index >= 0) {
        const message = pending.splice(index, 1)[0];
        clearInterval(timer);
        resolve(message);
      } else if (Date.now() > deadline) {
        clearInterval(timer);
        reject(new Error("等待引擎响应超时"));
      }
    }, 10);
  });
}

async function main() {
  const hello = await waitFor((message) => message.hello !== undefined);
  console.log("引擎:", hello.hello, hello.version, "规则:", hello.rule);

  send({ cmd: "new" });
  await waitFor((message) => message.ok && message.state && message.state.ply === 0);

  for (let ply = 0; ply < 20; ply += 1) {
    send({ cmd: "moves" });
    const listing = await waitFor((message) => Array.isArray(message.moves));
    if (listing.count === 0) {
      console.log("没有合法着法，结束。");
      break;
    }
    // 自己选一步：这里随便挑“吃子优先”的第一步
    const capture = listing.moves.find((move) => move.is_capture);
    const choice = capture || listing.moves[Math.floor(Math.random() * listing.moves.length)];
    console.log(String(ply + 1).padStart(3) + ". " + choice.notation);
    send({ cmd: "move", move: choice.notation });
    const played = await waitFor((message) => message.played !== undefined || message.ok === false);
    if (!played.ok) {
      console.log("走子被拒绝:", played.error);
      break;
    }
    if (played.state.game_over) {
      console.log("对局结束:", played.state.result.text);
      break;
    }
  }

  send({ cmd: "board" });
  const board = await waitFor((message) => typeof message.text === "string" && message.fen);
  console.log(board.text);

  send({ cmd: "record" });
  const record = await waitFor((message) => typeof message.text === "string" && message.result !== undefined);
  console.log("棋谱预览：");
  console.log(record.text.split("\n").filter((line) => line.trim()).slice(-4).join("\n"));

  send({ cmd: "quit" });
  await waitFor((message) => message.quit === true).catch(() => {});
  engine.kill();
}

main().catch((error) => {
  console.error("出错:", error.message);
  engine.kill();
  process.exitCode = 1;
});
