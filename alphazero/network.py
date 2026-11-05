"""AlphaZero 的策略/价值网络（numpy 手写前向与反向传播 + Adam）。

简化点（相对论文）：
* 主干是两层 3x3 卷积（same padding），16~32 通道；
* 策略头不是固定 121x121 的动作网格，而是「给当前局面的每个合法着法打分」：
  用起点与终点的格点特征 [h_from, h_to, h_from - h_to] 过一个共享的小 MLP，
  softmax 之后得到先验概率。这样参数量与棋盘动作空间解耦，小网络也能用；
* 价值头用全局平均池化 + 最大池化的拼接，过一个小 MLP，tanh 输出 [-1, 1]；
* 优化器是 Adam，损失 = 策略交叉熵 + 价值均方误差 + L2 正则。

权重保存为 JSON（可读、可移植、不依赖 numpy 版本）。
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoding import PLANES, encode

__all__ = ["AZNetwork", "softmax"]

FORMAT = "sqrtchess-az-net"
VERSION = 1


def softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax。"""
    if logits.size == 0:
        return logits
    shifted = logits - float(logits.max())
    exp = np.exp(shifted)
    return exp / float(exp.sum())


def _he(shape: Sequence[int], fan_in: int, rng: np.random.Generator) -> np.ndarray:
    scale = math.sqrt(2.0 / max(1, fan_in))
    return (rng.standard_normal(tuple(shape)) * scale).astype(np.float32)


class AZNetwork:
    """策略 + 价值网络。"""

    def __init__(self, channels: int = 24, policy_hidden: int = 32, value_hidden: int = 32,
                 seed: int = 0, weight_decay: float = 1e-4):
        self.channels = int(channels)
        self.policy_hidden = int(policy_hidden)
        self.value_hidden = int(value_hidden)
        self.weight_decay = float(weight_decay)
        rng = np.random.default_rng(seed)
        c = self.channels
        self.params: Dict[str, np.ndarray] = {
            "w1": _he((c, PLANES, 3, 3), PLANES * 9, rng),
            "b1": np.zeros(c, dtype=np.float32),
            "w2": _he((c, c, 3, 3), c * 9, rng),
            "b2": np.zeros(c, dtype=np.float32),
            "pw1": _he((self.policy_hidden, 3 * c), 3 * c, rng),
            "pb1": np.zeros(self.policy_hidden, dtype=np.float32),
            "pw2": (rng.standard_normal(self.policy_hidden) * 0.01).astype(np.float32),
            "pb2": np.float32(0.0),
            "vw1": _he((self.value_hidden, 2 * c), 2 * c, rng),
            "vb1": np.zeros(self.value_hidden, dtype=np.float32),
            "vw2": (rng.standard_normal(self.value_hidden) * 0.01).astype(np.float32),
            "vb2": np.float32(0.0),
        }
        self._reset_optimizer()

    # ---------------------------------------------------------------- 工具

    def _reset_optimizer(self) -> None:
        self.step = 0
        self.m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.v = {name: np.zeros_like(value) for name, value in self.params.items()}

    @property
    def parameter_count(self) -> int:
        return int(sum(int(np.size(value)) for value in self.params.values()))

    def copy(self) -> "AZNetwork":
        clone = AZNetwork(self.channels, self.policy_hidden, self.value_hidden,
                          seed=0, weight_decay=self.weight_decay)
        for name, value in self.params.items():
            clone.params[name] = value.copy()
        clone._reset_optimizer()
        return clone

    # ---------------------------------------------------------------- 前向

    @staticmethod
    def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """same padding 的 3x3 卷积；返回 (输出, 填充后的输入) 供反向使用。"""
        padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        out = np.zeros((w.shape[0], x.shape[1], x.shape[2]),
                       dtype=np.result_type(x.dtype, w.dtype))
        for dy in range(3):
            for dx in range(3):
                patch = padded[:, dy:dy + x.shape[1], dx:dx + x.shape[2]]
                out += np.tensordot(w[:, :, dy, dx], patch, axes=([1], [0]))
        out += b[:, None, None]
        return out, padded

    def forward(self, features: np.ndarray, moves: Sequence) -> Tuple[np.ndarray, float, Dict]:
        """一次前向：返回 (每个合法着法的 logits, 价值, 缓存)。"""
        params = self.params
        z1, padded_x = self._conv3x3(features, params["w1"], params["b1"])
        h1 = np.maximum(z1, 0)
        z2, padded_h1 = self._conv3x3(h1, params["w2"], params["b2"])
        h2 = np.maximum(z2, 0)
        c = self.channels

        cache: Dict[str, object] = {
            "features": features, "padded_x": padded_x, "z1": z1, "h1": h1,
            "padded_h1": padded_h1, "z2": z2, "h2": h2, "moves": list(moves),
        }

        logits = np.zeros(0, dtype=np.float32)
        if moves:
            from_xy = np.array([[move.frm % 11, move.frm // 11] for move in moves])
            to_xy = np.array([[move.to % 11, move.to // 11] for move in moves])
            emb_from = h2[:, from_xy[:, 1], from_xy[:, 0]].T          # (M, C)
            emb_to = h2[:, to_xy[:, 1], to_xy[:, 0]].T                # (M, C)
            joined = np.concatenate([emb_from, emb_to, emb_from - emb_to], axis=1)
            zp = joined @ params["pw1"].T + params["pb1"]
            act_p = np.maximum(zp, 0)
            logits = act_p @ params["pw2"] + params["pb2"]
            cache.update({"from_xy": from_xy, "to_xy": to_xy, "joined": joined,
                          "zp": zp, "act_p": act_p, "logits": logits})

        pooled = np.concatenate([h2.mean(axis=(1, 2)), h2.max(axis=(1, 2))])
        zv = params["vw1"] @ pooled + params["vb1"]
        act_v = np.maximum(zv, 0)
        value = float(np.tanh(float(params["vw2"] @ act_v + params["vb2"])))
        cache.update({"pooled": pooled, "zv": zv, "act_v": act_v, "value": value})
        return logits, value, cache

    def evaluate(self, game) -> Tuple[List[float], float]:
        """对局面的评估：返回 (与合法着法对齐的先验概率, 价值)。"""
        moves = game.legal_moves()
        if not moves:
            result = game.result
            if result is None or result.winner is None:
                return [], 0.0
            return [], 1.0 if result.winner == game.side else -1.0
        logits, value, _ = self.forward(encode(game), moves)
        return softmax(logits).tolist(), value

    # ---------------------------------------------------------------- 反向

    def backward(self, cache: Dict, pi: np.ndarray, z: float) -> Dict[str, np.ndarray]:
        """计算一个样本的梯度（策略交叉熵 + 价值均方误差）。"""
        params = self.params
        grads = {name: np.zeros_like(value) for name, value in params.items()}
        c = self.channels
        h2 = cache["h2"]
        dh2 = np.zeros_like(h2)

        # ---- 价值头
        value = cache["value"]
        d_value = 2.0 * (value - z) * (1.0 - value * value)
        act_v = cache["act_v"]
        grads["vw2"] += d_value * act_v
        grads["vb2"] += d_value
        d_act_v = d_value * params["vw2"] * (cache["zv"] > 0)
        grads["vw1"] += np.outer(d_act_v, cache["pooled"])
        grads["vb1"] += d_act_v
        d_pooled = params["vw1"].T @ d_act_v
        dh2 += (d_pooled[:c] / float(11 * 11))[:, None, None]
        flat = h2.reshape(c, -1)
        argmax = flat.argmax(axis=1)
        for channel in range(c):
            dh2[channel].reshape(-1)[argmax[channel]] += d_pooled[c + channel]

        # ---- 策略头
        moves = cache["moves"]
        if moves:
            logits = cache["logits"]
            probs = softmax(logits)
            d_logits = probs - pi
            grads["pw2"] += cache["act_p"].T @ d_logits
            grads["pb2"] += float(d_logits.sum())
            d_act_p = np.outer(d_logits, params["pw2"]) * (cache["zp"] > 0)
            grads["pw1"] += d_act_p.T @ cache["joined"]
            grads["pb1"] += d_act_p.sum(axis=0)
            d_joined = d_act_p @ params["pw1"]
            d_from = d_joined[:, :c] + d_joined[:, 2 * c:]
            d_to = d_joined[:, c:2 * c] - d_joined[:, 2 * c:]
            for index, move in enumerate(moves):
                fx, fy = int(cache["from_xy"][index][0]), int(cache["from_xy"][index][1])
                tx, ty = int(cache["to_xy"][index][0]), int(cache["to_xy"][index][1])
                dh2[:, fy, fx] += d_from[index]
                dh2[:, ty, tx] += d_to[index]

        # ---- 主干
        dz2 = dh2 * (cache["z2"] > 0)
        grads["b2"] += dz2.sum(axis=(1, 2))
        padded_h1 = cache["padded_h1"]
        dh1_padded = np.zeros_like(padded_h1)
        for dy in range(3):
            for dx in range(3):
                kernel = params["w2"][:, :, dy, dx]
                patch = padded_h1[:, dy:dy + 11, dx:dx + 11].reshape(c, -1)
                grads["w2"][:, :, dy, dx] += dz2.reshape(c, -1) @ patch.T
                dh1_padded[:, dy:dy + 11, dx:dx + 11] += (
                    kernel.T @ dz2.reshape(c, -1)).reshape(c, 11, 11)
        dh1 = dh1_padded[:, 1:12, 1:12]

        dz1 = dh1 * (cache["z1"] > 0)
        grads["b1"] += dz1.sum(axis=(1, 2))
        padded_x = cache["padded_x"]
        for dy in range(3):
            for dx in range(3):
                kernel = params["w1"][:, :, dy, dx]
                patch = padded_x[:, dy:dy + 11, dx:dx + 11].reshape(PLANES, -1)
                grads["w1"][:, :, dy, dx] += dz1.reshape(c, -1) @ patch.T
        return grads

    # ---------------------------------------------------------------- 训练

    def train_batch(self, batch: Sequence[Tuple[np.ndarray, Sequence, np.ndarray, float]],
                    lr: float, l2: float = None) -> Dict[str, float]:
        """在一个 mini-batch 上做一次 Adam 更新。"""
        total = len(batch)
        if total == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "samples": 0}
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        policy_loss = 0.0
        value_loss = 0.0
        for features, moves, pi, z in batch:
            logits, value, cache = self.forward(features, moves)
            probs = softmax(logits)
            policy_loss += float(-np.sum(pi * np.log(probs + 1e-9)))
            value_loss += float((value - z) ** 2)
            sample_grads = self.backward(cache, pi, z)
            for name in grads:
                grads[name] += sample_grads[name]

        scale = 1.0 / total
        decay = self.weight_decay if l2 is None else l2
        self.step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        bias1 = 1.0 - beta1 ** self.step
        bias2 = 1.0 - beta2 ** self.step
        for name, grad in grads.items():
            g = grad * scale + decay * self.params[name]
            m = self.m[name]
            v = self.v[name]
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * g * g
            self.params[name] -= (lr * (m / bias1) / (np.sqrt(v / bias2) + eps)).astype(np.float32)
        return {"loss": (policy_loss + value_loss) / total,
                "policy_loss": policy_loss / total,
                "value_loss": value_loss / total,
                "samples": total}

    # ---------------------------------------------------------------- 存取

    def to_dict(self) -> Dict:
        return {
            "format": FORMAT,
            "version": VERSION,
            "config": {"channels": self.channels, "policy_hidden": self.policy_hidden,
                       "value_hidden": self.value_hidden, "weight_decay": self.weight_decay},
            "weights": {name: np.asarray(value, dtype=np.float32).tolist()
                        for name, value in self.params.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def save(self, path) -> None:
        """原子写入：先写临时文件再替换，避免其它进程读到写了一半的权重。"""
        path = str(path)
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False)
        os.replace(temp, path)

    @classmethod
    def from_dict(cls, data: Dict) -> "AZNetwork":
        if data.get("format") != FORMAT:
            raise ValueError("不是根号棋 AlphaZero 网络文件: %r" % (data.get("format"),))
        config = data.get("config", {})
        net = cls(channels=int(config.get("channels", 24)),
                  policy_hidden=int(config.get("policy_hidden", 32)),
                  value_hidden=int(config.get("value_hidden", 32)),
                  seed=0, weight_decay=float(config.get("weight_decay", 1e-4)))
        for name, value in data["weights"].items():
            if name not in net.params:
                raise ValueError("未知参数: %s" % name)
            net.params[name] = np.asarray(value, dtype=np.float32)
        net._reset_optimizer()
        return net

    @classmethod
    def from_json(cls, text: str) -> "AZNetwork":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path) -> "AZNetwork":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
