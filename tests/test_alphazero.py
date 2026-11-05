"""AlphaZero 部分的测试：编码、网络、搜索、自我对弈、训练与评估。"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sqrtchess import (BLACK, CaptureAI, Game, NUM_SQUARES, WHITE, parse_square, replay,
                       square_name)
from alphazero import AZNetwork, MCTS, TrainConfig, Trainer, encode, play_game, play_match
from alphazero.encoding import PLANES
from alphazero.network import softmax
from alphazero.players import load_player
from alphazero.selfplay import play_baseline_game
from alphazero.trainer import _worker_play

_PIECE_RE = re.compile(r"^([WB])(\d+)([a-k](?:11|10|[1-9]))$")


def position(description: str, side: int = WHITE, limit=100) -> Game:
    cells = [0] * NUM_SQUARES
    for token in description.split():
        match = _PIECE_RE.match(token)
        if not match:
            raise ValueError("无法解析: %r" % (token,))
        color = WHITE if match.group(1) == "W" else BLACK
        cells[parse_square(match.group(3))] = color * int(match.group(2))
    return Game(cells, side=side, no_capture_limit=limit)


class TestEncoding(unittest.TestCase):
    def test_shape_and_planes(self):
        game = Game.new()
        features = encode(game)
        self.assertEqual(features.shape, (PLANES, 11, 11))
        self.assertEqual(float(features[0].sum()), 18.0)   # 白方 18 子
        self.assertEqual(float(features[1].sum()), 18.0)   # 黑方 18 子
        self.assertAlmostEqual(float(features[2].sum()), 56.0 / 112.0, places=5)
        self.assertAlmostEqual(float(features[3].sum()), 56.0 / 112.0, places=5)
        self.assertAlmostEqual(float(features[6].mean()), 1.0, places=6)   # 轮到白方
        self.assertAlmostEqual(float(features[7].mean()), 0.0, places=6)   # 无吃子进度
        self.assertAlmostEqual(float(features[8].mean()), 56.0 / 112.0, places=5)
        self.assertAlmostEqual(float(features[9].mean()), 56.0 / 112.0, places=5)

    def test_side_plane_follows_turn(self):
        game = Game.new()
        game.push("1b2-b3")
        features = encode(game)
        self.assertEqual(float(features[6].mean()), 0.0)   # 轮到黑方

    def test_mobility_plane_marks_stuck_pieces(self):
        game = position("W3a1 B1k11")
        features = encode(game)
        square = parse_square("a1")
        x, y = square % 11, square // 11
        self.assertEqual(float(features[4, y, x]), 0.0)
        self.assertEqual(float(features[5, y, x]), 1.0)
        other = parse_square("k11")
        self.assertGreater(float(features[4, other // 11, other % 11]), 0.0)

    def test_no_capture_progress(self):
        game = Game.new()
        game.no_capture = 50
        features = encode(game)
        self.assertAlmostEqual(float(features[7].mean()), 0.5, places=5)


class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.network = AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=1)
        self.game = Game.new()
        for _ in range(4):
            moves = self.game.legal_moves()
            self.game.push(moves[0])

    def test_evaluate_returns_normalised_priors(self):
        moves = self.game.legal_moves()
        priors, value = self.network.evaluate(self.game)
        self.assertEqual(len(priors), len(moves))
        self.assertAlmostEqual(sum(priors), 1.0, places=5)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "net.json"
            self.network.save(path)
            loaded = AZNetwork.load(path)
        moves = self.game.legal_moves()
        a = self.network.evaluate(self.game)
        b = loaded.evaluate(self.game)
        self.assertEqual(len(a[0]), len(b[0]))
        for left, right in zip(a[0], b[0]):
            self.assertAlmostEqual(left, right, places=6)
        self.assertAlmostEqual(a[1], b[1], places=6)

    def test_gradients_match_finite_differences(self):
        network = AZNetwork(channels=3, policy_hidden=4, value_hidden=4, seed=5)
        for name in list(network.params):
            network.params[name] = network.params[name].astype(np.float64)
        moves = self.game.legal_moves()
        features = encode(self.game).astype(np.float64)
        rng = np.random.default_rng(0)
        pi = rng.dirichlet(np.ones(len(moves)))
        z = 1.0
        grads = network.backward(network.forward(features, moves)[2], pi, z)

        def loss():
            logits, value, _ = network.forward(features, moves)
            return float(-np.sum(pi * np.log(softmax(logits) + 1e-12)) + (value - z) ** 2)

        eps = 1e-6
        for name in ("pw1", "pw2", "vw1", "w2", "w1"):
            array = network.params[name]
            for _ in range(2):
                index = tuple(int(rng.integers(0, size)) for size in array.shape)
                original = float(array[index])
                array[index] = original + eps
                plus = loss()
                array[index] = original - eps
                minus = loss()
                array[index] = original
                numeric = (plus - minus) / (2 * eps)
                analytic = float(grads[name][index])
                self.assertAlmostEqual(numeric, analytic, places=5,
                                       msg="%s%s 梯度不符" % (name, index))

    def test_training_reduces_loss(self):
        network = AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=2)
        rng = np.random.default_rng(1)
        batch = []
        for _ in range(3):
            game = Game.new()
            for _ in range(3):
                game.push(game.legal_moves()[0])
            moves = game.legal_moves()
            batch.append((encode(game), moves,
                          rng.dirichlet(np.ones(len(moves))).astype(np.float32), 1.0))
        first = network.train_batch(batch, lr=0.005)
        for _ in range(200):
            last = network.train_batch(batch, lr=0.005)
        self.assertLess(last["loss"], first["loss"] * 0.8,
                        "训练 200 步后损失应明显下降：%.3f -> %.3f"
                        % (first["loss"], last["loss"]))


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.network = AZNetwork(channels=8, policy_hidden=8, value_hidden=8, seed=3)

    def test_finds_immediate_win(self):
        game = position("W9a1 B1a4")   # 吃掉 a4 后黑方无子，白方胜
        search = MCTS(self.network, sims=24, seed=0)
        move = search.choose(game, temperature=0.0)
        self.assertEqual(move.text(True), "a1xa4")

    def test_visit_counts_and_policy(self):
        game = Game.new()
        search = MCTS(self.network, sims=20, seed=1)
        root = search.search(game)
        total = sum(child.N for child in root.children.values())
        self.assertEqual(total, 20)
        policy = MCTS.visit_policy(root, temperature=1.0)
        self.assertEqual(len(policy), len(game.legal_moves()))
        self.assertAlmostEqual(sum(policy.values()), 1.0, places=6)
        greedy = MCTS.visit_policy(root, temperature=0.0)
        self.assertAlmostEqual(sum(greedy.values()), 1.0, places=6)
        self.assertEqual(sum(1 for value in greedy.values() if value > 0), 1)

    def test_dirichlet_noise_changes_priors(self):
        game = Game.new()
        search = MCTS(self.network, sims=1, seed=2)
        root = search.search(game)
        before = dict(root.priors)
        search.add_dirichlet_noise(root, alpha=0.3, eps=0.5)
        changed = sum(1 for move, value in root.priors.items() if abs(value - before[move]) > 1e-9)
        self.assertGreater(changed, 0)
        self.assertAlmostEqual(sum(root.priors.values()), 1.0, places=5)

    def test_terminal_position(self):
        game = position("W3a1 B1k11")
        self.assertTrue(game.is_over)
        search = MCTS(self.network, sims=5, seed=0)
        self.assertIsNone(search.choose(game))


class TestCaptureBaseline(unittest.TestCase):
    def test_takes_free_capture(self):
        game = position("W9a1 B1a4 B1k11")
        engine = CaptureAI(depth=4, seed=0)
        self.assertEqual(engine.choose(game).text(True), "a1xa4")

    def test_avoids_losing_capture(self):
        # 白 5 吃 f7 的 1 之后会被黑 9 吃回（亏 5），应该选择不吃
        game = position("W5e5 B1f7 B9f4 B1k11")
        engine = CaptureAI(depth=4, seed=0)
        move = engine.choose(game)
        self.assertNotEqual(move.to, parse_square("f7"))

    def test_rank_moves_sorted(self):
        game = position("W9a1 B1a4 B1k11")
        engine = CaptureAI(depth=3, seed=0)
        ranked = engine.rank_moves(game)
        self.assertEqual(len(ranked), len(game.legal_moves()))
        scores = [score for score, _ in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(ranked[0][1].text(True), "a1xa4")


class TestSelfPlay(unittest.TestCase):
    def setUp(self):
        self.network = AZNetwork(channels=8, policy_hidden=8, value_hidden=8, seed=4)

    def test_samples_are_consistent(self):
        data = play_game(self.network, sims=6, max_plies=10, temp_plies=3, seed=5)
        self.assertGreater(len(data.samples), 0)
        for sample in data.samples:
            game = Game.from_fen(sample.fen)
            legal = {game.move_dict(move)["notation"] for move in game.legal_moves()}
            self.assertTrue(set(text for text, _ in sample.policy) <= legal)
            self.assertAlmostEqual(sum(p for _, p in sample.policy), 1.0, places=3)
            self.assertIn(round(sample.z, 3), (-1.0, -0.1, 0.0, 0.1, 1.0))
        report = replay(data.record)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(len(report.moves), len(data.moves))

    def test_truncated_game_uses_adjudication(self):
        data = play_game(self.network, sims=4, max_plies=6, temp_plies=2, seed=6)
        self.assertTrue(data.truncated)
        self.assertEqual(data.reason, "truncated")
        self.assertTrue(all(sample.z in (-1.0, 0.0, 1.0) for sample in data.samples))

    def test_baseline_games_have_soft_targets(self):
        engine = CaptureAI(depth=2, seed=0)
        data = play_baseline_game(engine, seed=1, max_plies=20, epsilon=0.2)
        self.assertGreater(len(data.samples), 0)
        for sample in data.samples:
            total = sum(probability for _, probability in sample.policy)
            self.assertAlmostEqual(total, 1.0, places=4)


class TestArena(unittest.TestCase):
    def test_baseline_beats_random(self):
        result = play_match(load_player("baseline", depth=2), load_player("random", seed=3),
                            games=4, max_plies=120, seed=1)
        self.assertEqual(result.games, 4)
        self.assertGreaterEqual(result.score_a, 0.5)
        self.assertLessEqual(result.score_a, 1.0)

    def test_truncated_games_are_adjudicated(self):
        result = play_match(load_player("random", seed=1), load_player("random", seed=2),
                            games=2, max_plies=6, seed=3)
        self.assertTrue(all(reason in ("truncated-adjudicated", "no-capture-limit",
                                       "no-moves", "all-captured")
                            for reason in result.reasons))


class TestPlayersAndTrainer(unittest.TestCase):
    def test_load_player_variants(self):
        self.assertIsInstance(load_player("baseline"), CaptureAI)
        self.assertIsNotNone(load_player("random"))
        self.assertIsNotNone(load_player("greedy"))
        with self.assertRaises(ValueError):
            load_player("no-such-player")

    def test_load_player_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "net.json"
            AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=0).save(path)
            player = load_player(str(path), sims=4)
            game = Game.new()
            move = player.choose(game)
            self.assertTrue(game.is_legal(move))

    def test_worker_payload(self):
        task = ("az", AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=0).to_json(),
                11, {"sims": 3, "temp_plies": 2, "c_puct": 1.5, "limit": 100,
                     "max_plies": 8, "baseline_depth": 2})
        result = _worker_play(task)
        self.assertGreater(len(result["samples"]), 0)
        self.assertIn("moves", result)

    def test_tiny_training_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = TrainConfig(iterations=1, games_per_iter=2, sims=4, temp_plies=2,
                                 max_plies=16, arena_games=2, arena_sims=3,
                                 baseline_games=1, bootstrap_games=1, workers=1,
                                 channels=6, policy_hidden=8, value_hidden=8,
                                 train_steps=4, batch_size=4, out_dir=tmp,
                                 arena_max_plies=16, keep_records=1, seed=3)
            trainer = Trainer(config, verbose=False)
            summary = trainer.run()
            self.assertTrue(Path(tmp, "best.json").exists())
            self.assertTrue(Path(tmp, "metrics.jsonl").exists())
            self.assertTrue(Path(tmp, "config.json").exists())
            self.assertEqual(len(summary["history"]), 1)
            self.assertGreater(summary["buffer"], 0)
            record = summary["history"][0]
            self.assertIn("gate", record)
            self.assertIn("anchor_baseline", record)


class TestMaterialPrior(unittest.TestCase):
    def test_material_term_sign(self):
        from alphazero.mcts import material_term

        game = position("W9a1 B1k11", side=WHITE)
        self.assertGreater(material_term(game), 0.0)      # 白方物质占优
        game2 = position("W9a1 B1k11", side=BLACK)
        self.assertLess(material_term(game2), 0.0)        # 黑方落后
        game3 = position("W1a1 B1k11", side=WHITE)
        self.assertAlmostEqual(material_term(game3), 0.0, places=6)

    def test_material_prior_steers_search_to_capture(self):
        # 未训练的网络价值接近 0，此时物质先验应该让搜索选中白吃的一步
        network = AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=11)
        game = position("W9a1 B1a4 B1k11")
        search = MCTS(network, sims=30, seed=0, material_weight=1.0)
        move = search.choose(game, temperature=0.0)
        self.assertEqual(move.text(True), "a1xa4")


class TestSparring(unittest.TestCase):
    def test_sparring_records_only_network_decisions(self):
        from alphazero.selfplay import play_sparring_game

        network = AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=12)
        opponent = CaptureAI(depth=2, seed=1)
        data = play_sparring_game(network, opponent, sims=4, max_plies=14, temp_plies=4,
                                  seed=3, net_is_white=True)
        self.assertGreater(len(data.samples), 0)
        for sample in data.samples:
            self.assertEqual(sample.side, "w")     # 只记录网络（白方）的决策
        # 黑方交给基线，所以样本数应当少于总步数
        self.assertLess(len(data.samples), max(1, data.plies))
        for sample in data.samples:
            game = Game.from_fen(sample.fen)
            legal = {game.move_dict(move)["notation"] for move in game.legal_moves()}
            self.assertTrue(set(text for text, _ in sample.policy) <= legal)

    def test_sparring_alternates_colour(self):
        from alphazero.selfplay import play_sparring_game

        network = AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=13)
        opponent = CaptureAI(depth=2, seed=2)
        data = play_sparring_game(network, opponent, sims=4, max_plies=10, temp_plies=2,
                                  seed=4, net_is_white=False)
        for sample in data.samples:
            self.assertEqual(sample.side, "b")


class TestDrawValueSemantics(unittest.TestCase):
    """按和棋原因区分价值目标：无法动子 ±0.1，100 步无吃子 0。"""

    def test_no_moves_draw_is_small_symmetric(self):
        from alphazero.values import sample_value, terminal_value

        game = position("W3a1 B1k11")     # 白方全部棋子都动不了 → 规则 5 判和
        self.assertTrue(game.is_over)
        self.assertEqual(game.result.reason, "no-moves")
        self.assertIsNone(game.result.winner)
        self.assertAlmostEqual(sample_value(game, "w", None, "no-moves"), -0.1, places=6)
        self.assertAlmostEqual(sample_value(game, "b", None, "no-moves"), 0.1, places=6)
        self.assertAlmostEqual(terminal_value(game), -0.1, places=6)   # 轮到白方，白方是卡死方

    def test_no_capture_draw_is_zero(self):
        from alphazero.values import sample_value, terminal_value

        game = position("W1a1 B1k11", limit=3)
        game.no_capture = 2
        game.push("a1-a2")
        self.assertEqual(game.result.reason, "no-capture-limit")
        self.assertAlmostEqual(sample_value(game, "w", None, "no-capture-limit"), 0.0, places=6)
        self.assertAlmostEqual(sample_value(game, "b", None, "no-capture-limit"), 0.0, places=6)
        self.assertAlmostEqual(terminal_value(game), 0.0, places=6)

    def test_win_loss_unchanged(self):
        from alphazero.values import sample_value

        game = position("W1a1 B2a2")
        game.push("a1xa2")               # 吃光黑方，白胜
        self.assertEqual(game.result.winner, WHITE)
        self.assertAlmostEqual(sample_value(game, "w", WHITE, "all-captured"), 1.0, places=6)
        self.assertAlmostEqual(sample_value(game, "b", WHITE, "all-captured"), -1.0, places=6)

    def test_play_game_uses_reason_based_values(self):
        network = AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=21)
        values = set()
        for seed in range(6):
            data = play_game(network, sims=4, max_plies=10, temp_plies=2, seed=seed)
            for sample in data.samples:
                values.add(round(sample.z, 3))
        self.assertTrue(values <= {-1.0, -0.1, 0.0, 0.1, 1.0}, values)


class TestOpponentPool(unittest.TestCase):
    def test_sparring_uses_varied_opponents(self):
        from alphazero.trainer import _worker_play

        weights = AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=0).to_json()
        options = {"sims": 3, "temp_plies": 2, "c_puct": 1.5, "limit": 100, "max_plies": 8,
                   "baseline_depth": 2, "material_weight": 0.45, "draw_no_moves_value": 0.1,
                   "spar_opponents": ["baseline", "random", "greedy", "search2"]}
        names = set()
        for seed in range(4):
            result = _worker_play(("spar", weights, seed, options))
            names.add(result["opponent"])
            self.assertGreater(len(result["samples"]), 0)
        self.assertEqual(names, {"baseline", "random", "greedy", "search2"})

    def test_init_model_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_model = Path(tmp) / "seed.json"
            net = AZNetwork(channels=6, policy_hidden=8, value_hidden=8, seed=5)
            net.params["pb2"] = np.float32(0.25)
            net.save(seed_model)
            out_dir = Path(tmp) / "out"
            config = TrainConfig(iterations=1, games_per_iter=1, sims=3, temp_plies=2,
                                 max_plies=8, arena_games=0, baseline_games=0,
                                 bootstrap_games=0, anchor_others_games=0, workers=1,
                                 channels=6, policy_hidden=8, value_hidden=8,
                                 train_steps=1, batch_size=2, out_dir=str(out_dir),
                                 arena_max_plies=8, keep_records=0, seed=1,
                                 init_model=str(seed_model))
            Trainer(config, verbose=False).run()
            self.assertTrue((out_dir / "best.json").exists())
            loaded = AZNetwork.load(out_dir / "best.json")
            self.assertAlmostEqual(float(loaded.params["pb2"]), 0.25, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)