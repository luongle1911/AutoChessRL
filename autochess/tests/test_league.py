"""Giai đoạn 2: chọn đối thủ theo nhóm, kho bản chụp (chụp, thay, chạy tiếp), huấn luyện tự đấu + bản neo."""
import os

import numpy as np
import pytest

from autochess.rl.opponents import OpponentPool, make_opponent

from .helpers import CAT, COMBAT


class _FakePolicyOpp:
    kind = "policy"

    def __init__(self, name):
        self.name = name


def test_group_weights_split_between_bots_and_snapshots():
    pool = OpponentPool([make_opponent(CAT, COMBAT, "tempo"), make_opponent(CAT, COMBAT, "fast9"),
                         _FakePolicyOpp("a"), _FakePolicyOpp("b"), _FakePolicyOpp("c")],
                        group_weights={"bot": 0.3, "policy": 0.7})
    pool.winrate[:] = [0.9, 0.2, 0.5, 0.5, 0.1]
    w = pool.weights()
    assert abs(w[:2].sum() - 0.3) < 1e-9 and abs(w[2:].sum() - 0.7) < 1e-9
    assert w[1] > w[0] and w[4] > w[2]                       # trong nhóm vẫn ưu tiên đối thủ khó
    # không có nhóm "policy" thì bot nhận hết
    only = OpponentPool([make_opponent(CAT, COMBAT, "tempo")], group_weights={"bot": 0.3, "policy": 0.7})
    assert only.weights().tolist() == [1.0]
    n = np.bincount(pool.sample(20000, np.random.default_rng(0)), minlength=5) / 20000
    assert abs(n[:2].sum() - 0.3) < 0.02


torch = pytest.importorskip("torch")


def _ckpt(tmp_path, name="seed.pt"):
    from autochess.rl.nets import build_net
    from autochess.rl.config import load_config
    cfg = load_config()
    cfg["versions"] = dict(catalog=CAT.version, combat=COMBAT.version)
    path = os.path.join(tmp_path, name)
    torch.save(dict(net=build_net(CAT, cfg["net"], 2).state_dict(), config=cfg), path)
    return path, cfg


def test_league_snapshot_replace_and_resume(tmp_path):
    from autochess.rl.league import League
    from autochess.rl.nets import build_net
    seed, cfg = _ckpt(tmp_path)
    pool = OpponentPool([make_opponent(CAT, COMBAT, "tempo")], group_weights={"bot": 0.3, "policy": 0.7})
    lg = League(pool, CAT, COMBAT, str(tmp_path), dict(snapshot_every=2, max_snapshots=2, min_games=5))
    lg.add_seed(seed, "neo")
    net = build_net(CAT, cfg["net"], 2)
    assert lg.maybe_snapshot(net, cfg, 1) is None
    for u in (2, 4):
        assert lg.maybe_snapshot(net, cfg, u)
    assert pool.names == ["tempo", "neo", "snap_00002", "snap_00004"]
    # chưa bản nào đủ ván: thay bản cũ nhất
    lg.maybe_snapshot(net, cfg, 6)
    assert pool.names == ["tempo", "neo", "snap_00006", "snap_00004"]
    # đủ ván: thay bản agent thắng nhiều nhất; bản neo không bao giờ bị thay
    pool.games[:] = 10
    pool.winrate[:] = [0.5, 0.99, 0.2, 0.8]
    lg.maybe_snapshot(net, cfg, 8)
    assert pool.names == ["tempo", "neo", "snap_00006", "snap_00008"]
    # chạy tiếp: dựng lại đúng các ô
    st, pst = lg.state_dict(), pool.state_dict()
    pool2 = OpponentPool([make_opponent(CAT, COMBAT, "tempo")], group_weights={"bot": 0.3, "policy": 0.7})
    lg2 = League(pool2, CAT, COMBAT, str(tmp_path), dict(snapshot_every=2, max_snapshots=2))
    lg2.load_state_dict(st)
    pool2.load_state_dict(pst)
    assert pool2.names == pool.names and np.allclose(pool2.winrate, pool.winrate)
    assert pool2.opponents[1].protected and not pool2.opponents[2].protected


def test_selfplay_training_with_league_and_anchors(tmp_path):
    from autochess.rl import train
    seed, _ = _ckpt(tmp_path)
    common = ["opponents.names=[tempo,bank]", "opponents.selfplay_prob=0.5", f"league.seeds={{neo: {seed}}}",
              "league.snapshot_every=1", "league.max_snapshots=2", "reward.churn_cost=0.002",
              "ppo.ent_coef_final=0.001", "env.num_games=12", "env.steps=32", "ppo.minibatch_size=256",
              "eval.every=2", "eval.games=8", "eval.B=8", "eval.matrix_games=4", "eval.opponents=[tempo,random]",
              f"eval.anchors={{neo: {seed}}}", "save_every=1"]
    train.main(["--set", f"run_dir={tmp_path}/runs", "name=sp", f"init={seed}", "total_decisions=2500", *common])
    run = os.path.join(tmp_path, "runs", os.listdir(os.path.join(tmp_path, "runs"))[0])
    ck = torch.load(os.path.join(run, "checkpoints", "latest.pt"), weights_only=False)
    names = [e["name"] for e in ck["league"]]
    assert "neo" in names and any(n.startswith("snap_") for n in names)
    assert os.path.exists(os.path.join(run, "eval_matrix.json"))
    train.main(["--resume", run, "--set", "total_decisions=4000"])
    ck2 = torch.load(os.path.join(run, "checkpoints", "latest.pt"), weights_only=False)
    assert ck2["decisions"] >= 4000 and [e["name"] for e in ck2["league"]][0] == "neo"
