"""Phần RL cần PyTorch: mạng, chính sách, PPO, vòng huấn luyện. Bỏ qua nếu chưa cài PyTorch."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autochess.core import reset, step, legal_actions, rng
from autochess.rl.observe import observe, flatten_batch, KEYS
from autochess.rl.nets import EntityNet, FlatNet, build_net
from autochess.rl.policy import Policy, to_tensors
from autochess.rl.opponents import OpponentPool, make_opponent
from autochess.rl.rollout import Rollout
from autochess.rl.gae import compute_gae
from autochess.rl.ppo import ppo_update, flatten_rollout
from autochess.rl.config import load_config

from .helpers import CAT, COMBAT, deterministic_actions


def _obs(P, B=6, T=300, seed=0):
    s = reset(CAT, rng.make_seeds(seed, B), P)
    for t in range(T):
        s, _ = step(CAT, COMBAT, s, deterministic_actions(s, t))
    return flatten_batch(observe(CAT, s, COMBAT))


def test_entity_net_runs_for_any_player_count():
    torch.manual_seed(0)
    net = EntityNet(CAT)
    n0 = sum(p.numel() for p in net.parameters())
    assert 100_000 < n0 < 600_000
    for P in (2, 4, 8):
        o = to_tensors(_obs(P))
        logits, value = net(o)
        assert logits.shape == (6 * P, 44) and value.shape == (6 * P,)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_flat_net_runs():
    net = FlatNet(CAT, 2)
    logits, value = net(to_tensors(_obs(2)))
    assert logits.shape == (12, 44)


def test_policy_never_picks_masked_actions_and_is_deterministic():
    net = EntityNet(CAT)
    pol = Policy(net)
    o = _obs(4, B=16)
    a1, lp1, v1 = pol.act(o, np.random.default_rng(5))
    a2, lp2, v2 = pol.act(o, np.random.default_rng(5))
    assert (a1 == a2).all() and np.allclose(lp1, lp2)
    assert o["mask"][np.arange(len(a1)), a1].all()
    assert (lp1 <= 1e-6).all()


def test_opponent_permutation_invariance():
    net = EntityNet(CAT).eval()
    o = _obs(4, B=4)
    perm = [2, 0, 1]
    p = dict(o)
    for k in ("opp", "opp_board_id", "opp_board_star", "opp_alive"):
        p[k] = o[k][:, perm]
    with torch.no_grad():
        l1, v1 = net(to_tensors(o))
        l2, v2 = net(to_tensors(p))
    assert torch.allclose(l1, l2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-5)


def test_shop_slot_swap_swaps_buy_logits():
    net = EntityNet(CAT).eval()
    o = _obs(2, B=4)
    p = dict(o)
    sw = [3, 1, 2, 0, 4]
    p["shop"] = o["shop"][:, sw]
    p["shop_id"] = o["shop_id"][:, sw]
    with torch.no_grad():
        l1, v1 = net(to_tensors(o))
        l2, v2 = net(to_tensors(p))
    assert torch.allclose(l1[:, sw], l2[:, :5], atol=1e-5)
    assert torch.allclose(l1[:, 5:], l2[:, 5:], atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)


@pytest.mark.parametrize("kind", ["entity", "flat"])
def test_ppo_update_smoke(kind):
    torch.manual_seed(0)
    net = build_net(CAT, {"kind": kind}, 2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
    pool = OpponentPool([make_opponent(CAT, COMBAT, "tempo"), make_opponent(CAT, COMBAT, "random")])
    ro = Rollout(CAT, COMBAT, pool, num_games=16, num_players=2, steps=48, seed=3)
    pol = Policy(net)
    cfg = load_config()["ppo"] | {"minibatch_size": 256, "update_epochs": 2}
    for _ in range(2):
        out = ro.collect(pol)
        buf = out["buf"]
        T, B, S = buf["rewards"].shape
        f = lambda x: x.reshape(T, B * S)
        adv, ret = compute_gae(f(buf["rewards"]), f(buf["values"]), f(buf["dones"]), f(buf["gammas"]),
                               f(buf["lambdas"]), f(buf["active"]), out["last_values"].reshape(-1))
        batch = flatten_rollout(buf, adv.reshape(T, B, S), ret.reshape(T, B, S))
        st = ppo_update(net, opt, batch, cfg, np.random.default_rng(0))
        for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac"):
            assert np.isfinite(st[k]), k
        assert st["decisions"] > 0


def test_train_and_resume(tmp_path):
    from autochess.rl import train
    args = ["--set", f"run_dir={tmp_path}", "name=t", "total_decisions=3000", "env.num_games=16",
            "env.steps=32", "ppo.minibatch_size=256", "eval.every=1", "eval.games=8", "eval.B=8",
            "eval.matrix_games=4", "eval.opponents=[random]", "save_every=1", "opponents.names=[random,tempo]"]
    train.main(args)
    runs = os.listdir(tmp_path)
    assert len(runs) == 1
    run = os.path.join(tmp_path, runs[0])
    assert os.path.exists(os.path.join(run, "checkpoints", "latest.pt"))
    train.main(["--resume", run, "--set", "total_decisions=5000"])
    ck = torch.load(os.path.join(run, "checkpoints", "latest.pt"), weights_only=False)
    assert ck["decisions"] >= 5000


def test_bc_then_train_init(tmp_path):
    from autochess.rl import bc, train
    common = [f"run_dir={tmp_path}", "env.num_games=16", "ppo.minibatch_size=256", "eval.B=8",
              "eval.matrix_games=4", "eval.opponents=[tempo]"]
    bc.main(["--set", *common, "bc.samples=3000", "bc.steps=32", "bc.eval_every=1", "bc.eval_games=4"])
    run = [d for d in os.listdir(tmp_path) if d.startswith("bc_")][0]
    ck = os.path.join(tmp_path, run, "checkpoints", "latest.pt")
    assert os.path.exists(ck)
    train.main(["--set", *common, "name=t", f"init={ck}", "ppo.value_warmup=1", "total_decisions=2000",
                "env.steps=32", "eval.every=0", "save_every=1", "opponents.names=[tempo]"])
