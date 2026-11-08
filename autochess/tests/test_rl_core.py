"""Phần RL không cần PyTorch: quan sát, phần thưởng, GAE, vòng thu thập."""
import numpy as np
import pytest

from autochess.core import reset, step, legal_actions, rng, END_TURN
from autochess.bots.random_bot import random_actions
from autochess.rl.observe import observe, select_seats, opponent_index, self_dim, opp_dim, KEYS
from autochess.rl.reward import RewardTracker, placement_reward, potential
from autochess.rl.gae import compute_gae
from autochess.rl.opponents import OpponentPool, make_opponent
from autochess.rl.rollout import Rollout, RandomPolicy

from .helpers import CAT, COMBAT, deterministic_actions


def _advance(P, B, T, seed=0):
    s = reset(CAT, rng.make_seeds(seed, B), P)
    for t in range(T):
        s, _ = step(CAT, COMBAT, s, deterministic_actions(s, t))
    return s


@pytest.mark.parametrize("P", [2, 4, 8])
def test_observe_shapes_and_ranges(P):
    s = _advance(P, 16, 300)
    o = observe(CAT, s, COMBAT)
    assert set(o) == set(KEYS)
    assert o["self"].shape == (16, P, self_dim(CAT))
    assert o["opp"].shape == (16, P, P - 1, opp_dim(CAT))
    assert o["opp_board_id"].shape == (16, P, P - 1, 9)
    for k in ("self", "units", "shop", "opp"):
        assert np.isfinite(o[k]).all()
        assert o[k].min() >= -1.5 and o[k].max() <= 3.0, k


def test_observe_batch_consistent():
    s = _advance(4, 8, 400)
    full = observe(CAT, s, COMBAT)
    for b in range(8):
        one = observe(CAT, s.rows(b), COMBAT)
        for k in KEYS:
            assert np.array_equal(full[k][b], one[k][0]), k


def test_observe_opponent_order_and_seat_selection():
    P = 4
    s = _advance(P, 4, 350)
    o = observe(CAT, s, COMBAT)
    oi = opponent_index(P)
    for p in range(P):
        np.testing.assert_allclose(o["opp"][:, p, :, 1], s.hp[:, oi[p]] / 100.0, rtol=1e-6)
    seats = np.array([[2], [0], [3], [1]])
    sel = observe(CAT, s, COMBAT, seats=seats)
    for b in range(4):
        assert np.array_equal(sel["self"][b, 0], o["self"][b, seats[b, 0]])


def test_observe_hides_opponent_bench_and_shop():
    """Đổi hàng chờ và shop của đối thủ không được làm đổi quan sát của mình."""
    s = _advance(2, 4, 250)
    a = observe(CAT, s, COMBAT)
    t = s.copy()
    # đổi chỗ một quân hàng chờ của đối thủ (ghế 1) với kho, và đổi shop của họ
    for b in range(4):
        for j in range(9):
            if t.units[b, 1, j] >= 0:
                c = t.units[b, 1, j] // 3
                t.pool[b, c] += [1, 3, 9][t.units[b, 1, j] % 3]
                t.units[b, 1, j] = -1
        for k in range(5):
            if t.shop[b, 1, k] >= 0:
                t.pool[b, t.shop[b, 1, k]] += 1
                t.shop[b, 1, k] = -1
    b_ = observe(CAT, t, COMBAT)
    for k in KEYS:
        if k == "mask":
            continue
        assert np.array_equal(a[k][:, 0], b_[k][:, 0]), k


def test_placement_reward_zero_sum():
    for P in (2, 4, 8):
        r = placement_reward(np.arange(1, P + 1), P)
        assert r[0] == 1 and r[-1] == -1 and abs(r.sum()) < 1e-6


def test_shaping_sums_to_zero_per_episode():
    """Với gamma = 1, tổng thưởng phụ của mỗi tập = 0; tổng thưởng = thưởng thứ hạng."""
    P, B = 4, 16
    tr = RewardTracker(CAT, P, gamma_round=1.0, lam=1.0, beta=0.1)
    s = reset(CAT, rng.make_seeds(3, B), P)
    seats = np.tile(np.arange(P), (B, 1))
    acc = np.zeros((B, P))
    totals, placed = [], []
    g = np.random.default_rng(0)
    for _ in range(4000):
        pre = tr.before(s, seats)
        a = random_actions(legal_actions(CAT, s), g, end_bias=0.2)
        s, info = step(CAT, COMBAT, s, a)
        r, done, _, _ = tr.after(s, info, pre)
        acc += r
        for b, p in zip(*np.nonzero(done)):
            totals.append(acc[b, p])
            placed.append(placement_reward(s.placement[b, p], P))
            acc[b, p] = 0
        if len(totals) > 200:
            break
    np.testing.assert_allclose(totals, placed, atol=1e-5)


def test_gae_matches_manual_with_transparent_steps():
    # một cột: bước 0 active, bước 1 trong suốt, bước 2 active và kết thúc tập, bước 3 tập mới
    r = np.array([[0.1], [0.0], [1.0], [0.2]], np.float32)
    v = np.array([[0.5], [9.9], [0.4], [0.3]], np.float32)
    d = np.array([[0], [0], [1], [0]], bool)
    g = np.array([[0.9], [1.0], [0.9], [1.0]], np.float32)
    l = np.array([[0.8], [1.0], [0.8], [1.0]], np.float32)
    act = np.array([[1], [0], [1], [1]], bool)
    adv, ret = compute_gae(r, v, d, g, l, act, np.array([0.7], np.float32))
    a3 = 0.2 + 1.0 * 0.7 - 0.3
    a2 = 1.0 - 0.4                       # kết thúc tập: không nhìn sang bước 3
    a0 = (0.1 + 0.9 * 0.4 - 0.5) + 0.9 * 0.8 * a2   # bước 1 trong suốt
    np.testing.assert_allclose(adv[:, 0], [a0, 0.0, a2, a3], rtol=1e-6)
    np.testing.assert_allclose(ret[[0, 2, 3], 0], adv[[0, 2, 3], 0] + v[[0, 2, 3], 0], rtol=1e-6)


def test_pool_prioritizes_hard_opponents():
    pool = OpponentPool([make_opponent(CAT, COMBAT, n) for n in ("tempo", "hyper", "random")])
    pool.winrate[:] = [0.4, 0.7, 0.98]
    w = pool.weights()
    assert w[0] > w[1] > w[2] > 0


@pytest.mark.parametrize("selfplay", [0.0, 0.5])
def test_rollout_runs_and_books_games(selfplay):
    pool = OpponentPool([make_opponent(CAT, COMBAT, n) for n in ("tempo", "fast9", "random")])
    ro = Rollout(CAT, COMBAT, pool, num_games=24, num_players=2, steps=64, seed=1, selfplay_prob=selfplay)
    finished = []
    for _ in range(6):
        out = ro.collect(RandomPolicy())
        finished += out["finished"]
        buf = out["buf"]
        assert buf["obs"]["mask"][buf["active"]].any(-1).all()
        assert not buf["decision"][~buf["active"]].any()
        assert np.isfinite(buf["rewards"]).all()
        assert (buf["values"][~buf["active"]] == 0).all()
        T, B, S = buf["rewards"].shape
        adv, ret = compute_gae(buf["rewards"].reshape(T, -1), buf["values"].reshape(T, -1),
                               buf["dones"].reshape(T, -1), buf["gammas"].reshape(T, -1),
                               buf["lambdas"].reshape(T, -1), buf["active"].reshape(T, -1),
                               out["last_values"].reshape(-1))
        assert np.isfinite(adv).all()
    assert len(finished) > 0
    assert pool.games.sum() > 0
    for f in finished:
        assert all(1 <= x <= 2 for x in f["agent_placement"])


def test_agent_player_in_runner_and_tournament():
    """AgentPlayer (với chính sách bất kỳ) ngồi được vào ghế của giao diện chơi thử và của giải đấu."""
    from autochess.play.runner import GameRunner
    from autochess.play.seats import ArrayBotSeat
    from autochess.rl.agent import AgentPlayer
    from autochess import tournament as T
    ag = AgentPlayer(RandomPolicy(), CAT, COMBAT, seed=1)
    seats = [ArrayBotSeat(ag, "agent"), ArrayBotSeat(AgentPlayer(RandomPolicy(), CAT, COMBAT, seed=2), "agent")]
    r = GameRunner(CAT, COMBAT, seats, seed=4, debug=True)
    r.run_until(lambda rr: False, max_steps=20000)
    assert r.done
    res = T.play(CAT, COMBAT, ("agent", "tempo"), 16, seed=3, B=8, players={"agent": ag})
    assert len(res["placements"]) == 16
    s = ag.summary()
    assert s["decisions_per_round"] > 0


def test_bot_data_collector():
    from autochess.rl.botdata import BotDataCollector
    col = BotDataCollector(CAT, COMBAT, num_games=32, seed=2)
    d = col.collect(80)
    M = len(d["actions"])
    assert M > 500
    assert d["obs"]["mask"][np.arange(M), d["actions"]].all(), "hành động bot phải hợp lệ"
    assert (d["obs"]["mask"].sum(-1) > 1).all(), "chỉ giữ bước có lựa chọn thật"
    assert set(np.unique(d["bot"])) <= {0, 1, 2, 3}


def test_churn_cost_only_on_sell_and_move():
    from autochess.core.actions import SELL0, ROLL
    pool = OpponentPool([make_opponent(CAT, COMBAT, "tempo")])
    outs = []
    for c in (0.0, 0.01):
        ro = Rollout(CAT, COMBAT, pool, num_games=8, num_players=2, steps=60, seed=11, churn_cost=c)
        outs.append(ro.collect(RandomPolicy())["buf"])
    a, b = outs
    assert np.array_equal(a["actions"], b["actions"])          # cùng seed: cùng diễn biến
    churn = a["decision"] & (a["actions"] >= SELL0) & (a["actions"] < ROLL)
    assert churn.any()
    np.testing.assert_allclose(a["rewards"] - b["rewards"], 0.01 * churn, atol=1e-6)


def test_entropy_schedule():
    pytest.importorskip("torch")
    from autochess.rl.train import ent_at
    assert ent_at({"ent_coef": 0.01}, 50, 100) == 0.01
    p = {"ent_coef": 0.01, "ent_coef_final": 0.002}
    assert ent_at(p, 0, 100) == 0.01 and abs(ent_at(p, 50, 100) - 0.006) < 1e-12 and ent_at(p, 200, 100) == 0.002
