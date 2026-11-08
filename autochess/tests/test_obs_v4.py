"""Huấn luyện trên luật bản 4: quan sát bản 4, mạng bản 4, chỉ số hành vi cuối ván, sảnh "recent"."""
import os

import numpy as np
import pytest

from autochess.core import load_catalog, FormulaCombat, reset, step, legal_actions, rng as crng
from autochess.core.catalog import DATA_DIR, BENCH_SIZE
from autochess.core.combat import board_rating, unit_final_stats
from autochess.bots.scripted import make_bots
from autochess.rl.observe import (observe, keys, UNIT_F4, SHOP_F4, SELF4, OPP4, STAT_UNIT, STAT_TEAM, OPP_KEYS,
                                  pad_opponents)

CAT4 = load_catalog(f"{DATA_DIR}/catalog_v4.json")
COMBAT4 = FormulaCombat(catalog=CAT4)


def _mid_game(B=8, P=8, steps=250, seed=5):
    s = reset(CAT4, crng.make_seeds(seed, B), P)
    bots = make_bots(CAT4, ["tempo", "bank", "tank"])
    names = list(bots)
    for _ in range(steps):
        m = legal_actions(CAT4, s)
        a = np.zeros((B, P), np.int64)
        for p in range(P):
            a[:, p] = bots[names[p % 3]].act(s, m, np.array([p]))[:, 0]
        s, _ = step(CAT4, COMBAT4, s, a, mask=m)
    return s


S = _mid_game()


def test_v4_shapes_and_keys():
    o = observe(CAT4, S, COMBAT4, version=4)
    assert set(o) == set(keys(4))
    B, P = S.B, S.P
    assert o["units"].shape == (B, P, 18, UNIT_F4)
    assert o["shop"].shape == (B, P, 5, SHOP_F4)
    assert o["self"].shape == (B, P, SELF4)
    assert o["opp"].shape == (B, P, P - 1, OPP4)
    for k, v in o.items():
        assert np.isfinite(v.astype(np.float64)).all(), k
        if v.dtype == np.float32:
            assert np.abs(v).max() < 20, k


def test_v4_unit_and_team_stats_match_combat():
    o = observe(CAT4, S, COMBAT4, version=4)
    board = S.units[..., BENCH_SIZE:].astype(np.int64)
    us = unit_final_stats(CAT4, board)
    assert np.allclose(o["units"][..., BENCH_SIZE:, -5:] * STAT_UNIT, us, rtol=1e-4, atol=1e-2)
    # hàng chờ: chỉ số gốc x sao (giáp, kháng phép không nhân)
    b, p, s = np.argwhere(S.units[..., :BENCH_SIZE] >= 0)[0]
    u = int(S.units[b, p, s])
    want = CAT4.stats[u // 3].astype(np.float64).copy()
    want[:3] *= CAT4.star_mult[u % 3]
    assert np.allclose(o["units"][b, p, s, -5:] * STAT_UNIT, want, rtol=1e-4)
    # chỉ số đội của mình nằm ở self, của đối thủ ở opp (ghế kế tiếp = đối thủ 0)
    team = o["self"][..., 12:17]
    assert np.allclose(o["opp"][:, 0, 0, 9:14], team[:, 1] * S.alive[:, 1, None], atol=1e-6)
    assert np.allclose(o["self"][..., 17:22].sum(-1)[S.units[..., BENCH_SIZE:].max(-1) >= 0], 1.0, atol=1e-5)
    assert (team >= 0).all() and team[..., 0].max() * STAT_TEAM[0] > 1000


def test_v4_shop_gain_matches_board_rating():
    o = observe(CAT4, S, COMBAT4, version=4)
    b, p = 0, 0
    board = S.units[b, p, BENCH_SIZE:].astype(np.int64)
    base = board_rating(CAT4, board) if (board >= 0).any() else 0.0
    occ = board >= 0
    n_board = int(occ.sum())
    for i, c in enumerate(S.shop[b, p]):
        if c < 0:
            assert o["shop"][b, p, i, -1] == 0
            continue
        cand = board.copy()
        if n_board < S.level[b, p] and not occ.all():
            cand[np.argmax(~occ)] = int(c) * 3
        else:
            pw = np.where(occ, CAT4.power[np.where(occ, board // 3, 0), np.where(occ, board % 3, 0)], np.inf)
            cand[pw.argmin()] = int(c) * 3
        assert np.isclose(o["shop"][b, p, i, -1] * 20, board_rating(CAT4, cand) - base, rtol=1e-4, atol=1e-4)


def test_v4_seat_selection_consistent():
    full = observe(CAT4, S, COMBAT4, version=4)
    seats = np.tile([3, 6], (S.B, 1))
    part = observe(CAT4, S, COMBAT4, version=4, seats=seats)
    for k in part:
        assert np.allclose(part[k], full[k][np.arange(S.B)[:, None], seats]), k


def test_v4_trait_level_normalized_by_levels():
    o = observe(CAT4, S, COMBAT4, version=4)
    lv = np.asarray(CAT4.trait_levels)
    assert (o["traits"][..., 1] <= 1.0 + 1e-6).all()
    # tộc riêng (1 mốc): kích hoạt thì mức = 1
    on = o["traits"][..., 4] > 0
    assert np.allclose(o["traits"][..., 1][on & (lv == 1)], 1.0)


def test_v4_requires_stats_catalog():
    c3 = load_catalog(f"{DATA_DIR}/catalog_v3.json")
    s = reset(c3, crng.make_seeds(1, 2), 2)
    with pytest.raises(ValueError):
        observe(c3, s, version=4)


def test_v4_pad_four_player_to_eight():
    s4 = reset(CAT4, crng.make_seeds(2, 2), 4)
    o = observe(CAT4, s4, COMBAT4, version=4, seats=np.zeros((2, 1), np.int64))
    flat = {k: v[:, 0] for k, v in o.items()}
    pad = pad_opponents(flat, 7)
    assert pad["opp"].shape == (2, 7, OPP4) and not pad["opp_alive"][:, 3:].any()
    assert all(pad[k].shape[1] == 7 for k in OPP_KEYS if k in pad)


torch = pytest.importorskip("torch")


def test_net_v4_forward_permutation_and_policy_independent_of_bench_order():
    from autochess.rl.nets import build_net, trait_stats_v4
    from autochess.rl.observe import flatten_batch
    from autochess.rl.policy import to_tensors
    for mixer in ("none", "attn"):
        net = build_net(CAT4, {"kind": "entity2", "mixer": mixer}, 8)
        assert net.obs_version == 4 and net.opp_bench
        o = flatten_batch(observe(CAT4, S, COMBAT4, version=4, seats=np.tile([0, 4], (S.B, 1))))
        x = to_tensors(o, "cpu")
        with torch.no_grad():
            l1, v1 = net(x)
            perm = np.random.default_rng(0).permutation(7)
            x2 = {k: (v[:, perm] if k in OPP_KEYS else v) for k, v in x.items()}
            l2, v2 = net(x2)
        assert l1.shape == (o["self"].shape[0], 44)
        assert torch.allclose(l1, l2, atol=1e-4) and torch.allclose(v1, v2, atol=1e-4)
    ts = trait_stats_v4(CAT4)
    k = CAT4.trait_names.index("Hộ vệ")
    assert ts.shape == (CAT4.K, 46) and np.isclose(ts[k, 0], 2 / 6) and ts[k, 3] == 0


def test_bc_collector_v4_eight_players():
    from autochess.rl.botdata import BotDataCollector
    col = BotDataCollector(CAT4, COMBAT4, bots=("tempo", "bank", "tank"), num_games=4, num_players=8, seed=1,
                           obs_version=4)
    d = col.collect(6)
    assert len(d["actions"]) > 0 and d["obs"]["units"].shape[-1] == UNIT_F4
    assert set(np.unique(d["bot"])) <= {0, 1, 2}


def test_late_game_behavior_metrics():
    from autochess.rl.agent import AgentPlayer
    from autochess.rl.evaluate import late_line

    class Fake:
        obs_version = 4

    ag = AgentPlayer(Fake(), CAT4, COMBAT4)
    st = _mid_game(B=4, steps=400)
    ag.LATE_ROUND = 1
    sel = np.ones((st.B, 2), bool)
    seats = np.tile([0, 1], (st.B, 1))
    ag._record_board(st, seats, sel)
    s = ag.late_summary()
    assert s["late_active_traits"] > 0 and 1.0 <= s["comp_diversity"] <= 8
    assert abs(sum(ag.stats["main_trait"].values()) - 8) == 0
    assert "lối chính" in late_line(s)


def test_resolve_recent_lobbies(tmp_path):
    from autochess.rl.train import resolve_recent
    import torch as _t
    from autochess.rl.nets import build_net
    from autochess.rl.config import load_config
    cfg = load_config([], [])
    net = build_net(CAT4, cfg["net"], 8)
    d = tmp_path / "snapshots"
    out, used = resolve_recent({"B": ["recent"] * 3, "A": ["tempo"]}, str(d), {}, CAT4, COMBAT4)
    assert out == {"A": ["tempo"]} and used == {}
    os.makedirs(d)
    for u in (25, 50):
        _t.save(dict(net=net.state_dict(), config=cfg, update=u), d / f"snap_{u:05d}.pt")
    cache = {}
    out, used = resolve_recent({"B": ["recent"] * 3 + ["bank"]}, str(d), cache, CAT4, COMBAT4)
    assert out["B"] == ["snap_00050", "snap_00025", "snap_00050", "bank"] and set(used) == {"snap_00050", "snap_00025"}
