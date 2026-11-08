"""Bất biến kiểm ở mỗi bước, trên đội hình ngẫu nhiên (Hypothesis tự thu nhỏ ca lỗi)."""
import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from battlesim import Simulator, load_config, make_batch  # noqa: E402
from battlesim.tests.helpers import bounce_counts, random_team  # noqa: E402
from battlesim import hexgrid  # noqa: E402
from battlesim.layout import (  # noqa: E402
    F_DEALT, F_HP, F_HP_MAX, F_TAKEN, I_ALIVE, I_BLOCKED, I_CELL, I_SIDE, I_CHAMP, I_RANGE,
    EV_ATTACK, EV_DEATH, EV_MOVE, U,
)

CFG = load_config()
MAX_MISSED = 10   # số bước liên tiếp tối đa có đường tới chỗ đánh mà không đi được
MAX_BOUNCE = 3    # số lần liên tiếp tối đa đi về ô vừa rời mà chưa ra đòn


def team_strategy():
    unit = st.tuples(st.integers(0, CFG.n_champs - 1), st.integers(1, 3))
    return st.integers(1, 8).flatmap(
        lambda k: st.tuples(st.lists(unit, min_size=k, max_size=k),
                            st.lists(st.integers(0, 27), min_size=k, max_size=k, unique=True))
    ).map(lambda t: [(c, s, h) for (c, s), h in zip(t[0], t[1])])


def check_invariants(st_, b):
    F, I, occ = st_.F[b], st_.I[b], st_.occ[b]
    used = I[:, I_CHAMP] >= 0
    alive = I[:, I_ALIVE] == 1
    assert np.all(F[used, F_HP] <= F[used, F_HP_MAX] + 1e-3)
    assert np.all(F[used & ~alive, F_HP] == 0), "tướng chết phải còn 0 máu"
    assert np.all(F[alive, F_HP] > 0)
    cells = I[alive, I_CELL]
    assert len(np.unique(cells)) == len(cells), "hai tướng trên một ô"
    expect = np.full(56, -1)
    expect[cells] = np.flatnonzero(alive)
    assert np.array_equal(occ.astype(int), expect), "mảng chiếm chỗ lệch với ô của tướng"
    for sd in (0, 1):
        mine = used & (I[:, I_SIDE] == sd)
        theirs = used & (I[:, I_SIDE] != sd)
        assert np.isclose(F[mine, F_DEALT].sum(), F[theirs, F_TAKEN].sum(), rtol=1e-5, atol=1e-3)


def reachable_attack_spot(u, occ0, I0):
    """Lúc ra quyết định (đầu bước), từ ô của tướng u có đường qua ô trống tới một ô
    đứng được để đánh một kẻ địch còn sống hay không."""
    start = I0[u, I_CELL]
    rng_u = I0[u, I_RANGE]
    enemies = [I0[v, I_CELL] for v in range(U)
               if I0[v, I_ALIVE] == 1 and I0[v, I_SIDE] != I0[u, I_SIDE]]
    seen, frontier = {start}, [start]
    while frontier:
        nxt = []
        for c in frontier:
            for n in hexgrid.NEIGHBORS[c]:
                if n < 0 or n in seen or occ0[n] >= 0:
                    continue
                if any(hexgrid.DIST[n, e] <= rng_u for e in enemies):
                    return True
                seen.add(n)
                nxt.append(n)
        frontier = nxt
    return False


def update_missed(missed, occ0, I0, I1):
    """Đếm số bước liên tiếp tướng muốn đi mà không đi được, dù lúc ra quyết định đã có
    đường tới chỗ đứng đánh. Chờ khi mọi chỗ đánh đều bị chiếm không bị đếm."""
    for u in range(U):
        tried = I1[u, I_ALIVE] == 1 and I1[u, I_BLOCKED] > I0[u, I_BLOCKED]
        missed[u] = missed[u] + 1 if tried and reachable_attack_spot(u, occ0, I0) else 0


@settings(max_examples=100, deadline=None)
@given(team_strategy(), team_strategy(), st.sampled_from([0, 1]), st.booleans())
def test_invariants_every_step(team_a, team_b, level, stochastic):
    sim = Simulator(CFG, level=level, stochastic=stochastic, log=True, max_events=8192)
    state = sim.init_state(make_batch([team_a], [team_b]))
    death = {}
    missed = np.zeros(U, dtype=int)
    res = -1
    for s in range(1, CFG.max_steps + 1):
        occ0, I0 = state.occ[0].copy(), state.I[0].copy()
        res = sim.step(state, s)[0]
        check_invariants(state, 0)
        update_missed(missed, occ0, I0, state.I[0])
        assert missed.max() < MAX_MISSED, f"tướng {missed.argmax()} kẹt giả {missed.max()} bước"
        if res >= 0:
            break
    # trận luôn kết thúc trong giới hạn bước (hết giờ cũng là kết thúc: cả hai cùng thua)
    assert s <= CFG.max_steps
    # tướng đã chết không hành động, không bị đánh sau khi chết
    assert state.evc[0] <= state.ev.shape[1], "nhật ký bị tràn"
    ev = state.ev[0, :state.evc[0]]
    assert max(bounce_counts(ev)) < MAX_BOUNCE, "tướng đi qua lại giữa hai ô" 
    for row in ev[ev[:, 1] == EV_DEATH]:
        death[int(row[2])] = row[0]
    for row in ev[(ev[:, 1] == EV_ATTACK) | (ev[:, 1] == EV_MOVE)]:
        src = int(row[2])
        assert src not in death or row[0] <= death[src]
        if row[1] == EV_ATTACK:
            dst = int(row[3])
            assert dst not in death or row[0] <= death[dst]


def test_seed_reproducible_in_stochastic_mode(cfg, np_rng):
    ta = [random_team(np_rng, cfg.n_champs, 6) for _ in range(30)]
    tb = [random_team(np_rng, cfg.n_champs, 6) for _ in range(30)]
    batch = make_batch(ta, tb, seeds=np.arange(30) * 7 + 1)
    sim = Simulator(cfg, stochastic=True, log=True)
    r1, r2 = sim.run(batch), sim.run(batch)
    assert np.array_equal(r1.result, r2.result)
    assert np.array_equal(r1.dmg_dealt, r2.dmg_dealt)
    assert all(np.array_equal(a, b) for a, b in zip(r1.events, r2.events))


def test_stochastic_seeds_differ(cfg, np_rng):
    team_a = random_team(np_rng, cfg.n_champs, 6)
    team_b = random_team(np_rng, cfg.n_champs, 6)
    batch = make_batch([team_a] * 50, [team_b] * 50, seeds=np.arange(50))
    r = Simulator(cfg, stochastic=True).run(batch)
    assert len(np.unique(r.dmg_dealt.sum(axis=1))) > 1
