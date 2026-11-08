"""Bot luật vector hóa: chỉ chọn hành động hợp lệ, giữ bất biến, tất định, đúng khi chạy một ván."""
import numpy as np
import pytest

from autochess.core import reset, step, legal_actions, state_hash, rng, END_TURN
from autochess.bots.scripted import PRESETS, ScriptedBot
from autochess.play.runner import GameRunner
from autochess.play.seats import ArrayBotSeat

from .helpers import CAT, COMBAT

NAMES = list(PRESETS)


def _lineup_actions(bots, lineup, s, m):
    a = np.full((s.B, s.P), END_TURN, np.int64)
    for n in dict.fromkeys(lineup):
        seats = [i for i, x in enumerate(lineup) if x == n]
        a[:, seats] = bots[n].act(s, m, seats)
    return a


@pytest.mark.parametrize("P", [2, 4, 8])
def test_bots_legal_and_invariants(P):
    bots = {n: ScriptedBot(CAT, PRESETS[n]) for n in NAMES}
    lineup = [NAMES[i % len(NAMES)] for i in range(P)]
    s = reset(CAT, rng.make_seeds(40 + P, 32), P)
    finished = 0
    for _ in range(3000):
        m = legal_actions(CAT, s)
        a = _lineup_actions(bots, lineup, s, m)
        assert np.take_along_axis(m, a[..., None], -1).all(), "bot chọn hành động không hợp lệ"
        s, info = step(CAT, COMBAT, s, a, debug=True)
        finished += int(info["reset"].sum())
        if finished >= 32:
            break
    assert finished >= 32, "ván của bot không kết thúc"


def test_bots_deterministic_and_batch_consistent():
    bots = {n: ScriptedBot(CAT, PRESETS[n]) for n in NAMES}
    lineup = NAMES[:4]          # 4 ghế: bốn bot đầu (bot "bank" được thử ở các kiểm thử khác)
    seeds = rng.make_seeds(9, 6)

    def run(sd):
        s = reset(CAT, sd, 4)
        for _ in range(400):
            m = legal_actions(CAT, s)
            s, _ = step(CAT, COMBAT, s, _lineup_actions(bots, lineup, s, m), mask=m)
        return s

    batch = run(seeds)
    assert state_hash(batch) == state_hash(run(seeds))
    for b in range(6):
        assert state_hash(batch, b) == state_hash(run(seeds[b:b + 1]), 0)


def test_array_bot_seat_in_runner():
    seats = [ArrayBotSeat(ScriptedBot(CAT, PRESETS[n]), n) for n in ("hyper", "fast9")]
    r = GameRunner(CAT, COMBAT, seats, seed=3, debug=True)
    r.run_until(lambda rr: False, max_steps=5000)
    assert r.done
    log = r.to_log()
    assert log["seats"] == ["bot:hyper", "bot:fast9"]


def test_tank_bot_sells_out_then_rebuilds():
    """Bot "bán máu": từ vòng 3 tới lúc dựng lại không có quân nào và giữ vàng >= 46; dựng lại thì sân đầy."""
    from autochess.core import load_catalog, FormulaCombat, reset, step, legal_actions, rng
    from autochess.core.catalog import DATA_DIR
    from autochess.bots.scripted import make_bots, PRESETS
    cat = load_catalog(f"{DATA_DIR}/catalog_v3.json")
    combat = FormulaCombat(catalog=cat)
    lineup = ("tank", "bank", "fast9", "hyper", "tempo", "roll7", "bank", "fast9")
    bots = make_bots(cat, sorted(set(lineup)))
    s = reset(cat, rng.make_seeds(3, 8), 8)
    rb = PRESETS["tank"].rebuild_round
    seen_tank = seen_rebuilt = False
    for _ in range(3000):
        if s.done.all():
            break
        m = legal_actions(cat, s)
        a = np.full((s.B, 8), 43)
        for p, n in enumerate(lineup):
            a[:, p] = bots[n].act(s, m, [p])[:, 0]
        pre = s.copy()
        s, info = step(cat, combat, s, a, mask=m)
        for b in np.flatnonzero(info["combat"]):
            if not pre.alive[b, 0]:
                continue
            n_units = int((pre.units[b, 0] >= 0).sum())
            r = int(pre.round[b])
            if 4 <= r < rb and pre.hp[b, 0] > PRESETS["tank"].rebuild_hp:
                assert n_units == 0, (b, r, n_units)
                if r >= 8:
                    assert pre.gold[b, 0] >= 46, (b, r, int(pre.gold[b, 0]))
                seen_tank = True
            if r == rb + 1:
                assert int((pre.units[b, 0, 9:] >= 0).sum()) >= 5      # đủ vàng thì đầy sân; thiếu thì ít nhất 5 quân
                seen_rebuilt = True
    assert seen_tank and seen_rebuilt
