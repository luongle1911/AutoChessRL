"""Lớp chơi thử: lệnh, view, undo, replay."""
import numpy as np

from autochess.bots import BOTS
from autochess.core import state_hash
from autochess.core.actions import BUY0, SELL0, MOVE0, ROLL, BUY_XP, END_TURN
from autochess.play.adapter import parse_command, to_view, diff_events, action_to_command
from autochess.play.replay import save_log, load_log, replay_frames
from autochess.play.runner import GameRunner
from autochess.play.seats import HumanSeat, ScriptedSeat, Command

from .helpers import CAT, COMBAT, fresh, give


def test_parse_command():
    assert parse_command("b 3") == BUY0 + 3
    assert parse_command("s 10") == SELL0 + 10
    assert parse_command("m 1") == MOVE0 + 1
    assert parse_command("r") == ROLL and parse_command("x") == BUY_XP and parse_command("e") == END_TURN
    assert parse_command("a 17") == 17
    for bad in ("b 5", "s 18", "m", "r 2", "god", "", "b x"):
        assert parse_command(bad) is None, bad
    for a in range(44):
        assert parse_command(action_to_command(a)) == a


def test_view_hides_opponent_private_info():
    s = fresh(P=3)
    give(s, 0, 1, 2, "Lê Kiếm")
    v = to_view(CAT, s, 0, 0, combat=COMBAT)
    o = v["opponents"][0]
    assert o["seat"] == 1 and "shop" not in o
    # hàng chờ đối thủ công khai (như ĐTCL; agent giai đoạn 3 cũng thấy)
    assert o["bench"][2]["name"] == "Lê Kiếm"
    assert "pool" not in v and "matchup" not in v
    vg = to_view(CAT, s, 0, 0, god=True, combat=COMBAT)
    assert vg["opponents"][0]["bench"][2]["name"] == "Lê Kiếm"
    assert "pool" in vg and vg["matchup"] is not None
    # đối thủ sắp tương đối theo ghế người quan sát
    assert [o["seat"] for o in to_view(CAT, s, 0, 2)["opponents"]] == [0, 1]


def test_matchup_row_flag():
    s = fresh(P=2)
    v = to_view(CAT, s, 0, 0, combat=COMBAT, matchup_row=False)
    assert "win_prob" not in v["opponents"][0] and v["expected_win"] is None


def _scripted_human(script):
    it = iter(script)

    def fn(view, mask):
        for a in it:
            if isinstance(a, Command) or mask[a]:
                return a
        return END_TURN
    return fn


def test_undo_restores_exact_state():
    seats = [HumanSeat(_scripted_human([BUY0, BUY0 + 1, ROLL, BUY_XP, END_TURN] * 50)),
             ScriptedSeat(BOTS["econ"], "econ", seed=1)]
    r = GameRunner(CAT, COMBAT, seats, seed=5)
    hashes = [state_hash(r.state)]
    for _ in range(6):
        r.tick()
        hashes.append(state_hash(r.state))
    k = r.undo()
    assert k >= 1
    assert state_hash(r.state) == hashes[6 - k]


def test_bot_game_runs_to_end_and_replays_identically(tmp_path):
    seats = [ScriptedSeat(BOTS["econ"], "econ", seed=i) if i % 2 == 0 else
             ScriptedSeat(BOTS["random"], "random", seed=i) for i in range(4)]
    r = GameRunner(CAT, COMBAT, seats, seed=42)
    r.run_until(lambda _: False)
    assert r.done
    assert sorted(r.state.placement[0].tolist()) == [1, 2, 3, 4]
    final = state_hash(r.state)
    path = save_log(r.to_log(), str(tmp_path / "g.json"))
    log = load_log(path)
    runner2, frames = replay_frames(log, CAT, COMBAT)
    assert state_hash(frames[-1][0]) == final
    assert state_hash(runner2.state) == final


def test_diff_events_agree_with_core_events():
    seats = [ScriptedSeat(BOTS["econ"], "econ", seed=0), ScriptedSeat(BOTS["econ"], "econ", seed=1)]
    r = GameRunner(CAT, COMBAT, seats, seed=8)
    merges = levels = 0
    while not r.done and r.t < 400:
        before = r.view(0)
        info = r.tick()
        after = r.view(0)
        d = diff_events(before, after)
        core_merge = sum(e["type"] == "merge" and e["seat"] == 0 for e in info["events"])
        core_level = any(e["type"] == "level_up" and e["seat"] == 0 for e in info["events"])
        if core_merge:
            merges += 1
            assert any(x.startswith("ghép") for x in d), (info["events"], d)
        if core_level:
            levels += 1
            assert any(x.startswith("lên cấp") for x in d)
        lost = sum(e["hp_lost"] for e in info["events"] if e["type"] == "combat" and e["loser"] == 0)
        if lost:
            assert any(x.startswith("máu") for x in d)
    assert merges > 0 and levels > 0
