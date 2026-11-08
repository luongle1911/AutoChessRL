"""Kiểm thử từng luật: dựng trạng thái tay rồi kiểm tra một hành động."""
import numpy as np
import pytest

from autochess.core import step, check_invariants, legal_actions, explain, IllegalActionError, meet_prob
from autochess.core.actions import BUY0, SELL0, MOVE0, ROLL, BUY_XP, END_TURN
from autochess.core.combat import board_features
from autochess.core.pairing import matchings_table, draw_pairs

from .helpers import CAT, COMBAT, fresh, give, set_shop, code, name_id, acts


def ok(s):
    errs = check_invariants(CAT, s)
    assert not errs, errs


def strong_board(s, b, p, names=("Long Vương", "Tử Vi", "Thần Tiễn", "Huyết Ảnh", "Cự Thạch")):
    s.level[b, p] = 9
    for k, n in enumerate(names):
        give(s, b, p, 9 + k, n, 2 if CAT.cost[name_id(n)] < 5 else 1)


# ---------------------------------------------------------------- ghép sao
def test_merge_two_on_board_one_bought():
    s = fresh()
    s.level[0, 0] = 3
    give(s, 0, 0, 9, "Lê Kiếm")
    give(s, 0, 0, 10, "Lê Kiếm")
    set_shop(s, 0, 0, ["Lê Kiếm", None, None, None, None])
    ok(s)
    s, info = step(CAT, COMBAT, s, acts(2, p0=BUY0 + 0, p1=BUY0 + 0), debug=True)
    u = s.units[0, 0]
    assert u[9] == code("Lê Kiếm", 2), "phải giữ ô sân đầu tiên"
    assert u[10] == -1 and u[0] == -1
    assert any(e["type"] == "merge" and e["star"] == 2 for e in info["events"])


def test_merge_chain_to_three_star():
    s = fresh()
    s.level[0, 0] = 3
    give(s, 0, 0, 9, "Lê Kiếm", 2)
    give(s, 0, 0, 1, "Lê Kiếm", 2)
    give(s, 0, 0, 2, "Lê Kiếm")
    give(s, 0, 0, 10, "Lê Kiếm")
    set_shop(s, 0, 0, ["Lê Kiếm", None, None, None, None])
    ok(s)
    s, info = step(CAT, COMBAT, s, acts(2, p0=BUY0), debug=True)
    u = s.units[0, 0]
    assert u[9] == code("Lê Kiếm", 3)
    assert (u >= 0).sum() == 1
    stars = [e["star"] for e in info["events"] if e["type"] == "merge"]
    assert stars == [2, 3]


def test_merge_prefers_board_slot_even_if_bench_first():
    s = fresh()
    s.level[0, 0] = 3
    give(s, 0, 0, 0, "Mộc Lan")
    give(s, 0, 0, 14, "Mộc Lan")
    set_shop(s, 0, 0, [None, "Mộc Lan", None, None, None])
    s, _ = step(CAT, COMBAT, s, acts(2, p0=BUY0 + 1), debug=True)
    assert s.units[0, 0, 14] == code("Mộc Lan", 2)
    assert (s.units[0, 0] >= 0).sum() == 1


# ---------------------------------------------------------------- shop
def test_roll_skips_exhausted_tier():
    s = fresh()
    for c in CAT.champs_by_tier[0]:
        s.pool[0, c] = 0
    s.shop[0, 0] = -1
    s.level[0, 0] = 1          # cấp 1: 100% bậc 1, mà bậc 1 cạn -> shop trống
    s, _ = step(CAT, COMBAT, s, acts(2, p0=ROLL))
    assert (s.shop[0, 0] == -1).all()
    s.level[0, 0] = 3          # cấp 3: 75% bậc 1 / 25% bậc 2 -> chỉ còn bậc 2
    s.ended_turn[0, 0] = False
    s, _ = step(CAT, COMBAT, s, acts(2, p0=ROLL))
    assert (CAT.cost[s.shop[0, 0]] == 2).all()


def test_roll_costs_and_returns_shop_to_pool():
    s = fresh()
    g = int(s.gold[0, 0])
    s, _ = step(CAT, COMBAT, s, acts(2, p0=ROLL), debug=True)  # debug kiểm tra bảo toàn tướng
    assert s.gold[0, 0] == g - CAT.roll_cost


def test_shop_odds_statistics():
    """Cấp 5: tỷ lệ bậc ~ 45/33/20/2/0."""
    from autochess.core.step import roll_shop
    s = fresh(B=4000, seed=9)
    s.level[:, 0] = 5
    roll_shop(CAT, s, 0, np.arange(4000), 7)
    tiers = CAT.cost[s.shop[:, 0].ravel()] - 1
    freq = np.bincount(tiers, minlength=5) / tiers.size
    assert np.allclose(freq, CAT.shop_odds[4], atol=0.02), freq


# ---------------------------------------------------------------- kinh tế
def _end_round(s, P=2):
    s, info = step(CAT, COMBAT, s, np.full((s.B, P), END_TURN), debug=True)
    assert info["combat"].all()
    return s, info


def test_interest_at_47_gold():
    s = fresh()
    s.gold[0, 0] = 47
    s, _ = _end_round(s)
    # thu nhập = 5 + lãi 4 + chuỗi (sau trận chuỗi là +-1 -> thưởng 0)
    assert s.gold[0, 0] == 47 + 5 + 4


def test_interest_capped():
    s = fresh()
    s.gold[0, 0] = 90
    s, _ = _end_round(s)
    assert s.gold[0, 0] == 90 + 5 + 5


def test_loss_streak_bonus():
    s = fresh()
    s.gold[0, 0] = 0
    s.streak[0, 0] = -4
    strong_board(s, 0, 1)
    ok(s)
    s, _ = _end_round(s)
    assert s.streak[0, 0] == -5
    assert s.gold[0, 0] == 5 + 0 + 3


def test_round_xp():
    s = fresh()
    s, _ = _end_round(s)
    assert s.level[0, 0] == 2 and s.xp[0, 0] == 0   # +2 XP ở đầu vòng 2 -> lên cấp 2


def test_buy_xp_multi_level_and_cap():
    s = fresh()
    s, info = step(CAT, COMBAT, s, acts(2, p0=BUY_XP), debug=True)
    assert s.level[0, 0] == 3 and s.xp[0, 0] == 0
    s.level[0, 0], s.gold[0, 0] = 9, 50
    m = legal_actions(CAT, s)
    assert not m[0, 0, BUY_XP]
    assert "tối đa" in explain(CAT, s, 0, 0, BUY_XP)


def test_sell_two_star_refund():
    s = fresh()
    give(s, 0, 0, 3, "Lê Kiếm", 2)
    pool_before = int(s.pool[0, name_id("Lê Kiếm")])
    g = int(s.gold[0, 0])
    s, _ = step(CAT, COMBAT, s, acts(2, p0=SELL0 + 3), debug=True)
    assert s.gold[0, 0] == g + 3
    assert s.pool[0, name_id("Lê Kiếm")] == pool_before + 3


# ---------------------------------------------------------------- mặt nạ
def test_buy_masked_when_bench_full():
    s = fresh()
    for j, n in enumerate(["Lê Kiếm", "Mộc Lan", "Hỏa Nhi", "Thạch Sơn", "Phong Ảnh",
                           "Lam Tuyết", "Thiết Trụ", "Ưng Nhãn", "Bạch Hổ"]):
        give(s, 0, 0, j, n)
    m, parts = legal_actions(CAT, s, return_parts=True)
    assert not m[0, 0, BUY0:SELL0].any()
    assert "hàng chờ đầy" in explain(CAT, s, 0, 0, BUY0)
    assert not m[0, 0, MOVE0 + 9:ROLL].any()   # sân trống, không có gì để xuống


def test_move_up_masked_when_board_full():
    s = fresh()
    give(s, 0, 0, 9, "Lê Kiếm")
    give(s, 0, 0, 0, "Mộc Lan")
    m = legal_actions(CAT, s)
    assert not m[0, 0, MOVE0 + 0]        # cấp 1, sân đã có 1
    assert m[0, 0, MOVE0 + 9]            # xuống được
    assert "sân đã đủ" in explain(CAT, s, 0, 0, MOVE0 + 0)


def test_illegal_action_raises_in_debug_and_is_skipped_otherwise():
    s = fresh()
    with pytest.raises(IllegalActionError):
        step(CAT, COMBAT, s.copy(), acts(2, p0=SELL0 + 4), debug=True)
    s2, _ = step(CAT, COMBAT, s.copy(), acts(2, p0=SELL0 + 4, p1=ROLL), debug=False)
    assert s2.ended_turn[0, 0]           # bị thay bằng kết thúc lượt


def test_only_end_turn_after_ending():
    s = fresh()
    s, _ = step(CAT, COMBAT, s, acts(2, p0=END_TURN, p1=ROLL))
    m = legal_actions(CAT, s)
    assert m[0, 0].sum() == 1 and m[0, 0, END_TURN]
    assert not s.ended_turn[0, 1]


def test_action_cap_auto_ends_turn():
    s = fresh()
    s.level[0, 0] = 1
    give(s, 0, 0, 0, "Lê Kiếm")
    s, _ = step(CAT, COMBAT, s, acts(2, p0=MOVE0 + 0, p1=END_TURN), debug=True)
    for t in range(1, CAT.max_actions_per_round):
        assert s.round[0] == 1
        src = int(np.flatnonzero(s.units[0, 0] >= 0)[0])
        s, info = step(CAT, COMBAT, s, acts(2, p0=MOVE0 + src), debug=True)
    assert s.round[0] == 2, "chạm 30 hành động -> tự kết thúc lượt -> giao tranh"
    assert any(e["type"] == "end_turn" and e["auto"] for e in info["events"])


# ---------------------------------------------------------------- giao tranh
def test_win_prob_antisymmetric_and_damage_formula():
    s = fresh(P=4, B=3)
    strong_board(s, 0, 1)
    give(s, 1, 2, 9, "Lê Kiếm")
    win, dmg = COMBAT.evaluate(CAT, s)
    off = ~np.eye(4, dtype=bool)
    assert np.allclose((win + win.transpose(0, 2, 1))[:, off], 1)
    nb = board_features(CAT, s)["n_board"]
    for b in range(3):
        for i in range(4):
            for j in range(4):
                if i != j:
                    exp = CAT.stage_damage(s.round[b]) + np.ceil(nb[b, j] * win[b, j, i] - 1e-6)
                    assert dmg[b, i, j] == exp


def test_empty_board_strength_is_one():
    s = fresh()
    f = board_features(CAT, s)
    assert (f["strength"] == 1).all()
    win, _ = COMBAT.evaluate(CAT, s)
    assert np.isclose(win[0, 0, 1], 0.5)


def test_trait_counts_distinct_champions():
    s = fresh()
    s.level[0, 0] = 9
    give(s, 0, 0, 9, "Lê Kiếm")
    give(s, 0, 0, 10, "Lê Kiếm")          # trùng tướng: chỉ đếm 1
    give(s, 0, 0, 11, "Thạch Sơn")
    f = board_features(CAT, s)
    dau_si = CAT.trait_names.index("Đấu sĩ")
    assert f["trait_count"][0, 0, dau_si] == 2
    assert f["trait_level"][0, 0, dau_si] == 1


def test_simultaneous_elimination_order_by_hp():
    s = fresh(P=4)
    s.hp[0, 0], s.hp[0, 1] = 3, 2
    give(s, 0, 1, 4, "Mộc Lan")           # quân ở hàng chờ cũng phải về kho
    strong_board(s, 0, 2)
    strong_board(s, 0, 3, names=("Long Vương", "Thiên Cung", "Hắc Phong", "Ngân Giáp"))
    s.last_opponent[0, 0], s.last_opponent[0, 1] = 1, 0   # cấm 0 gặp 1
    s.last_opponent[0, 2], s.last_opponent[0, 3] = 3, 2
    ok(s)
    s, info = step(CAT, COMBAT, s, np.full((1, 4), END_TURN), debug=True)
    assert not s.alive[0, 0] and not s.alive[0, 1]
    assert s.placement[0, 0] == 3 and s.placement[0, 1] == 4
    assert (s.units[0, :2] == -1).all() and (s.shop[0, :2] == -1).all()


def test_game_ends_last_standing_and_auto_reset():
    s = fresh()
    s.hp[0, 0] = 1
    strong_board(s, 0, 1)
    s, info = _end_round(s)
    assert s.done[0]
    assert s.placement[0].tolist() == [2, 1]
    seed = int(s.seed[0])
    s, info = step(CAT, COMBAT, s, acts(2, p0=ROLL), debug=True)  # hành động bị bỏ qua
    assert info["reset"][0] and info["final_placement"][0].tolist() == [2, 1]
    assert not s.done[0] and s.round[0] == 1 and int(s.seed[0]) != seed
    assert (s.hp[0] == 100).all() and s.gold[0, 0] == CAT.start_gold


def test_max_rounds_ranks_by_hp():
    s = fresh(P=3)
    s.round[0] = CAT.max_rounds
    s.hp[0] = [40, 90, 60]
    s, info = step(CAT, COMBAT, s, np.full((1, 3), END_TURN), debug=True)
    assert s.done[0]
    order = np.argsort(-s.hp[0].astype(int), kind="stable")
    assert [int(s.placement[0, i]) for i in order] == [1, 2, 3]


# ---------------------------------------------------------------- ghép cặp
def test_matching_counts():
    t4 = matchings_table(4)
    assert len(t4[0b1111][0]) == 3
    t3 = matchings_table(3)
    assert len(t3[0b111][0]) == 6 and t3[0b111][1].any(1).all()   # luôn có 1 trận ghost
    t8 = matchings_table(8)
    assert len(t8[0xFF][0]) == 105
    assert len(t8[0x7F][0]) == 7 * 6 * 15


def test_two_players_always_meet():
    s = fresh(P=2, B=5)
    mp = meet_prob(s)
    assert np.allclose(mp[:, 0, 1], 1) and np.allclose(mp[:, 1, 0], 1)


def test_no_rematch_with_four_alive():
    s = fresh(P=4, B=200, seed=5)
    s.last_opponent[:] = np.array([1, 0, 3, 2])
    opp, ghost = draw_pairs(s, np.ones(200, bool))
    assert not (opp == s.last_opponent).any()
    assert not ghost.any()
    mp = meet_prob(s)
    assert np.allclose(mp[:, 0, 1], 0) and np.allclose(mp[:, 0, 2], 0.5)
    assert np.allclose(mp.sum(-1), 1)


def test_meet_prob_matches_empirical_with_ghost():
    s = fresh(P=5, B=1, seed=1)
    s.last_opponent[0] = [1, 0, 3, 2, -1]
    mp = meet_prob(s)[0]
    counts = np.zeros((5, 5))
    for r in range(1, 3001):
        s.round[0] = r
        opp, _ = draw_pairs(s, np.ones(1, bool))
        for i, j in enumerate(opp[0]):
            counts[i, j] += 1
    assert np.allclose(counts / 3000, mp, atol=0.03)
