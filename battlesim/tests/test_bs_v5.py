"""Luật bản 5, phần giao tranh: khối hiệu ứng, trình mô phỏng nấc 2 (sim2), dữ liệu catalog_v5, trình xem lại."""
import copy

import numpy as np
import pytest

from battlesim.team import CombatData, Unit
from battlesim.effects import compile_effects, describe, ST_AS
from battlesim.sim2 import F_DEALT, F_HP, I_CELL, I_ATKS, RES_SIDE0, RES_SIDE1
from battlesim import hexgrid

DATA = CombatData.load()


# ------------------------------------------------------------------ dữ liệu
def test_catalog_v5_shape():
    assert DATA.n_shop == 59 and DATA.N == 67 and DATA.K == 24 and len(DATA.items) == 44
    comb = [i for i in DATA.items if i["kind"] == "combined"]
    assert len(comb) == 36 and len({tuple(sorted(i["recipe"])) for i in comb}) == 36
    assert all(len(DATA.skill_rows[c]) >= 1 for c in range(DATA.n_shop))
    for r in DATA.rows:            # mọi khối có mô tả đọc được
        assert describe(r)
    # tướng đắt có kỹ năng diện rộng hoặc nhiều thành phần hơn tướng rẻ (trung bình số khối)
    n = {c: np.mean([len(DATA.skill_rows[x["id"]]) for x in DATA.champs if x["cost"] == c]) for c in range(1, 6)}
    assert n[5] > n[1]
    assert DATA.combine("Kiếm", "Cung") == DATA.item_id("Diệt Khổng Lồ")
    assert DATA.combine("Găng", "Găng") == DATA.item_id("Găng Thợ Săn")


def test_trait_levels_count_distinct_champions():
    team = [Unit("Hermes", 1, 0), Unit("Hermes", 2, 1), Unit("Tengu", 1, 2)]
    lv = DATA.trait_levels(team)
    assert lv[DATA.trait_id("Sát thủ")] == 0          # 2 tướng khác nhau -> mốc 2
    assert lv[DATA.trait_id("Hy Lạp")] == -1          # 1 Hy Lạp
    lv2 = DATA.trait_levels(team + [Unit("Valkyrie", 1, 3, emblems=["Sát thủ"]),
                                    Unit("Karna", 1, 4, emblems=["Sát thủ"])])
    assert lv2[DATA.trait_id("Sát thủ")] == 1         # thêm 2 ấn -> mốc 4


# ------------------------------------------------------------------ đối xứng, tái lập
def _random_team(rng, n):
    from battlesim.viewer2 import random_team
    return random_team(DATA, rng, n)


def test_swap_sides_mirrors_result():
    rng = np.random.default_rng(1)
    A = [_random_team(rng, 6) for _ in range(6)]
    B = [_random_team(rng, 6) for _ in range(6)]
    sim = DATA.simulator()
    r1 = sim.run(DATA.build_batch(A, B))
    r2 = sim.run(DATA.build_batch(B, A))
    mirror = np.where(r1.result == 2, 2, 1 - r1.result)
    assert (r2.result == mirror).all()
    assert (r1.end_step == r2.end_step).all()


def test_identical_teams_both_lose_or_draw():
    rng = np.random.default_rng(3)
    T = _random_team(rng, 5)
    r = DATA.simulator().run(DATA.build_batch([T], [copy.deepcopy(T)]))
    assert r.result[0] == 2


def test_stochastic_reproducible_by_seed():
    rng = np.random.default_rng(2)
    A, B = _random_team(rng, 6), _random_team(rng, 6)
    sim = DATA.simulator(stochastic=True)
    r1 = sim.run(DATA.build_batch([A, A], [B, B], seeds=[5, 6]))
    r2 = sim.run(DATA.build_batch([A, A], [B, B], seeds=[5, 6]))
    assert (r1.result == r2.result).all() and np.allclose(r1.dealt, r2.dealt)


# ------------------------------------------------------------------ cơ chế, bằng một bộ luật tí hon
def _tiny(attacker_skill=(), dummy_effects=(), attacker_items=(), items=()):
    """Hai tướng: 'Đánh' (công 100, tốc 1, giáp 0) và 'Hình nhân' (máu lớn, không đánh, giáp 0)."""
    raw = copy.deepcopy(DATA.raw)
    st = dict(hp=500, ad=100, ap=100, armor=0, mr=0, range=1, crit_chance=0.0, crit_dmg=1.4, mana_start=0,
              mana_max=0, **{"as": 1.0})
    raw["champions"] = [
        dict(id=0, name="Đánh", cost=1, role="Đấu sĩ", traits=[], stats=dict(st, mana_max=50 if attacker_skill else 0),
             skill=dict(name="x", text="", effects=list(attacker_skill))),
        dict(id=1, name="Hình nhân", cost=1, role="Hộ vệ", traits=[], stats=dict(st, hp=1e6, ad=0, **{"as": 0.0}),
             skill=dict(name="y", text="", effects=list(dummy_effects))),
    ]
    raw["monsters"] = []
    raw["items"] = raw["items"] + [dict(id=44 + i, name=f"đồ{i}", kind="component", effects=list(e))
                                   for i, e in enumerate(items)]
    return CombatData(raw)


def _duel(d, steps, a_items=(), b_items=()):
    sim = d.simulator()
    batch = d.build_batch([[Unit(0, 1, 24, items=list(a_items))]], [[Unit(1, 1, 24, items=list(b_items))]])
    st = sim.init_state(batch)
    for s in range(1, steps + 1):
        sim.step(st, s)
    return st


def test_basic_attack_damage_and_rate():
    st = _duel(_tiny(), 50)                   # 5 giây, tốc đánh 1: đòn ở 1, 2, 3, 4, 5 giây
    assert st.I[0, 0, I_ATKS] == 5
    assert np.isclose(st.F[0, 0, F_DEALT], 500.0)


def test_shield_absorbs_then_hp():
    d = _tiny(dummy_effects=[], items=[[dict(when="start", to="self", do="shield", values=[250], duration=99)]])
    st = _duel(d, 50, b_items=["đồ0"])
    assert np.isclose(1e6 - st.F[0, 10, F_HP], 500.0 - 250.0)


def test_stun_stops_attacks():
    d = _tiny(attacker_skill=[dict(when="cast", to="target", do="stun", values=[3.0])],
              dummy_effects=[])
    # hình nhân không đánh nên cho kẻ đánh tự choáng mình bằng đồ: mỗi đòn thứ 2 tự choáng 1 giây
    d2 = _tiny(items=[[dict(when="attack_n", period=2, to="self", do="stun", values=[1.5])]])
    a = _duel(_tiny(), 100).I[0, 0, I_ATKS]
    b = _duel(d2, 100, a_items=["đồ0"]).I[0, 0, I_ATKS]
    assert b < a


def test_overtime_increases_damage():
    d = _tiny()
    st = _duel(d, 100)
    early = st.F[0, 0, F_DEALT]
    st2 = _duel(d, 300)
    late_10s = st2.F[0, 0, F_DEALT] - _duel(d, 200).F[0, 0, F_DEALT]
    assert np.isclose(early, 1000.0)
    assert late_10s > 2.0 * early             # giây 20-30: hệ số 1 -> 3


def test_wound_cuts_healing():
    heal = [dict(when="periodic", period=1, to="self", do="heal", values=[60])]
    d = _tiny(dummy_effects=[], items=[heal, [dict(when="attack", to="target", do="wound", values=[2], duration=2)]])
    no_wound = _duel(d, 50, b_items=["đồ0"]).F[0, 10, F_HP]
    wound = _duel(d, 50, a_items=["đồ1"], b_items=["đồ0"]).F[0, 10, F_HP]
    assert wound < no_wound


def test_stacking_attack_speed():
    d = _tiny(items=[[dict(when="attack", to="self", do="stat", stat="as", mode="mul", values=[0.1],
                           max_stacks=99)]])
    a = _duel(d, 100).I[0, 0, I_ATKS]
    b = _duel(d, 100, a_items=["đồ0"]).I[0, 0, I_ATKS]
    assert b >= a + 3


def test_skill_cast_uses_mana():
    d = _tiny(attacker_skill=[dict(when="cast", to="target", do="damage", dtype="true", values=[1000])])
    st = _duel(d, 60)                         # 10 năng lượng / đòn, đầy 50 sau 5 đòn -> tung chiêu ở bước 6 giây
    assert st.F[0, 0, F_DEALT] >= 1000.0 + 5 * 100.0 - 1e-3


def test_assassin_dashes_to_backline_at_start():
    A = [Unit("Hermes", 1, 24), Unit("Tengu", 1, 25)]       # 2 Sát thủ: lướt ra sau đầu trận
    B = [Unit("Sobek", 1, 24), Unit("Valkyrie", 1, 3)]      # Valkyrie đứng hàng sau
    sim = DATA.simulator()
    st = sim.init_state(DATA.build_batch([A], [B]))
    val_cell = st.I[0, 11, I_CELL]
    for u in (0, 1):
        assert hexgrid.DIST[st.I[0, u, I_CELL], val_cell] == 1


# ------------------------------------------------------------------ trình xem lại
def test_viewer2_writes_html(tmp_path):
    from battlesim.viewer2 import record, write_html
    rng = np.random.default_rng(4)
    rep = record(DATA, _random_team(rng, 5), _random_team(rng, 5), seed=1)
    p = write_html(rep, tmp_path / "r.html")
    html = p.read_text(encoding="utf-8")
    assert "Năng lượng" in html and len(rep["frames"]) == rep["meta"]["end_step"] + 1
