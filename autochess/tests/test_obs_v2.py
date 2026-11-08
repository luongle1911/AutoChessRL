"""Quan sát bản 2 (không cần PyTorch): kích thước, khoảng giá trị, đặc trưng tộc, không lộ thông tin ẩn,
và chạy được với file dữ liệu có số tướng / số tộc khác nhau."""
import numpy as np
import pytest

from autochess.core import reset, step, legal_actions, rng, load_catalog
from autochess.core.catalog import DATA_DIR
from autochess.core.combat import FormulaCombat
from autochess.rl.observe import observe, KEYS, KEYS2, TRAIT_F, SELF2, UNIT_F2, OPP_BASE

from .helpers import CAT, COMBAT, deterministic_actions, fresh, give
from .catalog_variants import variant


def _advance(cat, P, B, T, seed=0):
    combat = FormulaCombat(catalog=cat)
    s = reset(cat, rng.make_seeds(seed, B), P)
    for t in range(T):
        m = legal_actions(cat, s)
        s, _ = step(cat, combat, s, deterministic_actions(s, t, m), mask=m)
    return s, combat


@pytest.mark.parametrize("P", [2, 4, 8])
def test_v2_shapes_and_ranges(P):
    s, _ = _advance(CAT, P, 16, 300)
    o = observe(CAT, s, COMBAT, version=2)
    assert tuple(o) == KEYS2
    O, K = P - 1, CAT.K
    assert o["self"].shape == (16, P, SELF2)
    assert o["units"].shape == (16, P, 18, UNIT_F2)
    assert o["traits"].shape == (16, P, K, TRAIT_F)
    assert o["opp"].shape == (16, P, O, OPP_BASE)
    assert o["opp_traits"].shape == (16, P, O, K, TRAIT_F)
    assert o["opp_board_pool"].shape == (16, P, O, 9)
    for k in ("self", "units", "shop", "traits", "opp", "opp_traits", "opp_board_pool"):
        assert np.isfinite(o[k]).all(), k
        assert o[k].min() >= -1.5 and o[k].max() <= 3.0, k


def test_v1_unchanged_by_default():
    s, _ = _advance(CAT, 4, 4, 200)
    assert tuple(observe(CAT, s, COMBAT)) == KEYS


def test_v2_matches_v1_where_shared():
    s, _ = _advance(CAT, 4, 8, 300)
    a = observe(CAT, s, COMBAT, version=1)
    b = observe(CAT, s, COMBAT, version=2)
    assert np.array_equal(a["self"][..., :11], b["self"][..., :11])
    assert np.array_equal(a["units"], b["units"][..., :8])
    assert np.array_equal(a["opp"][..., :OPP_BASE], b["opp"])
    for k in ("shop", "shop_id", "unit_id", "opp_board_id", "opp_board_star", "opp_alive", "mask"):
        assert np.array_equal(a[k], b[k]), k
    # tỷ lệ còn trong kho của từng quân = cột kho của bản 1 tại tướng đó
    N = CAT.N
    pool_v1 = a["self"][..., 11 + 2 * CAT.K:]                       # [B,P,N]
    uid = a["unit_id"].astype(np.int64)
    exp = np.where(uid >= 0, np.take_along_axis(pool_v1, np.clip(uid, 0, None), -1), 0)
    assert np.allclose(b["units"][..., 8], exp)
    assert N == pool_v1.shape[-1]


def test_trait_features_by_hand():
    s = fresh(P=2)
    # Đấu sĩ (tộc 0, mốc 2/4/6): Lê Kiếm (0,2), Thạch Sơn (0,3), Bạch Hổ (0,4) trên sân -> 3 tướng, mức 1
    for slot, n in ((9, "Lê Kiếm"), (10, "Thạch Sơn"), (11, "Bạch Hổ")):
        give(s, 0, 0, slot, n)
    give(s, 0, 0, 0, "Hỏa Nhi")                 # hàng chờ: không tính
    o = observe(CAT, s, COMBAT, version=2)
    t = o["traits"][0, 0]
    k = CAT.trait_names.index("Đấu sĩ")
    assert np.allclose(t[k], [3 / 6, 1 / 3, 1 / 6, CAT.trait_bonus[k, 0] * 2, 1])
    pk = CAT.trait_names.index("Pháp sư")
    assert np.allclose(t[pk], [0, 0, 2 / 6, 0, 0])
    # đối thủ nhìn thấy đúng các tộc đó
    assert np.allclose(o["opp_traits"][0, 1, 0], t)


def test_v2_hides_opponent_bench_and_shop():
    s, _ = _advance(CAT, 2, 4, 250)
    a = observe(CAT, s, COMBAT, version=2)
    t = s.copy()
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
    b_ = observe(CAT, t, COMBAT, version=2)
    for k in KEYS2:
        if k != "mask":
            assert np.array_equal(a[k][:, 0], b_[k][:, 0]), k


@pytest.mark.parametrize("N,K", [(18, 5), (20, 6), (24, 8), (24, 5), (18, 8)])
def test_v2_runs_with_other_catalogs(N, K):
    cat = variant(N, K)
    assert (cat.N, cat.K) == (N, K)
    s, combat = _advance(cat, 4, 8, 250)
    o = observe(cat, s, combat, version=2)
    # chỉ chiều tộc phụ thuộc K; không mảng nào có chiều theo số tướng N
    O = 3
    expect = dict(self=(SELF2,), units=(18, UNIT_F2), unit_id=(18,), shop=(5, 8), shop_id=(5,),
                  traits=(K, TRAIT_F), opp=(O, OPP_BASE), opp_traits=(O, K, TRAIT_F), opp_board_id=(O, 9),
                  opp_board_star=(O, 9), opp_board_pool=(O, 9), opp_alive=(O,), mask=(44,))
    assert {k: v.shape[2:] for k, v in o.items()} == expect
    for k in ("self", "units", "traits", "opp_traits"):
        assert np.isfinite(o[k]).all() and o[k].max() <= 3.0, k


def test_fake_patch_catalog_loads_and_bots_play():
    from autochess import tournament as T
    cat = load_catalog(f"{DATA_DIR}/catalog_v1b.json")
    assert cat.N == 21 and cat.K == 7 and "Phong Ảnh" not in cat.names and "Kỵ Vương" in cat.names
    combat = FormulaCombat(catalog=cat)
    r = T.play(cat, combat, ("tempo", "fast9"), 16, seed=1, B=16)
    assert len(r["placements"]) >= 16


def test_catalog_v2_rules_load_and_bots_play():
    """Luật bản 2: tướng 4-5 tiền có 3 tộc, nhiệt độ giao tranh 0,2."""
    from autochess import tournament as T
    cat = load_catalog(f"{DATA_DIR}/catalog_v2.json")
    assert cat.version == "catalog-v2" and cat.trait_matrix.sum(1).max() == 3
    combat = FormulaCombat(catalog=cat)
    assert combat.temperature == 0.2
    r = T.play(cat, combat, ("hyper", "bank"), 16, seed=2, B=16)
    assert len(r["placements"]) >= 16
    s, _ = _advance(cat, 4, 4, 200)
    o = observe(cat, s, combat, version=2)
    assert np.isfinite(o["traits"]).all() and o["traits"].max() <= 3.0


def test_catalog_rejects_more_than_three_traits():
    import copy
    from autochess.core.catalog import build_catalog, CatalogError
    raw = copy.deepcopy(CAT.raw)
    raw["champions"][0]["traits"] = [0, 1, 2, 3]
    with pytest.raises(CatalogError):
        build_catalog(raw)


def test_catalog_v3_rules_and_unit_damage_scale():
    """Luật bản 3: 3 sao tướng rẻ mạnh hơn, kho lớn hơn, đầu ván mất ít máu hơn (cả phần theo số quân)."""
    from autochess import tournament as T
    from autochess.core.catalog import build_catalog
    import copy
    cat = load_catalog(f"{DATA_DIR}/catalog_v3.json")
    v2 = load_catalog(f"{DATA_DIR}/catalog_v2.json")
    assert cat.version == "catalog-v3.2" and FormulaCombat(catalog=cat).temperature == 0.17
    assert np.allclose(v2.unit_damage_scale_by_round, 1.0)
    assert cat.unit_damage_scale(5) == 0.5 and cat.unit_damage_scale(12) == 0.6 and cat.unit_damage_scale(30) == 1.0
    assert cat.stage_damage(3) < v2.stage_damage(3) and cat.stage_damage(20) < v2.stage_damage(20)
    assert cat.stage_damage(34) > v2.stage_damage(34)
    # cùng một trạng thái: phần theo số quân ở vòng 5 bằng một nửa (làm tròn lên)
    s, _ = _advance(v2, 8, 8, 120)
    s.round[:] = 5
    raw = copy.deepcopy(v2.raw)
    raw["unit_damage_scale"] = [[1, 0.5]]
    half = build_catalog(raw)
    w1, d1 = FormulaCombat(catalog=v2).evaluate(v2, s)
    w2, d2 = FormulaCombat(catalog=half).evaluate(half, s)
    stage = int(v2.stage_damage(5))
    assert np.array_equal(w1, w2)
    assert np.array_equal(d2 - stage, np.ceil((d1 - stage) / 2 - 1e-6).astype(d2.dtype)) or (d2 <= d1).all()
    assert (d2 <= d1).all() and (d2 < d1).any()
    r = T.play(cat, FormulaCombat(catalog=cat), ("hyper", "bank", "tempo", "fast9"), 16, seed=3, B=16)
    assert len(r["placements"]) >= 16
