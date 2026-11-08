import copy
import json

import numpy as np
import pytest

from autochess.core.catalog import build_catalog, load_catalog, DEFAULT_CATALOG, CatalogError

from .helpers import CAT


def test_shapes():
    assert CAT.N == 20 and CAT.K == 6
    assert [len(t) for t in CAT.champs_by_tier] == [5, 5, 4, 3, 3]
    assert CAT.power.shape == (20, 3)
    assert CAT.shop_odds.shape == (9, 5)
    assert np.allclose(CAT.shop_odds.sum(1), 1)
    assert CAT.xp_to_next[-1] == 0


def test_counter_antisymmetric():
    assert np.allclose(CAT.counter, -CAT.counter.T)


def test_rejects_broken_counter():
    raw = json.load(open(DEFAULT_CATALOG, encoding="utf-8"))
    bad = copy.deepcopy(raw)
    bad["counter_matrix"][0][1] = 0.5  # không còn phản đối xứng
    with pytest.raises(CatalogError):
        build_catalog(bad)


def test_rejects_bad_odds():
    raw = json.load(open(DEFAULT_CATALOG, encoding="utf-8"))
    bad = copy.deepcopy(raw)
    bad["shop_odds"][3][0] += 1
    with pytest.raises(CatalogError):
        build_catalog(bad)


def test_pool_sizes():
    p2 = CAT.pool_size(2)
    assert [int(p2[CAT.champs_by_tier[t][0]]) for t in range(5)] == [12, 10, 8, 6, 5]
    p8 = CAT.pool_size(8)
    assert [int(p8[CAT.champs_by_tier[t][0]]) for t in range(5)] == [29, 22, 18, 12, 10]
    p5 = CAT.pool_size(5)
    assert (p2 <= p5).all() and (p5 <= p8).all()


def test_every_trait_reachable_to_top_breakpoint():
    counts = CAT.trait_matrix.sum(0)
    assert (counts >= CAT.trait_breakpoints[:, -1]).all(), counts


def test_stage_damage():
    assert CAT.stage_damage(1) == 2 and CAT.stage_damage(6) == 5 and CAT.stage_damage(50) == 12
