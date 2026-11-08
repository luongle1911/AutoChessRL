"""Đối xứng: bắt cả lỗi thiên vị thứ tự lẫn lỗi xoay bàn cờ."""
import numpy as np
import pytest

from battlesim import Simulator, make_batch, swap_sides
from battlesim.layout import U_PER_SIDE

from battlesim.tests.helpers import random_team

FLIP = {0: 1, 1: 0, 2: 2}


def _swap_units(x):
    return np.concatenate([x[:, U_PER_SIDE:], x[:, :U_PER_SIDE]], axis=1)


@pytest.mark.parametrize("level", [0, 1])
def test_swapping_sides_mirrors_result(cfg, np_rng, level):
    n = 200
    ta = [random_team(np_rng, cfg.n_champs, np_rng.integers(1, 9)) for _ in range(n)]
    tb = [random_team(np_rng, cfg.n_champs, np_rng.integers(1, 9)) for _ in range(n)]
    sim = Simulator(cfg, level=level)
    r1 = sim.run(make_batch(ta, tb))
    r2 = sim.run(make_batch(tb, ta))         # đổi phe: B ở phe 0, A ở phe 1
    assert np.array_equal(r2.result, np.vectorize(FLIP.get)(r1.result))
    assert np.array_equal(r2.end_step, r1.end_step)
    assert np.array_equal(_swap_units(r2.alive), r1.alive)
    assert np.allclose(_swap_units(r2.dmg_dealt), r1.dmg_dealt, rtol=1e-6)
    assert np.array_equal(_swap_units(r2.death_step), r1.death_step)


def test_swap_sides_helper_matches(cfg, np_rng):
    ta = [random_team(np_rng, cfg.n_champs, 5) for _ in range(50)]
    tb = [random_team(np_rng, cfg.n_champs, 5) for _ in range(50)]
    sim = Simulator(cfg)
    b = make_batch(ta, tb)
    r1, r2 = sim.run(b), sim.run(swap_sides(b))
    assert np.array_equal(r2.result, np.vectorize(FLIP.get)(r1.result))
    assert np.allclose(r2.dmg_dealt, r1.dmg_dealt, rtol=1e-6)


@pytest.mark.parametrize("level", [0, 1])
def test_identical_teams_never_produce_a_winner(cfg, np_rng, level):
    teams = [random_team(np_rng, cfg.n_champs, np_rng.integers(1, 9)) for _ in range(200)]
    r = Simulator(cfg, level=level).run(make_batch(teams, teams))
    assert np.all(r.result == 2)
    assert np.allclose(r.dmg_dealt[:, :U_PER_SIDE], r.dmg_dealt[:, U_PER_SIDE:], rtol=1e-6)
