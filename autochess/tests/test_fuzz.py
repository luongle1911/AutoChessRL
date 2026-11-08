"""Kiểm thử ngẫu nhiên: bot ngẫu nhiên, bất biến sau mỗi bước, tất định, nhất quán theo lô.

Số ván mặc định nhỏ để chạy nhanh. Chạy đủ 10.000 ván như thiết kế:
    FUZZ_GAMES=10000 python -m pytest autochess/tests/test_fuzz.py -q
"""
import os

import numpy as np
import pytest

from autochess.core import reset, step, check_invariants, state_hash, legal_actions, rng
from autochess.core.actions import END_TURN
from autochess.bots.random_bot import random_actions

from .helpers import CAT, COMBAT, deterministic_actions

FUZZ_GAMES = int(os.environ.get("FUZZ_GAMES", "256"))


@pytest.mark.parametrize("P", [2, 3, 4, 8])
def test_random_games_keep_invariants(P):
    B = 64
    games = FUZZ_GAMES if P == 2 else max(64, FUZZ_GAMES // 4)
    s = reset(CAT, rng.make_seeds(100 + P, B), P)
    g = np.random.default_rng(P)
    finished, steps = 0, 0
    places = []
    while finished < games:
        m = legal_actions(CAT, s)
        a = random_actions(m, g, end_bias=0.05 if steps % 2 else 0.0)
        s, info = step(CAT, COMBAT, s, a, debug=True)      # debug: bất biến sau mỗi bước
        if info["reset"].any():
            places.append(info["final_placement"][info["reset"]])
        finished += int(info["reset"].sum())
        steps += 1
        assert steps < 200_000
    pl = np.concatenate(places)
    assert (np.sort(pl, 1) == np.arange(1, P + 1)).all()


def _run(seeds, P, T):
    s = reset(CAT, seeds, P)
    for t in range(T):
        s, _ = step(CAT, COMBAT, s, deterministic_actions(s, t))
    return s


def test_deterministic_same_seed_same_hash():
    seeds = rng.make_seeds(7, 4)
    a = _run(seeds, 3, 400)
    b = _run(seeds, 3, 400)
    assert state_hash(a) == state_hash(b)


def test_batch_consistency():
    """Ván b trong lô B cho kết quả giống hệt khi chạy riêng với B = 1."""
    P, T = 4, 600
    seeds = rng.make_seeds(11, 8)
    batch = _run(seeds, P, T)
    for b in range(8):
        alone = _run(seeds[b:b + 1], P, T)
        assert state_hash(batch, b) == state_hash(alone, 0), f"ván {b} lệch"


def test_games_reach_the_end():
    s = reset(CAT, rng.make_seeds(3, 32), 2)
    g = np.random.default_rng(0)
    ended = np.zeros(32, bool)
    for _ in range(5000):
        a = random_actions(legal_actions(CAT, s), g, end_bias=0.3)
        s, info = step(CAT, COMBAT, s, a)
        ended |= info["reset"]
        if ended.all():
            break
    assert ended.all()
