"""Kiểm thử đáp án mẫu: mọi tối ưu lõi phải cho kết quả GIỐNG HỆT bản gốc.

golden.json lưu mã băm trạng thái của từng ván (seed cố định, hành động tất định) tại vài
mốc bước, cho P = 2 và P = 8. Tạo lại đáp án mẫu (chỉ khi CỐ Ý đổi luật):
    GOLDEN_REGEN=1 python3 -m pytest autochess/tests/test_golden.py -q
"""
import json
import os

import numpy as np
import pytest

from autochess.core import reset, step, state_hash, rng

from .helpers import CAT, COMBAT, deterministic_actions

GOLDEN = os.path.join(os.path.dirname(__file__), "golden.json")
CASES = {"P2": (2, 50, 900, 11), "P8": (8, 50, 1500, 12)}   # P, số ván, số bước, seed gốc
CHECKPOINTS = (150, 400, 900, 1500)


def run_case(P, B, T, base):
    s = reset(CAT, rng.make_seeds(base, B), P)
    out, placements = {}, []
    for t in range(T):
        s, info = step(CAT, COMBAT, s, deterministic_actions(s, t))
        if info["reset"].any():
            placements.append(info["final_placement"][info["reset"]].tolist())
        if t + 1 in CHECKPOINTS:
            out[str(t + 1)] = [state_hash(s, b) for b in range(B)]
    out["placements"] = placements
    return out


def _regen():
    data = {k: run_case(*v) for k, v in CASES.items()}
    with open(GOLDEN, "w") as f:
        json.dump(data, f)
    return data


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_golden(case):
    if os.environ.get("GOLDEN_REGEN") or not os.path.exists(GOLDEN):
        _regen()
    with open(GOLDEN) as f:
        want = json.load(f)[case]
    got = run_case(*CASES[case])
    assert got["placements"] == want["placements"], "thứ hạng các ván đã xong bị lệch"
    for k in want:
        if k == "placements":
            continue
        bad = [b for b, (x, y) in enumerate(zip(got[k], want[k])) if x != y]
        assert not bad, f"bước {k}: {len(bad)} ván lệch, ví dụ ván {bad[:5]}"


def test_step_with_precomputed_mask_is_identical():
    """step(..., mask=m) phải cho kết quả giống hệt step tự tính mặt nạ."""
    from autochess.core import legal_actions
    seeds = rng.make_seeds(5, 32)
    a, b = reset(CAT, seeds, 4), reset(CAT, seeds, 4)
    for t in range(800):
        m = legal_actions(CAT, a)
        act = deterministic_actions(a, t, m)
        a, _ = step(CAT, COMBAT, a, act, mask=m)
        b, _ = step(CAT, COMBAT, b, act)
    assert state_hash(a) == state_hash(b)
