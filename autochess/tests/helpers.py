"""Tiện ích dựng trạng thái tay cho kiểm thử."""
from __future__ import annotations

import numpy as np

from autochess.core import load_catalog, FormulaCombat, reset, legal_actions, rng
from autochess.core.actions import END_TURN

CAT = load_catalog()
COMBAT = FormulaCombat(catalog=CAT)


def fresh(P=2, B=1, seed=123):
    return reset(CAT, rng.make_seeds(seed, B), P)


def name_id(name: str) -> int:
    return CAT.names.index(name)


def code(name: str, star: int = 1) -> int:
    return name_id(name) * 3 + star - 1


def give(s, b, p, slot, name, star=1):
    """Đặt một quân vào ô `slot`, lấy bản gốc từ kho để giữ bảo toàn."""
    c = name_id(name)
    copies = [1, 3, 9][star - 1]
    assert s.pool[b, c] >= copies
    s.pool[b, c] -= copies
    s.units[b, p, slot] = c * 3 + star - 1


def set_shop(s, b, p, names):
    """Đặt shop của người p (trả shop cũ về kho trước)."""
    for k in range(5):
        old = s.shop[b, p, k]
        if old >= 0:
            s.pool[b, old] += 1
        s.shop[b, p, k] = -1
    for k, n in enumerate(names):
        if n is None:
            continue
        c = name_id(n)
        s.pool[b, c] -= 1
        s.shop[b, p, k] = c


def acts(P, **kw):
    """Hành động [1, P]: mặc định kết thúc lượt; acts(2, p0=3) -> người 0 làm hành động 3."""
    a = np.full((1, P), END_TURN)
    for k, v in kw.items():
        a[0, int(k[1:])] = v
    return a


def deterministic_actions(state, t: int, mask=None):
    """Chọn hành động hợp lệ chỉ phụ thuộc (seed của ván, bước t): giống hệt dù chạy trong lô hay một mình."""
    if mask is None:
        mask = legal_actions(CAT, state)
    B, P, A = mask.shape
    u = rng.uniform(state.seed[:, None, None], rng.BOT, t, np.arange(P)[None, :, None], np.arange(A)[None, None, :])
    u = u + 0.15 * (np.arange(A) == END_TURN)  # nghiêng nhẹ về kết thúc lượt cho ván đi nhanh hơn
    return np.where(mask, u, -1).argmax(-1)
