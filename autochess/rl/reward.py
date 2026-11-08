"""Phần thưởng và chiết khấu theo vòng.

Thưởng chính (cuối ván, hoặc lúc ghế bị loại): thứ hạng chia đều từ +1 (hạng 1) xuống -1 (hạng P).
Thưởng phụ dạng thế năng (Ng, Harada, Russell 1999):

    Phi(s)     = beta * (máu mình - máu trung bình các đối thủ) / máu tối đa      (đối thủ bị loại: máu 0)
    thưởng phụ = gamma_t * Phi(s_{t+1}) - Phi(s_t),   Phi(trạng thái cuối) = 0

Với gamma = 1, tổng thưởng phụ của một ván đúng bằng 0: agent không "ăn" được thưởng phụ, nó chỉ
nhận tín hiệu sớm hơn. gamma_t = gamma_round nếu bước t có giao tranh, = 1 nếu không (các hành
động trong cùng một vòng không chiết khấu lẫn nhau). lambda của GAE cũng theo quy tắc đó.
"""
from __future__ import annotations

import numpy as np


def placement_reward(placement: np.ndarray, P: int) -> np.ndarray:
    """Hạng 1 -> +1, hạng P -> -1, tuyến tính. Tổng của cả sảnh bằng 0."""
    return (1.0 - 2.0 * (np.asarray(placement, np.float32) - 1.0) / (P - 1)).astype(np.float32)


def potential(cat, state, beta: float) -> np.ndarray:
    """Phi[B, P] theo chênh lệch máu (người đã bị loại có máu 0)."""
    hp = state.hp.astype(np.float32)
    P = hp.shape[1]
    opp_mean = (hp.sum(1, keepdims=True) - hp) / (P - 1)
    return (beta * (hp - opp_mean) / float(cat.start_hp)).astype(np.float32)


def step_discounts(info: dict, gamma_round: float, lam: float):
    """gamma_t[B], lambda_t[B]: chỉ trôi ở bước có giao tranh."""
    cb = info["combat"]
    g = np.where(cb, gamma_round, 1.0).astype(np.float32)
    l = np.where(cb, lam, 1.0).astype(np.float32)
    return g, l


class RewardTracker:
    """Tính thưởng cho các ghế được theo dõi (seats[B, S]) quanh một lần gọi step.

        pre = tracker.before(state, seats)          # trước step (step ghi đè state tại chỗ)
        state, info = step(...)
        r, done, g, l = tracker.after(state, info, pre)
    """

    def __init__(self, cat, P: int, gamma_round: float = 0.995, lam: float = 0.95, beta: float = 0.1):
        self.cat, self.P = cat, P
        self.gamma_round, self.lam, self.beta = gamma_round, lam, beta

    def before(self, state, seats: np.ndarray) -> dict:
        b = np.arange(state.B)[:, None]
        used = seats >= 0
        ss = np.where(used, seats, 0)
        active = used & state.alive[b, ss] & ~state.done[:, None]
        phi = potential(self.cat, state, self.beta)[b, ss] if self.beta else np.zeros(seats.shape, np.float32)
        return dict(ss=ss, active=active, phi=np.where(active, phi, 0).astype(np.float32))

    def after(self, state, info: dict, pre: dict):
        b = np.arange(state.B)[:, None]
        ss, active = pre["ss"], pre["active"]
        place = state.placement[b, ss]
        done = active & (place > 0)
        r = np.where(done, placement_reward(np.maximum(place, 1), self.P), 0).astype(np.float32)
        g, l = step_discounts(info, self.gamma_round, self.lam)
        if self.beta:
            phi_next = np.where(done, 0, potential(self.cat, state, self.beta)[b, ss])
            r = r + np.where(active, g[:, None] * phi_next - pre["phi"], 0)
        S = ss.shape[1]
        return (r.astype(np.float32), done, np.repeat(g[:, None], S, 1), np.repeat(l[:, None], S, 1))
