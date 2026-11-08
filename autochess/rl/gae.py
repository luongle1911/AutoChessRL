"""GAE theo vòng, với các bước "trong suốt".

Mọi mảng có dạng [T, N] (N = số ván x số ghế agent, đã trải phẳng).
    active[t]   ghế còn trong ván ở bước t (còn sống, ván chưa xong). Bước không active (đã bị loại,
                đang chờ ván reset, ô ghế không dùng) là trong suốt: không có advantage, và chuỗi tính
                đi xuyên qua nó.
    dones[t]    tập của ghế kết thúc ngay sau bước t (bị loại hoặc hết ván): không nhìn sang bước sau.
    gammas[t], lambdas[t]  chỉ khác 1 ở bước có giao tranh (xem reward.py).
last_values[N]  giá trị critic ở trạng thái sau bước cuối (0 nếu ghế không active ở đó).

    delta_t = r_t + gamma_t * V(next) * (1 - done_t) - V_t
    A_t     = delta_t + gamma_t * lambda_t * (1 - done_t) * A_next
"""
from __future__ import annotations

import numpy as np


def compute_gae(rewards, values, dones, gammas, lambdas, active, last_values):
    T, N = rewards.shape
    adv = np.zeros((T, N), np.float32)
    last_gae = np.zeros(N, np.float32)
    next_v = np.asarray(last_values, np.float32).copy()
    for t in range(T - 1, -1, -1):
        a = active[t]
        nonterm = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gammas[t] * next_v * nonterm - values[t]
        gae = delta + gammas[t] * lambdas[t] * nonterm * last_gae
        adv[t] = np.where(a, gae, 0.0)
        last_gae = np.where(a, gae, last_gae)
        next_v = np.where(a, values[t], next_v)
    returns = np.where(active, adv + values, 0.0).astype(np.float32)
    return adv, returns
