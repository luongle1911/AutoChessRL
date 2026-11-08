"""Sinh các file dữ liệu biến thể (khác số tướng N, số tộc K, thứ tự id) từ catalog_v1 cho kiểm thử."""
from __future__ import annotations

import copy

import numpy as np

from autochess.core.catalog import build_catalog

from .helpers import CAT


def variant_raw(n_champs: int = 20, n_traits: int = 6, permute: bool = False, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    raw = copy.deepcopy(CAT.raw)
    K0 = len(raw["traits"])
    champs = raw["champions"]
    traits = raw["traits"]
    C = np.asarray(raw["counter_matrix"], np.float64)

    # ---- số tộc
    if n_traits < K0:
        traits = traits[:n_traits]
        C = C[:n_traits, :n_traits]
        for c in champs:
            c["traits"] = [k for k in c["traits"] if k < n_traits] or [c["id"] % n_traits]
    elif n_traits > K0:
        for k in range(K0, n_traits):
            traits.append({"id": k, "name": f"Tộc thêm {k}", "breakpoints": [2, 4, 6], "bonus": [0.1, 0.25, 0.45]})
        newC = np.zeros((n_traits, n_traits))
        newC[:K0, :K0] = C
        A = np.triu(rng.uniform(-0.2, 0.2, (n_traits, n_traits)), 1)
        A[:K0, :K0] = 0
        C = newC + A - A.T
        new = list(range(K0, n_traits))
        for i, c in enumerate(champs):          # mỗi tộc mới có vài tướng
            k = new[i % len(new)]
            c["traits"] = (c["traits"][:1] + [k]) if i % 3 else [k]

    # ---- số tướng
    N0 = len(champs)
    if n_champs < N0:
        by_cost = {}
        for c in champs:
            by_cost.setdefault(c["cost"], []).append(c)
        drop = set()
        while N0 - len(drop) > n_champs:      # bỏ dần ở bậc đông nhất, bậc nào cũng còn ít nhất 1
            cost = max(by_cost, key=lambda k: len([c for c in by_cost[k] if c["name"] not in drop]))
            left = [c for c in by_cost[cost] if c["name"] not in drop]
            drop.add(left[-1]["name"])
        champs = [c for c in champs if c["name"] not in drop]
    for i in range(n_champs - len(champs)):
        cost = 1 + i % 5
        base = 10 * (1.6 ** (cost - 1))
        champs.append({"id": -1, "name": f"Tướng thêm {i}", "cost": cost,
                       "traits": sorted({i % n_traits, (i + 2) % n_traits}),
                       "power": [round(base, 1), round(base * 1.8, 1), round(base * 3.2, 1)]})

    # ---- đánh lại id (theo giá, hoặc ngẫu nhiên nếu permute)
    K = len(traits)
    tperm = rng.permutation(K) if permute else np.arange(K)      # tperm[mới] = cũ
    inv = np.argsort(tperm)                                      # inv[cũ] = mới
    traits = [dict(traits[int(o)], id=j) for j, o in enumerate(tperm)]
    C = C[np.ix_(tperm, tperm)]
    for c in champs:
        c["traits"] = [int(inv[k]) for k in c["traits"]]
    order = rng.permutation(len(champs)) if permute else sorted(range(len(champs)), key=lambda i: champs[i]["cost"])
    champs = [dict(champs[int(i)], id=j) for j, i in enumerate(order)]

    raw.update(champions=champs, traits=traits, counter_matrix=C.tolist(),
               version=f"test-N{len(champs)}-K{K}{'-perm' if permute else ''}")
    return raw


def variant(n_champs: int = 20, n_traits: int = 6, permute: bool = False, seed: int = 0):
    return build_catalog(variant_raw(n_champs, n_traits, permute, seed))


def translate_state(state, cat_a, cat_b):
    """Chép trạng thái từ catalog A sang catalog B có cùng tên tướng nhưng khác id."""
    m = np.array([cat_b.names.index(n) for n in cat_a.names])
    s = state.copy()
    u = state.units.astype(np.int64)
    s.units[...] = np.where(u >= 0, m[np.clip(u, 0, None) // 3] * 3 + u % 3, -1)
    sh = state.shop.astype(np.int64)
    s.shop[...] = np.where(sh >= 0, m[np.clip(sh, 0, None)], -1)
    s.pool[:, m] = state.pool
    return s
