"""Ghép cặp đấu và xác suất gặp.

Luật: bốc đều ngẫu nhiên trong các cách ghép hợp lệ. Khi còn >= 4 người sống, cách ghép
hợp lệ là cách không để ai gặp lại đối thủ vòng trước. Số người lẻ: một người đấu bản sao
đội hình (ghost) của một người khác. P = 2: luôn gặp nhau.

Mỗi cách ghép là một cặp mảng (opp[P], ghost[P]): opp[i] = j nghĩa là i đấu j (hoặc ghost
của j nếu ghost[i]); -1 cho người đã bị loại. Với P <= 8, mọi cách ghép của mọi tập người
sống được liệt kê sẵn một lần.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import rng


def _perfect_matchings(players: tuple) -> list:
    if not players:
        return [[]]
    a, rest = players[0], players[1:]
    out = []
    for k, b in enumerate(rest):
        for m in _perfect_matchings(rest[:k] + rest[k + 1:]):
            out.append([(a, b)] + m)
    return out


@lru_cache(maxsize=None)
def matchings_table(P: int) -> list:
    """table[alive_code] = (opp[M, P] int8, ghost[M, P] bool)."""
    table = []
    for code in range(1 << P):
        alive = tuple(i for i in range(P) if code >> i & 1)
        rows_opp, rows_ghost = [], []
        if len(alive) >= 2:
            if len(alive) % 2 == 0:
                setups = [(None, None, alive)]
            else:
                setups = [(i, j, tuple(x for x in alive if x != i))
                          for i in alive for j in alive if j != i]
            for gi, gj, rest in setups:
                for m in _perfect_matchings(rest):
                    opp = np.full(P, -1, np.int8)
                    gh = np.zeros(P, bool)
                    for a, b in m:
                        opp[a], opp[b] = b, a
                    if gi is not None:
                        opp[gi], gh[gi] = gj, True
                    rows_opp.append(opp)
                    rows_ghost.append(gh)
        else:
            rows_opp.append(np.full(P, -1, np.int8))
            rows_ghost.append(np.zeros(P, bool))
        opp_arr = np.stack(rows_opp)
        gh_arr = np.stack(rows_ghost)
        opp_arr.setflags(write=False)
        gh_arr.setflags(write=False)
        table.append((opp_arr, gh_arr))
    return table


def _alive_code(alive: np.ndarray) -> np.ndarray:
    P = alive.shape[1]
    return (alive.astype(np.int64) << np.arange(P)).sum(1)


def _valid(opp_m: np.ndarray, last_opp: np.ndarray, n_alive: int) -> np.ndarray:
    """valid[rows, M]: cách ghép m không để ai gặp lại đối thủ vòng trước (nếu >= 4 người sống)."""
    R, M = len(last_opp), len(opp_m)
    if n_alive < 4:
        return np.ones((R, M), bool)
    clash = (opp_m[None, :, :] >= 0) & (opp_m[None, :, :] == last_opp[:, None, :])
    v = ~clash.any(-1)
    none = ~v.any(1)
    v[none] = True  # không có cách nào hợp lệ (không xảy ra với >= 4 người) -> cho phép tất cả
    return v


def _groups(state, rows: np.ndarray):
    codes = _alive_code(state.alive)
    idx = np.flatnonzero(rows)
    for code in np.unique(codes[idx]):
        yield int(code), idx[codes[idx] == code]


def meet_prob(state, rows: np.ndarray | None = None) -> np.ndarray:
    """meet_prob[B, P, P]: xác suất i gặp j (kể cả gặp ghost của j) ở vòng hiện tại."""
    B, P = state.alive.shape
    out = np.zeros((B, P, P), np.float32)
    rows = np.ones(B, bool) if rows is None else rows
    table = matchings_table(P)
    for code, bi in _groups(state, rows):
        opp_m, _ = table[code]
        n_alive = bin(code).count("1")
        v = _valid(opp_m, state.last_opponent[bi], n_alive).astype(np.float32)  # [R, M]
        onehot = (opp_m[:, :, None] == np.arange(P)).astype(np.float32)           # [M, P, P]
        out[bi] = np.einsum("rm,mij->rij", v, onehot) / v.sum(1)[:, None, None]
    return out


def draw_pairs(state, rows: np.ndarray):
    """Bốc cặp đấu cho các ván `rows`. Trả về opp[B, P] int8 (-1 không đấu), ghost[B, P] bool."""
    B, P = state.alive.shape
    opp = np.full((B, P), -1, np.int8)
    ghost = np.zeros((B, P), bool)
    table = matchings_table(P)
    for code, bi in _groups(state, rows):
        opp_m, gh_m = table[code]
        n_alive = bin(code).count("1")
        v = _valid(opp_m, state.last_opponent[bi], n_alive)
        n_valid = v.sum(1)
        u = rng.uniform(state.seed[bi], rng.PAIRING, state.round[bi])
        k = np.minimum((u * n_valid).astype(np.int64), n_valid - 1)
        m = np.argmax(np.cumsum(v, 1) > k[:, None], axis=1)
        opp[bi] = opp_m[m]
        ghost[bi] = gh_m[m]
    return opp, ghost
