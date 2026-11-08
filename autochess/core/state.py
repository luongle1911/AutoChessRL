"""Bộ mảng trạng thái của B ván. Không có đối tượng tướng hay người chơi."""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .catalog import Catalog, NUM_SLOTS, SHOP_SIZE, MAX_PLAYERS


@dataclass(eq=False)
class State:
    units: np.ndarray               # [B, P, 18] int16: tướng*3 + (sao-1); -1 trống. 0-8 chờ, 9-17 sân
    shop: np.ndarray                # [B, P, 5]  int16: chỉ số tướng; -1 đã mua/trống
    pool: np.ndarray                # [B, N]     int16: số bản còn trong kho chung
    gold: np.ndarray                # [B, P]     int16
    level: np.ndarray               # [B, P]     int16
    xp: np.ndarray                  # [B, P]     int16
    hp: np.ndarray                  # [B, P]     int16
    streak: np.ndarray              # [B, P]     int16: >0 chuỗi thắng, <0 chuỗi thua
    alive: np.ndarray               # [B, P]     bool
    placement: np.ndarray           # [B, P]     int8: hạng khi bị loại / kết thúc; 0 = chưa có
    ended_turn: np.ndarray          # [B, P]     bool
    actions_this_round: np.ndarray  # [B, P]     int16
    round: np.ndarray               # [B]        int16, bắt đầu từ 1
    last_opponent: np.ndarray       # [B, P]     int8: đối thủ vòng trước (-1 chưa có)
    last_result: np.ndarray         # [B, P, 2]  int16: (+1 thắng / -1 thua / 0 không đấu, máu mất)
    done: np.ndarray                # [B]        bool
    seed: np.ndarray                # [B]        uint64

    @property
    def B(self) -> int:
        return self.units.shape[0]

    @property
    def P(self) -> int:
        return self.units.shape[1]

    def copy(self) -> "State":
        return State(**{f.name: getattr(self, f.name).copy() for f in fields(self)})

    def rows(self, idx) -> "State":
        """Cắt (bản sao) một tập ván, giữ chiều B. idx: int, slice hoặc mảng."""
        if isinstance(idx, (int, np.integer)):
            idx = slice(int(idx), int(idx) + 1)
        return State(**{f.name: getattr(self, f.name)[idx].copy() for f in fields(self)})

    def assign_rows(self, mask: np.ndarray, other: "State") -> None:
        """Ghi các ván của `other` (đúng bằng số hàng True của mask) vào các hàng mask."""
        for f in fields(self):
            getattr(self, f.name)[mask] = getattr(other, f.name)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


FIELD_NAMES = [f.name for f in fields(State)]


def empty_state(catalog: Catalog, B: int, P: int) -> State:
    N = catalog.N
    return State(
        units=np.full((B, P, NUM_SLOTS), -1, np.int16),
        shop=np.full((B, P, SHOP_SIZE), -1, np.int16),
        pool=np.zeros((B, N), np.int16),
        gold=np.zeros((B, P), np.int16),
        level=np.ones((B, P), np.int16),
        xp=np.zeros((B, P), np.int16),
        hp=np.zeros((B, P), np.int16),
        streak=np.zeros((B, P), np.int16),
        alive=np.zeros((B, P), bool),
        placement=np.zeros((B, P), np.int8),
        ended_turn=np.zeros((B, P), bool),
        actions_this_round=np.zeros((B, P), np.int16),
        round=np.zeros(B, np.int16),
        last_opponent=np.full((B, P), -1, np.int8),
        last_result=np.zeros((B, P, 2), np.int16),
        done=np.zeros(B, bool),
        seed=np.zeros(B, np.uint64),
    )


def reset(catalog: Catalog, seeds, num_players: int) -> State:
    """Khởi tạo B ván mới (vòng 1, đã roll shop miễn phí)."""
    from .step import start_round  # tránh import vòng

    seeds = np.atleast_1d(np.asarray(seeds, dtype=np.uint64))
    if not 2 <= num_players <= MAX_PLAYERS:
        raise ValueError(f"num_players phải trong 2..{MAX_PLAYERS}")
    B, P = len(seeds), num_players
    s = empty_state(catalog, B, P)
    s.seed[:] = seeds
    s.pool[:] = catalog.pool_size(P)[None, :]
    s.gold[:] = catalog.start_gold
    s.hp[:] = catalog.start_hp
    s.alive[:] = True
    s.round[:] = 1
    start_round(catalog, s, np.ones(B, bool))
    return s
