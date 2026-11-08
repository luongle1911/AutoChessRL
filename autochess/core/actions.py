"""Bảng 44 hành động và mặt nạ hợp lệ.

  0-4    mua ô shop i
  5-22   bán quân ở ô j (0-8 hàng chờ, 9-17 sân)
  23-40  chuyển quân ở ô j: chờ -> sân hoặc sân -> chờ (vào ô trống đầu tiên)
  41     roll shop
  42     mua kinh nghiệm
  43     kết thúc lượt (luôn hợp lệ; với người đã xong lượt/bị loại = bỏ qua)
"""
from __future__ import annotations

import numpy as np

from .catalog import Catalog, BENCH_SIZE, NUM_SLOTS, SHOP_SIZE, MAX_LEVEL

BUY0 = 0
SELL0 = BUY0 + SHOP_SIZE          # 5
MOVE0 = SELL0 + NUM_SLOTS         # 23
ROLL = MOVE0 + NUM_SLOTS          # 41
BUY_XP = ROLL + 1                 # 42
END_TURN = BUY_XP + 1             # 43
NUM_ACTIONS = END_TURN + 1        # 44

_BIG = np.int16(10_000)


def action_name(a: int) -> str:
    a = int(a)
    if BUY0 <= a < SELL0:
        return f"mua ô shop {a - BUY0}"
    if SELL0 <= a < MOVE0:
        return f"bán ô {a - SELL0}"
    if MOVE0 <= a < ROLL:
        return f"chuyển ô {a - MOVE0}"
    return {ROLL: "roll", BUY_XP: "mua XP", END_TURN: "kết thúc lượt"}.get(a, f"?{a}")


def mask_parts(catalog: Catalog, state) -> dict:
    """Các mặt nạ thành phần (để UI giải thích vì sao một lệnh bị từ chối)."""
    units, shop = state.units, state.shop
    active = state.alive & ~state.ended_turn & ~state.done[:, None]           # [B,P]
    shop_has = shop >= 0                                                      # [B,P,5]
    shop_cost = np.where(shop_has, catalog.cost[np.maximum(shop, 0)], _BIG)
    gold = state.gold
    occupied = units >= 0                                                     # [B,P,18]
    bench_space = (~occupied[..., :BENCH_SIZE]).any(-1)                       # [B,P]
    board_count = occupied[..., BENCH_SIZE:].sum(-1)                          # [B,P]
    return dict(
        active=active,
        shop_has=shop_has,
        gold_for_buy=gold[..., None] >= shop_cost,
        bench_space=bench_space,
        occupied=occupied,
        board_room=board_count < state.level,
        board_count=board_count,
        gold_for_roll=gold >= catalog.roll_cost,
        gold_for_xp=gold >= catalog.xp_cost,
        below_max_level=state.level < MAX_LEVEL,
    )


def legal_actions(catalog: Catalog, state, return_parts: bool = False):
    """mask[B, P, 44] bool."""
    pt = mask_parts(catalog, state)
    B, P = state.gold.shape
    act = pt["active"]
    m = np.zeros((B, P, NUM_ACTIONS), bool)
    m[..., BUY0:SELL0] = act[..., None] & pt["shop_has"] & pt["gold_for_buy"] & pt["bench_space"][..., None]
    m[..., SELL0:MOVE0] = act[..., None] & pt["occupied"]
    occ = pt["occupied"]
    m[..., MOVE0:MOVE0 + BENCH_SIZE] = act[..., None] & occ[..., :BENCH_SIZE] & pt["board_room"][..., None]
    m[..., MOVE0 + BENCH_SIZE:ROLL] = act[..., None] & occ[..., BENCH_SIZE:] & pt["bench_space"][..., None]
    m[..., ROLL] = act & pt["gold_for_roll"]
    m[..., BUY_XP] = act & pt["gold_for_xp"] & pt["below_max_level"]
    m[..., END_TURN] = True
    if return_parts:
        return m, pt
    return m


def explain(catalog: Catalog, state, b: int, p: int, a: int, parts: dict | None = None) -> str:
    """Lý do (tiếng Việt) một hành động bị che; chuỗi rỗng nếu hợp lệ."""
    if parts is None:
        parts = mask_parts(catalog, state)
    g = lambda k: parts[k][b, p]
    a = int(a)
    if not 0 <= a < NUM_ACTIONS:
        return f"không có hành động số {a}"
    if a == END_TURN:
        return ""
    if state.done[b]:
        return "ván đã kết thúc"
    if not state.alive[b, p]:
        return "bạn đã bị loại"
    if state.ended_turn[b, p]:
        return "bạn đã kết thúc lượt vòng này"
    if a < SELL0:
        i = a - BUY0
        if not g("shop_has")[i]:
            return f"ô shop {i} trống (đã mua)"
        if not g("gold_for_buy")[i]:
            c = int(catalog.cost[state.shop[b, p, i]])
            return f"thiếu vàng: cần {c}, có {int(state.gold[b, p])}"
        if not g("bench_space"):
            return "hàng chờ đầy (giai đoạn 1 không cho mua khi hàng chờ đầy)"
        return ""
    if a < MOVE0:
        j = a - SELL0
        return "" if g("occupied")[j] else f"ô {j} không có quân"
    if a < ROLL:
        j = a - MOVE0
        if not g("occupied")[j]:
            return f"ô {j} không có quân"
        if j < BENCH_SIZE and not g("board_room"):
            return f"sân đã đủ {int(g('board_count'))}/{int(state.level[b, p])} quân (= cấp)"
        if j >= BENCH_SIZE and not g("bench_space"):
            return "hàng chờ đầy, không xuống được"
        return ""
    if a == ROLL:
        return "" if g("gold_for_roll") else f"thiếu vàng: roll cần {catalog.roll_cost}"
    if a == BUY_XP:
        if not g("below_max_level"):
            return "đã cấp tối đa"
        return "" if g("gold_for_xp") else f"thiếu vàng: mua XP cần {catalog.xp_cost}"
    return ""
