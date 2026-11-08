"""Bot luật đơn giản giữ lãi: lấp sân bằng quân mạnh nhất, mua tướng đã có, giữ vàng để ăn lãi,
lên cấp theo vòng, roll khi máu thấp. Đọc view (không đọc mảng lõi)."""
from __future__ import annotations

import numpy as np

from autochess.core.actions import BUY0, SELL0, MOVE0, ROLL, BUY_XP, END_TURN


def _reserve(view) -> int:
    me = view["me"]
    if me["hp"] < 35:
        return 0
    return min(50, 10 * (view["round"] // 2))


def _target_level(view) -> int:
    return min(9, 3 + view["round"] // 3)


def act(view, mask: np.ndarray, rng: np.random.Generator) -> int:
    me = view["me"]
    gold, level = me["gold"], me["level"]
    board = [u for u in view["board"] if u]
    bench = [u for u in view["bench"] if u]
    reserve = _reserve(view)

    def ok(a):
        return bool(mask[a])

    # 1. sân còn chỗ: đưa quân mạnh nhất từ hàng chờ lên
    if bench and len(board) < level:
        u = max(bench, key=lambda u: u["power"])
        if ok(MOVE0 + u["slot"]):
            return MOVE0 + u["slot"]
    # 2. sân đầy nhưng hàng chờ có quân mạnh hơn quân yếu nhất trên sân: cho quân yếu xuống
    if bench and board and len(board) >= level:
        strong = max(bench, key=lambda u: u["power"])
        weak = min(board, key=lambda u: u["power"])
        if strong["power"] > weak["power"]:
            if ok(MOVE0 + weak["slot"]):
                return MOVE0 + weak["slot"]
            weakest_bench = min(bench, key=lambda u: u["power"])
            if ok(SELL0 + weakest_bench["slot"]):
                return SELL0 + weakest_bench["slot"]
    # 3. mua: ưu tiên tướng đã có (để ghép sao), sau đó lấp sân nếu còn dư vàng
    owned_shop = [s for s in view["shop"] if s and s["owned"] > 0]
    for s in sorted(owned_shop, key=lambda s: (-s["owned"], -s["cost"])):
        if ok(BUY0 + s["slot"]) and (gold - s["cost"] >= reserve // 2 or me["hp"] < 50):
            return BUY0 + s["slot"]
    if len(board) + len(bench) < level + 2:
        for s in sorted([s for s in view["shop"] if s], key=lambda s: -s["cost"]):
            if ok(BUY0 + s["slot"]) and gold - s["cost"] >= reserve:
                return BUY0 + s["slot"]
    # 4. hàng chờ đầy: bán quân yếu nhất
    if len(bench) >= 9:
        weakest = min(bench, key=lambda u: u["power"])
        return SELL0 + weakest["slot"]
    # 5. lên cấp theo lịch
    if level < _target_level(view) and gold - 4 >= reserve and ok(BUY_XP):
        return BUY_XP
    # 6. roll khi máu thấp, hoặc vàng dư nhiều
    if ok(ROLL) and ((me["hp"] < 35 and gold >= 2) or gold - 2 >= reserve + 20):
        return ROLL
    return END_TURN
