"""Bot luật có tham số, vector hóa trên B ván và nhiều ghế cùng lúc.

Một lớp `ScriptedBot` dùng chung; bốn lối chơi chỉ khác nhau ở bộ tham số `BotParams`:

    tempo   đỉnh vòng 1-11  : không giữ vàng, không roll, dồn vàng lên cấp, mua quân mạnh nhất
    hyper   đỉnh vòng 7-11  : roll hết vàng từ vòng 3 ở cấp 4-6, săn 2 tướng 1-2 tiền lên 3 sao
    roll7   đỉnh vòng 12-14 : giữ lãi, lên cấp 7 ở vòng 12, roll xuống săn tướng 2-3 tiền
    fast9   đỉnh vòng 15+   : giữ lãi tối đa, lên cấp 8 ở vòng 16, săn tướng 4-5 tiền
    bank    "tích vàng rồi xả": giữ lãi tới 50, lên cấp đều, săn tướng 2-4 tiền ghép sao, máu dưới 50 thì
            tiêu hết (lối chơi của người thử nghiệm đã thắng agent giai đoạn 1; thêm vào nhóm đối thủ giai đoạn 2)
    tank    "bán máu" (lối chơi người thử nghiệm đã về nhất trên luật 3.1): đánh 2 vòng đầu, từ vòng 3 bán hết quân,
            thua liên tục để ăn thưởng chuỗi thua + lãi tối đa, chỉ mua XP bằng vàng trên 50 (tới cấp 8); tới vòng
            `rebuild_round` (hoặc máu <= `rebuild_hp`) thì lên cấp 8 và roll hết vàng dựng đội tướng 3-5 tiền
            (ép tướng 3 tiền lên 2 sao); sau đó vẫn mua quân để ghép sao nhưng chỉ roll bằng vàng trên `roll_floor`
            (vừa thua thì roll xuống `loss_floor`)

Bot đọc thẳng mảng trạng thái (không dựng view) và không có ngẫu nhiên: cùng trạng thái
thì cùng hành động. Mỗi bước, bot chọn hành động hợp lệ ĐẦU TIÊN theo thứ tự ưu tiên:

    1. hàng chờ quá đầy      -> bán quân ít giá trị nhất ở hàng chờ
    2. sân còn chỗ           -> đưa quân mạnh nhất từ hàng chờ lên
    3. hàng chờ có quân mạnh hơn quân yếu nhất trên sân -> cho quân yếu xuống
    4. mua                   -> ô shop có điểm cao nhất (tướng đang săn, ghép sao, lấp sân)
    5. lên cấp theo kế hoạch -> mua XP
    6. roll theo kế hoạch    -> roll
    7. kết thúc lượt

Bot có giai đoạn "bán máu" (tank_from > 0): trong giai đoạn đó chỉ bán quân, mua XP bằng vàng trên `tank_floor`,
rồi kết thúc lượt. Hết giai đoạn đó mà sân chưa đủ quân thì cả hai sàn về 0 (roll xuống dựng đội).

"Sàn vàng" quyết định bot được tiêu tới đâu:
    floor_low  : cho mua quân phụ (lấp sân, nâng cấp nhẹ) - thường bằng mức giữ lãi
    floor_high : cho mua tướng đang săn, ghép sao, lên cấp và roll
Máu dưới `panic_hp` thì cả hai sàn về 0 (roll hết để cứu máu).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autochess.core.actions import BUY0, SELL0, MOVE0, ROLL, BUY_XP, END_TURN
from autochess.core.catalog import BENCH_SIZE, NUM_SLOTS, SHOP_SIZE, STAR_COPIES, MAX_LEVEL

_NEG = -1e9


def _plan(points: dict, max_rounds: int, default: int) -> np.ndarray:
    """Bảng bậc thang theo vòng: points = {vòng bắt đầu: giá trị}. Chỉ số = vòng."""
    out = np.full(max_rounds + 2, default, np.int64)
    for r in sorted(points):
        out[r:] = points[r]
    return out


@dataclass(frozen=True)
class BotParams:
    name: str
    level_plan: dict          # {vòng: cấp mục tiêu}
    econ_plan: dict           # {vòng: vàng giữ lại khi mua quân phụ}  -> floor_low
    spend_plan: dict          # {vòng: vàng giữ lại khi săn/lên cấp/roll} -> floor_high
    roll_from: int            # vòng bắt đầu được roll
    chase_tiers: tuple        # bậc (0..4) của tướng muốn săn lên sao
    chase_k: int              # số tướng săn cùng lúc
    panic_hp: int             # máu dưới mức này: tiêu hết
    bench_keep: int = 7       # hàng chờ đạt mức này thì bán bớt quân rác
    stop_roll_when_done: bool = True  # ngừng roll khi mọi tướng đang săn đã 3 sao
    tank_from: int = 0        # > 0: từ vòng này bỏ trống sân, thua liên tục (lối "bán máu")
    rebuild_round: int = 99   # vòng dựng lại đội
    rebuild_hp: int = 0       # hoặc khi máu xuống tới mức này
    tank_floor: int = 50      # trong giai đoạn bán máu chỉ mua XP bằng vàng trên mức này
    loss_floor: int = -1      # >= 0: vừa thua thì sàn tiêu (floor_high) hạ xuống mức này
    tank_level_cap: int = 8   # trong giai đoạn bán máu, vàng dư trên tank_floor dùng mua XP tới cấp này
    body_min_cost: int = 1    # chỉ mua tướng giá >= mức này để lấp sân / thay quân yếu (ghép sao thì vẫn mua)
    roll_floor: int = -1      # >= 0: roll chỉ khi còn trên mức này (mua quân vẫn theo econ/spend_plan)


PRESETS = {
    # Đã chỉnh bằng giải đấu (python3 -m autochess.tournament) trên catalog-v1.
    "tempo": BotParams(
        name="tempo",
        level_plan={1: 3, 2: 4, 4: 5, 6: 6, 9: 7, 18: 8, 26: 9},
        econ_plan={1: 0},
        spend_plan={1: 0},
        roll_from=99, chase_tiers=(0, 1, 2), chase_k=3, panic_hp=0, bench_keep=6,
        stop_roll_when_done=False),
    "hyper": BotParams(
        name="hyper",
        level_plan={1: 2, 3: 4, 5: 5, 8: 6, 20: 7, 26: 8},
        econ_plan={1: 0},
        spend_plan={1: 0},
        roll_from=3, chase_tiers=(0, 1), chase_k=2, panic_hp=0, bench_keep=7),
    "roll7": BotParams(
        name="roll7",
        level_plan={1: 2, 5: 4, 8: 5, 10: 6, 12: 7, 20: 8, 28: 9},
        econ_plan={1: 0, 3: 10, 5: 30, 7: 50},
        spend_plan={1: 0, 3: 10, 5: 30, 7: 50, 12: 10, 14: 50},
        roll_from=12, chase_tiers=(1, 2), chase_k=3, panic_hp=35, bench_keep=7),
    "fast9": BotParams(
        name="fast9",
        level_plan={1: 2, 6: 4, 9: 5, 12: 6, 14: 7, 16: 8, 22: 9},
        econ_plan={1: 0, 3: 10, 5: 30, 7: 50},
        spend_plan={1: 0, 3: 10, 5: 30, 7: 50, 16: 10, 18: 50, 22: 10, 24: 50},
        roll_from=16, chase_tiers=(3, 4), chase_k=3, panic_hp=25, bench_keep=7),
    "bank": BotParams(
        name="bank",
        level_plan={1: 3, 3: 4, 6: 5, 10: 6, 14: 7, 17: 8, 24: 9},
        econ_plan={1: 0, 3: 10, 5: 30, 7: 50},
        spend_plan={1: 0, 3: 10, 5: 30, 7: 50},
        roll_from=8, chase_tiers=(1, 2, 3), chase_k=3, panic_hp=50, bench_keep=7),
    # Chép lối chơi của người thử nghiệm (log 07/11, luật 3.1): cấp 4/5/6/7/8 ở vòng 4/8/9/12/14, bán hết quân ở
    # vòng 3, dựng lại ở vòng 14 với cấp 8, về nhất. Bot dựng ở vòng 13: thử 13/14/15 với 7 agent thì 13 và 15 tốt hơn 14.
    "tank": BotParams(
        name="tank",
        level_plan={1: 3, 4: 4, 8: 5, 9: 6, 12: 7, 14: 8, 26: 9},
        econ_plan={1: 0},
        spend_plan={1: 0},
        roll_from=14, chase_tiers=(2, 3, 4), chase_k=4, panic_hp=0, bench_keep=7,
        stop_roll_when_done=False, tank_from=3, rebuild_round=13, rebuild_hp=20, tank_floor=50,
        loss_floor=20, tank_level_cap=8, body_min_cost=3, roll_floor=50),
}


class ScriptedBot:
    """Bot luật vector hóa. act(state, mask, seats) -> actions[B, len(seats)]."""

    def __init__(self, catalog, params: BotParams):
        self.cat = catalog
        self.p = params
        self.name = params.name
        R = catalog.max_rounds
        self.level_target = np.minimum(_plan(params.level_plan, R, 1), MAX_LEVEL)
        self.floor_low = _plan(params.econ_plan, R, 0)
        self.floor_high = _plan(params.spend_plan, R, 0)
        self.chase_tier_mask = np.isin(catalog.tier, params.chase_tiers)       # [N]
        self.power = catalog.power.astype(np.float64)                           # [N,3]
        self.cost = catalog.cost.astype(np.int64)

    # ------------------------------------------------------------------ chính
    def act(self, state, mask: np.ndarray, seats=None) -> np.ndarray:
        """state: State (B ván); mask: [B, P, 44]; seats: các ghế bot này ngồi (mặc định tất cả)."""
        cat, prm = self.cat, self.p
        B, P = state.B, state.P
        seats = np.arange(P) if seats is None else np.atleast_1d(np.asarray(seats))
        S = seats.size
        m = mask[:, seats]                                         # [B,S,44]
        u = state.units[:, seats].astype(np.int64)                 # [B,S,18]
        shop = state.shop[:, seats].astype(np.int64)               # [B,S,5]
        gold = state.gold[:, seats].astype(np.int64)
        level = state.level[:, seats].astype(np.int64)
        hp = state.hp[:, seats].astype(np.int64)
        rnd = np.broadcast_to(np.minimum(state.round, cat.max_rounds)[:, None], (B, S))
        pool = state.pool.astype(np.int64)                         # [B,N]
        N = cat.N

        occ = u >= 0
        champ = np.where(occ, u // 3, 0)
        star = np.where(occ, u % 3, 0)
        val = np.where(occ, self.power[champ, star], np.inf)       # để tìm min
        val_max = np.where(occ, self.power[champ, star], -np.inf)  # để tìm max
        n_bench = occ[..., :BENCH_SIZE].sum(-1)
        n_board = occ[..., BENCH_SIZE:].sum(-1)

        # số bản gốc đang sở hữu của từng tướng [B,S,N]
        onehot = (champ[..., None] == np.arange(N)) & occ[..., None]
        copies = np.where(occ, STAR_COPIES[star], 0)
        owned = (onehot * copies[..., None]).sum(-2)

        # tướng đang săn: trong các bậc mục tiêu, nhiều bản nhất (hòa thì kho còn nhiều hơn)
        score = np.where(self.chase_tier_mask, owned * 1000 + pool[:, None, :], -1)
        order = np.argsort(-score, axis=-1, kind="stable")[..., :prm.chase_k]
        chase = np.zeros((B, S, N), bool)
        np.put_along_axis(chase, order, True, axis=-1)
        chase &= self.chase_tier_mask
        need = chase & (owned < 9)                                 # chưa 3 sao
        chase_left = need.any(-1)

        panic = hp < prm.panic_hp
        f_low = np.where(panic, 0, self.floor_low[rnd])
        f_high = np.where(panic, 0, self.floor_high[rnd])
        f_roll = f_high if prm.roll_floor < 0 else np.where(panic, 0, np.full((B, S), prm.roll_floor))
        if prm.loss_floor >= 0:
            lost = state.streak[:, seats] < 0
            f_high = np.where(lost, np.minimum(f_high, prm.loss_floor), f_high)
            f_roll = np.where(lost, np.minimum(f_roll, prm.loss_floor), f_roll)

        act = np.full((B, S), END_TURN, np.int64)
        decided = np.zeros((B, S), bool)

        def take(cond, a):
            nonlocal act, decided
            a = np.broadcast_to(np.asarray(a, np.int64), (B, S))
            ok = cond & ~decided & np.take_along_axis(m, a[..., None], -1)[..., 0]
            act = np.where(ok, a, act)
            decided |= ok

        if prm.tank_from > 0:
            # giai đoạn bán máu: bán hết quân, mua XP bằng vàng dư trên tank_floor, rồi kết thúc lượt
            tank = (rnd >= prm.tank_from) & (rnd < prm.rebuild_round) & (hp > prm.rebuild_hp)
            first = occ.argmax(-1)
            take(tank & occ.any(-1), SELL0 + first)
            take(tank & (level < prm.tank_level_cap) & (gold - cat.xp_cost >= prm.tank_floor), BUY_XP)
            decided |= tank
            # vừa hết giai đoạn bán máu mà sân chưa đủ quân: tiêu hết để dựng đội
            rebuild = ~tank & (rnd >= prm.tank_from) & ((n_board + n_bench) < level)
            f_low = np.where(rebuild, 0, f_low)
            f_high = np.where(rebuild, 0, f_high)
            f_roll = np.where(rebuild, 0, f_roll)

        # giá trị giữ của quân hàng chờ: tướng đang săn > tướng có cặp > sức mạnh
        unit_owned = np.take_along_axis(owned, champ, -1)          # [B,S,18]
        unit_chase = np.take_along_axis(chase, champ, -1)
        keep = np.where(occ, val_max + 1000 * unit_chase + 300 * ((unit_owned >= 2) & (unit_owned < 9)
                                                                   & (star < 2)), np.inf)

        # 1. hàng chờ quá đầy -> bán quân hàng chờ ít giá trị nhất (không bán tướng đang săn trừ khi đầy hẳn)
        bk = keep[..., :BENCH_SIZE]
        j = bk.argmin(-1)
        jk = np.take_along_axis(bk, j[..., None], -1)[..., 0]
        take((n_bench >= prm.bench_keep) & ((jk < 1000) | (n_bench >= BENCH_SIZE)), SELL0 + j)

        # 2. sân còn chỗ -> đưa quân mạnh nhất lên
        bv = val_max[..., :BENCH_SIZE]
        jb = bv.argmax(-1)
        take((n_board < level) & (n_bench > 0), MOVE0 + jb)

        # 3. hàng chờ có quân mạnh hơn quân yếu nhất trên sân -> cho quân yếu xuống
        best_bench = np.take_along_axis(bv, jb[..., None], -1)[..., 0]
        wv = val[..., BENCH_SIZE:]
        jw = wv.argmin(-1)
        weakest = np.take_along_axis(wv, jw[..., None], -1)[..., 0]
        take((n_board >= level) & (n_bench > 0) & (n_bench < BENCH_SIZE) & (best_bench > weakest),
             MOVE0 + BENCH_SIZE + jw)

        # 4. mua
        has = shop >= 0
        sc = np.where(has, shop, 0)
        c_cost = self.cost[sc]                                      # [B,S,5]
        c_owned = np.take_along_axis(owned, sc, -1)
        c_need = np.take_along_axis(need, sc, -1)
        c_pow = self.power[sc, 0]
        weakest_f = np.where(n_board > 0, weakest, 0.0)
        if cat.stats is not None:
            # luật bản 4: chấm quân định mua bằng mức tăng độ mạnh của cả đội (tính cả tộc hệ và chỉ số), như thể
            # đặt nó vào ô trống trên sân (hoặc thay quân yếu nhất nếu sân đã đủ)
            gain = self._gain(u[..., BENCH_SIZE:], sc, has, n_board, level, jw)            # [B,S,5]
            c_pow = gain
            weakest_f = np.zeros_like(weakest_f)
        # body_min_cost chỉ áp dụng sau giai đoạn đầu (bot bán máu vẫn mua quân rẻ ở 2 vòng đầu)
        min_cost = np.where(rnd >= max(prm.tank_from, 1), prm.body_min_cost, 1)[..., None]
        bodies = ((n_board + n_bench) < level)[..., None] & (c_cost >= min_cost)
        if cat.stats is not None:
            upgrade = (n_board >= level)[..., None] & (c_pow > 0.03 * self._base[..., None]) & (c_cost >= min_cost)
        else:
            upgrade = (n_board >= level)[..., None] & (c_pow > weakest_f[..., None] * 1.15) & (c_cost >= min_cost)
        completes = (c_owned % 3 == 2) & (c_owned < 9)               # bản này ghép thành sao mới
        pair = (c_owned % 3 == 1) & (c_owned < 9) & self.chase_tier_mask[sc]
        buy_score = np.where(c_need, 400 + c_owned,
                    np.where(completes, 300 + c_pow,
                    np.where(pair, 150 + c_pow,
                    np.where(bodies, 50 + c_pow,
                    np.where(upgrade, 10 + c_pow - weakest_f[..., None], _NEG)))))
        hi = buy_score >= 150
        floor = np.where(hi, f_high[..., None], f_low[..., None])
        room = np.where(hi, n_bench[..., None] < BENCH_SIZE, n_bench[..., None] + 1 < prm.bench_keep)
        buy_ok = has & (gold[..., None] - c_cost >= floor) & room & m[..., BUY0:SELL0]
        buy_score = np.where(buy_ok, buy_score, _NEG)
        k = buy_score.argmax(-1)
        take(np.take_along_axis(buy_score, k[..., None], -1)[..., 0] > _NEG / 2, BUY0 + k)

        # 5. lên cấp theo kế hoạch
        take((level < self.level_target[rnd]) & (gold - cat.xp_cost >= f_high), BUY_XP)

        # 6. roll
        want_roll = (rnd >= prm.roll_from) | panic
        if prm.stop_roll_when_done:
            want_roll &= chase_left | panic
        take(want_roll & (gold - cat.roll_cost >= f_roll), ROLL)

        return act

    def _gain(self, board, sc, has, n_board, level, jw):
        """Mức tăng độ mạnh đội khi thêm từng quân trong shop (1 sao) [B,S,5]."""
        from autochess.core.combat import board_rating
        cat = self.cat
        base = board_rating(cat, board)                                              # [B,S]
        self._base = base
        empty = board < 0
        first_empty = empty.argmax(-1)
        slot = np.where((n_board < level) & empty.any(-1), first_empty, jw)          # [B,S]
        cand = np.repeat(board[..., None, :], sc.shape[-1], -2).copy()              # [B,S,5,9]
        np.put_along_axis(cand, np.broadcast_to(slot[..., None, None], cand.shape[:-1] + (1,)),
                          (sc * 3)[..., None], -1)
        g = board_rating(cat, cand) - base[..., None]
        return np.where(has, g, -1e9)

    def act_one(self, state, mask_b: np.ndarray, b: int, seat: int) -> int:
        """Một ván, một ghế (cho lớp chơi thử)."""
        sb = state.rows(b) if state.B > 1 else state
        full = np.zeros((1, sb.P, mask_b.shape[-1]), bool)
        full[0, seat] = mask_b
        return int(self.act(sb, full, [seat])[0, 0])


def make_bots(catalog, names=None) -> dict:
    names = names or list(PRESETS)
    return {n: ScriptedBot(catalog, PRESETS[n]) for n in names}
