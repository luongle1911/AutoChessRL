"""Pha chuẩn bị, ghép sao, pha giao tranh, đầu vòng, kết thúc ván, tự reset.

API:
    state, info = step(catalog, combat, state, actions)   # actions: [B, P] int
Bản NumPy ghi đè `state` tại chỗ và trả lại chính nó.

info:
    events          list[dict] khi B == 1 hoặc debug=True, ngược lại None
    reset           [B] bool   ván đã kết thúc ở bước trước và vừa được khởi tạo lại
    final_placement [B, P]     thứ hạng của ván cũ (chỉ có nghĩa ở các hàng reset)
    combat          [B] bool   ván có pha giao tranh trong lần gọi này
    opp, ghost      [B, P]     cặp đấu vừa bốc (-1 = không đấu)
    win_prob        [B, P, P]  ma trận xác suất thắng lúc giao tranh (None nếu không có trận)
"""
from __future__ import annotations

import numpy as np

from . import rng
from .actions import (BUY0, SELL0, MOVE0, ROLL, BUY_XP, END_TURN, NUM_ACTIONS,
                      legal_actions, action_name)
from .catalog import Catalog, BENCH_SIZE, NUM_SLOTS, SHOP_SIZE, MAX_LEVEL, STAR_COPIES
from .pairing import draw_pairs
from .state import State, reset

# Thứ tự ưu tiên giữ quân khi ghép sao: sân trước (9..17), rồi hàng chờ (0..8)
_KEEP_PREF = np.r_[BENCH_SIZE:NUM_SLOTS, 0:BENCH_SIZE]
_ROLL_PURPOSES = np.array([rng.ROLL_TIER, rng.ROLL_CHAMP]).reshape(1, 2, 1)
_SHOP_K = np.arange(SHOP_SIZE).reshape(1, 1, SHOP_SIZE)


class _RowView:
    """Lát cắt chỉ-đọc gồm các mảng mà module giao tranh cần, cho một tập ván."""

    def __init__(self, s: State, bi: np.ndarray):
        self.units = s.units[bi]
        self.round = s.round[bi]
        self.alive = s.alive[bi]
        self.hp = s.hp[bi]
        self.level = s.level[bi]


class IllegalActionError(ValueError):
    pass


class InvariantError(AssertionError):
    pass


# ---------------------------------------------------------------- shop
def roll_shop(catalog: Catalog, s: State, p: int, bi: np.ndarray, step_idx) -> None:
    """Trả shop cũ của người p (ở các ván bi) về kho rồi bốc 5 ô mới. Vector hóa trên bi.

    Toàn bộ số ngẫu nhiên của 5 ô được băm trong MỘT lần gọi (cùng bộ chỉ số như từng ô
    riêng lẻ, nên kết quả giống hệt); chỉ bước chọn tướng phải tuần tự vì kho giảm dần.
    """
    if bi.size == 0:
        return
    old = s.shop[bi, p, :]
    for k in range(SHOP_SIZE):
        c = old[:, k]
        h = c >= 0
        s.pool[bi[h], c[h]] += 1
    s.shop[bi, p, :] = -1
    n = bi.size
    odds = catalog.shop_odds[s.level[bi, p] - 1]          # [n,5]
    step_idx = np.broadcast_to(np.asarray(step_idx), bi.shape)
    # u[:, 0, k] = uniform(seed, ROLL_TIER, vòng, bước, p, k); u[:, 1, k] = ... ROLL_CHAMP ...
    u = rng.uniform(s.seed[bi, None, None], _ROLL_PURPOSES, s.round[bi, None, None],
                    step_idx[:, None, None], p, _SHOP_K)  # [n,2,5]
    pool = s.pool[bi].astype(np.int64)                    # [n,N], cập nhật cục bộ rồi ghi lại
    tier_left = pool @ catalog.tier_onehot                # [n,5]
    tier_of = catalog.tier.astype(np.int64)
    ar = np.arange(n)
    for k in range(SHOP_SIZE):
        w = odds * (tier_left > 0)                         # bậc cạn kho bị loại, giữ tỷ lệ tương đối
        cw = np.cumsum(w, 1)
        tot = cw[:, -1]
        ok = tot > 0
        tier = np.argmax(cw > (u[:, 0, k] * tot)[:, None], 1)
        cp = np.cumsum(pool * (tier_of[None, :] == tier[:, None]), 1)
        champ = np.argmax(cp > (u[:, 1, k] * cp[:, -1])[:, None], 1)
        r = ar[ok]
        c = champ[ok]
        s.shop[bi[r], p, k] = c
        pool[r, c] -= 1
        tier_left[r, tier_of[c]] -= 1
    s.pool[bi] = pool


def _level_up(catalog: Catalog, s: State, bi: np.ndarray, p: int, ev) -> None:
    for _ in range(MAX_LEVEL):
        lv = s.level[bi, p]
        need = catalog.xp_to_next[lv - 1]
        up = (lv < MAX_LEVEL) & (s.xp[bi, p] >= need)
        if not up.any():
            break
        r = bi[up]
        s.xp[r, p] -= need[up]
        s.level[r, p] += 1
        if ev is not None:
            for b in r:
                ev.append(dict(type="level_up", b=int(b), seat=p, level=int(s.level[b, p])))
    maxed = bi[s.level[bi, p] >= MAX_LEVEL]
    s.xp[maxed, p] = 0


def _merge(s: State, p: int, bi: np.ndarray, champ: np.ndarray, ev) -> None:
    """Ghép 3 bản cùng mã thành 1 bản sao kế tiếp (lặp 2 lần), ưu tiên giữ ô trên sân."""
    for star in (0, 1):
        if bi.size == 0:
            return
        code = champ * 3 + star
        u = s.units[bi, p, :]
        eq = u == code[:, None]
        do = eq.sum(1) >= 3
        if not do.any():
            return
        rows, eqd, cd = bi[do], eq[do], code[do]
        keeper = _KEEP_PREF[np.argmax(eqd[:, _KEEP_PREF], axis=1)]
        newu = np.where(eqd, -1, u[do]).astype(np.int16)
        newu[np.arange(len(rows)), keeper] = cd + 1
        s.units[rows, p, :] = newu
        if ev is not None:
            for b, c, kslot in zip(rows, champ[do], keeper):
                ev.append(dict(type="merge", b=int(b), seat=p, champ=int(c), star=star + 2, slot=int(kslot)))
        bi, champ = rows, champ[do]


def _clear_player(catalog: Catalog, s: State, bi: np.ndarray, p: int) -> None:
    """Trả toàn bộ quân và shop của người p về kho chung."""
    for j in range(NUM_SLOTS):
        c = s.units[bi, p, j]
        h = c >= 0
        s.pool[bi[h], c[h] // 3] += STAR_COPIES[c[h] % 3]
    for k in range(SHOP_SIZE):
        c = s.shop[bi, p, k]
        h = c >= 0
        s.pool[bi[h], c[h]] += 1
    s.units[bi, p, :] = -1
    s.shop[bi, p, :] = -1


# ---------------------------------------------------------------- đầu vòng
def start_round(catalog: Catalog, s: State, rows: np.ndarray, ev=None) -> None:
    """Thu nhập, +XP, roll miễn phí, đặt lại cờ lượt. Vòng 1 chỉ roll (vàng khởi đầu đã có)."""
    bi = np.flatnonzero(rows)
    if bi.size == 0:
        return
    not_first = s.round[bi] > 1
    for p in range(s.P):
        al = s.alive[bi, p]
        r = bi[al & not_first]
        if r.size:
            g = s.gold[r, p]
            interest = np.minimum(g // catalog.interest_step, catalog.interest_max)
            bonus = catalog.streak_bonus[np.minimum(np.abs(s.streak[r, p]), 5)]
            income = catalog.base_income + interest + bonus
            s.gold[r, p] += income.astype(np.int16)
            if ev is not None:
                for b, i_, b_, t_ in zip(r, interest, bonus, income):
                    ev.append(dict(type="income", b=int(b), seat=p, base=catalog.base_income,
                                   interest=int(i_), streak=int(b_), total=int(t_)))
            s.xp[r, p] += catalog.xp_per_round
            _level_up(catalog, s, r, p, ev)
        roll_shop(catalog, s, p, bi[al], rng.FREE_ROLL)
    s.ended_turn[bi] = ~s.alive[bi]
    s.actions_this_round[bi] = 0


# ---------------------------------------------------------------- giao tranh
def _combat_phase(catalog: Catalog, combat, s: State, cb: np.ndarray, ev, info) -> None:
    bi = np.flatnonzero(cb)
    P = s.P
    opp, ghost = draw_pairs(s, cb)
    # Chỉ đánh giá các ván có giao tranh (thường ~10% lô), rồi trải về kích thước [B, P, P].
    win_sub, dmg_sub = combat.evaluate(catalog, _RowView(s, bi))
    win = np.zeros((s.B, P, P), win_sub.dtype)
    dmg = np.zeros((s.B, P, P), dmg_sub.dtype)
    win[bi], dmg[bi] = win_sub, dmg_sub
    info["opp"], info["ghost"], info["win_prob"] = opp, ghost, win
    pre_hp = s.hp.copy()
    s.last_result[bi] = 0

    for i in range(P):
        j = opp[bi, i].astype(np.int64)
        f = j >= 0
        r, j = bi[f], j[f]
        if r.size == 0:
            continue
        g = ghost[r, i]
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        u = rng.uniform(s.seed[r], rng.COMBAT, s.round[r], lo, hi, g * (i + 1))
        lo_wins = u < win[r, lo, hi]
        i_wins = np.where(g, u < win[r, i, j], np.where(i == lo, lo_wins, ~lo_wins))
        lost = np.where(i_wins, 0, dmg[r, i, j]).astype(np.int16)
        s.hp[r, i] -= lost
        st = s.streak[r, i]
        s.streak[r, i] = np.where(i_wins, np.maximum(st, 0) + 1, np.minimum(st, 0) - 1)
        s.last_result[r, i, 0] = np.where(i_wins, 1, -1)
        s.last_result[r, i, 1] = lost
        s.last_opponent[r, i] = j
        if ev is not None:
            for b, jj, gg, w, l in zip(r, j, g, i_wins, lost):
                if gg or i < jj:
                    ev.append(dict(type="combat", b=int(b), round=int(s.round[b]), a=i, opp=int(jj),
                                   ghost=bool(gg), p_a_wins=float(win[b, i, jj]), a_won=bool(w),
                                   hp_lost=int(l) if not w else int(dmg[b, jj, i]) if not gg else 0,
                                   loser=(i if not w else (int(jj) if not gg else -1))))

    # bị loại (chỉ tính trên các ván có người bị loại)
    newly = s.alive & (s.hp <= 0) & cb[:, None]
    er = np.flatnonzero(newly.any(1))
    if er.size:
        nw, ph = newly[er], pre_hp[er]
        n_after = (s.alive[er] & ~nw).sum(1)
        for i in range(P):
            rank = np.zeros(er.size, np.int64)
            for k in range(P):
                rank += nw[:, k] & ((ph[:, k] > ph[:, i]) | ((ph[:, k] == ph[:, i]) & (k < i)))
            s.placement[er, i] = np.where(nw[:, i], n_after + 1 + rank, s.placement[er, i])
            _clear_player(catalog, s, er[nw[:, i]], i)
        s.alive &= ~newly
        s.hp[newly] = 0
        s.ended_turn[newly] = True
        if ev is not None:
            for b, i in zip(*np.nonzero(newly)):
                ev.append(dict(type="eliminated", b=int(b), seat=int(i), placement=int(s.placement[b, i])))

    # kết thúc ván
    n_alive = s.alive.sum(1)
    end_last = cb & (n_alive <= 1)
    s.placement[:] = np.where(end_last[:, None] & s.alive, 1, s.placement)
    end_max = cb & ~end_last & (s.round >= catalog.max_rounds)
    mr = np.flatnonzero(end_max)
    if mr.size:
        al, hp = s.alive[mr], s.hp[mr]
        for i in range(P):
            rank = np.zeros(mr.size, np.int64)
            for k in range(P):
                rank += al[:, k] & ((hp[:, k] > hp[:, i]) | ((hp[:, k] == hp[:, i]) & (k < i)))
            s.placement[mr, i] = np.where(al[:, i], 1 + rank, s.placement[mr, i])
    ended = end_last | end_max
    s.done |= ended
    if ev is not None:
        for b in np.flatnonzero(ended):
            ev.append(dict(type="game_over", b=int(b), placement=s.placement[b].tolist(),
                           reason="max_rounds" if end_max[b] else "last_standing"))

    cont = cb & ~s.done
    s.round[cont] += 1
    if ev is not None:
        for b in np.flatnonzero(cont):
            ev.append(dict(type="new_round", b=int(b), round=int(s.round[b])))
    start_round(catalog, s, cont, ev)


# ---------------------------------------------------------------- step
def step(catalog: Catalog, combat, state: State, actions, debug: bool = False, mask=None):
    """mask: tùy chọn, mặt nạ đã tính bởi legal_actions trên CHÍNH trạng thái này (tránh tính lại)."""
    s = state
    B, P = s.B, s.P
    act = np.array(actions, dtype=np.int64).reshape(B, P)
    ev = [] if (debug or B == 1) else None
    info = dict(events=ev, combat=np.zeros(B, bool), opp=None, ghost=None, win_prob=None)

    # 1. tự reset các ván đã kết thúc (hành động của chúng ở bước này bị bỏ qua)
    rs = s.done.copy()
    info["reset"] = rs
    info["final_placement"] = s.placement.copy()
    if rs.any():
        fresh = reset(catalog, rng.hash64(s.seed[rs], rng.RESEED), P)
        s.assign_rows(rs, fresh)
        act[rs] = END_TURN
        if ev is not None:
            for b in np.flatnonzero(rs):
                ev.append(dict(type="reset", b=int(b)))

    # 2. kiểm tra mặt nạ (mặt nạ truyền vào vẫn đúng cho ván vừa reset: hành động của
    #    chúng đã bị đặt thành END_TURN, luôn hợp lệ)
    if mask is None or debug:
        mask = legal_actions(catalog, s)
    in_range = (act >= 0) & (act < NUM_ACTIONS)
    ok = in_range & np.take_along_axis(mask, np.clip(act, 0, NUM_ACTIONS - 1)[..., None], 2)[..., 0]
    if not ok.all():
        if debug:
            b, p = map(int, np.argwhere(~ok)[0])
            raise IllegalActionError(f"ván {b}, người {p}: hành động {act[b, p]} "
                                     f"({action_name(act[b, p]) if in_range[b, p] else '?'}) không hợp lệ")
        act = np.where(ok, act, END_TURN)
    active = s.alive & ~s.ended_turn & ~s.done[:, None] & ~rs[:, None]

    # 3. áp dụng theo thứ tự ghế cố định 0..P-1 (mọi người rút chung một kho)
    for p in range(P):
        a = act[:, p]
        # mua
        bi = np.flatnonzero(a < SELL0)
        if bi.size:
            slot = a[bi] - BUY0
            champ = s.shop[bi, p, slot].astype(np.int64)
            s.gold[bi, p] -= catalog.cost[champ]
            s.shop[bi, p, slot] = -1
            dest = np.argmax(s.units[bi, p, :BENCH_SIZE] < 0, axis=1)
            s.units[bi, p, dest] = champ * 3
            if ev is not None:
                for b, c, d in zip(bi, champ, dest):
                    ev.append(dict(type="buy", b=int(b), seat=p, champ=int(c), slot=int(d)))
            _merge(s, p, bi, champ, ev)
        # bán
        bi = np.flatnonzero((a >= SELL0) & (a < MOVE0))
        if bi.size:
            j = a[bi] - SELL0
            code = s.units[bi, p, j].astype(np.int64)
            champ, star = code // 3, code % 3
            copies = STAR_COPIES[star]
            s.gold[bi, p] += catalog.cost[champ] * copies
            s.pool[bi, champ] += copies
            s.units[bi, p, j] = -1
            if ev is not None:
                for b, c, st, jj in zip(bi, champ, star, j):
                    ev.append(dict(type="sell", b=int(b), seat=p, champ=int(c), star=int(st) + 1, slot=int(jj),
                                   refund=int(catalog.cost[c] * STAR_COPIES[st])))
        # chuyển
        bi = np.flatnonzero((a >= MOVE0) & (a < ROLL))
        if bi.size:
            j = a[bi] - MOVE0
            up = j < BENCH_SIZE
            dest = np.where(up, BENCH_SIZE + np.argmax(s.units[bi, p, BENCH_SIZE:] < 0, axis=1),
                            np.argmax(s.units[bi, p, :BENCH_SIZE] < 0, axis=1))
            s.units[bi, p, dest] = s.units[bi, p, j]
            s.units[bi, p, j] = -1
            if ev is not None:
                for b, jj, d in zip(bi, j, dest):
                    ev.append(dict(type="move", b=int(b), seat=p, src=int(jj), dst=int(d)))
        # roll
        bi = np.flatnonzero(a == ROLL)
        if bi.size:
            s.gold[bi, p] -= catalog.roll_cost
            roll_shop(catalog, s, p, bi, s.actions_this_round[bi, p])
            if ev is not None:
                for b in bi:
                    ev.append(dict(type="roll", b=int(b), seat=p))
        # mua XP
        bi = np.flatnonzero(a == BUY_XP)
        if bi.size:
            s.gold[bi, p] -= catalog.xp_cost
            s.xp[bi, p] += catalog.xp_per_buy
            if ev is not None:
                for b in bi:
                    ev.append(dict(type="buy_xp", b=int(b), seat=p))
            _level_up(catalog, s, bi, p, ev)
        # kết thúc lượt
        e = (a == END_TURN) & active[:, p]
        s.ended_turn[e, p] = True
        if ev is not None:
            for b in np.flatnonzero(e):
                ev.append(dict(type="end_turn", b=int(b), seat=p, auto=False))
        # bộ đếm + giới hạn hành động mỗi vòng
        s.actions_this_round[:, p] += active[:, p]
        cap = active[:, p] & ~s.ended_turn[:, p] & (s.actions_this_round[:, p] >= catalog.max_actions_per_round)
        s.ended_turn[cap, p] = True
        if ev is not None:
            for b in np.flatnonzero(cap):
                ev.append(dict(type="end_turn", b=int(b), seat=p, auto=True))

    # 4. ai cũng xong lượt -> giao tranh ngay trong lần gọi này
    cb = ~s.done & (s.ended_turn | ~s.alive).all(1) & ~rs
    if cb.any():
        info["combat"] = cb
        _combat_phase(catalog, combat, s, cb, ev, info)

    if debug:
        from .invariants import check_invariants
        errs = check_invariants(catalog, s)
        if errs:
            raise InvariantError("; ".join(errs[:10]))
    return s, info
