"""Mô phỏng giao tranh nấc 2 (luật bản 5): năng lượng, kỹ năng, đồ, tộc hệ, lõi qua một bộ thông dịch khối hiệu
ứng chung (battlesim/effects.py), cùng hiệp phụ tăng sát thương. Viết theo kiểu tương thích Numba như sim.py.

Đầu vào mỗi trận (lô B trận, U = 20 ô, 10 mỗi phe; dựng bằng battlesim.team.build_batch):
    champ, star, side, half_cell, seed       như sim.py
    base [B,U,NST]                           chỉ số gốc (đã nhân sao)
    padd, pmul [B,U,NST]                     cộng / nhân cố định cả trận (đồ, tộc hệ, lõi gộp sẵn)
    slot_eff, slot_lv [B,U,E]                các khối hiệu ứng động của từng quân (mã khối, bậc 0..3)
    EI, EF                                   bảng khối hiệu ứng (compile_effects)
Chỉ số hiện tại = (gốc + cộng) x (1 + nhân), cộng/nhân gồm phần cố định và các hiệu ứng có thời hạn.

Một bước thời gian (0,1 giây):
  1. đếm lùi (đòn đánh, choáng, khiên, hiệu ứng hết hạn...)
  2. hiệu ứng theo thời gian / ngưỡng máu / hạ địch ở bước trước
  3. chọn mục tiêu (bỏ qua kẻ địch đang mất dấu)
  4. tung chiêu nếu đầy năng lượng (không đánh trong bước đó)
  5. đánh hoặc xin đi một ô (như sim.py); đòn đánh kích hoạt hiệu ứng "mỗi đòn", "chí mạng"
  6. giải quyết tranh ô;  7. "khi bị đánh";  8. thiêu
  9. áp mọi tác động trong hàng đợi CÙNG LÚC (chỉ số -> khiên -> hồi máu -> khống chế -> năng lượng -> sát thương
     -> lướt), rồi xử lý chết.  Vì mọi tác động trong bước đều tính trên trạng thái đầu bước rồi mới áp, hai phe
     không phe nào được lợi do thứ tự trong mảng.
Hiệp phụ: từ giây ot_start, mọi sát thương nhân (1 + (t - ot_start) / ot_span); hết max_steps thì cả hai thua.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import hexgrid
from .jit import njit2 as njit, prange2 as prange, HAS_NUMBA
from .sim import _bfs as _bfs_j, _first_goal as _first_goal_j, _rand as _rand_j, _log as _log_j

# các hàm dùng lại của sim.py: bản Python thường khi sim2 không biên dịch
_bfs = getattr(_bfs_j, 'py_func', _bfs_j)
_first_goal = getattr(_first_goal_j, 'py_func', _first_goal_j)
_rand = getattr(_rand_j, 'py_func', _rand_j)
_log = getattr(_log_j, 'py_func', _log_j)
import os as _os
if HAS_NUMBA and _os.environ.get('BATTLESIM_JIT2') == '1':
    _bfs, _first_goal, _rand, _log = _bfs_j, _first_goal_j, _rand_j, _log_j
from .effects import (
    NST, ST_HP, ST_AD, ST_AP, ST_AS, ST_ARMOR, ST_MR, ST_CRIT_CH, ST_CRIT_DMG, ST_MANA_MAX, ST_RANGE, ST_VAMP,
    ST_AMP, ST_DR, ST_AMP_BIG, ST_AMP_SHIELD, ST_ABIL_CRIT, ST_MANA_ATK, ST_MANA_START,
    TR_START, TR_ATTACK, TR_ATTACK_N, TR_CRIT, TR_CAST, TR_PERIODIC, TR_AT_TIME, TR_HP_BELOW, TR_HIT_TAKEN,
    TR_KILL,
    TG_SELF, TG_TARGET, TG_ENEMIES_R, TG_ALLIES_R, TG_AROUND_TARGET, TG_LOWEST_ALLY, TG_FARTHEST,
    TG_NEAREST_N, TG_NEAR_TARGET_N, TG_DENSEST, TG_ALL_ALLIES, TG_ROW_ALLIES, TG_ALL_ENEMIES,
    K_DAMAGE, K_HEAL, K_SHIELD, K_STUN, K_STAT, K_MANA, K_WOUND, K_BURN, K_DASH, K_STEALTH, K_CC_IMMUNE,
    SC_NONE, SC_AD, SC_AP, SC_TGT_HP, SC_SELF_HP, SC_TGT_MISSING,
    EI_TRIG, EI_TARGET, EI_KIND, EI_STAT, EI_MODE, EI_SCALE, EI_COUNT, EI_RADIUS, EI_STACKS, EI_COND, EI_KEY,
    EF_V0, EF_DUR, EF_PERIOD,
)

U = 20
U_PER_SIDE = 10
N_CELLS = 56
N_HALF = 28
E_SLOTS = 24          # khối hiệu ứng động mỗi quân
M_SLOTS = 32          # hiệu ứng chỉ số đang có mỗi quân
MAXQ = 512            # tác động chờ áp mỗi bước mỗi trận
EPS = 1e-4
NEVER = 1e9

# ---- F: trạng thái thực ----
(F_HP, F_SHIELD, F_SHIELD_T, F_MANA, F_LOCK, F_ATK_CD, F_MOVE_CD, F_STUN, F_STEALTH, F_CCIMM, F_WOUND,
 F_BURN, F_BURN_T, F_CRIT_ACC, F_DEALT, F_DEALT_PHYS, F_DEALT_MAG, F_DEALT_TRUE, F_TAKEN, F_HEALED,
 F_SHIELDED, F_HITMAX) = range(22)
NF = 22
# ---- I: trạng thái nguyên ----
(I_CHAMP, I_STAR, I_SIDE, I_ORD, I_CELL, I_TARGET, I_ALIVE, I_BLOCKED, I_DEATH, I_CASTS, I_AVOID, I_GOAL,
 I_PREV, I_ATKS, I_HITS, I_LASTHIT, I_KILLS, I_DIRTY) = range(18)
NI = 18
# ---- hàng đợi ----
(Q_KIND, Q_SRC, Q_DST, Q_STAT, Q_MODE, Q_KEY, Q_STACKS, Q_ABIL) = range(8)
NQI = 8
NQF = 2               # lượng, thời hạn
# ---- tham số chung GP ----
(GP_DT, GP_MOVE_STEPS, GP_MAX_STEPS, GP_OT_START, GP_OT_SPAN, GP_MANA_ATK, GP_MANA_LOCK, GP_MANA_DMG,
 GP_MANA_DMG_CAP, GP_WOUND, GP_BIG_HP, GP_CAST_TIME, GP_AS_CAP) = range(13)
NGP = 13
# ---- sự kiện ----
EV_MOVE, EV_ATTACK, EV_DEATH, EV_CAST, EV_ABIL, EV_HEAL, EV_SHIELD, EV_STUN, EV_DASH = range(9)
RES_SIDE0, RES_SIDE1, RES_BOTH_LOSE = 0, 1, 2


# --------------------------------------------------------------------------- kernel
@njit
def _recompute(b, u, S, BASE, PADD, PMUL, MI, MF, F, I, as_cap):
    old_hp = S[b, u, ST_HP]
    for k in range(S.shape[2]):
        a = BASE[b, u, k] + PADD[b, u, k]
        m = 1.0 + PMUL[b, u, k]
        for j in range(MI.shape[2]):
            if MI[b, u, j, 0] == k:
                v = MF[b, u, j, 0] * MI[b, u, j, 3]
                if MI[b, u, j, 1] == 0:
                    a += v
                else:
                    m += v
        val = a * m
        if k == ST_AS:
            if val > as_cap:
                val = as_cap
            if val < 0.0:
                val = 0.0
        elif k == ST_CRIT_CH:
            if val > 1.0:
                val = 1.0
        elif k == ST_ARMOR or k == ST_MR or k == ST_HP or k == ST_AP:
            if val < 0.0:
                val = 0.0
        elif k == ST_RANGE:
            if val < 1.0:
                val = 1.0
        S[b, u, k] = val
    new_hp = S[b, u, ST_HP]
    if new_hp > old_hp and I[b, u, I_ALIVE] == 1:
        F[b, u, F_HP] += new_hp - old_hp
    if F[b, u, F_HP] > new_hp:
        F[b, u, F_HP] = new_hp
    I[b, u, I_DIRTY] = 0


@njit
def _add_mod(b, u, stat, mode, key, val, maxst, dur, t, MI, MF):
    free = -1
    for j in range(MI.shape[2]):
        if MI[b, u, j, 0] == stat and MI[b, u, j, 1] == mode and MI[b, u, j, 2] == key:
            st = MI[b, u, j, 3] + 1
            if st > maxst:
                st = maxst
            MI[b, u, j, 3] = st
            MF[b, u, j, 0] = val
            if dur > 0:
                MF[b, u, j, 1] = t + dur
            else:
                MF[b, u, j, 1] = -1.0
            return
        if free < 0 and MI[b, u, j, 0] < 0:
            free = j
    if free < 0:
        return
    MI[b, u, free, 0] = stat
    MI[b, u, free, 1] = mode
    MI[b, u, free, 2] = key
    MI[b, u, free, 3] = 1
    MF[b, u, free, 0] = val
    if dur > 0:
        MF[b, u, free, 1] = t + dur
    else:
        MF[b, u, free, 1] = -1.0


@njit
def _row_of(cell, side):
    """Hàng tính từ phía đội mình: 0 = hàng sau cùng, 3 = hàng đầu."""
    r = cell // 7
    if side == 0:
        return r
    return 7 - r


@njit
def _targets(b, u, tg, count, radius, I, F, S, dist, TG):
    """Ghi các quân bị nhắm vào TG[b, :n]; trả về n."""
    n_units = I.shape[1]
    sd = I[b, u, I_SIDE]
    cu = I[b, u, I_CELL]
    n = 0
    if tg == TG_SELF:
        TG[b, 0] = u
        return 1
    t = I[b, u, I_TARGET]
    if tg == TG_TARGET:
        if t >= 0 and I[b, t, I_ALIVE] == 1:
            TG[b, 0] = t
            return 1
        return 0
    if tg == TG_ENEMIES_R or tg == TG_ALL_ENEMIES:
        for v in range(n_units):
            if I[b, v, I_ALIVE] == 1 and I[b, v, I_SIDE] != sd and F[b, v, F_STEALTH] <= 0:
                if tg == TG_ALL_ENEMIES or dist[cu, I[b, v, I_CELL]] <= radius:
                    TG[b, n] = v
                    n += 1
        return n
    if tg == TG_ALLIES_R or tg == TG_ALL_ALLIES or tg == TG_ROW_ALLIES:
        ru = cu // 7
        for v in range(n_units):
            if I[b, v, I_ALIVE] == 1 and I[b, v, I_SIDE] == sd:
                ok = True
                if tg == TG_ALLIES_R:
                    ok = dist[cu, I[b, v, I_CELL]] <= radius
                elif tg == TG_ROW_ALLIES:
                    ok = I[b, v, I_CELL] // 7 == ru
                if ok:
                    TG[b, n] = v
                    n += 1
        return n
    if tg == TG_AROUND_TARGET:
        if t < 0 or I[b, t, I_ALIVE] == 0:
            return 0
        ct = I[b, t, I_CELL]
        for v in range(n_units):
            if I[b, v, I_ALIVE] == 1 and I[b, v, I_SIDE] != sd and F[b, v, F_STEALTH] <= 0:
                if v == t or dist[ct, I[b, v, I_CELL]] <= radius:
                    TG[b, n] = v
                    n += 1
        return n
    if tg == TG_LOWEST_ALLY:
        best = -1
        bv = 2.0
        bo = 1000
        for v in range(n_units):
            if I[b, v, I_ALIVE] == 1 and I[b, v, I_SIDE] == sd:
                frac = F[b, v, F_HP] / max(S[b, v, ST_HP], 1.0)
                o = I[b, v, I_ORD]
                if frac < bv - 1e-6 or (abs(frac - bv) <= 1e-6 and o < bo):
                    best = v
                    bv = frac
                    bo = o
        if best >= 0:
            TG[b, 0] = best
            return 1
        return 0
    if tg == TG_FARTHEST:
        best = -1
        bd = -1
        bo = 1000
        for v in range(n_units):
            if I[b, v, I_ALIVE] == 1 and I[b, v, I_SIDE] != sd and F[b, v, F_STEALTH] <= 0:
                d = dist[cu, I[b, v, I_CELL]]
                o = I[b, v, I_ORD]
                if d > bd or (d == bd and o < bo):
                    best = v
                    bd = d
                    bo = o
        if best >= 0:
            TG[b, 0] = best
            return 1
        return 0
    if tg == TG_NEAREST_N or tg == TG_NEAR_TARGET_N:
        center = cu
        skip = -1
        if tg == TG_NEAR_TARGET_N:
            if t < 0 or I[b, t, I_ALIVE] == 0:
                return 0
            center = I[b, t, I_CELL]
            skip = t
        for k in range(count):
            best = -1
            bd = 1000
            bo = 1000
            for v in range(n_units):
                if I[b, v, I_ALIVE] == 0 or I[b, v, I_SIDE] == sd or F[b, v, F_STEALTH] > 0 or v == skip:
                    continue
                taken = False
                for j in range(n):
                    if TG[b, j] == v:
                        taken = True
                if taken:
                    continue
                d = dist[center, I[b, v, I_CELL]]
                o = I[b, v, I_ORD]
                if d <= radius and (d < bd or (d == bd and o < bo)):
                    best = v
                    bd = d
                    bo = o
            if best < 0:
                break
            TG[b, n] = best
            n += 1
        return n
    if tg == TG_DENSEST:
        best = -1
        bc = -1
        bo = 1000
        for v in range(n_units):
            if I[b, v, I_ALIVE] == 0 or I[b, v, I_SIDE] == sd or F[b, v, F_STEALTH] > 0:
                continue
            c = 0
            for w in range(n_units):
                if I[b, w, I_ALIVE] == 1 and I[b, w, I_SIDE] != sd and F[b, w, F_STEALTH] <= 0:
                    if dist[I[b, v, I_CELL], I[b, w, I_CELL]] <= radius:
                        c += 1
            o = I[b, v, I_ORD]
            if c > bc or (c == bc and o < bo):
                best = v
                bc = c
                bo = o
        if best < 0:
            return 0
        cb = I[b, best, I_CELL]
        for w in range(n_units):
            if I[b, w, I_ALIVE] == 1 and I[b, w, I_SIDE] != sd and F[b, w, F_STEALTH] <= 0:
                if dist[cb, I[b, w, I_CELL]] <= radius:
                    TG[b, n] = w
                    n += 1
        return n
    return 0


@njit
def _push(b, QI, QF, qn, kind, src, dst, stat, mode, key, stacks, abil, amount, dur):
    k = qn[b]
    if k >= QI.shape[1]:
        return
    QI[b, k, Q_KIND] = kind
    QI[b, k, Q_SRC] = src
    QI[b, k, Q_DST] = dst
    QI[b, k, Q_STAT] = stat
    QI[b, k, Q_MODE] = mode
    QI[b, k, Q_KEY] = key
    QI[b, k, Q_STACKS] = stacks
    QI[b, k, Q_ABIL] = abil
    QF[b, k, 0] = amount
    QF[b, k, 1] = dur
    qn[b] = k + 1


@njit
def _crit(b, u, F, S, rng, stochastic):
    """Có chí mạng không. Tất định: cộng dồn tỷ lệ, đủ 1 thì chí mạng (đúng kỳ vọng, không cần RNG)."""
    ch = S[b, u, ST_CRIT_CH]
    if ch <= 0:
        return False
    if stochastic:
        return _rand(rng, b) < ch
    F[b, u, F_CRIT_ACC] += ch
    if F[b, u, F_CRIT_ACC] >= 1.0 - 1e-6:
        F[b, u, F_CRIT_ACC] -= 1.0
        return True
    return False


@njit
def _damage_amount(b, u, v, raw, dtype, F, S, GP, ot):
    if dtype == 0:
        raw = raw * 100.0 / (100.0 + S[b, v, ST_ARMOR])
    elif dtype == 1:
        raw = raw * 100.0 / (100.0 + S[b, v, ST_MR])
    amp = 1.0 + S[b, u, ST_AMP]
    if S[b, v, ST_HP] >= GP[GP_BIG_HP]:
        amp += S[b, u, ST_AMP_BIG]
    if F[b, v, F_SHIELD] > 0:
        amp += S[b, u, ST_AMP_SHIELD]
    dr = S[b, v, ST_DR]
    if dr > 0.9:
        dr = 0.9
    return raw * amp * (1.0 - dr) * ot


@njit
def _fire(b, u, e, lv, EI, EF, I, F, S, dist, TG, QI, QF, qn, rng, stochastic, GP, ot, ev, evc, s, log):
    """Đưa tác động của khối e (bậc lv) do quân u gây ra vào hàng đợi."""
    cond = EI[e, EI_COND]
    if cond != 0:
        row = _row_of(I[b, u, I_CELL], I[b, u, I_SIDE])
        if cond == 1 and row > 1:
            return
        if cond == 2 and row < 2:
            return
    kind = EI[e, EI_KIND]
    n = _targets(b, u, EI[e, EI_TARGET], EI[e, EI_COUNT], EI[e, EI_RADIUS], I, F, S, dist, TG)
    val0 = EF[e, EF_V0 + lv]
    dur = EF[e, EF_DUR]
    sc = EI[e, EI_SCALE]
    abil = 0
    if EI[e, EI_TRIG] == TR_CAST:
        abil = 1
    for j in range(n):
        v = TG[b, j]
        amt = val0
        if sc == SC_AD:
            amt = val0 * S[b, u, ST_AD]
        elif sc == SC_AP:
            amt = val0 * S[b, u, ST_AP] / 100.0
        elif sc == SC_TGT_HP:
            amt = val0 * S[b, v, ST_HP]
        elif sc == SC_SELF_HP:
            amt = val0 * S[b, u, ST_HP]
        elif sc == SC_TGT_MISSING:
            amt = val0 * (S[b, v, ST_HP] - F[b, v, F_HP])
        if kind == K_DAMAGE:
            if abil == 1 and S[b, u, ST_ABIL_CRIT] > 0:
                if _crit(b, u, F, S, rng, stochastic):
                    amt = amt * S[b, u, ST_CRIT_DMG]
            amt = _damage_amount(b, u, v, amt, EI[e, EI_MODE], F, S, GP, ot)
            if log:
                _log(ev, evc, b, s, EV_ABIL, u, v, amt, I[b, v, I_CELL])
        _push(b, QI, QF, qn, kind, u, v, EI[e, EI_STAT], EI[e, EI_MODE], EI[e, EI_KEY], EI[e, EI_STACKS],
              abil, amt, dur)


@njit
def _fire_trigger(b, u, trig, SE, SL, ST, SC, EI, EF, I, F, S, dist, TG, QI, QF, qn, rng, stochastic, GP, ot,
                  ev, evc, s, log):
    for j in range(SE.shape[2]):
        e = SE[b, u, j]
        if e < 0:
            continue
        if EI[e, EI_TRIG] != trig:
            continue
        per = EF[e, EF_PERIOD]
        if trig == TR_ATTACK_N:
            SC[b, u, j] += 1
            n = int(per)
            if n < 1:
                n = 1
            if SC[b, u, j] % n != 0:
                continue
        elif per > 0:                      # thời gian hồi nội tại cho "mỗi đòn", "khi bị đánh"...
            if ST[b, u, j] > 0:
                continue
            ST[b, u, j] = per
        _fire(b, u, e, SL[b, u, j], EI, EF, I, F, S, dist, TG, QI, QF, qn, rng, stochastic, GP, ot,
              ev, evc, s, log)


@njit
def _dash(b, u, v, I, occ, dist, nb):
    """Quân u lướt tới ô trống cạnh v, ưu tiên ô xa chỗ u đang đứng nhất (ra sau lưng v)."""
    if v < 0 or I[b, v, I_ALIVE] == 0:
        return -1
    cu = I[b, u, I_CELL]
    cv = I[b, v, I_CELL]
    sd = I[b, u, I_SIDE]
    best = -1
    bd = -1
    for kk in range(6):
        k = kk
        if sd == 1:
            k = (kk + 3) % 6
        c = nb[cv, k]
        if c < 0 or occ[b, c] >= 0:
            continue
        d = dist[cu, c]
        if d > bd:
            best = c
            bd = d
    if best < 0:
        return -1
    occ[b, cu] = -1
    occ[b, best] = u
    I[b, u, I_CELL] = best
    I[b, u, I_GOAL] = -1
    I[b, u, I_PREV] = -1
    I[b, u, I_TARGET] = v
    return best


@njit
def _apply(b, s, t, QI, QF, qn, I, F, S, BASE, PADD, PMUL, MI, MF, occ, dist, nb, GP, ev, evc, log):
    n_units = I.shape[1]
    nq = qn[b]
    as_cap = GP[GP_AS_CAP]
    # 1. chỉ số
    for k in range(nq):
        if QI[b, k, Q_KIND] == K_STAT:
            v = QI[b, k, Q_DST]
            if I[b, v, I_ALIVE] == 0:
                continue
            _add_mod(b, v, QI[b, k, Q_STAT], QI[b, k, Q_MODE], QI[b, k, Q_KEY], QF[b, k, 0],
                     QI[b, k, Q_STACKS], QF[b, k, 1], t, MI, MF)
            I[b, v, I_DIRTY] = 1
    for v in range(n_units):
        if I[b, v, I_DIRTY] == 1:
            _recompute(b, v, S, BASE, PADD, PMUL, MI, MF, F, I, as_cap)
    # 2. khiên, hồi máu, khống chế, năng lượng
    for k in range(nq):
        kind = QI[b, k, Q_KIND]
        v = QI[b, k, Q_DST]
        src = QI[b, k, Q_SRC]
        if I[b, v, I_ALIVE] == 0:
            continue
        amt = QF[b, k, 0]
        dur = QF[b, k, 1]
        if kind == K_SHIELD:
            F[b, v, F_SHIELD] += amt
            d = dur
            if d <= 0:
                d = 99.0
            if F[b, v, F_SHIELD_T] < d:
                F[b, v, F_SHIELD_T] = d
            if src >= 0:
                F[b, src, F_SHIELDED] += amt
            if log:
                _log(ev, evc, b, s, EV_SHIELD, src, v, amt, I[b, v, I_CELL])
        elif kind == K_HEAL:
            if F[b, v, F_WOUND] > 0:
                amt = amt * (1.0 - GP[GP_WOUND])
            room = S[b, v, ST_HP] - F[b, v, F_HP]
            if amt > room:
                amt = room
            if amt > 0:
                F[b, v, F_HP] += amt
                if src >= 0:
                    F[b, src, F_HEALED] += amt
                if log:
                    _log(ev, evc, b, s, EV_HEAL, src, v, amt, I[b, v, I_CELL])
        elif kind == K_STUN:
            if F[b, v, F_CCIMM] <= 0 and F[b, v, F_STUN] < amt:
                F[b, v, F_STUN] = amt
                if log:
                    _log(ev, evc, b, s, EV_STUN, src, v, amt, I[b, v, I_CELL])
        elif kind == K_STEALTH:
            if F[b, v, F_STEALTH] < amt:
                F[b, v, F_STEALTH] = amt
        elif kind == K_CC_IMMUNE:
            if F[b, v, F_CCIMM] < amt:
                F[b, v, F_CCIMM] = amt
            F[b, v, F_STUN] = 0.0
        elif kind == K_WOUND:
            d = dur
            if d <= 0:
                d = amt
            if F[b, v, F_WOUND] < d:
                F[b, v, F_WOUND] = d
        elif kind == K_BURN:
            if F[b, v, F_BURN] < amt:
                F[b, v, F_BURN] = amt
            if F[b, v, F_BURN_T] < dur:
                F[b, v, F_BURN_T] = dur
        elif kind == K_MANA:
            F[b, v, F_MANA] += amt
    # 3. sát thương
    for k in range(nq):
        if QI[b, k, Q_KIND] != K_DAMAGE:
            continue
        v = QI[b, k, Q_DST]
        src = QI[b, k, Q_SRC]
        if I[b, v, I_ALIVE] == 0:
            continue
        amt = QF[b, k, 0]
        if amt <= 0:
            continue
        F[b, v, F_TAKEN] += amt
        if src >= 0:
            F[b, src, F_DEALT] += amt
            dt_ = QI[b, k, Q_MODE]
            if dt_ == 0:
                F[b, src, F_DEALT_PHYS] += amt
            elif dt_ == 1:
                F[b, src, F_DEALT_MAG] += amt
            else:
                F[b, src, F_DEALT_TRUE] += amt
            if amt > F[b, v, F_HITMAX]:
                F[b, v, F_HITMAX] = amt
                I[b, v, I_LASTHIT] = src
        rest = amt
        sh = F[b, v, F_SHIELD]
        if sh > 0:
            a = min(sh, rest)
            F[b, v, F_SHIELD] = sh - a
            rest = rest - a
        F[b, v, F_HP] -= rest
        if F[b, v, F_LOCK] <= 0:
            g = amt * GP[GP_MANA_DMG]
            if g > GP[GP_MANA_DMG_CAP]:
                g = GP[GP_MANA_DMG_CAP]
            F[b, v, F_MANA] += g
        if src >= 0 and I[b, src, I_ALIVE] == 1:
            vamp = S[b, src, ST_VAMP]
            if vamp > 0:
                h = amt * vamp
                if F[b, src, F_WOUND] > 0:
                    h = h * (1.0 - GP[GP_WOUND])
                room = S[b, src, ST_HP] - F[b, src, F_HP]
                if h > room:
                    h = room
                if h > 0:
                    F[b, src, F_HP] += h
                    F[b, src, F_HEALED] += h
    # 4. lướt
    for k in range(nq):
        if QI[b, k, Q_KIND] != K_DASH:
            continue
        u = QI[b, k, Q_SRC]
        if I[b, u, I_ALIVE] == 0 or F[b, u, F_HP] <= 0:
            continue
        c = _dash(b, u, QI[b, k, Q_DST], I, occ, dist, nb)
        if c >= 0 and log:
            _log(ev, evc, b, s, EV_DASH, u, QI[b, k, Q_DST], 0.0, c)
    qn[b] = 0
    # 5. chết
    for v in range(n_units):
        if I[b, v, I_ALIVE] == 1 and F[b, v, F_HP] <= 0:
            I[b, v, I_ALIVE] = 0
            F[b, v, F_HP] = 0.0
            occ[b, I[b, v, I_CELL]] = -1
            I[b, v, I_DEATH] = s
            k = I[b, v, I_LASTHIT]
            if k >= 0 and I[b, k, I_ALIVE] == 1:
                I[b, k, I_KILLS] += 1
            if log:
                _log(ev, evc, b, s, EV_DEATH, v, -1, 0.0, I[b, v, I_CELL])


@njit
def _init_one(b, champ, star, side, half_cell, seed, BASE, PADD, PMUL, SE, SL, F, I, S, MI, MF, ST, SC, occ,
              rng, evc, qn, EI, EF, rot):
    n_units = F.shape[1]
    for c in range(occ.shape[1]):
        occ[b, c] = -1
    x = np.uint64(seed[b]) ^ np.uint64(0x9E3779B97F4A7C15)
    if x == np.uint64(0):
        x = np.uint64(1)
    rng[b] = x
    evc[b] = 0
    qn[b] = 0
    n0 = 0
    n1 = 0
    for u in range(n_units):
        for j in range(F.shape[2]):
            F[b, u, j] = 0.0
        for j in range(I.shape[2]):
            I[b, u, j] = 0
        for j in range(MI.shape[2]):
            MI[b, u, j, 0] = -1
            MI[b, u, j, 1] = 0
            MI[b, u, j, 2] = -1
            MI[b, u, j, 3] = 0
            MF[b, u, j, 0] = 0.0
            MF[b, u, j, 1] = -1.0
        for j in range(S.shape[2]):
            S[b, u, j] = 0.0
        I[b, u, I_CHAMP] = -1
        I[b, u, I_CELL] = -1
        I[b, u, I_TARGET] = -1
        I[b, u, I_DEATH] = -1
        I[b, u, I_AVOID] = -1
        I[b, u, I_GOAL] = -1
        I[b, u, I_PREV] = -1
        I[b, u, I_LASTHIT] = -1
        c = champ[b, u]
        if c < 0:
            continue
        sd = side[b, u]
        if sd == 0:
            o = n0
            n0 += 1
            cell = half_cell[b, u]
        else:
            o = n1
            n1 += 1
            cell = rot[half_cell[b, u]]
        occ[b, cell] = u
        I[b, u, I_CHAMP] = c
        I[b, u, I_STAR] = star[b, u]
        I[b, u, I_SIDE] = sd
        I[b, u, I_ORD] = o
        I[b, u, I_CELL] = cell
        I[b, u, I_ALIVE] = 1
        for k in range(S.shape[2]):
            S[b, u, k] = (BASE[b, u, k] + PADD[b, u, k]) * (1.0 + PMUL[b, u, k])
        if S[b, u, ST_RANGE] < 1.0:
            S[b, u, ST_RANGE] = 1.0
        if S[b, u, ST_CRIT_CH] > 1.0:
            S[b, u, ST_CRIT_CH] = 1.0
        F[b, u, F_HP] = S[b, u, ST_HP]
        F[b, u, F_MANA] = S[b, u, ST_MANA_START]
        if S[b, u, ST_AS] > 0:
            F[b, u, F_ATK_CD] = 1.0 / S[b, u, ST_AS]
        else:
            F[b, u, F_ATK_CD] = NEVER
        for j in range(SE.shape[2]):
            e = SE[b, u, j]
            ST[b, u, j] = 0.0
            SC[b, u, j] = 0
            if e >= 0 and EI[e, EI_TRIG] == TR_PERIODIC:
                ST[b, u, j] = EF[e, EF_PERIOD]


@njit
def _step_one(b, s, F, I, S, BASE, PADD, PMUL, MI, MF, SE, SL, ST, SC, EI, EF, occ, rng, ev, evc,
              want, key, minkey, cnt, bdist, bpar, bq, TG, QI, QF, qn, dist, nb, rot, GP, stochastic, log):
    n_units = F.shape[1]
    dt = GP[GP_DT]
    move_steps = int(GP[GP_MOVE_STEPS])
    t = s * dt
    ot = 1.0
    if t > GP[GP_OT_START]:
        ot = 1.0 + (t - GP[GP_OT_START]) / GP[GP_OT_SPAN]
    as_cap = GP[GP_AS_CAP]

    # 1. đếm lùi
    for u in range(n_units):
        if I[b, u, I_ALIVE] == 0:
            continue
        F[b, u, F_ATK_CD] -= dt
        if F[b, u, F_MOVE_CD] > 0:
            F[b, u, F_MOVE_CD] -= 1.0
        for col in (F_STUN, F_STEALTH, F_CCIMM, F_WOUND, F_LOCK, F_SHIELD_T, F_BURN_T):
            if F[b, u, col] > 0:
                F[b, u, col] -= dt
        if F[b, u, F_SHIELD_T] <= 0:
            F[b, u, F_SHIELD] = 0.0
        if F[b, u, F_BURN_T] <= 0:
            F[b, u, F_BURN] = 0.0
        F[b, u, F_HITMAX] = 0.0
        I[b, u, I_HITS] = 0
        for j in range(SE.shape[2]):
            if SE[b, u, j] >= 0 and ST[b, u, j] > 0:
                ST[b, u, j] -= dt
        dirty = False
        for j in range(MI.shape[2]):
            if MI[b, u, j, 0] >= 0 and MF[b, u, j, 1] >= 0 and MF[b, u, j, 1] <= t + 1e-6:
                MI[b, u, j, 0] = -1
                MI[b, u, j, 2] = -1
                MI[b, u, j, 3] = 0
                dirty = True
        if dirty:
            _recompute(b, u, S, BASE, PADD, PMUL, MI, MF, F, I, as_cap)

    # 2. hiệu ứng theo thời gian, ngưỡng máu, hạ địch
    for u in range(n_units):
        if I[b, u, I_ALIVE] == 0:
            continue
        for j in range(SE.shape[2]):
            e = SE[b, u, j]
            if e < 0:
                continue
            trig = EI[e, EI_TRIG]
            if trig == TR_PERIODIC:
                if ST[b, u, j] <= EPS:
                    ST[b, u, j] += EF[e, EF_PERIOD]
                    _fire(b, u, e, SL[b, u, j], EI, EF, I, F, S, dist, TG, QI, QF, qn, rng, stochastic, GP, ot,
                          ev, evc, s, log)
            elif trig == TR_AT_TIME:
                if SC[b, u, j] == 0 and t >= EF[e, EF_PERIOD] - 1e-6:
                    SC[b, u, j] = 1
                    _fire(b, u, e, SL[b, u, j], EI, EF, I, F, S, dist, TG, QI, QF, qn, rng, stochastic, GP, ot,
                          ev, evc, s, log)
            elif trig == TR_HP_BELOW:
                if SC[b, u, j] == 0 and F[b, u, F_HP] < EF[e, EF_PERIOD] * S[b, u, ST_HP]:
                    SC[b, u, j] = 1
                    _fire(b, u, e, SL[b, u, j], EI, EF, I, F, S, dist, TG, QI, QF, qn, rng, stochastic, GP, ot,
                          ev, evc, s, log)
        while I[b, u, I_KILLS] > 0:
            I[b, u, I_KILLS] -= 1
            _fire_trigger(b, u, TR_KILL, SE, SL, ST, SC, EI, EF, I, F, S, dist, TG, QI, QF, qn, rng, stochastic,
                          GP, ot, ev, evc, s, log)

    # 3. chọn mục tiêu (giữ mục tiêu cũ tới khi nó chết hoặc mất dấu)
    for u in range(n_units):
        if I[b, u, I_ALIVE] == 0:
            continue
        tt = I[b, u, I_TARGET]
        if tt >= 0 and I[b, tt, I_ALIVE] == 1 and F[b, tt, F_STEALTH] <= 0:
            continue
        cu = I[b, u, I_CELL]
        sd = I[b, u, I_SIDE]
        best = -1
        bd = 1000
        bo = 1000
        for v in range(n_units):
            if I[b, v, I_ALIVE] == 0 or I[b, v, I_SIDE] == sd or F[b, v, F_STEALTH] > 0:
                continue
            d = dist[cu, I[b, v, I_CELL]]
            o = I[b, v, I_ORD]
            if d < bd or (d == bd and o < bo):
                best = v
                bd = d
                bo = o
        I[b, u, I_TARGET] = best
        I[b, u, I_GOAL] = -1

    # 4. tung chiêu
    for u in range(n_units):
        want[b, u] = -2                         # -2 = chưa làm gì trong bước
        if I[b, u, I_ALIVE] == 0 or F[b, u, F_STUN] > 0 or F[b, u, F_LOCK] > 0:
            continue
        mm = S[b, u, ST_MANA_MAX]
        if mm <= 0 or F[b, u, F_MANA] < mm - 1e-6 or I[b, u, I_TARGET] < 0:
            continue
        F[b, u, F_MANA] = 0.0
        F[b, u, F_LOCK] = GP[GP_MANA_LOCK]
        I[b, u, I_CASTS] += 1
        if F[b, u, F_ATK_CD] < GP[GP_CAST_TIME]:
            F[b, u, F_ATK_CD] = GP[GP_CAST_TIME]
        want[b, u] = -3                         # -3 = đã tung chiêu, không đánh / đi
        if log:
            _log(ev, evc, b, s, EV_CAST, u, I[b, u, I_TARGET], 0.0, I[b, u, I_CELL])
        _fire_trigger(b, u, TR_CAST, SE, SL, ST, SC, EI, EF, I, F, S, dist, TG, QI, QF, qn, rng, stochastic,
                      GP, ot, ev, evc, s, log)

    # 5. đánh hoặc xin đi một ô
    for c in range(minkey.shape[1]):
        minkey[b, c] = 1 << 30
        cnt[b, c] = 0
    for u in range(n_units):
        casted = want[b, u] == -3
        want[b, u] = -1
        if casted or I[b, u, I_ALIVE] == 0:
            continue
        if F[b, u, F_MOVE_CD] > 0 or F[b, u, F_STUN] > 0:
            continue
        tt = I[b, u, I_TARGET]
        if tt < 0:
            continue
        cu = I[b, u, I_CELL]
        ct = I[b, tt, I_CELL]
        rng_u = int(S[b, u, ST_RANGE] + 1e-6)
        d = dist[cu, ct]
        if d <= rng_u:
            I[b, u, I_BLOCKED] = 0
            if F[b, u, F_ATK_CD] <= EPS and S[b, u, ST_AS] > 0:
                dmg = S[b, u, ST_AD]
                crit = _crit(b, u, F, S, rng, stochastic)
                if crit:
                    dmg = dmg * S[b, u, ST_CRIT_DMG]
                post = _damage_amount(b, u, tt, dmg, 0, F, S, GP, ot)
                _push(b, QI, QF, qn, K_DAMAGE, u, tt, 0, 0, -1, 1, 0, post, 0.0)
                I[b, tt, I_HITS] += 1
                I[b, u, I_ATKS] += 1
                cd = F[b, u, F_ATK_CD]
                if cd < 0:
                    cd = 0.0
                F[b, u, F_ATK_CD] = cd + 1.0 / S[b, u, ST_AS]
                if F[b, u, F_LOCK] <= 0:
                    F[b, u, F_MANA] += GP[GP_MANA_ATK] + S[b, u, ST_MANA_ATK]
                if log:
                    _log(ev, evc, b, s, EV_ATTACK, u, tt, post, ct)
                _fire_trigger(b, u, TR_ATTACK, SE, SL, ST, SC, EI, EF, I, F, S, dist, TG, QI, QF, qn, rng,
                              stochastic, GP, ot, ev, evc, s, log)
                _fire_trigger(b, u, TR_ATTACK_N, SE, SL, ST, SC, EI, EF, I, F, S, dist, TG, QI, QF, qn, rng,
                              stochastic, GP, ot, ev, evc, s, log)
                if crit:
                    _fire_trigger(b, u, TR_CRIT, SE, SL, ST, SC, EI, EF, I, F, S, dist, TG, QI, QF, qn, rng,
                                  stochastic, GP, ot, ev, evc, s, log)
        else:
            sd = I[b, u, I_SIDE]
            avoid = I[b, u, I_AVOID]
            I[b, u, I_AVOID] = -1
            back = I[b, u, I_PREV]
            if I[b, u, I_BLOCKED] >= move_steps:
                back = -1
            nv = _bfs(b, cu, avoid, back, sd, occ, nb, bdist, bpar, bq)
            goal = I[b, u, I_GOAL]
            if goal >= 0 and (bdist[b, goal] <= 0 or dist[goal, ct] > rng_u):
                goal = -1
            if goal < 0:
                goal = _first_goal(b, nv, ct, rng_u, dist, bq)
            if goal < 0:
                bl = 1000
                bo = 1000
                for v in range(n_units):
                    if I[b, v, I_ALIVE] == 0 or I[b, v, I_SIDE] == sd or F[b, v, F_STEALTH] > 0:
                        continue
                    g = _first_goal(b, nv, I[b, v, I_CELL], rng_u, dist, bq)
                    if g < 0:
                        continue
                    lv = bdist[b, g]
                    o = I[b, v, I_ORD]
                    if lv < bl or (lv == bl and o < bo):
                        bl = lv
                        bo = o
                        goal = g
                        tt = v
                if goal >= 0:
                    I[b, u, I_TARGET] = tt
            I[b, u, I_GOAL] = goal
            if goal < 0:
                I[b, u, I_BLOCKED] += 1
            else:
                plen = bdist[b, goal]
                step = goal
                while bpar[b, step] != cu:
                    step = bpar[b, step]
                want[b, u] = step
                own_cell = cu
                if sd == 1:
                    own_cell = rot[cu]
                kv = np.int64(plen) * 1024 + np.int64(I[b, u, I_ORD]) * 64 + np.int64(own_cell)
                key[b, u] = kv
                if kv < minkey[b, step]:
                    minkey[b, step] = kv
                    cnt[b, step] = 1
                elif kv == minkey[b, step]:
                    cnt[b, step] += 1

    # 6. giải quyết tranh ô
    for u in range(n_units):
        n = want[b, u]
        if n < 0:
            continue
        if key[b, u] == minkey[b, n] and cnt[b, n] == 1:
            old = I[b, u, I_CELL]
            occ[b, old] = -1
            occ[b, n] = u
            I[b, u, I_CELL] = n
            I[b, u, I_PREV] = old
            F[b, u, F_MOVE_CD] = move_steps
            I[b, u, I_BLOCKED] = 0
            if log:
                _log(ev, evc, b, s, EV_MOVE, u, -1, 0.0, n)
        else:
            I[b, u, I_BLOCKED] += 1
            if key[b, u] == minkey[b, n]:
                I[b, u, I_AVOID] = n

    # 7. khi bị đánh
    for u in range(n_units):
        if I[b, u, I_ALIVE] == 1 and I[b, u, I_HITS] > 0:
            for h in range(I[b, u, I_HITS]):
                _fire_trigger(b, u, TR_HIT_TAKEN, SE, SL, ST, SC, EI, EF, I, F, S, dist, TG, QI, QF, qn, rng,
                              stochastic, GP, ot, ev, evc, s, log)

    # 8. thiêu (sát thương chuẩn)
    for u in range(n_units):
        if I[b, u, I_ALIVE] == 1 and F[b, u, F_BURN] > 0:
            _push(b, QI, QF, qn, K_DAMAGE, -1, u, 0, 2, -1, 1, 0, F[b, u, F_BURN] * dt * ot, 0.0)

    # 9. áp cùng lúc, xử lý chết
    _apply(b, s, t, QI, QF, qn, I, F, S, BASE, PADD, PMUL, MI, MF, occ, dist, nb, GP, ev, evc, log)

    a0 = 0
    a1 = 0
    for u in range(n_units):
        if I[b, u, I_ALIVE] == 1:
            if I[b, u, I_SIDE] == 0:
                a0 += 1
            else:
                a1 += 1
    if a0 == 0 and a1 == 0:
        return RES_BOTH_LOSE
    if a1 == 0:
        return RES_SIDE0
    if a0 == 0:
        return RES_SIDE1
    return -1


@njit
def _start(b, F, I, S, BASE, PADD, PMUL, MI, MF, SE, SL, ST, SC, EI, EF, occ, rng, ev, evc, TG, QI, QF, qn,
           dist, nb, GP, stochastic, log):
    """Hiệu ứng "đầu trận" (bước 0): khiên đầu trận, lướt của sát thủ, giảm giáp quanh mình..."""
    for u in range(F.shape[1]):
        if I[b, u, I_ALIVE] == 1:
            _fire_trigger(b, u, TR_START, SE, SL, ST, SC, EI, EF, I, F, S, dist, TG, QI, QF, qn, rng, stochastic,
                          GP, 1.0, ev, evc, 0, log)
    _apply(b, 0, 0.0, QI, QF, qn, I, F, S, BASE, PADD, PMUL, MI, MF, occ, dist, nb, GP, ev, evc, log)


@njit(parallel=True)
def _run_all(champ, star, side, half_cell, seed, BASE, PADD, PMUL, SE, SL, F, I, S, MI, MF, ST, SC, occ, rng,
             ev, evc, want, key, minkey, cnt, bdist, bpar, bq, TG, QI, QF, qn, result, end_step,
             EI, EF, rot, dist, nb, GP, stochastic, log):
    max_steps = int(GP[GP_MAX_STEPS])
    for b in prange(champ.shape[0]):
        _init_one(b, champ, star, side, half_cell, seed, BASE, PADD, PMUL, SE, SL, F, I, S, MI, MF, ST, SC,
                  occ, rng, evc, qn, EI, EF, rot)
        _start(b, F, I, S, BASE, PADD, PMUL, MI, MF, SE, SL, ST, SC, EI, EF, occ, rng, ev, evc, TG, QI, QF, qn,
               dist, nb, GP, stochastic, log)
        r = -1
        last = max_steps
        for s in range(1, max_steps + 1):
            r = _step_one(b, s, F, I, S, BASE, PADD, PMUL, MI, MF, SE, SL, ST, SC, EI, EF, occ, rng, ev, evc,
                          want, key, minkey, cnt, bdist, bpar, bq, TG, QI, QF, qn, dist, nb, rot, GP,
                          stochastic, log)
            if r >= 0:
                last = s
                break
        if r < 0:
            r = RES_BOTH_LOSE
        result[b] = r
        end_step[b] = last


# --------------------------------------------------------------------------- lớp Python
@dataclass
class Batch2:
    champ: np.ndarray       # (B, U) int16, -1 = trống
    star: np.ndarray        # (B, U) int8, 1-3
    side: np.ndarray        # (B, U) int8
    half_cell: np.ndarray   # (B, U) int8
    seed: np.ndarray        # (B,) int64
    base: np.ndarray        # (B, U, NST) float32
    padd: np.ndarray
    pmul: np.ndarray
    slot_eff: np.ndarray    # (B, U, E) int32, -1 = trống
    slot_lv: np.ndarray     # (B, U, E) int8

    @property
    def size(self):
        return self.champ.shape[0]


@dataclass
class State2:
    F: np.ndarray
    I: np.ndarray
    S: np.ndarray
    MI: np.ndarray
    MF: np.ndarray
    ST: np.ndarray
    SC: np.ndarray
    occ: np.ndarray
    rng: np.ndarray
    ev: np.ndarray
    evc: np.ndarray
    want: np.ndarray
    key: np.ndarray
    minkey: np.ndarray
    cnt: np.ndarray
    bdist: np.ndarray
    bpar: np.ndarray
    bq: np.ndarray
    TG: np.ndarray
    QI: np.ndarray
    QF: np.ndarray
    qn: np.ndarray


@dataclass
class Result2:
    result: np.ndarray        # (B,) 0/1 = phe thắng, 2 = cả hai cùng thua
    end_step: np.ndarray
    alive: np.ndarray         # (B, U) bool
    star: np.ndarray
    dealt: np.ndarray         # (B, U, 3) vật lý / phép / chuẩn
    taken: np.ndarray
    healed: np.ndarray
    shielded: np.ndarray
    casts: np.ndarray
    death_step: np.ndarray
    events: list = None


class Simulator2:
    """Trình mô phỏng nấc 2. gp: dict tham số chung (xem DEFAULT_GP)."""
    DEFAULT_GP = dict(dt=0.1, move_time=0.5, max_time=40.0, overtime_start=20.0, overtime_span=5.0,
                      mana_per_attack=10.0, mana_lock=1.0, mana_from_damage=0.02, mana_from_damage_cap=15.0,
                      wound=0.33, big_hp=1600.0, cast_time=0.3, as_cap=5.0)

    def __init__(self, EI, EF, gp=None, stochastic=False, log=False, max_events=8192):
        g = dict(self.DEFAULT_GP)
        g.update(gp or {})
        self.g = g
        GP = np.zeros(NGP, np.float32)
        GP[GP_DT] = g["dt"]
        GP[GP_MOVE_STEPS] = max(1, int(round(g["move_time"] / g["dt"])))
        GP[GP_MAX_STEPS] = int(round(g["max_time"] / g["dt"]))
        GP[GP_OT_START] = g["overtime_start"]
        GP[GP_OT_SPAN] = g["overtime_span"]
        GP[GP_MANA_ATK] = g["mana_per_attack"]
        GP[GP_MANA_LOCK] = g["mana_lock"]
        GP[GP_MANA_DMG] = g["mana_from_damage"]
        GP[GP_MANA_DMG_CAP] = g["mana_from_damage_cap"]
        GP[GP_WOUND] = g["wound"]
        GP[GP_BIG_HP] = g["big_hp"]
        GP[GP_CAST_TIME] = g["cast_time"]
        GP[GP_AS_CAP] = g["as_cap"]
        self.GP = GP
        self.EI = np.ascontiguousarray(EI, np.int32)
        self.EF = np.ascontiguousarray(EF, np.float32)
        self.stochastic = bool(stochastic)
        self.log = bool(log)
        self.max_events = int(max_events) if log else 1

    @property
    def max_steps(self):
        return int(self.GP[GP_MAX_STEPS])

    def _alloc(self, n):
        return State2(
            F=np.zeros((n, U, NF), np.float32), I=np.zeros((n, U, NI), np.int16),
            S=np.zeros((n, U, NST), np.float32),
            MI=np.zeros((n, U, M_SLOTS, 4), np.int16), MF=np.zeros((n, U, M_SLOTS, 2), np.float32),
            ST=np.zeros((n, U, E_SLOTS), np.float32), SC=np.zeros((n, U, E_SLOTS), np.int32),
            occ=np.full((n, N_CELLS), -1, np.int8), rng=np.zeros(n, np.uint64),
            ev=np.zeros((n, self.max_events, 6), np.float32), evc=np.zeros(n, np.int32),
            want=np.full((n, U), -1, np.int16), key=np.zeros((n, U), np.int32),
            minkey=np.zeros((n, N_CELLS), np.int32), cnt=np.zeros((n, N_CELLS), np.int16),
            bdist=np.zeros((n, N_CELLS), np.int16), bpar=np.zeros((n, N_CELLS), np.int16),
            bq=np.zeros((n, N_CELLS), np.int16), TG=np.zeros((n, U), np.int16),
            QI=np.zeros((n, MAXQ, NQI), np.int32), QF=np.zeros((n, MAXQ, NQF), np.float32),
            qn=np.zeros(n, np.int32))

    def validate(self, batch):
        used = batch.champ >= 0
        if np.any((batch.star[used] < 1) | (batch.star[used] > 3)):
            raise ValueError("Số sao phải trong 1-3")
        if np.any((batch.half_cell[used] < 0) | (batch.half_cell[used] >= N_HALF)):
            raise ValueError("Ô nửa sân phải trong 0-27")
        board = np.where(batch.side == 0, batch.half_cell, hexgrid.ROT[np.clip(batch.half_cell, 0, N_HALF - 1)])
        for b in range(batch.size):
            cells = board[b][used[b]]
            if len(np.unique(cells)) != len(cells):
                raise ValueError(f"Trận {b}: hai tướng cùng một ô")
        if np.any(batch.slot_eff >= len(self.EI)):
            raise ValueError("Mã khối hiệu ứng vượt bảng")

    def _args(self, batch, st):
        return (batch.champ, batch.star, batch.side, batch.half_cell, batch.seed, batch.base, batch.padd,
                batch.pmul, batch.slot_eff, batch.slot_lv)

    def run(self, batch):
        self.validate(batch)
        n = batch.size
        st = self._alloc(n)
        result = np.zeros(n, np.int8)
        end_step = np.zeros(n, np.int16)
        _run_all(batch.champ, batch.star, batch.side, batch.half_cell, batch.seed, batch.base, batch.padd,
                 batch.pmul, batch.slot_eff, batch.slot_lv, st.F, st.I, st.S, st.MI, st.MF, st.ST, st.SC,
                 st.occ, st.rng, st.ev, st.evc, st.want, st.key, st.minkey, st.cnt, st.bdist, st.bpar, st.bq,
                 st.TG, st.QI, st.QF, st.qn, result, end_step, self.EI, self.EF, hexgrid.ROT, hexgrid.DIST,
                 hexgrid.NEIGHBORS, self.GP, self.stochastic, self.log)
        return self.collect(st, result, end_step)

    # ---- từng bước (kiểm thử, trình xem lại)
    def init_state(self, batch):
        self.validate(batch)
        st = self._alloc(batch.size)
        for b in range(batch.size):
            _init_one(b, batch.champ, batch.star, batch.side, batch.half_cell, batch.seed, batch.base, batch.padd,
                      batch.pmul, batch.slot_eff, batch.slot_lv, st.F, st.I, st.S, st.MI, st.MF, st.ST, st.SC,
                      st.occ, st.rng, st.evc, st.qn, self.EI, self.EF, hexgrid.ROT)
            _start(b, st.F, st.I, st.S, batch.base, batch.padd, batch.pmul, st.MI, st.MF, batch.slot_eff,
                   batch.slot_lv, st.ST, st.SC, self.EI, self.EF, st.occ, st.rng, st.ev, st.evc, st.TG, st.QI,
                   st.QF, st.qn, hexgrid.DIST, hexgrid.NEIGHBORS, self.GP, self.stochastic, self.log)
        self._batch = batch
        return st

    def step(self, st, s, active=None):
        batch = self._batch
        n = st.F.shape[0]
        out = np.full(n, -1, np.int8)
        for b in range(n):
            if active is not None and not active[b]:
                continue
            out[b] = _step_one(b, s, st.F, st.I, st.S, batch.base, batch.padd, batch.pmul, st.MI, st.MF,
                               batch.slot_eff, batch.slot_lv, st.ST, st.SC, self.EI, self.EF, st.occ, st.rng,
                               st.ev, st.evc, st.want, st.key, st.minkey, st.cnt, st.bdist, st.bpar, st.bq,
                               st.TG, st.QI, st.QF, st.qn, hexgrid.DIST, hexgrid.NEIGHBORS, hexgrid.ROT, self.GP,
                               self.stochastic, self.log)
        return out

    def collect(self, st, result, end_step):
        I, F = st.I, st.F
        events = None
        if self.log:
            events = [st.ev[b, :min(st.evc[b], self.max_events)].copy() for b in range(len(result))]
        return Result2(result=result, end_step=end_step, alive=I[:, :, I_ALIVE] == 1,
                       star=I[:, :, I_STAR].astype(np.int8),
                       dealt=np.stack([F[:, :, F_DEALT_PHYS], F[:, :, F_DEALT_MAG], F[:, :, F_DEALT_TRUE]], -1),
                       taken=F[:, :, F_TAKEN].copy(), healed=F[:, :, F_HEALED].copy(),
                       shielded=F[:, :, F_SHIELDED].copy(), casts=I[:, :, I_CASTS].copy(),
                       death_step=I[:, :, I_DEATH].copy(), events=events)
