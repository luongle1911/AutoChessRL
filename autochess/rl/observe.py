"""Quan sát cho agent: một bộ mảng nhìn từ ghế của từng người chơi (không phải một hàng số phẳng).

    obs = observe(catalog, state, combat, matchup_row=True, seats=None, version=2)

seats = None : mọi ghế, mảng có dạng [B, P, ...]
seats = [B, S] (int): chỉ các ghế được chọn ở từng ván, mảng có dạng [B, S, ...]
version = 1  : quan sát của mạng bản 1 (EntityNetV1, FlatNet), có cột theo từng tướng và từng tộc
version = 2  : quan sát của mạng bản 2 (EntityNetV2), không còn cột nào theo từng tướng hay từng tộc
version = 3  : bản 2 + hàng chờ của đối thủ (giai đoạn 3, như ĐTCL thật); kho "nhìn thấy được" trừ thêm
               quân trên hàng chờ đối thủ
version = 4  : bản 3 + chỉ số (luật bản 4, catalog có stats): cùng các khóa, nhưng
               units [.., 18, 14]  thêm 5 chỉ số của quân (trên sân: chỉ số cuối sau sao và tộc hệ;
                                   hàng chờ: chỉ số gốc x sao), chia STAT_UNIT
               shop  [.., 5, 14]   thêm 5 chỉ số gốc (1 sao) + mức tăng độ mạnh đội nếu đặt nó lên sân
                                   (vào ô trống, hoặc thay quân yếu nhất khi sân đủ), như bot luật chấm
               self  [.., 22]      thêm 5 chỉ số đội (tổng máu, tổng STVL, tổng STP, giáp TB, kháng phép TB;
                                   chia STAT_TEAM) + tỷ lệ 5 hệ vai trò trên sân
               opp   [.., O, 19]   như self: thêm chỉ số đội và tỷ lệ hệ của từng đối thủ (thấy được từ sân)
               traits              cột mức kích hoạt chia cho số mốc của tộc đó (tộc có 1-4 mốc)
               Bảng hiệu ứng tộc (mốc, hiệu ứng theo chỉ số, phạm vi) là dữ liệu tĩnh: mạng đọc từ catalog.

Các khóa (O = P - 1 đối thủ, xếp theo ghế tương đối: ghế kế tiếp trước; K = số tộc):
    khóa            bản 1               bản 2
    self            [.., 11 + 2K + N]   [.., 12]          chỉ số bản thân
    units           [.., 18, 8]         [.., 18, 9]       18 ô quân của mình (bản 2 thêm tỷ lệ còn trong kho)
    unit_id         [.., 18]            như bản 1         int16 id tướng, -1 = ô trống
    shop            [.., 5, 8]          như bản 1         5 ô shop
    shop_id         [.., 5]             như bản 1         int16
    traits          -                   [.., K, 5]        tộc của mình (xem TRAIT_F)
    opp             [.., O, 9 + K]      [.., O, 9]        chỉ số từng đối thủ
    opp_traits      -                   [.., O, K, 5]     tộc của từng đối thủ (từ sân, thấy được)
    opp_board_id    [.., O, 9]          như bản 1         int16 quân trên sân đối thủ
    opp_board_star  [.., O, 9]          như bản 1         int8 0/1/2 = 1/2/3 sao (0 cho ô trống)
    opp_board_pool  -                   [.., O, 9]        tỷ lệ còn trong kho của quân đó
    opp_alive       [.., O]             như bản 1         bool
    mask            [.., 44]            như bản 1         bool
    opp_bench_id    -                   bản 3: [.., O, 9] int16 quân trên hàng chờ đối thủ
    opp_bench_star  -                   bản 3: [.., O, 9] int8
    opp_bench_pool  -                   bản 3: [.., O, 9] tỷ lệ còn trong kho của quân đó

critic_info (chỉ cho critic, xem nets.EntityNetV2; mặc định "none" = không có khóa này):
    crit_opp        [.., O, 3]   "pairing": đối thủ thật của vòng này (một-nóng), có phải ghost không;
                                 "pairing+dice": thêm số ngẫu nhiên quyết định trận đó, quy về phía mình
                                 (mình thắng khi số này < xác suất thắng). Cả hai băm từ seed, số vòng và cặp
                                 ghế, không phụ thuộc hành động trong lượt.

Không chứa thông tin ẩn (ngoài crit_opp): shop của đối thủ (bản 1-2: cả hàng chờ của đối thủ), đối thủ thật của vòng này.
Mọi số được chia bằng hằng số cố định (không chuẩn hóa theo thống kê chạy).
"""
from __future__ import annotations

import numpy as np

from autochess.core.actions import legal_actions
from autochess.core.catalog import BENCH_SIZE, NUM_SLOTS, SHOP_SIZE, STAR_COPIES
from autochess.core.combat import board_features
from autochess.core import rng as crng
from autochess.core.pairing import draw_pairs, meet_prob

UNIT_F = 8   # sao (3, một-nóng), đang ở sân, ô trống, giá/5, số bản gốc mình có/9, sức mạnh/100
UNIT_F2 = 9  # bản 2: thêm tỷ lệ bản còn trong kho (theo kho nhìn thấy được) của tướng trong ô
SHOP_F = 8   # ô trống, giá/5, đủ vàng, số bản mình có/9, mua thì ghép sao, mua thì thành cặp,
             # số tộc của nó đang có trên sân mình/2, tỷ lệ bản còn lại (theo kho nhìn thấy được)
TRAIT_F = 5  # số tướng khác nhau trên sân/6, mức kích hoạt/3, còn thiếu tới mốc kế/6 (0 nếu đã tối đa),
             # mức thưởng đang có x2, đang kích hoạt. Tộc có số tướng = 0 coi là trống.
SELF_BASE = 11
SELF2 = 12   # bản 2: 11 số đầu như bản 1 + log sức mạnh đội/7
OPP_BASE = 9
# bản 4 (luật có chỉ số)
N_ROLES = 5
STAT_UNIT = np.array([2000.0, 200.0, 200.0, 100.0, 100.0], np.float32)     # máu, STVL, STP, giáp, kháng phép
STAT_TEAM = np.array([10000.0, 1000.0, 1000.0, 100.0, 100.0], np.float32)  # tổng máu/STVL/STP, giáp/kháng TB
TEAM_KEYS = ("team_hp", "team_ad", "team_ap", "team_armor", "team_mr")
UNIT_F4 = UNIT_F2 + 5
SHOP_F4 = SHOP_F + 6
SELF4 = SELF2 + 5 + N_ROLES
OPP4 = OPP_BASE + 5 + N_ROLES
KEYS = ("self", "units", "unit_id", "shop", "shop_id", "opp", "opp_board_id", "opp_board_star",
        "opp_alive", "mask")
KEYS2 = ("self", "units", "unit_id", "shop", "shop_id", "traits", "opp", "opp_traits", "opp_board_id",
         "opp_board_star", "opp_board_pool", "opp_alive", "mask")
KEYS3 = KEYS2 + ("opp_bench_id", "opp_bench_star", "opp_bench_pool")
# các khóa có chiều đối thủ ngay sau chiều ghế (dùng khi hoán vị đối thủ trong kiểm thử, và khi đệm
# quan sát ván 4 người lên 7 đối thủ để học chung với ván 8 người)
OPP_KEYS = ("opp", "opp_traits", "opp_board_id", "opp_board_star", "opp_board_pool", "opp_alive",
            "opp_bench_id", "opp_bench_star", "opp_bench_pool", "crit_opp")
CRITIC_INFO = ("none", "pairing", "pairing+dice")
# giá trị đệm cho một "đối thủ đã bị loại" (mặc định 0 / False)
PAD_VALUE = {"opp_board_id": -1, "opp_bench_id": -1}

# chỉ số cột trong `opp` (mạng dùng MEET_COL để gộp theo xác suất gặp)
OPP_ALIVE_COL, MEET_COL, WIN_COL = 0, 6, 7


def keys(version: int = 1) -> tuple:
    return {1: KEYS, 2: KEYS2, 3: KEYS3, 4: KEYS3}[version]


def self_dim(cat) -> int:
    return SELF_BASE + 2 * cat.K + cat.N


def opp_dim(cat) -> int:
    return OPP_BASE + cat.K


def opponent_index(P: int) -> np.ndarray:
    """[P, P-1]: đối thủ thứ o của ghế p là ghế (p + 1 + o) mod P."""
    return (np.arange(P)[:, None] + 1 + np.arange(P - 1)[None, :]) % P


def observe(cat, state, combat=None, matchup_row: bool = True, seats=None, mask=None,
            feats=None, version: int = 1, critic_info: str = "none") -> dict:
    if version not in (1, 2, 3, 4):
        raise ValueError(f"không có quan sát bản {version}")
    if version == 4 and cat.stats is None:
        raise ValueError("quan sát bản 4 cần luật có chỉ số (catalog_v4)")
    if critic_info not in CRITIC_INFO:
        raise ValueError(f"critic_info phải là một trong {CRITIC_INFO}")
    B, P = state.B, state.P
    N, K = cat.N, cat.K
    f = board_features(cat, state) if feats is None else feats
    m = legal_actions(cat, state) if mask is None else mask

    # ---------------- quân của mình
    u = state.units.astype(np.int64)                                     # [B,P,18]
    occ = u >= 0
    champ = np.where(occ, u // 3, 0)
    star = np.where(occ, u % 3, 0)
    copies = np.where(occ, STAR_COPIES[star], 0)
    onehot = (champ[..., None] == np.arange(N)) & occ[..., None]         # [B,P,18,N]
    owned = (onehot * copies[..., None]).sum(-2)                          # [B,P,N] bản gốc
    unit_owned = np.take_along_axis(owned, champ, -1)                     # [B,P,18]
    on_board = np.zeros(NUM_SLOTS, bool)
    on_board[BENCH_SIZE:] = True
    units = np.zeros((B, P, NUM_SLOTS, UNIT_F), np.float32)
    for s in range(3):
        units[..., s] = occ & (star == s)
    units[..., 3] = occ & on_board
    units[..., 4] = ~occ
    units[..., 5] = np.where(occ, cat.cost[champ] / 5.0, 0)
    units[..., 6] = np.where(occ, unit_owned / 9.0, 0)
    units[..., 7] = np.where(occ, cat.power[champ, star] / 100.0, 0)
    unit_id = np.where(occ, champ, -1).astype(np.int16)

    # ---------------- bản thân
    gold = state.gold.astype(np.float32)
    lvl = state.level.astype(np.int64)
    xp_next = cat.xp_to_next[lvl - 1].astype(np.float32)
    pool_size = cat.pool_size(P).astype(np.float32)                       # [N]
    # Kho "nhìn thấy được": kích thước kho trừ những bản mình thấy (quân của mình, shop của mình,
    # sân đối thủ). Không dùng state.pool trực tiếp vì nó lộ shop và hàng chờ của đối thủ.
    oi = opponent_index(P)                                                # [P,O]
    board_copies = (onehot[:, :, BENCH_SIZE:] * copies[:, :, BENCH_SIZE:, None]).sum(-2)   # [B,P,N]
    sh0 = state.shop.astype(np.int64)
    shop_counts = ((sh0[..., None] == np.arange(N)) & (sh0[..., None] >= 0)).sum(-2)       # [B,P,N]
    seen = owned + shop_counts + board_copies[:, oi].sum(2)
    if version >= 3:
        bench_copies = (onehot[:, :, :BENCH_SIZE] * copies[:, :, :BENCH_SIZE, None]).sum(-2)
        seen = seen + bench_copies[:, oi].sum(2)
    pool_frac = np.clip(pool_size - seen, 0, None) / pool_size            # [B,P,N]
    n_bench = occ[..., :BENCH_SIZE].sum(-1)
    me = np.zeros((B, P, self_dim(cat) if version == 1 else SELF2), np.float32)
    me[..., 0] = np.minimum(gold, 150) / 50.0
    me[..., 1] = np.minimum(state.gold // cat.interest_step, cat.interest_max) / 5.0
    me[..., 2] = lvl / 9.0
    me[..., 3] = np.where(xp_next > 0, state.xp / np.maximum(xp_next, 1), 0)
    me[..., 4] = state.hp / float(cat.start_hp)
    me[..., 5] = np.clip(state.streak / 5.0, -1.2, 1.2)
    me[..., 6] = state.round[:, None] / 30.0
    me[..., 7] = (cat.max_actions_per_round - state.actions_this_round) / float(cat.max_actions_per_round)
    me[..., 8] = f["n_board"] / np.maximum(lvl, 1)
    me[..., 9] = n_bench / 9.0
    me[..., 10] = state.alive
    if version == 1:
        me[..., 11:11 + K] = f["trait_count"] / 6.0
        me[..., 11 + K:11 + 2 * K] = f["trait_level"] / 3.0
        me[..., 11 + 2 * K:] = pool_frac
    else:
        me[..., 11] = np.log(f["strength"]) / 7.0
        unit_pool = np.where(occ, np.take_along_axis(pool_frac, champ, -1), 0)
        units = np.concatenate([units, unit_pool[..., None].astype(np.float32)], -1)
    if version == 4:
        us = np.zeros((B, P, NUM_SLOTS, 5), np.float32)
        mult = cat.star_mult[star[..., :BENCH_SIZE]][..., None]                            # sao: máu, STVL, STP
        bench_st = cat.stats[champ[..., :BENCH_SIZE]] * np.where(np.arange(5) < 3, mult, 1.0)
        us[..., :BENCH_SIZE, :] = np.where(occ[..., :BENCH_SIZE, None], bench_st, 0)
        us[..., BENCH_SIZE:, :] = f["unit_stats"]
        units = np.concatenate([units, us / STAT_UNIT], -1)
        tsv = np.stack([f[k] for k in TEAM_KEYS], -1).astype(np.float32) / STAT_TEAM      # [B,P,5]
        role = np.zeros((B, P, N_ROLES), np.float32)
        rs = f["role_share"]
        role[..., :rs.shape[-1]] = rs
        team_vec = np.concatenate([tsv, role], -1)                                         # [B,P,10]
        me = np.concatenate([me, team_vec], -1)

    # ---------------- shop
    sh = state.shop.astype(np.int64)                                      # [B,P,5]
    has = sh >= 0
    c = np.where(has, sh, 0)
    c_cost = cat.cost[c]
    c_owned = np.take_along_axis(owned, c, -1)
    mine = (f["trait_count"] > 0)                                          # [B,P,K]
    overlap = (cat.trait_matrix[c].astype(bool) & mine[:, :, None, :]).sum(-1)
    pool_left = np.take_along_axis(pool_frac, c, -1)
    shop = np.zeros((B, P, SHOP_SIZE, SHOP_F), np.float32)
    shop[..., 0] = ~has
    shop[..., 1] = np.where(has, c_cost / 5.0, 0)
    shop[..., 2] = has & (state.gold[..., None] >= c_cost)
    shop[..., 3] = np.where(has, c_owned / 9.0, 0)
    shop[..., 4] = has & (c_owned % 3 == 2) & (c_owned < 9)
    shop[..., 5] = has & (c_owned % 3 == 1) & (c_owned < 9)
    shop[..., 6] = np.where(has, overlap / 2.0, 0)
    shop[..., 7] = np.where(has, pool_left, 0)       # tỷ lệ còn lại (theo kho nhìn thấy được)
    shop_id = np.where(has, sh, -1).astype(np.int16)
    if version == 4:
        base_st = np.where(has[..., None], cat.stats[c], 0) / STAT_UNIT
        if seats is None:
            gain = shop_rating_gain(cat, u[..., BENCH_SIZE:], c, has, f["n_board"], f["strength"], lvl)
        else:   # chỉ tính cho các ghế được hỏi (phần đắt nhất của quan sát bản 4)
            sq = np.asarray(seats, np.int64)
            bq = np.arange(B)[:, None]
            gain = np.zeros((B, P, SHOP_SIZE))
            gain[bq, sq] = shop_rating_gain(cat, u[bq, sq, BENCH_SIZE:], c[bq, sq], has[bq, sq],
                                            f["n_board"][bq, sq], f["strength"][bq, sq], lvl[bq, sq])
        shop = np.concatenate([shop, base_st.astype(np.float32), (gain / 20.0)[..., None].astype(np.float32)], -1)

    # ---------------- đối thủ
    pr = np.arange(P)[:, None]
    alive = state.alive
    o_alive = alive[:, oi]                                                # [B,P,O]
    meet = meet_prob(state)[:, pr, oi]
    if matchup_row and combat is not None:
        win = combat.evaluate(cat, state)[0][:, pr, oi] * o_alive
    else:
        win = np.zeros_like(meet)
    opp = np.zeros((B, P, P - 1, opp_dim(cat) if version == 1 else OPP_BASE), np.float32)
    opp[..., OPP_ALIVE_COL] = o_alive
    opp[..., 1] = state.hp[:, oi] / float(cat.start_hp)
    opp[..., 2] = state.level[:, oi] / 9.0
    opp[..., 3] = np.minimum(state.gold[:, oi], 150) / 50.0
    opp[..., 4] = np.clip(state.streak[:, oi] / 5.0, -1.2, 1.2)
    opp[..., 5] = np.log(f["strength"])[:, oi] / 7.0 * o_alive
    opp[..., MEET_COL] = meet
    opp[..., WIN_COL] = win
    opp[..., 8] = state.last_opponent[:, :, None] == oi[None]
    if version == 1:
        opp[..., OPP_BASE:] = f["trait_level"][:, oi, :] / 3.0
    if version == 4:
        opp = np.concatenate([opp, team_vec[:, oi] * o_alive[..., None]], -1)
    board = u[..., BENCH_SIZE:]                                           # [B,P,9]
    bocc = board >= 0
    opp_board_id = np.where(bocc, board // 3, -1)[:, oi].astype(np.int16)  # [B,P,O,9]
    opp_board_star = np.where(bocc, board % 3, 0)[:, oi].astype(np.int8)

    if version == 1:
        out = dict(self=me, units=units, unit_id=unit_id, shop=shop, shop_id=shop_id, opp=opp,
                   opp_board_id=opp_board_id, opp_board_star=opp_board_star, opp_alive=o_alive, mask=m)
    else:
        tr = trait_features(cat, f, per_level=version == 4)               # [B,P,K,5]
        O = P - 1
        c = np.clip(opp_board_id, 0, None).astype(np.int64)
        pf = np.broadcast_to(pool_frac[:, :, None, :], (B, P, O, N))
        opp_board_pool = np.where(opp_board_id >= 0, np.take_along_axis(pf, c, -1), 0).astype(np.float32)
        out = dict(self=me, units=units, unit_id=unit_id, shop=shop, shop_id=shop_id, traits=tr,
                   opp=opp, opp_traits=tr[:, oi], opp_board_id=opp_board_id, opp_board_star=opp_board_star,
                   opp_board_pool=opp_board_pool, opp_alive=o_alive, mask=m)
        if version >= 3:
            bench = u[..., :BENCH_SIZE]
            bo = bench >= 0
            ob_id = np.where(bo, bench // 3, -1)[:, oi].astype(np.int16)        # [B,P,O,9]
            out["opp_bench_id"] = ob_id
            out["opp_bench_star"] = np.where(bo, bench % 3, 0)[:, oi].astype(np.int8)
            cb = np.clip(ob_id, 0, None).astype(np.int64)
            out["opp_bench_pool"] = np.where(ob_id >= 0, np.take_along_axis(pf, cb, -1), 0).astype(np.float32)
    if critic_info != "none":
        out["crit_opp"] = critic_opponent_info(state, oi, dice=critic_info == "pairing+dice")
    if seats is not None:
        seats = np.asarray(seats, np.int64)
        out = select_seats(out, seats)
    return out


def critic_opponent_info(state, oi: np.ndarray, dice: bool) -> np.ndarray:
    """[B, P, O, 3]: (là đối thủ thật vòng này, là ghost, số ngẫu nhiên của trận quy về phía mình).
    Lõi bốc cặp và xúc xắc lúc giao tranh từ đúng các giá trị này (seed, số vòng, người còn sống, đối thủ vòng
    trước), mà các giá trị đó không đổi trong lượt, nên tính trước được và không phụ thuộc hành động."""
    B, P = state.alive.shape
    popp, pgh = draw_pairs(state, np.ones(B, bool))                       # [B,P]
    tgt = oi[None]                                                        # [1,P,O]
    is_opp = popp[:, :, None].astype(np.int64) == tgt
    gh = pgh[:, :, None] & is_opp
    out = np.zeros(is_opp.shape + (3,), np.float32)
    out[..., 0] = is_opp
    out[..., 1] = gh
    if dice:
        pp = np.arange(P)[None, :, None]
        lo, hi = np.minimum(pp, tgt), np.maximum(pp, tgt)
        u = crng.uniform(state.seed[:, None, None], crng.COMBAT, state.round[:, None, None], lo, hi,
                         np.where(gh, pp + 1, 0))
        # lõi: ghế lo thắng khi u < p(lo thắng hi); ghost: mình thắng khi u < p(mình thắng)
        out[..., 2] = np.where(gh | (pp < tgt), u, 1.0 - u) * is_opp
    return out


def shop_rating_gain(cat, board, c, has, n_board, strength, lvl) -> np.ndarray:
    """[B, P, 5] mức tăng độ mạnh đội (team_rating) khi đặt quân 1 sao của từng ô shop lên sân: vào ô trống đầu
    tiên nếu sân còn chỗ theo cấp, không thì thay quân yếu nhất (như bot luật chấm khi mua)."""
    from autochess.core.combat import board_rating
    occ = board >= 0
    base = np.where(n_board > 0, strength, 0.0)
    ch = np.where(occ, board // 3, 0)
    pw = np.where(occ, cat.power[ch, np.where(occ, board % 3, 0)], np.inf)
    weakest = pw.argmin(-1)
    slot = np.where((n_board < lvl) & (~occ).any(-1), (~occ).argmax(-1), weakest)              # [B,P]
    S = c.shape[-1]
    cand = np.repeat(board[..., None, :], S, -2).copy()                                         # [B,P,5,9]
    np.put_along_axis(cand, np.broadcast_to(slot[..., None, None], cand.shape[:-1] + (1,)), (c * 3)[..., None], -1)
    g = board_rating(cat, cand) - base[..., None]
    return np.where(has, g, 0.0)


def trait_features(cat, f: dict, per_level: bool = False) -> np.ndarray:
    """[B, P, K, TRAIT_F] từ board_features: đặc trưng của từng tộc trên sân mỗi người.
    per_level (bản 4): cột mức kích hoạt chia cho số mốc của tộc thay vì 3."""
    cnt = f["trait_count"].astype(np.int64)                               # [B,P,K]
    lvl = f["trait_level"].astype(np.int64)
    bp = np.asarray(cat.trait_breakpoints, np.int64)                      # [K,3]
    nxt = np.concatenate([bp, bp[:, -1:]], 1)                             # mốc kế (mức 3: không còn)
    nxt_bp = np.take_along_axis(np.broadcast_to(nxt, cnt.shape[:-1] + nxt.shape), lvl[..., None], -1)[..., 0]
    n_lv = np.asarray(cat.trait_levels if cat.trait_levels is not None else np.full(cat.K, bp.shape[1]), np.int64)
    missing = np.where(lvl < n_lv, nxt_bp - cnt, 0)
    bonus_tab = np.concatenate([np.zeros((cat.K, 1), np.float32), np.asarray(cat.trait_bonus, np.float32)], 1)
    bonus = np.take_along_axis(np.broadcast_to(bonus_tab, cnt.shape[:-1] + bonus_tab.shape), lvl[..., None], -1)[..., 0]
    out = np.zeros(cnt.shape + (TRAIT_F,), np.float32)
    out[..., 0] = cnt / 6.0
    out[..., 1] = lvl / (n_lv if per_level else 3.0)
    out[..., 2] = missing / 6.0
    out[..., 3] = bonus * 2.0
    out[..., 4] = lvl > 0
    return out


def select_seats(obs: dict, seats: np.ndarray) -> dict:
    """Lấy các ghế seats[B, S] từ quan sát của mọi ghế [B, P, ...] -> [B, S, ...]."""
    b = np.arange(seats.shape[0])[:, None]
    return {k: v[b, seats] for k, v in obs.items()}


def pad_opponents(obs: dict, O: int) -> dict:
    """Đệm chiều đối thủ (ngay sau chiều mẫu, mảng [M, O', ...]) lên O bằng "đối thủ đã bị loại":
    dùng để học chung ván 4 người với ván 8 người. Mạng bỏ qua đối thủ có opp_alive = False."""
    out = {}
    for k, v in obs.items():
        if k in OPP_KEYS and v.shape[1] < O:
            pad = np.full((v.shape[0], O - v.shape[1]) + v.shape[2:], PAD_VALUE.get(k, 0), v.dtype)
            v = np.concatenate([v, pad], 1)
        out[k] = v
    return out


def flatten_batch(obs: dict) -> dict:
    """[B, S, ...] -> [B*S, ...] (gộp hai chiều đầu)."""
    return {k: v.reshape((-1,) + v.shape[2:]) for k, v in obs.items()}
