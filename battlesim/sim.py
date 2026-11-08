"""Mô phỏng giao tranh theo lô, viết theo kiểu tương thích Numba.

Nấc luật (level):
  0 — không có không gian: mọi tướng luôn trong tầm, không di chuyển.
  1 — lưới lục giác: chọn kẻ địch gần nhất, tìm đường (BFS) vòng qua đồng đội, tranh ô.
  2 — (M4) thanh năng lượng, kỹ năng, tộc/hệ.  Chưa cài.
stochastic=False: chí mạng tính theo giá trị kỳ vọng (trận tất định).
stochastic=True : chí mạng tung ngẫu nhiên theo seed (nấc 3).

Một bước thời gian:
  1. giảm bộ đếm   2. chọn mục tiêu   3. đánh hoặc xin đi một ô
  4. giải quyết tranh ô   5. tung chiêu (nấc 2)   6. áp sát thương cùng lúc, xử lý chết
  7. kiểm tra kết thúc

Các quy tắc giữ đối xứng giữa hai phe (để không phe nào được lợi giả):
  - sát thương ghi vào mảng đệm, áp cùng lúc ở bước 6;
  - hòa khoảng cách khi chọn mục tiêu: phá bằng thứ tự trong đội (I_ORD), không bằng chỉ số mảng;
  - tìm đường: phe 1 thử hướng theo thứ tự của phe 0 xoay 180° (k -> k+3);
  - tranh ô: tướng có đường tới chỗ đánh ngắn hơn được đi; hòa thì thứ tự trong đội nhỏ hơn
    được đi; vẫn hòa (khác phe) thì so vị trí nhìn từ phía đội mình (phe 1 xoay 180°), ô nhỏ hơn
    được đi. Chỉ hai tướng đứng đối xứng hoàn toàn mới còn hòa; khi đó không ai đi và cả hai
    tránh ô đó ở bước sau.
"""
from dataclasses import dataclass

import numpy as np

from . import hexgrid
from .jit import njit, prange
from .layout import (
    U, U_PER_SIDE, N_CELLS, N_HALF, NF, NI,
    S_HP, S_AD, S_AS, S_ARMOR, S_MR, S_CRIT_CH, S_CRIT_MULT, S_MANA_MAX, S_MANA_START,
    F_HP, F_HP_MAX, F_AD, F_CRIT_CH, F_CRIT_MULT, F_AS, F_ARMOR, F_MR, F_MANA, F_MANA_MAX,
    F_ATK_CD, F_MOVE_CD, F_STUN, F_SHIELD, F_PEND, F_DEALT, F_TAKEN,
    I_CHAMP, I_STAR, I_SIDE, I_ORD, I_CELL, I_TARGET, I_RANGE, I_ALIVE, I_BLOCKED,
    I_DEATH, I_CASTS, I_AVOID, I_GOAL, I_PREV,
    EV_MOVE, EV_ATTACK, EV_DEATH, RES_SIDE0, RES_SIDE1, RES_BOTH_LOSE,
)

EPS = 1e-4
NEVER = 1e9           # thời gian hồi đòn của tướng có tốc đánh 0 (hình nộm)


# --------------------------------------------------------------------------- #
# Kernel (tương thích Numba)
# --------------------------------------------------------------------------- #

@njit
def _rand(rng, b):
    """xorshift64: số thực trong [0, 1)."""
    x = rng[b]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    rng[b] = x
    return float(x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit
def _log(ev, evc, b, s, typ, src, dst, val, cell):
    k = evc[b]
    if k < ev.shape[1]:
        ev[b, k, 0] = s
        ev[b, k, 1] = typ
        ev[b, k, 2] = src
        ev[b, k, 3] = dst
        ev[b, k, 4] = val
        ev[b, k, 5] = cell
    evc[b] = k + 1     # vẫn đếm khi tràn để phát hiện thiếu chỗ


@njit
def _bfs(b, start, avoid, avoid2, sd, occ, nb, bdist, bpar, bq):
    """Loang theo chiều rộng từ ô start qua các ô trống (chiếm chỗ tính ở đầu bước).
    Thứ tự thử hướng của phe 1 là thứ tự của phe 0 xoay 180°, nên hai phe tìm đường đối xứng.
    Trả về số ô đã thăm; bq[b, :n] là các ô theo thứ tự thăm (khoảng cách tăng dần)."""
    for c in range(bdist.shape[1]):
        bdist[b, c] = -1
    bdist[b, start] = 0
    bpar[b, start] = -1
    bq[b, 0] = start
    head = 0
    tail = 1
    while head < tail:
        c = bq[b, head]
        head += 1
        for kk in range(6):
            k = kk
            if sd == 1:
                k = (kk + 3) % 6
            n = nb[c, k]
            if n < 0 or bdist[b, n] >= 0 or occ[b, n] >= 0 or n == avoid or n == avoid2:
                continue
            bdist[b, n] = bdist[b, c] + 1
            bpar[b, n] = c
            bq[b, tail] = n
            tail += 1
    return tail


@njit
def _first_goal(b, n_visited, target_cell, attack_range, dist, bq):
    """Ô đầu tiên (theo thứ tự loang, tức đường ngắn nhất) đứng được để đánh mục tiêu; -1 nếu không có."""
    for i in range(1, n_visited):
        c = bq[b, i]
        if dist[c, target_cell] <= attack_range:
            return c
    return -1


@njit
def _init_one(b, champ, star, side, half_cell, seed, F, I, occ, rng, evc,
              stats, attack_range, star_mult, rot, stochastic):
    n_units = F.shape[1]
    for c in range(occ.shape[1]):
        occ[b, c] = -1
    x = np.uint64(seed[b]) ^ np.uint64(0x9E3779B97F4A7C15)
    if x == np.uint64(0):
        x = np.uint64(1)
    rng[b] = x
    evc[b] = 0

    n0 = 0
    n1 = 0
    for u in range(n_units):
        for j in range(F.shape[2]):
            F[b, u, j] = 0.0
        for j in range(I.shape[2]):
            I[b, u, j] = 0
        I[b, u, I_CHAMP] = -1
        I[b, u, I_CELL] = -1
        I[b, u, I_TARGET] = -1
        I[b, u, I_DEATH] = -1
        I[b, u, I_AVOID] = -1
        I[b, u, I_GOAL] = -1
        I[b, u, I_PREV] = -1
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

        st = star[b, u] - 1
        hp = stats[c, S_HP] * star_mult[st, 0]
        ad = stats[c, S_AD] * star_mult[st, 1]
        cc = stats[c, S_CRIT_CH]
        cm = stats[c, S_CRIT_MULT]
        if not stochastic:
            ad = ad * (1.0 + cc * (cm - 1.0))
        atk_speed = stats[c, S_AS]

        F[b, u, F_HP] = hp
        F[b, u, F_HP_MAX] = hp
        F[b, u, F_AD] = ad
        F[b, u, F_CRIT_CH] = cc
        F[b, u, F_CRIT_MULT] = cm
        F[b, u, F_AS] = atk_speed
        F[b, u, F_ARMOR] = stats[c, S_ARMOR]
        F[b, u, F_MR] = stats[c, S_MR]
        F[b, u, F_MANA] = stats[c, S_MANA_START]
        F[b, u, F_MANA_MAX] = stats[c, S_MANA_MAX]
        if atk_speed > 0:
            F[b, u, F_ATK_CD] = 1.0 / atk_speed   # đòn đầu tiên sau một chu kỳ
        else:
            F[b, u, F_ATK_CD] = NEVER

        I[b, u, I_CHAMP] = c
        I[b, u, I_STAR] = star[b, u]
        I[b, u, I_SIDE] = sd
        I[b, u, I_ORD] = o
        I[b, u, I_CELL] = cell
        I[b, u, I_RANGE] = attack_range[c]
        I[b, u, I_ALIVE] = 1


@njit
def _step_one(b, s, F, I, occ, rng, ev, evc, want, key, minkey, cnt, bdist, bpar, bq,
              dist, nb, rot, level, stochastic, log, dt, move_steps):
    """Chạy bước thời gian thứ s (bắt đầu từ 1) cho trận b.
    Trả về -1 nếu trận chưa kết thúc, ngược lại là mã kết quả."""
    n_units = F.shape[1]

    # 1. giảm bộ đếm
    for u in range(n_units):
        if I[b, u, I_ALIVE] == 0:
            continue
        F[b, u, F_ATK_CD] -= dt
        if F[b, u, F_MOVE_CD] > 0:
            F[b, u, F_MOVE_CD] -= 1.0
        if F[b, u, F_STUN] > 0:
            F[b, u, F_STUN] -= dt

    # 2. chọn mục tiêu (giữ mục tiêu cũ tới khi nó chết)
    for u in range(n_units):
        if I[b, u, I_ALIVE] == 0:
            continue
        t = I[b, u, I_TARGET]
        if t >= 0 and I[b, t, I_ALIVE] == 1:
            continue
        cu = I[b, u, I_CELL]
        sd = I[b, u, I_SIDE]
        best = -1
        bd = 1000
        bo = 1000
        for v in range(n_units):
            if I[b, v, I_ALIVE] == 0 or I[b, v, I_SIDE] == sd:
                continue
            d = 0
            if level >= 1:
                d = dist[cu, I[b, v, I_CELL]]
            o = I[b, v, I_ORD]
            if d < bd or (d == bd and o < bo):
                best = v
                bd = d
                bo = o
        I[b, u, I_TARGET] = best

    # 3. đánh hoặc xin đi một ô
    for c in range(minkey.shape[1]):
        minkey[b, c] = 1 << 30
        cnt[b, c] = 0
    for u in range(n_units):
        want[b, u] = -1
        if I[b, u, I_ALIVE] == 0:
            continue
        if F[b, u, F_MOVE_CD] > 0 or F[b, u, F_STUN] > 0:
            continue
        t = I[b, u, I_TARGET]
        if t < 0:
            continue
        cu = I[b, u, I_CELL]
        ct = I[b, t, I_CELL]
        d = 0
        if level >= 1:
            d = dist[cu, ct]
        if d <= I[b, u, I_RANGE]:
            I[b, u, I_BLOCKED] = 0
            if F[b, u, F_ATK_CD] <= EPS:
                dmg = F[b, u, F_AD]
                if stochastic:
                    if _rand(rng, b) < F[b, u, F_CRIT_CH]:
                        dmg = dmg * F[b, u, F_CRIT_MULT]
                post = dmg * 100.0 / (100.0 + F[b, t, F_ARMOR])
                F[b, t, F_PEND] += post
                F[b, u, F_DEALT] += post
                cd = F[b, u, F_ATK_CD]
                if cd < 0:
                    cd = 0.0
                F[b, u, F_ATK_CD] = cd + 1.0 / F[b, u, F_AS]
                if log:
                    _log(ev, evc, b, s, EV_ATTACK, u, t, post, ct)
        elif level >= 1:
            # tìm đường ngắn nhất (qua ô trống) tới một ô đứng được để đánh mục tiêu
            sd = I[b, u, I_SIDE]
            rng_u = I[b, u, I_RANGE]
            avoid = I[b, u, I_AVOID]
            I[b, u, I_AVOID] = -1
            # không quay về ô vừa rời, trừ khi đã chờ đủ một nhịp đi
            back = I[b, u, I_PREV]
            if I[b, u, I_BLOCKED] >= move_steps:
                back = -1
            nv = _bfs(b, cu, avoid, back, sd, occ, nb, bdist, bpar, bq)
            # giữ chỗ đứng đánh đã cam kết nếu còn dùng được: tới được và vẫn trong tầm mục tiêu
            goal = I[b, u, I_GOAL]
            if goal >= 0 and (bdist[b, goal] <= 0 or dist[goal, ct] > rng_u):
                goal = -1
            if goal < 0:
                goal = _first_goal(b, nv, ct, rng_u, dist, bq)
            if goal < 0:
                # mục tiêu bị vây kín: đổi sang kẻ địch tới được bằng đường ngắn nhất
                bl = 1000
                bo = 1000
                for v in range(n_units):
                    if I[b, v, I_ALIVE] == 0 or I[b, v, I_SIDE] == sd:
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
                        t = v
                if goal >= 0:
                    I[b, u, I_TARGET] = t
            I[b, u, I_GOAL] = goal
            if goal < 0:
                I[b, u, I_BLOCKED] += 1
            else:
                plen = bdist[b, goal]
                step = goal
                while bpar[b, step] != cu:
                    step = bpar[b, step]
                want[b, u] = step
                # khóa tranh ô: đường ngắn hơn, rồi thứ tự trong đội, rồi vị trí nhìn từ phía đội mình
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

    # 4. giải quyết tranh ô
    if level >= 1:
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
                    I[b, u, I_AVOID] = n             # hòa tuyệt đối: bước sau tránh ô này

    # 5. tung chiêu — nấc 2 (M4)

    # 6. áp sát thương cùng lúc
    for u in range(n_units):
        p = F[b, u, F_PEND]
        if p <= 0:
            continue
        F[b, u, F_PEND] = 0.0
        F[b, u, F_TAKEN] += p
        sh = F[b, u, F_SHIELD]
        if sh > 0:
            a = min(sh, p)
            F[b, u, F_SHIELD] = sh - a
            p = p - a
        F[b, u, F_HP] -= p
    for u in range(n_units):
        if I[b, u, I_ALIVE] == 1 and F[b, u, F_HP] <= 0:
            I[b, u, I_ALIVE] = 0
            F[b, u, F_HP] = 0.0
            occ[b, I[b, u, I_CELL]] = -1
            I[b, u, I_DEATH] = s
            if log:
                _log(ev, evc, b, s, EV_DEATH, u, -1, 0.0, I[b, u, I_CELL])

    # 7. kiểm tra kết thúc
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


@njit(parallel=True)
def _run_all(champ, star, side, half_cell, seed, F, I, occ, rng, ev, evc,
             want, key, minkey, cnt, bdist, bpar, bq, result, end_step,
             stats, attack_range, star_mult, rot, dist, nb,
             level, stochastic, log, dt, move_steps, max_steps):
    n_battles = champ.shape[0]
    for b in prange(n_battles):
        _init_one(b, champ, star, side, half_cell, seed, F, I, occ, rng, evc,
                  stats, attack_range, star_mult, rot, stochastic)
        r = -1
        last = max_steps
        for s in range(1, max_steps + 1):
            r = _step_one(b, s, F, I, occ, rng, ev, evc, want, key, minkey, cnt, bdist, bpar, bq,
                          dist, nb, rot, level, stochastic, log, dt, move_steps)
            if r >= 0:
                last = s
                break
        if r < 0:
            r = RES_BOTH_LOSE      # hết giờ: cả hai cùng thua
        result[b] = r
        end_step[b] = last


# --------------------------------------------------------------------------- #
# Lớp Python bên ngoài
# --------------------------------------------------------------------------- #

@dataclass
class Batch:
    """Đầu vào theo hợp đồng dữ liệu: mỗi mảng có dạng (B, U)."""
    champ: np.ndarray      # int8, -1 = trống
    star: np.ndarray       # int8, 1–3
    side: np.ndarray       # int8, 0/1
    half_cell: np.ndarray  # int8, 0–27 trên nửa sân của đội đó
    seed: np.ndarray       # (B,) int64

    @property
    def size(self):
        return self.champ.shape[0]


@dataclass
class State:
    F: np.ndarray
    I: np.ndarray
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


@dataclass
class BattleResult:
    result: np.ndarray        # (B,) int8: 0/1 = phe thắng, 2 = cả hai cùng thua
    end_step: np.ndarray      # (B,) int16
    alive: np.ndarray         # (B, U) bool
    star: np.ndarray          # (B, U) int8 — cùng với alive để tính sát thương lên người chơi
    dmg_dealt: np.ndarray     # (B, U) float32
    dmg_taken: np.ndarray     # (B, U) float32
    death_step: np.ndarray    # (B, U) int16, -1 = còn sống
    casts: np.ndarray         # (B, U) int16
    events: list = None       # list (B) mảng (n, 6) nếu bật nhật ký
    events_overflow: np.ndarray = None


def make_batch(teams_a, teams_b, seeds=None):
    """teams_a[b], teams_b[b]: danh sách (mã tướng, sao, ô nửa sân) của phe 0 và phe 1."""
    n = len(teams_a)
    champ = np.full((n, U), -1, dtype=np.int8)
    star = np.ones((n, U), dtype=np.int8)
    side = np.zeros((n, U), dtype=np.int8)
    half = np.zeros((n, U), dtype=np.int8)
    side[:, U_PER_SIDE:] = 1
    for b in range(n):
        for offset, team in ((0, teams_a[b]), (U_PER_SIDE, teams_b[b])):
            if len(team) > U_PER_SIDE:
                raise ValueError(f"Trận {b}: một đội quá {U_PER_SIDE} tướng")
            for k, (c, st, h) in enumerate(team):
                champ[b, offset + k] = c
                star[b, offset + k] = st
                half[b, offset + k] = h
    if seeds is None:
        seeds = np.arange(n, dtype=np.int64)
    return Batch(champ, star, side, half, np.asarray(seeds, dtype=np.int64))


def swap_sides(batch):
    """Đổi phe hai đội (giữ nguyên ô nửa sân và thứ tự trong đội)."""
    return Batch(batch.champ.copy(), batch.star.copy(), (1 - batch.side).astype(np.int8),
                 batch.half_cell.copy(), batch.seed.copy())


class Simulator:
    def __init__(self, cfg, level=1, stochastic=False, log=False, max_events=4096):
        if level not in (0, 1):
            raise NotImplementedError("Nấc 2 trở lên sẽ có ở M4")
        self.cfg = cfg
        self.level = int(level)
        self.stochastic = bool(stochastic)
        self.log = bool(log)
        self.max_events = int(max_events) if log else 1

    # ---- kiểm tra đầu vào ----
    def validate(self, batch):
        champ, star, side, half = batch.champ, batch.star, batch.side, batch.half_cell
        if champ.shape[1] != U:
            raise ValueError(f"Cần {U} ô tướng mỗi trận")
        used = champ >= 0
        if np.any(champ[used] >= self.cfg.n_champs):
            raise ValueError("Mã tướng vượt quá số tướng trong cấu hình")
        if np.any((star[used] < 1) | (star[used] > 3)):
            raise ValueError("Số sao phải trong 1–3")
        if np.any((half[used] < 0) | (half[used] >= N_HALF)):
            raise ValueError("Ô nửa sân phải trong 0–27")
        if np.any((side[used] != 0) & (side[used] != 1)):
            raise ValueError("Phe phải là 0 hoặc 1")
        board = np.where(side == 0, half, hexgrid.ROT[np.clip(half, 0, N_HALF - 1)])
        for b in range(champ.shape[0]):
            cells = board[b][used[b]]
            if len(np.unique(cells)) != len(cells):
                raise ValueError(f"Trận {b}: hai tướng cùng một ô")

    # ---- cấp phát ----
    def _alloc(self, n):
        return State(
            F=np.zeros((n, U, NF), dtype=np.float32),
            I=np.zeros((n, U, NI), dtype=np.int16),
            occ=np.full((n, N_CELLS), -1, dtype=np.int8),
            rng=np.zeros(n, dtype=np.uint64),
            ev=np.zeros((n, self.max_events, 6), dtype=np.float32),
            evc=np.zeros(n, dtype=np.int32),
            want=np.full((n, U), -1, dtype=np.int16),
            key=np.zeros((n, U), dtype=np.int32),
            minkey=np.zeros((n, N_CELLS), dtype=np.int32),
            cnt=np.zeros((n, N_CELLS), dtype=np.int16),
            bdist=np.zeros((n, N_CELLS), dtype=np.int16),
            bpar=np.zeros((n, N_CELLS), dtype=np.int16),
            bq=np.zeros((n, N_CELLS), dtype=np.int16),
        )

    # ---- chạy trọn trận ----
    def run(self, batch):
        self.validate(batch)
        n = batch.size
        st = self._alloc(n)
        result = np.zeros(n, dtype=np.int8)
        end_step = np.zeros(n, dtype=np.int16)
        cfg = self.cfg
        _run_all(batch.champ, batch.star, batch.side, batch.half_cell, batch.seed,
                 st.F, st.I, st.occ, st.rng, st.ev, st.evc,
                 st.want, st.key, st.minkey, st.cnt, st.bdist, st.bpar, st.bq, result, end_step,
                 cfg.stats, cfg.attack_range, cfg.star_mult, hexgrid.ROT,
                 hexgrid.DIST, hexgrid.NEIGHBORS,
                 self.level, self.stochastic, self.log, cfg.dt, cfg.move_steps, cfg.max_steps)
        return self.collect(st, result, end_step)

    # ---- chạy từng bước (cho kiểm thử và trình xem lại) ----
    def init_state(self, batch):
        self.validate(batch)
        st = self._alloc(batch.size)
        for b in range(batch.size):
            _init_one(b, batch.champ, batch.star, batch.side, batch.half_cell, batch.seed,
                      st.F, st.I, st.occ, st.rng, st.evc,
                      self.cfg.stats, self.cfg.attack_range, self.cfg.star_mult,
                      hexgrid.ROT, self.stochastic)
        return st

    def step(self, st, s, active=None):
        """Chạy bước s cho các trận còn đang đánh. Trả về mã kết quả (-1 = chưa xong)."""
        n = st.F.shape[0]
        out = np.full(n, -1, dtype=np.int8)
        for b in range(n):
            if active is not None and not active[b]:
                continue
            out[b] = _step_one(b, s, st.F, st.I, st.occ, st.rng, st.ev, st.evc,
                               st.want, st.key, st.minkey, st.cnt, st.bdist, st.bpar, st.bq,
                               hexgrid.DIST, hexgrid.NEIGHBORS, hexgrid.ROT, self.level,
                               self.stochastic, self.log, self.cfg.dt, self.cfg.move_steps)
        return out

    def collect(self, st, result, end_step):
        I, F = st.I, st.F
        events, overflow = None, None
        if self.log:
            events = [st.ev[b, :min(st.evc[b], self.max_events)].copy() for b in range(len(result))]
            overflow = st.evc > self.max_events
        return BattleResult(
            result=result, end_step=end_step,
            alive=I[:, :, I_ALIVE] == 1,
            star=I[:, :, I_STAR].astype(np.int8),
            dmg_dealt=F[:, :, F_DEALT].copy(),
            dmg_taken=F[:, :, F_TAKEN].copy(),
            death_step=I[:, :, I_DEATH].copy(),
            casts=I[:, :, I_CASTS].copy(),
            events=events, events_overflow=overflow,
        )
