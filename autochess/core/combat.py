"""Module giao tranh: hợp đồng duy nhất mà lõi biết.

evaluate(catalog, state) -> (win_prob[B, P, P], damage[B, P, P])
  win_prob[b, i, j] = xác suất i thắng j;  win_prob[b, i, j] + win_prob[b, j, i] = 1
  damage[b, i, j]   = máu i mất nếu thua j
Thay ruột (agent xếp quân, mạng dự đoán...) không phải sửa lõi.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .catalog import Catalog, BENCH_SIZE, POWER_SCALE


class CombatModel(Protocol):
    """`state` là State hoặc lát cắt chỉ-đọc của các ván đang giao tranh, có ít nhất các mảng:
    units [n, P, 18], round [n], alive [n, P], hp [n, P], level [n, P]."""
    version: str

    def evaluate(self, catalog: Catalog, state) -> tuple[np.ndarray, np.ndarray]:
        ...


def board_features(catalog: Catalog, state) -> dict:
    """Đặc trưng sân đấu dùng chung cho combat, UI và quan sát. Luật bản 4 (catalog.stats) thêm chỉ số đội
    (team_stats) và strength = độ mạnh suy ra từ chỉ số."""
    board = state.units[..., BENCH_SIZE:]                     # [B,P,9]
    f = _board_features(catalog, board)
    if catalog.stats is not None:
        ts = team_stats(catalog, board, f, keep_units=True)
        f.update(ts)
        f["strength"] = np.where(f["n_board"] > 0, team_rating(ts), 1.0)
    return f


def _board_features(catalog: Catalog, board: np.ndarray) -> dict:
    occ = board >= 0
    champ = np.where(occ, board // 3, 0)
    star = np.where(occ, board % 3, 0)
    power = np.where(occ, catalog.power[champ, star], 0.0).sum(-1)   # [B,P]
    has = ((champ[..., None] == np.arange(catalog.N)) & occ[..., None]).any(-2)  # [B,P,N]
    trait_count = has.astype(np.int16) @ catalog.trait_matrix.astype(np.int16)    # [B,P,K]
    trait_level = (trait_count[..., None] >= catalog.trait_breakpoints).sum(-1)    # [B,P,K] 0..L
    bonus_tab = np.concatenate([np.zeros((catalog.K, 1), np.float32), catalog.trait_bonus], 1)  # [K,L+1]
    bonus = bonus_tab[np.arange(catalog.K), trait_level].sum(-1)                  # [B,P]
    n_board = occ.sum(-1)
    strength = np.where(n_board > 0, power * (1.0 + bonus), 1.0)
    return dict(power=power, bonus=bonus, strength=strength, trait_count=trait_count,
                trait_level=trait_level, n_board=n_board, has=has)


def unit_final_stats(catalog: Catalog, board: np.ndarray, f: dict | None = None) -> np.ndarray:
    """Chỉ số cuối của từng quân trên sân [..., 9, 5] (máu, STVL, STP, giáp, kháng phép) sau sao và hiệu ứng tộc.
    Ô trống = 0. Chỉ dùng với luật bản 4 (catalog.stats)."""
    f = f if f is not None else _board_features(catalog, board)
    occ = board >= 0
    champ = np.where(occ, board // 3, 0)
    star = np.where(occ, board % 3, 0)
    base = catalog.stats[champ].astype(np.float64)                                  # [...,9,5]
    mult = catalog.star_mult[star].astype(np.float64)                               # [...,9]
    base[..., :3] *= mult[..., None]
    lvl = f["trait_level"].astype(np.int64)                                        # [...,K]
    eff_tab = np.concatenate([np.zeros((catalog.K, 1) + catalog.trait_effects.shape[2:], np.float32),
                              catalog.trait_effects], 1)                           # [K,L+1,5,2]
    E = eff_tab[np.arange(catalog.K), lvl].astype(np.float64)                      # [...,K,5,2]
    team = catalog.trait_team.astype(bool)
    # hiệu ứng riêng: mỗi tướng có <= 3 tộc -> lấy thẳng 3 hàng (hàng K = 0) thay vì nhân với ma trận [9, K]
    Em = np.concatenate([np.where(team[:, None, None], 0.0, E), np.zeros(E.shape[:-3] + (1,) + E.shape[-2:])], -3)
    idx = _champ_trait_idx(catalog)[champ]                                          # [...,9,3]
    flat = Em.reshape(Em.shape[:-3] + (-1, 10))                                     # [...,K+1,10]
    g = np.take_along_axis(flat[..., None, :, :], idx[..., None], -2)               # [...,9,3,10]
    eff = (g[..., 0, :] + g[..., 1, :] + g[..., 2, :]).reshape(g.shape[:-2] + (5, 2))
    eff = eff + E[..., team, :, :].sum(-3)[..., None, :, :]
    out = base * (1.0 + eff[..., 0]) + eff[..., 1]
    return np.where(occ[..., None], out, 0.0)


def _champ_trait_idx(catalog: Catalog) -> np.ndarray:
    """[N, 3] chỉ số các tộc của từng tướng, đệm bằng K (hàng hiệu ứng 0)."""
    idx = getattr(catalog, "_trait_idx3", None)
    if idx is None:
        idx = np.full((catalog.N, 3), catalog.K, np.int64)
        for c in range(catalog.N):
            ks = np.flatnonzero(catalog.trait_matrix[c])
            idx[c, :len(ks)] = ks
        object.__setattr__(catalog, "_trait_idx3", idx)
    return idx


def team_stats(catalog: Catalog, board: np.ndarray, f: dict | None = None, keep_units: bool = False) -> dict:
    """Chỉ số cả đội: tổng máu / STVL / STP, giáp và kháng phép trung bình theo trọng số máu, tỷ lệ các hệ.
    keep_units: trả thêm unit_stats [..., 9, 5] (chỉ số cuối từng quân trên sân, quan sát bản 4 dùng)."""
    f = f if f is not None else _board_features(catalog, board)
    u = unit_final_stats(catalog, board, f)
    hp = u[..., 0].sum(-1)
    w = np.maximum(hp, 1e-9)
    out = dict(team_hp=hp, team_ad=u[..., 1].sum(-1), team_ap=u[..., 2].sum(-1),
               team_armor=(u[..., 0] * u[..., 3]).sum(-1) / w, team_mr=(u[..., 0] * u[..., 4]).sum(-1) / w)
    if keep_units:
        out["unit_stats"] = u
    if catalog.role is not None:
        occ = board >= 0
        r = np.where(occ, catalog.role[np.where(occ, board // 3, 0)], -1)
        R = len(catalog.role_traits)
        cnt = (r[..., None] == np.arange(R)).sum(-2).astype(np.float64)              # [...,R]
        out["role_share"] = cnt / np.maximum(occ.sum(-1, keepdims=True), 1)
    return out


def team_rating(ts: dict) -> np.ndarray:
    """Độ mạnh một số của đội (cùng thang với catalog.power): căn của máu hiệu dụng x sát thương lên đối thủ
    có 50 giáp / 50 kháng phép."""
    ehp = ts["team_hp"] * (1.0 + (ts["team_armor"] + ts["team_mr"]) / 200.0)
    dps = (ts["team_ad"] + ts["team_ap"]) * (100.0 / 150.0)
    return np.sqrt(ehp * dps) / POWER_SCALE


def board_rating(catalog: Catalog, board: np.ndarray) -> np.ndarray:
    """Độ mạnh của các sân bất kỳ [..., 9] (bot dùng để chấm quân định mua)."""
    return team_rating(team_stats(catalog, board))


def time_to_kill(ts: dict, armor_scale: float = 100.0) -> np.ndarray:
    """TTK[..., i, j] = thời gian đội i hạ đội j = máu j / (STVL i giảm bởi giáp j + STP i giảm bởi kháng phép j)."""
    hp = ts["team_hp"] + 1.0
    ad, ap = ts["team_ad"], ts["team_ap"]
    red_a = armor_scale / (armor_scale + ts["team_armor"])
    red_m = armor_scale / (armor_scale + ts["team_mr"])
    dps = ad[..., :, None] * red_a[..., None, :] + ap[..., :, None] * red_m[..., None, :] + 1e-3
    return hp[..., None, :] / dps


class FormulaCombat:
    """Giao tranh bằng công thức (chưa mô phỏng). Chọn theo file dữ liệu (khóa "combat"):

    formula-v1 (luật 1-3): sức mạnh x thưởng tộc, khắc chế tộc, sigmoid với nhiệt độ T.
    ttk-v1 (luật bản 4): mỗi đội có tổng máu, sát thương vật lý, sát thương phép, giáp và kháng phép trung bình
        (sau sao và hiệu ứng tộc). Thời gian i hạ j = máu j / (STVL i x 100/(100 + giáp j) + STP i x 100/(100 +
        kháng phép j)). logit(i thắng j) = (log TTK(j hạ i) - log TTK(i hạ j) + khắc chế hệ) / T, khắc chế hệ =
        tỷ lệ hệ của i x role_counter x tỷ lệ hệ của j.
    """

    version = "formula-v1"

    def __init__(self, temperature: float | None = None, catalog: Catalog | None = None):
        cfg = catalog.raw.get("combat", {}) if catalog is not None else {}
        self.mode = "ttk" if "ttk-v1" in cfg else "formula"
        if self.mode == "ttk":
            self.version = "ttk-v1"
        c = cfg.get(self.version, {})
        if temperature is None:
            temperature = c.get("temperature", 0.3)
        self.temperature = float(temperature)
        self.armor_scale = float(c.get("armor_scale", 100.0))

    def evaluate(self, catalog: Catalog, state):
        f = board_features(catalog, state)
        if self.mode == "ttk":
            ttk = time_to_kill(f, self.armor_scale)                         # [B,P,P] i hạ j
            lt = np.log(ttk)
            c = np.einsum("bir,rs,bjs->bij", f["role_share"], catalog.role_counter.astype(np.float64),
                          f["role_share"])
            z = (np.swapaxes(lt, 1, 2) - lt + c) / self.temperature
        else:
            logS = np.log(f["strength"])                                       # [B,P]
            a = f["trait_level"].astype(np.float32) / 3.0                      # [B,P,K]
            # einsum (vòng lặp C cố định) thay vì matmul: matmul float32 đi qua BLAS, thứ tự cộng có thể
            # khác giữa các máy -> lệch bit cuối -> ván phát lại khác máy có thể khác kết quả.
            c = np.einsum("bik,kl,bjl->bij", a, catalog.counter, a)            # [B,P,P] phản đối xứng
            z = (logS[:, :, None] - logS[:, None, :] + c) / self.temperature
        win = 1.0 / (1.0 + np.exp(-z))
        # damage[i, j] (i thua j) = sát thương giai đoạn + ceil(số quân của j * P(j thắng i) * hệ số vòng)
        stage = catalog.stage_damage(state.round).astype(np.float32)      # [B]
        winner_units = f["n_board"][:, None, :].astype(np.float32)        # j theo cột
        p_j_beats_i = np.swapaxes(win, 1, 2)
        unit_part = winner_units * p_j_beats_i
        if catalog.unit_damage_scale_by_round is not None:
            unit_part = unit_part * catalog.unit_damage_scale(state.round)[:, None, None]
        dmg = stage[:, None, None] + np.ceil(unit_part - 1e-6)
        return win, dmg.astype(np.int16)
