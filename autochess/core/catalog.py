"""Nạp file dữ liệu tĩnh (JSON) thành các mảng hằng.

File JSON là nguồn sự thật duy nhất cho luật và nội dung. Lõi đọc các mảng ở đây;
UI đọc tên tướng/tộc cũng từ đây. Không mảng nào thay đổi trong lúc chạy.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_CATALOG = os.path.join(DATA_DIR, "catalog_v1.json")

STAT_NAMES = ("hp", "ad", "ap", "armor", "mr")
NO_BREAKPOINT = 99   # đệm cho tộc có ít mốc hơn

# Kích thước cố định, gắn với bảng 44 hành động (không đọc từ JSON).
SHOP_SIZE = 5
BENCH_SIZE = 9
BOARD_SIZE = 9
NUM_SLOTS = BENCH_SIZE + BOARD_SIZE  # ô 0-8 hàng chờ, 9-17 sân
MAX_LEVEL = 9
NUM_TIERS = 5
MAX_STAR = 3
STAR_COPIES = np.array([1, 3, 9], dtype=np.int16)  # số bản gốc của quân 1/2/3 sao
MAX_PLAYERS = 8


class CatalogError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class Catalog:
    version: str
    raw: dict
    names: list
    trait_names: list
    cost: np.ndarray               # [N] int16, 1..5
    tier: np.ndarray               # [N] int8, 0..4 (= cost - 1)
    power: np.ndarray              # [N, 3] float32, sức mạnh theo sao (luật bản 4: suy ra từ 5 chỉ số)
    trait_matrix: np.ndarray       # [N, K] int8 (0/1)
    trait_breakpoints: np.ndarray  # [K, L] int16 (L = số mốc nhiều nhất; mốc thiếu đệm NO_BREAKPOINT)
    trait_bonus: np.ndarray        # [K, L] float32 (luật bản 4: tóm tắt một số từ hiệu ứng tộc)
    counter: np.ndarray            # [K, K] float32, phản đối xứng
    shop_odds: np.ndarray          # [9, 5] float64, hàng L-1 = tỷ lệ ở cấp L (tổng = 1)
    xp_to_next: np.ndarray         # [9] int16, xp_to_next[L-1]; cấp 9 = 0 (không lên nữa)
    stage_damage_by_round: np.ndarray  # [max_rounds + 1] int16, chỉ số = vòng
    tier_onehot: np.ndarray        # [N, 5] int16
    champs_by_tier: list           # 5 mảng chỉ số tướng
    pool_by_cost: dict             # {P: [5]} từ JSON
    # kinh tế & luật
    base_income: int
    interest_step: int
    interest_max: int
    streak_bonus: np.ndarray       # [6] int16, chỉ số = min(|chuỗi|, 5)
    roll_cost: int
    xp_cost: int
    xp_per_buy: int
    xp_per_round: int
    start_gold: int
    start_hp: int
    max_rounds: int
    max_actions_per_round: int
    # hệ số nhân phần sát thương theo số quân thắng còn sống, theo vòng (JSON "unit_damage_scale",
    # mặc định 1 ở mọi vòng)
    unit_damage_scale_by_round: np.ndarray = None
    # luật bản 4 (None với file dữ liệu cũ)
    stats: np.ndarray = None           # [N, 5] float32 chỉ số 1 sao: máu, STVL, STP, giáp, kháng phép
    star_mult: np.ndarray = None       # [3] hệ số nhân máu / sát thương theo sao
    trait_levels: np.ndarray = None    # [K] số mốc thật của từng tộc
    trait_effects: np.ndarray = None   # [K, L, 5, 2] (phần trăm, cộng thẳng) cho từng chỉ số ở từng mốc
    trait_team: np.ndarray = None      # [K] bool, hiệu ứng cho cả đội (False: chỉ tướng mang tộc)
    role: np.ndarray = None            # [N] int8, chỉ số hệ (0..R-1) của tướng, -1 nếu không có
    role_traits: np.ndarray = None     # [R] id tộc của các hệ
    role_counter: np.ndarray = None    # [R, R] khắc chế giữa các hệ (phản đối xứng)
    _pool_cache: dict = field(default_factory=dict, repr=False)

    @property
    def N(self) -> int:
        return len(self.names)

    @property
    def K(self) -> int:
        return len(self.trait_names)

    def pool_size(self, num_players: int) -> np.ndarray:
        """Kích thước kho chung ban đầu của từng tướng [N] cho P người chơi."""
        if num_players in self._pool_cache:
            return self._pool_cache[num_players]
        if not 2 <= num_players <= MAX_PLAYERS:
            raise CatalogError(f"số người chơi phải trong 2..{MAX_PLAYERS}")
        keys = sorted(self.pool_by_cost)
        if num_players in self.pool_by_cost:
            by_cost = np.array(self.pool_by_cost[num_players], dtype=np.float64)
        else:
            lo = max(k for k in keys if k < num_players)
            hi = min(k for k in keys if k > num_players)
            t = (num_players - lo) / (hi - lo)
            by_cost = (1 - t) * np.array(self.pool_by_cost[lo]) + t * np.array(self.pool_by_cost[hi])
        by_cost = np.round(by_cost).astype(np.int16)
        out = by_cost[self.tier]
        out.setflags(write=False)
        self._pool_cache[num_players] = out
        return out

    def stage_damage(self, rnd) -> np.ndarray:
        rnd = np.clip(np.asarray(rnd), 0, self.max_rounds)
        return self.stage_damage_by_round[rnd]

    def unit_damage_scale(self, rnd) -> np.ndarray:
        rnd = np.clip(np.asarray(rnd), 0, self.max_rounds)
        return self.unit_damage_scale_by_round[rnd]


def load_catalog(path: str | None = None) -> Catalog:
    path = path or DEFAULT_CATALOG
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return build_catalog(raw)


def build_catalog(raw: dict) -> Catalog:
    champs = sorted(raw["champions"], key=lambda c: c["id"])
    traits = sorted(raw["traits"], key=lambda t: t["id"])
    N, K = len(champs), len(traits)
    _check([c["id"] for c in champs] == list(range(N)), "id tướng phải là 0..N-1 liên tục")
    _check([t["id"] for t in traits] == list(range(K)), "id tộc phải là 0..K-1 liên tục")
    names = [c["name"] for c in champs]
    _check(len(set(names)) == N, "tên tướng bị trùng")

    cost = np.array([c["cost"] for c in champs], dtype=np.int16)
    _check(((cost >= 1) & (cost <= NUM_TIERS)).all(), "giá tướng phải trong 1..5")
    tier = (cost - 1).astype(np.int8)
    for t in range(NUM_TIERS):
        _check((tier == t).any(), f"bậc {t + 1} không có tướng nào")

    has_stats = all("stats" in c for c in champs)
    stats = star_mult = None
    if has_stats:
        stats = np.array([[c["stats"][k] for k in STAT_NAMES] for c in champs], dtype=np.float32)
        _check((stats[:, 0] > 0).all() and (stats >= 0).all(), "chỉ số phải không âm, máu dương")
        star_mult = np.array(raw.get("star_mult", [1.0, 1.8, 3.24]), dtype=np.float32)
        _check(star_mult.shape == (MAX_STAR,) and (np.diff(star_mult) > 0).all(), "star_mult: 3 số tăng dần")
    if "power" in champs[0]:
        power = np.array([c["power"] for c in champs], dtype=np.float32)
    else:
        _check(has_stats, "mỗi tướng cần power (luật cũ) hoặc stats (luật bản 4)")
        power = unit_power(stats, star_mult)
    _check(power.shape == (N, MAX_STAR), "power phải có 3 giá trị (1/2/3 sao)")
    _check((power > 0).all() and (np.diff(power, axis=1) > 0).all(), "power phải dương và tăng theo sao")

    trait_matrix = np.zeros((N, K), dtype=np.int8)
    for c in champs:
        _check(1 <= len(c["traits"]) <= 3 and len(set(c["traits"])) == len(c["traits"]),
               f"{c['name']}: mỗi tướng có 1-3 tộc khác nhau")
        for k in c["traits"]:
            _check(0 <= k < K, f"{c['name']}: tộc {k} không tồn tại")
            trait_matrix[c["id"], k] = 1

    L = max(len(t["breakpoints"]) for t in traits)
    levels = np.array([len(t["breakpoints"]) for t in traits], dtype=np.int16)
    _check((levels >= 1).all() and L <= 6, "mỗi tộc có 1-6 mốc")
    bp = np.full((K, L), NO_BREAKPOINT, dtype=np.int16)
    for t in traits:
        b = list(t["breakpoints"])
        _check(all(x < y for x, y in zip(b, b[1:])) and b[0] >= 1, f"{t['name']}: mốc tộc phải tăng dần, >= 1")
        bp[t["id"], :len(b)] = b
    effects = team = None
    if all("effects" in t for t in traits):
        effects = np.zeros((K, L, len(STAT_NAMES), 2), dtype=np.float32)
        team = np.array([t.get("scope", "members") == "team" for t in traits])
        for t in traits:
            _check(t.get("scope", "members") in ("members", "team"), f"{t['name']}: scope là members hoặc team")
            for stat, mode, vals in t["effects"]:
                _check(stat in STAT_NAMES and mode in ("pct", "flat"), f"{t['name']}: hiệu ứng {stat}/{mode} không hợp lệ")
                _check(len(vals) == len(t["breakpoints"]), f"{t['name']}: mỗi mốc một giá trị hiệu ứng")
                j = STAT_NAMES.index(stat)
                for lv, v in enumerate(vals):
                    if mode == "pct":
                        effects[t["id"], lv, j, 0] += v / 100.0
                    else:
                        effects[t["id"], lv, j, 1] += v
    if all("bonus" in t for t in traits):
        bonus = np.zeros((K, L), dtype=np.float32)
        for t in traits:
            _check(len(t["bonus"]) == len(t["breakpoints"]), f"{t['name']}: mỗi mốc một mức thưởng")
            bonus[t["id"], :len(t["bonus"])] = t["bonus"]
            bonus[t["id"], len(t["bonus"]):] = t["bonus"][-1]
    else:
        _check(effects is not None, "mỗi tộc cần bonus (luật cũ) hoặc effects (luật bản 4)")
        # tóm tắt một số cho quan sát / bot cũ: phần trăm cộng lại + cộng thẳng / 100; tộc cả đội x 2
        summ = effects[..., 0].sum(-1) + effects[..., 1].sum(-1) / 100.0
        bonus = (summ * np.where(team, 2.0, 1.0)[:, None] / 2.0).astype(np.float32)
        for k in range(K):
            bonus[k, levels[k]:] = bonus[k, levels[k] - 1]

    role = role_traits = role_counter = None
    if "role_counter" in raw:
        rc = raw["role_counter"]
        role_traits = np.array(rc["roles"], dtype=np.int16)
        role_counter = np.array(rc["matrix"], dtype=np.float32)
        R = len(role_traits)
        _check(role_counter.shape == (R, R), "role_counter.matrix phải là R x R")
        _check(np.allclose(role_counter, -role_counter.T, atol=1e-6), "role_counter phải phản đối xứng")
        role = np.full(N, -1, dtype=np.int8)
        for c in champs:
            mine = [r for r, k in enumerate(role_traits) if k in c["traits"]]
            _check(len(mine) <= 1, f"{c['name']}: mỗi tướng tối đa một hệ vai trò")
            if mine:
                role[c["id"]] = mine[0]
        counter = np.zeros((K, K), dtype=np.float32)
        counter[np.ix_(role_traits, role_traits)] = role_counter
    else:
        counter = np.array(raw["counter_matrix"], dtype=np.float32)
    _check(counter.shape == (K, K), "counter_matrix phải là K x K")
    _check(np.allclose(counter, -counter.T, atol=1e-6), "counter_matrix phải phản đối xứng (C[i][j] = -C[j][i])")

    odds = np.array(raw["shop_odds"], dtype=np.float64)
    _check(odds.shape == (MAX_LEVEL, NUM_TIERS), "shop_odds phải là 9 x 5")
    _check(np.allclose(odds.sum(1), 100), "mỗi hàng shop_odds phải cộng lại 100")
    odds = odds / 100.0

    xp = list(raw["xp_to_next"])
    _check(len(xp) == MAX_LEVEL - 1 and all(v > 0 for v in xp), "xp_to_next cần 8 ngưỡng dương")
    xp_to_next = np.array(xp + [0], dtype=np.int16)

    rules = raw.get("rules", {})
    eco = raw["economy"]
    max_rounds = int(rules.get("max_rounds", 50))
    stage = np.zeros(max_rounds + 1, dtype=np.int16)
    sd = sorted(raw["stage_damage"])
    for start, dmg in sd:
        stage[start:] = dmg
    _check(sd[0][0] <= 1, "stage_damage phải bắt đầu từ vòng 1")
    uscale = np.ones(max_rounds + 1, dtype=np.float32)
    for start, sc in sorted(raw.get("unit_damage_scale", [])):
        _check(0 <= sc <= 3, "unit_damage_scale: hệ số trong [0, 3]")
        uscale[int(start):] = sc

    pool_by_cost = {int(k): v for k, v in raw["pool_size_by_cost"].items()}
    _check(2 in pool_by_cost and MAX_PLAYERS in pool_by_cost, "pool_size_by_cost cần khóa '2' và '8'")
    for v in pool_by_cost.values():
        _check(len(v) == NUM_TIERS and all(x >= 3 for x in v), "pool_size_by_cost: 5 số, mỗi số >= 3")

    streak = np.array(eco["streak_bonus"], dtype=np.int16)
    _check(len(streak) == 6, "streak_bonus cần 6 giá trị (chỉ số = min(|chuỗi|, 5))")

    tier_onehot = np.zeros((N, NUM_TIERS), dtype=np.int16)
    tier_onehot[np.arange(N), tier] = 1
    champs_by_tier = [np.flatnonzero(tier == t) for t in range(NUM_TIERS)]

    for a in (cost, tier, power, trait_matrix, bp, bonus, counter, odds, xp_to_next, stage, tier_onehot, streak, uscale,
              levels, stats, star_mult, effects, team, role, role_traits, role_counter):
        if a is not None:
            a.setflags(write=False)

    return Catalog(
        version=raw["version"], raw=raw, names=names, trait_names=[t["name"] for t in traits],
        cost=cost, tier=tier, power=power, trait_matrix=trait_matrix,
        trait_breakpoints=bp, trait_bonus=bonus, counter=counter, shop_odds=odds,
        xp_to_next=xp_to_next, stage_damage_by_round=stage, tier_onehot=tier_onehot,
        champs_by_tier=champs_by_tier, pool_by_cost=pool_by_cost,
        base_income=int(eco["base_income"]), interest_step=int(eco["interest_step"]),
        interest_max=int(eco["interest_max"]), streak_bonus=streak,
        roll_cost=int(eco["roll_cost"]), xp_cost=int(eco["xp_cost"]),
        xp_per_buy=int(eco["xp_per_buy"]), xp_per_round=int(eco["xp_per_round"]),
        start_gold=int(eco["start_gold"]), start_hp=int(rules.get("start_hp", 100)),
        max_rounds=max_rounds, max_actions_per_round=int(rules.get("max_actions_per_round", 30)),
        unit_damage_scale_by_round=uscale,
        stats=stats, star_mult=star_mult, trait_levels=levels, trait_effects=effects, trait_team=team,
        role=role, role_traits=role_traits, role_counter=role_counter,
    )


POWER_SCALE = 17.0   # power = sqrt(máu hiệu dụng x sát thương hiệu dụng) / POWER_SCALE (1 tiền 1 sao ~ 8-11)


def unit_power(stats: np.ndarray, star_mult: np.ndarray) -> np.ndarray:
    """Sức mạnh một số của từng tướng theo sao [N, 3], suy ra từ 5 chỉ số (cho bot, quan sát, giao diện):
    căn của (máu hiệu dụng trước đòn hỗn hợp) x (sát thương lên đối thủ có 50 giáp / 50 kháng phép)."""
    st = np.asarray(stats, np.float64)
    m = np.asarray(star_mult, np.float64)
    ehp = st[:, 0:1] * m * (1.0 + (st[:, 3:4] + st[:, 4:5]) / 200.0)
    dps = (st[:, 1:2] + st[:, 2:3]) * m * (100.0 / 150.0)
    return (np.sqrt(ehp * dps) / POWER_SCALE).astype(np.float32)


def _check(cond, msg):
    if not cond:
        raise CatalogError(msg)
