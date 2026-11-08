"""Đọc file cấu hình tướng / tộc-hệ thành các bảng numpy.

Mọi con số của luật (chỉ số tướng, hệ số sao, thời gian di chuyển...) nằm trong
file cấu hình; code mô phỏng không viết cứng con số nào.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .layout import N_STATS, STAT_KEYS

SKILL_TYPES = ("single_damage", "area_damage", "heal_lowest_ally", "self_shield", "stun_target")
TRAIT_STATS = ("hp", "ad", "as", "armor", "mr")   # chỉ số tộc/hệ có thể cộng (nấc 2)
MAX_TRAITS_PER_CHAMP = 3

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "champions.json"


@dataclass
class Config:
    names: list
    stats: np.ndarray          # (C, N_STATS) float32
    attack_range: np.ndarray   # (C,) int8
    skill_type: np.ndarray     # (C,) int8, chỉ số trong SKILL_TYPES
    cost: np.ndarray           # (C,) int8
    champ_traits: np.ndarray   # (C, 3) int8, -1 = không có
    trait_names: list
    trait_bp: np.ndarray       # (T, MAXBP) int8, -1 = không dùng
    trait_bonus: np.ndarray    # (T, MAXBP, len(TRAIT_STATS)) float32
    star_mult: np.ndarray      # (3, 2) float32: [sao-1, (máu, công)]
    dt: float
    max_steps: int
    move_steps: int            # số bước để đi một ô
    raw: dict = field(repr=False, default_factory=dict)

    def champ_id(self, name):
        return self.names.index(name)

    @property
    def n_champs(self):
        return len(self.names)


def from_dict(d):
    g = d["global"]
    dt = float(g["dt"])
    traits = d.get("traits", [])
    trait_names = [t["name"] for t in traits]
    max_bp = max([len(t["breakpoints"]) for t in traits], default=1)
    trait_bp = np.full((len(traits), max_bp), -1, dtype=np.int8)
    trait_bonus = np.zeros((len(traits), max_bp, len(TRAIT_STATS)), dtype=np.float32)
    for ti, t in enumerate(traits):
        bps = t["breakpoints"]
        trait_bp[ti, :len(bps)] = bps
        for key, vals in t.get("bonus", {}).items():
            if key not in TRAIT_STATS:
                raise ValueError(f"Tộc/hệ {t['name']}: chỉ số '{key}' không hỗ trợ")
            if len(vals) != len(bps):
                raise ValueError(f"Tộc/hệ {t['name']}: số giá trị '{key}' khác số mốc")
            trait_bonus[ti, :len(vals), TRAIT_STATS.index(key)] = vals

    champs = d["champions"]
    C = len(champs)
    stats = np.zeros((C, N_STATS), dtype=np.float32)
    attack_range = np.zeros(C, dtype=np.int8)
    skill_type = np.zeros(C, dtype=np.int8)
    cost = np.zeros(C, dtype=np.int8)
    champ_traits = np.full((C, MAX_TRAITS_PER_CHAMP), -1, dtype=np.int8)
    for ci, ch in enumerate(champs):
        for si, key in enumerate(STAT_KEYS):
            stats[ci, si] = ch.get(key, 0.0)
        attack_range[ci] = ch["range"]
        skill_type[ci] = SKILL_TYPES.index(ch.get("skill", "single_damage"))
        cost[ci] = ch.get("cost", 1)
        tlist = ch.get("traits", [])
        if len(tlist) > MAX_TRAITS_PER_CHAMP:
            raise ValueError(f"{ch['name']}: quá {MAX_TRAITS_PER_CHAMP} tộc/hệ")
        for k, tn in enumerate(tlist):
            champ_traits[ci, k] = trait_names.index(tn)

    sm = g["star_mult"]
    star_mult = np.array([sm["hp"], sm["ad"]], dtype=np.float32).T.copy()  # (3, 2)

    return Config(
        names=[c["name"] for c in champs], stats=stats, attack_range=attack_range,
        skill_type=skill_type, cost=cost, champ_traits=champ_traits,
        trait_names=trait_names, trait_bp=trait_bp, trait_bonus=trait_bonus,
        star_mult=star_mult, dt=dt, max_steps=int(g["max_steps"]),
        move_steps=max(1, int(round(float(g["move_time"]) / dt))), raw=d,
    )


def load_config(path=DEFAULT_PATH):
    with open(path, encoding="utf-8") as f:
        return from_dict(json.load(f))
