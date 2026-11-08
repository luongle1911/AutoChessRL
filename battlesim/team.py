"""Dựng trận cho trình mô phỏng nấc 2 từ file luật bản 5: chỉ số gốc theo sao, đồ, tộc hệ đang kích hoạt, lõi.

    data = CombatData.load()                     # autochess/data/catalog_v5.json
    team_a = [Unit("Zeus", 2, 3, items=["Mũ Phù Thủy"]), Unit("Athena", 1, 24)]
    batch = data.build_batch([team_a], [team_b])
    res = data.simulator().run(batch)

Tộc hệ: đếm số tướng KHÁC NHAU trên sân mang tộc (cộng ấn tộc nếu có), lấy mốc cao nhất đạt được; hiệu ứng
"members" gắn cho các thành viên, "team" gắn cho cả đội. Khối chỉ cộng/nhân chỉ số cố định cả trận được gộp
thẳng vào chỉ số (padd / pmul); các khối còn lại vào ô hiệu ứng động của từng quân.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .effects import (NST, STATS, ST_AD, ST_AS, ST_HP, compile_effects, is_static, MAX_LEVELS)
from .sim2 import Batch2, Simulator2, U, U_PER_SIDE, E_SLOTS

DEFAULT_CATALOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "autochess", "data", "catalog_v5.json")


@dataclass
class Unit:
    champ: object                 # tên hoặc mã tướng
    star: int = 1
    cell: int = 0                 # ô nửa sân 0-27 (hàng 0 = sau, hàng 3 = đầu; ô = hàng*7 + cột)
    items: list = field(default_factory=list)    # tên hoặc mã đồ (tối đa 3)
    emblems: list = field(default_factory=list)  # tên tộc thêm cho tướng này (ấn tộc hệ)


@dataclass
class TeamExtra:
    """Phần cả đội ngoài tướng: lõi viết bằng khối hiệu ứng (áp cho mọi quân), cộng số đếm tộc."""
    effects: list = field(default_factory=list)  # khối hiệu ứng (dict)
    trait_bonus: dict = field(default_factory=dict)  # {tên tộc: +n}


class CombatData:
    def __init__(self, raw: dict):
        self.raw = raw
        self.version = raw["version"]
        self.champs = sorted(raw["champions"] + raw.get("monsters", []), key=lambda c: c["id"])
        self.n_shop = len(raw["champions"])        # tướng mua được; sau đó là quái (vòng đánh quái)
        self.traits = sorted(raw["traits"], key=lambda t: t["id"])
        self.items = sorted(raw["items"], key=lambda i: i["id"])
        self.names = [c["name"] for c in self.champs]
        self.trait_names = [t["name"] for t in self.traits]
        self.item_names = [i["name"] for i in self.items]
        self.N, self.K = len(self.champs), len(self.traits)
        self.gp = {k: v for k, v in raw["combat_v5"].items() if k != "note"}
        self.star_hp = np.array(raw["star_mult"]["hp"], np.float32)
        self.star_ad = np.array(raw["star_mult"]["ad"], np.float32)
        rows = []

        def add(blocks):
            ids = []
            for blk in blocks:
                ids.append(len(rows))
                rows.append(blk)
            return ids
        self.skill_rows = [add(c["skill"]["effects"]) for c in self.champs]
        self.trait_rows = [add(t["effects"]) for t in self.traits]
        self.item_rows = [add(i["effects"]) for i in self.items]
        self.rows = rows
        self.EI, self.EF = compile_effects(rows)
        self.static = np.array([is_static(r) for r in rows])
        self.base = np.zeros((self.N, NST), np.float32)
        for c in self.champs:
            for k, v in c["stats"].items():
                self.base[c["id"], STATS.index(k)] = v
        self.champ_traits = [list(c["traits"]) for c in self.champs]
        self.bp = [list(t["breakpoints"]) for t in self.traits]
        self.recipe = {tuple(sorted(i["recipe"])): i["id"] for i in self.items if i["kind"] == "combined"}

    @classmethod
    def load(cls, path=DEFAULT_CATALOG):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # ------------------------------------------------------------ tra cứu
    def champ_id(self, c):
        return int(c) if not isinstance(c, str) else self.names.index(c)

    def item_id(self, i):
        return int(i) if not isinstance(i, str) else self.item_names.index(i)

    def trait_id(self, t):
        return int(t) if not isinstance(t, str) else self.trait_names.index(t)

    def combine(self, a, b):
        """Mã đồ ghép từ hai mảnh (None nếu không ghép được)."""
        return self.recipe.get(tuple(sorted((self.item_id(a), self.item_id(b)))))

    def simulator(self, stochastic=False, log=False, **kw):
        return Simulator2(self.EI, self.EF, gp=self.gp, stochastic=stochastic, log=log, **kw)

    # ------------------------------------------------------------ tộc hệ
    def trait_levels(self, team: list[Unit], extra: TeamExtra | None = None) -> np.ndarray:
        """[K] bậc kích hoạt (-1 = chưa) của một đội."""
        count = np.zeros(self.K, np.int32)
        seen = set()
        for u in team:
            c = self.champ_id(u.champ)
            if c not in seen:
                seen.add(c)
                for k in self.champ_traits[c]:
                    count[k] += 1
            for t in u.emblems:
                k = self.trait_id(t)
                if k not in self.champ_traits[c]:
                    count[k] += 1
        if extra is not None:
            for t, n in extra.trait_bonus.items():
                count[self.trait_id(t)] += n
        lv = np.full(self.K, -1, np.int32)
        for k in range(self.K):
            for j, b in enumerate(self.bp[k]):
                if count[k] >= b:
                    lv[k] = j
        return lv

    # ------------------------------------------------------------ dựng lô
    def build_batch(self, teams_a, teams_b, seeds=None, extras_a=None, extras_b=None) -> Batch2:
        n = len(teams_a)
        champ = np.full((n, U), -1, np.int16)
        star = np.ones((n, U), np.int8)
        side = np.zeros((n, U), np.int8)
        side[:, U_PER_SIDE:] = 1
        half = np.zeros((n, U), np.int8)
        base = np.zeros((n, U, NST), np.float32)
        padd = np.zeros((n, U, NST), np.float32)
        pmul = np.zeros((n, U, NST), np.float32)
        se = np.full((n, U, E_SLOTS), -1, np.int32)
        sl = np.zeros((n, U, E_SLOTS), np.int8)
        for b in range(n):
            for off, team, extra in ((0, teams_a[b], (extras_a or [None] * n)[b]),
                                     (U_PER_SIDE, teams_b[b], (extras_b or [None] * n)[b])):
                if len(team) > U_PER_SIDE:
                    raise ValueError(f"trận {b}: một đội quá {U_PER_SIDE} tướng")
                lv = self.trait_levels(team, extra)
                team_rows = []          # (khối, bậc) áp cho cả đội
                for k in range(self.K):
                    if lv[k] >= 0 and self.traits[k]["scope"] == "team":
                        team_rows += [(r, lv[k]) for r in self.trait_rows[k]]
                extra_rows = []
                if extra is not None and extra.effects:
                    extra_rows = [(r, 0) for r in self._extra_rows(extra.effects)]
                for j, u in enumerate(team):
                    i = off + j
                    c = self.champ_id(u.champ)
                    st = int(u.star)
                    champ[b, i] = c
                    star[b, i] = st
                    half[b, i] = int(u.cell)
                    base[b, i] = self.base[c]
                    base[b, i, ST_HP] *= self.star_hp[st - 1]
                    base[b, i, ST_AD] *= self.star_ad[st - 1]
                    rows = [(r, st - 1) for r in self.skill_rows[c]]
                    mine = set(self.champ_traits[c]) | {self.trait_id(t) for t in u.emblems}
                    for k in sorted(mine):
                        if lv[k] >= 0 and self.traits[k]["scope"] == "members":
                            rows += [(r, lv[k]) for r in self.trait_rows[k]]
                    if len(u.items) > 3:
                        raise ValueError(f"{self.names[c]}: quá 3 món đồ")
                    for it in u.items:
                        rows += [(r, 0) for r in self.item_rows[self.item_id(it)]]
                    rows += team_rows + extra_rows
                    slot = 0
                    for r, lvl in rows:
                        lvl = min(int(lvl), MAX_LEVELS - 1)
                        if self.static[r]:
                            s = self.EI[r, 3]
                            v = self.EF[r, lvl]
                            if self.EI[r, 4] == 0:
                                padd[b, i, s] += v
                            else:
                                pmul[b, i, s] += v
                        else:
                            if slot >= E_SLOTS:
                                raise ValueError(f"{self.names[c]}: quá {E_SLOTS} khối hiệu ứng động")
                            se[b, i, slot] = r
                            sl[b, i, slot] = lvl
                            slot += 1
        if seeds is None:
            seeds = np.arange(n, dtype=np.int64)
        return Batch2(champ, star, side, half, np.asarray(seeds, np.int64), base, padd, pmul, se, sl)

    def _extra_rows(self, blocks):
        """Khối của lõi (thêm vào bảng nếu chưa có)."""
        ids = []
        for blk in blocks:
            if blk in self.rows:
                ids.append(self.rows.index(blk))
                continue
            self.rows.append(blk)
            ids.append(len(self.rows) - 1)
        if len(self.rows) != len(self.EI):
            self.EI, self.EF = compile_effects(self.rows)
            self.static = np.array([is_static(r) for r in self.rows])
        return ids

    # ------------------------------------------------------------ chữ
    def parse_team(self, text: str) -> list[Unit]:
        """'Tên*sao@ô[Đồ1+Đồ2], ...'  ví dụ 'Zeus*2@3[Mũ Phù Thủy+Ngọc Lam], Athena@24'."""
        team = []
        for part in [p.strip() for p in text.split(",") if p.strip()]:
            items = []
            if "[" in part:
                part, rest = part.split("[", 1)
                items = [x.strip() for x in rest.rstrip("]").split("+") if x.strip()]
            left, cell = part.rsplit("@", 1)
            st = 1
            if "*" in left:
                left, st = left.rsplit("*", 1)
            team.append(Unit(left.strip(), int(st), int(cell), items=items))
        return team
