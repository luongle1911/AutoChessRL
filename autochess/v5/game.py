"""Ván đấu luật bản 5 để chơi thử (một ván, viết thẳng bằng Python cho dễ đọc; bản vector hóa cho huấn luyện làm
sau). Giao tranh do battlesim/sim2.py mô phỏng.

Vòng: giai đoạn 1: 1-1 chọn tướng chung, 1-2..1-4 đánh quái. Từ giai đoạn 2: x-1..x-3 đấu người (chọn lõi đầu
vòng 2-1, 3-2, 4-2), x-4 chọn tướng chung, x-5, x-6 đấu người, x-7 đánh quái.

Pha của ván (game.phase):
    "augment"   đang chờ người chọn lõi (bot chọn ngay)
    "carousel"  chọn tướng chung, lần lượt từ người ít máu nhất
    "plan"      lượt mua bán / xếp quân / gắn đồ; người bấm "sẵn sàng" thì giao tranh rồi sang vòng sau
    "over"      hết ván
Ghế 0 là người chơi nếu human=True; các ghế khác do bot luật (autochess/v5/bots.py) đi.
"""
from __future__ import annotations

import hashlib
import numpy as np

from battlesim.team import CombatData, Unit as SimUnit, TeamExtra

ROLE_NAMES = ("Hộ vệ", "Đấu sĩ", "Sát thủ", "Xạ thủ", "Pháp sư")


class ActionError(Exception):
    pass


def _seed(*parts) -> int:
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFFFFFFFFFF


class Player:
    def __init__(self, idx, name, cfg):
        self.idx, self.name = idx, name
        self.hp = cfg["start_hp"]
        self.gold = cfg["start_gold"]
        self.level, self.xp = 1, 0
        self.streak = 0                 # >0 chuỗi thắng, <0 chuỗi thua
        self.units: list[dict] = []     # {uid, champ, star, bench|None, cell|None, items[], emblems[]}
        self.inventory: list[dict] = [] # {"kind": "item", "id"} | {"kind": "emblem", "trait"}
        self.shop: list[int] = [-1] * 5
        self.augments: list[int] = []
        self.gold_per_round = self.xp_per_round = self.interest_bonus = self.free_rolls = 0
        self.free_rolls_left = 0
        self.trait_bonus: dict = {}
        self.effects: list = []
        self.alive, self.placement = True, 0
        self.last = None                # kết quả vòng trước
        self.components_got = 0
        self.uid = 0

    def board(self):
        return [u for u in self.units if u["cell"] is not None]

    def bench(self):
        return [u for u in self.units if u["bench"] is not None]


class Game:
    def __init__(self, data: CombatData | None = None, n_players=8, seed=0, human=True, bots=None, names=None,
                 log_fn=None):
        from .bots import make_bot
        self.data = data or CombatData.load()
        d, raw = self.data, self.data.raw
        self.cfg = raw["game_v5"]
        self.eco = raw["economy"]
        self.odds = np.array(raw["shop_odds"], np.float64) / 100.0
        self.xp_next = raw["xp_to_next"]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.P = n_players
        self.cost = np.array([c["cost"] for c in d.champs[:d.n_shop]])
        pool_by_cost = raw["pool_size_by_cost"]["8" if n_players > 4 else "2"]
        self.pool = np.array([pool_by_cost[c - 1] for c in self.cost], np.int64)
        self.by_cost = [np.flatnonzero(self.cost == c) for c in range(1, 6)]
        self.components = [i["id"] for i in d.items if i["kind"] == "component"]
        self.combined = [i["id"] for i in d.items if i["kind"] == "combined"]
        self.augs = raw["augments"]
        for a in self.augs:              # đăng ký trước các khối hiệu ứng của lõi (bảng hiệu ứng biên dịch một lần)
            if a.get("effects"):
                self.data._extra_rows(a["effects"])
        self.human = human
        styles = bots or ["tempo", "bank", "tank", "tempo", "bank", "tank", "tempo", "bank"]
        self.players = [Player(i, (names[i] if names else ("Bạn" if (human and i == 0) else f"bot {styles[i % len(styles)]}")),
                               self.cfg) for i in range(n_players)]
        self.bots = {i: make_bot(styles[(i - (1 if human else 0)) % len(styles)], self, i)
                     for i in range(n_players) if not (human and i == 0)}
        if not names:
            for i in self.bots:
                self.players[i].name = f"bot {self.bots[i].style} {i}"
        tiers = self.cfg["augment_tier_patterns"]
        self.aug_tiers = tiers[int(self.rng.integers(len(tiers)))]
        self.stage, self.rnd = 1, 1
        self.phase = None
        self.log: list[str] = []
        self.log_fn = log_fn
        self.fights: list[dict] = []        # trận của vòng gần nhất
        self.history: list[dict] = []       # mọi trận có ghế 0 (để xem lại)
        self.pending_aug: dict[int, list[int]] = {}
        self.carousel = None
        self.start_round()

    # ================================================================ tiện ích
    def say(self, msg):
        line = f"[{self.label}] {msg}"
        self.log.append(line)
        self.log = self.log[-300:]
        if self.log_fn:
            self.log_fn(line)

    @property
    def label(self):
        return f"{self.stage}-{self.rnd}"

    def kind(self, stage=None, rnd=None):
        s, r = stage or self.stage, rnd or self.rnd
        if (s == 1 and r == 1) or (s >= 2 and r == 4):
            return "carousel"
        if (s == 1 and r >= 2) or r == 7:
            return "pve"
        return "pvp"

    def alive(self):
        return [p for p in self.players if p.alive]

    def cname(self, c):
        return self.data.names[c]

    def iname(self, entry):
        if entry["kind"] == "emblem":
            return f"Ấn {self.data.trait_names[entry['trait']]}"
        return self.data.items[entry["id"]]["name"]

    # ================================================================ vòng
    def start_round(self):
        k = self.kind()
        if k == "carousel":
            self.start_carousel()
            return
        for p in self.alive():
            self.income(p)
            p.xp += 2 + p.xp_per_round
            self.level_up(p)
            p.free_rolls_left = p.free_rolls
            self.refresh_shop(p)
        if self.label in self.cfg["augment_rounds"]:
            self.offer_augments()
            if self.pending_aug:
                self.phase = "augment"
                return
        self.begin_plan()

    def begin_plan(self):
        self.phase = "plan"
        for i, bot in self.bots.items():
            if self.players[i].alive:
                bot.plan()

    def income(self, p):
        if self.stage == 1:
            g = self.cfg["stage1_income"]
        else:
            interest = min(p.gold // self.eco["interest_step"], self.eco["interest_max"] + p.interest_bonus)
            sb = self.eco["streak_bonus"]
            g = self.eco["base_income"] + interest + sb[min(abs(p.streak), len(sb) - 1)]
        p.gold += g + p.gold_per_round

    def level_up(self, p):
        while p.level < 9 and p.xp >= self.xp_next[p.level - 1]:
            p.xp -= self.xp_next[p.level - 1]
            p.level += 1
        if p.level >= 9:
            p.xp = 0

    # ================================================================ shop, kho chung
    def draw_champ(self, level):
        odds = self.odds[level - 1].copy()
        for _ in range(10):
            tier = int(self.rng.choice(5, p=odds / odds.sum()))
            ids = self.by_cost[tier]
            w = self.pool[ids].astype(np.float64)
            if w.sum() > 0:
                return int(self.rng.choice(ids, p=w / w.sum()))
            odds[tier] = 0
            if odds.sum() <= 0:
                break
        return -1

    def refresh_shop(self, p):
        p.shop = [self.draw_champ(p.level) for _ in range(5)]

    # ================================================================ hành động (người và bot dùng chung)
    def act(self, pi, a: dict):
        p = self.players[pi]
        t = a.get("type")
        if self.phase == "augment":
            if t != "augment":
                raise ActionError("đang chọn lõi")
            return self.choose_augment(pi, int(a["index"]))
        if self.phase == "carousel":
            if t != "carousel":
                raise ActionError("đang chọn tướng chung")
            return self.carousel_pick(pi, int(a["index"]))
        if self.phase != "plan" or not p.alive:
            raise ActionError("không phải lượt mua bán")
        if t == "buy":
            return self.buy(p, int(a["slot"]))
        if t == "sell":
            return self.sell(p, int(a["uid"]))
        if t == "roll":
            return self.roll(p)
        if t == "xp":
            return self.buy_xp(p)
        if t == "move":
            return self.move(p, int(a["uid"]), a["to"])
        if t == "equip":
            return self.equip(p, int(a["inv"]), int(a["uid"]))
        if t == "unequip":
            return self.unequip(p, int(a["uid"]), int(a["slot"]))
        if t == "ready":
            if pi != 0 or not self.human:
                return None
            return self.resolve_round()
        raise ActionError(f"không hiểu hành động {t}")

    def unit(self, p, uid):
        for u in p.units:
            if u["uid"] == uid:
                return u
        raise ActionError("không có quân này")

    def free_bench(self, p):
        used = {u["bench"] for u in p.units if u["bench"] is not None}
        for i in range(self.cfg["bench_slots"]):
            if i not in used:
                return i
        return None

    def add_unit(self, p, c, star=1, items=None):
        """Thêm quân vào hàng chờ (rồi gộp sao nếu đủ 3). Trả về False nếu hàng chờ đầy và không gộp được."""
        same = [u for u in p.units if u["champ"] == c and u["star"] == star]
        slot = self.free_bench(p)
        if slot is None and len(same) < 2:
            return False
        p.uid += 1
        u = dict(uid=p.uid, champ=c, star=star, bench=slot, cell=None, items=list(items or []), emblems=[])
        if slot is None:          # gộp ngay, không cần chỗ
            u["bench"] = -1
        p.units.append(u)
        self.merge(p)
        return True

    def merge(self, p):
        changed = True
        while changed:
            changed = False
            for u in list(p.units):
                if u["star"] >= 3:
                    continue
                same = [v for v in p.units if v["champ"] == u["champ"] and v["star"] == u["star"]]
                if len(same) < 3:
                    continue
                three = sorted(same, key=lambda v: (v["cell"] is None, v["uid"]))[:3]
                keep = three[0]
                items, emblems = [], []
                for v in three:
                    items += v["items"]
                    emblems += v["emblems"]
                    if v is not keep:
                        p.units.remove(v)
                keep["star"] += 1
                if keep["cell"] is None and (keep["bench"] is None or keep["bench"] < 0):
                    keep["bench"] = self.free_bench(p)
                keep["items"], keep["emblems"] = [], []
                for it in items:
                    if not self._put_item(keep, it):
                        p.inventory.append(dict(kind="item", id=it))
                for k in emblems:
                    if len(keep["items"]) + len(keep["emblems"]) < 3 and k not in keep["emblems"]:
                        keep["emblems"].append(k)
                    else:
                        p.inventory.append(dict(kind="emblem", trait=k))
                self.say(f"{p.name}: {self.cname(keep['champ'])} lên {keep['star']} sao")
                changed = True
                break

    def _put_item(self, u, it):
        """Gắn đồ it lên quân u (tự ghép mảnh). False nếu hết chỗ."""
        d = self.data
        if d.items[it]["kind"] == "component":
            for j, x in enumerate(u["items"]):
                if d.items[x]["kind"] == "component":
                    u["items"][j] = d.combine(x, it)
                    return True
        if len(u["items"]) + len(u["emblems"]) >= self.cfg["max_items"]:
            return False
        u["items"].append(it)
        return True

    def buy(self, p, slot):
        c = p.shop[slot] if 0 <= slot < 5 else -1
        if c < 0:
            raise ActionError("ô shop trống")
        cost = int(self.cost[c])
        if p.gold < cost:
            raise ActionError("không đủ vàng")
        if self.pool[c] <= 0:
            raise ActionError("hết tướng này trong kho chung")
        if not self.add_unit(p, c):
            raise ActionError("hàng chờ đầy")
        p.gold -= cost
        self.pool[c] -= 1
        p.shop[slot] = -1
        return True

    def sell_value(self, c, star):
        cost = int(self.cost[c])
        v = cost * 3 ** (star - 1)
        return v - 1 if (star > 1 and cost > 1) else v

    def sell(self, p, uid):
        u = self.unit(p, uid)
        p.gold += self.sell_value(u["champ"], u["star"])
        self.pool[u["champ"]] += 3 ** (u["star"] - 1)
        for it in u["items"]:
            p.inventory.append(dict(kind="item", id=it))
        for k in u["emblems"]:
            p.inventory.append(dict(kind="emblem", trait=k))
        p.units.remove(u)
        return True

    def roll(self, p):
        if p.free_rolls_left > 0:
            p.free_rolls_left -= 1
        elif p.gold >= self.eco["roll_cost"]:
            p.gold -= self.eco["roll_cost"]
        else:
            raise ActionError("không đủ vàng")
        self.refresh_shop(p)
        return True

    def buy_xp(self, p):
        if p.level >= 9:
            raise ActionError("đã cấp tối đa")
        if p.gold < self.eco["xp_cost"]:
            raise ActionError("không đủ vàng")
        p.gold -= self.eco["xp_cost"]
        p.xp += self.eco["xp_per_buy"]
        self.level_up(p)
        return True

    def move(self, p, uid, to):
        """to = {"bench": i} hoặc {"cell": c}; ô có quân thì đổi chỗ."""
        u = self.unit(p, uid)
        if "bench" in to:
            dest = ("bench", int(to["bench"]))
            if not 0 <= dest[1] < self.cfg["bench_slots"]:
                raise ActionError("ô hàng chờ không hợp lệ")
        else:
            dest = ("cell", int(to["cell"]))
            if not 0 <= dest[1] < 28:
                raise ActionError("ô bàn cờ không hợp lệ")
        other = next((v for v in p.units if v is not u and v[dest[0]] == dest[1]), None)
        src = ("cell", u["cell"]) if u["cell"] is not None else ("bench", u["bench"])
        if other is None and dest[0] == "cell" and src[0] == "bench" and len(p.board()) >= p.level:
            raise ActionError(f"sân đã đủ {p.level} quân (bằng cấp)")
        u["cell"] = u["bench"] = None
        u[dest[0]] = dest[1]
        if other is not None:
            other["cell"] = other["bench"] = None
            other[src[0]] = src[1]
        return True

    def equip(self, p, inv, uid):
        if not 0 <= inv < len(p.inventory):
            raise ActionError("không có món này")
        e = p.inventory[inv]
        u = self.unit(p, uid)
        if e["kind"] == "emblem":
            if e["trait"] in self.data.champ_traits[u["champ"]] or e["trait"] in u["emblems"]:
                raise ActionError("tướng đã có tộc hệ này")
            if len(u["items"]) + len(u["emblems"]) >= self.cfg["max_items"]:
                raise ActionError("tướng đã đủ 3 món")
            u["emblems"].append(e["trait"])
        elif not self._put_item(u, e["id"]):
            raise ActionError("tướng đã đủ 3 món")
        p.inventory.pop(inv)
        return True

    def unequip(self, p, uid, slot):
        """Tháo đồ (luật bản 5 cho tháo / đổi đồ tự do). slot < số đồ: đồ; còn lại: ấn tộc."""
        u = self.unit(p, uid)
        if len(p.inventory) >= self.cfg["item_bench"]:
            raise ActionError("kho đồ đầy")
        if slot < len(u["items"]):
            p.inventory.append(dict(kind="item", id=u["items"].pop(slot)))
        elif slot - len(u["items"]) < len(u["emblems"]):
            p.inventory.append(dict(kind="emblem", trait=u["emblems"].pop(slot - len(u["items"]))))
        else:
            raise ActionError("không có đồ ở ô này")
        return True

    # ================================================================ lõi
    def offer_augments(self):
        k = self.cfg["augment_rounds"].index(self.label)
        tier = self.aug_tiers[k]
        pool = [a["id"] for a in self.augs if a["tier"] == tier]
        self.pending_aug = {}
        for p in self.alive():
            opts = [a for a in pool if a not in p.augments]
            self.pending_aug[p.idx] = [int(x) for x in self.rng.choice(opts, size=3, replace=False)]
        for i, bot in self.bots.items():
            if i in self.pending_aug:
                self.choose_augment(i, bot.pick_augment(self.pending_aug[i]), advance=False)

    def choose_augment(self, pi, index, advance=True):
        opts = self.pending_aug.get(pi)
        if not opts:
            raise ActionError("không có lõi để chọn")
        aid = opts[index]
        p = self.players[pi]
        a = self.augs[aid]
        p.augments.append(aid)
        econ = a.get("econ", {})
        p.gold += econ.get("gold_now", 0)
        p.gold_per_round += econ.get("gold_per_round", 0)
        p.interest_bonus += econ.get("interest_max", 0)
        p.xp_per_round += econ.get("xp_per_round", 0)
        p.free_rolls += econ.get("free_rolls", 0)
        for _ in range(econ.get("components", 0)):
            p.inventory.append(dict(kind="item", id=int(self.rng.choice(self.components))))
        for _ in range(econ.get("combined", 0)):
            p.inventory.append(dict(kind="item", id=int(self.rng.choice(self.combined))))
        if "emblem" in econ:
            p.inventory.append(dict(kind="emblem", trait=self.data.trait_id(econ["emblem"])))
        for t, n in a.get("trait_bonus", {}).items():
            p.trait_bonus[t] = p.trait_bonus.get(t, 0) + n
        p.effects += a.get("effects", [])
        del self.pending_aug[pi]
        self.say(f"{p.name} chọn lõi {a['name']}")
        if advance and not self.pending_aug and self.phase == "augment":
            self.begin_plan()
        return True

    # ================================================================ chọn tướng chung
    def start_carousel(self):
        costs = self.cfg["carousel_costs"][str(min(self.stage, 8))]
        opts = []
        for _ in range(9):
            for _try in range(20):
                c = int(self.rng.choice(self.by_cost[int(self.rng.choice(costs)) - 1]))
                if self.pool[c] > 0:
                    break
            self.pool[c] -= 1
            opts.append(dict(champ=c, item=int(self.rng.choice(self.components)), taken_by=None))
        alive = self.alive()
        tie = self.rng.permutation(len(alive))
        order = [p.idx for _, _, p in sorted(zip([p.hp for p in alive], tie, alive), key=lambda x: (x[0], x[1]))]
        self.carousel = dict(options=opts, order=order, pos=0)
        self.phase = "carousel"
        self.say("chọn tướng chung: " + ", ".join(self.cname(o["champ"]) for o in opts))
        self._carousel_bots()

    def _carousel_bots(self):
        car = self.carousel
        while car["pos"] < len(car["order"]):
            pi = car["order"][car["pos"]]
            if pi not in self.bots:
                return             # chờ người chơi
            free = [i for i, o in enumerate(car["options"]) if o["taken_by"] is None]
            self.carousel_pick(pi, self.bots[pi].pick_carousel(free), auto=True)
        self.end_carousel()

    def carousel_pick(self, pi, index, auto=False):
        car = self.carousel
        if car is None or car["order"][car["pos"]] != pi:
            raise ActionError("chưa tới lượt bạn chọn")
        o = car["options"][index]
        if o["taken_by"] is not None:
            raise ActionError("tướng này đã có người lấy")
        p = self.players[pi]
        o["taken_by"] = pi
        self.pool[o["champ"]] += 1                 # trả lại kho rồi lấy như mua
        if self.add_unit(p, o["champ"], items=[o["item"]]):
            self.pool[o["champ"]] -= 1
        else:
            p.inventory.append(dict(kind="item", id=o["item"]))
        p.components_got += 1
        self.say(f"{p.name} lấy {self.cname(o['champ'])} + {self.data.items[o['item']]['name']}")
        car["pos"] += 1
        if not auto:
            self._carousel_bots()
        return True

    def end_carousel(self):
        for o in self.carousel["options"]:
            if o["taken_by"] is None:
                self.pool[o["champ"]] += 1
        self.carousel = None
        self.advance()

    # ================================================================ giao tranh
    def auto_fill(self, p):
        """Như ĐTCL: sân còn chỗ theo cấp thì tự đưa quân mạnh nhất ở hàng chờ lên trước khi đánh."""
        while len(p.board()) < p.level and p.bench():
            u = max(p.bench(), key=lambda v: (self.cost[v["champ"]] * 3 ** (v["star"] - 1), -v["uid"]))
            used = {v["cell"] for v in p.board()}
            rng_ = self.data.champs[u["champ"]]["stats"]["range"]
            prefs = [21 + i for i in (3, 2, 4, 1, 5, 0, 6)] + [14 + i for i in range(7)] if rng_ <= 1 else \
                [i for i in (3, 2, 4, 1, 5, 0, 6)] + [7 + i for i in range(7)]
            cell = next((c for c in prefs + list(range(28)) if c not in used), None)
            u["bench"], u["cell"] = None, cell

    def team_of(self, p):
        team = [SimUnit(u["champ"], u["star"], u["cell"], items=list(u["items"]),
                        emblems=[self.data.trait_names[k] for k in u["emblems"]]) for u in p.board()]
        return team, TeamExtra(effects=list(p.effects), trait_bonus=dict(p.trait_bonus))

    def monster_team(self):
        name_id = {n: i for i, n in enumerate(self.data.names)}
        return [SimUnit(name_id[n], s, c) for n, s, c in self.cfg["pve"].get(self.label, self.cfg["pve"]["7-7"])]

    def resolve_round(self):
        if self.phase != "plan":
            raise ActionError("không ở lượt mua bán")
        k = self.kind()
        alive = self.alive()
        for p in alive:
            self.auto_fill(p)
        fights = []          # (a, b|None ghost/monster, team_a, team_b, extra_a, extra_b, kind)
        if k == "pve":
            mon = self.monster_team()
            for p in alive:
                ta, ea = self.team_of(p)
                fights.append(dict(a=p.idx, b=None, ghost=None, ta=ta, tb=mon, ea=ea, eb=None, kind="pve"))
        else:
            idx = [p.idx for p in alive]
            order = list(self.rng.permutation(idx))
            for _ in range(20):      # tránh gặp lại đối thủ vòng trước
                pairs = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
                bad = any(self.players[a].last and self.players[a].last.get("opp") == b for a, b in pairs)
                if not bad:
                    break
                order = list(self.rng.permutation(idx))
            for i in range(0, len(order) - 1, 2):
                a, b = int(order[i]), int(order[i + 1])
                ta, ea = self.team_of(self.players[a])
                tb, eb = self.team_of(self.players[b])
                fights.append(dict(a=a, b=b, ghost=None, ta=ta, tb=tb, ea=ea, eb=eb, kind="pvp"))
            if len(order) % 2 == 1:
                a = int(order[-1])
                g = int(self.rng.choice([i for i in idx if i != a]))
                ta, ea = self.team_of(self.players[a])
                tb, eb = self.team_of(self.players[g])
                fights.append(dict(a=a, b=None, ghost=g, ta=ta, tb=tb, ea=ea, eb=eb, kind="ghost"))
        n = len(fights)
        seeds = [_seed(self.seed, self.label, i) for i in range(n)]
        batch = self.data.build_batch([f["ta"] for f in fights], [f["tb"] for f in fights], seeds=seeds,
                                      extras_a=[f["ea"] for f in fights], extras_b=[f["eb"] for f in fights])
        sim = self.data.simulator(stochastic=True)
        res = sim.run(batch)
        dmg_stage = self.cfg["stage_damage"][str(min(self.stage, 8))]
        for i, f in enumerate(fights):
            f["seed"] = seeds[i]
            r = int(res.result[i])
            f["result"] = r
            f["time"] = round(float(res.end_step[i]) * 0.1, 1)
            surv_a = int(res.alive[i, :10].sum())
            surv_b = int(res.alive[i, 10:].sum())
            f["survivors"] = (surv_a, surv_b)
            f["dealt"] = res.dealt[i].sum(-1).round().tolist()
            pa = self.players[f["a"]]
            pb = self.players[f["b"]] if f["b"] is not None else None
            label_b = "quái" if f["kind"] == "pve" else (pb.name if pb else f"bóng {self.players[f['ghost']].name}")
            if r == 0:
                f["dmg"] = (0, dmg_stage + surv_a)
                if pb:
                    pb.hp -= f["dmg"][1]
            elif r == 1:
                f["dmg"] = (dmg_stage + surv_b, 0)
                pa.hp -= f["dmg"][0]
            else:
                f["dmg"] = (dmg_stage, dmg_stage if pb else 0)
                pa.hp -= f["dmg"][0]
                if pb:
                    pb.hp -= f["dmg"][1]
            if f["kind"] != "pve":
                self._streak(pa, r == 0)
                if pb:
                    self._streak(pb, r == 1)
            else:
                self.loot(pa, full=(r == 0))
            pa.last = dict(opp=f["b"], win=r == 0, dmg=f["dmg"][0], label=label_b, time=f["time"])
            if pb:
                pb.last = dict(opp=f["a"], win=r == 1, dmg=f["dmg"][1], label=pa.name, time=f["time"])
            res_txt = {0: f"{pa.name} thắng", 1: f"{label_b} thắng", 2: "hai bên cùng thua"}[r]
            self.say(f"{pa.name} vs {label_b}: {res_txt} ({f['time']} giây), máu mất {f['dmg'][0]}/{f['dmg'][1]}")
        self.fights = fights
        for f in fights:
            if f["a"] == 0 or f["b"] == 0:
                self.history.append(dict(label=self.label, **{k: f[k] for k in ("a", "b", "ghost", "ta", "tb", "ea",
                                                                                 "eb", "seed", "result", "kind")}))
        if self.stage == 1 and self.rnd == 4:
            for p in alive:          # bảo hiểm rơi đồ giai đoạn 1
                while p.components_got < self.cfg["loot"]["stage1_min_components"]:
                    p.inventory.append(dict(kind="item", id=int(self.rng.choice(self.components))))
                    p.components_got += 1
        self.eliminate()
        if self.phase != "over":
            self.advance()
        # người chơi đã bị loại: cho các bot đánh nốt tới hết ván
        while self.human and not self.players[0].alive and self.phase == "plan":
            self.resolve_round()
        return True

    def _streak(self, p, win):
        if win:
            p.streak = p.streak + 1 if p.streak > 0 else 1
        else:
            p.streak = p.streak - 1 if p.streak < 0 else -1

    def loot(self, p, full=True):
        L = self.cfg["loot"]
        drops = []
        if self.stage == 1:
            n = L["stage1_orbs"].get(self.label, 1)
            if not full:
                n = max(1, n // 2)
            for _ in range(n):
                x = self.rng.random()
                if x < 0.55:
                    drops.append(("item", int(self.rng.choice(self.components))))
                elif x < 0.85:
                    drops.append(("gold", int(self.rng.integers(2, 5))))
                else:
                    drops.append(("champ", int(self.rng.choice(self.by_cost[0]))))
        else:
            nc = L["later_components"] + (1 if self.stage >= L["late_extra_stage"] else 0)
            gold = L["later_gold"]
            if not full:
                nc, gold = max(1, nc // 2), gold // 2
            drops = [("item", int(self.rng.choice(self.components))) for _ in range(nc)] + [("gold", gold)]
        txt = []
        for kind, v in drops:
            if kind == "item":
                p.inventory.append(dict(kind="item", id=v))
                p.components_got += 1
                txt.append(self.data.items[v]["name"])
            elif kind == "gold":
                p.gold += v
                txt.append(f"{v} vàng")
            else:
                if self.pool[v] > 0 and self.add_unit(p, v):
                    self.pool[v] -= 1
                    txt.append(self.cname(v))
                else:
                    p.gold += 1
                    txt.append("1 vàng")
        if p.idx == 0:
            self.say(f"rơi từ quái: {', '.join(txt)}")

    def eliminate(self):
        dying = [p for p in self.alive() if p.hp <= 0]
        n_alive = len(self.alive())
        for rank, p in enumerate(sorted(dying, key=lambda q: -q.hp)):
            p.alive = False
            p.placement = n_alive - len(dying) + 1 + rank
            for u in p.units:
                self.pool[u["champ"]] += 3 ** (u["star"] - 1)
            p.units = []
            self.say(f"{p.name} bị loại, hạng {p.placement}")
        left = self.alive()
        if len(left) <= 1 or (self.stage >= self.cfg["max_stage"] and self.rnd >= 7):
            for rank, p in enumerate(sorted(left, key=lambda q: -q.hp)):
                p.placement = rank + 1
                p.alive = rank == 0 and len(left) == 1
            self.phase = "over"
            self.say("hết ván: " + ", ".join(f"{p.placement}. {p.name}" for p in sorted(self.players, key=lambda q: q.placement)))

    def advance(self):
        last = 4 if self.stage == 1 else 7
        if self.rnd >= last:
            self.stage, self.rnd = self.stage + 1, 1
        else:
            self.rnd += 1
        self.start_round()

    # ================================================================ chạy toàn bot (kiểm thử)
    def run_bots(self, max_rounds=200):
        for _ in range(max_rounds):
            if self.phase == "over":
                break
            if self.phase == "plan":
                self.resolve_round()
        return sorted(self.players, key=lambda q: q.placement)

    # ================================================================ tộc hệ của một người
    def traits_of(self, p):
        team, extra = self.team_of(p)
        lv = self.data.trait_levels(team, extra) if team else np.full(self.data.K, -1)
        count = {}
        seen = set()
        for u in p.board():
            if u["champ"] not in seen:
                seen.add(u["champ"])
                for k in self.data.champ_traits[u["champ"]]:
                    count[k] = count.get(k, 0) + 1
            for k in u["emblems"]:
                if k not in self.data.champ_traits[u["champ"]]:
                    count[k] = count.get(k, 0) + 1
        for t, n in p.trait_bonus.items():
            k = self.data.trait_id(t)
            count[k] = count.get(k, 0) + n
        return count, lv
