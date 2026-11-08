"""Bot luật cho ván bản 5. Ba lối kinh tế như các bản trước:
    tempo  lên cấp sớm, giữ ~20 vàng, roll phần dư từ giai đoạn 3, roll mạnh ở giai đoạn 4
    bank   giữ 50 vàng ăn lãi, chỉ lên cấp bằng phần dư; roll mạnh từ 4-1 (hoặc khi máu dưới 40)
    tank   chịu thua sớm giữ vàng (chuỗi thua), lên 8 ở giai đoạn 4 rồi roll
Chung cho cả ba:
    chọn "tộc chính" theo các tướng đang có (2 tộc thường nặng ký nhất), mua tướng ghép sao / hợp tộc chính / mạnh;
    đưa lên sân các quân điểm cao nhất; ghép đồ theo bảng ưu tiên cho chủ lực (quân đắt nhất tầm xa) và đỡ đòn;
    xếp quân: tầm gần hàng đầu (Hộ vệ giữa), Sát thủ góc hàng 2, tầm xa hàng sau (chủ lực ở góc);
    lõi theo lối (bank: kinh tế, tempo: chỉ số, tank: tộc hệ / đồ); chọn tướng chung theo tộc chính rồi theo đồ.
"""
from __future__ import annotations

import numpy as np

ITEM_PREF = {
    "Pháp sư": ["Ngọc Lam", "Mũ Phù Thủy", "Găng Ngọc", "Trượng Thiên Thần", "Nanh Rắn", "Bùa Dịch Hạch",
                "Kiếm Súng", "Bàn Tay Công Lý"],
    "Xạ thủ": ["Cuồng Cung", "Vô Cực Kiếm", "Diệt Khổng Lồ", "Lời Thì Thầm", "Cung Bão", "Kiếm Tử Thần",
               "Huyết Kiếm", "Bàn Tay Công Lý"],
    "Sát thủ": ["Vô Cực Kiếm", "Kiếm Tử Thần", "Huyết Kiếm", "Bàn Tay Công Lý", "Áo Choàng Bóng Đêm",
                "Găng Thợ Săn", "Diệt Khổng Lồ"],
    "Đấu sĩ": ["Huyết Kiếm", "Găng Bạo Lực", "Ý Chí Titan", "Kiếm Tử Thần", "Thương Thần", "Bàn Tay Công Lý"],
    "tank": ["Áo Gai", "Vuốt Rồng", "Giáp Máu", "Giáp Thạch Quỷ", "Ngọc Cứu Rỗi", "Áo Choàng Lửa", "Tim Băng",
             "Khăn Hư Vô", "Vương Miện Hộ Vệ", "Giáp Kiên Định", "Ngọc Lôi", "Khăn Giải Thuật"],
}
AUG_PREF = {"bank": ["Kinh tế", "Đồ", "Chỉ số", "Tộc hệ"], "tempo": ["Chỉ số", "Đồ", "Tộc hệ", "Kinh tế"],
            "tank": ["Tộc hệ", "Đồ", "Chỉ số", "Kinh tế"]}
TARGET_LEVEL = {"tempo": {1: 3, 2: 5, 3: 7, 4: 8, 5: 8, 6: 9, 7: 9, 8: 9},
                "bank": {1: 3, 2: 4, 3: 6, 4: 8, 5: 8, 6: 9, 7: 9, 8: 9},
                "tank": {1: 3, 2: 3, 3: 5, 4: 8, 5: 8, 6: 9, 7: 9, 8: 9}}


def make_bot(style, game, idx):
    return RuleBot(style, game, idx)


class RuleBot:
    def __init__(self, style, game, idx):
        if style not in TARGET_LEVEL:
            raise ValueError(f"không có bot {style}")
        self.style, self.g, self.i = style, game, idx

    @property
    def p(self):
        return self.g.players[self.i]

    # ------------------------------------------------------------ đánh giá
    def unit_value(self, c, star):
        return float(self.g.cost[c]) * (3.0 ** (star - 1)) ** 0.8

    def main_traits(self):
        d, g = self.g.data, self.g
        w = {}
        for u in self.p.units:
            for k in d.champ_traits[u["champ"]]:
                if d.traits[k]["kind"] == "unique":
                    continue
                w[k] = w.get(k, 0.0) + 1.0 + 0.3 * g.cost[u["champ"]]
        return [k for k, _ in sorted(w.items(), key=lambda x: -x[1])[:2]]

    def score(self, u, main):
        d = self.g.data
        s = self.unit_value(u["champ"], u["star"])
        s += 1.5 * sum(1 for k in d.champ_traits[u["champ"]] if k in main)
        s += 0.8 * (len(u["items"]) + len(u["emblems"]))
        return s

    # ------------------------------------------------------------ lượt mua bán
    def plan(self):
        g, p = self.g, self.p
        stage = g.stage
        st = self.style
        reserve = {"tempo": 20, "bank": 50, "tank": 50}[st]
        rolling = (st == "tempo" and stage >= 3) or (st != "tempo" and (stage >= 4 or p.hp < 40)) or stage >= 5
        floor = 10 if (stage >= 4 or p.hp < 40) else reserve
        if st == "tank" and stage == 4 and g.rnd <= 2:
            floor = 0
        if p.hp < 25:
            floor = 0
        # lên cấp
        tgt = TARGET_LEVEL[st][min(stage, 8)]
        while p.level < tgt and p.gold >= 4 and (p.gold - 4 >= min(floor, 20) or stage >= 4):
            g.buy_xp(p)
        self.shop_pass(floor if not rolling else 0)
        # roll
        tries = 0
        while rolling and tries < 30 and (p.free_rolls_left > 0 or p.gold - 2 >= floor):
            g.roll(p)
            self.shop_pass(floor)
            tries += 1
        while p.free_rolls_left > 0:
            g.roll(p)
            self.shop_pass(floor)
        self.arrange()
        self.items()
        self.position()

    def shop_pass(self, floor):
        g, p, d = self.g, self.p, self.g.data
        main = self.main_traits()
        for slot in range(5):
            c = p.shop[slot]
            if c < 0:
                continue
            cost = int(g.cost[c])
            owned = [u for u in p.units if u["champ"] == c and u["star"] < 3]
            fits = any(k in main for k in d.champ_traits[c])
            full = g.free_bench(p) is None
            want = bool(owned) or fits or len(p.units) < p.level + 2
            short = len(p.units) < p.level          # thiếu quân trên sân: mua bất kể vàng dự trữ
            spend_ok = p.gold - cost >= floor or (owned and p.gold >= cost) or short
            if not want or not spend_ok or p.gold < cost:
                continue
            if full and not (owned and sum(1 for u in owned if u["star"] == 1) >= 2):
                # hàng chờ đầy: bán quân hàng chờ yếu nhất không hợp tộc
                weak = sorted(p.bench(), key=lambda u: self.score(u, main))
                if weak and self.score(weak[0], main) < self.unit_value(c, 1) + (1.5 if fits else 0):
                    g.sell(p, weak[0]["uid"])
                else:
                    continue
            try:
                g.buy(p, slot)
            except Exception:
                pass

    def arrange(self):
        """Đưa quân điểm cao nhất lên sân (đủ cấp), phần còn lại xuống hàng chờ; bán bớt nếu thừa."""
        g, p = self.g, self.p
        main = self.main_traits()
        ranked = sorted(p.units, key=lambda u: -self.score(u, main))
        # một tướng chỉ tính tộc một lần: ưu tiên các tướng khác nhau
        picked, seen = [], set()
        for u in ranked:
            if len(picked) >= p.level:
                break
            if u["champ"] in seen and any(v["champ"] not in seen for v in ranked if v not in picked):
                continue
            picked.append(u)
            seen.add(u["champ"])
        for u in ranked:
            if len(picked) < p.level and u not in picked:
                picked.append(u)
        for u in p.units:
            u["cell"] = None
            u["bench"] = None
        for j, u in enumerate([u for u in p.units if u not in picked]):
            u["bench"] = j
        for j, u in enumerate(picked):
            u["cell"] = 100 + j          # tạm, position() đặt lại
        extra = [u for u in p.units if u["bench"] is not None and u["bench"] >= g.cfg["bench_slots"]]
        for u in extra:
            g.sell(p, u["uid"])

    def roles(self):
        d = self.g.data
        board = self.p.board()
        if not board:
            return None, None
        ranged = [u for u in board if d.champs[u["champ"]]["stats"]["range"] > 1]
        pool = ranged or board
        carry = max(pool, key=lambda u: (self.unit_value(u["champ"], u["star"]), -u["uid"]))
        melee = [u for u in board if d.champs[u["champ"]]["stats"]["range"] <= 1 and u is not carry]
        tank = max(melee, key=lambda u: (d.champs[u["champ"]]["role"] == "Hộ vệ",
                                         self.unit_value(u["champ"], u["star"]), -u["uid"])) if melee else None
        return carry, tank

    def items(self):
        g, p, d = self.g, self.p, self.g.data
        carry, tank = self.roles()
        if carry is None:
            return
        name_id = {n: i for i, n in enumerate(d.item_names)}
        for target, pref in ((carry, ITEM_PREF.get(d.champs[carry["champ"]]["role"], ITEM_PREF["Sát thủ"])),
                             (tank, ITEM_PREF["tank"])):
            if target is None:
                continue
            for nm in pref:
                if len(target["items"]) + len(target["emblems"]) >= 3:
                    break
                it = name_id[nm]
                inv = [e["id"] for e in p.inventory if e["kind"] == "item"]
                if it in inv:
                    g.equip(p, [j for j, e in enumerate(p.inventory) if e["kind"] == "item" and e["id"] == it][0],
                            target["uid"])
                    continue
                a, b = d.items[it]["recipe"]
                if a in inv and b in inv and (a != b or inv.count(a) >= 2):
                    j = next(j for j, e in enumerate(p.inventory) if e["kind"] == "item" and e["id"] == a)
                    g.equip(p, j, target["uid"])
                    j = next(j for j, e in enumerate(p.inventory) if e["kind"] == "item" and e["id"] == b)
                    g.equip(p, j, target["uid"])
        # đồ ghép còn trong kho (từ lõi) và ấn tộc: gắn cho quân trên sân còn chỗ
        for e in list(p.inventory):
            j = p.inventory.index(e)
            if e["kind"] == "item" and d.items[e["id"]]["kind"] == "component":
                continue
            for u in sorted(p.board(), key=lambda u: -self.unit_value(u["champ"], u["star"])):
                try:
                    g.equip(p, j, u["uid"])
                    break
                except Exception:
                    continue
        # cuối ván (giai đoạn 5+): gắn hết mảnh lẻ cho chủ lực / đỡ đòn
        if g.stage >= 5:
            for target in (carry, tank):
                while target is not None and any(e["kind"] == "item" for e in p.inventory):
                    j = next(j for j, e in enumerate(p.inventory) if e["kind"] == "item")
                    try:
                        g.equip(p, j, target["uid"])
                    except Exception:
                        break

    def position(self):
        d, p = self.g.data, self.p
        carry, tank = self.roles()
        front = [21 + c for c in (3, 2, 4, 1, 5, 0, 6)]
        second = [14 + c for c in (3, 2, 4, 1, 5, 0, 6)]
        back = [0, 6, 1, 5, 2, 4, 3] + [7 + c for c in (0, 6, 1, 5, 2, 4, 3)]
        used = set()

        def take(prefs):
            for c in prefs:
                if c not in used:
                    used.add(c)
                    return c
            for c in range(28):
                if c not in used:
                    used.add(c)
                    return c

        board = p.board()
        order = sorted(board, key=lambda u: (u is not tank, u is not carry, -self.unit_value(u["champ"], u["star"])))
        for u in order:
            ch = d.champs[u["champ"]]
            if ch["stats"]["range"] > 1:
                u["cell"] = take(back)
            elif ch["role"] == "Sát thủ":
                u["cell"] = take([14, 20, 21, 27] + second)
            else:
                u["cell"] = take(front)

    # ------------------------------------------------------------ lõi, chọn tướng chung
    def pick_augment(self, opts):
        augs = self.g.augs
        main = [self.g.data.trait_names[k] for k in self.main_traits()]
        best, bs = 0, -1e9
        for j, a in enumerate(opts):
            A = augs[a]
            s = -AUG_PREF[self.style].index(A["group"]) * 10.0
            if A["group"] == "Tộc hệ":
                tr = list(A.get("trait_bonus", {})) + ([A["econ"]["emblem"]] if "emblem" in A.get("econ", {}) else [])
                s += 15.0 if any(t in main for t in tr) else -5.0
            s += A["tier"] + float(self.g.rng.random()) * 0.1
            if s > bs:
                best, bs = j, s
        return best

    def pick_carousel(self, free):
        g, d = self.g, self.g.data
        main = self.main_traits()
        opts = g.carousel["options"]

        def sc(i):
            o = opts[i]
            c = o["champ"]
            s = float(g.cost[c])
            s += 3.0 * sum(1 for k in d.champ_traits[c] if k in main)
            s += 2.0 * sum(1 for u in self.p.units if u["champ"] == c)
            return s + float(g.rng.random()) * 0.1
        return max(free, key=sc)
