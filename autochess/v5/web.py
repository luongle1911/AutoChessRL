"""Giao diện web chơi thử luật bản 5 (bạn ngồi ghế 0, 7 ghế còn lại là bot luật).

    python3 -m autochess.v5.web --seed 1
    # rồi mở http://localhost:8005

API (JSON):
  GET  /api/catalog       dữ liệu tĩnh: tướng (chỉ số, kỹ năng), tộc hệ (mốc, hiệu ứng), đồ (công thức), lõi
  GET  /api/state         trạng thái ván nhìn từ ghế 0
  POST /api/act           {type: buy|sell|roll|xp|move|equip|unequip|ready|augment|carousel, ...}
  POST /api/new {seed}    ván mới
  GET  /replay/<n>        xem lại trận thứ n của bạn (trình xem của battlesim)
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from battlesim.effects import describe
from battlesim.team import CombatData
from .game import Game, ActionError

HERE = os.path.dirname(os.path.abspath(__file__))
TIER_VN = {1: "bạc", 2: "vàng", 3: "kim cương"}


def catalog_json(g: Game) -> dict:
    d = g.data
    champs = []
    for c in d.champs[:d.n_shop]:
        st = c["stats"]
        champs.append(dict(id=c["id"], name=c["name"], cost=c["cost"], role=c["role"],
                           traits=[d.trait_names[k] for k in c["traits"]], stats=st,
                           skill=c["skill"]["name"], skill_text=c["skill"]["text"],
                           skill_detail=["; ".join(describe(b, lv) for b in c["skill"]["effects"]) for lv in range(3)]))
    traits = []
    for t in d.traits:
        nlv = len(t["breakpoints"])
        traits.append(dict(name=t["name"], kind=t["kind"], breakpoints=t["breakpoints"], scope=t["scope"],
                           text=t.get("text", ""),
                           levels=["; ".join(describe(b, lv) for b in t["effects"]) for lv in range(nlv)],
                           members=[c["name"] for c in d.champs[:d.n_shop] if t["id"] in c["traits"]]))
    comp = {i["id"]: i["name"] for i in d.items if i["kind"] == "component"}
    items = []
    for i in d.items:
        own = i["effects"][2:] if i["kind"] == "combined" else i["effects"]
        items.append(dict(id=i["id"], name=i["name"], kind=i["kind"], group=i.get("group", "Mảnh"),
                          recipe=[comp[k] for k in i.get("recipe", [])],
                          text="; ".join(describe(b, 0) for b in own),
                          stats="; ".join(describe(b, 0) for b in i["effects"][:2]) if i["kind"] == "combined" else ""))
    augs = [dict(id=a["id"], name=a["name"], tier=TIER_VN[a["tier"]], group=a["group"], text=a["text"]) for a in g.augs]
    return dict(version=d.version, champions=champs, traits=traits, items=items, augments=augs,
                shop_odds=g.raw_odds, xp_to_next=g.xp_next, rules=g.cfg["note"])


def _unit(g, u):
    d = g.data
    c = d.champs[u["champ"]]
    return dict(uid=u["uid"], champ=u["champ"], name=c["name"], star=u["star"], cell=u["cell"], bench=u["bench"],
                cost=c["cost"], role=c["role"], range=c["stats"]["range"],
                items=[dict(id=i, name=d.items[i]["name"], kind=d.items[i]["kind"]) for i in u["items"]],
                emblems=[d.trait_names[k] for k in u["emblems"]], sell=g.sell_value(u["champ"], u["star"]))


def view(g: Game) -> dict:
    d = g.data
    me = g.players[0]
    count, lv = g.traits_of(me)
    traits = []
    for k, n in sorted(count.items(), key=lambda x: (-int(lv[x[0]] >= 0), -x[1])):
        t = d.traits[k]
        traits.append(dict(name=t["name"], count=n, breakpoints=t["breakpoints"], level=int(lv[k]),
                           kind=t["kind"]))
    eco = g.eco
    interest = min(me.gold // eco["interest_step"], eco["interest_max"] + me.interest_bonus)
    out = dict(
        label=g.label, phase=g.phase, kind=g.kind(), stage=g.stage, rnd=g.rnd, seed=g.seed,
        me=dict(hp=me.hp, gold=me.gold, level=me.level, xp=me.xp,
                xp_next=g.xp_next[me.level - 1] if me.level < 9 else 0, streak=me.streak, alive=me.alive,
                placement=me.placement, free_rolls=me.free_rolls_left, interest=interest,
                shop=[None if c < 0 else dict(slot=i, champ=c, name=d.names[c], cost=int(g.cost[c]),
                                              role=d.champs[c]["role"],
                                              traits=[d.trait_names[k] for k in d.champ_traits[c]],
                                              owned=sum(1 for u in me.units if u["champ"] == c))
                      for i, c in enumerate(me.shop)],
                units=[_unit(g, u) for u in me.units],
                inventory=[dict(idx=j, name=g.iname(e), kind=e["kind"],
                                item=(e["id"] if e["kind"] == "item" else None),
                                component=(e["kind"] == "item" and d.items[e["id"]]["kind"] == "component"))
                           for j, e in enumerate(me.inventory)],
                augments=[dict(name=g.augs[a]["name"], text=g.augs[a]["text"], group=g.augs[a]["group"])
                          for a in me.augments],
                traits=traits, board_cap=me.level, on_board=len(me.board())),
        players=[dict(idx=p.idx, name=p.name, hp=p.hp, level=p.level, streak=p.streak, alive=p.alive,
                      placement=p.placement, gold=p.gold,
                      last=p.last, augments=[g.augs[a]["name"] for a in p.augments],
                      board=[_unit(g, u) for u in p.board()], bench=[_unit(g, u) for u in p.bench()])
                 for p in g.players],
        log=g.log[-60:],
        fights=[],
        history=[dict(n=i, label=h["label"], result=h["result"], mine_side=0 if h["a"] == 0 else 1,
                      kind=h["kind"]) for i, h in enumerate(g.history)][-12:],
    )
    for f in g.fights:
        pa = g.players[f["a"]]
        nb = "quái" if f["kind"] == "pve" else (g.players[f["b"]].name if f["b"] is not None
                                                else f"bóng của {g.players[f['ghost']].name}")
        out["fights"].append(dict(a=pa.name, b=nb, result=f["result"], time=f["time"], dmg=list(f["dmg"]),
                                  mine=f["a"] == 0 or f["b"] == 0, survivors=list(f["survivors"])))
    if g.phase == "augment" and 0 in g.pending_aug:
        out["augment_options"] = [dict(index=j, name=g.augs[a]["name"], text=g.augs[a]["text"],
                                       group=g.augs[a]["group"], tier=TIER_VN[g.augs[a]["tier"]])
                                  for j, a in enumerate(g.pending_aug[0])]
    if g.carousel is not None:
        car = g.carousel
        out["carousel"] = dict(
            options=[dict(index=j, champ=o["champ"], name=d.names[o["champ"]], cost=int(g.cost[o["champ"]]),
                          traits=[d.trait_names[k] for k in d.champ_traits[o["champ"]]],
                          item=d.items[o["item"]]["name"],
                          taken_by=(g.players[o["taken_by"]].name if o["taken_by"] is not None else None))
                     for j, o in enumerate(car["options"])],
            order=[g.players[i].name for i in car["order"]], pos=car["pos"],
            my_turn=car["pos"] < len(car["order"]) and car["order"][car["pos"]] == 0)
    if g.phase == "over":
        out["standings"] = [dict(name=p.name, placement=p.placement, hp=p.hp)
                            for p in sorted(g.players, key=lambda q: q.placement)]
    return out


class App:
    def __init__(self, seed=1, players=8, bots=None):
        self.data = CombatData.load()
        self.players, self.bots = players, bots
        self.lock = threading.Lock()
        self.new(seed)

    def new(self, seed):
        g = Game(self.data, n_players=self.players, seed=seed, human=True, bots=self.bots)
        g.raw_odds = self.data.raw["shop_odds"]
        self.game = g

    def replay_html(self, n):
        from battlesim.viewer2 import record, write_html
        import tempfile
        h = self.game.history[n]
        rep = record(self.data, h["ta"], h["tb"], seed=h["seed"], stochastic=True, extra_a=h["ea"], extra_b=h["eb"])
        path = os.path.join(tempfile.gettempdir(), f"autochess_v5_replay_{os.getpid()}.html")
        write_html(rep, path)
        with open(path, "rb") as f:
            return f.read()


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if self.path == "/api/catalog":
                return self._send(200, catalog_json(app.game))
            if self.path == "/api/state":
                with app.lock:
                    return self._send(200, view(app.game))
            if self.path.startswith("/replay/"):
                try:
                    n = int(self.path.split("/")[-1])
                    with app.lock:
                        html = app.replay_html(n)
                    return self._send(200, html, "text/html; charset=utf-8")
                except Exception as e:      # noqa: BLE001
                    return self._send(404, {"error": str(e)})
            self._send(404, {"error": "không có"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            with app.lock:
                if self.path == "/api/new":
                    app.new(int(body.get("seed", int(time.time()) % 100000)))
                    return self._send(200, view(app.game))
                if self.path == "/api/act":
                    err = None
                    try:
                        app.game.act(0, body)
                    except ActionError as e:
                        err = str(e)
                    out = view(app.game)
                    out["error"] = err
                    return self._send(200, out)
            self._send(404, {"error": "không có"})
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-p", "--players", type=int, default=8)
    ap.add_argument("--bots", default="tempo,bank,tank", help="lối của các bot, xoay vòng")
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    app = App(a.seed, a.players, a.bots.split(","))
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    print(f"mở trình duyệt: http://localhost:{a.port}   ({app.data.version}; Ctrl+C để thoát)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
