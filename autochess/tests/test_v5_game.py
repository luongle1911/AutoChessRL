"""Ván đấu luật bản 5 (autochess/v5): vòng, kinh tế, gộp sao, đồ, lõi, chọn tướng chung, bot, giao diện web."""
import json
import threading
import urllib.request

import numpy as np
import pytest

from autochess.v5.game import Game, ActionError
from battlesim.team import CombatData

DATA = CombatData.load()


def _copies(g):
    owned = np.zeros_like(g.pool)
    for p in g.players:
        for u in p.units:
            owned[u["champ"]] += 3 ** (u["star"] - 1)
    return owned


def test_bot_game_finishes_with_valid_placements():
    g = Game(DATA, seed=11, human=False)
    total = g.pool.copy() + _copies(g)          # sau 1-1 (chọn tướng chung đã chạy)
    g.run_bots()
    assert g.phase == "over"
    assert sorted(p.placement for p in g.players) == list(range(1, 9))
    assert (g.pool >= 0).all()
    assert (g.pool + _copies(g) == total).all(), "kho chung + quân đang có phải giữ nguyên"
    assert all(len(p.augments) >= 1 for p in g.players)


def test_schedule_kinds():
    g = Game(DATA, seed=1, human=False)
    assert g.kind(1, 1) == "carousel" and g.kind(1, 2) == "pve" and g.kind(2, 1) == "pvp"
    assert g.kind(2, 4) == "carousel" and g.kind(2, 7) == "pve" and g.kind(3, 5) == "pvp"


def test_buy_merge_items_and_unequip():
    g = Game(DATA, seed=2, human=True)
    while g.phase == "carousel":
        g.act(0, {"type": "carousel", "index": next(i for i, o in enumerate(g.carousel["options"])
                                                   if o["taken_by"] is None)})
    p = g.players[0]
    p.gold = 100
    c = int(g.by_cost[0][0])
    p.shop = [c, c, c, -1, -1]
    before = len(p.units)
    for s in range(3):
        g.act(0, {"type": "buy", "slot": s})
    stars = [u["star"] for u in p.units if u["champ"] == c]
    assert 2 in stars and len(p.units) == before + 1
    u = next(u for u in p.units if u["champ"] == c)
    kiem, cung = DATA.item_id("Kiếm"), DATA.item_id("Cung")
    p.inventory = [dict(kind="item", id=kiem), dict(kind="item", id=cung)]
    g.act(0, {"type": "equip", "inv": 0, "uid": u["uid"]})
    g.act(0, {"type": "equip", "inv": 0, "uid": u["uid"]})
    assert u["items"] == [DATA.item_id("Diệt Khổng Lồ")] and p.inventory == []
    g.act(0, {"type": "unequip", "uid": u["uid"], "slot": 0})          # tháo đồ tự do
    assert u["items"] == [] and p.inventory[0]["id"] == DATA.item_id("Diệt Khổng Lồ")
    gold = p.gold
    g.act(0, {"type": "sell", "uid": u["uid"]})
    assert p.gold == gold + g.sell_value(c, 2)


def test_board_limit_and_move_swap():
    g = Game(DATA, seed=3, human=True)
    while g.phase == "carousel":
        g.act(0, {"type": "carousel", "index": next(i for i, o in enumerate(g.carousel["options"])
                                                   if o["taken_by"] is None)})
    p = g.players[0]
    p.gold = 100
    for s in range(5):
        if p.shop[s] >= 0:
            try:
                g.act(0, {"type": "buy", "slot": s})
            except ActionError:
                pass
    bench = [u for u in p.units if u["bench"] is not None]
    assert len(bench) >= 2
    g.act(0, {"type": "move", "uid": bench[0]["uid"], "to": {"cell": 24}})
    with pytest.raises(ActionError):       # cấp 1 (hoặc 2): sân đã đủ
        for u in bench[1:]:
            g.act(0, {"type": "move", "uid": u["uid"], "to": {"cell": 20 + bench.index(u)}})
    on = next(u for u in p.units if u["cell"] == 24)
    off = next(u for u in p.units if u["bench"] is not None)
    g.act(0, {"type": "move", "uid": off["uid"], "to": {"cell": 24}})      # đổi chỗ
    assert off["cell"] == 24 and on["bench"] is not None


def test_carousel_order_lowest_hp_first():
    g = Game(DATA, seed=4, human=False)
    while not (g.phase == "plan" and g.kind() == "pvp" and g.stage == 2 and g.rnd == 3):
        g.resolve_round()
    for i, p in enumerate(g.players):
        p.hp = 50 + i
    g.resolve_round()                       # 2-3 -> 2-4 chọn tướng chung (bot chọn hết)
    picks = [l.split("] ")[1].split(" lấy ")[0] for l in g.log if " lấy " in l and l.startswith("[2-4]")]
    hp = {p.name: p.hp for p in g.players if p.alive}
    assert len(picks) == len(hp)
    assert [hp[n] for n in picks] == sorted(hp[n] for n in picks), "ít máu chọn trước"


def test_web_api(tmp_path, monkeypatch):
    from http.server import ThreadingHTTPServer
    from autochess.v5.web import App, make_handler
    app = App(seed=5)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path):
        return json.loads(urllib.request.urlopen(base + path, timeout=60).read())

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())
    try:
        cat = get("/api/catalog")
        assert len(cat["champions"]) == 59 and len(cat["items"]) == 44 and cat["augments"]
        s = get("/api/state")
        assert s["phase"] == "carousel" and s["carousel"]["my_turn"] in (True, False)
        while s["phase"] == "carousel":
            idx = next(o["index"] for o in s["carousel"]["options"] if not o["taken_by"])
            s = post("/api/act", {"type": "carousel", "index": idx})
        assert s["phase"] == "plan" and s["label"] == "1-2"
        s = post("/api/act", {"type": "ready"})
        assert s["label"] == "1-3" and s["fights"] and s["history"]
        html = urllib.request.urlopen(base + "/replay/0", timeout=60).read().decode("utf-8")
        assert "Xem lại trận" in html
        s = post("/api/act", {"type": "buy", "slot": 9})
        assert s["error"]
    finally:
        srv.shutdown()
