"""Luật bản 4: 59 tướng, 5 chỉ số, tộc hệ có hiệu ứng chỉ số, giao tranh ttk-v1, và giao diện web."""
import json
import threading
import urllib.request

import numpy as np
import pytest

from autochess.core import load_catalog, FormulaCombat
from autochess.core.catalog import DATA_DIR, NO_BREAKPOINT
from autochess.core.combat import board_features, team_stats, unit_final_stats

CAT4 = load_catalog(f"{DATA_DIR}/catalog_v4.json")
NAME = {n: i for i, n in enumerate(CAT4.names)}


def _board(names, star=0):
    b = np.full(9, -1, np.int64)
    for i, n in enumerate(names):
        b[i] = NAME[n] * 3 + star
    return b


class _S:
    def __init__(self, boards, rnd=10):
        B = np.asarray(boards)
        self.units = np.concatenate([np.full(B.shape[:-1] + (9,), -1), B], -1)[None].astype(np.int16)
        self.round = np.array([rnd])


def test_catalog_v4_shape():
    c = CAT4
    assert c.version == "catalog-v4" and c.N == 59 and c.K == 24
    assert [len(t) for t in c.champs_by_tier] == [13, 13, 13, 12, 8]
    assert c.pool_size(8)[c.champs_by_tier[0][0]] == 30 and c.pool_size(8)[c.champs_by_tier[4][0]] == 9
    assert FormulaCombat(catalog=c).version == "ttk-v1"
    assert (c.role >= 0).all(), "mỗi tướng có đúng một hệ vai trò"
    n_members = c.trait_matrix.sum(0)
    uniques = [k for k, t in enumerate(sorted(c.raw["traits"], key=lambda t: t["id"])) if t["kind"] == "unique"]
    assert len(uniques) == 8 and (n_members[uniques] == 1).all()
    for k in range(c.K):    # mỗi mốc đều đạt được
        assert n_members[k] >= c.trait_breakpoints[k, c.trait_levels[k] - 1]
    assert (c.trait_breakpoints[c.trait_levels == 1, 1:] == NO_BREAKPOINT).all()


def test_star_and_cost_scaling():
    p = CAT4.power
    for t in range(4):      # ở mỗi mức sao, tướng đắt hơn mạnh hơn (trung bình theo bậc)
        a, b = p[CAT4.champs_by_tier[t]].mean(0), p[CAT4.champs_by_tier[t + 1]].mean(0)
        assert (b > a).all()
    t1, t2, t4, t5 = (p[CAT4.champs_by_tier[t]].mean(0) for t in (0, 1, 3, 4))
    assert 0.8 < t1[2] / t4[1] < 1.1, "3 sao tướng 1 tiền ~ 2 sao tướng 4 tiền"
    assert t4[1] < t2[2] < t5[1] * 1.05, "3 sao tướng 2 tiền giữa 2 sao tướng 4 và 5 tiền"


def test_trait_effects_members_and_team():
    # 2 Hộ vệ (Sobek, Chử Đồng Tử): mỗi Hộ vệ +25 giáp/kháng phép; Ai Cập 1 chưa kích hoạt
    b = _board(["Sobek", "Chử Đồng Tử", "Valkyrie"])
    u = unit_final_stats(CAT4, b[None])[0]
    base = CAT4.stats
    assert u[0, 3] == base[NAME["Sobek"], 3] + 25 and u[1, 4] == base[NAME["Chử Đồng Tử"], 4] + 25
    assert u[2, 3] == base[NAME["Valkyrie"], 3], "Xạ thủ không mang tộc Hộ vệ"
    # Ai Cập 2 (hiệu ứng cả đội): cả Valkyrie cũng +10 giáp
    b2 = _board(["Sobek", "Anubis", "Valkyrie"])
    u2 = unit_final_stats(CAT4, b2[None])[0]
    assert u2[2, 3] == base[NAME["Valkyrie"], 3] + 10
    # 2 sao: máu và sát thương x1,8, giáp giữ nguyên
    u3 = unit_final_stats(CAT4, _board(["Valkyrie"], star=1)[None])[0]
    assert np.isclose(u3[0, 0], base[NAME["Valkyrie"], 0] * 1.8) and u3[0, 3] == base[NAME["Valkyrie"], 3]
    # trùng tướng chỉ tính một lần cho tộc
    f = board_features(CAT4, _S([_board(["Sobek", "Sobek"])]))
    assert f["trait_count"][0, 0, CAT4.trait_names.index("Hộ vệ")] == 1


def test_ttk_combat_counters_and_symmetry():
    phys = _board(["Hermes", "Tengu", "Valkyrie", "Karna", "Camazotz", "Garuda"])
    tank = _board(["Sobek", "Chử Đồng Tử", "Xolotl", "Heimdall", "Anubis", "Dagda"])
    mage = _board(["Brigid", "Kitsune", "Bastet", "Freyja", "Thoth", "Hằng Nga"])
    empty = np.full(9, -1)
    combat = FormulaCombat(catalog=CAT4)
    win, dmg = combat.evaluate(CAT4, _S([phys, tank, mage, empty]))
    w = win[0]
    assert np.allclose(w + w.T, 1.0)
    assert (w[:3, 3] > 0.99).all() and np.isclose(w[3, 3], 0.5)
    ts = team_stats(CAT4, np.stack([phys, tank, mage]))
    assert ts["team_armor"][1] > 100 and ts["team_ad"][0] > 3 * ts["team_ad"][2]
    # giáp khắc sát thương vật lý: đội đỡ đòn thắng đội vật lý, đội phép không bị đội đỡ đòn khắc mạnh như thế
    assert w[1, 0] > 0.6 and w[2, 1] > w[0, 1]
    assert (dmg[0] >= 0).all()


def test_bots_play_full_games_on_v4():
    from autochess import tournament as T
    combat = FormulaCombat(catalog=CAT4)
    r = T.play(CAT4, combat, ("tempo", "hyper", "roll7", "fast9", "bank", "tank", "bank", "fast9"), 16, seed=4, B=16)
    pl = r["placements"]
    assert len(pl) >= 16 and all(sorted(row) == list(range(1, 9)) for row in pl.tolist())


def test_old_rules_unchanged():
    c3 = load_catalog(f"{DATA_DIR}/catalog_v3.json")
    assert FormulaCombat(catalog=c3).version == "formula-v1" and c3.stats is None
    assert c3.trait_breakpoints.shape == (c3.K, 3)
    f = board_features(c3, _S([np.full(9, -1)]))
    assert "team_hp" not in f


def test_web_api_play_a_few_steps(tmp_path, monkeypatch):
    import argparse
    from http.server import ThreadingHTTPServer
    from autochess.play.web import WebApp, make_handler
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(catalog=f"{DATA_DIR}/catalog_v4.json", players=4, seed=3, bot="tempo,bank,tank",
                              port=0, host="127.0.0.1", no_matchup=False, no_debug=False)
    app = WebApp(args)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path):
        return json.loads(urllib.request.urlopen(base + path, timeout=30).read())

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())
    try:
        cat = get("/api/catalog")
        assert len(cat["champions"]) == 59 and cat["traits"][0]["effects"][0].startswith("+25")
        s = get("/api/state")
        assert s["waiting"] and s["view"]["round"] == 1 and len(s["labels"]) == 4
        buy = [a for a in s["legal"] if a < 5][0]
        s2 = post("/api/act", {"a": buy})
        assert any(u for u in s2["view"]["bench"]) and s2["actions_taken"] == 1
        s3 = post("/api/undo", {})
        assert s3["actions_taken"] == 0
        r0 = s3["view"]["round"]
        s4 = post("/api/act", {"a": 43})          # kết thúc lượt: bot đi hết vòng, giao tranh
        assert s4["view"]["round"] == r0 + 1 and s4["rounds"]
        s5 = post("/api/act", {"a": 99})          # hành động không hợp lệ bị từ chối
        assert s5["rejected"] == 99
    finally:
        app.game.stop()
        srv.shutdown()


def test_v4_1_same_structure_rebalanced_traits():
    from autochess.data.balance import round_robin, STAGES
    c41 = load_catalog(f"{DATA_DIR}/catalog_v4_1.json")
    assert c41.version == "catalog-v4.1" and c41.names == CAT4.names and c41.trait_names == CAT4.trait_names
    assert (c41.trait_matrix == CAT4.trait_matrix).all() and np.allclose(c41.stats, CAT4.stats)
    assert (c41.trait_breakpoints == CAT4.trait_breakpoints).all()
    k = CAT4.trait_names.index("Đấu sĩ")
    assert c41.trait_effects[k, 3, 0, 0] < CAT4.trait_effects[k, 3, 0, 0], "Đấu sĩ 8 bị giảm"
    wins = {}
    for st in STAGES.values():
        for r in round_robin(c41, st):
            wins.setdefault(r["trait"], []).append(r["win"])
    avg = {t: float(np.mean(v)) for t, v in wins.items()}
    assert all(0.4 <= a <= 0.6 for a in avg.values()), avg


def test_v4_net_loads_on_v4_1():
    torch = pytest.importorskip("torch")
    from autochess.rl.nets import build_net
    c41 = load_catalog(f"{DATA_DIR}/catalog_v4_1.json")
    a, b = build_net(CAT4, {"kind": "entity2"}, 8), build_net(c41, {"kind": "entity2"}, 8)
    b.load_state_dict(a.state_dict())          # cùng kích thước: thẻ tộc chỉ đổi giá trị đọc từ file luật
    assert not torch.allclose(a.trait_stats, b.trait_stats)
