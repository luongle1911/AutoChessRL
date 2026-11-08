"""Giao diện web để người chơi thử (chạy trên máy mình, không cần cài thêm gói nào).

    python3 -m autochess.play.web --catalog autochess/data/catalog_v4.json -p 8 --seed 1 \\
        --bot tempo,bank,tank,fast9,roll7,hyper,bank
    # rồi mở http://localhost:8000

Bạn ngồi ghế 0. Ván chạy ở một luồng nền: bot đi các bước của chúng, tới lượt bạn thì luồng dừng chờ lệnh
từ trình duyệt (POST /api/act). Log ván ghi vào logs/ như giao diện terminal (phân tích / phát lại được).

API (JSON):
  GET  /api/catalog       dữ liệu tĩnh: tướng (chỉ số theo sao, tộc), tộc hệ (mốc, hiệu ứng, mô tả, thành viên),
                          tỷ lệ shop theo cấp, kinh nghiệm theo cấp, kho chung
  GET  /api/state         trạng thái hiện tại nhìn từ ghế 0 (+ hành động hợp lệ, kết quả giao tranh gần nhất)
  POST /api/act {a}       làm hành động a (0-43); trả về trạng thái khi tới lượt bạn lần tới (hoặc hết ván)
  POST /api/undo          lùi về trước lệnh gần nhất của bạn
  POST /api/new {seed}    ván mới
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from autochess.core import load_catalog, FormulaCombat
from autochess.core.catalog import NO_BREAKPOINT, STAT_NAMES
from .adapter import champ_stats, diff_events
from .replay import save_log
from .runner import GameRunner
from .seats import Command, Seat

HERE = os.path.dirname(os.path.abspath(__file__))
STAT_VN = {"hp": "máu", "ad": "STVL", "ap": "STP", "armor": "giáp", "mr": "kháng phép"}


class WebSeat(Seat):
    kind = "human"

    def __init__(self, game: "WebGame"):
        self.game = game

    def act(self, obs, mask, view):
        return self.game.wait_for_action(view, mask)


class WebGame:
    """Một ván chạy ở luồng nền; ghế 0 là người chơi qua trình duyệt."""

    def __init__(self, app: "WebApp", seed: int):
        self.app, self.seed = app, seed
        cat, combat = app.catalog, app.combat
        seats = [WebSeat(self)]
        self.labels = ["Bạn"]
        bots = app.args.bot.split(",")
        for i in range(1, app.args.players):
            name = bots[(i - 1) % len(bots)]
            seats.append(app.make_bot(name, i, seed))
            self.labels.append(_label(name))
        self.runner = GameRunner(cat, combat, seats, seed, debug=not app.args.no_debug,
                                 matchup_row=not app.args.no_matchup)
        self.cond = threading.Condition()
        self.q: queue.Queue = queue.Queue()
        self.version = 0
        self.need = None            # (view, mask) khi đang chờ người
        self.prev_view = None
        self.events: list[str] = []
        self.rounds: list[dict] = []
        self.log_path = None
        self.error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ luồng ván
    def wait_for_action(self, view, mask):
        with self.cond:
            if self.prev_view is not None:
                ev = diff_events(self.prev_view, view)
                if ev:
                    self.events = (self.events + [f"vòng {view['round']}: " + e for e in ev])[-60:]
            self.prev_view = view
            self.need = (view, mask)
            self.version += 1
            self.cond.notify_all()
        a = self.q.get()
        with self.cond:
            self.need = None
        return a

    def _loop(self):
        r = self.runner
        try:
            while not r.done:
                res = r.tick()
                if isinstance(res, Command):
                    if res.name == "quit":
                        return
                    if res.name == "undo":
                        r.undo()
                        self.prev_view = None
                    continue
                if res["combat"][0]:
                    self._record_combat(res)
            self.log_path = save_log(r.to_log())
        except Exception as e:      # hiện lỗi lên trình duyệt thay vì treo
            import traceback
            self.error = traceback.format_exc()
            print(self.error)
        finally:
            with self.cond:
                self.version += 1
                self.cond.notify_all()

    def _record_combat(self, info):
        s = self.runner.state
        rows = []
        for p in range(s.P):
            j = int(s.last_opponent[0, p])
            res = int(s.last_result[0, p, 0])
            if res == 0:
                continue
            rows.append(dict(seat=p, opponent=j, ghost=bool(info["ghost"][0, p]) if "ghost" in info else False,
                             win=res > 0, hp_lost=int(s.last_result[0, p, 1]), hp=int(s.hp[0, p])))
        self.rounds = (self.rounds + [dict(round=int(s.round[0]) - 1, results=rows)])[-5:]

    # ------------------------------------------------------------ cho HTTP
    def act(self, payload, timeout=120.0):
        with self.cond:
            v0 = self.version
            if self.need is None:
                return self.snapshot()
        self.q.put(payload)
        deadline = time.time() + timeout
        with self.cond:
            while self.version == v0 and time.time() < deadline:
                self.cond.wait(timeout=0.5)
        return self.snapshot()

    def snapshot(self) -> dict:
        r = self.runner
        with self.cond:
            need = self.need
        out = dict(labels=self.labels, events=self.events[-30:], rounds=self.rounds, seed=self.seed,
                   done=r.done, error=self.error, waiting=need is not None, log=self.log_path,
                   actions_taken=sum(r.human_flags))
        if need is not None:
            view, mask = need
            out["view"] = view
            out["legal"] = np.flatnonzero(mask).tolist()
        else:
            out["view"] = r.view(0, god=r.done)
            out["legal"] = []
        if r.done:
            out["placements"] = [int(x) for x in r.state.placement[0]]
        return out

    def stop(self):
        self.q.put(Command("quit"))


def _label(name: str) -> str:
    if name.startswith("agent:"):
        path = name[len("agent:"):]
        parts = path.replace("\\", "/").split("/")
        run = next((p for p in parts if p.startswith(("p2_", "p3_", "p4_", "bc_", "v2rules"))), parts[0])
        return f"agent {run[:14]}/{os.path.splitext(parts[-1])[0].replace('update_', '')}"
    return f"bot {name}"


class WebApp:
    def __init__(self, args):
        from .cli import CLI
        self.args = args
        self.cli = CLI(argparse.Namespace(catalog=args.catalog, temperature=None, no_color=True))
        self.catalog, self.combat = self.cli.catalog, self.cli.combat
        self.game = WebGame(self, args.seed)
        self.lock = threading.Lock()

    def make_bot(self, name, seat, seed):
        return self.cli.make_bot(name, seat, seed)

    def new_game(self, seed):
        with self.lock:
            self.game.stop()
            self.game = WebGame(self, seed)
        return self.game

    def catalog_json(self) -> dict:
        cat = self.catalog
        P = self.args.players
        members = {k: [int(c) for c in np.flatnonzero(cat.trait_matrix[:, k])] for k in range(cat.K)}
        traits = []
        for k, t in enumerate(sorted(cat.raw["traits"], key=lambda t: t["id"])):
            bp = [int(x) for x in cat.trait_breakpoints[k] if x < NO_BREAKPOINT]
            eff = []
            for lv in range(len(bp)):
                parts = []
                for stat, mode, vals in t.get("effects", []):
                    v = vals[lv]
                    parts.append(f"+{v}% {STAT_VN[stat]}" if mode == "pct" else f"+{v} {STAT_VN[stat]}")
                if not parts and "bonus" in t:
                    parts.append(f"+{round(100 * t['bonus'][lv])}% sức mạnh")
                eff.append(", ".join(parts))
            traits.append(dict(id=k, name=t["name"], kind=t.get("kind", "trait"), breakpoints=bp,
                               scope=t.get("scope", "members"), effects=eff, text=t.get("text", ""),
                               members=members[k]))
        champs = []
        for c in range(cat.N):
            raw = next(x for x in cat.raw["champions"] if x["id"] == c)
            champs.append(dict(id=c, name=cat.names[c], cost=int(cat.cost[c]), role=raw.get("role"),
                               traits=[int(k) for k in np.flatnonzero(cat.trait_matrix[c])],
                               stats=[champ_stats(cat, c, s) for s in (1, 2, 3)],
                               power=[round(float(x), 1) for x in cat.power[c]]))
        return dict(version=cat.version, combat=self.combat.version, stat_names=list(STAT_NAMES),
                    stat_vn=STAT_VN, champions=champs, traits=traits,
                    shop_odds=(np.asarray(cat.shop_odds) * 100).round().astype(int).tolist(),
                    xp_to_next=[int(x) for x in cat.xp_to_next], pool=[int(x) for x in cat.pool_size(P)],
                    streak_bonus=[int(x) for x in cat.streak_bonus], interest_step=cat.interest_step,
                    interest_max=cat.interest_max, base_income=cat.base_income,
                    max_actions=cat.max_actions_per_round, players=P)


def make_handler(app: WebApp):
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
                with open(os.path.join(HERE, "web_index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if self.path == "/api/catalog":
                return self._send(200, app.catalog_json())
            if self.path == "/api/state":
                g = app.game
                deadline = time.time() + 5
                while not g.snapshot()["waiting"] and not g.runner.done and g.error is None and time.time() < deadline:
                    time.sleep(0.05)
                return self._send(200, g.snapshot())
            self._send(404, {"error": "không có"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            g = app.game
            if self.path == "/api/act":
                a = int(body.get("a", -1))
                snap = g.snapshot()
                if not snap["waiting"] or a not in snap["legal"]:
                    snap["rejected"] = a
                    return self._send(200, snap)
                return self._send(200, g.act(a))
            if self.path == "/api/undo":
                return self._send(200, g.act(Command("undo")))
            if self.path == "/api/new":
                seed = int(body.get("seed", int(time.time()) % 100000))
                g = app.new_game(seed)
                deadline = time.time() + 30
                while not g.snapshot()["waiting"] and not g.runner.done and time.time() < deadline:
                    time.sleep(0.05)
                return self._send(200, g.snapshot())
            self._send(404, {"error": "không có"})
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("-p", "--players", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bot", default="tempo,bank,tank,fast9,roll7,hyper,bank",
                    help="bot cho các ghế 1.., vd tempo,bank hoặc agent:<checkpoint.pt>")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-matchup", action="store_true", help="ẩn xác suất thắng từng đối thủ")
    ap.add_argument("--no-debug", action="store_true", help="tắt kiểm tra bất biến sau mỗi bước (nhanh hơn)")
    args = ap.parse_args(argv)
    app = WebApp(args)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"mở trình duyệt: http://localhost:{args.port}   (luật {app.catalog.version}, giao tranh {app.combat.version}; "
          f"Ctrl+C để thoát)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
