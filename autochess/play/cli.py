"""Giao diện dòng lệnh để chơi thử và soi lỗi luật.

    python -m autochess.play.cli play  [-p 2] [--seed 1] [--bot econ]
    python -m autochess.play.cli watch [-p 4] [--seed 1] [--bot econ,random]
    python -m autochess.play.cli replay logs/game_xxx.json
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from autochess.bots import BOTS, ARRAY_BOTS, ScriptedBot
from autochess.core import load_catalog, FormulaCombat, explain, state_hash, mask_parts
from autochess.core.actions import BUY0, SELL0, MOVE0, ROLL, BUY_XP, END_TURN
from autochess.core.state import FIELD_NAMES
from .adapter import STARS, parse_command, action_to_command, diff_events
from .replay import save_log, load_log, replay_frames, version_warnings
from .runner import GameRunner
from .seats import Command, HumanSeat, ScriptedSeat, ArrayBotSeat


# ---------------------------------------------------------------- màu
class Style:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else str(s)

    def bold(self, s): return self._w("1", s)
    def dim(self, s): return self._w("2", s)
    def red(self, s): return self._w("31", s)
    def green(self, s): return self._w("32", s)
    def yellow(self, s): return self._w("33", s)
    def cyan(self, s): return self._w("36", s)
    def mag(self, s): return self._w("35", s)


TIER_COLOR = {1: "dim", 2: "green", 3: "cyan", 4: "mag", 5: "yellow"}


# ---------------------------------------------------------------- hiển thị
class Renderer:
    def __init__(self, style: Style, human_seat: int | None = None):
        self.S = style
        self.human_seat = human_seat

    def who(self, p: int) -> str:
        return f"Người {p}" + (" (bạn)" if p == self.human_seat else "")

    def unit(self, u) -> str:
        if not u:
            return "—"
        name = getattr(self.S, TIER_COLOR[u["cost"]])(u["name"])
        return f"{name}{self.S.yellow(STARS[u['star']])}"

    def units_line(self, units, with_slot=True) -> str:
        items = [f"[{u['slot']}] {self.unit(u)}" if with_slot else self.unit(u) for u in units if u]
        return "  ".join(items) if items else self.S.dim("(trống)")

    def traits_line(self, traits) -> str:
        if not traits:
            return self.S.dim("(chưa có)")
        out = []
        for t in traits:
            s = f"{t['name']} {t['count']}/{t['next']}"
            out.append(self.S.green(s + " ✓") if t["level"] > 0 else s)
        return "   ".join(out)

    @staticmethod
    def streak(n: int) -> str:
        if n > 0:
            return f"Chuỗi thắng {n}"
        if n < 0:
            return f"Chuỗi thua {-n}"
        return "Chưa có chuỗi"

    def last_result(self, lr) -> str:
        if lr["opponent"] < 0 or lr["result"] == 0:
            return ""
        if lr["result"] > 0:
            return self.S.green(f"vòng trước thắng Người {lr['opponent']}")
        return self.S.red(f"vòng trước thua Người {lr['opponent']} (-{lr['hp_lost']} máu)")

    def view(self, v: dict) -> str:
        S, me = self.S, v["me"]
        L = []
        phase = "Đã kết thúc" if v["done"] else ("Đã xong lượt, chờ người khác" if me["ended"] else "Pha chuẩn bị")
        god = S.mag(" · GOD") if v["god"] else ""
        L.append(S.bold(f"=== Vòng {v['round']} · {phase} · {self.who(v['seat'])} · ván {v['num_players']} người ===") + god)
        xp = f"XP {me['xp']}/{me['xp_next']}" if me["xp_next"] else "tối đa"
        L.append(f"Máu {S.bold(me['hp'])} | Vàng {S.yellow(me['gold'])} (+{me['interest_next']} lãi) | "
                 f"Cấp {me['level']} ({xp}) | {self.streak(me['streak'])} | còn {me['actions_left']} hành động")
        shop = []
        for i, s in enumerate(v["shop"]):
            if s is None:
                shop.append(f"[{i}] —")
            else:
                name = getattr(S, TIER_COLOR[s["cost"]])(s["name"])
                own = S.green(f" ·có {s['owned']}") if s["owned"] else ""
                shop.append(f"[{i}] {name} ({s['cost']}v){own}")
        L.append("Shop:    " + "  ".join(shop))
        L.append(f"Sân {me['board_count']}/{me['level']}: " + self.units_line(v["board"]))
        L.append("Chờ:     " + self.units_line(v["bench"]))
        L.append("Tộc:     " + self.traits_line(me["traits"]))
        lr = self.last_result(me["last_result"])
        if lr:
            L.append("Kết quả: " + lr)
        L.append("Đối thủ:" if v["done"] else "Có thể gặp:")
        for o in v["opponents"]:
            if not o["alive"]:
                L.append(S.dim(f"  Người {o['seat']} · đã bị loại, hạng {o['placement']}"))
                continue
            parts = [f"  {self.who(o['seat'])}", f"Máu {o['hp']}", f"Cấp {o['level']}", f"Vàng {o['gold']}",
                     self.streak(o["streak"])]
            if not v["done"]:
                if o["meet_prob"] == 0:
                    parts.append(S.dim("không gặp vòng này"))
                else:
                    parts.append(f"gặp {o['meet_prob']:.0%}")
                if "win_prob" in o and o["meet_prob"] > 0:
                    wp = o["win_prob"]
                    col = S.green if wp >= 0.5 else S.red
                    parts.append(col(f"thắng ~{wp:.0%}"))
            L.append(" · ".join(parts))
            L.append(f"      Sân: {self.units_line(o['board'], with_slot=False)}")
            if any(o.get("bench") or []):
                L.append(f"      Chờ: {self.units_line(o['bench'], with_slot=False)}")
            if o["traits"]:
                L.append(f"      Tộc: {self.traits_line(o['traits'])}")
            if v["god"]:
                sh = "  ".join(f"{s['name']}({s['cost']}v)" if s else "—" for s in o["shop"])
                L.append(S.mag(f"      [god] Shop: {sh}"))
        if v["expected_win"] is not None:
            L.append(f"Kỳ vọng thắng vòng này: ~{v['expected_win']:.0%}")
        if v["god"]:
            pool = "  ".join(f"{p['name']} {p['left']}/{p['total']}" for p in v["pool"])
            L.append(S.mag("[god] Kho: ") + pool)
            if v.get("matchup"):
                P = v["num_players"]
                L.append(S.mag("[god] Ma trận thắng (hàng thắng cột) / xác suất gặp:"))
                for i in range(P):
                    row = "  ".join("  —  " if i == j else f"{v['matchup'][i][j]:.2f}/{v['meet_prob'][i][j]:.2f}"
                                    for j in range(P))
                    L.append(S.mag(f"   {i}: ") + row)
        return "\n".join(L)

    def events(self, events, catalog, viewer: int | None, god: bool) -> list[str]:
        S, out = self.S, []
        names = catalog.names
        private = {"buy", "sell", "move", "roll", "buy_xp", "merge", "income", "end_turn"}
        for e in events or []:
            t = e["type"]
            if t in private and not god and e.get("seat") != viewer:
                continue
            p = e.get("seat")
            if t == "buy":
                out.append(f"{self.who(p)} mua {names[e['champ']]} vào ô {e['slot']}")
            elif t == "sell":
                out.append(f"{self.who(p)} bán {names[e['champ']]} {STARS[e['star']]} (+{e['refund']} vàng)")
            elif t == "merge":
                out.append(S.yellow(f"{self.who(p)}: 3 {names[e['champ']]} {STARS[e['star'] - 1]} "
                                    f"ghép thành {names[e['champ']]} {STARS[e['star']]} (ô {e['slot']})"))
            elif t == "level_up":
                out.append(S.cyan(f"{self.who(p)} lên cấp {e['level']}"))
            elif t == "income":
                out.append(f"{self.who(p)} +{e['total']} vàng (cơ bản {e['base']}, lãi {e['interest']}, chuỗi {e['streak']})")
            elif t == "end_turn" and e["auto"]:
                out.append(S.dim(f"{self.who(p)} chạm giới hạn hành động, tự kết thúc lượt"))
            elif t == "combat":
                a, b = e["a"], e["opp"]
                tgt = f"bản sao của Người {b}" if e["ghost"] else self.who(b)
                head = f"{self.who(a)} vs {tgt} · trước trận {self.who(a)} thắng ~{e['p_a_wins']:.0%} → "
                if e["a_won"]:
                    res = f"{self.who(a)} thắng" + ("" if e["ghost"] else f", {self.who(b)} mất {e['hp_lost']} máu")
                else:
                    res = f"{self.who(a)} thua, mất {e['hp_lost']} máu"
                out.append(S.bold("⚔ ") + head + res)
            elif t == "eliminated":
                out.append(S.red(f"✖ {self.who(p)} bị loại, hạng {e['placement']}"))
            elif t == "game_over":
                why = "chạm vòng tối đa" if e["reason"] == "max_rounds" else "còn một người"
                out.append(S.bold(f"■ Hết ván ({why}). Thứ hạng: ") +
                           ", ".join(f"{self.who(i)}: {r}" for i, r in enumerate(e["placement"])))
            elif t == "new_round":
                out.append(S.bold(f"--- Vòng {e['round']} ---"))
        return out


HELP = """Lệnh hành động (mỗi lệnh = đúng một số hiệu):
  b i   mua ô shop i (0-4)            s j   bán quân ở ô j (0-8 chờ, 9-17 sân)
  m j   chuyển quân ở ô j lên/xuống   r     roll (2 vàng)
  x     mua 4 XP (4 vàng)             e     kết thúc lượt
  a n   gửi thẳng số hiệu n (gỡ lỗi)
Lệnh gỡ lỗi (không vào log):
  ?          liệt kê hành động hợp lệ      help   bảng này
  undo       lùi một hành động của bạn     god    bật/tắt nhìn toàn bộ
  raw        in mảng trạng thái            check  chạy bất biến + băm trạng thái
  save [f]   ghi log ván                   quit   thoát"""

DEBUG_WORDS = {"undo", "god", "raw", "check", "save", "quit", "q", "exit"}


class CLI:
    def __init__(self, args):
        self.args = args
        self.catalog = load_catalog(args.catalog)
        self.combat = FormulaCombat(catalog=self.catalog,
                                    temperature=getattr(args, "temperature", None))
        color = sys.stdout.isatty() and not args.no_color and not os.environ.get("NO_COLOR")
        self.S = Style(color)
        self.god = False

    # ------------------------------------------------------------ chung
    def make_bot(self, name: str, seat: int, seed: int):
        if name.startswith("agent:"):
            # agent đã huấn luyện: --bot agent:runs/<tên>/checkpoints/latest.pt (cần PyTorch)
            from autochess.rl.agent import AgentPlayer
            from autochess.rl.policy import Policy, load_checkpoint
            net, cfg = load_checkpoint(name[len("agent:"):], self.catalog)
            player = AgentPlayer(Policy(net), self.catalog, self.combat,
                                 matchup_row=cfg["env"]["matchup_row"], seed=seed * 1000 + seat)
            return ArrayBotSeat(player, "agent")
        if name in ARRAY_BOTS:
            return ArrayBotSeat(ScriptedBot(self.catalog, ARRAY_BOTS[name]), name)
        if name not in BOTS:
            raise SystemExit(f"không có bot '{name}'. Có: {', '.join([*BOTS, *ARRAY_BOTS])}")
        return ScriptedSeat(BOTS[name], name, seed=seed * 1000 + seat)

    def legal_list(self, view, mask) -> str:
        out = []
        for a in np.flatnonzero(mask):
            cmd = action_to_command(a)
            if a < SELL0:
                s = view["shop"][a - BUY0]
                desc = f"mua {s['name']} ({s['cost']}v)"
            elif a < MOVE0:
                j = a - SELL0
                u = (view["bench"] + view["board"])[j]
                desc = f"bán {u['name']} {STARS[u['star']]}"
            elif a < ROLL:
                j = a - MOVE0
                u = (view["bench"] + view["board"])[j]
                desc = f"{'lên sân' if j < 9 else 'xuống hàng chờ'}: {u['name']} {STARS[u['star']]}"
            else:
                desc = {ROLL: "roll", BUY_XP: "mua XP", END_TURN: "kết thúc lượt"}[a]
            out.append(f"  {cmd:<5} {desc}")
        return "\n".join(out)

    def print_raw(self, runner):
        np.set_printoptions(linewidth=140)
        for name in FIELD_NAMES:
            print(f"{name}: {getattr(runner.state, name)[0].tolist()}")

    def do_check(self, runner):
        errs = runner.check()
        print(self.S.green("bất biến: ổn") if not errs else self.S.red("VI PHẠM:\n  " + "\n  ".join(errs)))
        print(f"băm trạng thái: {state_hash(runner.state)[:16]}  ·  bước {runner.t}")

    def do_save(self, runner, arg="", auto=False):
        if auto and getattr(self, "_saved_t", None) == runner.t:
            return
        path = save_log(runner.to_log(), arg or None)
        self._saved_t = runner.t
        print(f"đã ghi log: {path}")

    # ------------------------------------------------------------ play
    def human_input(self, view, mask):
        r = self.runner
        if self.god:
            view = r.view(view["seat"], god=True)
        print()
        print(self.R.view(view))
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                return Command("quit")
            if not text:
                continue
            if text == "?":
                print(self.legal_list(view, mask))
                continue
            if text in ("help", "h"):
                print(HELP)
                continue
            a = parse_command(text, view)
            if a is not None:
                if mask[a]:
                    return a
                why = explain(self.catalog, r.state, 0, view["seat"], a)
                print(self.S.red(f"không được: {why}"))
                continue
            word, _, arg = text.partition(" ")
            if word in DEBUG_WORDS:
                return Command(word, arg.strip())
            if word in ("b", "s", "m", "a"):
                print(self.S.red("sai cú pháp/chỉ số: b 0-4, s 0-17, m 0-17, a 0-43"))
            else:
                print(self.S.red("lệnh không hiểu; gõ help"))

    def handle_command(self, cmd: Command, runner) -> bool:
        """Trả về False nếu thoát."""
        n = cmd.name
        if n in ("quit", "q", "exit"):
            return False
        if n == "undo":
            k = runner.undo()
            print(f"đã lùi {k} bước" if k else "không còn gì để lùi")
        elif n == "god":
            self.god = not self.god
            print(f"god: {'bật' if self.god else 'tắt'}")
        elif n == "raw":
            self.print_raw(runner)
        elif n == "check":
            self.do_check(runner)
        elif n == "save":
            self.do_save(runner, cmd.arg)
        return True

    def play(self):
        a = self.args
        bots = a.bot.split(",")
        seats = [HumanSeat(self.human_input)]
        for i in range(1, a.players):
            seats.append(self.make_bot(bots[(i - 1) % len(bots)], i, a.seed))
        self.R = Renderer(self.S, human_seat=0)
        self.runner = r = GameRunner(self.catalog, self.combat, seats, a.seed, debug=not a.no_debug,
                                     matchup_row=not a.no_matchup)
        print(self.S.bold(f"Ván {a.players} người, seed {a.seed}, giao tranh {self.combat.version}. Gõ help để xem lệnh."))
        while not r.done:
            before = r.view(0)
            res = r.tick()
            if isinstance(res, Command):
                if not self.handle_command(res, r):
                    self.do_save(r, auto=True)
                    return
                continue
            lines = self.R.events(res["events"], self.catalog, viewer=0, god=self.god)
            if res["combat"][0] and not lines:
                lines = diff_events(before, r.view(0))
            for line in lines:
                print("  " + line)
        print()
        print(self.R.view(r.view(0, god=True)))
        self.do_save(r, auto=True)

    # ------------------------------------------------------------ watch
    def summary_table(self, runner) -> str:
        s = runner.state
        lines = [self.S.bold(f"=== Vòng {int(s.round[0])} · bước {runner.t} ===")]
        for p in range(runner.P):
            v = runner.view(p)
            me = v["me"]
            status = (f"hạng {me['placement']}" if me["placement"] else
                      "xong lượt" if me["ended"] else "đang đi")
            lines.append(f"  [{p}] {runner.seats[p].kind:<10} Máu {me['hp']:>3} Vàng {me['gold']:>3} "
                         f"Cấp {me['level']} {self.R.streak(me['streak']):<14} {status:<10} "
                         f"Sân: {self.R.units_line(v['board'], with_slot=False)}")
        return "\n".join(lines)

    def watch(self):
        a = self.args
        bots = a.bot.split(",")
        seats = [self.make_bot(bots[i % len(bots)], i, a.seed) for i in range(a.players)]
        self.R = Renderer(self.S)
        r = GameRunner(self.catalog, self.combat, seats, a.seed, debug=not a.no_debug,
                       matchup_row=not a.no_matchup)
        self.god, focus = True, 0
        print(self.S.bold("watch: Enter = 1 bước · r = hết vòng · a = hết ván · v N = xem ghế N · "
                          "g = god · check · save · q"))
        print(self.summary_table(r))
        while True:
            try:
                text = input("watch> ").strip()
            except EOFError:
                break
            word, _, arg = text.partition(" ")
            infos = []
            if word == "":
                if not r.done:
                    infos.append(r.tick())
            elif word == "r":
                rnd = int(r.state.round[0])
                while not r.done and int(r.state.round[0]) == rnd:
                    infos.append(r.tick())
            elif word == "a":
                while not r.done:
                    infos.append(r.tick())
            elif word == "v" and arg.isdigit() and int(arg) < r.P:
                focus = int(arg)
                print(self.R.view(r.view(focus, god=self.god)))
                continue
            elif word == "g":
                self.god = not self.god
                print(f"god: {'bật' if self.god else 'tắt'}")
                continue
            elif word == "check":
                self.do_check(r)
                continue
            elif word == "save":
                self.do_save(r, arg)
                continue
            elif word == "raw":
                self.print_raw(r)
                continue
            elif word in ("q", "quit", "exit"):
                break
            else:
                print("lệnh không hiểu")
                continue
            for info in infos:
                for line in self.R.events(info["events"], self.catalog, viewer=focus, god=self.god):
                    print("  " + line)
            print(self.summary_table(r))
            if r.done:
                print(self.S.bold("Ván đã kết thúc."))
        if r.t:
            self.do_save(r, auto=True)

    # ------------------------------------------------------------ replay
    def replay(self):
        log = load_log(self.args.file)
        for w in version_warnings(log, self.catalog, self.combat):
            print(self.S.red("cảnh báo: " + w))
        runner, frames = replay_frames(log, self.catalog, self.combat, debug=not self.args.no_debug)
        self.R = Renderer(self.S, human_seat=0 if log["seats"] and log["seats"][0] == "human" else None)
        # mốc đầu mỗi vòng: bước t mà sau đó state ở vòng mới (hoặc t = 0)
        round_start = {}
        for t, (st, _) in enumerate(frames):
            round_start.setdefault(int(st.round[0]), t)
        rounds = sorted(round_start)
        t, focus, self.god = 0, 0, True
        print(self.S.bold(f"replay {self.args.file}: {len(frames) - 1} bước, {len(rounds)} vòng, "
                          f"{log['num_players']} người ({', '.join(log['seats'])})"))
        print("n/p = vòng sau/trước · j N = tới vòng N · s = 1 bước · v N = xem ghế N · g = god · check · q")

        def show():
            runner.state = frames[t][0].copy()
            print(self.R.view(runner.view(focus, god=self.god)))
            print(self.S.dim(f"bước {t}/{len(frames) - 1}"))

        show()
        while True:
            try:
                text = input("replay> ").strip()
            except EOFError:
                break
            word, _, arg = text.partition(" ")
            cur_round = int(frames[t][0].round[0])
            if word == "n":
                nxt = [r for r in rounds if r > cur_round]
                new_t = round_start[nxt[0]] if nxt else len(frames) - 1
                for k in range(t + 1, new_t + 1):
                    for line in self.R.events(frames[k][1]["events"], self.catalog, focus, self.god):
                        print("  " + line)
                t = new_t
            elif word == "p":
                prv = [r for r in rounds if r < cur_round]
                t = round_start[prv[-1]] if prv else 0
            elif word == "j" and arg.isdigit() and int(arg) in round_start:
                t = round_start[int(arg)]
            elif word == "s":
                if t < len(frames) - 1:
                    t += 1
                    for line in self.R.events(frames[t][1]["events"], self.catalog, focus, self.god):
                        print("  " + line)
                    acts = log["actions"][t - 1]
                    print(self.S.dim("  hành động: " + ", ".join(f"{i}:{action_to_command(x)}" for i, x in enumerate(acts))))
            elif word == "v" and arg.isdigit() and int(arg) < log["num_players"]:
                focus = int(arg)
            elif word == "g":
                self.god = not self.god
            elif word == "check":
                runner.state = frames[t][0].copy()
                self.do_check(runner)
                continue
            elif word in ("q", "quit", "exit"):
                break
            else:
                print("lệnh không hiểu")
                continue
            show()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autochess", description="Auto Chess RL — chơi thử lõi vĩ mô")
    ap.add_argument("--catalog", default=None, help="đường dẫn file dữ liệu (mặc định catalog_v1.json)")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--no-debug", action="store_true", help="tắt kiểm tra bất biến sau mỗi bước")
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("play", "watch"):
        sp = sub.add_parser(name)
        sp.add_argument("-p", "--players", type=int, default=2 if name == "play" else 4)
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--bot", default="econ", help="tên bot, có thể nhiều: econ,random,tempo,hyper,roll7,fast9, hoặc agent:<checkpoint.pt>")
        sp.add_argument("--no-matchup", action="store_true", help="tắt cờ obs.matchup_row (ẩn xác suất thắng)")
        sp.add_argument("--temperature", type=float, default=None)
    sp = sub.add_parser("replay")
    sp.add_argument("file")
    args = ap.parse_args(argv)
    cli = CLI(args)
    getattr(cli, args.mode)()


if __name__ == "__main__":
    main()
