"""Xem lại một trận của luật bản 5 (kỹ năng, đồ, tộc hệ, hiệp phụ) trong trình duyệt.

    python -m battlesim.viewer2 --a "Zeus*2@3[Mũ Phù Thủy+Ngọc Lam], Athena*2@24[Áo Gai]" \\
                                --b "Thor*2@24[Huyết Kiếm], Lugh*2@2[Cuồng Cung]"
    python -m battlesim.viewer2 --random 7 --seed 3
    python -m battlesim.viewer2 --list           # danh sách tướng, kỹ năng, đồ

Cú pháp đội: 'Tên*sao@ô[Đồ1+Đồ2+Đồ3]', sao mặc định 1; ô nửa sân 0-27 (hàng 0 = sau, hàng 3 = đầu,
ô = hàng x 7 + cột). File HTML ghi vào runs/replays/ rồi tự mở.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np

from .effects import ST_HP, ST_MANA_MAX, ST_RANGE, ST_AD, ST_AP, ST_AS, ST_ARMOR, ST_MR, describe
from .sim2 import (F_HP, F_MANA, F_SHIELD, F_STUN, F_DEALT, F_STEALTH, I_ALIVE, I_CELL, I_TARGET, I_BLOCKED,
                   I_CHAMP, I_SIDE, I_ORD, I_STAR, U, EV_DASH, EV_MOVE, RES_SIDE0, RES_SIDE1)
from .team import CombatData
from .viewer import _TEMPLATE

DEFAULT_DIR = Path("runs") / "replays"


def record(data: CombatData, team_a, team_b, seed=0, stochastic=False, extra_a=None, extra_b=None):
    batch = data.build_batch([team_a], [team_b], seeds=[seed], extras_a=[extra_a], extras_b=[extra_b])
    sim = data.simulator(stochastic=stochastic, log=True, max_events=32768)
    st = sim.init_state(batch)

    def snap(s):
        F, I, S = st.F[0], st.I[0], st.S[0]
        return {"s": s, "cell": I[:, I_CELL].tolist(), "hp": np.round(F[:, F_HP], 1).tolist(),
                "alive": I[:, I_ALIVE].tolist(), "target": I[:, I_TARGET].tolist(),
                "blocked": I[:, I_BLOCKED].tolist(), "dealt": np.round(F[:, F_DEALT], 1).tolist(),
                "hpmax": np.round(S[:, ST_HP], 1).tolist(), "mana": np.round(F[:, F_MANA], 1).tolist(),
                "manamax": np.round(S[:, ST_MANA_MAX], 1).tolist(), "shield": np.round(F[:, F_SHIELD], 1).tolist(),
                "stun": (F[:, F_STUN] > 0).astype(int).tolist(), "stealth": (F[:, F_STEALTH] > 0).astype(int).tolist(),
                "stats": np.round(S[:, [ST_AD, ST_AP, ST_AS, ST_ARMOR, ST_MR]], 2).tolist()}

    frames = [snap(0)]
    start_cells = list(frames[0]["cell"])
    result, end = 2, sim.max_steps
    for s in range(1, sim.max_steps + 1):
        r = int(sim.step(st, s)[0])
        frames.append(snap(s))
        if r >= 0:
            result, end = r, s
            break
    # trạng thái trước hiệu ứng đầu trận (sát thủ lướt ở bước 0): ô ban đầu theo đội hình
    init_cells = list(np.where(batch.side[0] == 0, batch.half_cell[0],
                               __import__("battlesim").hexgrid.ROT[np.clip(batch.half_cell[0], 0, 27)]))
    I0 = st.I[0]
    teams = (team_a, team_b)
    units = []
    for u in range(U):
        c = int(batch.champ[0, u])
        if c < 0:
            continue
        side = int(batch.side[0, u])
        unit = teams[side][u - 10 * side]
        ch = data.champs[c]
        items = [data.items[data.item_id(i)]["name"] for i in unit.items]
        units.append({"slot": u, "side": side, "ord": int(I0[u, I_ORD]), "name": ch["name"], "star": int(unit.star),
                      "range": int(round(frames[0]["stats"][u][0] * 0 + st.S[0, u, ST_RANGE])),
                      "hp_max": frames[0]["hpmax"][u], "items": items, "cost": ch["cost"],
                      "role": ch["role"], "traits": [data.trait_names[k] for k in ch["traits"]],
                      "skill": f"{ch['skill']['name']}: {ch['skill']['text']}",
                      "skill_detail": "; ".join(describe(b, int(unit.star) - 1) for b in ch["skill"]["effects"])})
    events = []
    for e in st.ev[0, :min(int(st.evc[0]), st.ev.shape[1])].tolist():
        if int(e[1]) == EV_DASH:          # lướt: vẽ như một bước đi tức thì
            events.append([e[0], EV_MOVE, e[2], -1, 0.0, e[5]])
            events.append([e[0], 9, e[2], e[3], 0.0, e[5]])
        else:
            events.append(e)
    # khung 0 vẽ từ vị trí xếp quân ban đầu (trước khi sát thủ lướt)
    frames[0]["cell"] = [int(init_cells[u]) if batch.champ[0, u] >= 0 else -1 for u in range(U)]
    for e in events:
        if e[0] == 0 and e[1] == EV_MOVE:
            e[0] = 0.0
    lv_a, lv_b = data.trait_levels(team_a, extra_a), data.trait_levels(team_b, extra_b)
    traits = [[f"{data.trait_names[k]} {data.bp[k][lv[k]]}" for k in range(data.K) if lv[k] >= 0]
              for lv in (lv_a, lv_b)]
    return {"meta": {"seed": int(seed), "dt": sim.g["dt"], "move_steps": int(round(sim.g["move_time"] / sim.g["dt"])),
                     "max_steps": sim.max_steps, "result": result, "end_step": end, "stochastic": stochastic,
                     "overtime": sim.g["overtime_start"], "version": data.version, "traits": traits, "level": 2},
            "units": units, "frames": frames, "events": events, "warnings": []}


_PATCHES = [
    # tiêu đề
    ("<title>Xem lại trận</title>", "<title>Xem lại trận (bản 5)</title>"),
    # bảng: thêm cột năng lượng
    ("<th>Tướng</th><th>Máu</th><th>Gây ra</th><th>Mục tiêu</th>",
     "<th>Tướng</th><th>Máu</th><th>Năng lượng</th><th>Gây ra</th><th>Mục tiêu</th>"),
    ("""      <td>${Math.round(fr.dealt[u.slot])}</td><td>${fr.alive[u.slot] ? tn : 'đã chết'}</td></tr>`;""",
     """      <td>${fr.manamax[u.slot] > 0 ? Math.round(fr.mana[u.slot]) + '/' + Math.round(fr.manamax[u.slot]) : '—'}</td>
      <td>${Math.round(fr.dealt[u.slot])}</td><td>${fr.alive[u.slot] ? tn : 'đã chết'}</td></tr>`;"""),
    ("""const hp = Math.max(0, fr.hp[u.slot]), pct = (100 * hp / u.hp_max).toFixed(0);""",
     """const hp = Math.max(0, fr.hp[u.slot]), pct = (100 * hp / fr.hpmax[u.slot]).toFixed(0);"""),
    ("""${u.name} <span class="tag">${u.star}★</span></td>""",
     """${u.name} <span class="tag">${u.star}★</span>${u.items.length ? ' <span class="tag">' + u.items.length + ' đồ</span>' : ''}${fr.shield[u.slot] > 0 ? ' <span class="tag">khiên ' + Math.round(fr.shield[u.slot]) + '</span>' : ''}${fr.stun[u.slot] ? ' <span class="tag">choáng</span>' : ''}</td>"""),
    # thanh năng lượng dưới thanh máu, khiên trên thanh máu
    ("""  const blocked = el('text', {'text-anchor': 'middle', y: R * 0.95, 'font-size': 8, fill: css('--warn')}, g);""",
     """  const blocked = el('text', {'text-anchor': 'middle', y: R * 0.95, 'font-size': 8, fill: css('--warn')}, g);
  const mbg = el('rect', {x: -R * 0.6, y: -R * 0.95 + 5, width: R * 1.2, height: 3, rx: 1.5, fill: css('--line')}, g);
  const mana = el('rect', {x: -R * 0.6, y: -R * 0.95 + 5, width: 0, height: 3, rx: 1.5, fill: '#7c5cff'}, g);
  const shield = el('rect', {x: -R * 0.6, y: -R * 0.95 - 3, width: 0, height: 3, rx: 1.5, fill: css('--muted')}, g);
  const fx = el('text', {'text-anchor': 'middle', y: -R * 1.15, 'font-size': 9, 'font-weight': 700, fill: '#7c5cff'}, g);"""),
    ("""  uEls[u.slot] = {g, ring, bar, blocked};""", """  uEls[u.slot] = {g, ring, bar, blocked, mana, shield, fx};"""),
    ("""    e.bar.setAttribute('width', (R * 1.2 * Math.max(0, fr.hp[u.slot]) / u.hp_max).toFixed(2));""",
     """    e.bar.setAttribute('width', (R * 1.2 * Math.max(0, fr.hp[u.slot]) / fr.hpmax[u.slot]).toFixed(2));
    e.mana.setAttribute('width', fr.manamax[u.slot] > 0 ? (R * 1.2 * Math.min(1, fr.mana[u.slot] / fr.manamax[u.slot])).toFixed(2) : 0);
    e.shield.setAttribute('width', (R * 1.2 * Math.min(1, fr.shield[u.slot] / fr.hpmax[u.slot])).toFixed(2));
    let lab = fr.stun[u.slot] ? 'choáng' : '';
    for (const ev of D.events) if (ev[1] === 3 && ev[2] === u.slot && T - ev[0] >= 0 && T - ev[0] < 8) lab = 'TUNG CHIÊU';
    e.fx.textContent = fr.alive[u.slot] ? lab : '';
    e.g.setAttribute('opacity', fr.stealth[u.slot] && fr.alive[u.slot] ? 0.45 : op);"""),
    # vẽ kỹ năng (tím, nét đứt) và hồi máu (xanh lá)
    ("""    if (e[1] !== EV_ATTACK) continue;
    const age = T - e[0];""",
     """    if (e[1] !== EV_ATTACK && e[1] !== 4 && e[1] !== 5) continue;
    if (e[2] < 0 || !bySlot[e[2]] || !bySlot[e[3]]) continue;
    const age = T - e[0];"""),
    ("""    const a = posAt(e[2], T), b = posAt(e[3], T), col = sideColor(bySlot[e[2]]);
    if (age <= 3) el('line', {x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: col, 'stroke-width': 2, 'stroke-opacity': (1 - age / 3).toFixed(2), 'stroke-linecap': 'round'}, gFx);""",
     """    const a = posAt(e[2], T), b = posAt(e[3], T);
    const col = e[1] === 4 ? '#7c5cff' : e[1] === 5 ? css('--ok') : sideColor(bySlot[e[2]]);
    if (age <= 3 && e[2] !== e[3]) el('line', {x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: col, 'stroke-width': e[1] === 4 ? 3 : 2, 'stroke-dasharray': e[1] === 1 ? '' : '4 3', 'stroke-opacity': (1 - age / 3).toFixed(2), 'stroke-linecap': 'round'}, gFx);"""),
    ("""    t.textContent = Math.round(e[4]);""", """    t.textContent = (e[1] === 5 ? '+' : '') + Math.round(e[4]);"""),
    # nhật ký
    ("""const NAMES = {0: 'đi tới ô', 1: 'đánh', 2: 'chết', 3: 'tung chiêu'};""",
     """const NAMES = {0: 'đi tới ô', 1: 'đánh', 2: 'chết', 3: 'tung chiêu', 4: 'kỹ năng/đồ trúng', 5: 'hồi máu cho', 6: 'tạo khiên cho', 7: 'làm choáng', 9: 'lướt tới cạnh'};"""),
    ("""    const src = bySlot[e[2]], dst = bySlot[e[3]];
    let txt = `bước ${e[0]}: ${src.name} ${NAMES[e[1]]}`;
    if (e[1] === EV_ATTACK) txt += ` ${dst.name}, ${Math.round(e[4])} sát thương`;""",
     """    const src = bySlot[e[2]], dst = bySlot[e[3]];
    if (!src) continue;
    let txt = `bước ${e[0]}: ${src.name} ${NAMES[e[1]] || e[1]}`;
    if (e[1] === EV_ATTACK || e[1] === 4) txt += ` ${dst ? dst.name : ''}, ${Math.round(e[4])} sát thương`;
    if (e[1] === 5 || e[1] === 6) txt += ` ${dst ? dst.name : ''} ${Math.round(e[4])}`;
    if (e[1] === 7 || e[1] === 9) txt += ` ${dst ? dst.name : ''}`;"""),
    # khung thông tin tướng đang chọn
    ("""    <h2 style="margin-top:14px">Sự kiện quanh bước hiện tại</h2>""",
     """    <div id="info" class="legend"></div>
    <h2 style="margin-top:14px">Sự kiện quanh bước hiện tại</h2>"""),
    ("""  renderTable(fr);
  renderLog();""",
     """  renderTable(fr);
  renderLog();
  const inf = document.getElementById('info');
  if (selected !== null) {
    const u = bySlot[selected], st = fr.stats[selected];
    inf.innerHTML = `<b>${u.name}</b> (${u.cost} tiền, ${u.role}; ${u.traits.join(', ')})<br>` +
      `<b>Kỹ năng</b> ${u.skill}<br><span>${u.skill_detail}</span><br>` +
      `<b>Đồ:</b> ${u.items.join(', ') || '—'}<br>` +
      `<b>Chỉ số lúc này:</b> công ${Math.round(st[0])}, sức mạnh phép ${Math.round(st[1])}, tốc đánh ${st[2].toFixed(2)}, giáp ${Math.round(st[3])}, kháng phép ${Math.round(st[4])}, máu tối đa ${Math.round(fr.hpmax[selected])}`;
  } else {
    inf.innerHTML = `<b>Tộc hệ kích hoạt</b> — xanh: ${D.meta.traits[0].join(', ') || '—'}; đỏ: ${D.meta.traits[1].join(', ') || '—'}. Bấm vào một tướng để xem kỹ năng, đồ và chỉ số.`;
  }"""),
    ("""      Một bước = 0,1 giây; tướng đi một ô mất <span id="ms"></span> bước.""",
     """      Một bước = 0,1 giây; tướng đi một ô mất <span id="ms"></span> bước.
      Thanh tím dưới thanh máu là năng lượng, thanh xám trên là khiên; nét đứt tím là kỹ năng hoặc hiệu ứng đồ trúng,
      số xanh lá là hồi máu. Từ giây 20 là hiệp phụ: mọi sát thương tăng dần (gấp đôi ở giây 25, gấp ba ở giây 30)."""),
]


def write_html(replay, path):
    html = _TEMPLATE
    for a, b in _PATCHES:
        if a not in html:
            raise RuntimeError(f"không vá được giao diện: {a[:60]!r}")
        html = html.replace(a, b)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(replay, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    path.write_text(html.replace("/*__DATA__*/", data), encoding="utf-8")
    return path


def show(data, team_a, team_b, seed=0, stochastic=False, path=None, open_browser=True):
    rep = record(data, team_a, team_b, seed=seed, stochastic=stochastic)
    path = write_html(rep, path or DEFAULT_DIR / f"replay5_{datetime.now():%Y%m%d_%H%M%S}.html")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path, rep


def random_team(data, rng, size, items=True):
    from .team import Unit
    cells = rng.choice(28, size=size, replace=False)
    champs = rng.choice(data.N, size=size, replace=False)
    combined = [i["name"] for i in data.items if i["kind"] == "combined"]
    team = []
    for c, cell in zip(champs, cells):
        it = list(rng.choice(combined, size=int(rng.integers(0, 3)), replace=False)) if items else []
        team.append(Unit(int(c), int(rng.integers(1, 3)), int(cell), items=it))
    return team


def list_all(data):
    print(f"{data.version}\n\nTƯỚNG (tên · giá · hệ · tộc · tầm · kỹ năng)")
    for c in data.champs:
        tr = ", ".join(data.trait_names[k] for k in c["traits"])
        print(f"  {c['name']:<16} {c['cost']} tiền  {c['role']:<8} {tr:<28} tầm {c['stats']['range']}  "
              f"{c['skill']['name']}: {c['skill']['text']}")
    print("\nĐỒ GHÉP (ghép từ · tên · hiệu ứng riêng)")
    comp = [i["name"] for i in data.items if i["kind"] == "component"]
    for i in data.items:
        if i["kind"] == "combined":
            a, b = (comp[k] for k in i["recipe"])
            print(f"  {a} + {b:<10} {i['name']:<20} {i['group']}")
    print("\nMẢNH: " + ", ".join(comp))


def duel(data, ta, tb, n, seed=0):
    """N trận giữa hai đội (chí mạng ngẫu nhiên); in tỷ lệ thắng, thời lượng, sát thương TB từng tướng."""
    sim = data.simulator(stochastic=True)
    res = sim.run(data.build_batch([ta] * n, [tb] * n, seeds=np.arange(n) + seed * 1000))
    w0, w1 = (res.result == 0).mean(), (res.result == 1).mean()
    print(f"{n} trận: xanh thắng {100 * w0:.0f}%, đỏ thắng {100 * w1:.0f}%, cùng thua {100 * (1 - w0 - w1):.0f}%; "
          f"trung bình {res.end_step.mean() * sim.g['dt']:.1f} giây, quân còn sống khi thắng "
          f"{res.alive.sum(1)[res.result < 2].mean():.1f}")
    dealt = res.dealt.sum(-1).mean(0)
    for side, team in ((0, ta), (1, tb)):
        print("  " + ("xanh" if side == 0 else "đỏ") + ": " + ", ".join(
            f"{data.names[data.champ_id(u.champ)]} {dealt[10 * side + j]:.0f}" for j, u in enumerate(team)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--random", type=int, default=0, help="hai đội ngẫu nhiên N tướng (kèm đồ ngẫu nhiên)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true", help="chí mạng ngẫu nhiên")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--out")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--list", action="store_true", help="in danh sách tướng và đồ")
    ap.add_argument("--games", type=int, default=0,
                    help="đấu N trận (chí mạng ngẫu nhiên, mỗi trận một seed) và in tỷ lệ thắng thay vì mở trận xem")
    a = ap.parse_args(argv)
    data = CombatData.load(a.catalog) if a.catalog else CombatData.load()
    if a.list:
        return list_all(data)
    if a.random:
        rng = np.random.default_rng(a.seed)
        ta, tb = random_team(data, rng, a.random), random_team(data, rng, a.random)
    elif a.a and a.b:
        ta, tb = data.parse_team(a.a), data.parse_team(a.b)
    else:
        ap.error("cần --random N, hoặc cả --a và --b (hoặc --list)")
    if a.games:
        return duel(data, ta, tb, a.games, a.seed)
    path, rep = show(data, ta, tb, seed=a.seed, stochastic=a.stochastic, path=a.out, open_browser=not a.no_open)
    res = {RES_SIDE0: "xanh (phe 0) thắng", RES_SIDE1: "đỏ (phe 1) thắng"}.get(rep["meta"]["result"], "hai bên cùng thua")
    print(f"{res} ở giây {rep['meta']['end_step'] * rep['meta']['dt']:.1f} · {path}")


if __name__ == "__main__":
    main()
