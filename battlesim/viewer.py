"""Trình xem lại trận: chạy một trận, ghi trạng thái từng bước, xuất file HTML tự chứa.

Trạng thái mỗi bước được chụp thẳng từ mảng của mô phỏng (không dựng lại từ nhật ký),
nên trình xem lại hiện đúng những gì mô phỏng thấy — kể cả lỗi.

Dùng trong Python:
    from battlesim.viewer import show
    show(cfg, team_a, team_b)                    # mở trình duyệt

Dùng từ dòng lệnh (từ thư mục gốc dự án):
    python -m battlesim.viewer --random 6
    python -m battlesim.viewer --a "Khiên Sắt*2@24, Cung Thủ@3" --b "Kiếm Khách@24, Pháp Sư@3"
"""
import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import load_config
from .layout import (
    F_DEALT, F_HP, F_HP_MAX, I_ALIVE, I_BLOCKED, I_CELL, I_CHAMP, I_ORD, I_RANGE, I_SIDE,
    I_STAR, I_TARGET, U,
)
from .sim import Simulator, make_batch

DEFAULT_DIR = Path("runs") / "replays"


def record(cfg, team_a, team_b, seed=0, level=1, stochastic=False):
    """Chạy một trận theo từng bước và trả về dict dữ liệu cho trình xem lại."""
    sim = Simulator(cfg, level=level, stochastic=stochastic, log=True, max_events=16384)
    batch = make_batch([team_a], [team_b], seeds=[seed])
    st = sim.init_state(batch)

    def snap(s):
        F, I = st.F[0], st.I[0]
        return {
            "s": s,
            "cell": I[:, I_CELL].tolist(),
            "hp": np.round(F[:, F_HP], 1).tolist(),
            "alive": I[:, I_ALIVE].tolist(),
            "target": I[:, I_TARGET].tolist(),
            "blocked": I[:, I_BLOCKED].tolist(),
            "dealt": np.round(F[:, F_DEALT], 1).tolist(),
        }

    frames = [snap(0)]
    result = 2
    end = cfg.max_steps
    for s in range(1, cfg.max_steps + 1):
        r = int(sim.step(st, s)[0])
        frames.append(snap(s))
        if r >= 0:
            result, end = r, s
            break

    I0 = sim.init_state(batch).I[0]   # trạng thái ban đầu, để lấy thông tin tĩnh
    F0 = sim.init_state(batch).F[0]
    units = []
    for u in range(U):
        c = int(I0[u, I_CHAMP])
        if c < 0:
            continue
        units.append({
            "slot": u, "side": int(I0[u, I_SIDE]), "ord": int(I0[u, I_ORD]),
            "name": cfg.names[c], "star": int(I0[u, I_STAR]), "range": int(I0[u, I_RANGE]),
            "hp_max": round(float(F0[u, F_HP_MAX]), 1),
        })

    n_ev = min(int(st.evc[0]), st.ev.shape[1])
    events = st.ev[0, :n_ev].tolist()

    # cảnh báo: hai tướng còn sống cùng một ô
    warnings = []
    for fr in frames:
        cells = [fr["cell"][u["slot"]] for u in units if fr["alive"][u["slot"]]]
        if len(cells) != len(set(cells)):
            warnings.append(f"Bước {fr['s']}: hai tướng cùng một ô")
    if st.evc[0] > st.ev.shape[1]:
        warnings.append("Nhật ký sự kiện bị tràn, thiếu đường đánh ở cuối trận")

    return {
        "meta": {"level": level, "stochastic": stochastic, "seed": int(seed), "dt": cfg.dt,
                 "move_steps": cfg.move_steps, "max_steps": cfg.max_steps,
                 "result": result, "end_step": end},
        "units": units, "frames": frames, "events": events, "warnings": warnings,
    }


def write_html(replay, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(replay, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = _TEMPLATE.replace("/*__DATA__*/", data)
    path.write_text(html, encoding="utf-8")
    return path


def show(cfg, team_a, team_b, seed=0, level=1, stochastic=False, path=None, open_browser=True):
    replay = record(cfg, team_a, team_b, seed=seed, level=level, stochastic=stochastic)
    if path is None:
        path = DEFAULT_DIR / f"replay_{datetime.now():%Y%m%d_%H%M%S}.html"
    path = write_html(replay, path)
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


# --------------------------------------------------------------------------- #
# Dòng lệnh
# --------------------------------------------------------------------------- #

def parse_team(text, cfg):
    """'Tên*sao@ô, ...' — sao mặc định 1; tên có thể thay bằng mã số tướng."""
    team = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "@" not in part:
            raise ValueError(f"Thiếu ô (@0–27) trong '{part}'")
        left, cell = part.rsplit("@", 1)
        star = 1
        if "*" in left:
            left, star = left.rsplit("*", 1)
            star = int(star)
        left = left.strip()
        cid = int(left) if left.isdigit() else cfg.champ_id(left)
        team.append((cid, star, int(cell)))
    return team


def random_team(rng, n_champs, size):
    cells = rng.choice(28, size=size, replace=False)
    return [(int(rng.integers(n_champs)), int(rng.integers(1, 4)), int(c)) for c in cells]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Xem lại một trận giao tranh trong trình duyệt")
    ap.add_argument("--a", help="đội phe 0, ví dụ 'Khiên Sắt*2@24, Cung Thủ@3'")
    ap.add_argument("--b", help="đội phe 1")
    ap.add_argument("--random", type=int, default=0, help="sinh hai đội ngẫu nhiên N tướng")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=1, choices=[0, 1])
    ap.add_argument("--stochastic", action="store_true", help="chí mạng ngẫu nhiên (nấc 3)")
    ap.add_argument("--out", help="đường dẫn file HTML")
    ap.add_argument("--no-open", action="store_true", help="không tự mở trình duyệt")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.random:
        rng = np.random.default_rng(args.seed)
        team_a = random_team(rng, cfg.n_champs, args.random)
        team_b = random_team(rng, cfg.n_champs, args.random)
    elif args.a and args.b:
        team_a, team_b = parse_team(args.a, cfg), parse_team(args.b, cfg)
    else:
        ap.error("cần --random N, hoặc cả --a và --b")

    path = show(cfg, team_a, team_b, seed=args.seed, level=args.level,
                stochastic=args.stochastic, path=args.out, open_browser=not args.no_open)
    print(path)


_TEMPLATE = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Xem lại trận</title>
<style>
:root{
  --bg:#f6f5f2;--panel:#ffffff;--ink:#1d1d1b;--muted:#6b6a65;--line:#dcd9d2;
  --hex:#ecebe6;--hex-edge:#cfccc4;--s0:#2563c9;--s1:#cf3f32;--warn:#b45309;--ok:#15803d;
  --range:rgba(37,99,201,.13);--sel:#111;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#161615;--panel:#1f1f1d;--ink:#ecebe6;--muted:#9c9a93;--line:#34332f;
  --hex:#262624;--hex-edge:#3a3935;--s0:#5b93f0;--s1:#f07063;--warn:#f59e0b;--ok:#4ade80;
  --range:rgba(91,147,240,.18);--sel:#fff;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650}
.meta{color:var(--muted);font-size:13px}
.result{font-weight:600}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:16px;padding:16px 20px;max-width:1280px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
svg{width:100%;height:auto;display:block}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
button,select{font:inherit;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--muted)}
#scrub{flex:1;min-width:160px}
.clock{font-variant-numeric:tabular-nums;min-width:120px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-weight:500;padding:4px 6px;border-bottom:1px solid var(--line)}
td{padding:4px 6px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
tr.dead td{opacity:.4}
tr.sel td{background:var(--range)}
tr{cursor:pointer}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;width:70px;display:inline-block;vertical-align:middle;margin-right:6px}
.bar>i{display:block;height:100%}
.warn{color:var(--warn);font-weight:600}
.legend{color:var(--muted);font-size:12.5px;margin:8px 2px 0}
.legend b{color:var(--ink);font-weight:600}
.tag{font-size:11px;padding:1px 5px;border-radius:4px;background:var(--line);color:var(--muted)}
#log{max-height:220px;overflow:auto;font-size:12.5px;color:var(--muted);margin-top:8px}
#log div.now{color:var(--ink)}
h2{font-size:13px;margin:0 0 6px;font-weight:600}
</style>
</head>
<body>
<header>
  <h1>Xem lại trận</h1>
  <span class="result" id="result"></span>
  <span class="meta" id="meta"></span>
  <span class="warn" id="warnings"></span>
</header>
<main>
  <section class="card">
    <svg id="board" viewBox="0 0 100 100" role="img" aria-label="Bàn cờ lục giác"></svg>
    <div class="controls">
      <button id="play" title="Phím cách">▶ Chạy</button>
      <button id="back" title="Mũi tên trái">◀ 1 bước</button>
      <button id="fwd" title="Mũi tên phải">1 bước ▶</button>
      <select id="speed" title="Tốc độ">
        <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
        <option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option>
      </select>
      <input type="range" id="scrub" min="0" step="0.1" value="0">
      <span class="clock" id="clock"></span>
    </div>
    <p class="legend">
      <b>Đang xem gì:</b> bàn 8 hàng × 7 cột ô lục giác. Phe 0 (<span style="color:var(--s0)">xanh</span>) đứng nửa dưới,
      phe 1 (<span style="color:var(--s1)">đỏ</span>) nửa trên, vạch giữa là ranh giới hai nửa sân.
      Mỗi vòng tròn là một tướng: chữ là tên viết tắt, chấm nhỏ là số sao, thanh trên đầu là máu.
      Đường mảnh là đòn đánh vừa ra, số bay lên là sát thương sau giáp. Bấm vào một tướng để tô các ô trong tầm đánh của nó.
      Một bước = 0,1 giây; tướng đi một ô mất <span id="ms"></span> bước.
    </p>
  </section>
  <aside class="card">
    <h2>Tướng</h2>
    <table>
      <thead><tr><th>Tướng</th><th>Máu</th><th>Gây ra</th><th>Mục tiêu</th></tr></thead>
      <tbody id="units"></tbody>
    </table>
    <h2 style="margin-top:14px">Sự kiện quanh bước hiện tại</h2>
    <div id="log"></div>
  </aside>
</main>
<script type="application/json" id="data">/*__DATA__*/</script>
<script>
(() => {
const D = JSON.parse(document.getElementById('data').textContent);
const ROWS = 8, COLS = 7, R = 26, W = Math.sqrt(3) * R, PAD = 14;
const EV_MOVE = 0, EV_ATTACK = 1, EV_DEATH = 2;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
};

// --- hình học: giống hexgrid.py (odd-r), lật dọc để phe 0 ở dưới ---
const rowOf = c => Math.floor(c / COLS), colOf = c => c % COLS;
function center(c) {
  const r = rowOf(c), q = colOf(c);
  return [PAD + W / 2 + W * (q + 0.5 * (r & 1)), PAD + R + 1.5 * R * (ROWS - 1 - r)];
}
function axial(c) { const r = rowOf(c), q = colOf(c) - ((r - (r & 1)) >> 1); return [q, r]; }
function dist(a, b) {
  const [qa, ra] = axial(a), [qb, rb] = axial(b);
  const dq = qa - qb, dr = ra - rb;
  return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
}
const hexPts = (x, y) => Array.from({length: 6}, (_, k) => {
  const a = Math.PI / 180 * (60 * k - 30);
  return `${(x + (R - 1) * Math.cos(a)).toFixed(2)},${(y + (R - 1) * Math.sin(a)).toFixed(2)}`;
}).join(' ');

const svg = document.getElementById('board');
const VW = PAD * 2 + W * (COLS + 0.5), VH = PAD * 2 + R * 2 + 1.5 * R * (ROWS - 1);
svg.setAttribute('viewBox', `0 0 ${VW.toFixed(1)} ${VH.toFixed(1)}`);

const gHex = el('g', {}, svg), gFx = el('g', {}, svg), gUnits = el('g', {}, svg), gTxt = el('g', {}, svg);
const hexEls = [];
for (let c = 0; c < ROWS * COLS; c++) {
  const [x, y] = center(c);
  hexEls.push(el('polygon', {points: hexPts(x, y), fill: css('--hex'), stroke: css('--hex-edge'), 'stroke-width': 1}, gHex));
}
// ranh giới hai nửa sân (giữa hàng 3 và hàng 4)
const yMid = (center(3 * COLS)[1] + center(4 * COLS)[1]) / 2;
el('line', {x1: PAD / 2, x2: VW - PAD / 2, y1: yMid, y2: yMid, stroke: css('--muted'), 'stroke-dasharray': '4 4', 'stroke-width': 1}, gHex);

// --- dữ liệu tĩnh của tướng ---
const units = D.units, bySlot = {};
units.forEach(u => bySlot[u.slot] = u);
const abbr = s => s.split(/\s+/).map(w => w[0]).join('').toUpperCase().slice(0, 2);
const sideColor = u => css(u.side === 0 ? '--s0' : '--s1');

// các lần di chuyển của từng tướng để vẽ chuyển động mượt
const moves = {};
units.forEach(u => moves[u.slot] = []);
let prevCell = {};
units.forEach(u => prevCell[u.slot] = D.frames[0].cell[u.slot]);
for (const e of D.events) {
  if (e[1] !== EV_MOVE) continue;
  const u = e[2], to = e[5];
  moves[u].push({s: e[0], from: prevCell[u], to});
  prevCell[u] = to;
}
const deathStep = {};
for (const e of D.events) if (e[1] === EV_DEATH) deathStep[e[2]] = e[0];

function posAt(slot, T) {
  const ms = moves[slot];
  let p = center(D.frames[0].cell[slot]);
  for (const m of ms) {
    if (m.s > T) break;
    const f = Math.min(1, (T - m.s) / D.meta.move_steps);
    const a = center(m.from), b = center(m.to);
    p = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
  }
  return p;
}

// phần tử SVG cho từng tướng
const uEls = {};
for (const u of units) {
  const g = el('g', {style: 'cursor:pointer'}, gUnits);
  const ring = el('circle', {r: R * 0.62, fill: sideColor(u), 'fill-opacity': 0.18, stroke: sideColor(u), 'stroke-width': 2}, g);
  const label = el('text', {'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-size': 12, 'font-weight': 650, fill: css('--ink')}, g);
  label.textContent = abbr(u.name);
  const stars = el('text', {'text-anchor': 'middle', y: R * 0.42, 'font-size': 8, fill: sideColor(u)}, g);
  stars.textContent = '•'.repeat(u.star);
  const barBg = el('rect', {x: -R * 0.6, y: -R * 0.95, width: R * 1.2, height: 4, rx: 2, fill: css('--line')}, g);
  const bar = el('rect', {x: -R * 0.6, y: -R * 0.95, width: R * 1.2, height: 4, rx: 2, fill: sideColor(u)}, g);
  const blocked = el('text', {'text-anchor': 'middle', y: R * 0.95, 'font-size': 8, fill: css('--warn')}, g);
  g.addEventListener('click', () => select(u.slot));
  uEls[u.slot] = {g, ring, bar, blocked};
}

// --- trạng thái phát ---
const END = D.meta.end_step;
let T = 0, playing = false, speed = 1, selected = null, lastTs = null;
const scrub = document.getElementById('scrub');
scrub.max = END;

function frameAt(T) { return D.frames[Math.min(D.frames.length - 1, Math.max(0, Math.floor(T + 1e-6)))]; }

function render() {
  const fr = frameAt(T);
  // tầm đánh của tướng đang chọn
  for (let c = 0; c < hexEls.length; c++) hexEls[c].setAttribute('fill', css('--hex'));
  if (selected !== null && fr.alive[selected]) {
    const u = bySlot[selected], here = fr.cell[selected];
    for (let c = 0; c < hexEls.length; c++)
      if (dist(here, c) <= u.range) hexEls[c].setAttribute('fill', css('--range'));
  }
  // tướng
  for (const u of units) {
    const e = uEls[u.slot], [x, y] = posAt(u.slot, T);
    const d = deathStep[u.slot];
    let op = 1;
    if (d !== undefined && T >= d) op = Math.max(0.12, 1 - (T - d) / 6);
    e.g.setAttribute('transform', `translate(${x.toFixed(2)},${y.toFixed(2)})`);
    e.g.setAttribute('opacity', op);
    e.bar.setAttribute('width', (R * 1.2 * Math.max(0, fr.hp[u.slot]) / u.hp_max).toFixed(2));
    e.ring.setAttribute('stroke', u.slot === selected ? css('--sel') : sideColor(u));
    e.ring.setAttribute('stroke-width', u.slot === selected ? 3 : 2);
    e.blocked.textContent = fr.alive[u.slot] && fr.blocked[u.slot] > 0 ? `chặn ${fr.blocked[u.slot]}` : '';
  }
  // đòn đánh và số sát thương gần thời điểm T
  gFx.replaceChildren(); gTxt.replaceChildren();
  const stack = {};   // số đòn đã vẽ trên cùng mục tiêu trong cùng bước, để xếp số sát thương không chồng nhau
  for (const e of D.events) {
    if (e[1] !== EV_ATTACK) continue;
    const age = T - e[0];
    if (age < 0 || age > 6) continue;
    const sk = e[0] + ':' + e[3], k = stack[sk] = (stack[sk] || 0) + 1;
    const a = posAt(e[2], T), b = posAt(e[3], T), col = sideColor(bySlot[e[2]]);
    if (age <= 3) el('line', {x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: col, 'stroke-width': 2, 'stroke-opacity': (1 - age / 3).toFixed(2), 'stroke-linecap': 'round'}, gFx);
    const t = el('text', {x: b[0] + 8 + (k - 1) * 4, y: b[1] - 10 - age * 3 - (k - 1) * 11, 'font-size': 10, 'font-weight': 600, fill: col, opacity: (1 - age / 6).toFixed(2)}, gTxt);
    t.textContent = Math.round(e[4]);
  }
  // bảng và đồng hồ
  scrub.value = T;
  document.getElementById('clock').textContent = `bước ${Math.floor(T)} / ${END} · ${(Math.floor(T) * D.meta.dt).toFixed(1)} giây`;
  renderTable(fr);
  renderLog();
}

function renderTable(fr) {
  const rows = [...units].sort((a, b) => a.side - b.side || a.ord - b.ord).map(u => {
    const hp = Math.max(0, fr.hp[u.slot]), pct = (100 * hp / u.hp_max).toFixed(0);
    const t = fr.target[u.slot], tn = t >= 0 && bySlot[t] ? abbr(bySlot[t].name) + (bySlot[t].side ? ' (đỏ)' : ' (xanh)') : '—';
    const cls = (fr.alive[u.slot] ? '' : 'dead') + (u.slot === selected ? ' sel' : '');
    return `<tr class="${cls}" data-slot="${u.slot}">
      <td><span class="dot" style="background:${sideColor(u)}"></span>${u.name} <span class="tag">${u.star}★</span></td>
      <td><span class="bar"><i style="width:${pct}%;background:${sideColor(u)}"></i></span>${Math.round(hp)}</td>
      <td>${Math.round(fr.dealt[u.slot])}</td><td>${fr.alive[u.slot] ? tn : 'đã chết'}</td></tr>`;
  });
  const tb = document.getElementById('units');
  tb.innerHTML = rows.join('');
  tb.querySelectorAll('tr').forEach(r => r.addEventListener('click', () => select(+r.dataset.slot)));
}

const NAMES = {0: 'đi tới ô', 1: 'đánh', 2: 'chết', 3: 'tung chiêu'};
function renderLog() {
  const s = Math.floor(T), lines = [];
  for (const e of D.events) {
    if (e[0] < s - 5 || e[0] > s) continue;
    const src = bySlot[e[2]], dst = bySlot[e[3]];
    let txt = `bước ${e[0]}: ${src.name} ${NAMES[e[1]]}`;
    if (e[1] === EV_ATTACK) txt += ` ${dst.name}, ${Math.round(e[4])} sát thương`;
    if (e[1] === EV_MOVE) txt += ` ${e[5]}`;
    lines.push(`<div class="${e[0] === s ? 'now' : ''}">${txt}</div>`);
  }
  document.getElementById('log').innerHTML = lines.join('') || '<div>—</div>';
}

function select(slot) { selected = selected === slot ? null : slot; render(); }
function setT(v) { T = Math.min(END, Math.max(0, v)); render(); }
function tick(ts) {
  if (!playing) { lastTs = null; return; }
  if (lastTs !== null) setT(T + (ts - lastTs) / 1000 / D.meta.dt * speed);
  lastTs = ts;
  if (T >= END) { playing = false; updatePlay(); return; }
  requestAnimationFrame(tick);
}
function updatePlay() { document.getElementById('play').textContent = playing ? '❚❚ Dừng' : '▶ Chạy'; }
function toggle() { if (T >= END) T = 0; playing = !playing; updatePlay(); if (playing) requestAnimationFrame(tick); }

document.getElementById('play').onclick = toggle;
document.getElementById('back').onclick = () => { playing = false; updatePlay(); setT(Math.ceil(T) - 1); };
document.getElementById('fwd').onclick = () => { playing = false; updatePlay(); setT(Math.floor(T) + 1); };
document.getElementById('speed').onchange = e => speed = +e.target.value;
scrub.oninput = e => { playing = false; updatePlay(); setT(+e.target.value); };
document.addEventListener('keydown', e => {
  if (e.key === ' ') { e.preventDefault(); toggle(); }
  if (e.key === 'ArrowLeft') document.getElementById('back').click();
  if (e.key === 'ArrowRight') document.getElementById('fwd').click();
});

// --- tiêu đề ---
const RES = {0: 'Phe 0 (xanh) thắng', 1: 'Phe 1 (đỏ) thắng', 2: 'Cả hai cùng thua'};
document.getElementById('result').textContent = RES[D.meta.result] + (D.meta.result === 2 && END >= D.meta.max_steps ? ' — hết giờ' : '');
document.getElementById('meta').textContent =
  `nấc ${D.meta.level} · ${D.meta.stochastic ? 'chí mạng ngẫu nhiên' : 'tất định'} · seed ${D.meta.seed} · ${END} bước`;
document.getElementById('warnings').textContent = D.warnings.length ? '⚠ ' + D.warnings.slice(0, 3).join('; ') : '';
document.getElementById('ms').textContent = D.meta.move_steps;
render();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
