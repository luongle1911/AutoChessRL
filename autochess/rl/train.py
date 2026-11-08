"""Huấn luyện agent vĩ mô bằng PPO.

    python3 -m autochess.rl.train --config configs/p2_random.yaml configs/laptop.yaml   # bước 3: thắng bot ngẫu nhiên
    python3 -m autochess.rl.train --config configs/p2_bots.yaml configs/laptop.yaml     # giai đoạn 1
    python3 -m autochess.rl.train --resume runs/<tên>                                    # chạy tiếp
    python3 -m autochess.rl.train --config ... --set net.kind=flat --set name=flat_baseline
    python3 -m autochess.rl.train --config ... --set net.mixer=attn name=attn   # thử attention giữa thực thể
    python3 -m autochess.rl.train --config configs/p3_lobby.yaml configs/pc.yaml   # PC: thu thập nhiều tiến trình

Mạng mặc định là bản 2 (net.kind=entity2). Khởi tạo từ mạng học bắt chước (rl.bc) nên kèm
ppo.value_warmup (chỉ học critic, chính sách giữ nguyên) và ppo.lr_warmup (tốc độ học tăng dần sau đó):
    --set init=runs/bc_<thời gian>/checkpoints/latest.pt ppo.value_warmup=50 ppo.lr_warmup=50
Đổi file dữ liệu (--set catalog=...) vẫn nạp được checkpoint bản 2: phần riêng được ghép theo tên tướng/tộc.

Mỗi lần chạy tạo runs/<tên>_<thời gian>/ gồm: config.yaml (cấu hình đã gộp, phiên bản dữ liệu và
module giao tranh), metrics.jsonl (mỗi lần cập nhật), eval.jsonl (mỗi lần đánh giá), checkpoints/,
và tensorboard/ nếu đã cài tensorboard.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from autochess.core import load_catalog, FormulaCombat
from .compat import checkpoint_config, load_into
from .config import load_config, apply_sets
from .evaluate import bot_matrix, evaluate, evaluate_lobbies, format_eval, format_lobbies, load_anchors
from .league import League
from .gae import compute_gae
from .nets import build_net
from .opponents import OpponentPool, make_opponent
from .observe import pad_opponents
from .policy import Policy
from .ppo import ppo_update, flatten_rollout
from .rollout import Lobbies, Rollout


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def _dump_yaml(cfg: dict, path: str) -> None:
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        with open(path.replace(".yaml", ".json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


class Logger:
    def __init__(self, run_dir: str):
        self.f = open(os.path.join(run_dir, "metrics.jsonl"), "a")
        self.fe = open(os.path.join(run_dir, "eval.jsonl"), "a")
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(run_dir, "tensorboard"))
        except Exception:
            pass

    def log(self, row: dict, step: int, kind: str = "train") -> None:
        f = self.f if kind == "train" else self.fe
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        if self.tb is not None:
            for k, v in _flat(row).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                    self.tb.add_scalar(f"{kind}/{k}", v, step)


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "/"))
        else:
            out[key] = v
    return out


def _format_vs(g: dict, pool) -> str:
    """Tỷ lệ thắng trong các ván vừa chơi: bot luật từng con, các bản cũ gộp lại, số ván tự đấu."""
    kinds = dict(zip(pool.names, pool.kinds()))
    bots = [f"{k[7:]}:{100 * v:.0f}" for k, v in g.items() if k.startswith("win_vs/") and kinds.get(k[7:]) != "policy"]
    pol = [v for k, v in g.items() if k.startswith("win_vs/") and kinds.get(k[7:]) == "policy"]
    out = " ".join(bots)
    if pol:
        out += f" bản-cũ:{100 * float(np.mean(pol)):.0f}"
    if g.get("selfplay"):
        out += f" · tự đấu {g['selfplay']} ván"
    for P in (4, 8):
        if f"place/{P}" in g:
            out += f" · {P} người hạng TB {g[f'place/{P}']:.2f}"
    return out


def _summarize_games(finished: list, names: list) -> dict:
    if not finished:
        return {}
    out = {"games": len(finished), "rounds": float(np.mean([f["rounds"] for f in finished])),
           "selfplay": sum(bool(f.get("selfplay")) for f in finished)}
    for n in names:
        sc = [float(a < o) for f in finished for a in f["agent_placement"]
              for opp, o in zip(f["opponents"], f["opp_placement"]) if opp == n]
        if sc:
            out[f"win_vs/{n}"] = float(np.mean(sc))
    # ván nhiều người có đối thủ từ kho: hạng trung bình và tỷ lệ top nửa của agent
    for P in sorted({f.get("players", 2) for f in finished}):
        if P <= 2:
            continue
        pl = [a for f in finished if f.get("players") == P and not f.get("selfplay") for a in f["agent_placement"]]
        if pl:
            out[f"place/{P}"] = float(np.mean(pl))
            out[f"top_half/{P}"] = float(np.mean(np.asarray(pl) <= P // 2))
    return out


def make_rollouts(cat, combat, pool, cfg: dict, update: int):
    """Rollout (một loại sảnh) hoặc Lobbies (env.mix, vd {4: 0.25}: 25% số ván là sảnh 4 người)."""
    env, rw, oc = cfg["env"], cfg["reward"], cfg["opponents"]
    P = cfg["num_players"]
    mix = {int(k): float(v) for k, v in (env.get("mix") or {}).items() if float(v) > 0 and int(k) != P}
    sizes = {P: env["num_games"]}
    for q, frac in mix.items():
        n = max(1, int(round(env["num_games"] * frac)))
        sizes[q] = n
        sizes[P] -= n
    common = dict(gamma_round=rw["gamma_round"], lam=rw["lam"], beta=rw["beta"], matchup_row=env["matchup_row"],
                  selfplay_prob=oc["selfplay_prob"], churn_cost=rw.get("churn_cost", 0.0),
                  agent_seats=tuple(oc.get("agent_seats", (1, 1))), skip_waiting=env.get("skip_waiting", False))
    ros = [Rollout(cat, combat, pool, n, q, env["steps"], seed=cfg["seed"] + 1000 * update + 7 * k, **common)
           for k, (q, n) in enumerate(sizes.items())]
    return ros[0] if len(ros) == 1 else Lobbies(ros)


def collect_batch(ro, policy, O: int):
    """Thu thập một đợt, tính GAE cho từng loại sảnh, đệm chiều đối thủ lên O rồi ghép thành một lô PPO."""
    outs = ro.collect(policy) if isinstance(ro, Lobbies) else [ro.collect(policy)]
    batches = []
    for out in outs:
        buf = out["buf"]
        T_, B_, S_ = buf["rewards"].shape
        flat = lambda x: x.reshape(T_, B_ * S_)
        adv, ret = compute_gae(flat(buf["rewards"]), flat(buf["values"]), flat(buf["dones"]),
                               flat(buf["gammas"]), flat(buf["lambdas"]), flat(buf["active"]),
                               out["last_values"].reshape(-1))
        b = flatten_rollout(buf, adv.reshape(T_, B_, S_), ret.reshape(T_, B_, S_))
        b["obs"] = pad_opponents(b["obs"], O)
        batches.append(b)
    if len(batches) == 1:
        batch = batches[0]
    else:
        batch = {k: np.concatenate([b[k] for b in batches]) for k in batches[0] if k != "obs"}
        batch["obs"] = {k: np.concatenate([b["obs"][k] for b in batches]) for k in batches[0]["obs"]}
    info = dict(seconds=sum(o["seconds"] for o in outs), policy_seconds=sum(o["policy_seconds"] for o in outs),
                finished=[f for o in outs for f in o["finished"]])
    return batch, info


def lr_at(pcfg: dict, update: int, decisions: int, total: int) -> float:
    """Tốc độ học: giảm tuyến tính theo số quyết định (anneal_lr); sau value_warmup lần cập nhật chỉ học
    critic, tăng dần trong lr_warmup lần (bắt đầu từ lr / lr_warmup)."""
    lr = pcfg["lr"]
    if pcfg.get("anneal_lr", True):
        lr *= max(1.0 - decisions / total, 0.02)
    vw, lw = int(pcfg.get("value_warmup", 0)), int(pcfg.get("lr_warmup", 0))
    if lw > 0 and update >= vw:
        lr *= min(1.0, (update - vw + 1) / lw)
    return lr


def ent_at(pcfg: dict, decisions: int, total: int) -> float:
    """Hệ số entropy: hằng số, hoặc giảm tuyến tính từ ent_coef về ent_coef_final theo số quyết định."""
    final = pcfg.get("ent_coef_final")
    if final is None:
        return pcfg["ent_coef"]
    f = min(decisions / max(total, 1), 1.0)
    return pcfg["ent_coef"] + (final - pcfg["ent_coef"]) * f


def save_checkpoint(path, net, opt, cfg, update, decisions, pool, league=None) -> None:
    tmp = path + ".tmp"
    torch.save(dict(net=net.state_dict(), optimizer=opt.state_dict(), config=cfg, update=update,
                    decisions=decisions, pool=pool.state_dict(),
                    league=league.state_dict() if league is not None else None), tmp)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], dest="sets", help="ghi đè, vd ppo.lr=1e-4")
    ap.add_argument("--resume", default=None, help="thư mục một lần chạy cũ")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)

    if a.resume:
        ck_path = os.path.join(a.resume, "checkpoints", "latest.pt")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = apply_sets(checkpoint_config(ck), a.sets)
        run_dir = a.resume
    else:
        ck = None
        cfg = load_config(a.config, a.sets)
        run_dir = os.path.join(cfg["run_dir"], f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    cat = load_catalog(cfg.get("catalog"))
    combat = FormulaCombat(catalog=cat)
    cfg["versions"] = dict(catalog=cat.version, combat=combat.version, git=_git_commit())
    if not a.resume:
        _dump_yaml(cfg, os.path.join(run_dir, "config.yaml"))

    if cfg["threads"]:
        torch.set_num_threads(int(cfg["threads"]))
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    P = cfg["num_players"]

    net = build_net(cat, cfg["net"], P).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["ppo"]["lr"], eps=1e-5)
    oc = cfg["opponents"]
    lc = cfg.get("league", {})
    use_league = bool(lc.get("seeds")) or int(lc.get("snapshot_every", 0)) > 0
    pool = OpponentPool([make_opponent(cat, combat, n, seed=cfg["seed"] + i) for i, n in enumerate(oc["names"])],
                        prioritize=oc["prioritize"], power=oc["power"], floor=oc["floor"],
                        group_weights=lc.get("group_weights") if use_league else None)
    league = League(pool, cat, combat, run_dir, lc, matchup_row=cfg["env"]["matchup_row"],
                    seed=cfg["seed"]) if use_league else None
    update, decisions = 0, 0
    init = cfg.get("init")
    if ck is None and init:
        # khởi tạo từ một checkpoint khác (ví dụ mạng đã học bắt chước bot bằng rl.bc)
        ick = torch.load(init, map_location=device, weights_only=False)
        ikind = checkpoint_config(ick)["net"]["kind"]
        if ikind != cfg["net"]["kind"]:
            raise SystemExit(f"init là mạng '{ikind}' nhưng cấu hình là '{cfg['net']['kind']}' (--set net.kind=...)")
        rep = load_into(net, ick["net"])
        print(f"khởi tạo mạng từ {init}")
        if rep.changed or rep.warnings:
            print(rep)
    if ck is not None:
        rep = load_into(net, ck["net"])
        if rep.changed:
            # hàng phần riêng đã đổi nên trạng thái Adam cũ không còn khớp: bắt đầu lại trạng thái Adam
            print(rep)
            print("file dữ liệu đã đổi: không nạp trạng thái optimizer")
        else:
            opt.load_state_dict(ck["optimizer"])
        if league is not None and ck.get("league"):
            league.load_state_dict(ck["league"])
        pool.load_state_dict(ck["pool"])
        update, decisions = ck["update"], ck["decisions"]
        rng = np.random.default_rng(cfg["seed"] + update)
    if league is not None and ck is None:
        for name, path in (lc.get("seeds") or {}).items():
            league.add_seed(path, name)
    env = cfg["env"]
    workers = int(cfg.get("workers", 0))
    if workers > 0:
        from .vec import VecRollout
        ro = VecRollout(cfg, workers, update)
    else:
        ro = make_rollouts(cat, combat, pool, cfg, update)
    policy = Policy(net, device)
    logger = Logger(run_dir)
    ev_cfg = cfg["eval"]
    matrix = None
    anchors = None
    n_params = sum(p.numel() for p in net.parameters())
    print(f"chạy: {run_dir} · mạng {cfg['net']['kind']} {n_params:,} tham số · thiết bị {device} · "
          f"{env['num_games']} ván x {env['steps']} bước mỗi đợt"
          + (f" (pha sảnh {env['mix']})" if env.get("mix") else "") + f" · đối thủ {pool.names}"
          + (f" · tự đấu {100 * oc['selfplay_prob']:.0f}%" if oc["selfplay_prob"] else "")
          + (f" · thu thập {workers} tiến trình" if workers > 0 else ""))

    total = int(cfg["total_decisions"])
    while decisions < total:
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg["ppo"], update, decisions, total)
        if workers > 0:
            batch, out = ro.collect(net, pool, league, update)
        else:
            batch, out = collect_batch(ro, policy, P - 1)
        t1 = time.perf_counter()
        pcfg = dict(cfg["ppo"])
        pcfg["ent_coef"] = ent_at(cfg["ppo"], decisions, total)
        if update < int(cfg["ppo"].get("value_warmup", 0)):
            pcfg["pg_coef"] = 0.0    # vài lần cập nhật đầu chỉ học critic (critic của mạng bắt chước chưa học gì)
        st = ppo_update(net, opt, batch, pcfg, rng, device)
        t2 = time.perf_counter()
        update += 1
        decisions += st["decisions"]
        row = dict(update=update, decisions=decisions, lr=opt.param_groups[0]["lr"], ent_coef=pcfg["ent_coef"],
                   rollout_s=round(out["seconds"], 2), policy_s=round(out["policy_seconds"], 2),
                   update_s=round(t2 - t1, 2), decisions_per_s=round(st["decisions"] / (t2 - t0), 1),
                   **{k: v for k, v in st.items() if k not in ("samples", "decisions")},
                   games=_summarize_games(out["finished"], pool.names),
                   pool=dict(zip(pool.names, [round(float(w), 3) for w in pool.winrate])))
        logger.log(row, decisions)
        g = row["games"]
        vs = _format_vs(g, pool)
        print(f"[{update:5d}] {decisions:>11,} qđ · {row['decisions_per_s']:>7,.0f} qđ/s · "
              f"KL {st['approx_kl']:.4f} kẹp {st['clipfrac']:.2f} entropy {st['entropy']:.2f} "
              f"EV {st['explained_variance']:.2f} · {g.get('games', 0)} ván, thắng% {vs}")

        if league is not None:
            msg = league.maybe_snapshot(net, cfg, update)
            if msg:
                print(f"  kho: {msg} · thắng% trước bản cũ: {league.summary()}")
        if not a.no_eval and ev_cfg["every"] and update % ev_cfg["every"] == 0:
            if anchors is None:
                anchors = load_anchors(cat, combat, ev_cfg.get("anchors") or {})
                recent_cache = {}
            if matrix is None and ev_cfg.get("matrix", True):
                names = tuple(ev_cfg["opponents"]) + tuple(a_ for a_ in anchors if a_ not in ev_cfg["opponents"])
                cache = "bot_matrix.json" if not anchors else "eval_matrix.json"
                print(f"  (tính bảng thắng giữa các đối thủ cố định {list(names)} một lần, lưu vào {cache})")
                matrix = bot_matrix(cat, combat, names=names, games=ev_cfg["matrix_games"], B=ev_cfg["B"],
                                    cache=os.path.join(run_dir, cache), players=anchors)
            te = time.perf_counter()
            if ev_cfg["games"]:
                res = evaluate(cat, combat, policy, opponents=tuple(ev_cfg["opponents"]), games=ev_cfg["games"],
                               seed=ev_cfg["seed"], B=ev_cfg["B"], matchup_row=env["matchup_row"], matrix=matrix,
                               anchors=anchors)
            else:       # eval.games = 0: bỏ phần 1 đấu 1, chỉ đánh giá trong sảnh
                res = dict(milestones={})
            lob_text = ""
            if ev_cfg.get("lobbies"):
                lobbies, recent = resolve_recent(ev_cfg["lobbies"], os.path.join(run_dir, "snapshots"),
                                                 recent_cache, cat, combat)
                lob = evaluate_lobbies(cat, combat, policy, lobbies, players={**anchors, **recent},
                                       games=ev_cfg.get("lobby_games", 128), B=ev_cfg.get("lobby_B", 64),
                                       seed=ev_cfg["seed"], matchup_row=env["matchup_row"],
                                       elo_anchor=ev_cfg.get("lobby_elo_anchor"))
                res["lobby"] = lob
                m5 = ev_cfg.get("m5_lobby")
                if m5 and m5 in lob["lobbies"]:
                    res["milestones"]["m5"] = bool(lob["lobbies"][m5]["top_half"] > 0.6)
                lob_text = format_lobbies(lob) + "\n"
            text = lob_text + (format_eval(res) if ev_cfg["games"] else "")
            res.update(update=update, decisions=decisions, seconds=round(time.perf_counter() - te, 1))
            logger.log(res, decisions, kind="eval")
            print(f"  ── đánh giá ({res['seconds']:.0f} giây) ──\n  " + text.replace("\n", "\n  "))
        if update % cfg["save_every"] == 0 or decisions >= total:
            ckdir = os.path.join(run_dir, "checkpoints")
            save_checkpoint(os.path.join(ckdir, "latest.pt"), net, opt, cfg, update, decisions, pool, league)
            if ev_cfg["every"] and update % ev_cfg["every"] == 0:
                save_checkpoint(os.path.join(ckdir, f"update_{update:05d}.pt"), net, opt, cfg, update,
                                decisions, pool, league)
    if workers > 0:
        ro.close()
    print(f"xong: {decisions:,} quyết định, {update} lần cập nhật. Checkpoint: {run_dir}/checkpoints/latest.pt")


def resolve_recent(lobbies: dict, snap_dir: str, cache: dict, cat, combat) -> tuple[dict, dict]:
    """Thay tên "recent" trong các sảnh đánh giá bằng các bản chụp mới nhất của lần chạy này (mỗi ghế "recent"
    một bản khác nhau, mới nhất trước; thiếu thì lặp lại). Chưa có bản chụp nào thì bỏ các sảnh đó.
    Trả về (sảnh, {tên bản chụp: AgentPlayer}); cache giữ các bản đã nạp."""
    import glob
    files = sorted(glob.glob(os.path.join(snap_dir, "snap_*.pt")), reverse=True)
    out, used = {}, {}
    for name, opps in lobbies.items():
        n_recent = sum(1 for o in opps if o == "recent")
        if n_recent and not files:
            continue
        pick = [files[i % len(files)] for i in range(n_recent)]
        row, k = [], 0
        for o in opps:
            if o == "recent":
                path = pick[k]
                k += 1
                nm = os.path.splitext(os.path.basename(path))[0]
                if nm not in cache:
                    cache[nm] = load_anchors(cat, combat, {nm: path})[nm]
                used[nm] = cache[nm]
                row.append(nm)
            else:
                row.append(o)
        out[name] = row
    for nm in list(cache):          # bỏ các bản không còn dùng
        if nm not in used:
            del cache[nm]
    return out, used


if __name__ == "__main__":
    main()
