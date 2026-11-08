"""Học bắt chước bot luật để khởi động agent (phương án dự phòng của giai đoạn 1 trong tài liệu thiết kế).

Khi agent bắt đầu từ chính sách gần như ngẫu nhiên, nó thua bot luật 100% số ván: phần thưởng cuối ván lúc
nào cũng là -1 và gần như không có tín hiệu để học. Học bắt chước cho mạng một điểm xuất phát ở trình độ
xấp xỉ bot (trộn cả bốn lối chơi), từ đó PPO có cả ván thắng lẫn ván thua để học tiếp.

    python3 -m autochess.rl.bc --config configs/p2_bots.yaml configs/laptop.yaml
    python3 -m autochess.rl.train --config configs/p2_bots.yaml configs/laptop.yaml \\
        --set init=runs/bc_<thời gian>/checkpoints/latest.pt ppo.value_warmup=50 ppo.lr_warmup=50

Dữ liệu được sinh liên tục (không lưu xuống đĩa): mỗi vòng lặp chạy `bc.steps` bước của 256 ván bot đấu
nhau, học `bc.epochs` lượt trên lô đó rồi bỏ. Chỉ học phần chính sách (critic học ở PPO, nên dùng
ppo.value_warmup để vài lần cập nhật đầu chỉ học critic, rồi ppo.lr_warmup để chính sách không bị kéo
đi quá nhanh ngay khi PPO bắt đầu).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from autochess.core import load_catalog, FormulaCombat
from .botdata import BotDataCollector
from .config import load_config
from .evaluate import bot_matrix, evaluate, evaluate_lobbies, format_eval, format_lobbies
from .nets import build_net
from .policy import Policy, to_tensors, masked_logits
from .train import _dump_yaml


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.sets)
    bc = cfg["bc"]
    cat = load_catalog(cfg.get("catalog"))
    combat = FormulaCombat(catalog=cat)
    run_dir = os.path.join(cfg["run_dir"], f"bc_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    cfg["versions"] = dict(catalog=cat.version, combat=combat.version)
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
    opt = torch.optim.Adam(net.parameters(), lr=bc["lr"], eps=1e-5)
    col = BotDataCollector(cat, combat, bots=tuple(bc["bots"]), num_games=cfg["env"]["num_games"],
                           num_players=P, seed=cfg["seed"], matchup_row=cfg["env"]["matchup_row"],
                           obs_version=getattr(net, "obs_version", 1))
    aux_fn = getattr(net, "aux_loss", None)
    mb = int(cfg["ppo"]["minibatch_size"])
    log = open(os.path.join(run_dir, "bc.jsonl"), "a")
    seen, it = 0, 0
    total = int(bc["samples"])
    ev = cfg["eval"]
    matrix = None
    print(f"học bắt chước: {run_dir} · bot {bc['bots']} · mục tiêu {total:,} mẫu")
    while seen < total:
        t0 = time.perf_counter()
        data = col.collect(int(bc["steps"]))
        t1 = time.perf_counter()
        M = len(data["actions"])
        if M == 0:
            continue
        for g in opt.param_groups:
            g["lr"] = bc["lr"] * max(1.0 - seen / total, 0.05)
        net.train()
        losses, accs = [], []
        for _ in range(int(bc["epochs"])):
            perm = rng.permutation(M)
            for s in range(0, M, mb):
                idx = perm[s:s + mb]
                obs = to_tensors({k: v[idx] for k, v in data["obs"].items()}, device)
                target = torch.as_tensor(data["actions"][idx], device=device)
                logits, _ = net(obs)
                ml = masked_logits(logits, obs["mask"])
                loss = F.cross_entropy(ml, target)
                if aux_fn is not None:
                    loss = loss + aux_fn()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["ppo"]["max_grad_norm"])
                opt.step()
                losses.append(loss.item())
                accs.append((ml.argmax(-1) == target).float().mean().item())
        it += 1
        seen += M
        row = dict(iter=it, samples=seen, loss=float(np.mean(losses)), acc=float(np.mean(accs)),
                   collect_s=round(t1 - t0, 2), train_s=round(time.perf_counter() - t1, 2))
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"[{it:4d}] {seen:>10,} mẫu · loss {row['loss']:.3f} · trùng hành động bot {100 * row['acc']:.1f}% · "
              f"thu thập {row['collect_s']}s, học {row['train_s']}s")
        last = seen >= total
        if bc["eval_every"] and (it % bc["eval_every"] == 0 or last) and P > 2 and ev.get("lobbies"):
            # nhiều người: đánh giá trong các sảnh chỉ gồm bot luật (sảnh có bản neo / "recent" bỏ qua)
            from autochess.bots.scripted import PRESETS
            lob = {n: o for n, o in ev["lobbies"].items() if all(x in PRESETS for x in o)}
            res = evaluate_lobbies(cat, combat, Policy(net, device), lob, games=bc["eval_games"],
                                   B=ev.get("lobby_B", 64), seed=ev["seed"], matchup_row=cfg["env"]["matchup_row"])
            print("  ── đánh giá ──\n  " + format_lobbies(res).replace("\n", "\n  "))
            log.write(json.dumps(dict(iter=it, eval=res)) + "\n")
            log.flush()
        elif bc["eval_every"] and (it % bc["eval_every"] == 0 or last):
            if matrix is None:
                matrix = bot_matrix(cat, combat, games=ev["matrix_games"], B=ev["B"],
                                    cache=os.path.join(run_dir, "bot_matrix.json"))
            res = evaluate(cat, combat, Policy(net, device), opponents=tuple(ev["opponents"]),
                           games=bc["eval_games"], seed=ev["seed"], B=ev["B"],
                           matchup_row=cfg["env"]["matchup_row"], matrix=matrix)
            print("  ── đánh giá ──\n  " + format_eval(res).replace("\n", "\n  "))
            log.write(json.dumps(dict(iter=it, eval=res)) + "\n")
            log.flush()
        if it % 20 == 0 or last:
            torch.save(dict(net=net.state_dict(), config=cfg, source="bc", samples=seen),
                       os.path.join(run_dir, "checkpoints", "latest.pt"))
    if bc.get("export"):
        import shutil
        os.makedirs(os.path.dirname(bc["export"]) or ".", exist_ok=True)
        shutil.copyfile(os.path.join(run_dir, "checkpoints", "latest.pt"), bc["export"])
        print(f"đã chép checkpoint cuối sang {bc['export']}")
    print(f"xong: {seen:,} mẫu. Dùng tiếp với: --set init={run_dir}/checkpoints/latest.pt "
          f"ppo.value_warmup=50 ppo.lr_warmup=50")


if __name__ == "__main__":
    main()
