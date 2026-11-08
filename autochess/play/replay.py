"""Đọc/ghi log ván. Log không lưu trạng thái: phát lại bằng cách chạy lại lõi."""
from __future__ import annotations

import json
import os
import time

from .runner import GameRunner, LOG_FORMAT
from .seats import ReplaySeat


def save_log(log: dict, path: str | None = None, folder: str = "logs") -> str:
    if path is None:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"game_{log['seed']}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    else:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False)
    return path


def load_log(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    if log.get("format") != LOG_FORMAT:
        raise ValueError(f"không phải log {LOG_FORMAT}")
    return log


def version_warnings(log: dict, catalog, combat) -> list[str]:
    w = []
    if log["catalog_version"] != catalog.version:
        w.append(f"log dùng dữ liệu {log['catalog_version']}, đang nạp {catalog.version}: phát lại có thể lệch")
    if log["combat_version"] != combat.version:
        w.append(f"log dùng giao tranh {log['combat_version']}, đang dùng {combat.version}: phát lại có thể lệch")
    return w


def replay_frames(log: dict, catalog, combat, debug: bool = True):
    """Chạy lại log. Trả về (runner, frames) với frames[t] = (bản sao state sau t bước, info của bước t)."""
    P = log["num_players"]
    cols = [[row[i] for row in log["actions"]] for i in range(P)]
    runner = GameRunner(catalog, combat, [ReplaySeat(c) for c in cols], log["seed"], debug=debug,
                        matchup_row=log.get("matchup_row", True))
    human = set(log.get("human_steps", []))
    frames = [(runner.state.copy(), None)]
    for t in range(len(log["actions"])):
        acts, _ = runner.collect()
        info = runner.advance(acts, t in human)
        frames.append((runner.state.copy(), info))
    return runner, frames
