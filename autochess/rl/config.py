"""Cấu hình huấn luyện: giá trị mặc định + các file YAML ghi đè theo thứ tự + --set khóa=giá trị.

    cfg = load_config(["configs/p2_bots.yaml", "configs/laptop.yaml"], ["ppo.lr=1e-4"])
"""
from __future__ import annotations

import copy
import json

DEFAULTS = {
    "name": "run",
    "num_players": 2,
    "seed": 1,
    "total_decisions": 20_000_000,    # dừng khi số bước có quyết định thật đạt mức này
    "device": "auto",                 # auto | cpu | cuda
    "threads": 0,                     # số luồng PyTorch (0 = mặc định)
    "workers": 0,                     # > 0: thu thập bằng ngần ấy tiến trình con (rl/vec.py), cho PC nhiều nhân
    "run_dir": "runs",
    "catalog": None,                  # file dữ liệu (None = autochess/data/catalog_v1.json)
    "env": {"num_games": 256, "steps": 128, "matchup_row": True,
            "mix": {},                        # {số người: tỷ lệ ván}, vd {4: 0.25} khi num_players = 8
            "skip_waiting": False},           # bỏ các bước chờ lặp lại (nhanh hơn nhiều ở ván 8 người)
    "reward": {"gamma_round": 0.995, "lam": 0.95, "beta": 0.1,
               "churn_cost": 0.0},             # thưởng âm cho mỗi lần bán/chuyển quân (vd 0.002)
    "ppo": {"lr": 3.0e-4, "anneal_lr": True, "update_epochs": 4, "minibatch_size": 2048,
            "clip": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
            "ent_coef_final": None,         # nếu đặt: hệ số entropy giảm tuyến tính từ ent_coef về giá trị này
            "norm_adv": True, "target_kl": None,
            "value_warmup": 0,              # số lần cập nhật đầu chỉ học critic (sau khi khởi tạo từ rl.bc)
            "freeze_policy_in_warmup": True,  # khi đó chỉ cập nhật tham số riêng của critic, chính sách giữ nguyên
            "lr_warmup": 0},                # sau value_warmup: tốc độ học tăng dần từ ~0 lên lr trong ngần ấy lần
    # entity2 = mạng bản 2 (dùng lại được khi đổi file dữ liệu); entity = bản 1; flat = đối chứng
    "net": {"kind": "entity2", "emb": 16, "ent": 64, "hid": 256, "self_hid": 128, "card_hid": 32,
            "separate_critic": False, "mixer": "none", "mixer_layers": 2, "mixer_heads": 4,
            "res_l2": 0.01,
            "opp_bench": False,               # giai đoạn 3: thấy hàng chờ đối thủ (quan sát bản 3)
            "critic_info": "none"},           # none | pairing | pairing+dice: thông tin ẩn chỉ cho critic
    "opponents": {"names": ["tempo", "hyper", "roll7", "fast9"], "prioritize": True,
                  "power": 2.0, "floor": 0.05, "selfplay_prob": 0.0,
                  "agent_seats": [1, 1]},      # ván không tự đấu: agent ngồi từ ... đến ... ghế
    "eval": {"every": 25, "games": 400, "B": 256, "opponents": ["tempo", "hyper", "roll7", "fast9", "random"],
             "matrix_games": 400, "seed": 12345,
             "anchors": {}},                 # {tên: checkpoint}: bản neo đóng băng, đấu thêm khi đánh giá
    # giai đoạn 2: kho bản chụp của agent (dùng khi seeds khác rỗng hoặc snapshot_every > 0 và có tự đấu)
    "league": {"seeds": {},                  # {tên: checkpoint}: bản neo trong kho đối thủ, không bị bỏ
               "snapshot_every": 0,           # chụp mạng vào kho mỗi ngần ấy lần cập nhật (0 = tắt)
               "max_snapshots": 10,           # số bản chụp tối đa (không tính bản neo)
               "min_games": 100,              # khi kho đầy: chỉ thay bản đã đấu ít nhất ngần ấy ván
               "group_weights": {"bot": 0.3, "policy": 0.7}},   # chia phần không tự đấu giữa bot luật / bản cũ
    "save_every": 25,
    "init": None,                     # checkpoint để khởi tạo mạng (vd từ rl.bc), chỉ khi chạy mới
    "bc": {"samples": 2_000_000, "steps": 64, "epochs": 2, "lr": 1.0e-3, "eval_every": 40,
           "eval_games": 100, "bots": ["tempo", "hyper", "roll7", "fast9"],
           "export": None},                  # nếu đặt: chép checkpoint cuối sang đường dẫn này (cố định, cho config sau)
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(text: str):
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        try:
            return json.loads(text)
        except ValueError:
            return text


def _load_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("Cần PyYAML để đọc file .yaml: pip install pyyaml") from e
    return yaml.safe_load(text) or {}


def load_config(paths=(), sets=()) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    for p in paths:
        cfg = deep_merge(cfg, _load_file(p))
    return apply_sets(cfg, sets)


def apply_sets(cfg: dict, sets=()) -> dict:
    """Ghi đè từng khóa dạng a.b.c=giá trị (giá trị đọc theo cú pháp YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in sets:
        key, _, val = item.partition("=")
        d = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = _parse_value(val)
    return cfg
