"""Cập nhật PPO (bám theo PPO rời rạc của CleanRL), có mặt nạ hành động.

Mẫu đầu vào là các bước "active" (ghế còn trong ván). Phần chính sách và entropy chỉ tính trên các
bước "decision" (có hơn một hành động hợp lệ); phần critic tính trên mọi bước active.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .policy import to_tensors, masked_logits
from .nets import critic_parameters


def flatten_rollout(buf: dict, adv: np.ndarray, ret: np.ndarray) -> dict:
    """Từ bộ đệm [T, B, S, ...] lấy ra các mẫu active, trải phẳng thành [M, ...]. Nếu bộ đệm có "key"
    (Rollout(skip_waiting=True)) thì chỉ lấy các bước chạy mạng thật (bỏ các bước chờ lặp lại)."""
    act = buf["active"] & buf["key"] if "key" in buf else buf["active"]
    out = {"obs": {k: v[act] for k, v in buf["obs"].items()}}
    for k in ("actions", "logp", "values", "decision"):
        out[k] = buf[k][act]
    out["adv"] = adv.reshape(act.shape)[act]
    out["returns"] = ret.reshape(act.shape)[act]
    return out


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = np.var(target)
    return float("nan") if var == 0 else float(1 - np.var(target - pred) / var)


def ppo_update(net: nn.Module, optimizer, batch: dict, cfg: dict, rng: np.random.Generator,
               device="cpu") -> dict:
    M = len(batch["actions"])
    mb = int(cfg["minibatch_size"])
    eps, ent_coef, vf_coef = cfg["clip"], cfg["ent_coef"], cfg["vf_coef"]
    pg_coef = cfg.get("pg_coef", 1.0)        # 0 khi "khởi động critic": chỉ học critic, giữ nguyên chính sách
    stats = dict(pg_loss=[], v_loss=[], entropy=[], approx_kl=[], clipfrac=[], aux_loss=[])
    # khởi động critic: chỉ cập nhật tham số riêng của critic để chính sách (đã học bắt chước) không bị kéo đi
    frozen = []
    if pg_coef == 0 and cfg.get("freeze_policy_in_warmup", True):
        keep = {id(p) for p in critic_parameters(net)}
        frozen = [p for p in net.parameters() if id(p) not in keep]
    aux_fn = getattr(net, "aux_loss", None)
    net.train()
    stop = False
    for epoch in range(int(cfg["update_epochs"])):
        perm = rng.permutation(M)
        for start in range(0, M, mb):
            idx = perm[start:start + mb]
            obs = to_tensors({k: v[idx] for k, v in batch["obs"].items()}, device)
            a = torch.as_tensor(batch["actions"][idx], device=device)
            old_logp = torch.as_tensor(batch["logp"][idx], device=device)
            adv = torch.as_tensor(batch["adv"][idx], device=device)
            ret = torch.as_tensor(batch["returns"][idx], device=device)
            dec = torch.as_tensor(batch["decision"][idx], device=device).float()
            n_dec = dec.sum().clamp(min=1.0)

            logits, value = net(obs)
            logp_all = torch.log_softmax(masked_logits(logits, obs["mask"]), -1)
            new_logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
            entropy = -(logp_all.exp() * logp_all).sum(-1)
            logratio = (new_logp - old_logp) * dec
            ratio = logratio.exp()

            if cfg.get("norm_adv", True):
                mean = (adv * dec).sum() / n_dec
                std = (((adv - mean) ** 2) * dec).sum().div(n_dec).sqrt()
                adv = (adv - mean) / (std + 1e-8)
            pg1 = -adv * ratio
            pg2 = -adv * torch.clamp(ratio, 1 - eps, 1 + eps)
            pg_loss = (torch.max(pg1, pg2) * dec).sum() / n_dec
            v_loss = 0.5 * ((value - ret) ** 2).mean()
            ent = (entropy * dec).sum() / n_dec
            loss = pg_coef * (pg_loss - ent_coef * ent) + vf_coef * v_loss
            aux = aux_fn() if aux_fn is not None and not frozen else None
            if aux is not None:
                loss = loss + aux

            optimizer.zero_grad()
            loss.backward()
            for p in frozen:
                p.grad = None      # Adam bỏ qua tham số không có gradient
            nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            with torch.no_grad():
                kl = (((ratio - 1) - logratio) * dec).sum() / n_dec
                cf = ((((ratio - 1).abs() > eps).float()) * dec).sum() / n_dec
            stats["pg_loss"].append(pg_loss.item())
            stats["v_loss"].append(v_loss.item())
            stats["entropy"].append(ent.item())
            stats["approx_kl"].append(kl.item())
            stats["clipfrac"].append(cf.item())
            stats["aux_loss"].append(aux.item() if aux is not None else 0.0)
            target_kl = cfg.get("target_kl")
            if target_kl and kl.item() > 1.5 * target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    out["explained_variance"] = explained_variance(batch["values"], batch["returns"])
    out["samples"] = M
    out["decisions"] = int(batch["decision"].sum())
    out["early_stop"] = stop
    return out
