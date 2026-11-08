"""Bọc mạng thành "chính sách": quan sát NumPy vào, hành động NumPy ra.

Việc bốc hành động dùng bộ sinh số NumPy (mẹo Gumbel), nên kết quả tất định theo seed và không phụ
thuộc trạng thái ngẫu nhiên của PyTorch.
"""
from __future__ import annotations

import numpy as np
import torch

from autochess.core.actions import END_TURN

NEG = -1e9


def to_tensors(obs: dict, device="cpu") -> dict:
    return {k: torch.as_tensor(np.ascontiguousarray(v), device=device) for k, v in obs.items()}


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask.bool(), NEG)


class Policy:
    def __init__(self, net, device="cpu", chunk: int = 16384):
        self.net = net
        self.device = torch.device(device)
        self.chunk = chunk
        self.obs_version = getattr(net, "obs_version", 1)    # quan sát bản nào (observe(..., version=))
        self.critic_info = getattr(net, "critic_info", "none")  # khi thu thập để học: observe(critic_info=)

    @torch.no_grad()
    def _forward(self, obs: dict):
        self.net.eval()
        M = len(obs["mask"])
        outs_l, outs_v = [], []
        for i in range(0, M, self.chunk):
            part = {k: v[i:i + self.chunk] for k, v in obs.items()}
            t = to_tensors(part, self.device)
            logits, value = self.net(t)
            outs_l.append(torch.log_softmax(masked_logits(logits, t["mask"]), -1).float().cpu().numpy())
            outs_v.append(value.float().cpu().numpy())
        return np.concatenate(outs_l), np.concatenate(outs_v)

    def act(self, obs: dict, rng: np.random.Generator, greedy: bool = False):
        logp_all, value = self._forward(obs)
        if greedy:
            a = logp_all.argmax(-1)
        else:
            u = rng.random(logp_all.shape)
            gumbel = -np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))
            a = np.where(obs["mask"], logp_all + gumbel, -np.inf).argmax(-1)
        logp = np.take_along_axis(logp_all, a[:, None], -1)[:, 0]
        return a.astype(np.int64), logp.astype(np.float32), value.astype(np.float32)

    def value(self, obs: dict):
        return self._forward(obs)[1].astype(np.float32)


def load_checkpoint(path: str, cat, device="cpu", verbose: bool = True):
    """Nạp mạng từ checkpoint của train.py / bc.py, ghép theo tên nếu file dữ liệu đã đổi.
    Trả về (net, cfg)."""
    from .compat import load_compatible
    net, cfg, rep = load_compatible(path, cat, device)
    if verbose and (rep.changed or rep.warnings):
        print(rep)
    return net, cfg
