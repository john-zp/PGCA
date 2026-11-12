from __future__ import annotations

import torch


def pairwise_sqdist(xy: torch.Tensor) -> torch.Tensor:
    diff = xy[:, None, :] - xy[None, :, :]
    return (diff ** 2).sum(dim=-1)


def sinkhorn(a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, eps: float = 0.05, iters: int = 60):
    B, N = a.shape
    K = torch.exp(-C / eps).clamp_min(1e-12)
    u = torch.ones(B, N, device=a.device) / N
    v = torch.ones(B, N, device=a.device) / N
    KT = K.t()
    for _ in range(iters):
        u = a / (v @ KT + 1e-12)
        v = b / (u @ K + 1e-12)
        u = u.clamp_min(1e-12)
        v = v.clamp_min(1e-12)
    gamma = u.unsqueeze(2) * K.unsqueeze(0) * v.unsqueeze(1)
    return gamma, u, v, K


def sb_entropic_interpolation(u: torch.Tensor, v: torch.Tensor, K: torch.Tensor, steps: int = 3):
    P = K / (K.sum(dim=1, keepdim=True) + 1e-12)
    rhos = []
    T = steps + 1
    for m in range(1, T):
        f_t = u @ torch.matrix_power(P, m)
        g_t = v @ torch.matrix_power(P, T - m)
        rho = (f_t * g_t).clamp_min(1e-12)
        rho = rho / (rho.sum(dim=1, keepdim=True) + 1e-12)
        rhos.append(rho)
    return rhos


def sb_losses(attn: torch.Tensor, roi: torch.Tensor, xy: torch.Tensor, eps: float = 0.05, iters: int = 60, steps: int = 3, beta: float = 0.3):
    C = pairwise_sqdist(xy)
    gamma, u, v, K = sinkhorn(attn, roi, C, eps=eps, iters=iters)
    transport_cost = (gamma * C).sum(dim=(1, 2))
    loss_sb_static = (beta * transport_cost).mean()

    rhos = sb_entropic_interpolation(u, v, K, steps=steps)
    loss_path = 0.0
    for rho in rhos:
        kl = (attn * (attn.clamp_min(1e-12).log() - rho.clamp_min(1e-12).log())).sum(dim=1).mean()
        loss_path += kl / max(len(rhos), 1)
    return loss_sb_static, loss_path
