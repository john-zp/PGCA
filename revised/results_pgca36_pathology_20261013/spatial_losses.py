"""Stable static OT and horizon-consistent finite-state bridge interpolation."""
import torch
import torch.nn.functional as F


def normalize_nonempty(weights):
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid nonnegative spatial weights")
    mass = weights.sum(dim=-1, keepdim=True)
    valid = mass.squeeze(-1) > 0
    return weights[valid] / mass[valid], valid


def spatial_distributions(attention, roi, grid):
    weights = attention.squeeze(1)
    small = F.interpolate(roi, size=grid, mode="nearest").flatten(1)
    _, attention_valid = normalize_nonempty(weights)
    _, roi_valid = normalize_nonempty(small)
    valid = attention_valid & roi_valid
    if not valid.any():
        return None, None, valid
    a = weights[valid] / weights[valid].sum(-1, keepdim=True)
    b = small[valid] / small[valid].sum(-1, keepdim=True)
    return a, b, valid


def log_mass(distribution):
    tiny = torch.finfo(distribution.dtype).tiny
    return distribution.clamp_min(tiny).log().masked_fill(distribution == 0, float("-inf"))


def solve_coupling(a, b, log_kernel, iterations=200):
    if a.shape != b.shape or log_kernel.shape != (a.shape[-1], a.shape[-1]):
        raise ValueError("Incompatible coupling shapes")
    log_a, log_b = log_mass(a), log_mass(b)
    log_u, log_v = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel[None] + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(log_kernel[None] + log_u[:, :, None], dim=1)
    gamma = (log_u[:, :, None] + log_kernel[None] + log_v[:, None, :]).exp()
    return gamma, log_u, log_v


def costs(grid, device, dtype):
    h, w = grid
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=device, dtype=dtype), torch.linspace(0, 1, w, device=device, dtype=dtype), indexing="ij")
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return ((xy[:, None] - xy[None]) ** 2).sum(-1)


def marginal_error(gamma, a, b):
    return torch.maximum((gamma.sum(2) - a).abs().max(), (gamma.sum(1) - b).abs().max())


def bridge(a, b, transition, steps=3, iterations=200):
    n, horizon = transition.shape[0], steps + 1
    if (transition <= 0).any() or not torch.allclose(transition.sum(1), torch.ones(n, device=a.device, dtype=a.dtype), atol=1e-6, rtol=1e-6):
        raise ValueError("Reference transition must be positive and row stochastic")
    powers = [torch.eye(n, device=a.device, dtype=a.dtype)]
    for _ in range(horizon):
        powers.append(powers[-1] @ transition)
    log_reference = -torch.log(torch.tensor(float(n), device=a.device, dtype=a.dtype))
    log_endpoint_kernel = log_reference + powers[horizon].log()
    gamma, log_u, log_v = solve_coupling(a, b, log_endpoint_kernel, iterations)
    states = []
    for time in range(horizon + 1):
        # Forward probability mass uses row-vector propagation; backward potential
        # uses P^(T-t) @ v. No post-normalization hides endpoint or mass errors.
        log_forward = torch.logsumexp((log_u + log_reference)[:, :, None] + log_mass(powers[time])[None], dim=1)
        log_backward = torch.logsumexp(log_mass(powers[horizon - time])[None] + log_v[:, None, :], dim=2)
        states.append((log_forward + log_backward).exp())
    return states, gamma


def alignment(attention, roi, grid, with_path=False, epsilon=.05, iterations=200, beta=.3):
    a, b, valid = spatial_distributions(attention, roi, grid)
    if a is None:
        zero = attention.sum() * 0
        return zero, zero, {"valid_roi": 0, "marginal_error": 0.0, "endpoint_error": 0.0}
    C = costs(grid, a.device, a.dtype)
    gamma, _, _ = solve_coupling(a, b, -C / epsilon, iterations)
    error = marginal_error(gamma, a, b)
    static = beta * (gamma * C).sum((1, 2)).mean()
    path = static * 0
    endpoint_error = torch.zeros((), device=a.device, dtype=a.dtype)
    if with_path:
        transition = torch.softmax(-C / epsilon, dim=1)
        states, bridge_gamma = bridge(a, b, transition, iterations=iterations)
        endpoint_error = torch.maximum((states[0] - a).abs().max(), (states[-1] - b).abs().max())
        error = torch.maximum(error, marginal_error(bridge_gamma, a, b))
        tiny = torch.finfo(a.dtype).tiny
        path = torch.stack([(a * (a.clamp_min(tiny).log() - state.clamp_min(tiny).log())).sum(-1).mean() for state in states[1:-1]]).mean()
    if not torch.isfinite(static + path) or error.detach() > 1e-4 or endpoint_error.detach() > 1e-4:
        raise FloatingPointError("OT/path failed finite or marginal/endpoint tolerance")
    return static, path, {"valid_roi": int(valid.sum()), "marginal_error": float(error.detach()), "endpoint_error": float(endpoint_error.detach())}
