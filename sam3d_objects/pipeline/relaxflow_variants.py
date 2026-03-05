import math
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


# --------------------------- Gating schedules --------------------------- #
def gating_linear_cutoff(step_idx: int, total_steps: int, alpha0: float = 1.0, mid_ratio: float = 0.4) -> float:
    """Linear decay with early cutoff (matches the original default)."""
    total_steps = max(int(total_steps), 1)
    T_mid = int(mid_ratio * total_steps)
    if step_idx <= T_mid:
        return alpha0 * (1 - step_idx / float(total_steps))
    return 0.0


def gating_cosine(step_idx: int, total_steps: int, alpha0: float = 1.0, mid_ratio: float = 0.6) -> float:
    """Half-cosine decay; mid_ratio controls the cutoff point."""
    total_steps = max(int(total_steps), 1)
    cutoff = max(int(total_steps * mid_ratio), 1)
    if step_idx >= cutoff:
        return 0.0
    phase = math.pi * step_idx / float(cutoff)
    return alpha0 * 0.5 * (1.0 + math.cos(phase))


def gating_exponential(step_idx: int, total_steps: int, alpha0: float = 1.0, decay: float = 4.0, floor: float = 0.0) -> float:
    """Exponential decay that never goes below a small floor."""
    total_steps = max(int(total_steps), 1)
    ratio = step_idx / float(total_steps)
    return max(float(alpha0 * math.exp(-decay * ratio)), floor)


def gating_sigmoid(step_idx: int, total_steps: int, alpha0: float = 1.0, center: float = 0.5, sharpness: float = 10.0) -> float:
    """Sigmoid gate: starts flat, drops sharply around `center`."""
    total_steps = max(int(total_steps), 1)
    pos = step_idx / float(total_steps)
    return float(alpha0 / (1.0 + math.exp(sharpness * (pos - center))))


def gating_two_stage(step_idx: int, total_steps: int, alpha0: float = 1.0, warmup: float = 0.1, hold: float = 0.2) -> float:
    """
    Warmup -> hold -> linear decay.
    warmup/hold are fractions of total_steps.
    """
    total_steps = max(int(total_steps), 1)
    warm_steps = int(total_steps * warmup)
    hold_steps = int(total_steps * hold)
    decay_start = warm_steps + hold_steps

    if step_idx < warm_steps:
        return alpha0 * (step_idx + 1) / max(warm_steps, 1)
    if step_idx < decay_start:
        return alpha0
    remain = total_steps - decay_start
    if remain <= 0:
        return 0.0
    progress = (step_idx - decay_start) / float(remain)
    return alpha0 * max(1.0 - progress, 0.0)


def gating_inverse_sqrt(step_idx: int, total_steps: int, alpha0: float = 1.0, gamma: float = 5.0) -> float:
    """Inverse sqrt decay that keeps a gentle tail."""
    total_steps = max(int(total_steps), 1)
    ratio = step_idx / float(total_steps)
    return float(alpha0 / math.sqrt(1.0 + gamma * ratio))


GATING_SCHEDULES: Dict[str, Callable[..., float]] = {
    "linear_cutoff": gating_linear_cutoff,
    "cosine": gating_cosine,
    "exponential": gating_exponential,
    "sigmoid": gating_sigmoid,
    "two_stage": gating_two_stage,
    "inverse_sqrt": gating_inverse_sqrt,
}


def resolve_gating_schedule(
    schedule: Optional[Callable[[int, int], float]] = None,
    schedule_name: Optional[str] = None,
    fallback: Optional[Callable[[int, int], float]] = None,
) -> Callable[[int, int], float]:
    if schedule is not None and not isinstance(schedule, str):
        return schedule
    name = schedule_name or (schedule if isinstance(schedule, str) else None)
    if name:
        if name not in GATING_SCHEDULES:
            raise ValueError(f"Unknown gating schedule '{name}'. Available: {list(GATING_SCHEDULES)}")
        return GATING_SCHEDULES[name]
    if fallback is not None:
        return fallback
    return gating_linear_cutoff


# --------------------------- Flow blend variants --------------------------- #
def blend_linear(e_obs: torch.Tensor, e_prior: torch.Tensor, alpha: float, prior_weight: float, geometry_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
    # Check average magnitude (energy) of the update vectors
    mag_obs = e_obs.norm(p=2, dim=-1).mean().item()
    mag_prior = e_prior.norm(p=2, dim=-1).mean().item()

    print(f"DEBUG: e_obs magnitude: {mag_obs:.4f}")
    print(f"DEBUG: e_prior magnitude: {mag_prior:.4f}")
    if geometry_mask is not None:
        mask = geometry_mask.to(device=e_obs.device, dtype=e_obs.dtype)
        if mask.dim() == e_obs.dim() - 1:
            mask = mask.unsqueeze(-1)
        mask = mask.clamp(0.0, 1.0)
    alpha_t = torch.as_tensor(alpha, device=e_obs.device, dtype=e_obs.dtype)
    prior_w = torch.as_tensor(prior_weight, device=e_obs.device, dtype=e_obs.dtype)
    blended = (1.0 - alpha_t) * e_obs + alpha_t * prior_w * e_prior
    if geometry_mask is not None:
        return mask * e_obs + (1.0 - mask) * blended
    else:
        return blended


def blend_residual_pull(e_obs: torch.Tensor, e_prior: torch.Tensor, alpha: float, prior_weight: float, **_) -> torch.Tensor:
    """Moves along the residual direction; prior_weight scales the residual."""
    alpha_t = torch.as_tensor(alpha, device=e_obs.device, dtype=e_obs.dtype)
    prior_w = torch.as_tensor(prior_weight, device=e_obs.device, dtype=e_obs.dtype)
    return e_obs + alpha_t * prior_w * (e_prior - e_obs)


def blend_norm_weighted(
    e_obs: torch.Tensor,
    e_prior: torch.Tensor,
    alpha: float,
    prior_weight: float,
    norm_eps: float = 1e-4,
    **_,
) -> torch.Tensor:
    """Normalize by magnitude to prevent domination from high-variance branch."""
    alpha_t = torch.as_tensor(alpha, device=e_obs.device, dtype=e_obs.dtype)
    prior_w = torch.as_tensor(prior_weight, device=e_obs.device, dtype=e_obs.dtype)
    norm_obs = torch.linalg.norm(e_obs, dim=-1, keepdim=True).clamp_min(norm_eps)
    norm_prior = torch.linalg.norm(e_prior, dim=-1, keepdim=True).clamp_min(norm_eps)
    w_obs = 1.0 / norm_obs
    w_prior = prior_w / norm_prior
    norm = w_obs + w_prior
    w_obs = w_obs / norm
    w_prior = w_prior / norm
    return (1.0 - alpha_t) * w_obs * e_obs + alpha_t * w_prior * e_prior


def blend_cosine_guided(
    e_obs: torch.Tensor,
    e_prior: torch.Tensor,
    alpha: float,
    prior_weight: float,
    sim_eps: float = 1e-4,
    **_,
) -> torch.Tensor:
    """Use cosine similarity to modulate prior strength."""
    alpha_t = torch.as_tensor(alpha, device=e_obs.device, dtype=e_obs.dtype)
    prior_w = torch.as_tensor(prior_weight, device=e_obs.device, dtype=e_obs.dtype)
    sim = F.cosine_similarity(e_obs, e_prior, dim=-1, eps=sim_eps)
    gain = ((sim + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(-1)
    return (1.0 - alpha_t) * e_obs + alpha_t * prior_w * gain * e_prior


def blend_softmax_moe(
    e_obs: torch.Tensor,
    e_prior: torch.Tensor,
    alpha: float,
    prior_weight: float,
    temperature: float = 1.0,
    norm_eps: float = 1e-4,
    **_,
) -> torch.Tensor:
    """Mixture-of-experts on branch magnitudes."""
    alpha_t = torch.as_tensor(alpha, device=e_obs.device, dtype=e_obs.dtype)
    prior_w = torch.as_tensor(prior_weight, device=e_obs.device, dtype=e_obs.dtype)
    mag_obs = torch.linalg.norm(e_obs, dim=-1, keepdim=True).clamp_min(norm_eps)
    mag_prior = torch.linalg.norm(e_prior, dim=-1, keepdim=True).clamp_min(norm_eps)
    logits = torch.stack([mag_obs, mag_prior], dim=-1) / max(temperature, 1e-6)
    weights = torch.softmax(logits, dim=-1)
    w_obs = weights[..., 0:1]
    w_prior = weights[..., 1:2] * prior_w
    norm = (w_obs + w_prior).clamp_min(1e-6)
    w_obs = w_obs / norm
    w_prior = w_prior / norm
    return (1.0 - alpha_t) * w_obs * e_obs + alpha_t * w_prior * e_prior


def blend_tanh_clipped(
    e_obs: torch.Tensor,
    e_prior: torch.Tensor,
    alpha: float,
    prior_weight: float,
    clip_value: float = 2.0,
    **_,
) -> torch.Tensor:
    """Clamps prior velocity to stabilize extreme logits."""
    alpha_t = torch.as_tensor(alpha, device=e_obs.device, dtype=e_obs.dtype)
    prior_w = torch.as_tensor(prior_weight, device=e_obs.device, dtype=e_obs.dtype)
    clipped_prior = torch.tanh(e_prior / clip_value) * clip_value
    return (1.0 - alpha_t) * e_obs + alpha_t * prior_w * clipped_prior


FLOW_BLEND_FNS: Dict[str, Callable[..., torch.Tensor]] = {
    "linear": blend_linear,
    "residual_pull": blend_residual_pull,
    "norm_weighted": blend_norm_weighted,
    "cosine_guided": blend_cosine_guided,
    "softmax_moe": blend_softmax_moe,
    "tanh_clipped": blend_tanh_clipped,
}


def resolve_flow_blend(
    blend: Optional[Callable[..., torch.Tensor]] = None,
    blend_name: Optional[str] = None,
    fallback: Optional[Callable[..., torch.Tensor]] = None,
) -> Callable[..., torch.Tensor]:
    if blend is not None and not isinstance(blend, str):
        return blend
    name = blend_name or (blend if isinstance(blend, str) else None)
    if name:
        if name not in FLOW_BLEND_FNS:
            raise ValueError(f"Unknown flow blend '{name}'. Available: {list(FLOW_BLEND_FNS)}")
        return FLOW_BLEND_FNS[name]
    if fallback is not None:
        return fallback
    return blend_linear
