"""RelaxFlow pipeline extensions for Trellis. Author: Jiayin Zhu"""

import inspect
import math
import os
import sys
import types
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
from tqdm import tqdm

TRELLIS_ROOT = os.environ.get("TRELLIS_ROOT", "/home/jiayin/TRELLIS")
if TRELLIS_ROOT not in sys.path and os.path.isdir(TRELLIS_ROOT):
    sys.path.insert(0, TRELLIS_ROOT)

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.modules.attention.modules import MultiHeadAttention

import trellis.modules.attention.full_attn as full_attn
import trellis.modules.attention.modules as attn_modules
import trellis.modules.sparse.attention.full_attn as sparse_full_attn
import trellis.modules.sparse.attention.modules as sparse_attn_modules

from sam3d_objects.pipeline.relaxflow_variants import (
    FLOW_BLEND_FNS,
    resolve_flow_blend,
    resolve_gating_schedule,
)


def gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def _cast_for_blur(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float(), True
    return x, False


def gaussian_blur2d(logits: torch.Tensor, sigma: float = 2.5) -> torch.Tensor:
    if sigma <= 0:
        return logits
    bsz, head, q_len, k_len = logits.shape
    logits_f, cast_back = _cast_for_blur(logits)
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel = gaussian_kernel1d(kernel_size, sigma, logits_f.device).to(dtype=logits_f.dtype)
    logits_f = logits_f.view(-1, 1, q_len, k_len)
    padding = kernel_size // 2
    out = F.conv2d(logits_f, kernel[None, None, :, None], padding=(padding, 0), groups=1)
    out = F.conv2d(out, kernel[None, None, None, :], padding=(0, padding), groups=1)
    out = out.view(bsz, head, q_len, k_len)
    if cast_back:
        out = out.to(dtype=logits.dtype)
    return out


def gaussian_blur_logits(
    logits: torch.Tensor,
    sigma: float = 2.5,
    *,
    blur_q: bool = True,
    blur_k: bool = True,
) -> torch.Tensor:
    if sigma <= 0 or not (blur_q or blur_k):
        return logits
    bsz, head, q_len, k_len = logits.shape
    logits_f, cast_back = _cast_for_blur(logits)
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel = gaussian_kernel1d(kernel_size, sigma, logits_f.device).to(dtype=logits_f.dtype)
    out = logits_f.view(-1, 1, q_len, k_len)
    padding = kernel_size // 2
    if blur_q:
        out = F.conv2d(out, kernel[None, None, :, None], padding=(padding, 0), groups=1)
    if blur_k:
        out = F.conv2d(out, kernel[None, None, None, :], padding=(0, padding), groups=1)
    out = out.view(bsz, head, q_len, k_len)
    if cast_back:
        out = out.to(dtype=logits.dtype)
    return out


def gaussian_blur_feature_dim(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma is None or sigma <= 0:
        return x
    if x.dim() < 1:
        return x
    feat_dim = x.shape[-1]
    if feat_dim <= 1:
        return x
    kernel_size = int(2 * round(3 * float(sigma)) + 1)
    if kernel_size > feat_dim:
        kernel_size = feat_dim if (feat_dim % 2 == 1) else max(feat_dim - 1, 1)
    if kernel_size <= 1:
        return x
    kernel = gaussian_kernel1d(kernel_size, float(sigma), x.device).to(dtype=x.dtype)
    x_flat = x.reshape(-1, 1, feat_dim)
    out = F.conv1d(x_flat, kernel.view(1, 1, -1), padding=kernel_size // 2)
    return out.reshape(*x.shape)


def add_feature_noise(x: torch.Tensor, noise_type: str, strength: float) -> torch.Tensor:
    if strength is None or strength <= 0:
        return x
    noise_type = (noise_type or "gaussian").lower()
    if noise_type == "gaussian":
        noise = torch.randn_like(x) * float(strength)
    elif noise_type == "uniform":
        noise = (torch.rand_like(x) * 2.0 - 1.0) * float(strength)
    else:
        raise ValueError(f"Unknown feat_noise_type: {noise_type}")
    return x + noise


class RELAXFLOWAttnBlurContext:
    def __init__(
        self,
        model: torch.nn.Module,
        sigma: float,
        attn_type: str = "self",
        *,
        feat_blur_type: str = "none",
        feat_blur_sigma: float = 1.0,
        feat_noise_type: str = "gaussian",
        feat_noise_strength: float = 0.0,
    ):
        self.model = model
        self.sigma = sigma
        self.attn_type = attn_type
        self.feat_blur_type = (feat_blur_type or "none").lower()
        self.feat_blur_sigma = feat_blur_sigma
        self.feat_noise_type = (feat_noise_type or "gaussian").lower()
        self.feat_noise_strength = feat_noise_strength
        self._patches = []
        self._sdpa_patches = []
        self._sparse_patches = []
        self._active = False

    def _maybe_perturb(self, v: torch.Tensor) -> torch.Tensor:
        if self.feat_blur_type == "blur":
            return gaussian_blur_feature_dim(v, self.feat_blur_sigma)
        if self.feat_blur_type == "noise":
            return add_feature_noise(v, self.feat_noise_type, self.feat_noise_strength)
        return v

    def __enter__(self):
        do_logit_blur = self.sigma is not None and self.sigma > 0
        do_feat_blur = self.feat_blur_type == "blur" and self.feat_blur_sigma > 0
        do_feat_noise = self.feat_blur_type == "noise" and self.feat_noise_strength > 0
        if not (do_logit_blur or do_feat_blur or do_feat_noise):
            self._active = False
            return self

        self._active = True
        self._patches = []
        self._sdpa_patches = []
        self._sparse_patches = []

        if do_feat_blur or do_feat_noise:
            orig_sdpa_full = full_attn.scaled_dot_product_attention
            orig_sdpa_mod = attn_modules.scaled_dot_product_attention

            def sdpa_wrapper(*args, **kwargs):
                num_all_args = len(args) + len(kwargs)
                if num_all_args == 1:
                    qkv = args[0] if len(args) > 0 else kwargs["qkv"]
                    q, k, v = qkv.unbind(dim=2)
                    v = self._maybe_perturb(v)
                    qkv = torch.stack([q, k, v], dim=2)
                    if len(args) > 0:
                        return orig_sdpa_full(qkv)
                    return orig_sdpa_full(qkv=qkv)
                if num_all_args == 2:
                    q = args[0] if len(args) > 0 else kwargs["q"]
                    kv = args[1] if len(args) > 1 else kwargs["kv"]
                    k, v = kv.unbind(dim=2)
                    v = self._maybe_perturb(v)
                    kv = torch.stack([k, v], dim=2)
                    if len(args) >= 2:
                        return orig_sdpa_full(q, kv)
                    if len(args) == 1:
                        return orig_sdpa_full(q, kv=kv)
                    return orig_sdpa_full(q=q, kv=kv)
                if num_all_args == 3:
                    q = args[0] if len(args) > 0 else kwargs["q"]
                    k = args[1] if len(args) > 1 else kwargs["k"]
                    v = args[2] if len(args) > 2 else kwargs["v"]
                    v = self._maybe_perturb(v)
                    if len(args) >= 3:
                        return orig_sdpa_full(q, k, v)
                    if len(args) == 2:
                        return orig_sdpa_full(q, k, v=v)
                    if len(args) == 1:
                        return orig_sdpa_full(q, k=k, v=v)
                    return orig_sdpa_full(q=q, k=k, v=v)
                return orig_sdpa_full(*args, **kwargs)

            full_attn.scaled_dot_product_attention = sdpa_wrapper
            attn_modules.scaled_dot_product_attention = sdpa_wrapper
            self._sdpa_patches.append((full_attn, "scaled_dot_product_attention", orig_sdpa_full))
            self._sdpa_patches.append((attn_modules, "scaled_dot_product_attention", orig_sdpa_mod))

            orig_sparse_sdpa_full = sparse_full_attn.sparse_scaled_dot_product_attention
            orig_sparse_sdpa_mod = sparse_attn_modules.sparse_scaled_dot_product_attention

            def sparse_sdpa_wrapper(*args, **kwargs):
                num_all_args = len(args) + len(kwargs)
                if num_all_args == 1:
                    qkv = args[0] if len(args) > 0 else kwargs["qkv"]
                    try:
                        feats = qkv.feats
                        v = feats[:, 2]
                        v = self._maybe_perturb(v)
                        feats_new = feats.clone()
                        feats_new[:, 2] = v
                        qkv = qkv.replace(feats_new)
                    except Exception:
                        pass
                    if len(args) > 0:
                        return orig_sparse_sdpa_full(qkv)
                    return orig_sparse_sdpa_full(qkv=qkv)
                if num_all_args == 2:
                    q = args[0] if len(args) > 0 else kwargs["q"]
                    kv = args[1] if len(args) > 1 else kwargs["kv"]
                    try:
                        if hasattr(kv, "feats"):
                            feats = kv.feats
                            v = feats[:, 1]
                            v = self._maybe_perturb(v)
                            feats_new = feats.clone()
                            feats_new[:, 1] = v
                            kv = kv.replace(feats_new)
                        elif torch.is_tensor(kv) and kv.dim() == 5 and kv.shape[2] == 2:
                            k = kv[:, :, 0]
                            v = kv[:, :, 1]
                            v = self._maybe_perturb(v)
                            kv = torch.stack([k, v], dim=2)
                    except Exception:
                        pass
                    if len(args) >= 2:
                        return orig_sparse_sdpa_full(q, kv)
                    if len(args) == 1:
                        return orig_sparse_sdpa_full(q, kv=kv)
                    return orig_sparse_sdpa_full(q=q, kv=kv)
                if num_all_args == 3:
                    q = args[0] if len(args) > 0 else kwargs["q"]
                    k = args[1] if len(args) > 1 else kwargs["k"]
                    v = args[2] if len(args) > 2 else kwargs["v"]
                    try:
                        vv = v.feats if hasattr(v, "feats") else v
                        v_new = self._maybe_perturb(vv)
                        if hasattr(v, "replace"):
                            v = v.replace(v_new)
                        else:
                            v = v_new
                    except Exception:
                        pass
                    if len(args) >= 3:
                        return orig_sparse_sdpa_full(q, k, v)
                    if len(args) == 2:
                        return orig_sparse_sdpa_full(q, k, v=v)
                    if len(args) == 1:
                        return orig_sparse_sdpa_full(q, k=k, v=v)
                    return orig_sparse_sdpa_full(q=q, k=k, v=v)
                return orig_sparse_sdpa_full(*args, **kwargs)

            sparse_full_attn.sparse_scaled_dot_product_attention = sparse_sdpa_wrapper
            sparse_attn_modules.sparse_scaled_dot_product_attention = sparse_sdpa_wrapper
            self._sparse_patches.append(
                (sparse_full_attn, "sparse_scaled_dot_product_attention", orig_sparse_sdpa_full)
            )
            self._sparse_patches.append(
                (sparse_attn_modules, "sparse_scaled_dot_product_attention", orig_sparse_sdpa_mod)
            )

        if do_logit_blur or do_feat_blur or do_feat_noise:
            for module in self.model.modules():
                if not isinstance(module, MultiHeadAttention):
                    continue
                attn_kind = getattr(module, "_type", None)
                if self.attn_type not in ("both", attn_kind):
                    continue

                orig_forward = module.forward

                def blur_forward(this, x, context=None, indices=None):
                    bsz, q_len, _ = x.shape
                    if this._type == "self":
                        qkv = this.to_qkv(x)
                        qkv = qkv.reshape(bsz, q_len, 3, this.num_heads, -1)
                        if this.use_rope:
                            q, k, v = qkv.unbind(dim=2)
                            q, k = this.rope(q, k, indices)
                            qkv = torch.stack([q, k, v], dim=2)
                        q, k, v = qkv.unbind(dim=2)
                    else:
                        assert context is not None, "Cross-attention requires context"
                        kv_len = context.shape[1]
                        q = this.to_q(x)
                        kv = this.to_kv(context)
                        q = q.reshape(bsz, q_len, this.num_heads, -1)
                        kv = kv.reshape(bsz, kv_len, 2, this.num_heads, -1)
                        k, v = kv.unbind(dim=2)

                    if this.qk_rms_norm:
                        q = this.q_rms_norm(q)
                        k = this.k_rms_norm(k)

                    q_t = q.permute(0, 2, 1, 3)
                    k_t = k.permute(0, 2, 1, 3)
                    v_t = v.permute(0, 2, 1, 3)
                    scale = 1.0 / math.sqrt(this.head_dim)
                    attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
                    if do_logit_blur:
                        attn_scores = gaussian_blur2d(attn_scores, self.sigma)
                    attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
                    attn_probs = torch.softmax(attn_scores, dim=-1)
                    if do_feat_blur or do_feat_noise:
                        v_t = self._maybe_perturb(v_t)
                    out = torch.matmul(attn_probs, v_t)
                    out = out.permute(0, 2, 1, 3).reshape(bsz, q_len, -1)
                    return this.to_out(out)

                module.forward = types.MethodType(blur_forward, module)
                self._patches.append((module, orig_forward))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._active:
            return
        for module, orig in self._patches:
            module.forward = orig
        self._patches = []
        for mod, attr, orig in self._sdpa_patches:
            setattr(mod, attr, orig)
        self._sdpa_patches = []
        for mod, attr, orig in self._sparse_patches:
            setattr(mod, attr, orig)
        self._sparse_patches = []
        self._active = False


def _clone_noise(noise: Union[torch.Tensor, sp.SparseTensor]) -> Union[torch.Tensor, sp.SparseTensor]:
    if isinstance(noise, sp.SparseTensor):
        return noise.replace(noise.feats.clone(), noise.coords.clone())
    return noise.clone()


class TrellisImageTo3DPipelineRelaxFlow(TrellisImageTo3DPipeline):
    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipelineRelaxFlow":
        base = TrellisImageTo3DPipeline.from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipelineRelaxFlow()
        new_pipeline.__dict__ = base.__dict__
        return new_pipeline

    def _build_alpha_schedule(
        self, steps: int, gating_schedule: Callable[[int, int], float], gating_args: dict
    ) -> List[float]:
        steps = max(int(steps), 1)
        alphas: List[float] = []
        sig = inspect.signature(gating_schedule)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_var_kwargs:
            filtered_args = gating_args
        else:
            filtered_args = {k: v for k, v in gating_args.items() if k in sig.parameters}
        for idx in range(steps):
            alpha_val = gating_schedule(idx, steps, **filtered_args)
            alpha_float = float(alpha_val)
            alpha_clamped = max(0.0, min(1.0, alpha_float))
            alphas.append(alpha_clamped)
        return alphas

    def _pool_prior_tokens(
        self,
        tokens_list: List[torch.Tensor],
        mode: str,
        temperature: float,
        agreement_boost: float,
    ) -> torch.Tensor:
        if len(tokens_list) == 1:
            return tokens_list[0]
        mode = (mode or "concat").lower()
        if mode == "concat":
            return torch.cat(tokens_list, dim=1)
        tokens = torch.stack(tokens_list, dim=0)
        if mode == "mean":
            return tokens.mean(dim=0)
        if mode == "consensus":
            mean = tokens.mean(dim=0, keepdim=True)
            sim = F.cosine_similarity(tokens, mean, dim=-1)
            temp = max(float(temperature), 1e-6)
            weights = torch.softmax(sim / temp, dim=0)
            if agreement_boost and agreement_boost > 0:
                weights = weights * (1.0 + agreement_boost * sim.clamp(min=0.0))
                weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
            weights = weights.unsqueeze(-1)
            return (weights * tokens).sum(dim=0)
        raise ValueError(f"Unsupported prior pooling mode: {mode}")

    def _blend_velocity(
        self,
        blend_fn: Callable[..., torch.Tensor],
        e_obs: Union[torch.Tensor, sp.SparseTensor],
        e_prior: Union[torch.Tensor, sp.SparseTensor],
        alpha: float,
        prior_weight: float,
        blend_kwargs: Optional[dict] = None,
    ) -> Union[torch.Tensor, sp.SparseTensor]:
        blend_kwargs = blend_kwargs or {}
        if isinstance(e_obs, sp.SparseTensor):
            obs_feats = e_obs.feats
            prior_feats = e_prior.feats if isinstance(e_prior, sp.SparseTensor) else e_prior
            blended_feats = blend_fn(obs_feats, prior_feats, alpha, prior_weight, **blend_kwargs)
            return e_obs.replace(blended_feats)
        return blend_fn(e_obs, e_prior, alpha, prior_weight, **blend_kwargs)

    def _infer_velocity(
        self,
        sampler,
        model,
        x_t,
        t,
        cond: torch.Tensor,
        neg_cond: Optional[torch.Tensor],
        infer_params: Dict[str, object],
    ):
        sig = inspect.signature(sampler._inference_model)
        kwargs = {}
        if "neg_cond" in sig.parameters and neg_cond is not None:
            kwargs["neg_cond"] = neg_cond
        if "cfg_strength" in sig.parameters:
            kwargs["cfg_strength"] = infer_params.get("cfg_strength", 0.0)
        if "cfg_interval" in sig.parameters:
            kwargs["cfg_interval"] = infer_params.get("cfg_interval", (0.0, 1.0))
        return sampler._inference_model(model, x_t, t, cond=cond, **kwargs)

    def _sample_relaxflow_flow(
        self,
        model,
        sampler,
        noise,
        cond_obs: Dict[str, torch.Tensor],
        cond_prior: Dict[str, torch.Tensor],
        *,
        steps: int,
        rescale_t: float,
        gating_schedule: Callable[[int, int], float],
        gating_args: Dict[str, object],
        prior_weight: float,
        prior_blur_sigma: float,
        blur_attn_type: str,
        feat_blur_type: str,
        feat_blur_sigma: float,
        feat_noise_type: str,
        feat_noise_strength: float,
        flow_combine_fn: Callable[..., torch.Tensor],
        flow_blend_args: Dict[str, object],
        infer_params: Dict[str, object],
        verbose: bool,
        return_branch_latents: bool = False,
    ):
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        alpha_seq = self._build_alpha_schedule(steps, gating_schedule, gating_args)

        blur_context = RELAXFLOWAttnBlurContext(
            model,
            sigma=prior_blur_sigma,
            attn_type=blur_attn_type,
            feat_blur_type=feat_blur_type,
            feat_blur_sigma=feat_blur_sigma,
            feat_noise_type=feat_noise_type,
            feat_noise_strength=feat_noise_strength,
        )

        def run_branch(cond_bundle: Dict[str, torch.Tensor], use_blur: bool):
            sample = _clone_noise(noise)
            for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
                if use_blur:
                    with blur_context:
                        pred_v = self._infer_velocity(
                            sampler,
                            model,
                            sample,
                            t,
                            cond_bundle["cond"],
                            cond_bundle.get("neg_cond"),
                            infer_params,
                        )
                else:
                    pred_v = self._infer_velocity(
                        sampler,
                        model,
                        sample,
                        t,
                        cond_bundle["cond"],
                        cond_bundle.get("neg_cond"),
                        infer_params,
                    )
                sample = sample - (t - t_prev) * pred_v
            return sample

        if return_branch_latents:
            obs_only = run_branch(cond_obs, use_blur=False)
            prior_only = run_branch(cond_prior, use_blur=True)
        else:
            obs_only = None
            prior_only = None

        sample = _clone_noise(noise)
        loop = tqdm(t_pairs, desc="Sampling", disable=not verbose)
        for idx, (t, t_prev) in enumerate(loop):
            alpha_t = alpha_seq[min(idx, len(alpha_seq) - 1)]
            if prior_weight <= 0 or alpha_t <= 0:
                pred_v = self._infer_velocity(
                    sampler,
                    model,
                    sample,
                    t,
                    cond_obs["cond"],
                    cond_obs.get("neg_cond"),
                    infer_params,
                )
            else:
                pred_v_obs = self._infer_velocity(
                    sampler,
                    model,
                    sample,
                    t,
                    cond_obs["cond"],
                    cond_obs.get("neg_cond"),
                    infer_params,
                )
                with blur_context:
                    pred_v_prior = self._infer_velocity(
                        sampler,
                        model,
                        sample,
                        t,
                        cond_prior["cond"],
                        cond_prior.get("neg_cond"),
                        infer_params,
                    )
                pred_v = self._blend_velocity(
                    flow_combine_fn,
                    pred_v_obs,
                    pred_v_prior,
                    alpha_t,
                    prior_weight,
                    flow_blend_args,
                )
            sample = sample - (t - t_prev) * pred_v
        return sample, obs_only, prior_only

    def sample_sparse_structure_relaxflow(
        self,
        cond_obs: Dict[str, torch.Tensor],
        prior_tokens: torch.Tensor,
        *,
        num_samples: int = 1,
        gating_schedule: Callable[[int, int], float],
        prior_weight: float = 1.0,
        gating_args: Optional[dict] = None,
        inference_steps: Optional[int] = None,
        prior_blur_sigma: float = 2.5,
        blur_attn_type: str = "self",
        feat_blur_type: str = "none",
        feat_blur_sigma: float = 1.0,
        feat_noise_type: str = "gaussian",
        feat_noise_strength: float = 0.0,
        flow_combine_fn: Optional[Callable[..., torch.Tensor]] = None,
        flow_blend_args: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        return_branch_latents: bool = False,
    ):
        gating_args = gating_args or {}
        flow_blend_args = flow_blend_args or {}
        sampler_params = sampler_params or {}

        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)

        resolved_sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        steps = inference_steps or resolved_sampler_params.get("steps", 50)
        rescale_t = resolved_sampler_params.get("rescale_t", 1.0)
        verbose = resolved_sampler_params.get("verbose", True)
        infer_params = {
            k: resolved_sampler_params[k]
            for k in ("cfg_strength", "cfg_interval")
            if k in resolved_sampler_params
        }

        cond_prior = {
            "cond": prior_tokens,
            "neg_cond": torch.zeros_like(prior_tokens),
        }
        flow_combine_fn = resolve_flow_blend(
            flow_combine_fn, None, FLOW_BLEND_FNS["linear"]
        )

        main_latent, obs_latent, prior_latent = self._sample_relaxflow_flow(
            flow_model,
            self.sparse_structure_sampler,
            noise,
            cond_obs,
            cond_prior,
            steps=steps,
            rescale_t=rescale_t,
            gating_schedule=gating_schedule,
            gating_args=gating_args,
            prior_weight=prior_weight,
            prior_blur_sigma=prior_blur_sigma,
            blur_attn_type=blur_attn_type,
            feat_blur_type=feat_blur_type,
            feat_blur_sigma=feat_blur_sigma,
            feat_noise_type=feat_noise_type,
            feat_noise_strength=feat_noise_strength,
            flow_combine_fn=flow_combine_fn,
            flow_blend_args=flow_blend_args,
            infer_params=infer_params,
            verbose=verbose,
            return_branch_latents=return_branch_latents,
        )

        decoder = self.models["sparse_structure_decoder"]

        def _latent_to_coords(latent):
            occ = decoder(latent)
            coords = torch.argwhere(occ > 0)[:, [0, 2, 3, 4]].int()
            return coords

        coords_main = _latent_to_coords(main_latent)
        if not return_branch_latents:
            return coords_main

        coords_obs = _latent_to_coords(obs_latent) if obs_latent is not None else None
        coords_prior = _latent_to_coords(prior_latent) if prior_latent is not None else None
        return {
            "blend": coords_main,
            "obs_only": coords_obs,
            "prior_only": coords_prior,
        }

    def sample_slat_relaxflow(
        self,
        cond_obs: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        prior_tokens: torch.Tensor,
        *,
        gating_schedule: Callable[[int, int], float],
        prior_weight: float = 1.0,
        gating_args: Optional[dict] = None,
        inference_steps: Optional[int] = None,
        prior_blur_sigma: float = 2.5,
        blur_attn_type: str = "self",
        feat_blur_type: str = "none",
        feat_blur_sigma: float = 1.0,
        feat_noise_type: str = "gaussian",
        feat_noise_strength: float = 0.0,
        flow_combine_fn: Optional[Callable[..., torch.Tensor]] = None,
        flow_blend_args: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        return_branch_latents: bool = False,
    ):
        gating_args = gating_args or {}
        flow_blend_args = flow_blend_args or {}
        sampler_params = sampler_params or {}

        flow_model = self.models["slat_flow_model"]
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )

        resolved_sampler_params = {**self.slat_sampler_params, **sampler_params}
        steps = inference_steps or resolved_sampler_params.get("steps", 50)
        rescale_t = resolved_sampler_params.get("rescale_t", 1.0)
        verbose = resolved_sampler_params.get("verbose", True)
        infer_params = {
            k: resolved_sampler_params[k]
            for k in ("cfg_strength", "cfg_interval")
            if k in resolved_sampler_params
        }

        cond_prior = {
            "cond": prior_tokens,
            "neg_cond": torch.zeros_like(prior_tokens),
        }
        flow_combine_fn = resolve_flow_blend(
            flow_combine_fn, None, FLOW_BLEND_FNS["linear"]
        )

        main_slat, obs_slat, prior_slat = self._sample_relaxflow_flow(
            flow_model,
            self.slat_sampler,
            noise,
            cond_obs,
            cond_prior,
            steps=steps,
            rescale_t=rescale_t,
            gating_schedule=gating_schedule,
            gating_args=gating_args,
            prior_weight=prior_weight,
            prior_blur_sigma=prior_blur_sigma,
            blur_attn_type=blur_attn_type,
            feat_blur_type=feat_blur_type,
            feat_blur_sigma=feat_blur_sigma,
            feat_noise_type=feat_noise_type,
            feat_noise_strength=feat_noise_strength,
            flow_combine_fn=flow_combine_fn,
            flow_blend_args=flow_blend_args,
            infer_params=infer_params,
            verbose=verbose,
            return_branch_latents=return_branch_latents,
        )

        std = torch.tensor(self.slat_normalization["std"])[None].to(main_slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(main_slat.device)

        def _denorm(slat):
            if slat is None:
                return None
            return slat * std + mean

        main_slat = _denorm(main_slat)
        if not return_branch_latents:
            return main_slat
        return {
            "blend": main_slat,
            "obs_only": _denorm(obs_slat),
            "prior_only": _denorm(prior_slat),
        }

    @torch.no_grad()
    def run_relaxflow(
        self,
        image,
        prior_images: List,
        *,
        num_samples: int = 1,
        seed: Optional[int] = 42,
        sparse_structure_sampler_params: Optional[dict] = None,
        slat_sampler_params: Optional[dict] = None,
        formats: List[str] = None,
        preprocess_image: bool = True,
        prior_weight: float = 1.0,
        prior_blur_sigma: float = 2.5,
        blur_attn_type: str = "self",
        gating_schedule: Optional[Callable[[int, int], float]] = None,
        gating_schedule_name: Optional[str] = None,
        gating_args: Optional[dict] = None,
        flow_combine_fn: Optional[Callable[..., torch.Tensor]] = None,
        flow_blend_name: Optional[str] = None,
        flow_blend_args: Optional[dict] = None,
        stage1_inference_steps: Optional[int] = None,
        prior_pooling: str = "concat",
        prior_pooling_temperature: float = 0.1,
        prior_pooling_agreement_boost: float = 0.0,
        stage2_prior_weight: Optional[float] = None,
        stage2_prior_blur_sigma: Optional[float] = None,
        stage2_blur_attn_type: Optional[str] = None,
        stage2_gating_schedule: Optional[Callable[[int, int], float]] = None,
        stage2_gating_schedule_name: Optional[str] = None,
        stage2_gating_args: Optional[dict] = None,
        stage2_flow_combine_fn: Optional[Callable[..., torch.Tensor]] = None,
        stage2_flow_blend_name: Optional[str] = None,
        stage2_flow_blend_args: Optional[dict] = None,
        stage2_inference_steps: Optional[int] = None,
        stage2_prior_pooling: Optional[str] = None,
        stage2_prior_pooling_temperature: Optional[float] = None,
        stage2_prior_pooling_agreement_boost: Optional[float] = None,
        feat_blur_type: str = "none",
        feat_blur_sigma: float = 1.0,
        feat_noise_type: str = "gaussian",
        feat_noise_strength: float = 0.0,
        feat_blur_stage1: bool = False,
        return_branch_outputs: bool = False,
    ) -> dict:
        if not prior_images:
            raise ValueError("Must provide at least one prior image for RelaxFlow.")

        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if preprocess_image:
            image = self.preprocess_image(image)
            prior_images = [self.preprocess_image(img) for img in prior_images]

        cond_obs = self.get_cond([image])
        cond_prior_raw = self.get_cond(prior_images)
        prior_tokens_list = [cond_prior_raw["cond"][i : i + 1] for i in range(cond_prior_raw["cond"].shape[0])]

        prior_tokens = self._pool_prior_tokens(
            prior_tokens_list, prior_pooling, prior_pooling_temperature, prior_pooling_agreement_boost
        )

        stage2_prior_pooling = stage2_prior_pooling or prior_pooling
        stage2_prior_pooling_temperature = (
            prior_pooling_temperature
            if stage2_prior_pooling_temperature is None
            else stage2_prior_pooling_temperature
        )
        stage2_prior_pooling_agreement_boost = (
            prior_pooling_agreement_boost
            if stage2_prior_pooling_agreement_boost is None
            else stage2_prior_pooling_agreement_boost
        )
        prior_tokens_s2 = self._pool_prior_tokens(
            prior_tokens_list,
            stage2_prior_pooling,
            stage2_prior_pooling_temperature,
            stage2_prior_pooling_agreement_boost,
        )

        gating_args = gating_args or {}
        flow_blend_args = flow_blend_args or {}
        sparse_structure_sampler_params = sparse_structure_sampler_params or {}
        slat_sampler_params = slat_sampler_params or {}

        gating_schedule = resolve_gating_schedule(
            gating_schedule, gating_schedule_name, None
        )
        flow_combine_fn = resolve_flow_blend(
            flow_combine_fn, flow_blend_name, FLOW_BLEND_FNS["linear"]
        )

        stage2_prior_weight = prior_weight if stage2_prior_weight is None else stage2_prior_weight
        stage2_prior_blur_sigma = (
            prior_blur_sigma if stage2_prior_blur_sigma is None else stage2_prior_blur_sigma
        )
        stage2_blur_attn_type = (
            blur_attn_type if stage2_blur_attn_type is None else stage2_blur_attn_type
        )
        stage2_gating_schedule = resolve_gating_schedule(
            stage2_gating_schedule, stage2_gating_schedule_name, gating_schedule
        )
        stage2_gating_args = stage2_gating_args or gating_args
        stage2_flow_combine_fn = resolve_flow_blend(
            stage2_flow_combine_fn, stage2_flow_blend_name, flow_combine_fn
        )
        stage2_flow_blend_args = stage2_flow_blend_args or flow_blend_args

        stage1_feat_blur_type = feat_blur_type if feat_blur_stage1 else "none"
        stage1_feat_blur_sigma = feat_blur_sigma if feat_blur_stage1 else 0.0
        stage1_feat_noise_strength = feat_noise_strength if feat_blur_stage1 else 0.0

        logger.info("Stage 1: RelaxFlow sparse structure sampling")
        ss_bundle = self.sample_sparse_structure_relaxflow(
            cond_obs,
            prior_tokens,
            num_samples=num_samples,
            gating_schedule=gating_schedule,
            prior_weight=prior_weight,
            gating_args=gating_args,
            inference_steps=stage1_inference_steps,
            prior_blur_sigma=prior_blur_sigma,
            blur_attn_type=blur_attn_type,
            feat_blur_type=stage1_feat_blur_type,
            feat_blur_sigma=stage1_feat_blur_sigma,
            feat_noise_type=feat_noise_type,
            feat_noise_strength=stage1_feat_noise_strength,
            flow_combine_fn=flow_combine_fn,
            flow_blend_args=flow_blend_args,
            sampler_params=sparse_structure_sampler_params,
            return_branch_latents=return_branch_outputs,
        )

        logger.info("Stage 2: RelaxFlow SLAT sampling")
        if return_branch_outputs:
            coords_blend = ss_bundle["blend"]
            coords_obs = ss_bundle.get("obs_only")
            coords_prior = ss_bundle.get("prior_only")
        else:
            coords_blend = ss_bundle
            coords_obs = None
            coords_prior = None

        slat_bundle = self.sample_slat_relaxflow(
            cond_obs,
            coords_blend,
            prior_tokens_s2,
            gating_schedule=stage2_gating_schedule,
            prior_weight=stage2_prior_weight,
            gating_args=stage2_gating_args,
            inference_steps=stage2_inference_steps,
            prior_blur_sigma=stage2_prior_blur_sigma,
            blur_attn_type=stage2_blur_attn_type,
            feat_blur_type=feat_blur_type,
            feat_blur_sigma=feat_blur_sigma,
            feat_noise_type=feat_noise_type,
            feat_noise_strength=feat_noise_strength,
            flow_combine_fn=stage2_flow_combine_fn,
            flow_blend_args=stage2_flow_blend_args,
            sampler_params=slat_sampler_params,
            return_branch_latents=return_branch_outputs,
        )

        formats = formats or ["mesh", "gaussian", "radiance_field"]
        if return_branch_outputs:
            slat_blend = slat_bundle["blend"]
        else:
            slat_blend = slat_bundle
        outputs = self.decode_slat(slat_blend, formats)

        if not return_branch_outputs:
            return outputs

        branch_outputs: Dict[str, dict] = {}
        slat_blend_obs = slat_bundle.get("obs_only") if isinstance(slat_bundle, dict) else None
        slat_blend_prior = slat_bundle.get("prior_only") if isinstance(slat_bundle, dict) else None

        if slat_blend_obs is not None:
            branch_outputs["blend_then_obs"] = self.decode_slat(slat_blend_obs, formats)
        if slat_blend_prior is not None:
            branch_outputs["blend_then_prior"] = self.decode_slat(slat_blend_prior, formats)

        if coords_obs is not None:
            slat_obs = self.sample_slat(
                cond_obs,
                coords_obs,
                sampler_params=slat_sampler_params,
            )
            branch_outputs["obs_only"] = self.decode_slat(slat_obs, formats)

        if coords_prior is not None:
            prior_slat_bundle = self.sample_slat_relaxflow(
                cond_obs,
                coords_prior,
                prior_tokens_s2,
                gating_schedule=stage2_gating_schedule,
                prior_weight=stage2_prior_weight,
                gating_args=stage2_gating_args,
                inference_steps=stage2_inference_steps,
                prior_blur_sigma=stage2_prior_blur_sigma,
                blur_attn_type=stage2_blur_attn_type,
                feat_blur_type=feat_blur_type,
                feat_blur_sigma=feat_blur_sigma,
                feat_noise_type=feat_noise_type,
                feat_noise_strength=feat_noise_strength,
                flow_combine_fn=stage2_flow_combine_fn,
                flow_blend_args=stage2_flow_blend_args,
                sampler_params=slat_sampler_params,
                return_branch_latents=True,
            )
            slat_prior = prior_slat_bundle.get("prior_only") or prior_slat_bundle.get("blend")
            branch_outputs["prior_only"] = self.decode_slat(slat_prior, formats)

        return {"relaxflow": outputs, **branch_outputs}
