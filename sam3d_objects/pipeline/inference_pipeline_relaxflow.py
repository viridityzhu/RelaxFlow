# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Jiayin Zhu. Key implementation for the paper "RelaxFlow: A General Framework for 3D Object Generation".
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, List, Union, Dict
import contextlib
import random
from copy import deepcopy
from loguru import logger
from PIL import Image
import os
import imageio
import types
import functools
import math
import inspect
import time
import warnings

from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils
from .inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.data.utils import tree_tensor_map
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.pipeline.relaxflow_variants import (
    GATING_SCHEDULES,
    FLOW_BLEND_FNS,
    resolve_flow_blend,
    resolve_gating_schedule,
)
from sam3d_objects.pipeline.inference_utils import compute_geometry_visibility_mask
from sam3d_objects.utils.visualization import SceneVisualizer
from sam3d_objects.model.backbone.tdfy_dit.renderers import GaussianRenderer
from sam3d_objects.model.backbone.tdfy_dit.representations import Gaussian
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_multiply
from sam3d_objects.pipeline.layout_post_optimization_utils import denormalize_f
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds
LOG_LEVEL_STAGE = logger.level("STAGE", no=38, color="<green>", icon="🐍")


def intrinsics_from_fov(fov_deg: float, resolution: int) -> torch.Tensor:
    fov = math.radians(float(fov_deg))
    focal = 0.5 * float(resolution) / math.tan(fov / 2.0)
    cx = float(resolution) / 2.0
    cy = float(resolution) / 2.0
    return torch.tensor(
        [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

# -- Gaussian Blur util for attention logits --
# 构造一维高斯核，用来在注意力 logits 上执行平滑，削弱高频噪声
def gaussian_kernel1d(kernel_size, sigma, device):
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_blur2d(logits: torch.Tensor, sigma: float = 2.5) -> torch.Tensor:
    # 在 query / key 两个维度上做二维模糊，得到低频语义势场
    if sigma <= 0: return logits
    B, head, Q, K = logits.shape
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel = gaussian_kernel1d(kernel_size, sigma, logits.device)
    # Batchwise depthwise conv (each query vector gets its blur)
    logits = logits.view(-1, 1, Q, K)
    padding = kernel_size // 2
    # Blur query and key dims (2D)
    out = F.conv2d(logits, kernel[None, None, :, None], padding=(padding,0), groups=1)
    out = F.conv2d(out, kernel[None, None, None, :], padding=(0, padding), groups=1)
    out = out.view(B, head, Q, K)
    return out

def gaussian_blur_feature_dim(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply 1D gaussian blur along the last (feature) dimension.
    x: (..., D)
    """
    if sigma is None or sigma <= 0:
        return x
    if x.dim() < 1:
        return x
    D = x.shape[-1]
    if D <= 1:
        return x
    kernel_size = int(2 * round(3 * float(sigma)) + 1)
    # Clamp kernel size to <= D and keep it odd.
    if kernel_size > D:
        kernel_size = D if (D % 2 == 1) else max(D - 1, 1)
    if kernel_size <= 1:
        return x
    kernel = gaussian_kernel1d(kernel_size, float(sigma), x.device).to(dtype=x.dtype)
    # Conv1d expects (N, C, L). We blur each feature vector independently.
    x_flat = x.reshape(-1, 1, D)
    out = F.conv1d(x_flat, kernel.view(1, 1, -1), padding=kernel_size // 2)
    return out.reshape(*x.shape)

def add_feature_noise(x: torch.Tensor, noise_type: str, strength: float) -> torch.Tensor:
    """
    Add random noise along the feature dimension (same shape as x).
    - noise_type: 'gaussian' or 'uniform'
    - strength: noise magnitude (std for gaussian, half-range for uniform)
    """
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

# -- RelaxFlow Attention Blur Injection (monkeypatch) --
class RELAXFLOWAttnBlurContext:
    """
    Context manager to inject gaussian blur into attention logits during prior branch forward. 
    """
    def __init__(
        self,
        model,
        sigma,
        attn_type: str = "self",
        *,
        feat_blur_type: str = "none",
        feat_blur_sigma: float = 1.0,
        feat_noise_type: str = "gaussian",
        feat_noise_strength: float = 0.0,
        debug_v: bool = False,
        debug_tag: str = "",
    ):
        self.model = model  # The full model or just backbone/flow/transformer
        self.sigma = sigma
        self.attn_type = attn_type
        self.feat_blur_type = (feat_blur_type or "none").lower()
        self.feat_blur_sigma = feat_blur_sigma
        self.feat_noise_type = (feat_noise_type or "gaussian").lower()
        self.feat_noise_strength = feat_noise_strength
        self.debug_v = bool(debug_v)
        self.debug_tag = debug_tag
        self._patches = []
        self._active = False
        self._sdpa_patches = []
        self._debug_state = None
        self._debug_nohit_reported = False
        self._sparse_attn_patches = []

    def __enter__(self):
        do_logit_blur = self.sigma is not None and self.sigma > 0
        do_feat_blur = self.feat_blur_type == "blur" and self.feat_blur_sigma is not None and self.feat_blur_sigma > 0
        do_feat_noise = (
            self.feat_blur_type == "noise"
            and self.feat_noise_strength is not None
            and self.feat_noise_strength > 0
        )
        if not (do_logit_blur or do_feat_blur or do_feat_noise):
            self._active = False
            return self

        self._active = True
        # Clear previous patches so the context can be re-used across steps
        self._patches = []
        self._sdpa_patches = []
        self._debug_state = None
        if self.debug_v and (do_feat_blur or do_feat_noise):
            self._debug_state = {
                "sdpa_calls": 0,
                "mha_calls": 0,
                "first_logged": False,
                "patched_mha": 0,
                "total_mha": 0,
                "patched_sdpa": False,
                "sparse_calls": 0,
                "patched_sparse": False,
            }

        if do_feat_blur or do_feat_noise:
            #!deprecated
            raise NotImplementedError("This feature is deprecated and is removed. Please set do_feat_blur or do_feat_noise to False.")

        from sam3d_objects.model.backbone.tdfy_dit.modules.attention.modules import MultiHeadAttention  # type: ignore

        for m in self.model.modules():
            if isinstance(m, MultiHeadAttention):
                if self._debug_state is not None:
                    self._debug_state["total_mha"] += 1
                typ = getattr(m, "_type", None)
                if self.attn_type in ("both", typ):
                    orig_forward = m.forward

                    @functools.wraps(orig_forward)
                    def blur_forward(this, x, context=None, indices=None):
                        """
                        Re-implement attention with explicit logits computation and Gaussian blur.
                        Works for both self- and cross-attention.
                        """
                        B, L, C = x.shape
                        if this._type == "self":
                            qkv = this.to_qkv(x)
                            qkv = qkv.reshape(B, L, 3, this.num_heads, -1)
                            if this.use_rope:
                                q, k, v = qkv.unbind(dim=2)
                                q, k = this.rope(q, k, indices)
                                qkv = torch.stack([q, k, v], dim=2)
                            q, k, v = qkv.unbind(dim=2)
                        else:
                            assert context is not None, "Cross-attention requires context"
                            Lkv = context.shape[1]
                            q = this.to_q(x)
                            kv = this.to_kv(context)
                            q = q.reshape(B, L, this.num_heads, -1)
                            kv = kv.reshape(B, Lkv, 2, this.num_heads, -1)
                            k, v = kv.unbind(dim=2)

                        if this.qk_rms_norm:
                            q = this.q_rms_norm(q)
                            k = this.k_rms_norm(k)

                        # q, k, v: [B, Lq or Lk, H, D]
                        q_t = q.permute(0, 2, 1, 3)  # [B,H,Lq,D]
                        k_t = k.permute(0, 2, 1, 3)  # [B,H,Lk,D]
                        v_t = v.permute(0, 2, 1, 3)  # [B,H,Lk,D]

                        scale = 1.0 / math.sqrt(this.head_dim)
                        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [B,H,Lq,Lk]
                        if do_logit_blur:
                            attn_scores = gaussian_blur2d(attn_scores, self.sigma)
                        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
                        attn_probs = F.softmax(attn_scores, dim=-1)
                        if do_feat_blur or do_feat_noise:
                            raise NotImplementedError("This feature is deprecated and is removed. Please set do_feat_blur and do_feat_noise to False.")
                        out = torch.matmul(attn_probs, v_t)  # [B,H,Lq,D]
                        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)
                        out = this.to_out(out)
                        return out

                    m._original_forward = orig_forward  # type: ignore[attr-defined]
                    m.forward = types.MethodType(blur_forward, m)
                    self._patches.append((m, orig_forward))
                    if self._debug_state is not None:
                        self._debug_state["patched_mha"] += 1
        if self._debug_state is not None:
            logger.info(
                f"[RELAXFLOWAttnBlurContext:{self.debug_tag}] armed: "
                f"attn_type={self.attn_type}, logit_sigma={self.sigma}, "
                f"feat={self.feat_blur_type} (blur_sigma={self.feat_blur_sigma}, "
                f"noise={self.feat_noise_type},{self.feat_noise_strength}), "
                f"patched_sdpa={self._debug_state.get('patched_sdpa')}, "
                f"patched_sparse={self._debug_state.get('patched_sparse')}, "
                f"patched_mha={self._debug_state.get('patched_mha')}/{self._debug_state.get('total_mha')}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._active:
            return
        # Unpatch
        for m, orig in self._patches:
            m.forward = orig
        self._patches = []
        for mod, attr, orig in self._sdpa_patches:
            setattr(mod, attr, orig)
        self._sdpa_patches = []
        for mod, attr, orig in self._sparse_attn_patches:
            setattr(mod, attr, orig)
        self._sparse_attn_patches = []
        if self._debug_state is not None:
            sdpa_calls = self._debug_state.get("sdpa_calls", 0)
            mha_calls = self._debug_state.get("mha_calls", 0)
            sparse_calls = self._debug_state.get("sparse_calls", 0)
            if (sdpa_calls == 0 and mha_calls == 0) and not self._debug_nohit_reported:
                self._debug_nohit_reported = True
                logger.warning(
                    f"[RELAXFLOWAttnBlurContext:{self.debug_tag}] V-perturb NO-HIT (no attention calls captured). "
                    f"sdpa_calls=0, mha_calls=0, sparse_calls={sparse_calls}. "
                    f"patched_sdpa={self._debug_state.get('patched_sdpa')}, "
                    f"patched_sparse={self._debug_state.get('patched_sparse')}, "
                    f"patched_mha={self._debug_state.get('patched_mha')}/{self._debug_state.get('total_mha')}. "
                    "This usually means the stage2 model is using a different attention implementation "
                    "or the call path is bypassing our patched functions."
                )
            elif sdpa_calls > 0 or mha_calls > 0 or sparse_calls > 0:
                logger.info(
                    f"[RELAXFLOWAttnBlurContext:{self.debug_tag}] V-perturb summary: "
                    f"sdpa_calls={sdpa_calls}, sparse_calls={sparse_calls}, mha_calls={mha_calls}"
                )
        self._active = False

# -- Main pipeline extension --
class InferencePipelineRELAXFLOW(InferencePipelinePointMap):
    def __init__(
        self,
        *args,
        prior_blur_sigma: float = 2.5,
        blur_attn_type: str = "self",
        gating_schedule: Optional[Callable[[int, int], float]] = None,
        prior_use_cropped_only: bool = False,
        prior_only_cropped_img_and_mask: bool = True,
        prior_flow_combiner: Optional[
            Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor]
        ] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_blur_sigma = prior_blur_sigma
        self.blur_attn_type = blur_attn_type
        self.gating_schedule = gating_schedule
        # Prior tokens reuse the cropped-image positional embedding by default
        self.prior_use_cropped_only = prior_use_cropped_only
        self.prior_only_cropped_img_and_mask = prior_only_cropped_img_and_mask
        # Optional custom flow combiner
        self.prior_flow_combiner = prior_flow_combiner

    def _shape_autocast_context(self):
        # Keep the same autocast precision as the original shape model to avoid numerical inconsistency between prior and obs flows
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.shape_model_dtype)
        return contextlib.nullcontext()

    def _normalize_prior_inputs(self, prior_images, prior_masks):
        if prior_images is None:
            return [], None
        if isinstance(prior_images, (list, tuple)):
            prior_image_list = list(prior_images)
        else:
            prior_image_list = [prior_images]

        prior_mask_list = None
        if prior_masks is not None:
            if isinstance(prior_masks, (list, tuple)):
                prior_mask_list = list(prior_masks)
            else:
                prior_mask_list = [prior_masks]

            if len(prior_mask_list) == 0:
                prior_mask_list = None
            elif len(prior_mask_list) == 1 and len(prior_image_list) > 1:
                logger.info(
                    f"Broadcasting single prior mask across {len(prior_image_list)} priors"
                )
                prior_mask_list = prior_mask_list * len(prior_image_list)
            elif len(prior_mask_list) < len(prior_image_list):
                logger.warning(
                    f"Found {len(prior_mask_list)} prior masks for {len(prior_image_list)} priors; "
                    "missing masks will be treated as None."
                )
                prior_mask_list = prior_mask_list + [None] * (
                    len(prior_image_list) - len(prior_mask_list)
                )
            elif len(prior_mask_list) > len(prior_image_list):
                logger.warning(
                    f"Found {len(prior_mask_list)} prior masks for {len(prior_image_list)} priors; "
                    "extra masks will be ignored."
                )
                prior_mask_list = prior_mask_list[: len(prior_image_list)]

        return prior_image_list, prior_mask_list

    def _coerce_prior_image(self, img):
        if isinstance(img, (str, os.PathLike)):
            img = Image.open(img)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if isinstance(img, np.ndarray):
            arr = img
        elif isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
        else:
            raise TypeError(
                f"Unsupported prior image type {type(img)}; "
                "expected path/np.ndarray/PIL/Tensor."
            )

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if (
            arr.ndim == 3
            and arr.shape[0] in (3, 4)
            and arr.shape[-1] not in (3, 4)
        ):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8)

        return arr

    def _coerce_prior_mask(self, mask):
        if mask is None:
            return None
        if isinstance(mask, (str, os.PathLike)):
            mask = Image.open(mask)
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                f"Unsupported prior mask type {type(mask)}; "
                "expected path/np.ndarray/PIL/Tensor."
            )
        if mask.ndim == 3:
            if mask.shape[-1] in (1, 3, 4):
                mask = mask[..., -1]
            elif mask.shape[0] in (1, 3, 4):
                mask = mask[0]
        return mask

    def _resize_prior_mask(self, mask_arr, target_image, idx=None):
        if mask_arr is None:
            return None
        if mask_arr.ndim != 2:
            raise ValueError(
                f"Prior mask must be 2D after squeezing channels; got shape {mask_arr.shape}"
            )
        h, w = target_image.shape[:2]
        if mask_arr.shape[0] == h and mask_arr.shape[1] == w:
            return mask_arr.astype(bool)
        logger.info(
            f"Resizing prior mask {idx} from {mask_arr.shape[:2]} to {(h, w)}"
        )
        mask_img = Image.fromarray((mask_arr > 0).astype(np.uint8) * 255)
        mask_img = mask_img.resize((w, h), resample=Image.Resampling.NEAREST)
        return (np.array(mask_img) > 0).astype(bool)

    def _save_prior_mask(self, mask_arr, idx):
        if not self.save_render_dir:
            return
        os.makedirs(self.save_render_dir, exist_ok=True)
        prior_mask_path = os.path.join(self.save_render_dir, f"prior_mask_{idx}.png")
        prior_mask_np = (mask_arr > 0).astype(np.uint8) * 255
        imageio.imwrite(prior_mask_path, prior_mask_np)

    def _write_prior_pooling_debug(
        self,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        mode: str,
        temperature: float,
        agreement_boost: float,
        similarity: Optional[torch.Tensor],
        agreement: Optional[torch.Tensor],
        debug_dir: Optional[str],
        debug_tag: str,
        debug_stride: int,
        debug_max_tokens: Optional[int],
        eps: float = 1e-6,
    ) -> None:
        n_priors, batch_size, n_tokens = weights.shape[:3]
        weights = weights.detach().float()
        weights_min = float(weights.min().cpu())
        weights_max = float(weights.max().cpu())
        weights_mean = float(weights.mean().cpu())
        top1 = weights.max(dim=0).values
        top1_mean = float(top1.mean().cpu())
        entropy = -(weights * (weights.clamp_min(eps).log())).sum(dim=0)
        entropy = entropy / math.log(n_priors)
        entropy_mean = float(entropy.mean().cpu())
        logger.info(
            "Prior pooling[{}]: mode={}, temp={:.3f}, boost={:.3f}, priors={}, tokens={}, "
            "w(min/mean/max)={:.4f}/{:.4f}/{:.4f}, top1_mean={:.4f}, entropy_mean={:.4f}",
            debug_tag,
            mode,
            float(temperature),
            float(agreement_boost),
            n_priors,
            n_tokens,
            weights_min,
            weights_mean,
            weights_max,
            top1_mean,
            entropy_mean,
        )
        if similarity is not None:
            sim = similarity.detach().float()
            logger.info(
                "Prior pooling[{}]: similarity(min/mean/max)={:.4f}/{:.4f}/{:.4f}",
                debug_tag,
                float(sim.min().cpu()),
                float(sim.mean().cpu()),
                float(sim.max().cpu()),
            )
        if agreement is not None:
            agree = agreement.detach().float()
            logger.info(
                "Prior pooling[{}]: agreement(mean)={:.4f}",
                debug_tag,
                float(agree.mean().cpu()),
            )

        if not debug_dir:
            return
        os.makedirs(debug_dir, exist_ok=True)
        stride = max(1, int(debug_stride))
        max_tokens = None if debug_max_tokens is None else int(debug_max_tokens)
        heatmap = weights[:, 0, ::stride]
        if max_tokens is not None and heatmap.shape[1] > max_tokens:
            heatmap = heatmap[:, :max_tokens]
        denom = heatmap.max().clamp_min(eps)
        heatmap_img = (heatmap / denom * 255.0).byte().cpu().numpy()
        heatmap_path = os.path.join(
            debug_dir, f"prior_pooling_weights_{debug_tag}.png"
        )
        imageio.imwrite(heatmap_path, heatmap_img)
        npz_path = os.path.join(debug_dir, f"prior_pooling_weights_{debug_tag}.npz")
        payload = {"weights": heatmap.cpu().numpy()}
        if similarity is not None:
            payload["similarity"] = similarity[:, 0, ::stride].detach().cpu().numpy()
        if agreement is not None:
            agree = agreement[0, ::stride]
            if agree.ndim > 1:
                agree = agree.squeeze(-1)
            payload["agreement"] = agree.detach().cpu().numpy()
        np.savez_compressed(npz_path, **payload)

    def _align_tokens_to_reference(
        self,
        reference_tokens: torch.Tensor,
        tokens: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = F.normalize(reference_tokens.float(), dim=-1, eps=eps)
        tok = F.normalize(tokens.float(), dim=-1, eps=eps)
        sim = torch.bmm(ref, tok.transpose(1, 2))  # (B, T, T)
        batch_size, num_tokens = sim.shape[:2]
        perm = torch.empty((batch_size, num_tokens), dtype=torch.long, device=sim.device)
        matched_sim = torch.empty((batch_size, num_tokens), dtype=sim.dtype, device=sim.device)

        for b in range(batch_size):
            sim_b = sim[b]
            ref_best = sim_b.max(dim=1).values
            order = torch.argsort(ref_best, descending=True)
            taken = torch.zeros(num_tokens, dtype=torch.bool, device=sim_b.device)
            perm_b = torch.empty(num_tokens, dtype=torch.long, device=sim_b.device)
            match_b = torch.empty(num_tokens, dtype=sim_b.dtype, device=sim_b.device)
            for r in order:
                scores = sim_b[r].masked_fill(taken, -1e9)
                idx = torch.argmax(scores)
                perm_b[r] = idx
                match_b[r] = sim_b[r, idx]
                taken[idx] = True
            perm[b] = perm_b
            matched_sim[b] = match_b

        aligned = tokens.gather(1, perm.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return aligned, perm, matched_sim

    def _align_prior_tokens(
        self,
        tokens: torch.Tensor,
        *,
        debug: bool = False,
        debug_tag: str = "prior",
        eps: float = 1e-6,
    ) -> torch.Tensor:
        ref_tokens = tokens[0]
        aligned = [ref_tokens]
        match_means = []
        for idx in range(1, tokens.shape[0]):
            aligned_tokens, _, matched_sim = self._align_tokens_to_reference(
                ref_tokens, tokens[idx], eps=eps
            )
            aligned.append(aligned_tokens)
            if debug:
                match_means.append(float(matched_sim.mean().detach().cpu()))
        if debug and match_means:
            logger.info(
                "Prior pooling[{}]: alignment mean cosine per prior={}",
                debug_tag,
                [f"{m:.4f}" for m in match_means],
            )
        return torch.stack(aligned, dim=0)

    def _pool_prior_tokens(
        self,
        tokens: torch.Tensor,
        mode: str = "mean",
        temperature: float = 0.1,
        agreement_boost: float = 0.0,
        *,
        debug: bool = False,
        debug_dir: Optional[str] = None,
        debug_tag: str = "prior",
        debug_stride: int = 8,
        debug_max_tokens: Optional[int] = 256,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Pool prior tokens across multiple priors.

        tokens: (N, B, T, D)
        mode: "mean", "consensus", or "concat"
        """
        if tokens.shape[0] == 1:
            return tokens.squeeze(0)

        if (mode == "consensus" or agreement_boost > 0) and mode != "concat":
            tokens = self._align_prior_tokens(
                tokens,
                debug=debug,
                debug_tag=debug_tag,
                eps=eps,
            )

        similarity = None
        agreement = None

        if mode == "mean":
            pooled = tokens.mean(dim=0)
            if debug:
                weights = torch.full(
                    (tokens.shape[0], tokens.shape[1], tokens.shape[2]),
                    1.0 / tokens.shape[0],
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
        elif mode == "concat":
            pooled = tokens.permute(1, 0, 2, 3).reshape(
                tokens.shape[1], tokens.shape[0] * tokens.shape[2], tokens.shape[3]
            )
            if debug:
                logger.info(
                    "Prior pooling[{}]: mode=concat, priors={}, tokens_per_prior={}, "
                    "concat_tokens={}",
                    debug_tag,
                    tokens.shape[0],
                    tokens.shape[2],
                    pooled.shape[1],
                )
        elif mode == "consensus":
            if temperature is None or temperature <= 0:
                raise ValueError("prior_pooling_temperature must be > 0 for consensus pooling")
            normed = F.normalize(tokens, dim=-1, eps=eps)
            mean_norm = F.normalize(normed.mean(dim=0, keepdim=True), dim=-1, eps=eps)
            similarity = (normed * mean_norm).sum(dim=-1)  # (N, B, T)
            weights = torch.softmax(similarity / temperature, dim=0)
            pooled = (weights.unsqueeze(-1) * tokens).sum(dim=0)
        else:
            raise ValueError(f"Unknown prior pooling mode: {mode}")

        if mode != "concat" and agreement_boost and tokens.shape[0] > 1:
            # Boost tokens that are consistent across priors; suppress conflicting ones.
            var = tokens.var(dim=0, unbiased=False)
            var_norm = var.mean(dim=-1, keepdim=True)
            var_ref = var_norm.mean().clamp_min(eps)
            agreement = 1.0 / (1.0 + var_norm / var_ref)
            pooled = pooled * (1.0 + agreement_boost * agreement)
        elif mode != "concat" and debug and tokens.shape[0] > 1:
            var = tokens.var(dim=0, unbiased=False)
            var_norm = var.mean(dim=-1, keepdim=True)
            var_ref = var_norm.mean().clamp_min(eps)
            agreement = 1.0 / (1.0 + var_norm / var_ref)

        if debug and mode != "concat":
            if mode == "mean":
                weights = torch.full(
                    (tokens.shape[0], tokens.shape[1], tokens.shape[2]),
                    1.0 / tokens.shape[0],
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
            self._write_prior_pooling_debug(
                tokens,
                weights,
                mode,
                temperature,
                agreement_boost,
                similarity,
                agreement,
                debug_dir,
                debug_tag,
                debug_stride,
                debug_max_tokens,
                eps=eps,
            )

        return pooled

    def embed_prior_images(
        self,
        prior_image_list: List[Union[str, Image.Image, np.ndarray, torch.Tensor]],
        prior_mask_list: Optional[
            List[Union[str, Image.Image, np.ndarray, torch.Tensor]]
        ] = None,
        keep_cropped_only: Optional[bool] = None,
        only_cropped_img_and_mask: Optional[bool] = None,
        *,
        condition_embedder=None,
        preprocessor=None,
        pool_mode: str = "mean",
        pool_temperature: float = 0.1,
        pool_agreement_boost: float = 0.0,
        pool_debug: bool = False,
        pool_debug_dir: Optional[str] = None,
        pool_debug_tag: str = "prior",
        pool_debug_stride: int = 8,
        pool_debug_max_tokens: Optional[int] = 256,
    ) -> torch.Tensor:
        """
        Embed 1-N prior images as tokens using the same condition embedder,
        then pool across priors (mean or consensus). By default only the cropped/object token
        is kept via the embedder_fuser._filter_tokens_by_name hook.
        """
        prior_image_list, prior_mask_list = self._normalize_prior_inputs(
            prior_image_list, prior_mask_list
        )
        if not prior_image_list:
            raise ValueError("Must provide at least one prior image for RelaxFlow")
        condition_embedder = condition_embedder or self.condition_embedders["ss_condition_embedder"]
        preprocessor = preprocessor or self.ss_preprocessor
        if self.prior_only_cropped_img_and_mask:
            logger.info("Using only cropped image and mask for prior")
            keep_cropped_only = False
            only_cropped_img_and_mask = True
        elif self.prior_use_cropped_only:
            logger.info("Using only cropped image for prior")
            keep_cropped_only = True
            only_cropped_img_and_mask = False
        else:
            logger.info("Using full image for prior")
            keep_cropped_only = False
            only_cropped_img_and_mask = False
        processed = []
        with torch.no_grad():
            with self._shape_autocast_context():
                for idx, img in enumerate(prior_image_list):
                    prior_mask = (
                        prior_mask_list[idx]
                        if prior_mask_list is not None and idx < len(prior_mask_list)
                        else None
                    )
                    arr = self._coerce_prior_image(img)
                    prior_mask_arr = self._coerce_prior_mask(prior_mask)
                    if prior_mask_arr is not None:
                        prior_mask_arr = self._resize_prior_mask(prior_mask_arr, arr, idx=idx)
                        arr = self.merge_image_and_mask(arr, prior_mask_arr)
                    if arr.shape[-1] == 3:
                        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
                        arr = np.concatenate([arr, alpha], axis=-1)
                    arr = arr[..., :4]

                    item = self.preprocess_image(arr, preprocessor)
                    tokens = condition_embedder(
                        **item, only_cropped_img=keep_cropped_only, only_cropped_img_and_mask=only_cropped_img_and_mask
                    )
                    processed.append(tokens)

        stacked = torch.stack(processed, dim=0)  # (N, ...)
        return self._pool_prior_tokens(
            stacked,
            mode=pool_mode,
            temperature=pool_temperature,
            agreement_boost=pool_agreement_boost,
            debug=pool_debug,
            debug_dir=pool_debug_dir,
            debug_tag=pool_debug_tag,
            debug_stride=pool_debug_stride,
            debug_max_tokens=pool_debug_max_tokens,
        )  # (B, T, D)

    # ---- Rendering utils (fix orientation by rotating frames CCW 90 deg) ----
    def _render_turntable(
        self,
        sample,
        num_frames: int = 120,
        resolution: int = 512,
        backend: str = "inria",
        yaw_start_deg: float = -90,
        pitch_deg: float = 0,
        radius: float = 2.0,
        fov_deg: float = 40,
        bg_color=(0, 0, 0),
        path_type: str = "spiral",
        pitch_span_deg: float = 30.0,
        radius_span: float = 0.3,
    ) -> Dict[str, List[np.ndarray]]:
        yaws = (
            torch.linspace(0, 2 * torch.pi, num_frames)
            + math.radians(yaw_start_deg)
        ).tolist()
        if path_type == "turntable":
            pitchs = [math.radians(pitch_deg)] * num_frames
            radii = [radius] * num_frames
        else:
            t_vals = torch.linspace(0, 2 * torch.pi, num_frames)
            pitchs = (0.25 + 0.5 * torch.sin(t_vals)).tolist()
            radii = [radius] * num_frames
        extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitchs, radii, fov_deg
        )

        render_options = {
            "resolution": resolution,
            "bg_color": bg_color,
            "backend": backend,
        }
        ret = render_utils.render_frames(
            sample,
            extr,
            intr,
            render_options,
            verbose=False,
        )
        return ret["color"]

    def _render_turntable_rgba(
        self,
        sample,
        num_frames: int = 120,
        resolution: int = 512,
        backend: str = "inria",
        yaw_start_deg: float = -90,
        pitch_deg: float = 0,
        radius: float = 2.0,
        fov_deg: float = 40,
        bg_color=(0, 0, 0),
        path_type: str = "spiral",
    ) -> Dict[str, List[np.ndarray]]:
        yaws = (
            torch.linspace(0, 2 * torch.pi, num_frames)
            + math.radians(yaw_start_deg)
        ).tolist()
        if path_type == "turntable":
            pitchs = [math.radians(pitch_deg)] * num_frames
            radii = [radius] * num_frames
        else:
            t_vals = torch.linspace(0, 2 * torch.pi, num_frames)
            pitchs = (0.25 + 0.5 * torch.sin(t_vals)).tolist()
            radii = [radius] * num_frames
        extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitchs, radii, fov_deg
        )
        render_options = {
            "resolution": resolution,
            "bg_color": bg_color,
            "backend": backend,
        }
        ret = render_utils.render_frames_rgba(
            sample,
            extr,
            intr,
            render_options,
            verbose=False,
        )
        return ret["color"]

    def _save_video_and_ply(
        self,
        out_dict: dict,
        save_dir: str,
        name: str,
        render_backend: str = "inria",
        render_resolution: int = 512,
        render_num_frames: int = 120,
        render_frame_stride: int = 1,
        render_bg_color=(0, 0, 0),
        intrinsics: Optional[torch.Tensor] = None,
        original_image: Optional[np.ndarray] = None,
        save_overlay: bool = True,
        overlay_pose: Optional[dict] = None,
        render_path: str = "spiral",
        render_pitch_span_deg: float = 30.0,
        render_radius_span: float = 0.3,
    ):
        os.makedirs(save_dir, exist_ok=True)
        gs = out_dict.get("gs", None)
        if gs is not None:
            # ply_path = os.path.join(save_dir, f"{name}.ply")
            # gs.save_ply(ply_path)
            # logger.info(f"Saved PLY to {ply_path}")
            rgba_frames = []
            try:
                rgba_frames = self._render_turntable_rgba(
                    gs,
                    num_frames=render_num_frames,
                    resolution=render_resolution,
                    backend=render_backend,
                    bg_color=render_bg_color,
                    path_type=render_path,
                )
            except Exception as exc:
                logger.warning("RGBA render failed for {}: {}", name, exc)

            if rgba_frames:
                rgba_dir = os.path.join(save_dir, "frames_rgba")
                os.makedirs(rgba_dir, exist_ok=True)
                frame_stride = max(1, int(render_frame_stride))
                saved_frames: List[np.ndarray] = []
                for idx, frame in enumerate(rgba_frames):
                    if idx % frame_stride != 0:
                        continue
                    imageio.imwrite(os.path.join(rgba_dir, f"frame_{idx:04d}.png"), frame)
                    saved_frames.append(frame)
                if saved_frames:
                    video_path = os.path.join(save_dir, f"{name}.mp4")
                    with imageio.get_writer(video_path, fps=15) as writer:
                        for frame in saved_frames:
                            writer.append_data(frame[..., :3])
                    logger.info("Saved turntable video to {}", video_path)


    def merge_mask_to_rgba(self, image, mask):
        # Normalize image to HxWx3 if needed (e.g., grayscale inputs).
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        # Ensure mask is single-channel
        if mask.ndim == 3:
            mask = mask[..., -1]
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image


    def run(
        self,
        image: Union[None, Image.Image, np.ndarray, torch.Tensor],
        mask: Union[None, Image.Image, np.ndarray, torch.Tensor] = None,
        prior_images: Optional[
            List[Union[str, Image.Image, np.ndarray, torch.Tensor]]
        ] = None,
        prior_masks: Optional[
            List[Union[str, Image.Image, np.ndarray, torch.Tensor]]
        ] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=True,
        use_vertex_color=False,
        use_geometry_mask_condition_mask: bool = False,
        geometry_mask_soft_falloff: float = 3.0,
        geometry_mask_param_tolerance_scale: float = 1.5,
        geometry_mask_param_dilate_scale: float = 1.5,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        estimate_plane: bool = False,
        only_cropped_img: bool = False,
        only_cropped_img_and_mask: bool = False,
        image_as: str = "part",
        prior_weight: float = 1.0,
        prior_blur_sigma: Optional[float] = None,
        blur_attn_type: Optional[str] = None,
        feat_blur_type: str = "none",
        feat_blur_sigma: float = 1.0,
        feat_noise_type: str = "gaussian",
        feat_noise_strength: float = 0.0,
        feat_blur_stage1: bool = False,
        gating_args: Optional[dict] = None,
        gating_schedule: Optional[Callable[[int, int], float]] = None,
        gating_schedule_name: Optional[str] = None,
        flow_combine_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor]
        ] = None,
        flow_blend_args: Optional[dict] = None,
        flow_blend_name: Optional[str] = None,
        prior_use_cropped_only: Optional[bool] = None,
        prior_only_cropped_img_and_mask: Optional[bool] = None,
        prior_pooling: str = "mean",
        prior_pooling_temperature: float = 0.1,
        prior_pooling_agreement_boost: float = 0.0,
        prior_pooling_debug: bool = False,
        prior_pooling_debug_stride: int = 8,
        prior_pooling_debug_max_tokens: Optional[int] = 256,
        stage2_prior_weight: Optional[float] = None,
        stage2_prior_blur_sigma: Optional[float] = None,
        stage2_blur_attn_type: Optional[str] = None,
        stage2_gating_args: Optional[dict] = None,
        stage2_gating_schedule: Optional[Callable[[int, int], float]] = None,
        stage2_gating_schedule_name: Optional[str] = None,
        stage2_flow_combine_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor]
        ] = None,
        stage2_flow_blend_args: Optional[dict] = None,
        stage2_flow_blend_name: Optional[str] = None,
        stage2_prior_pooling: Optional[str] = "consensus",
        stage2_prior_pooling_temperature: Optional[float] = None,
        stage2_prior_pooling_agreement_boost: Optional[float] = 0.3,
        stage2_prior_pooling_debug: Optional[bool] = None,
        stage2_prior_pooling_debug_stride: Optional[int] = None,
        stage2_prior_pooling_debug_max_tokens: Optional[int] = None,
        compute_original: bool = True,
        compute_blend_prior_slat: bool = True,
        return_branch_outputs: bool = True,
        save_render_dir: Optional[str] = None,
        render_backend: str = "gsplat",
        render_resolution: int = 512,
        render_num_frames: int = 120,
        render_frame_stride: int = 1,
        render_bg_color=(0, 0, 0),
    ):
        """
        Run dual-flow RelaxFlow with semantic prior. prior_images is list of file paths/np/tensor.
        """
        prior_images, prior_masks = self._normalize_prior_inputs(prior_images, prior_masks)
        if prior_images is None or len(prior_images) == 0:
            raise ValueError("Must provide at least one prior image for RelaxFlow")
        if image_as not in ("part", "scene"):
            raise ValueError("image_as must be one of: 'part', 'scene'")
        if prior_use_cropped_only is not None:
            self.prior_use_cropped_only = prior_use_cropped_only
        if prior_only_cropped_img_and_mask is not None:
            self.prior_only_cropped_img_and_mask = prior_only_cropped_img_and_mask
        self.save_render_dir = save_render_dir

        image_rgba = self.merge_mask_to_rgba(image, mask) # omg, this one let [0,1] mask * 255; but merge_image_and_mask did not!!!!!!
        image_rgba_path = os.path.join(save_render_dir, "image_rgba.png")
        imageio.imwrite(image_rgba_path, image_rgba)
        logger.info(f"Saved image_rgba to {image_rgba_path}")

        prior_blur_sigma = (
            prior_blur_sigma if prior_blur_sigma is not None else self.prior_blur_sigma
        )
        blur_attn_type = (
            blur_attn_type if blur_attn_type is not None else self.blur_attn_type
        )
        gating_schedule = resolve_gating_schedule(
            gating_schedule, gating_schedule_name, self.gating_schedule
        )
        gating_args = gating_args or {}
        flow_combine_fn = resolve_flow_blend(
            flow_combine_fn, flow_blend_name, self.prior_flow_combiner
        )
        flow_blend_args = flow_blend_args or {}
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
        stage2_prior_pooling = (
            prior_pooling if stage2_prior_pooling is None else stage2_prior_pooling
        )
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
        stage2_prior_pooling_debug = (
            prior_pooling_debug
            if stage2_prior_pooling_debug is None
            else stage2_prior_pooling_debug
        )
        stage2_prior_pooling_debug_stride = (
            prior_pooling_debug_stride
            if stage2_prior_pooling_debug_stride is None
            else stage2_prior_pooling_debug_stride
        )
        stage2_prior_pooling_debug_max_tokens = (
            prior_pooling_debug_max_tokens
            if stage2_prior_pooling_debug_max_tokens is None
            else stage2_prior_pooling_debug_max_tokens
        )

        need_branch_latents = (
            return_branch_outputs
            or save_render_dir
        )
        pointmap_dict = self.compute_pointmap(image_rgba, pointmap)
        if estimate_plane:
            return self.estimate_plane(pointmap_dict, image_rgba)
        pointmap = pointmap_dict["pointmap"]
        obs_input_dict = self.preprocess_image(
            image_rgba, self.ss_preprocessor, pointmap=pointmap
        )
        # Allow routing obs image/mask into the "cropped" slot (image/mask) vs the "scene" slot (rgb_image/rgb_image_mask).
        # The downstream condition embedder expects keys like:
        #   image/mask/pointmap   -> cropped (object-centric) tokens
        #   rgb_image/rgb_image_mask/rgb_pointmap -> full-scene tokens
        # If image_as == "scene", we copy rgb_* into the cropped keys so cropped-only modes use scene content.
        if image_as == "scene":
            if "rgb_image" in obs_input_dict:
                obs_input_dict["image"] = obs_input_dict["rgb_image"]
            if "rgb_image_mask" in obs_input_dict:
                obs_input_dict["mask"] = obs_input_dict["rgb_image_mask"]
            if "rgb_pointmap" in obs_input_dict:
                obs_input_dict["pointmap"] = obs_input_dict["rgb_pointmap"]
            if "rgb_pointmap_scale" in obs_input_dict:
                obs_input_dict["pointmap_scale"] = obs_input_dict["rgb_pointmap_scale"]
            if "rgb_pointmap_shift" in obs_input_dict:
                obs_input_dict["pointmap_shift"] = obs_input_dict["rgb_pointmap_shift"]

        slat_input_dict = self.preprocess_image(image_rgba, self.slat_preprocessor)
        if image_as == "scene":
            if "rgb_image" in slat_input_dict:
                slat_input_dict["image"] = slat_input_dict["rgb_image"]
            if "rgb_image_mask" in slat_input_dict:
                slat_input_dict["mask"] = slat_input_dict["rgb_image_mask"]
        if seed is not None:
            global_seed, ss_seed, slat_seed = self._resolve_seed_triplet(seed)
            if global_seed is not None:
                torch.manual_seed(global_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(global_seed)
                np.random.seed(global_seed % (2**32 - 1))
                random.seed(global_seed)
            self.set_model_seeds(ss_seed, slat_seed)

        prior_tokens = self.embed_prior_images(
            prior_images,
            prior_mask_list=prior_masks,
            pool_mode=prior_pooling,
            pool_temperature=prior_pooling_temperature,
            pool_agreement_boost=prior_pooling_agreement_boost,
            pool_debug=prior_pooling_debug,
            pool_debug_dir=save_render_dir,
            pool_debug_tag="stage1",
            pool_debug_stride=prior_pooling_debug_stride,
            pool_debug_max_tokens=prior_pooling_debug_max_tokens,
        )
        prior_tokens_slat = self.embed_prior_images(
            prior_images,
            prior_mask_list=prior_masks,
            condition_embedder=self.condition_embedders["slat_condition_embedder"],
            preprocessor=self.slat_preprocessor,
            pool_mode=stage2_prior_pooling,
            pool_temperature=stage2_prior_pooling_temperature,
            pool_agreement_boost=stage2_prior_pooling_agreement_boost,
            pool_debug=stage2_prior_pooling_debug,
            pool_debug_dir=save_render_dir,
            pool_debug_tag="stage2",
            pool_debug_stride=stage2_prior_pooling_debug_stride,
            pool_debug_max_tokens=stage2_prior_pooling_debug_max_tokens,
        )
        ##########
        origingal_flag = compute_original
        outputs_ori = None
        if origingal_flag:
            # SAM3D-Objects: using original SAM 3D Object model and pipeline
            logger.log("STAGE", "Sampling original result")
            ss_original_return_dict = self.sample_sparse_structure(
                    obs_input_dict,
                    inference_steps=stage1_inference_steps,
                    use_distillation=use_stage1_distillation,
                )
            # We could probably use the decoder from the models themselves
            pointmap_scale = obs_input_dict.get("pointmap_scale", None)
            pointmap_shift = obs_input_dict.get("pointmap_shift", None)
            ss_original_return_dict.update(
                self.pose_decoder(
                    ss_original_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )

            logger.info(f"Rescaling scale by {ss_original_return_dict['downsample_factor']} after downsampling")
            ss_original_return_dict["scale"] = ss_original_return_dict["scale"] * ss_original_return_dict["downsample_factor"]
                
            coords_ori = ss_original_return_dict["coords"]
            slat_ori = self.sample_slat(
                slat_input_dict,
                coords_ori,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            outputs_ori = self.decode_slat(
                slat_ori, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs_ori = self.postprocess_slat_output(
                outputs_ori, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            self._save_video_and_ply(
                    outputs_ori,
                    os.path.join(save_render_dir, "original"),
                    "original",
                    render_backend=render_backend,
                    render_resolution=render_resolution,
                    render_num_frames=render_num_frames,
                    render_frame_stride=render_frame_stride,
                    render_bg_color=render_bg_color,
                    save_overlay=False,
                )
        ##########
        # using RelaxFlow pipeline
        logger.log("STAGE", "Stage 1: Sparse Structure (SS_RelaxFlow) Sampling")
        stage1_feat_blur_type = feat_blur_type if feat_blur_stage1 else "none"
        stage1_feat_blur_sigma = feat_blur_sigma if feat_blur_stage1 else 0.0
        stage1_feat_noise_strength = feat_noise_strength if feat_blur_stage1 else 0.0
        ss_sampled = self.sample_sparse_structure_relaxflow(
            obs_input_dict,
            prior_tokens,
            gating_schedule=gating_schedule,
            prior_weight=prior_weight,
            gating_args=gating_args,
            inference_steps=stage1_inference_steps,
            use_distillation=use_stage1_distillation,
            only_cropped_img=only_cropped_img,
            only_cropped_img_and_mask=only_cropped_img_and_mask,
            prior_blur_sigma=prior_blur_sigma,
            blur_attn_type=blur_attn_type,
            feat_blur_type=stage1_feat_blur_type,
            feat_blur_sigma=stage1_feat_blur_sigma,
            feat_noise_type=feat_noise_type,
            feat_noise_strength=stage1_feat_noise_strength,
            flow_combine_fn=flow_combine_fn,
            flow_blend_args=flow_blend_args,
            return_branch_latents=need_branch_latents,
        )
        if need_branch_latents and isinstance(ss_sampled, dict):
            ss_return_dict = ss_sampled["blend"]
            ss_return_obs = ss_sampled.get("obs_only", None)
            ss_return_prior = ss_sampled.get("prior_only", None)
        else:
            ss_return_dict = ss_sampled
            ss_return_obs = ss_return_prior = None

        # ####### Geometry Mask ####### #
        logger.log("STAGE", "Computing Geometry Mask")
        geometry_mask = None
        mask_debug_info = None
        linear_blend_fn = FLOW_BLEND_FNS.get("linear")

        def _unwrap_callable(fn):
            if hasattr(fn, "func"):
                return _unwrap_callable(fn.func)
            return fn

        stage2_fn_resolved = _unwrap_callable(stage2_flow_combine_fn)
        fn_name = getattr(stage2_fn_resolved, "__name__", stage2_fn_resolved.__class__.__name__)
        blend_supports_geometry_mask = stage2_fn_resolved in (
            linear_blend_fn,
        ) or fn_name in {"blend_linear"}
        assert blend_supports_geometry_mask, "Stage2 flow blend must support geometry mask"

        should_attempt_geometry_mask = True
        if should_attempt_geometry_mask:
            if ss_return_obs is None:
                logger.warning("[GeomMask] obs_only pose missing; geometry mask skipped.")
            else:
                pointmap_scale = obs_input_dict.get("pointmap_scale", None)
                pointmap_shift = obs_input_dict.get("pointmap_shift", None)
                pose_info_for_mask = self.pose_decoder(
                    ss_return_obs,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
                scale_for_mask = pose_info_for_mask.get("scale", None)
                if scale_for_mask is not None:
                    downsample_factor = ss_return_obs.get("downsample_factor", 1.0)
                    scale_for_mask = scale_for_mask * downsample_factor
                if (
                    pose_info_for_mask.get("translation") is not None
                    and pose_info_for_mask.get("rotation") is not None
                    and scale_for_mask is not None
                ):
                    H, W = image_rgba.shape[:2]
                    render_res = max(H, W)
                    intrinsics = intrinsics_from_fov(40.0, render_res)
                    if render_res != W or render_res != H:
                        sx = W / render_res
                        sy = H / render_res
                        intrinsics[0, 0] *= sx
                        intrinsics[0, 2] *= sx
                        intrinsics[1, 1] *= sy
                        intrinsics[1, 2] *= sy
                    debug_enabled = False
                    coords_for_mask = ss_return_dict.get("coords", None)

                    mask_result = compute_geometry_visibility_mask(
                        coords_for_mask,
                        pose_info_for_mask["translation"],
                        pose_info_for_mask["rotation"],
                        scale_for_mask,
                        intrinsics,
                        image_rgba.shape[:2],
                        condition_mask=image_rgba[:, :, 3] if use_geometry_mask_condition_mask else None,
                        soft_falloff=geometry_mask_soft_falloff, 
                        param_tolerance_scale=geometry_mask_param_tolerance_scale,
                        param_dilate_scale=geometry_mask_param_dilate_scale,
                        return_debug=debug_enabled,
                    )

                    if isinstance(mask_result, tuple):
                        geometry_mask, mask_debug_info = mask_result
                    else:
                        geometry_mask = mask_result

                    geometry_mask = geometry_mask.to(self.device)
                    mask_mean = float(geometry_mask.mean().detach().cpu())
                    mask_vis_frac = float((geometry_mask > 0.5).float().mean().detach().cpu())
                    logger.info(
                        f"Geometry mask stats: mean={mask_mean:.4f}, frac>0.5={mask_vis_frac:.4f}, "
                        f"soft_falloff={geometry_mask_soft_falloff}, param_tolerance_scale={geometry_mask_param_tolerance_scale}, param_dilate_scale={geometry_mask_param_dilate_scale}"
                    )
                else:
                    logger.warning("[GeomMask] Pose incomplete; geometry mask skipped.")
        else:
            logger.info("Geometry mask disabled (geometry_mask_soft_falloff unset).")

        if geometry_mask is not None and blend_supports_geometry_mask:
            stage2_flow_blend_args = dict(stage2_flow_blend_args)
            stage2_flow_blend_args["geometry_mask"] = geometry_mask
        elif geometry_mask is None and should_attempt_geometry_mask:
            logger.warning(
                "Geometry mask could not be computed; no diagnostics will be saved. "
                "Stage2 blend fn: %s (supports_mask=%s)",
                fn_name,
                blend_supports_geometry_mask,
            )
        elif geometry_mask is not None and not blend_supports_geometry_mask:
            logger.warning(
                "Geometry mask computed but not applied (stage2 blend '%s' unsupported).",
                fn_name,
            )

        # ###########
        logger.log("STAGE", "Stage 2: Texture & Refinement (SLAT_RelaxFlow) Sampling")
        # this will return 3 outputs: blend, obs_only, prior_only
        slat_relaxflow = self.sample_slat_relaxflow(
            slat_input_dict,
            ss_return_dict["coords"],
            prior_tokens_slat,
            gating_schedule=stage2_gating_schedule,
            prior_weight=stage2_prior_weight,
            gating_args=stage2_gating_args,
            inference_steps=stage2_inference_steps,
            use_distillation=use_stage2_distillation,
            prior_blur_sigma=stage2_prior_blur_sigma,
            blur_attn_type=stage2_blur_attn_type,
            feat_blur_type=feat_blur_type,
            feat_blur_sigma=feat_blur_sigma,
            feat_noise_type=feat_noise_type,
            feat_noise_strength=feat_noise_strength,
            flow_combine_fn=stage2_flow_combine_fn,
            flow_blend_args=stage2_flow_blend_args,
            return_branch_outputs=need_branch_latents,
            return_obs_branch=need_branch_latents,
            return_prior_branch=need_branch_latents and compute_blend_prior_slat,
        )

        if need_branch_latents and isinstance(slat_relaxflow, dict):
            slat_blend = slat_relaxflow["blend"]
            slat_obs_blend = slat_relaxflow.get("obs_only", None)
            slat_prior_blend = (
                slat_relaxflow.get("prior_only", None) if compute_blend_prior_slat else None
            )
        else:
            slat_blend = slat_relaxflow
            slat_obs_blend = slat_prior_blend = None

        # ######### Decode Full Output ######### #
        logger.log("STAGE", "Decoding Full Output")
        outputs = self._decode_full_output(
            ss_return_dict,
            obs_input_dict,
            pointmap_dict,
            slat_input_dict,
            image_rgba,
            stage2_inference_steps,
            use_stage2_distillation,
            decode_formats,
            with_mesh_postprocess,
            with_texture_baking,
            use_vertex_color,
            with_layout_postprocess,
            slat_latent=slat_blend,
        )

        # ######### Decode Branch Outputs ######### #
        branch_outputs = {}
        if need_branch_latents and ss_return_obs is not None and ss_return_prior is not None:
            logger.log("STAGE", "Decoding Obs Branch Outputs")
            # obs_ss + obs_slat
            slat_obs_ss = self.sample_slat(
                slat_input_dict,
                ss_return_obs["coords"],
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            branch_outputs["obs_only"] = self._decode_full_output(
                ss_return_obs,
                obs_input_dict,
                pointmap_dict,
                slat_input_dict,
                image_rgba,
                stage2_inference_steps,
                use_stage2_distillation,
                decode_formats,
                with_mesh_postprocess,
                with_texture_baking,
                use_vertex_color,
                with_layout_postprocess,
                slat_latent=slat_obs_ss,
            )
            # prior_ss + prior_slat (prior-only)
            logger.log("STAGE", "Decoding Prior Branch Outputs")
            prior_slat_bundle = self.sample_slat_relaxflow(
                slat_input_dict,
                ss_return_prior["coords"],
                prior_tokens_slat,
                gating_schedule=stage2_gating_schedule,
                prior_weight=stage2_prior_weight,
                gating_args=stage2_gating_args,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
                prior_blur_sigma=stage2_prior_blur_sigma,
                blur_attn_type=stage2_blur_attn_type,
                feat_blur_type=feat_blur_type,
                feat_blur_sigma=feat_blur_sigma,
                feat_noise_type=feat_noise_type,
                feat_noise_strength=feat_noise_strength,
                flow_combine_fn=stage2_flow_combine_fn,
                flow_blend_args={k: v for k, v in stage2_flow_blend_args.items() if k != "geometry_mask"},
                return_branch_outputs=True,
                return_obs_branch=False,
                return_prior_branch=True,
            )
            if isinstance(prior_slat_bundle, dict):
                slat_prior_ss = prior_slat_bundle.get("prior_only") or prior_slat_bundle.get("blend")
            else:
                slat_prior_ss = prior_slat_bundle

            branch_outputs["prior_only"] = self._decode_full_output(
                ss_return_prior,
                obs_input_dict,
                pointmap_dict,
                slat_input_dict,
                image_rgba,
                stage2_inference_steps,
                use_stage2_distillation,
                decode_formats,
                with_mesh_postprocess,
                with_texture_baking,
                use_vertex_color,
                with_layout_postprocess,
                slat_latent=slat_prior_ss,
            )

            # blend_ss + obs_slat (reuse obs slat sampled with blend coords)
            if slat_obs_blend is not None:
                logger.log("STAGE", "Decoding Blend Obs Branch Outputs")
                branch_outputs["blend_obs_slat"] = self._decode_full_output(
                    ss_return_dict,
                    obs_input_dict,
                    pointmap_dict,
                    slat_input_dict,
                    image_rgba,
                    stage2_inference_steps,
                    use_stage2_distillation,
                    decode_formats,
                    with_mesh_postprocess,
                    with_texture_baking,
                    use_vertex_color,
                    with_layout_postprocess,
                    slat_latent=slat_obs_blend,
                )

            # blend_ss + prior_slat
            if compute_blend_prior_slat and slat_prior_blend is not None:
                logger.log("STAGE", "Decoding Blend Prior Branch Outputs")
                branch_outputs["blend_prior_slat"] = self._decode_full_output(
                    ss_return_dict,
                    obs_input_dict,
                    pointmap_dict,
                    slat_input_dict,
                    image_rgba,
                    stage2_inference_steps,
                    use_stage2_distillation,
                    decode_formats,
                    with_mesh_postprocess,
                    with_texture_baking,
                    use_vertex_color,
                    with_layout_postprocess,
                    slat_latent=slat_prior_blend,
                )

        # ######### Save Render ######### #
        if save_render_dir:
            logger.log("STAGE", "Saving Render")
            # Use obs branch pose for all overlays when available
            overlay_pose = None
            if branch_outputs and "obs_only" in branch_outputs:
                overlay_pose = branch_outputs["obs_only"]

            self._save_video_and_ply(
                outputs,
                os.path.join(save_render_dir, "relaxflow"),
                "relaxflow",
                render_backend=render_backend,
                render_resolution=render_resolution,
                render_num_frames=render_num_frames,
                render_frame_stride=render_frame_stride,
                render_bg_color=render_bg_color,
                original_image=image_rgba,
                save_overlay=True,
                overlay_pose=overlay_pose,
            )
            if branch_outputs:
                if "obs_only" in branch_outputs:
                    logger.log("STAGE", "Saving Obs Branch Render")
                    self._save_video_and_ply(
                        branch_outputs["obs_only"],
                        os.path.join(save_render_dir, "obs_only"),
                        "obs_only",
                        render_backend=render_backend,
                        render_resolution=render_resolution,
                        render_num_frames=render_num_frames,
                        render_frame_stride=render_frame_stride,
                        render_bg_color=render_bg_color,
                        original_image=image_rgba,
                        save_overlay=False,
                        overlay_pose=overlay_pose,
                    )
                if "prior_only" in branch_outputs:
                    logger.log("STAGE", "Saving Prior Branch Render")
                    self._save_video_and_ply(
                        branch_outputs["prior_only"],
                        os.path.join(save_render_dir, "prior_only"),
                        "prior_only",
                        render_backend=render_backend,
                        render_resolution=render_resolution,
                        render_num_frames=render_num_frames,
                        render_frame_stride=render_frame_stride,
                        render_bg_color=render_bg_color,
                        original_image=image_rgba,
                        save_overlay=False,
                        overlay_pose=overlay_pose,
                    )
                if "blend_obs_slat" in branch_outputs:
                    logger.log("STAGE", "Saving Blend Obs Branch Render")
                    self._save_video_and_ply(
                        branch_outputs["blend_obs_slat"],
                        os.path.join(save_render_dir, "blend_obs_slat"),
                        "blend_obs_slat",
                        render_backend=render_backend,
                        render_resolution=render_resolution,
                        render_num_frames=render_num_frames,
                        render_frame_stride=render_frame_stride,
                        render_bg_color=render_bg_color,
                        original_image=image_rgba,
                        save_overlay=False,
                        overlay_pose=overlay_pose,
                    )
                if "blend_prior_slat" in branch_outputs:
                    if compute_blend_prior_slat:
                        logger.log("STAGE", "Saving Blend Prior Branch Render")
                        self._save_video_and_ply(
                            branch_outputs["blend_prior_slat"],
                            os.path.join(save_render_dir, "blend_prior_slat"),
                            "blend_prior_slat",
                            render_backend=render_backend,
                            render_resolution=render_resolution,
                            render_num_frames=render_num_frames,
                            render_frame_stride=render_frame_stride,
                            render_bg_color=render_bg_color,
                            original_image=image_rgba,
                            save_overlay=False,
                            overlay_pose=overlay_pose,
                        )

        if branch_outputs and return_branch_outputs:
            return {
                "relaxflow": outputs,
                "original": outputs_ori if origingal_flag else None,
                **branch_outputs,
            }
        if return_branch_outputs:
            return {
                "relaxflow": outputs,
                "original": outputs_ori if origingal_flag else None,
            }
        return outputs

    def sample_sparse_structure_relaxflow(
        self,
        obs_input_dict: dict,
        prior_tokens: torch.Tensor,
        gating_schedule: Callable[[int, int], float],
        prior_weight: float = 1.0,
        gating_args: Optional[dict] = None,
        inference_steps=None,
        use_distillation=False,
        only_cropped_img: bool = False,
        only_cropped_img_and_mask: bool = False,
        prior_blur_sigma: Optional[float] = None,
        blur_attn_type: Optional[str] = None,
        feat_blur_type: str = "none",
        feat_blur_sigma: float = 1.0,
        feat_noise_type: str = "gaussian",
        feat_noise_strength: float = 0.0,
        flow_combine_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor]
        ] = None,
        flow_blend_args: Optional[dict] = None,
        return_branch_latents: bool = False,
        **kwargs,
    ):
        """
        RelaxFlow sparse structure sampling:
        - obs branch uses normal conditions
        - prior branch uses prior_tokens (blurred attention)
        - dynamics = (1-alpha_t) * e_obs + alpha_t * prior_weight * e_prior
        """
        gating_args = gating_args or {}
        prior_blur_sigma = (
            self.prior_blur_sigma if prior_blur_sigma is None else prior_blur_sigma
        )
        blur_attn_type = blur_attn_type or self.blur_attn_type
        blend_fn = flow_combine_fn or self.prior_flow_combiner
        flow_blend_args = flow_blend_args or {}

        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]

        if use_distillation:
            logger.info("Using shortcut distillation mode")
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            logger.info("Not using shortcut distillation mode")
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm

        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = obs_input_dict["image"]
        bs = image.shape[0]
        logger.info(
            "RelaxFlow sampling: inference_steps={}, strength={}, interval={}, rescale_t={}, cfg_strength_pm={}",
            ss_generator.inference_steps,
            ss_generator.reverse_fn.strength,
            ss_generator.reverse_fn.interval,
            ss_generator.rescale_t,
            ss_generator.reverse_fn.strength_pm,
        )

        # Build latent shape dict (mm_dit or standard)
        if self.is_mm_dit():
            latent_shape_dict = {
                k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
            }
        else:
            latent_shape_dict = (bs,) + (4096, 8)

        # Condition args for observed branch (tokens)
        # CRITICAL: Match baseline's logic exactly
        if only_cropped_img:
            extra_condition_kwargs = {"only_cropped_img": True}
        elif only_cropped_img_and_mask:
            extra_condition_kwargs = {"only_cropped_img_and_mask": True}
        else:
            extra_condition_kwargs = None
        # observation branch uses official condition embedder, can choose to only keep cropped tokens
        condition_args_obs, condition_kwargs_obs = self.get_condition_input(
            self.condition_embedders["ss_condition_embedder"],
            obs_input_dict,
            self.ss_condition_input_mapping,
            extra_condition_kwargs=extra_condition_kwargs,
        )

        # Prior tokens: (B, T, D) -> ensure batch matches
        prior_tokens = prior_tokens.to(image.device)
        if prior_tokens.shape[0] == 1 and bs > 1:
            prior_tokens = prior_tokens.expand(bs, -1, -1)

        condition_args_prior = (prior_tokens,)
        condition_kwargs_prior: Dict = {}

        blur_context = RELAXFLOWAttnBlurContext(
            ss_generator,
            sigma=prior_blur_sigma,
            attn_type=blur_attn_type,
            feat_blur_type=feat_blur_type,
            feat_blur_sigma=feat_blur_sigma,
            feat_noise_type=feat_noise_type,
            feat_noise_strength=feat_noise_strength,
        )

        with torch.no_grad():
            with self._shape_autocast_context():
                x_obs_dict = None
                x_prior = None
                
                # Run obs_only branch if needed for return_branch_latents
                if return_branch_latents:
                    logger.info(
                        "Running obs_only branch (completely independent, before any other operations): "
                        f"strength={ss_generator.reverse_fn.strength}, "
                        f"strength_pm={ss_generator.reverse_fn.strength_pm}, "
                        f"interval={ss_generator.reverse_fn.interval}"
                    )
                    return_dict_obs = ss_generator(
                        latent_shape_dict,
                        image.device,
                        *condition_args_obs,
                        **condition_kwargs_obs,
                    )
                    if not self.is_mm_dit():
                        return_dict_obs = {"shape": return_dict_obs}
                    x_obs_dict = tree_tensor_map(lambda t: t.cpu(), return_dict_obs)
                
                if hasattr(ss_generator, "_prepare_t_and_d"):
                    logger.info("Using ShortCut-style")
                    t_seq_np, d_val = ss_generator._prepare_t_and_d()
                    t_seq = torch.as_tensor(
                        t_seq_np, device=image.device, dtype=torch.float32
                    )
                else:
                    logger.info("Using Standard FlowMatching-style")
                    t_seq = ss_generator._prepare_t().to(image.device)
                    d_val = None
                
                step_count = len(t_seq) - 1
                alpha_seq = self._build_alpha_schedule(
                    step_count, gating_schedule, gating_args
                )
                logger.info(f"Alpha sequence (length {len(alpha_seq)}): {alpha_seq}")
                
                x_0 = ss_generator._generate_noise(latent_shape_dict, image.device)
                x_0_prior = (
                    tree_tensor_map(lambda t: t.clone(), x_0) if return_branch_latents else None
                )
                
                if return_branch_latents:
                    logger.info("Running prior_only branch (before main loop)")
                    if hasattr(ss_generator, "_prepare_t_and_d"):
                        solver = ss_generator._solver
                        with blur_context:
                            for x_prior, t in solver.solve_iter(
                                ss_generator._generate_dynamics,
                                x_0_prior,
                                t_seq,
                                d_val,
                                *condition_args_prior,
                                **condition_kwargs_prior,
                            ):
                                pass
                    else:
                        solver = ss_generator._solver
                        with blur_context:
                            for x_prior, t in solver.solve_iter(
                                ss_generator._generate_dynamics,
                                x_0_prior,
                                t_seq,
                                *condition_args_prior,
                                **condition_kwargs_prior,
                            ):
                                pass
                    x_prior = tree_tensor_map(lambda t: t.cpu(), x_prior)

                step_state = {"i": 0}

                def _next_alpha():
                    idx = min(step_state["i"], len(alpha_seq) - 1)
                    step_state["i"] += 1
                    return alpha_seq[idx]

                if hasattr(ss_generator, "_prepare_t_and_d"):
                    def relaxflow_dynamics(x_t, t, d, *unused_args, **unused_kwargs):
                        alpha_t = _next_alpha()
                        logger.info(f"relaxflow_dynamics: t: {t}, alpha_t: {alpha_t}")
                        if prior_weight <= 0 or alpha_t <= 0:
                            logger.info(f"Prior weight <= 0 or alpha_t <= 0, using obs branch")
                            t_tensor = torch.tensor(
                                [float(t) * ss_generator.time_scale],
                                device=image.device,
                                dtype=torch.float32,
                            )
                            d_tensor = torch.tensor(
                                [d * ss_generator.time_scale],
                                device=image.device,
                                dtype=torch.float32,
                            )
                            return ss_generator.reverse_fn(
                                x_t,
                                t_tensor,
                                *condition_args_obs,
                                d=d_tensor,
                                **condition_kwargs_obs,
                            )
                        t_tensor = torch.tensor(
                            [float(t) * ss_generator.time_scale],
                            device=image.device,
                            dtype=torch.float32,
                        )
                        d_tensor = torch.tensor(
                            [d * ss_generator.time_scale],
                            device=image.device,
                            dtype=torch.float32,
                        )
                        #########
                        # obs branch: compute velocity using blend state x_t
                        e_obs = ss_generator.reverse_fn(
                            x_t,
                            t_tensor,
                            *condition_args_obs,
                            d=d_tensor,
                            **condition_kwargs_obs,
                        )
                        #########
                        # prior branch: compute velocity using blend state x_t
                        with blur_context:
                            e_prior = ss_generator.reverse_fn(
                                x_t,
                                t_tensor,
                                *condition_args_prior,
                                d=d_tensor,
                                **condition_kwargs_prior,
                            )

                        return tree_tensor_map(
                            lambda a, b, mt=alpha_t, pw=prior_weight: self._mode_blend_fn(
                                blend_fn, a, b, mt, pw, blend_kwargs=flow_blend_args
                            ),
                            e_obs,
                            e_prior,
                        )

                    solver = ss_generator._solver
                    for x_t, t in solver.solve_iter(
                        relaxflow_dynamics,
                        x_0,
                        t_seq,
                        d_val,
                    ):
                        pass
                else:
                    # Standard FlowMatching-style
                    def relaxflow_dynamics(x_t, t, *unused_args, **unused_kwargs):
                        alpha_t = _next_alpha()
                        if prior_weight <= 0 or alpha_t <= 0:
                            t_tensor = torch.tensor(
                                [float(t) * ss_generator.time_scale],
                                device=image.device,
                                dtype=torch.float32,
                            )
                            return ss_generator.reverse_fn(
                                x_t, t_tensor, *condition_args_obs, **condition_kwargs_obs
                            )
                        # normal case: compute velocities using blend state x_t
                        t_tensor = torch.tensor(
                            [float(t) * ss_generator.time_scale],
                            device=image.device,
                            dtype=torch.float32,
                        )
                        e_obs = ss_generator.reverse_fn(
                            x_t, t_tensor, *condition_args_obs, **condition_kwargs_obs
                        )
                        with blur_context:
                            e_prior = ss_generator.reverse_fn(
                                x_t,
                                t_tensor,
                                *condition_args_prior,
                                **condition_kwargs_prior,
                            )
                        return tree_tensor_map(
                            lambda a, b, mt=alpha_t, pw=prior_weight: self._mode_blend_fn(
                                blend_fn, a, b, mt, pw, blend_kwargs=flow_blend_args
                            ),
                            e_obs,
                            e_prior,
                        )

                    solver = ss_generator._solver
                    for x_t, t in solver.solve_iter(
                        relaxflow_dynamics,
                        x_0,
                        t_seq,
                    ):
                        pass

        def _latent_to_ss_dict(shape_latent_tensor):
            # This function handles tensor inputs from solver iteration
            # For mm_dit, shape_latent_tensor is already a dict from solver
            # For non-mm_dit, shape_latent_tensor is a tensor
            if isinstance(shape_latent_tensor, dict):
                # mm_dit case: already a dict
                out = tree_tensor_map(lambda t: t.to(self.device), shape_latent_tensor)
                shape_latent = out["shape"]
            else:
                # non-mm_dit case: tensor input
                out = {"shape": shape_latent_tensor.to(self.device)}
                shape_latent = out["shape"]
            
            ss = ss_decoder(
                shape_latent.permute(0, 2, 1)
                .contiguous()
                .view(shape_latent.shape[0], 8, 16, 16, 16)
            )
            coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
            out["coords_original"] = coords
            if self.downsample_ss_dist > 0:
                coords = prune_sparse_structure(
                    coords,
                    max_neighbor_axes_dist=self.downsample_ss_dist,
                )
            coords, downsample_factor = downsample_sparse_structure(coords)
            out["coords"] = coords
            out["downsample_factor"] = downsample_factor
            return out

        from sam3d_objects.pipeline.inference_utils import (
            prune_sparse_structure,
            downsample_sparse_structure,
        )

        main_dict = _latent_to_ss_dict(x_t)
        branch_dicts = {}
        if return_branch_latents:
            # For obs_only: process exactly like baseline
            if x_obs_dict is not None:
                shape_latent_obs = x_obs_dict["shape"].to(self.device)
                ss_obs = ss_decoder(
                    shape_latent_obs.permute(0, 2, 1)
                    .contiguous()
                    .view(shape_latent_obs.shape[0], 8, 16, 16, 16)
                )
                coords_obs = torch.argwhere(ss_obs > 0)[:, [0, 2, 3, 4]].int()
                branch_dicts["obs_only"] = dict(x_obs_dict)  # Copy the dict
                branch_dicts["obs_only"]["coords_original"] = coords_obs
                original_shape_obs = coords_obs.shape
                if self.downsample_ss_dist > 0:
                    coords_obs = prune_sparse_structure(
                        coords_obs,
                        max_neighbor_axes_dist=self.downsample_ss_dist,
                    )
                coords_obs, downsample_factor_obs = downsample_sparse_structure(coords_obs)
                logger.info(
                    f"Downsampled obs_only coords from {original_shape_obs[0]} to {coords_obs.shape[0]}"
                )
                branch_dicts["obs_only"]["coords"] = coords_obs
                branch_dicts["obs_only"]["downsample_factor"] = downsample_factor_obs
            
            # For prior_only: use _latent_to_ss_dict (from solver iteration)
            if x_prior is not None:
                branch_dicts["prior_only"] = _latent_to_ss_dict(x_prior)

        ss_generator.inference_steps = prev_inference_steps
        if return_branch_latents:
            # Ensure all branch tensors reside on self.device for downstream decoders
            branch_dicts = {
                k: tree_tensor_map(
                    lambda t: t.to(self.device) if torch.is_tensor(t) else t, v
                )
                for k, v in branch_dicts.items()
            }
            return {"blend": main_dict, **branch_dicts}
        return main_dict

    def sample_slat_relaxflow(
        self,
        slat_input: dict,
        coords: torch.Tensor,
        prior_tokens: torch.Tensor,
        gating_schedule: Callable[[int, int], float],
        prior_weight: float = 1.0,
        gating_args: Optional[dict] = None,
        inference_steps=None,
        use_distillation=False,
        prior_blur_sigma: Optional[float] = None,
        blur_attn_type: Optional[str] = None,
        feat_blur_type: str = "none",
        feat_blur_sigma: float = 1.0,
        feat_noise_type: str = "gaussian",
        feat_noise_strength: float = 0.0,
        flow_combine_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor]
        ] = None,
        flow_blend_args: Optional[dict] = None,
        return_branch_outputs: bool = False,
        return_obs_branch: bool = True,
        return_prior_branch: bool = True,
    ) -> Union[sp.SparseTensor, Dict[str, sp.SparseTensor]]:
        slat_generator = self.models["slat_generator"]
        gating_args = gating_args or {}
        prior_blur_sigma = (
            self.prior_blur_sigma if prior_blur_sigma is None else prior_blur_sigma
        )
        blur_attn_type = blur_attn_type or self.blur_attn_type
        blend_fn = flow_combine_fn or self.prior_flow_combiner
        flow_blend_args = flow_blend_args or {}

        if use_distillation:
            slat_generator.no_shortcut = False
            slat_generator.reverse_fn.strength = 0
        else:
            slat_generator.no_shortcut = True
            slat_generator.reverse_fn.strength = self.slat_cfg_strength

        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps

        image = slat_input["image"]
        bs = image.shape[0]
        device = image.device
        latent_shape = (bs,) + (coords.shape[0], 8)

        logger.info(
            "RelaxFlow SLAT sampling: inference_steps={}, strength={}, interval={}, rescale_t={}",
            slat_generator.inference_steps,
            slat_generator.reverse_fn.strength,
            slat_generator.reverse_fn.interval,
            slat_generator.rescale_t,
        )

        def _latent_to_slat(latent: torch.Tensor) -> sp.SparseTensor:
            # Normalize latent shape to (N, C) like the baseline sample_slat
            if isinstance(latent, (list, tuple)):
                latent = latent[0]
            # remove batch dim if present
            if latent.dim() == 3 and latent.shape[0] == 1:
                latent = latent[0]
            feats = latent.contiguous()
            coords_device = coords.to(device, non_blocking=True)
            slat_tensor = sp.SparseTensor(coords=coords_device, feats=feats).to(device)
            slat_tensor = slat_tensor * self.slat_std.to(device) + self.slat_mean.to(device)
            return slat_tensor

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                condition_args_obs, condition_kwargs_obs = self.get_condition_input(
                    self.condition_embedders["slat_condition_embedder"],
                    slat_input,
                    self.slat_condition_input_mapping,
                )
                condition_args_obs += (coords.cpu().numpy(),)

                prior_tokens = prior_tokens.to(device)
                if prior_tokens.shape[0] == 1 and bs > 1:
                    prior_tokens = prior_tokens.expand(bs, -1, -1)
                condition_args_prior = (prior_tokens, coords.cpu().numpy())
                condition_kwargs_prior: Dict = {}

                blur_context = RELAXFLOWAttnBlurContext(
                    slat_generator,
                    sigma=prior_blur_sigma,
                    attn_type=blur_attn_type,
                    feat_blur_type=feat_blur_type,
                    feat_blur_sigma=feat_blur_sigma,
                    feat_noise_type=feat_noise_type,
                    feat_noise_strength=feat_noise_strength,
                    debug_v=False,
                    debug_tag="stage2_slat_prior",
                )

                t_seq = slat_generator._prepare_t().to(device)
                alpha_seq = self._build_alpha_schedule(
                    len(t_seq) - 1, gating_schedule, gating_args
                )
                step_state = {"i": 0}

                def _next_alpha():
                    idx = min(step_state["i"], len(alpha_seq) - 1)
                    step_state["i"] += 1
                    return alpha_seq[idx]

                x_0 = slat_generator._generate_noise(latent_shape, device)
                run_obs_branch = return_branch_outputs and return_obs_branch
                run_prior_branch = return_branch_outputs and return_prior_branch
                x_0_obs = x_0.clone() if run_obs_branch else None
                x_0_prior = x_0.clone() if run_prior_branch else None
                solver = slat_generator._solver

                x_obs = None
                x_prior = None

                def _obs_only_dynamics(x_t, t, *unused_args, **unused_kwargs):
                    t_tensor = torch.tensor(
                        [float(t) * slat_generator.time_scale],
                        device=device,
                        dtype=torch.float32,
                    )
                    return slat_generator.reverse_fn(
                        x_t,
                        t_tensor,
                        *condition_args_obs,
                        **condition_kwargs_obs,
                    )

                def _prior_only_dynamics(x_t, t, *unused_args, **unused_kwargs):
                    t_tensor = torch.tensor(
                        [float(t) * slat_generator.time_scale],
                        device=device,
                        dtype=torch.float32,
                    )
                    with blur_context:
                        return slat_generator.reverse_fn(
                            x_t,
                            t_tensor,
                            *condition_args_prior,
                            **condition_kwargs_prior,
                        )

                if run_obs_branch:
                    for x_obs, t in solver.solve_iter(
                        _obs_only_dynamics,
                        x_0_obs,
                        t_seq,
                    ):
                        pass
                if run_prior_branch:
                    with blur_context:
                        for x_prior, t in solver.solve_iter(
                            _prior_only_dynamics,
                            x_0_prior,
                            t_seq,
                        ):
                            pass

                def relaxflow_dynamics(x_t, t, *unused_args, **unused_kwargs):
                    alpha_t = _next_alpha()
                    if prior_weight <= 0 or alpha_t <= 0:
                        t_tensor = torch.tensor(
                            [float(t) * slat_generator.time_scale],
                            device=device,
                            dtype=torch.float32,
                        )
                        return slat_generator.reverse_fn(
                            x_t,
                            t_tensor,
                            *condition_args_obs,
                            **condition_kwargs_obs,
                        )
                    t_tensor = torch.tensor(
                        [float(t) * slat_generator.time_scale],
                        device=device,
                        dtype=torch.float32,
                    )
                    e_obs = slat_generator.reverse_fn(
                        x_t, t_tensor, *condition_args_obs, **condition_kwargs_obs
                    )
                    with blur_context:
                        e_prior = slat_generator.reverse_fn(
                            x_t,
                            t_tensor,
                            *condition_args_prior,
                            **condition_kwargs_prior,
                        )
                    return tree_tensor_map(
                        lambda a, b, mt=alpha_t, pw=prior_weight: self._mode_blend_fn(
                            blend_fn, a, b, mt, pw, blend_kwargs=flow_blend_args
                        ),
                        e_obs,
                        e_prior,
                    )

                for x_t, t in solver.solve_iter(
                    relaxflow_dynamics,
                    x_0,
                    t_seq,
                ):
                    pass

        slat_generator.inference_steps = prev_inference_steps
        slat = _latent_to_slat(x_t)
        if not return_branch_outputs:
            return slat

        branch_outputs: Dict[str, sp.SparseTensor] = {"blend": slat}
        if x_obs is not None:
            branch_outputs["obs_only"] = _latent_to_slat(x_obs)
        if x_prior is not None:
            branch_outputs["prior_only"] = _latent_to_slat(x_prior)
        return branch_outputs

    def _mode_blend_fn(
        self,
        blend_fn: Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor],
        e_obs: torch.Tensor,
        e_prior: torch.Tensor,
        alpha_t: float,
        prior_weight: float,
        blend_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        blend_kwargs = blend_kwargs or {}
        return blend_fn(e_obs, e_prior, alpha_t, prior_weight, **blend_kwargs)

    def _decode_full_output(
        self,
        ss_return_dict: dict,
        obs_input_dict: dict,
        pointmap_dict: dict,
        slat_input_dict: dict,
        image_rgba,
        stage2_inference_steps,
        use_stage2_distillation,
        decode_formats,
        with_mesh_postprocess,
        with_texture_baking,
        use_vertex_color,
        with_layout_postprocess,
        slat_latent=None,
    ):
        pointmap = pointmap_dict["pointmap"]
        pts = type(self)._down_sample_img(pointmap)
        pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])

        pointmap_scale = obs_input_dict.get("pointmap_scale", None)
        pointmap_shift = obs_input_dict.get("pointmap_shift", None)
        ss_return_dict = dict(ss_return_dict)
        ss_return_dict.update(
            self.pose_decoder(
                ss_return_dict,
                scene_scale=pointmap_scale,
                scene_shift=pointmap_shift,
            )
        )
        ss_return_dict["scale"] = (
            ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
        )

        coords = ss_return_dict["coords"]
        slat = slat_latent or self.sample_slat(
            slat_input_dict,
            coords,
            inference_steps=stage2_inference_steps,
            use_distillation=use_stage2_distillation,
        )
        outputs = self.decode_slat(
            slat, self.decode_formats if decode_formats is None else decode_formats
        )
        outputs = self.postprocess_slat_output(
            outputs,
            with_mesh_postprocess,
            with_texture_baking,
            use_vertex_color,
        )
        glb = outputs.get("glb", None)

        try:
            if with_layout_postprocess and self.layout_post_optimization_method is not None:
                assert glb is not None, "require mesh to run postprocessing"
                postprocessed_pose = self.run_post_optimization(
                    deepcopy(glb),
                    pointmap_dict["intrinsics"],
                    ss_return_dict,
                    obs_input_dict,
                )
                ss_return_dict.update(postprocessed_pose)
        except Exception as e:
            logger.error(f"Error during layout post optimization: {e}", exc_info=True)

        return {
            **ss_return_dict,
            **outputs,
            "pointmap": pts.cpu().permute((1, 2, 0)),
            "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),
        }

    def _build_alpha_schedule(
        self, steps: int, gating_schedule: Callable[[int, int], float], gating_args: dict
    ) -> List[float]:
        # discretize the continuous time function into step sequence, match the solver's step length
        steps = max(int(steps), 1)
        alphas: List[float] = []
        # Filter kwargs to match the gating_schedule signature to avoid unexpected kwargs errors
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
