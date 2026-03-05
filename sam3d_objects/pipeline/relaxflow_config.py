"""
RelaxFlow Configuration Module
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import argparse

from .relaxflow_variants import GATING_SCHEDULES, FLOW_BLEND_FNS


@dataclass
class GatingConfig:
    """Configuration for gating schedule parameters."""
    
    schedule: str = "linear_cutoff"
    alpha0: float = 1.0
    mid_ratio: float = 0.4
    decay: float = 4.0
    center: float = 0.5
    sharpness: float = 10.0
    warmup: float = 0.1
    hold: float = 0.2
    gamma: float = 5.0

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "alpha0": self.alpha0,
            "mid_ratio": self.mid_ratio,
            "decay": self.decay,
            "center": self.center,
            "sharpness": self.sharpness,
            "warmup": self.warmup,
            "hold": self.hold,
            "gamma": self.gamma,
        }

    def validate(self):
        if self.schedule not in GATING_SCHEDULES:
            raise ValueError(
                f"Unknown gating schedule '{self.schedule}'. "
                f"Available: {list(GATING_SCHEDULES.keys())}"
            )
        if not 0 <= self.alpha0 <= 2:
            raise ValueError(f"alpha0 should be in [0, 2], got {self.alpha0}")
        if not 0 <= self.mid_ratio <= 1:
            raise ValueError(f"mid_ratio should be in [0, 1], got {self.mid_ratio}")


@dataclass
class FlowBlendConfig:
    """Configuration for flow blending function parameters."""
    
    blend_fn: str = "linear"
    # softmax_moe blend
    temperature: float = 1.0
    # norm_weighted / softmax_moe blend
    norm_eps: float = 1e-4
    # cosine_guided blend
    sim_eps: float = 1e-4
    # tanh_clipped blend
    clip_value: float = 2.0

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "norm_eps": self.norm_eps,
            "sim_eps": self.sim_eps,
            "clip_value": self.clip_value,
        }

    def validate(self):
        if self.blend_fn not in FLOW_BLEND_FNS:
            raise ValueError(
                f"Unknown flow blend '{self.blend_fn}'. "
                f"Available: {list(FLOW_BLEND_FNS.keys())}"
            )


@dataclass
class AttentionBlurConfig:
    """Configuration for attention blur parameters in prior branch."""
    
    blur_sigma: float = 1.5
    blur_attn_type: str = "self"  # "cross", "self", or "both"
    # V-feature perturbation (ablation)
    feat_blur_type: str = "none"  # "none", "blur", or "noise"
    feat_blur_sigma: float = 1.0
    feat_noise_type: str = "gaussian"  # "gaussian" or "uniform"
    feat_noise_strength: float = 0.0

    def validate(self):
        """Validate configuration values."""
        if self.blur_attn_type not in ("cross", "self", "both"):
            raise ValueError(
                f"blur_attn_type must be one of 'cross', 'self', 'both', got {self.blur_attn_type}"
            )
        if self.feat_blur_type not in ("none", "blur", "noise"):
            raise ValueError(
                f"feat_blur_type must be one of 'none', 'blur', 'noise', got {self.feat_blur_type}"
            )
        if self.feat_noise_type not in ("gaussian", "uniform"):
            raise ValueError(
                f"feat_noise_type must be 'gaussian' or 'uniform', got {self.feat_noise_type}"
            )


@dataclass
class PriorPoolingConfig:
    """Configuration for combining multiple prior images."""

    mode: str = "concat"  # "mean", "consensus", or "concat"
    temperature: float = 0.1
    agreement_boost: float = 0.0

    def validate(self):
        """Validate configuration values."""
        if self.mode not in ("mean", "consensus", "concat"):
            raise ValueError(
                f"prior_pooling mode must be 'mean', 'consensus', or 'concat', got {self.mode}"
            )
        if self.mode == "consensus" and self.temperature <= 0:
            raise ValueError(
                f"prior_pooling temperature must be > 0 for consensus, got {self.temperature}"
            )
        if self.agreement_boost < 0:
            raise ValueError(
                f"prior_pooling agreement_boost must be >= 0, got {self.agreement_boost}"
            )


@dataclass
class StageConfig:
    """Configuration for a single stage (Shape or Texture)."""
    
    prior_weight: float = 1.0
    inference_steps: Optional[int] = None
    gating: GatingConfig = field(default_factory=GatingConfig)
    flow_blend: FlowBlendConfig = field(default_factory=FlowBlendConfig)
    attention_blur: AttentionBlurConfig = field(default_factory=AttentionBlurConfig)
    prior_pooling: PriorPoolingConfig = field(default_factory=PriorPoolingConfig)

    def __post_init__(self):
        # Allow nested dicts to be passed in and converted to dataclasses
        if isinstance(self.gating, dict):
            self.gating = GatingConfig(**self.gating)
        if isinstance(self.flow_blend, dict):
            self.flow_blend = FlowBlendConfig(**self.flow_blend)
        if isinstance(self.attention_blur, dict):
            self.attention_blur = AttentionBlurConfig(**self.attention_blur)
        if isinstance(self.prior_pooling, dict):
            self.prior_pooling = PriorPoolingConfig(**self.prior_pooling)

    def get_gating_args(self) -> Dict[str, Any]:
        """Get gating schedule kwargs."""
        return self.gating.to_kwargs()

    def get_flow_blend_args(self) -> Dict[str, Any]:
        """Get flow blend kwargs."""
        return self.flow_blend.to_kwargs()

    def validate(self):
        """Validate all nested configurations."""
        if self.prior_weight < 0:
            raise ValueError(f"prior_weight must be >= 0, got {self.prior_weight}")
        self.gating.validate()
        self.flow_blend.validate()
        self.attention_blur.validate()
        self.prior_pooling.validate()

@dataclass
class GeometryMaskConfig:
    """Configuration for geometry mask parameters."""
    disable: bool = False
    use_condition_mask: bool = False
    soft_falloff: float = 3.0
    param_tolerance_scale: float = 1.5
    param_dilate_scale: float = 1.5

    def validate(self):
        """Validate configuration values."""
        if self.soft_falloff < 0:
            raise ValueError(f"soft_falloff must be >= 0, got {self.soft_falloff}")
        if self.param_tolerance_scale < 0:
            raise ValueError(f"param_tolerance_scale must be >= 0, got {self.param_tolerance_scale}")
        if self.param_dilate_scale < 0:
            raise ValueError(f"param_dilate_scale must be >= 0, got {self.param_dilate_scale}")

@dataclass
class RELAXFLOWConfig:
    """Complete RelaxFlow configuration."""
    
    stage1: StageConfig = field(default_factory=StageConfig)
    stage2: StageConfig = field(
        default_factory=lambda: StageConfig(
            prior_weight=0.75,
            prior_pooling=PriorPoolingConfig(
                mode="concat", temperature=0.1, agreement_boost=0.0
            ),
        )
    )
    geometry_mask: GeometryMaskConfig = field(default_factory=GeometryMaskConfig)
    seed: int = 42
    prior_mode: str = "cropped_and_mask"
    stage1_feat_blur: bool = False  # Apply feat blur/noise to stage1?
    prior_pooling_debug: bool = False
    prior_pooling_debug_stride: int = 8
    prior_pooling_debug_max_tokens: int = 256
    stage2_prior_pooling_debug: Optional[bool] = None
    stage2_prior_pooling_debug_stride: Optional[int] = None
    stage2_prior_pooling_debug_max_tokens: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.stage1, dict):
            self.stage1 = StageConfig(**self.stage1)
        if isinstance(self.stage2, dict):
            self.stage2 = StageConfig(**self.stage2)
        if isinstance(self.geometry_mask, dict):
            self.geometry_mask = GeometryMaskConfig(**self.geometry_mask)

    def validate(self):
        """Validate all configurations."""
        self.stage1.validate()
        self.stage2.validate()
        self.geometry_mask.validate()
        if self.prior_mode not in ("cropped", "full", "cropped_and_mask"):
            raise ValueError(
                f"prior_mode must be 'cropped', 'full', or 'cropped_and_mask', got {self.prior_mode}"
            )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RELAXFLOWConfig":
        """Create RELAXFLOWConfig from parsed argparse namespace."""
        def _get(attr: str, default):
            val = getattr(args, attr, None)
            return default if val is None else val

        stage1_gating = GatingConfig(
            schedule=_get("gating_schedule", "linear_cutoff"),
            alpha0=_get("gating_alpha0", 1.0),
            mid_ratio=_get("gating_mid_ratio", 0.4),
            decay=_get("gating_exp_k", 4.0),
            center=_get("gating_sigmoid_center", 0.5),
            sharpness=_get("gating_sigmoid_k", 10.0),
            warmup=_get("gating_warmup_ratio", 0.1),
            hold=_get("gating_hold_ratio", 0.2),
            gamma=_get("gating_inverse_gamma", 5.0),
        )

        stage1_flow_blend = FlowBlendConfig(
            blend_fn=_get("flow_blend", "linear"),
            temperature=_get("flow_blend_temperature", 1.0),
            norm_eps=_get("flow_blend_norm_eps", 1e-4),
            sim_eps=_get("flow_blend_sim_eps", 1e-4),
            clip_value=_get("flow_blend_clip", 2.0),
        )

        stage1_attn_blur = AttentionBlurConfig(
            blur_sigma=_get("prior_blur_sigma", 1.5),
            blur_attn_type=_get("blur_attn_type", "self"),
            feat_blur_type=_get("feat_blur_type", "none"),
            feat_blur_sigma=_get("feat_blur_sigma", 1.0),
            feat_noise_type=_get("feat_noise_type", "gaussian"),
            feat_noise_strength=_get("feat_noise_strength", 0.0),
        )
        stage1_prior_pooling = PriorPoolingConfig(
            mode=_get("prior_pooling", "concat"),
            temperature=_get("prior_pooling_temperature", 0.1),
            agreement_boost=_get("prior_pooling_agreement_boost", 0.0),
        )

        # Stage 2 gating config - use stage2 specific if provided, else inherit from stage1
        s2_gating_schedule = _get("stage2_gating_schedule", None) or _get("texture_gating_schedule", None)
        stage2_gating = GatingConfig(
            schedule=s2_gating_schedule if s2_gating_schedule else stage1_gating.schedule,
            alpha0=_get("stage2_gating_alpha0", stage1_gating.alpha0),
            mid_ratio=_get("stage2_gating_mid_ratio", stage1_gating.mid_ratio),
            decay=_get("stage2_gating_exp_k", stage1_gating.decay),
            center=_get("stage2_gating_sigmoid_center", stage1_gating.center),
            sharpness=_get("stage2_gating_sigmoid_k", stage1_gating.sharpness),
            warmup=_get("stage2_gating_warmup_ratio", stage1_gating.warmup),
            hold=_get("stage2_gating_hold_ratio", stage1_gating.hold),
            gamma=_get("stage2_gating_inverse_gamma", stage1_gating.gamma),
        )

        s2_flow_blend = _get("stage2_flow_blend", None) or _get("texture_flow_blend", None)
        stage2_flow_blend = FlowBlendConfig(
            blend_fn=s2_flow_blend if s2_flow_blend else stage1_flow_blend.blend_fn,
            temperature=_get("stage2_flow_blend_temperature", stage1_flow_blend.temperature),
            norm_eps=_get("stage2_flow_blend_norm_eps", stage1_flow_blend.norm_eps),
            sim_eps=_get("stage2_flow_blend_sim_eps", stage1_flow_blend.sim_eps),
            clip_value=_get("stage2_flow_blend_clip", stage1_flow_blend.clip_value),
        )

        # Stage 2 attention blur config
        stage2_attn_blur = AttentionBlurConfig(
            blur_sigma=_get("stage2_prior_blur_sigma", stage1_attn_blur.blur_sigma),
            blur_attn_type=_get("stage2_blur_attn_type", stage1_attn_blur.blur_attn_type),
            feat_blur_type=_get("feat_blur_type", "none"),
            feat_blur_sigma=_get("feat_blur_sigma", 1.0),
            feat_noise_type=_get("feat_noise_type", "gaussian"),
            feat_noise_strength=_get("feat_noise_strength", 0.0),
        )
        stage2_prior_pooling = PriorPoolingConfig(
            mode=_get("stage2_prior_pooling", "concat"),
            temperature=_get("stage2_prior_pooling_temperature", stage1_prior_pooling.temperature),
            agreement_boost=_get("stage2_prior_pooling_agreement_boost", 0.0),
        )

        stage1 = StageConfig(
            prior_weight=_get("prior_weight", 1.0),
            inference_steps=_get("stage1_inference_steps", None),
            gating=stage1_gating,
            flow_blend=stage1_flow_blend,
            attention_blur=stage1_attn_blur,
            prior_pooling=stage1_prior_pooling,
        )

        s2_prior_weight = _get("stage2_prior_weight", None)
        stage2 = StageConfig(
            prior_weight=s2_prior_weight if s2_prior_weight is not None else stage1.prior_weight,
            inference_steps=_get("stage2_inference_steps", None),
            gating=stage2_gating,
            flow_blend=stage2_flow_blend,
            attention_blur=stage2_attn_blur,
            prior_pooling=stage2_prior_pooling,
        )

        geometry_mask = GeometryMaskConfig(
            disable=_get("disable_geometry_mask", False),
            use_condition_mask=_get("use_geometry_mask_condition_mask", False),
            soft_falloff=_get("geometry_mask_soft_falloff", 3.0),
            param_tolerance_scale=_get("geometry_mask_param_tolerance_scale", 1.5),
            param_dilate_scale=_get("geometry_mask_param_dilate_scale", 1.5),
        )

        return cls(
            stage1=stage1,
            stage2=stage2,
            geometry_mask=geometry_mask,
            seed=_get("seed", 42),
            prior_mode=_get("prior_mode", "cropped_and_mask"),
            stage1_feat_blur=_get("feat_blur_stage1", False),
            prior_pooling_debug=_get("prior_pooling_debug", False),
            prior_pooling_debug_stride=_get("prior_pooling_debug_stride", 8),
            prior_pooling_debug_max_tokens=_get("prior_pooling_debug_max_tokens", 256),
            stage2_prior_pooling_debug=_get("stage2_prior_pooling_debug", None),
            stage2_prior_pooling_debug_stride=_get("stage2_prior_pooling_debug_stride", None),
            stage2_prior_pooling_debug_max_tokens=_get("stage2_prior_pooling_debug_max_tokens", None),
        )

    def to_run_kwargs(self) -> Dict[str, Any]:
        """
        Convert config to kwargs dict for InferencePipelineRELAXFLOW.run().
        This bridges the gap between the new structured config and the existing API.
        """
        return {
            # Stage 1 parameters
            "prior_weight": self.stage1.prior_weight,
            "prior_blur_sigma": self.stage1.attention_blur.blur_sigma,
            "blur_attn_type": self.stage1.attention_blur.blur_attn_type,
            "gating_schedule_name": self.stage1.gating.schedule,
            "gating_args": self.stage1.get_gating_args(),
            "flow_blend_name": self.stage1.flow_blend.blend_fn,
            "flow_blend_args": self.stage1.get_flow_blend_args(),
            "stage1_inference_steps": self.stage1.inference_steps,
            "prior_pooling": self.stage1.prior_pooling.mode,
            "prior_pooling_temperature": self.stage1.prior_pooling.temperature,
            "prior_pooling_agreement_boost": self.stage1.prior_pooling.agreement_boost,
            "prior_pooling_debug": self.prior_pooling_debug,
            "prior_pooling_debug_stride": self.prior_pooling_debug_stride,
            "prior_pooling_debug_max_tokens": self.prior_pooling_debug_max_tokens,
            # !deprecated: V-feature perturbation for stage1 (controlled by stage1_feat_blur flag)
            "feat_blur_stage1": self.stage1_feat_blur,
            "feat_blur_type": self.stage1.attention_blur.feat_blur_type if self.stage1_feat_blur 
                             else self.stage2.attention_blur.feat_blur_type,
            "feat_blur_sigma": self.stage1.attention_blur.feat_blur_sigma if self.stage1_feat_blur 
                              else self.stage2.attention_blur.feat_blur_sigma,
            "feat_noise_type": self.stage1.attention_blur.feat_noise_type if self.stage1_feat_blur 
                              else self.stage2.attention_blur.feat_noise_type,
            "feat_noise_strength": self.stage1.attention_blur.feat_noise_strength if self.stage1_feat_blur 
                                  else self.stage2.attention_blur.feat_noise_strength,
            # Geometry mask parameters
            "use_geometry_mask_condition_mask": self.geometry_mask.use_condition_mask,
            "geometry_mask_soft_falloff": self.geometry_mask.soft_falloff,
            "geometry_mask_param_tolerance_scale": self.geometry_mask.param_tolerance_scale,
            "geometry_mask_param_dilate_scale": self.geometry_mask.param_dilate_scale,
            "disable_geometry_mask": self.geometry_mask.disable,
            # Stage 2 parameters
            "stage2_prior_weight": self.stage2.prior_weight,
            "stage2_prior_blur_sigma": self.stage2.attention_blur.blur_sigma,
            "stage2_blur_attn_type": self.stage2.attention_blur.blur_attn_type,
            "stage2_gating_schedule_name": self.stage2.gating.schedule,
            "stage2_gating_args": self.stage2.get_gating_args(),
            "stage2_flow_blend_name": self.stage2.flow_blend.blend_fn,
            "stage2_flow_blend_args": self.stage2.get_flow_blend_args(),
            "stage2_inference_steps": self.stage2.inference_steps,
            "stage2_prior_pooling": self.stage2.prior_pooling.mode,
            "stage2_prior_pooling_temperature": self.stage2.prior_pooling.temperature,
            "stage2_prior_pooling_agreement_boost": self.stage2.prior_pooling.agreement_boost,
            "stage2_prior_pooling_debug": self.stage2_prior_pooling_debug,
            "stage2_prior_pooling_debug_stride": self.stage2_prior_pooling_debug_stride,
            "stage2_prior_pooling_debug_max_tokens": self.stage2_prior_pooling_debug_max_tokens,
            # Seed
            "seed": None if self.seed < 0 else self.seed,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dict for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RELAXFLOWConfig":
        """Create from nested dict."""
        return cls(**d)


def add_relaxflow_arguments(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    """Add RelaxFlow related arguments to an argparse parser."""
    p = prefix
    
    stage1_group = parser.add_argument_group("Stage 1 (Shape) RelaxFlow Configuration")
    
    stage1_group.add_argument(
        f"--{p}prior-weight",
        type=float,
        default=1.0,
        help="Prior weight for Stage 1 flow blending.",
    )
    stage1_group.add_argument(
        f"--{p}prior-blur-sigma",
        type=float,
        default=2.5,
        help="Gaussian blur sigma for prior-branch attention in Stage 1.",
    )
    stage1_group.add_argument(
        f"--{p}blur-attn-type",
        choices=["cross", "self", "both"],
        default="self",
        help="Which attention types to blur inside the prior branch (Stage 1).",
    )
    stage1_group.add_argument(
        f"--{p}gating-schedule",
        type=str,
        default="linear_cutoff",
        choices=list(GATING_SCHEDULES.keys()),
        help="Gating schedule for Stage 1 (shape) flow.",
    )
    stage1_group.add_argument(
        f"--{p}flow-blend",
        type=str,
        default="linear",
        choices=list(FLOW_BLEND_FNS.keys()),
        help="Velocity blending strategy for Stage 1.",
    )
    stage1_group.add_argument(
        f"--{p}stage1-inference-steps",
        type=int,
        default=None,
        help="Override flow steps for Stage 1 sparse structure sampling.",
    )
    
    # Stage 1 gating args
    gating1_group = parser.add_argument_group("Stage 1 Gating Parameters")
    gating1_group.add_argument(f"--{p}gating-alpha0", type=float, default=1.0,
                               help="Initial gating strength alpha_0 at t=0.")
    gating1_group.add_argument(f"--{p}gating-mid-ratio", type=float, default=0.4,
                               help="Fraction of steps before prior guidance shuts off.")
    gating1_group.add_argument(f"--{p}gating-exp-k", type=float, default=4.0,
                               help="Exponential decay rate for 'exponential' schedule.")
    gating1_group.add_argument(f"--{p}gating-sigmoid-center", type=float, default=0.5,
                               help="Center position for 'sigmoid' schedule.")
    gating1_group.add_argument(f"--{p}gating-sigmoid-k", type=float, default=10.0,
                               help="Sharpness for 'sigmoid' schedule.")
    gating1_group.add_argument(f"--{p}gating-warmup-ratio", type=float, default=0.1,
                               help="Warmup ratio for 'two_stage' schedule.")
    gating1_group.add_argument(f"--{p}gating-hold-ratio", type=float, default=0.2,
                               help="Hold ratio for 'two_stage' schedule.")
    gating1_group.add_argument(f"--{p}gating-inverse-gamma", type=float, default=5.0,
                               help="Gamma for 'inverse_sqrt' schedule.")
    
    # Stage 1 flow blend args
    blend1_group = parser.add_argument_group("Stage 1 Flow Blend Parameters")
    blend1_group.add_argument(f"--{p}flow-blend-temperature", type=float, default=1.0,
                              help="Temperature for softmax_moe blend.")
    blend1_group.add_argument(f"--{p}flow-blend-norm-eps", type=float, default=1e-4,
                              help="Epsilon for norm_weighted/softmax_moe blends.")
    blend1_group.add_argument(f"--{p}flow-blend-sim-eps", type=float, default=1e-4,
                              help="Epsilon for cosine_guided blend.")
    blend1_group.add_argument(f"--{p}flow-blend-clip", type=float, default=2.0,
                              help="Clipping value for tanh_clipped blend.")

    # Stage 1 prior pooling args
    pool1_group = parser.add_argument_group("Stage 1 Prior Pooling")
    pool1_group.add_argument(
        f"--{p}prior-pooling",
        type=str,
        choices=["mean", "consensus", "concat"],
        default="concat",
        help="How to pool multiple priors for Stage 1 (shape).",
    )
    pool1_group.add_argument(
        f"--{p}prior-pooling-temperature",
        type=float,
        default=0.1,
        help="Softmax temperature for consensus pooling (Stage 1).",
    )
    pool1_group.add_argument(
        f"--{p}prior-pooling-agreement-boost",
        type=float,
        default=0.0,
        help="Agreement boost for consistent tokens across priors (Stage 1).",
    )
    
    # ===================== Stage 2 =====================
    stage2_group = parser.add_argument_group("Stage 2 (Texture) RelaxFlow Configuration")
    
    stage2_group.add_argument(
        f"--{p}stage2-prior-weight",
        type=float,
        default=None,
        help="Prior weight for Stage 2 (defaults to Stage 1 value).",
    )
    stage2_group.add_argument(
        f"--{p}stage2-prior-blur-sigma",
        type=float,
        default=1.5,
        help="Blur sigma for Stage 2 (defaults to Stage 1 value).",
    )
    stage2_group.add_argument(
        f"--{p}stage2-blur-attn-type",
        choices=["cross", "self", "both"],
        default=None,
        help="Attention type for Stage 2 blur (defaults to Stage 1 value).",
    )
    stage2_group.add_argument(
        f"--{p}stage2-gating-schedule",
        "--texture-gating-schedule",
        type=str,
        default=None,
        choices=list(GATING_SCHEDULES.keys()),
        help="Gating schedule for Stage 2 (defaults to Stage 1).",
    )
    stage2_group.add_argument(
        f"--{p}stage2-flow-blend",
        "--texture-flow-blend",
        type=str,
        default=None,
        choices=list(FLOW_BLEND_FNS.keys()),
        help="Flow blend for Stage 2 (defaults to Stage 1).",
    )
    stage2_group.add_argument(
        f"--{p}stage2-inference-steps",
        type=int,
        default=None,
        help="Override diffusion steps for Stage 2 SLAT refinement.",
    )
    
    # Stage 2 gating args (separate from Stage 1)
    gating2_group = parser.add_argument_group("Stage 2 Gating Parameters (defaults to Stage 1 values)")
    gating2_group.add_argument(f"--{p}stage2-gating-alpha0", type=float, default=None,
                               help="Stage 2 gating alpha0.")
    gating2_group.add_argument(f"--{p}stage2-gating-mid-ratio", type=float, default=0.4,
                               help="Stage 2 gating mid_ratio.")
    gating2_group.add_argument(f"--{p}stage2-gating-exp-k", type=float, default=None,
                               help="Stage 2 exponential decay rate.")
    gating2_group.add_argument(f"--{p}stage2-gating-sigmoid-center", type=float, default=None,
                               help="Stage 2 sigmoid center.")
    gating2_group.add_argument(f"--{p}stage2-gating-sigmoid-k", type=float, default=None,
                               help="Stage 2 sigmoid sharpness.")
    gating2_group.add_argument(f"--{p}stage2-gating-warmup-ratio", type=float, default=None,
                               help="Stage 2 warmup ratio.")
    gating2_group.add_argument(f"--{p}stage2-gating-hold-ratio", type=float, default=None,
                               help="Stage 2 hold ratio.")
    gating2_group.add_argument(f"--{p}stage2-gating-inverse-gamma", type=float, default=None,
                               help="Stage 2 inverse_sqrt gamma.")
    
    # Stage 2 flow blend args
    blend2_group = parser.add_argument_group("Stage 2 Flow Blend Parameters (defaults to Stage 1 values)")
    blend2_group.add_argument(f"--{p}stage2-flow-blend-temperature", type=float, default=None,
                              help="Stage 2 softmax_moe temperature.")
    blend2_group.add_argument(f"--{p}stage2-flow-blend-norm-eps", type=float, default=None,
                              help="Stage 2 norm epsilon.")
    blend2_group.add_argument(f"--{p}stage2-flow-blend-sim-eps", type=float, default=None,
                              help="Stage 2 similarity epsilon.")
    blend2_group.add_argument(f"--{p}stage2-flow-blend-clip", type=float, default=None,
                              help="Stage 2 tanh clip value.")

    # Stage 2 prior pooling args
    pool2_group = parser.add_argument_group("Stage 2 Prior Pooling")
    pool2_group.add_argument(
        f"--{p}stage2-prior-pooling",
        type=str,
        choices=["mean", "consensus", "concat"],
        default="concat",
        help="How to pool multiple priors for Stage 2 (texture).",
    )
    pool2_group.add_argument(
        f"--{p}stage2-prior-pooling-temperature",
        type=float,
        default=None,
        help="Softmax temperature for consensus pooling (Stage 2). Defaults to Stage 1 value.",
    )
    pool2_group.add_argument(
        f"--{p}stage2-prior-pooling-agreement-boost",
        type=float,
        default=0.0,
        help="Agreement boost for consistent tokens across priors (Stage 2).",
    )

    # Prior pooling debug
    debug_group = parser.add_argument_group("Prior Pooling Debug")
    debug_group.add_argument(
        f"--{p}prior-pooling-debug",
        action="store_true",
        help="Enable debug logs and weight heatmaps for prior pooling (Stage 1).",
    )
    debug_group.add_argument(
        f"--{p}prior-pooling-debug-stride",
        type=int,
        default=8,
        help="Token stride when saving prior pooling heatmaps.",
    )
    debug_group.add_argument(
        f"--{p}prior-pooling-debug-max-tokens",
        type=int,
        default=256,
        help="Max tokens to include in prior pooling heatmaps.",
    )
    debug_group.add_argument(
        f"--{p}stage2-prior-pooling-debug",
        action="store_true",
        default=None,
        help="Enable debug logs and weight heatmaps for prior pooling (Stage 2).",
    )
    debug_group.add_argument(
        f"--{p}stage2-prior-pooling-debug-stride",
        type=int,
        default=None,
        help="Token stride for Stage 2 pooling heatmaps (defaults to Stage 1).",
    )
    debug_group.add_argument(
        f"--{p}stage2-prior-pooling-debug-max-tokens",
        type=int,
        default=None,
        help="Max tokens for Stage 2 pooling heatmaps (defaults to Stage 1).",
    )
    
    # ===================== V-Feature Perturbation  deprecated! =====================
    feat_group = parser.add_argument_group("V-Feature Perturbation (Ablation)")
    feat_group.add_argument(
        f"--{p}feat-blur-type",
        type=str,
        choices=["none", "blur", "noise"],
        default="none",
    )
    feat_group.add_argument(
        f"--{p}feat-blur-sigma",
        type=float,
        default=1.0,
    )
    feat_group.add_argument(
        f"--{p}feat-noise-type",
        type=str,
        choices=["gaussian", "uniform"],
        default="gaussian",
    )
    feat_group.add_argument(
        f"--{p}feat-noise-strength",
        type=float,
        default=0.0,
    )
    feat_group.add_argument(
        f"--{p}feat-blur-stage1",
        action="store_true",
    )
    
    # ===================== General =====================
    general_group = parser.add_argument_group("General RelaxFlow Settings")
    general_group.add_argument(
        f"--{p}prior-mode",
        type=str,
        default="cropped_and_mask",
        choices=["cropped", "full", "cropped_and_mask"],
        help="How to handle prior images.",
    )
    general_group.add_argument(
        f"--{p}seed",
        type=int,
        default=42,
        help="Random seed (-1 for stochastic).",
    )
    general_group.add_argument(
        f"--{p}disable-geometry-mask",
        action="store_true",
        help="Disable geometry visibility mask computation/blending in Stage 2.",
    )


def print_config_summary(config: RELAXFLOWConfig) -> str:
    """Generate a human-readable summary of the RelaxFlow configuration."""
    lines = [
        "=" * 60,
        "RelaxFlow Configuration Summary",
        "=" * 60,
        "",
        "Stage 1 (Shape):",
        f"  Prior Weight:      {config.stage1.prior_weight}",
        f"  Gating Schedule:   {config.stage1.gating.schedule}",
        f"    - alpha0:        {config.stage1.gating.alpha0}",
        f"    - mid_ratio:     {config.stage1.gating.mid_ratio}",
        f"  Flow Blend:        {config.stage1.flow_blend.blend_fn}",
        f"  Blur Sigma:        {config.stage1.attention_blur.blur_sigma}",
        f"  Blur Attn Type:    {config.stage1.attention_blur.blur_attn_type}",
        f"  Prior Pooling:     {config.stage1.prior_pooling.mode}",
        f"    - temperature:   {config.stage1.prior_pooling.temperature}",
        f"    - boost:         {config.stage1.prior_pooling.agreement_boost}",
        "",
        "Stage 2 (Texture):",
        f"  Prior Weight:      {config.stage2.prior_weight}",
        f"  Gating Schedule:   {config.stage2.gating.schedule}",
        f"    - alpha0:        {config.stage2.gating.alpha0}",
        f"    - mid_ratio:     {config.stage2.gating.mid_ratio}",
        f"  Flow Blend:        {config.stage2.flow_blend.blend_fn}",
        f"  Blur Sigma:        {config.stage2.attention_blur.blur_sigma}",
        f"  Blur Attn Type:    {config.stage2.attention_blur.blur_attn_type}",
        f"  Prior Pooling:     {config.stage2.prior_pooling.mode}",
        f"    - temperature:   {config.stage2.prior_pooling.temperature}",
        f"    - boost:         {config.stage2.prior_pooling.agreement_boost}",
        "",
        "Geometry Mask:",
        f"  Disabled:         {config.geometry_mask.disable}",
        f"  Use Condition Mask: {config.geometry_mask.use_condition_mask}",
        f"  Soft Falloff:      {config.geometry_mask.soft_falloff}",
        f"  Param Tolerance Scale: {config.geometry_mask.param_tolerance_scale}",
        f"  Param Dilate Scale: {config.geometry_mask.param_dilate_scale}",
        "",
        f"Seed:              {config.seed}",
        f"Prior Mode:        {config.prior_mode}",
        f"Stage1 Feat Blur:  {config.stage1_feat_blur}",
        "=" * 60,
    ]
    return "\n".join(lines)
