#!/usr/bin/env python3
"""
RelaxFlow Demo

Usage Examples:
    # Basic usage with defaults
    python demo_relaxflow.py --image input.png --prior-images prior.png

    # Different configurations for Stage 1 and Stage 2
    python demo_relaxflow.py --image input.png --prior-images prior.png \
        --gating-schedule linear_cutoff --gating-mid-ratio 0.5 \
        --stage2-gating-schedule two_stage --stage2-gating-mid-ratio 0.3

    # Show configuration summary without running
    python demo_relaxflow.py --show-config
"""

import argparse
import os
import sys

import imageio
from loguru import logger
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

# Import utilities
sys.path.append("notebook")
from inference import (  # noqa: E402
    load_image,
    load_mask,
    infer_mask_from_image,
    check_hydra_safety,
    WHITELIST_FILTERS,
    BLACKLIST_FILTERS,
)
from sam3d_objects.pipeline.relaxflow_config import (
    RELAXFLOWConfig,
    add_relaxflow_arguments,
    print_config_summary,
)

PIPELINE_TARGET = (
    "sam3d_objects.pipeline.inference_pipeline_relaxflow.InferencePipelineRELAXFLOW"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="RelaxFlow Demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ==================== Input/Output ====================
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--config",
        default="checkpoints/hf/checkpoints/pipeline.yaml",
        help="Path to the pipeline config file.",
    )
    io_group.add_argument(
        "--image",
        help="Path to the input image (RGBA/RGB).",
    )
    io_group.add_argument(
        "--mask",
        default=None,
        help="Optional binary mask. If omitted, uses alpha channel or rembg.",
    )
    io_group.add_argument(
        "--prior-images",
        nargs="+",
        help="One or more semantic prior images for RelaxFlow guidance.",
    )
    io_group.add_argument(
        "--prior-masks",
        nargs="+",
        default=None,
        help="Optional masks for prior images.",
    )
    io_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs/relaxflow",
        help="Output directory.",
    )
    io_group.add_argument(
        "--output-name",
        type=str,
        default="result",
        help="Output name prefix.",
    )

    # ==================== Pipeline Options ====================
    pipeline_group = parser.add_argument_group("Pipeline Options")
    pipeline_group.add_argument(
        "--image-as",
        type=str,
        default="part",
        choices=["scene", "part"],
        help="Route observed image into 'part' (cropped) or 'scene' condition slot.",
    )
    pipeline_group.add_argument(
        "--only-cropped-img",
        action="store_true",
        help="Use only cropped image token as condition.",
    )
    pipeline_group.add_argument(
        "--only-cropped-img-and-mask",
        action="store_true",
        help="Use cropped image and mask tokens as condition.",
    )
    pipeline_group.add_argument(
        "--stage1-only",
        action="store_true",
        help="Skip Stage 2 (SLAT) and output sparse structure only.",
    )
    pipeline_group.add_argument(
        "--compile",
        action="store_true",
        help="Compile the pipeline for faster inference.",
    )

    # ==================== Geometry Mask Options ====================
    geometry_mask_group = parser.add_argument_group("Geometry Mask Options")
    geometry_mask_group.add_argument(
        "--use-geometry-mask-condition-mask",
        action="store_true",
        help="Use geometry mask as condition mask.",
        default=False,
    )
    geometry_mask_group.add_argument(
        "--geometry-mask-soft-falloff",
        type=float,
        default=3.0,
        help="Soft falloff for geometry mask.",
    )
    geometry_mask_group.add_argument(
        "--geometry-mask-param-tolerance-scale",
        type=float,
        default=1.5,
        help="Parameter tolerance scale for geometry mask.",
    )
    geometry_mask_group.add_argument(
        "--geometry-mask-param-dilate-scale",
        type=float,
        default=1.5,
        help="Parameter dilate scale for geometry mask.",
    )

    # ==================== Rendering Options ====================
    render_group = parser.add_argument_group("Rendering Options")
    render_group.add_argument(
        "--render-backend",
        default="inria",
        choices=["inria", "gsplat"],
        help="Rendering backend for turntable video.",
    )
    render_group.add_argument(
        "--use-vertex-color",
        dest="use_vertex_color",
        action="store_true",
        default=True,
        help="Use vertex color baking for mesh.",
    )
    render_group.add_argument(
        "--no-vertex-color",
        dest="use_vertex_color",
        action="store_false",
        help="Skip vertex color baking.",
    )

    # ==================== Add RelaxFlow Configuration Arguments ====================
    add_relaxflow_arguments(parser)

    # ==================== Utility Options ====================
    util_group = parser.add_argument_group("Utility Options")
    util_group.add_argument(
        "--show-config",
        action="store_true",
        help="Print configuration summary and exit without running.",
    )

    return parser.parse_args()


def build_pipeline(config_path: str, compile_model: bool, prior_mode: str):
    """Load config and instantiate the RelaxFlow pipeline."""
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = compile_model
    config.workspace_dir = os.path.dirname(config_path)
    config["_target_"] = PIPELINE_TARGET

    # Set default blur params if not present
    if "prior_blur_sigma" not in config:
        config.prior_blur_sigma = 2.5
    if "blur_attn_type" not in config:
        config.blur_attn_type = "self"

    # Handle prior_mode
    config.pop("prior_mode", None)
    if prior_mode == "cropped":
        config.prior_use_cropped_only = True
        config.prior_only_cropped_img_and_mask = False
    elif prior_mode == "full":
        config.prior_use_cropped_only = False
        config.prior_only_cropped_img_and_mask = False
    elif prior_mode == "cropped_and_mask":
        config.prior_use_cropped_only = False
        config.prior_only_cropped_img_and_mask = True

    check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
    return instantiate(config)


def load_inputs(args):
    """Load image, mask, and prior images."""
    def _resize_mask(mask_arr, target_image):
        h, w = target_image.shape[:2]
        if mask_arr.shape[0] == h and mask_arr.shape[1] == w:
            return mask_arr.astype(bool)
        print(f"Resizing mask from {mask_arr.shape[:2]} to {(h, w)}")
        mask_img = Image.fromarray(mask_arr.astype(np.uint8) * 255)
        mask_img = mask_img.resize((w, h), resample=Image.Resampling.NEAREST)
        return (np.array(mask_img) > 0).astype(bool)

    def _expand_arg_list(values):
        if not values:
            return []
        expanded = []
        for value in values:
            if isinstance(value, str) and "," in value:
                expanded.extend([v for v in value.split(",") if v])
            elif isinstance(value, str) and ' ' in value.strip():
                expanded.extend([v for v in value.strip().split(" ") if v])
            else:
                expanded.append(value)
        return expanded

    logger.info("Loading inputs...")
    logger.info(f"Loading image from {args.image}")
    image = load_image(args.image)
    if args.mask:
        logger.info(f"Loading mask from {args.mask}")
        mask = load_mask(args.mask)
        mask = _resize_mask(mask, image)
    else:
        logger.info(f"Inferring mask from alpha/rembg")
        mask = infer_mask_from_image(image)

    prior_image_paths = _expand_arg_list(args.prior_images) or [args.image]
    logger.info(f"Loading prior images from {prior_image_paths}")
    prior_images = [load_image(p) for p in prior_image_paths]
    if args.prior_masks:
        prior_mask_paths = _expand_arg_list(args.prior_masks)
        logger.info(f"Loading prior masks from {prior_mask_paths}")
        if len(prior_mask_paths) == 1 and len(prior_images) > 1:
            logger.info("Broadcasting single prior mask across all prior images")
            base_mask = load_mask(prior_mask_paths[0])
            prior_masks = [_resize_mask(base_mask, pi) for pi in prior_images]
        else:
            if len(prior_mask_paths) != len(prior_images):
                raise ValueError(
                    "Number of prior masks must be 1 or match the number of prior images. "
                    f"Got {len(prior_mask_paths)} masks for {len(prior_images)} images."
                )
            prior_masks = [
                _resize_mask(load_mask(m), pi)
                for m, pi in zip(prior_mask_paths, prior_images)
            ]
    else:
        logger.info(f"Inferring prior masks from alpha/rembg")
        prior_masks = [infer_mask_from_image(pi) for pi in prior_images]

    return image, mask, prior_images, prior_masks


def main():
    args = parse_args()

    # Build structured config from args
    relaxflow_config = RELAXFLOWConfig.from_args(args)

    # Show config summary if requested
    if args.show_config:
        print(print_config_summary(relaxflow_config))
        return

    # Validate configuration
    try:
        relaxflow_config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Print & save configuration summary
    summary = print_config_summary(relaxflow_config)
    print(summary)
    os.makedirs(os.path.join(args.output_dir, args.output_name), exist_ok=True)
    with open(os.path.join(args.output_dir, args.output_name, "config_summary.txt"), "w") as f:
        f.write(summary)

    # Check for conflicting options
    if args.only_cropped_img and args.only_cropped_img_and_mask:
        raise ValueError(
            "--only-cropped-img and --only-cropped-img-and-mask are mutually exclusive"
        )

    # Derive prior toggles from mode
    if relaxflow_config.prior_mode == "cropped":
        prior_use_cropped_only = True
        prior_only_cropped_img_and_mask = False
    elif relaxflow_config.prior_mode == "full":
        prior_use_cropped_only = False
        prior_only_cropped_img_and_mask = False
    else:  # cropped_and_mask
        prior_use_cropped_only = False
        prior_only_cropped_img_and_mask = True

    # Build pipeline
    print("\nBuilding RelaxFlow pipeline...")
    pipeline = build_pipeline(args.config, args.compile, relaxflow_config.prior_mode)

    # Load inputs
    print("Loading inputs...")
    image, mask, prior_images, prior_masks = load_inputs(args)

    # Prepare output directory
    out_dir = os.path.join(args.output_dir, args.output_name)
    os.makedirs(out_dir, exist_ok=True)

    # Get run kwargs from config
    run_kwargs = relaxflow_config.to_run_kwargs()
    run_kwargs.pop("disable_geometry_mask", None)

    # Run RelaxFlow pipeline
    print("\nRunning RelaxFlow pipeline...")
    output = pipeline.run(
        image=image,
        mask=mask,
        prior_images=prior_images,
        prior_masks=prior_masks,
        stage1_only=args.stage1_only,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=True,
        use_vertex_color=args.use_vertex_color,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        estimate_plane=False,
        only_cropped_img=args.only_cropped_img,
        only_cropped_img_and_mask=args.only_cropped_img_and_mask,
        image_as=args.image_as,
        prior_use_cropped_only=prior_use_cropped_only,
        prior_only_cropped_img_and_mask=prior_only_cropped_img_and_mask,
        return_branch_outputs=True,
        save_render_dir=out_dir,
        render_backend=args.render_backend,
        **run_kwargs,
    )

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
