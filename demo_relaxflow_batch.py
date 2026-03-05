#!/usr/bin/env python3
"""
RelaxFlow Batch Runner.

Manifest format (JSON/JSONL):
[
  {
    "id": "sample_0001",
    "image": "/abs/path/to/image.png",
    "mask": "/abs/path/to/mask.png",
    "prior_images": ["/abs/path/to/prior.png"],
    "prior_masks": ["/abs/path/to/prior_mask.png"],
    "prior_text": "a wooden chair",
    "gt_mesh": "/abs/path/to/gt_mesh.obj",
    "gt_pointcloud": "/abs/path/to/gt_points.npy",
    "gt_images": ["/abs/path/to/gt_view_00.png", "..."],
    "gt_render_dir": "/abs/path/to/gt_views_dir"
  }
]
"""

import argparse
import math
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from hydra.utils import instantiate
from PIL import Image
import torch
import trimesh
from typing import Any

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
from sam3d_objects.utils.relaxflow_eval_utils import (
    CLIPImageSimilarity,
    CLIPScoreCalculator,
    FIDCalculator,
    KIDCalculator,
    compute_3d_metrics_from_pointclouds,
    cov_mmd_from_sets,
    fps_downsample,
    has_2d_metrics,
    save_batch_npz_files,
    compute_pfid_with_pointe,
)
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils
from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult

PIPELINE_TARGET_RELAXFLOW = (
    "sam3d_objects.pipeline.inference_pipeline_relaxflow.InferencePipelineRELAXFLOW"
)
PIPELINE_TARGET_BASE = (
    "sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap"
)

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


def _default_cache_root() -> str:
    env_cache = os.environ.get("RELAXFLOW_CACHE_ROOT")
    if env_cache:
        return env_cache
    return str(Path("outputs") / "cache")


def _path_is_under(path: str, root: Path) -> bool:
    try:
        return Path(path).resolve().as_posix().startswith(root.resolve().as_posix())
    except Exception:
        return False


def _set_cache_env(cache_root: Optional[str]) -> None:
    if not cache_root:
        return
    resolved_root = Path(cache_root).expanduser()
    if not resolved_root.is_absolute():
        resolved_root = (Path.cwd() / resolved_root).resolve()
    else:
        resolved_root = resolved_root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    hf_home = resolved_root / "hf_cache"
    hub_cache = hf_home / "hub"
    datasets_cache = hf_home / "datasets"
    torch_home = resolved_root / "torch_cache"
    point_e_cache = resolved_root / "point_e_cache"
    tmp_dir = resolved_root / "tmp"
    for path in (hf_home, hub_cache, datasets_cache, torch_home, point_e_cache, tmp_dir):
        path.mkdir(parents=True, exist_ok=True)

    home_root = Path.home()

    def _set_env(name: str, value: Path) -> None:
        current = os.environ.get(name)
        if not current or _path_is_under(current, home_root):
            os.environ[name] = str(value)

    _set_env("HF_HOME", hf_home)
    _set_env("HUGGINGFACE_HUB_CACHE", hub_cache)
    _set_env("TRANSFORMERS_CACHE", hub_cache)
    _set_env("HF_DATASETS_CACHE", datasets_cache)
    _set_env("TORCH_HOME", torch_home)
    _set_env("POINT_E_CACHE_DIR", point_e_cache)
    _set_env("TMPDIR", tmp_dir)
    _set_env("TEMP", tmp_dir)
    _set_env("TMP", tmp_dir)


def _expand_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        if "," in value:
            return [v for v in value.split(",") if v]
        if " " in value.strip():
            return [v for v in value.strip().split(" ") if v]
        return [value]
    return [value]


def _resolve_path(path: Optional[str], base_dir: Path, data_root: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    if data_root is not None:
        return str(data_root / path_obj)
    return str(base_dir / path_obj)


def _load_manifest(manifest_path: Path) -> List[Dict]:
    if manifest_path.suffix.lower() == ".jsonl":
        samples = []
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples
    with open(manifest_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "samples" in data:
        return list(data["samples"])
    if isinstance(data, list):
        return data
    raise ValueError("Manifest must be a JSON list or contain a 'samples' list.")


def _resize_mask(mask_arr: np.ndarray, target_image: np.ndarray) -> np.ndarray:
    h, w = target_image.shape[:2]
    if mask_arr.shape[0] == h and mask_arr.shape[1] == w:
        return mask_arr.astype(bool)
    mask_img = Image.fromarray(mask_arr.astype(np.uint8) * 255)
    mask_img = mask_img.resize((w, h), resample=Image.Resampling.NEAREST)
    return (np.array(mask_img) > 0).astype(bool)


def _mask_too_small(mask: np.ndarray, min_size: int = 2) -> bool:
    if mask is None:
        return True
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return True
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    return width < min_size or height < min_size


def _crop_and_center_object(image: np.ndarray, bg_threshold: int = 5) -> np.ndarray:
    if image is None:
        return image
    img = image
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        alpha = img[..., 3]
        mask = alpha > 0
        img = img[..., :3]
    else:
        mask = img.sum(axis=-1) > bg_threshold
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return img
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    size = max(h, w)
    out = np.zeros((size, size, 3), dtype=crop.dtype)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    out[y_off : y_off + h, x_off : x_off + w] = crop
    return out


def _resize_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    if image is None:
        return image
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((size, size), resample=Image.Resampling.BICUBIC)
    return np.array(pil_img)


def _prep_lpips_tensor(image: np.ndarray, size: int = 224) -> torch.Tensor:
    resized = _resize_image(image, size=size).astype(np.float32) / 255.0
    if resized.ndim == 2:
        resized = np.stack([resized, resized, resized], axis=-1)
    if resized.shape[-1] == 4:
        resized = resized[..., :3]
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _resolve_obs_object_image(
    entry: Dict,
    base_dir: Path,
    data_root: Optional[Path],
    image_path: Optional[str],
) -> Optional[str]:
    for key in ("obs_object_image", "rendered", "full_object_image"):
        if entry.get(key):
            return _resolve_path(entry.get(key), base_dir, data_root)
    if image_path:
        img_dir = Path(image_path).parent
        candidate = img_dir / "rendered.png"
        if candidate.exists():
            return str(candidate)
    return None


def _load_inputs(entry: Dict, data_root: Optional[Path], base_dir: Path):
    image_path = _resolve_path(entry.get("image"), base_dir, data_root)
    if image_path is None:
        raise ValueError("Each entry must define an 'image' path.")
    image = load_image(image_path)
    mask_path = _resolve_path(entry.get("mask"), base_dir, data_root)
    if mask_path:
        mask = load_mask(mask_path)
        mask = _resize_mask(mask, image)
    else:
        mask = infer_mask_from_image(image)

    prior_images = _expand_list(entry.get("prior_images") or entry.get("prior_image"))
    if not prior_images:
        logger.warning("No prior_images provided; defaulting to input image for {}", image_path)
        prior_images = [image_path]
    prior_image_paths = [_resolve_path(p, base_dir, data_root) for p in prior_images]
    prior_images = [load_image(p) for p in prior_image_paths]

    prior_masks = _expand_list(entry.get("prior_masks") or entry.get("prior_mask"))
    prior_mask_paths: List[str] = []
    if prior_masks:
        prior_mask_paths = [_resolve_path(p, base_dir, data_root) for p in prior_masks]
        if len(prior_mask_paths) == 1 and len(prior_images) > 1:
            base_mask = load_mask(prior_mask_paths[0])
            prior_masks = [_resize_mask(base_mask, pi) for pi in prior_images]
        elif len(prior_mask_paths) != len(prior_images):
            raise ValueError(
                f"prior_masks length must be 1 or match prior_images ({len(prior_images)})."
            )
        else:
            prior_masks = [
                _resize_mask(load_mask(p), pi) for p, pi in zip(prior_mask_paths, prior_images)
            ]
    else:
        prior_masks = [infer_mask_from_image(pi) for pi in prior_images]

    return (
        image_path,
        mask_path,
        image,
        mask,
        prior_images,
        prior_masks,
        prior_image_paths,
        prior_mask_paths,
    )


def _save_mask_array(mask: np.ndarray, path: Path) -> None:
    mask_u8 = (mask.astype(np.uint8) * 255)
    imageio.imwrite(str(path), mask_u8)


def _copy_inputs(
    input_dir: Path,
    image_path: Optional[str],
    mask_path: Optional[str],
    mask: np.ndarray,
    prior_image_paths: List[str],
    prior_mask_paths: List[str],
    prior_masks: List[np.ndarray],
) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)

    if image_path and Path(image_path).exists():
        shutil.copy2(image_path, input_dir / "image.png")

    if mask_path and Path(mask_path).exists():
        shutil.copy2(mask_path, input_dir / "mask.png")
    else:
        _save_mask_array(mask, input_dir / "mask.png")

    for idx, prior_path in enumerate(prior_image_paths):
        suffix = Path(prior_path).suffix or ".png"
        dst_path = input_dir / f"prior_{idx:02d}{suffix}"
        if prior_path and Path(prior_path).exists():
            shutil.copy2(prior_path, dst_path)

    if prior_mask_paths:
        for idx, prior_mask_path in enumerate(prior_mask_paths):
            suffix = Path(prior_mask_path).suffix or ".png"
            dst_path = input_dir / f"prior_mask_{idx:02d}{suffix}"
            if prior_mask_path and Path(prior_mask_path).exists():
                shutil.copy2(prior_mask_path, dst_path)
    else:
        for idx, prior_mask in enumerate(prior_masks):
            _save_mask_array(prior_mask, input_dir / f"prior_mask_{idx:02d}.png")


def _to_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(image)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def _render_standard_views(
    sample,
    out_dir: Path,
    resolution: int,
    backend: str,
    bg_color=(0, 0, 0),
) -> List[np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    yaws = [0.0, 0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi]
    pitchs = [30.0 * np.pi / 180.0] * 4
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, rs=2.0, fovs=40.0
    )
    render_options = {"resolution": resolution, "bg_color": bg_color, "backend": backend}
    res = render_utils.render_frames_rgba(
        sample, extrinsics, intrinsics, render_options, verbose=False
    )
    views = res.get("color", [])
    for i, view in enumerate(views):
        imageio.imwrite(str(out_dir / f"view_{i:02d}.png"), view)
    return views


def _load_rendered_frames(frames_dir: Path) -> List[np.ndarray]:
    if not frames_dir.exists():
        return []
    paths = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not paths:
        return []
    frames: List[np.ndarray] = []
    for path in paths:
        try:
            frames.append(np.array(imageio.imread(str(path))))
        except Exception:
            continue
    return frames


def _select_evenly_spaced(items: List[Any], count: int) -> List[Any]:
    if not items:
        return []
    if count <= 0 or len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int).tolist()
    selected: List[Any] = []
    last_idx = None
    for idx in indices:
        if last_idx is None or idx != last_idx:
            selected.append(items[idx])
            last_idx = idx
    return selected


def _save_frames_and_video(
    frames: List[np.ndarray],
    frames_dir: Path,
    video_path: Path,
    fps: int = 15,
    frame_stride: int = 1,
) -> None:
    if not frames:
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(frame_stride))
    saved_frames: List[np.ndarray] = []
    for idx, frame in enumerate(frames):
        if idx % stride != 0:
            continue
        imageio.imwrite(str(frames_dir / f"frame_{idx:04d}.png"), frame)
        saved_frames.append(frame)
    if not saved_frames:
        return
    with imageio.get_writer(str(video_path), fps=fps) as writer:
        for frame in saved_frames:
            if frame.ndim == 2:
                rgb_frame = np.stack([frame, frame, frame], axis=-1)
            elif frame.shape[-1] == 4:
                rgb_frame = frame[:, :, :3]
            else:
                rgb_frame = frame
            writer.append_data(rgb_frame)


def _format_duration(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"



def _render_spiral_views(
    sample,
    resolution: int,
    backend: str,
    num_frames: int,
    bg_color=(1, 1, 1),
    yaw_start_deg: float = -90.0,
    pitch_deg: float = 0.0,
    pitch_span_deg: float = 30.0,
    radius: float = 2.0,
    radius_span: float = 0.3,
    fov_deg: float = 40.0,
) -> List[np.ndarray]:
    yaws = torch.linspace(0, 2 * torch.pi, num_frames)
    t_vals = torch.linspace(0, 2 * torch.pi, num_frames)
    pitchs = (0.25 + 0.5 * torch.sin(t_vals)).tolist()
    radii = [radius] * num_frames
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws.tolist(), pitchs, radii, fov_deg
    )
    render_options = {
        "resolution": resolution,
        "bg_color": bg_color,
        "backend": backend,
    }
    res = render_utils.render_frames_rgba(sample, extr, intr, render_options, verbose=False)
    return res.get("color", [])


def _render_turntable_views(
    sample,
    resolution: int,
    backend: str,
    num_frames: int,
    bg_color=(0, 0, 0),
    yaw_start_deg: float = -90.0,
    pitch_deg: float = 0.0,
    radius: float = 2.0,
    fov_deg: float = 40.0,
) -> List[np.ndarray]:
    """Render turntable (fixed pitch) views around the object."""
    yaws = (
        torch.linspace(0, 2 * torch.pi, num_frames)
        + math.radians(yaw_start_deg)
    ).tolist()
    pitchs = [math.radians(pitch_deg)] * num_frames
    radii = [radius] * num_frames
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, radii, fov_deg
    )
    render_options = {
        "resolution": resolution,
        "bg_color": bg_color,
        "backend": backend,
    }
    res = render_utils.render_frames_rgba(sample, extr, intr, render_options, verbose=False)
    return res.get("color", [])


def _load_gt_images(
    entry: Dict, data_root: Optional[Path], base_dir: Path
) -> Tuple[List[np.ndarray], List[str]]:
    gt_images = _expand_list(entry.get("gt_images"))
    gt_render_dir = entry.get("gt_render_dir")
    paths: List[str] = []
    if gt_images:
        paths = [_resolve_path(p, base_dir, data_root) for p in gt_images]
    elif gt_render_dir:
        render_dir = Path(_resolve_path(gt_render_dir, base_dir, data_root))
        paths = sorted(
            str(p)
            for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in render_dir.glob(ext)
        )
    if not paths:
        return [], []
    images = []
    for p in paths:
        try:
            images.append(np.array(imageio.imread(p)))
        except Exception:
            continue
    return images, paths


def _copy_gt_images(paths: List[str], dst_dir: Path, frame_stride: int = 1) -> None:
    if not paths:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(frame_stride))
    for idx, p in enumerate(paths):
        if idx % stride != 0:
            continue
        src = Path(p)
        if not src.exists():
            continue
        shutil.copy2(src, dst_dir / src.name)


def _load_pointcloud(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    if path_obj.suffix == ".npy":
        return np.load(str(path_obj))
    if path_obj.suffix == ".pth":
        data = torch.load(str(path_obj), map_location="cpu")
        if isinstance(data, dict):
            data = data.get("points") or data.get("pc") or data.get("pointcloud")
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        if isinstance(data, np.ndarray):
            return data
    return None


def _mesh_to_pointcloud(mesh, device: str, points: int = 4096) -> Optional[torch.Tensor]:
    if mesh is None:
        return None
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            return None
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if hasattr(mesh, "vertices"):
        verts = mesh.vertices
        if isinstance(verts, torch.Tensor):
            verts = verts.detach().cpu().numpy()
        verts = np.asarray(verts)
    else:
        return None
    if verts.ndim != 2 or verts.shape[1] != 3:
        return None
    pts = torch.from_numpy(verts).float().unsqueeze(0)
    pts = pts.to(device)
    pts = fps_downsample(pts, points)
    return pts


def _atomic_json_dump(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _mean_metrics(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    keys = set().union(*[m.keys() for m in metric_dicts])
    avg: Dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in metric_dicts if key in m and not np.isnan(m[key])]
        if vals:
            avg[key] = float(np.mean(vals))
    return avg

def _run_vis_pass(
    pipeline,
    output: Dict,
    branch_name: str,
    save_dir: Path,
    render_backend: str,
    render_resolution: int,
    render_num_frames: int,
    render_frame_stride: int,
) -> None:
    """Run visualization rendering for a single branch output."""
    branch_out = output.get("relaxflow") if branch_name == "relaxflow" else output.get(branch_name)
    if branch_out is None:
        logger.warning("Missing branch output {} for visualization", branch_name)
        return

    gs = branch_out.get("gs") or (branch_out.get("gaussian") or [None])[0]
    if gs is None:
        logger.warning("Missing gaussian output for {}", branch_name)
        return

    branch_dir = save_dir / branch_name
    branch_dir.mkdir(parents=True, exist_ok=True)

    # Render spiral views (transparent background)
    try:
        spiral_frames = _render_spiral_views(
            gs,
            render_resolution,
            render_backend,
            render_num_frames,
            bg_color=(0, 0, 0),  # transparent bg
        )
        if spiral_frames:
            spiral_frames_dir = branch_dir / "spiral_frames_rgba"
            _save_frames_and_video(
                spiral_frames,
                spiral_frames_dir,
                branch_dir / f"{branch_name}_spiral.mp4",
                fps=15,
                frame_stride=render_frame_stride,
            )
            logger.info("Saved spiral video for {}", branch_name)
    except Exception as exc:
        logger.warning("Spiral render failed for {}: {}", branch_name, exc)

    # Render turntable views (transparent background)
    try:
        turntable_frames = _render_turntable_views(
            gs,
            render_resolution,
            render_backend,
            render_num_frames,
            bg_color=(0, 0, 0),  # transparent bg
        )
        if turntable_frames:
            turntable_frames_dir = branch_dir / "turntable_frames_rgba"
            _save_frames_and_video(
                turntable_frames,
                turntable_frames_dir,
                branch_dir / f"{branch_name}_turntable.mp4",
                fps=15,
                frame_stride=render_frame_stride,
            )
            logger.info("Saved turntable video for {}", branch_name)
    except Exception as exc:
        logger.warning("Turntable render failed for {}: {}", branch_name, exc)

    # Save mesh if available
    mesh_list = branch_out.get("mesh")
    glb_data = branch_out.get("glb")
    if mesh_list:
        try:
            mesh = mesh_list[0] if isinstance(mesh_list, (list, tuple)) else mesh_list
            if hasattr(mesh, "vertices"):
                mesh_path = branch_dir / f"{branch_name}.ply"
                if isinstance(mesh, trimesh.Trimesh):
                    mesh.export(str(mesh_path))
                elif hasattr(mesh, "save"):
                    mesh.save(str(mesh_path))
        except Exception as exc:
            logger.warning("Mesh save failed for {}: {}", branch_name, exc)
    if glb_data is not None:
        try:
            glb_path = branch_dir / f"{branch_name}.glb"
            if hasattr(glb_data, "export"):
                glb_data.export(str(glb_path))
            elif isinstance(glb_data, bytes):
                glb_path.write_bytes(glb_data)
        except Exception as exc:
            logger.warning("GLB save failed for {}: {}", branch_name, exc)


def build_pipeline(config_path: str, compile_model: bool, prior_mode: str, run_obs_only: bool = False):
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = compile_model
    config.workspace_dir = os.path.dirname(config_path)
    # Use base pipeline (no prior images needed) when run_obs_only is True
    config["_target_"] = PIPELINE_TARGET_BASE if run_obs_only else PIPELINE_TARGET_RELAXFLOW

    # Only add RELAXFLOW-specific parameters when not running obs_only mode
    if not run_obs_only:
        if "prior_blur_sigma" not in config:
            config.prior_blur_sigma = 2.5
        if "blur_attn_type" not in config:
            config.blur_attn_type = "self"

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
    else:
        for key in ["prior_blur_sigma", "blur_attn_type", "prior_mode",
                    "prior_use_cropped_only", "prior_only_cropped_img_and_mask"]:
            config.pop(key, None)

    check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
    return instantiate(config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch RelaxFlow runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--config",
        default="checkpoints/hf/checkpoints/pipeline.yaml",
        help="Path to the pipeline config file.",
    )
    io_group.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset manifest (JSON/JSONL).",
    )
    io_group.add_argument(
        "--data-root",
        default=None,
        help="Optional root directory to resolve relative paths in the manifest.",
    )
    io_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs/relaxflow_batch",
        help="Output directory.",
    )
    io_group.add_argument(
        "--output-name",
        type=str,
        default="run",
        help="Output name prefix.",
    )
    io_group.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of workers for sharded evaluation.",
    )
    io_group.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Worker rank in [0, world_size).",
    )
    io_group.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional limit of samples to process after sharding (-1 for all).",
    )

    pipeline_group = parser.add_argument_group("Pipeline Options")
    pipeline_group.add_argument(
        "--image-as",
        type=str,
        default="part",
        choices=["scene", "part"],
        help="Route observed image into 'part' or 'scene' condition slot.",
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
    pipeline_group.add_argument(
        "--compute-original",
        action="store_true",
        help="Also sample the original (non-relaxflow) branch.",
    )
    pipeline_group.add_argument(
        "--compute-blend-prior-slat",
        action="store_true",
        help="Also decode the blend_prior_slat branch.",
    )
    pipeline_group.add_argument(
        "--attn-blur-type",
        choices=["cross", "self", "both"],
        default="cross",
        help="Alias for --blur-attn-type (Stage 1). If set, also applies to Stage 2 unless overridden.",
    )
    pipeline_group.add_argument(
        "--stage2-attn-blur-type",
        choices=["cross", "self", "both"],
        default="cross",
        help="Alias for --stage2-blur-attn-type (Stage 2).",
    )
    pipeline_group.add_argument(
        "--run-obs-only",
        action="store_true",
        default=False,
        help="Run only the obs_only branch (vanilla SAM3D), skip RELAXFLOW blending.",
    )

    vis_group = parser.add_argument_group("Visualization Mode")
    vis_group.add_argument(
        "--vis",
        action="store_true",
        default=True,
        help="Enable visualization mode: run both blurred and no-blur, render spiral+turntable, skip metrics.",
    )

    render_group = parser.add_argument_group("Rendering Options")
    render_group.add_argument(
        "--render-backend",
        default="inria",
        choices=["inria", "gsplat"],
        help="Rendering backend for turntable video.",
    )
    render_group.add_argument(
        "--render-resolution",
        type=int,
        default=512,
        help="Resolution for standard view renders.",
    )
    render_group.add_argument(
        "--render-num-frames",
        type=int,
        default=120,
        help="Frames for turntable video.",
    )
    render_group.add_argument(
        "--render-frame-stride",
        type=int,
        default=2,
        help="Stride for saving rendered frames (e.g. 2 saves every other frame).",
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

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--eval-branches",
        nargs="+",
        default=["relaxflow", "blend_obs_slat", "obs_only"],
        help="Branches to evaluate.",
    )
    eval_group.add_argument(
        "--metrics-device",
        default="cuda",
        help="Device for metric computation.",
    )
    eval_group.add_argument(
        "--clip-text-model",
        default="openai/clip-vit-base-patch32",
        help="Model name for image-text CLIPScore.",
    )
    eval_group.add_argument(
        "--clip-image-model",
        default="openai/clip-vit-base-patch32",
        help="Model name for image-image CLIP similarity.",
    )
    eval_group.add_argument(
        "--omit-point-level-metrics",
        dest="omit_point_level_metrics",
        action="store_true",
        default=True,
        help="Omit point-level 3D metrics (default: enabled).",
    )
    eval_group.add_argument(
        "--keep-point-level-metrics",
        dest="omit_point_level_metrics",
        action="store_false",
        help="Enable point-level 3D metrics (Chamfer, F-score, voxel IoU, COV/MMD).",
    )
    eval_group.add_argument(
        "--eval-frames",
        type=int,
        default=10,
        help="Evenly spaced frames per branch for CLIP/LPIPS metrics.",
    )
    eval_group.add_argument(
        "--skip-pfid",
        action="store_true",
        help="Skip Point-E P-FID computation.",
    )
    eval_group.add_argument(
        "--cache-root",
        default=_default_cache_root(),
        help="Cache directory for HF and temp files.",
    )

    add_relaxflow_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.attn_blur_type is not None:
        args.blur_attn_type = args.attn_blur_type
        if args.stage2_attn_blur_type is None:
            args.stage2_blur_attn_type = args.attn_blur_type
    if args.stage2_attn_blur_type is not None:
        args.stage2_blur_attn_type = args.stage2_attn_blur_type

    if args.vis:
        logger.info("Visualization mode enabled: skipping metrics, rendering spiral+turntable")
        args.render_num_frames = 120
        args.render_frame_stride = 1

    run_obs_only = getattr(args, "run_obs_only", False)
    if run_obs_only:
        logger.info("Running obs_only mode: using base pipeline (no RELAXFLOW), only vanilla SAM3D.")
        args.eval_branches = ["obs_only"]
        args.compute_original = False
        args.compute_blend_prior_slat = False

    warnings.filterwarnings(
        "ignore",
        message="Bin size was too small in the coarse rasterization phase.*",
    )
    _set_cache_env(args.cache_root)
    cache_root = Path(args.cache_root).expanduser()
    if not cache_root.is_absolute():
        cache_root = (Path.cwd() / cache_root).resolve()
    else:
        cache_root = cache_root.resolve()
    point_e_cache_dir = os.environ.get("POINT_E_CACHE_DIR")
    if not point_e_cache_dir:
        point_e_cache_dir = str(cache_root / "point_e_cache")
    Path(point_e_cache_dir).mkdir(parents=True, exist_ok=True)

    if args.only_cropped_img and args.only_cropped_img_and_mask:
        raise ValueError("--only-cropped-img and --only-cropped-img-and-mask are mutually exclusive")

    metrics_device = args.metrics_device
    if metrics_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU for metrics.")
        metrics_device = "cpu"
    if args.skip_pfid:
        logger.warning("P-FID is skipped; remove --skip-pfid to enable it.")
    if args.omit_point_level_metrics:
        logger.info(
            "Omitting point-level 3D metrics (Chamfer, F-score, voxel IoU, COV/MMD)."
        )

    relaxflow_config = RELAXFLOWConfig.from_args(args)
    relaxflow_config.validate()
    summary = print_config_summary(relaxflow_config)
    print(summary)

    out_root = Path(args.output_dir) / args.output_name
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config_summary.txt").write_text(summary)

    metrics_suffix = f"rank{args.rank}" if args.world_size > 1 else None
    per_sample_path = (
        out_root / f"per_sample_metrics_{metrics_suffix}.json"
        if metrics_suffix
        else out_root / "per_sample_metrics.json"
    )
    summary_path = (
        out_root / f"summary_{metrics_suffix}.json"
        if metrics_suffix
        else out_root / "summary.json"
    )
    metadata_path = (
        out_root / f"sample_metadata_{metrics_suffix}.json"
        if metrics_suffix
        else out_root / "sample_metadata.json"
    )
    failed_path = (
        out_root / f"failed_samples_{metrics_suffix}.json"
        if metrics_suffix
        else out_root / "failed_samples.json"
    )
    label_summary_path = (
        out_root / f"label_summary_{metrics_suffix}.json"
        if metrics_suffix
        else out_root / "label_summary.json"
    )
    summary_table_path = (
        out_root / f"summary_table_{metrics_suffix}.tex"
        if metrics_suffix
        else out_root / "summary_table.tex"
    )
    label_table_path = (
        out_root / f"label_summary_table_{metrics_suffix}.tex"
        if metrics_suffix
        else out_root / "label_summary_table.tex"
    )

    manifest_path = Path(args.dataset)
    samples = _load_manifest(manifest_path)
    base_dir = manifest_path.parent
    data_root = Path(args.data_root) if args.data_root else None

    if args.world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world_size")

    if args.world_size > 1:
        samples = [s for idx, s in enumerate(samples) if idx % args.world_size == args.rank]
        logger.info(
            "Rank {} assigned {} samples (world_size={})",
            args.rank,
            len(samples),
            args.world_size,
        )

    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        logger.info("Limiting to {} samples", len(samples))

    logger.info("Building RelaxFlow pipeline once...")
    pipeline = build_pipeline(args.config, args.compile, relaxflow_config.prior_mode, run_obs_only)

    run_kwargs = relaxflow_config.to_run_kwargs()
    run_kwargs.pop("disable_geometry_mask", None)

    fid_calcs: Dict[str, FIDCalculator] = {}
    kid_calcs: Dict[str, KIDCalculator] = {}
    fid_kid_updates: Dict[str, int] = {}

    clip_text_calc = None
    clip_image_calc = None
    lpips_calc = None

    per_sample_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    sample_metadata: Dict[str, Dict[str, str]] = {}
    branch_pc_sets: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    branch_clip_text_scores: Dict[str, List[float]] = {}
    branch_clip_image_scores: Dict[str, List[float]] = {}
    failed_samples: List[Dict[str, str]] = []

    start_time = time.time()
    total_samples = len(samples)
    for idx, entry in enumerate(samples):
        sample_id = entry.get("id") or f"sample_{idx:05d}"
        elapsed = time.time() - start_time
        if idx > 0:
            avg_per_sample = elapsed / idx
            eta = avg_per_sample * (total_samples - idx)
            eta_str = _format_duration(eta)
        else:
            eta_str = "n/a"
        logger.info(
            "[PROGRESS] Sample {}/{}: {} (elapsed {}, ETA {})",
            idx + 1,
            total_samples,
            sample_id,
            _format_duration(elapsed),
            eta_str,
        )
        sample_dir = out_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        per_sample_metrics.setdefault(sample_id, {})
        sample_metadata.setdefault(sample_id, {})
        try:
            (
                image_path,
                mask_path,
                image,
                mask,
                prior_images,
                prior_masks,
                prior_image_paths,
                prior_mask_paths,
            ) = _load_inputs(entry, data_root, base_dir)
        except Exception as exc:
            logger.exception("Failed to load inputs for {}: {}", sample_id, exc)
            sample_metadata[sample_id]["error"] = str(exc)
            failed_samples.append({"id": sample_id, "error": str(exc)})
            _atomic_json_dump(per_sample_path, per_sample_metrics)
            _atomic_json_dump(metadata_path, sample_metadata)
            _atomic_json_dump(failed_path, failed_samples)
            continue

        if _mask_too_small(mask):
            msg = "mask bbox < 2px; skipping sample"
            logger.warning("{}: {}", sample_id, msg)
            sample_metadata[sample_id]["error"] = msg
            failed_samples.append({"id": sample_id, "error": msg})
            _atomic_json_dump(per_sample_path, per_sample_metrics)
            _atomic_json_dump(metadata_path, sample_metadata)
            _atomic_json_dump(failed_path, failed_samples)
            continue

        _copy_inputs(
            sample_dir / "inputs",
            image_path,
            mask_path,
            mask,
            prior_image_paths,
            prior_mask_paths,
            prior_masks,
        )

        if relaxflow_config.prior_mode == "cropped":
            prior_use_cropped_only = True
            prior_only_cropped_img_and_mask = False
        elif relaxflow_config.prior_mode == "full":
            prior_use_cropped_only = False
            prior_only_cropped_img_and_mask = False
        else:
            prior_use_cropped_only = False
            prior_only_cropped_img_and_mask = True

        if args.vis:
            vis_configs = [
                ("blurred", run_kwargs.copy()),
                ("no_blur", {**run_kwargs, "blur_attn_type": "self", "stage2_blur_attn_type": "self"}),
            ]
            for vis_name, vis_run_kwargs in vis_configs:
                vis_save_dir = sample_dir / vis_name
                vis_save_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Running {} pass for {}", vis_name, sample_id)
                try:
                    vis_output = pipeline.run(
                        image=image,
                        mask=mask,
                        prior_images=prior_images,
                        prior_masks=prior_masks,
                        stage1_only=args.stage1_only,
                        with_mesh_postprocess=False,
                        with_texture_baking=False,
                        with_layout_postprocess=True,
                        use_vertex_color=args.use_vertex_color,
                        only_cropped_img=args.only_cropped_img,
                        only_cropped_img_and_mask=args.only_cropped_img_and_mask,
                        image_as=args.image_as,
                        prior_use_cropped_only=prior_use_cropped_only,
                        prior_only_cropped_img_and_mask=prior_only_cropped_img_and_mask,
                        return_branch_outputs=True,
                        save_render_dir=str(vis_save_dir),
                        render_backend=args.render_backend,
                        render_resolution=args.render_resolution,
                        render_num_frames=args.render_num_frames,
                        render_bg_color=(0, 0, 0),  # Transparent background
                        render_frame_stride=args.render_frame_stride,
                        compute_original=args.compute_original,
                        compute_blend_prior_slat=args.compute_blend_prior_slat,
                        **vis_run_kwargs,
                    )
                except Exception as exc:
                    logger.exception("Pipeline failed for {}/{}: {}", sample_id, vis_name, exc)
                    sample_metadata[sample_id][f"error_{vis_name}"] = str(exc)
                    continue

                # Render both spiral and turntable for each branch
                vis_branches = ["relaxflow", "blend_obs_slat", "obs_only"]
                for branch in vis_branches:
                    _run_vis_pass(
                        pipeline,
                        vis_output,
                        branch,
                        vis_save_dir,
                        args.render_backend,
                        args.render_resolution,
                        args.render_num_frames,
                        args.render_frame_stride,
                    )
            # Skip metrics in vis mode
            _atomic_json_dump(metadata_path, sample_metadata)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        try:
            if run_obs_only:
                # Use base pipeline (InferencePipelinePointMap) - no prior images needed
                base_output = pipeline.run(
                    image=image,
                    mask=mask,
                    stage1_only=args.stage1_only,
                    with_mesh_postprocess=False,
                    with_texture_baking=False,
                    with_layout_postprocess=True,
                    use_vertex_color=args.use_vertex_color,
                    only_cropped_img=args.only_cropped_img,
                    only_cropped_img_and_mask=args.only_cropped_img_and_mask,
                )
                # Wrap in obs_only format for consistent downstream processing
                output = {"obs_only": base_output}
            else:
                # Use RELAXFLOW pipeline with prior images
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
                    only_cropped_img=args.only_cropped_img,
                    only_cropped_img_and_mask=args.only_cropped_img_and_mask,
                    image_as=args.image_as,
                    prior_use_cropped_only=prior_use_cropped_only,
                    prior_only_cropped_img_and_mask=prior_only_cropped_img_and_mask,
                    return_branch_outputs=True,
                    save_render_dir=str(sample_dir),
                    render_backend=args.render_backend,
                    render_resolution=args.render_resolution,
                    render_num_frames=args.render_num_frames,
                    render_bg_color=(1, 1, 1),
                    render_frame_stride=args.render_frame_stride,
                    compute_original=args.compute_original,
                    compute_blend_prior_slat=args.compute_blend_prior_slat,
                    **run_kwargs,
                )
        except Exception as exc:
            logger.exception("Pipeline failed for {}: {}", sample_id, exc)
            sample_metadata[sample_id]["error"] = str(exc)
            failed_samples.append({"id": sample_id, "error": str(exc)})
            _atomic_json_dump(per_sample_path, per_sample_metrics)
            _atomic_json_dump(metadata_path, sample_metadata)
            _atomic_json_dump(failed_path, failed_samples)
            continue

        per_sample_metrics[sample_id] = {}

        prior_text = entry.get("prior_text") or entry.get("text") or entry.get("prompt")
        if prior_text:
            sample_metadata[sample_id]["label"] = prior_text
            if clip_text_calc is None:
                try:
                    clip_text_calc = CLIPScoreCalculator(
                        device=metrics_device, model_name=args.clip_text_model
                    )
                except Exception as exc:
                    logger.warning("CLIPScore unavailable: {}", exc)
                    clip_text_calc = None
        obs_object_path = _resolve_obs_object_image(entry, base_dir, data_root, image_path)
        obs_object_img = None
        obs_object_img_proc = None
        if obs_object_path and Path(obs_object_path).exists():
            try:
                obs_object_img = np.array(imageio.imread(obs_object_path))
                obs_object_img_proc = _crop_and_center_object(obs_object_img)
            except Exception as exc:
                logger.warning("Failed to read obs object image for {}: {}", sample_id, exc)

        gt_pc = _load_pointcloud(
            _resolve_path(entry.get("gt_pointcloud"), base_dir, data_root)
        )
        gt_mesh_path = _resolve_path(entry.get("gt_mesh"), base_dir, data_root)
        gt_mesh = None
        if gt_mesh_path:
            try:
                gt_mesh = trimesh.load(gt_mesh_path, force="mesh")
            except Exception:
                gt_mesh = None

        gt_images, gt_image_paths = _load_gt_images(entry, data_root, base_dir)
        _copy_gt_images(
            gt_image_paths,
            sample_dir / "gt_renders",
            frame_stride=args.render_frame_stride,
        )
        if clip_image_calc is None and (gt_images or obs_object_img is not None):
            try:
                clip_image_calc = CLIPImageSimilarity(
                    device=metrics_device, model_name=args.clip_image_model
                )
            except Exception as exc:
                logger.warning("CLIP image similarity unavailable: {}", exc)
                clip_image_calc = None
        if lpips_calc is None and obs_object_img_proc is not None:
            if _HAS_LPIPS:
                try:
                    lpips_calc = LearnedPerceptualImagePatchSimilarity(
                        net_type="alex", normalize=True
                    ).to(metrics_device)
                except Exception as exc:
                    logger.warning("LPIPS unavailable: {}", exc)
                    lpips_calc = None
            else:
                logger.warning("LPIPS unavailable: torchmetrics LPIPS not installed.")

        for branch in args.eval_branches:
            branch_out = None
            if isinstance(output, dict):
                if branch == "relaxflow":
                    branch_out = output.get("relaxflow")
                else:
                    branch_out = output.get(branch)
            if branch_out is None:
                logger.warning("Missing branch output {} for {}", branch, sample_id)
                continue

            branch_dir = sample_dir / branch
            frames_dir = branch_dir / "frames_rgba"
            pred_views = _load_rendered_frames(frames_dir)
            if not pred_views:
                legacy_dir = branch_dir / "frames"
                pred_views = _load_rendered_frames(legacy_dir)
                if pred_views:
                    frames_dir = legacy_dir
            if not pred_views:
                gs = branch_out.get("gs") or (branch_out.get("gaussian") or [None])[0]
                if gs is None:
                    logger.warning("Missing gaussian output for {}/{}", sample_id, branch)
                    continue
                try:
                    pred_views = _render_spiral_views(
                        gs,
                        args.render_resolution,
                        args.render_backend,
                        args.render_num_frames,
                    )
                    _save_frames_and_video(
                        pred_views,
                        branch_dir / "frames_rgba",
                        branch_dir / f"{branch}.mp4",
                        fps=15,
                        frame_stride=args.render_frame_stride,
                    )
                except Exception as exc:
                    logger.warning("Render failed for {}/{}: {}", sample_id, branch, exc)
                    continue
            if not pred_views:
                logger.warning("No rendered views for {}/{}", sample_id, branch)
                continue

            metrics: Dict[str, float] = {}
            pred_views_eval = _select_evenly_spaced(pred_views, args.eval_frames)

            if gt_images and has_2d_metrics():
                try:
                    if branch not in fid_calcs:
                        fid_calcs[branch] = FIDCalculator(device=metrics_device)
                        kid_calcs[branch] = KIDCalculator(
                            device=metrics_device,
                            subsets=len(samples),
                            subset_size=min(len(samples), 1000),
                        )
                    fid_calcs[branch].update(gt_images, pred_views)
                    kid_calcs[branch].update(gt_images, pred_views)
                    fid_kid_updates[branch] = fid_kid_updates.get(branch, 0) + 1
                except Exception as exc:
                    logger.warning(
                        "FID/KID update failed for {}/{}: {}", sample_id, branch, exc
                    )

            if prior_text and clip_text_calc is not None:
                clip_score = clip_text_calc.compute(pred_views_eval, prior_text)
                metrics["clip_image_text"] = clip_score
                branch_clip_text_scores.setdefault(branch, []).append(clip_score)

            pred_views_proc = [_crop_and_center_object(v) for v in pred_views_eval]
            pred_clip_feats = None
            if clip_image_calc is not None:
                try:
                    pred_clip_feats = torch.cat(
                        [clip_image_calc.encode(_to_pil_rgb(v)) for v in pred_views_proc],
                        dim=0,
                    )
                except Exception as exc:
                    logger.warning(
                        "CLIP encoding failed for {}/{}: {}", sample_id, branch, exc
                    )
                    pred_clip_feats = None

            gt_images_proc = [_crop_and_center_object(img) for img in gt_images] if gt_images else []
            gt_ref_images = list(gt_images_proc)
            if obs_object_img_proc is not None:
                gt_ref_images.append(obs_object_img_proc)

            if gt_ref_images and pred_clip_feats is not None and clip_image_calc is not None:
                try:
                    gt_feats = torch.cat(
                        [clip_image_calc.encode(_to_pil_rgb(img)) for img in gt_ref_images],
                        dim=0,
                    )
                    gt_feat = gt_feats.mean(dim=0, keepdim=True)
                    gt_feat = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
                    sims = (pred_clip_feats * gt_feat).sum(dim=-1)
                    clip_img_score = float(sims.mean().item())
                    metrics["clip_image_image"] = clip_img_score
                    metrics["clip_image_image_gt_max"] = float(sims.max().item())
                    branch_clip_image_scores.setdefault(branch, []).append(clip_img_score)
                except Exception as exc:
                    logger.warning(
                        "GT CLIP image similarity failed for {}/{}: {}", sample_id, branch, exc
                    )

            if obs_object_img_proc is not None and pred_clip_feats is not None:
                try:
                    obs_feat = clip_image_calc.encode(_to_pil_rgb(obs_object_img_proc))
                    sims = (pred_clip_feats * obs_feat).sum(dim=-1)
                    metrics["clip_obs_image_max"] = float(sims.max().item())
                    metrics["clip_obs_image_mean"] = float(sims.mean().item())
                except Exception as exc:
                    logger.warning(
                        "Obs CLIP similarity failed for {}/{}: {}", sample_id, branch, exc
                    )

            if obs_object_img_proc is not None and lpips_calc is not None:
                try:
                    obs_tensor = _prep_lpips_tensor(obs_object_img_proc).to(metrics_device)
                    best_lpips = None
                    for view in pred_views_proc:
                        view_tensor = _prep_lpips_tensor(view).to(metrics_device)
                        lpips_val = float(lpips_calc(obs_tensor, view_tensor).item())
                        best_lpips = lpips_val if best_lpips is None else min(best_lpips, lpips_val)
                    if best_lpips is not None:
                        metrics["obs_lpips_min"] = best_lpips
                except Exception as exc:
                    logger.warning("Obs LPIPS failed for {}/{}: {}", sample_id, branch, exc)

            need_pointclouds = (not args.omit_point_level_metrics) or (not args.skip_pfid)
            if need_pointclouds:
                pred_mesh = None
                mesh_list = branch_out.get("mesh")
                if isinstance(mesh_list, (list, tuple)) and mesh_list:
                    pred_mesh = mesh_list[0]
                elif isinstance(mesh_list, MeshExtractResult):
                    pred_mesh = mesh_list
                if pred_mesh is None:
                    pred_mesh = branch_out.get("glb")

                pred_pc = _mesh_to_pointcloud(pred_mesh, metrics_device)
                if pred_pc is not None and (gt_pc is not None or gt_mesh is not None):
                    if gt_pc is None and gt_mesh is not None:
                        gt_pc = np.asarray(gt_mesh.vertices)
                    if gt_pc is not None:
                        gt_pts = torch.from_numpy(gt_pc).float().unsqueeze(0)
                        gt_pts = gt_pts.to(metrics_device)
                        gt_pts = fps_downsample(gt_pts, pred_pc.shape[1])
                        if not args.omit_point_level_metrics:
                            try:
                                metrics.update(
                                    compute_3d_metrics_from_pointclouds(
                                        pred_pc, gt_pts, device=metrics_device, use_icp=True
                                    )
                                )
                            except Exception as exc:
                                logger.warning(
                                    "3D metrics failed for {}/{}: {}", sample_id, branch, exc
                                )
                        if not args.skip_pfid or not args.omit_point_level_metrics:
                            branch_pc_sets.setdefault(branch, {"gen": [], "gt": []})
                            branch_pc_sets[branch]["gen"].append(
                                pred_pc.squeeze(0).detach().cpu()
                            )
                            branch_pc_sets[branch]["gt"].append(
                                gt_pts.squeeze(0).detach().cpu()
                            )

            per_sample_metrics[sample_id][branch] = metrics

        _atomic_json_dump(per_sample_path, per_sample_metrics)
        _atomic_json_dump(metadata_path, sample_metadata)
        _atomic_json_dump(failed_path, failed_samples)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary: Dict[str, Dict[str, float]] = {}
    for branch in args.eval_branches:
        branch_metrics = []
        for sample_id in per_sample_metrics:
            metrics = per_sample_metrics[sample_id].get(branch)
            if metrics:
                branch_metrics.append(metrics)

        avg: Dict[str, float] = _mean_metrics(branch_metrics)

        if branch in fid_calcs and fid_kid_updates.get(branch, 0) > 0:
            try:
                avg["fid_set"] = fid_calcs[branch].compute()
            except Exception as exc:
                logger.warning("FID/KID compute failed for {}: {}", branch, exc)
            if fid_kid_updates.get(branch, 0) > 1:
                try:
                    kid_mean, kid_std = kid_calcs[branch].compute()
                    if not np.isnan(kid_mean):
                        avg["kid_mean_set"] = kid_mean
                        avg["kid_std_set"] = kid_std
                    else:
                        logger.warning("KID returned NaN for {}; skipping.", branch)
                except Exception as exc:
                    logger.warning("KID compute failed for {}: {}", branch, exc)
            else:
                logger.warning("KID requires >=2 samples; skipping for {}.", branch)

        if branch in branch_clip_text_scores:
            avg["clip_image_text_mean"] = float(np.mean(branch_clip_text_scores[branch]))
        if branch in branch_clip_image_scores:
            avg["clip_image_image_mean"] = float(np.mean(branch_clip_image_scores[branch]))

        if branch in branch_pc_sets:
            if not args.omit_point_level_metrics:
                cov_mmd = cov_mmd_from_sets(
                    branch_pc_sets[branch]["gen"], branch_pc_sets[branch]["gt"]
                )
                if cov_mmd:
                    avg.update(cov_mmd)
            if not args.skip_pfid:
                pred_npz_dir = out_root / "pfid" / branch / "pred"
                gt_npz_dir = out_root / "pfid" / branch / "gt"
                save_batch_npz_files(branch_pc_sets[branch]["gen"], str(pred_npz_dir))
                save_batch_npz_files(branch_pc_sets[branch]["gt"], str(gt_npz_dir))
                pfid = compute_pfid_with_pointe(
                    str(pred_npz_dir),
                    str(gt_npz_dir),
                    cache_dir=point_e_cache_dir,
                )
                if pfid is not None:
                    avg["pfid"] = pfid
                else:
                    logger.warning(
                        "P-FID unavailable; check Point-E install/cache at {}",
                        point_e_cache_dir,
                    )

        if avg:
            summary[branch] = avg

    label_groups: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    for sample_id, branches in per_sample_metrics.items():
        label = sample_metadata.get(sample_id, {}).get("label")
        if not label:
            continue
        for branch, metrics in branches.items():
            if not metrics:
                continue
            label_groups.setdefault(label, {}).setdefault(branch, []).append(metrics)

    label_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, branches in label_groups.items():
        for branch, metrics_list in branches.items():
            avg = _mean_metrics(metrics_list)
            if avg:
                label_summary.setdefault(label, {})[branch] = avg

    _atomic_json_dump(per_sample_path, per_sample_metrics)
    _atomic_json_dump(summary_path, summary)
    _atomic_json_dump(metadata_path, sample_metadata)
    _atomic_json_dump(label_summary_path, label_summary)
    _atomic_json_dump(failed_path, failed_samples)

    logger.info("Saved metrics to {}", out_root)


if __name__ == "__main__":
    main()
