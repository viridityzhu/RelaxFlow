# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# not ideal to put that here
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", os.environ.get("CUDA_HOME", ""))
os.environ["LIDRA_SKIP_INIT"] = "true"
import sys
from typing import Union, Optional, List, Callable
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
import torch
import math
import utils3d
import shutil
import subprocess
import seaborn as sns
from PIL import Image
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from copy import deepcopy
from kaolin.visualize import IpyTurntableVisualizer
from kaolin.render.camera import Camera, CameraExtrinsics, PinholeIntrinsics
import builtins
from pytorch3d.transforms import quaternion_multiply, quaternion_invert

import sam3d_objects  # REMARK(Pierre) : do not remove this import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

from sam3d_objects.utils.visualization import SceneVisualizer

__all__ = ["Inference"]

_rembg_session = None

WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.fchdir,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]


class Inference:
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(self, config_file: str, compile: bool = False):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def merge_mask_to_rgba(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        pointmap=None,
        only_cropped_img: bool = False,
        only_cropped_img_and_mask: bool = False,
    ) -> dict:
        assert not (only_cropped_img and only_cropped_img_and_mask), "only_cropped_img and only_cropped_img_and_mask cannot be True at the same time"
        image = self.merge_mask_to_rgba(image, mask)
        # # save image_rgba to output directory
        # image_rgba_path = os.path.join(save_render_dir, "image_rgba.png")
        # # image_rgba is RGBA, save png with alpha channel
        # imageio.imwrite(image_rgba_path, image_rgba)
        # # save the alpha channel to output directory
        # alpha_channel_path = os.path.join(save_render_dir, "alpha_channel.png")
        # imageio.imwrite(alpha_channel_path, image_rgba[:, :, 3])
        # # print range of alpha channel, as well as range of rgb channels
        # logger.info(f"Alpha channel range: {image_rgba[:, :, 3].min()}, {image_rgba[:, :, 3].max()}")
        # logger.info(f"RGB channels range: {image_rgba[:, :, :3].min()}, {image_rgba[:, :, :3].max()}")
        # logger.info(f"Saved image_rgba to {image_rgba_path}")
        return self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
            only_cropped_img=only_cropped_img,
            only_cropped_img_and_mask=only_cropped_img_and_mask,
        )


def _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                ]
            ).cuda()
            * r
        )
        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 1, 0]).float().cuda(),
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_video(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2.0,
    fov=40,
    pitch_deg=0,
    yaw_start_deg=-90,
    backend="gsplat",
    fallback_backend="inria",
    **kwargs,
):

    yaws = (
        torch.linspace(0, 2 * torch.pi, num_frames) + math.radians(yaw_start_deg)
    ).tolist()
    pitch = [math.radians(pitch_deg)] * num_frames

    extr, intr = _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)

    render_options = {
        "resolution": resolution,
        "bg_color": bg_color,
        "backend": backend,
    }

    try:
        return render_utils.render_frames(
            sample,
            extr,
            intr,
            render_options,
            **kwargs,
        )
    except AttributeError as err:
        needs_fallback = (
            backend.lower() == "gsplat"
            and fallback_backend is not None
            and "fully_fused_projection" in str(err)
        )
        if not needs_fallback:
            raise

        print(
            f"gsplat backend failed with '{err}'. Falling back to '{fallback_backend}'."
        )
        render_options["backend"] = fallback_backend
        return render_utils.render_frames(
            sample,
            extr,
            intr,
            render_options,
            **kwargs,
        )


def ready_gaussian_for_video_rendering(scene_gs, in_place=False, fix_alignment=False):
    if fix_alignment:
        scene_gs = _fix_gaussian_alignment(scene_gs, in_place=in_place)
    scene_gs = normalized_gaussian(scene_gs, in_place=fix_alignment)
    return scene_gs


def _fix_gaussian_alignment(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    device = scene_gs._xyz.device
    dtype = scene_gs._xyz.dtype
    scene_gs._xyz = (
        scene_gs._xyz
        @ torch.tensor(
            [
                [-1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            device=device,
            dtype=dtype,
        ).T
    )
    return scene_gs


def normalized_gaussian(scene_gs, in_place=False, outlier_percentile=None):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    orig_xyz = scene_gs.get_xyz
    orig_scale = scene_gs.get_scaling

    active_mask = (scene_gs.get_opacity > 0.9).squeeze()
    inv_scale = (
        orig_xyz[active_mask].max(dim=0)[0] - orig_xyz[active_mask].min(dim=0)[0]
    ).max()
    norm_scale = orig_scale / inv_scale
    norm_xyz = orig_xyz / inv_scale

    if outlier_percentile is None:
        lower_bound_xyz = torch.min(norm_xyz[active_mask], dim=0)[0]
        upper_bound_xyz = torch.max(norm_xyz[active_mask], dim=0)[0]
    else:
        lower_bound_xyz = torch.quantile(
            norm_xyz[active_mask],
            outlier_percentile,
            dim=0,
        )
        upper_bound_xyz = torch.quantile(
            norm_xyz[active_mask],
            1.0 - outlier_percentile,
            dim=0,
        )

    center = (lower_bound_xyz + upper_bound_xyz) / 2
    norm_xyz = norm_xyz - center
    scene_gs.from_xyz(norm_xyz)
    scene_gs.mininum_kernel_size /= inv_scale.item()
    scene_gs.from_scaling(norm_scale)
    return scene_gs


def make_scene(*outputs, in_place=False):
    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_outs = []
    minimum_kernel_size = float("inf")
    for output in outputs:
        # move gaussians to scene frame of reference
        PC = SceneVisualizer.object_pointcloud(
            points_local=output["gaussian"][0].get_xyz.unsqueeze(0),
            quat_l2c=output["rotation"],
            trans_l2c=output["translation"],
            scale_l2c=output["scale"],
        )
        output["gaussian"][0].from_xyz(PC.points_list()[0])
        # must ... ROTATE
        output["gaussian"][0].from_rotation(
            quaternion_multiply(
                quaternion_invert(output["rotation"]),
                output["gaussian"][0].get_rotation,
            )
        )
        scale = output["gaussian"][0].get_scaling
        adjusted_scale = scale * output["scale"]
        assert (
            output["scale"][0, 0].item()
            == output["scale"][0, 1].item()
            == output["scale"][0, 2].item()
        )
        output["gaussian"][0].mininum_kernel_size *= output["scale"][0, 0].item()
        adjusted_scale = torch.maximum(
            adjusted_scale,
            torch.tensor(
                output["gaussian"][0].mininum_kernel_size * 1.1,
                device=adjusted_scale.device,
            ),
        )
        output["gaussian"][0].from_scaling(adjusted_scale)
        minimum_kernel_size = min(
            minimum_kernel_size,
            output["gaussian"][0].mininum_kernel_size,
        )
        all_outs.append(output)

    # merge gaussians
    scene_gs = all_outs[0]["gaussian"][0]
    scene_gs.mininum_kernel_size = minimum_kernel_size
    for out in all_outs[1:]:
        out_gs = out["gaussian"][0]
        scene_gs._xyz = torch.cat([scene_gs._xyz, out_gs._xyz], dim=0)
        scene_gs._features_dc = torch.cat(
            [scene_gs._features_dc, out_gs._features_dc], dim=0
        )
        scene_gs._scaling = torch.cat([scene_gs._scaling, out_gs._scaling], dim=0)
        scene_gs._rotation = torch.cat([scene_gs._rotation, out_gs._rotation], dim=0)
        scene_gs._opacity = torch.cat([scene_gs._opacity, out_gs._opacity], dim=0)

    return scene_gs


def check_target(
    target: str,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    if any(filt(target) for filt in whitelist_filters):
        if not any(filt(target) for filt in blacklist_filters):
            return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))


def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    # Ensure we always have at least 3 channels for downstream concatenation.
    if image.ndim == 2:  # grayscale
        image = np.stack([image, image, image], axis=-1)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask


def mask_from_alpha_channel(image: np.ndarray):
    """
    Extract a boolean mask from the alpha channel if present.
    Returns None when no alpha channel is available.
    """
    print("mask directly from alpha channel")
    if image.ndim == 3 and image.shape[-1] == 4:
        alpha = image[..., 3]
        return alpha > 0
    return None


def mask_from_rembg(image: np.ndarray):
    """
    Use rembg to estimate a foreground mask when no alpha channel exists.
    Returns a boolean mask.
    """
    global _rembg_session
    try:
        import rembg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "rembg is required to infer mask when no alpha channel is present."
        ) from exc
    print("*****inferring mask from rembg*****")

    pil_image = Image.fromarray(image)
    if pil_image.mode not in {"RGB", "RGBA"}:
        pil_image = pil_image.convert("RGB")

    if _rembg_session is None:
        _rembg_session = rembg.new_session("u2net")

    result = rembg.remove(pil_image, session=_rembg_session)
    result_np = np.array(result)
    if result_np.ndim == 2:
        alpha = result_np
    elif result_np.shape[-1] >= 4:
        alpha = result_np[..., 3]
    else:
        alpha = result_np[..., -1]
    return alpha > 0


def infer_mask_from_image(image: np.ndarray):
    """
    Prefer alpha-channel mask; fall back to rembg background removal.
    Ensures the returned mask is boolean (0/1) to avoid scaling bugs.
    """
    alpha_mask = mask_from_alpha_channel(image)
    if alpha_mask is not None:
        return alpha_mask
    try:
        return mask_from_rembg(image)
    except Exception as err:  # pragma: no cover - best-effort fallback
        print(f"rembg mask extraction failed ({err}); falling back to full mask.")
        return mask_to_all_true(image)


def mask_to_all_true(image):
    mask = np.ones_like(image)
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask

def load_single_mask(folder_path, index=0, extension=".png"):
    masks = load_masks(folder_path, [index], extension)
    return masks[0]


def load_masks(folder_path, indices_list=None, extension=".png"):
    masks = []
    indices_list = [] if indices_list is None else list(indices_list)
    if not len(indices_list) > 0:  # get all all masks if not provided
        idx = 0
        while os.path.exists(os.path.join(folder_path, f"{idx}{extension}")):
            indices_list.append(idx)
            idx += 1

    for idx in indices_list:
        mask_path = os.path.join(folder_path, f"{idx}{extension}")
        assert os.path.exists(mask_path), f"Mask path {mask_path} does not exist"
        mask = load_mask(mask_path)
        masks.append(mask)
    return masks


def display_image(image, masks=None):
    def imshow(image, ax):
        ax.axis("off")
        ax.imshow(image)

    grid = (1, 1) if masks is None else (2, 2)
    fig, axes = plt.subplots(*grid)
    if masks is not None:
        mask_colors = sns.color_palette("husl", len(masks))
        black_image = np.zeros_like(image[..., :3], dtype=float)  # background
        mask_display = np.copy(black_image)
        mask_union = np.zeros_like(image[..., :3])
        for i, mask in enumerate(masks):
            mask_display[mask] = mask_colors[i]
            mask_union |= mask[..., None] if mask.ndim == 2 else mask
        imshow(black_image, axes[0, 1])
        imshow(mask_display, axes[1, 0])
        imshow(image * mask_union, axes[1, 1])

    image_axe = axes if masks is None else axes[0, 0]
    imshow(image, image_axe)

    fig.tight_layout(pad=0)
    fig.show()


def interactive_visualizer(ply_path):
    with gr.Blocks() as demo:
        gr.Markdown("# 3D Gaussian Splatting (black-screen loading might take a while)")
        gr.Model3D(
            value=ply_path,  # splat file
            label="3D Scene",
        )
    demo.launch(share=True)
