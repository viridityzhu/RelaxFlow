# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Jiayin Zhu
import time
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, matrix_to_quaternion
from typing import Optional, Tuple, List, Union
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform, decompose_transform
from sam3d_objects.data.dataset.tdfy.pose_target import PoseTargetConverter
from loguru import logger
from sam3d_objects.pipeline.layout_post_optimization_utils import (
    run_ICP,
    compute_iou,
    set_seed,
    apply_transform,
    get_mesh,
    get_mask_renderer,
    run_alignment,
    run_render_compare,
    check_occlusion,
)


SLAT_STD = torch.tensor(
    [
        2.377650737762451,
        2.386378288269043,
        2.124418020248413,
        2.1748552322387695,
        2.663944721221924,
        2.371192216873169,
        2.6217446327209473,
        2.684523105621338,
    ]
)
SLAT_MEAN = torch.tensor(
    [
        -2.1687545776367188,
        -0.004347046371549368,
        -0.13352349400520325,
        -0.08418072760105133,
        -0.5271206498146057,
        0.7238689064979553,
        -1.1414450407028198,
        1.2039363384246826,
    ]
)

ROTATION_6D_MEAN = torch.tensor(
    [
        -0.06366084883674913,
        0.008438224692279752,
        0.00017084786438302483,
        0.0007126610473540038,
        -0.0030916726538816417,
        0.5166093753457688,
    ]
)
ROTATION_6D_STD = torch.tensor(
    [
        0.6656971967514863,
        0.6787012271867754,
        0.30345010594844524,
        0.4394504420678794,
        0.39817973931717104,
        0.6176286868761914,
    ]
)

def layout_post_optimization(
    Mesh,
    Quaternion,
    Translation,
    Scale,
    Mask,
    Point_Map,
    Intrinsics,
    Enable_shape_ICP=True,
    Enable_rendering_optimization=True,
    min_size=512,
    device=None,
):

    set_seed(100)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init transform and process mesh
    Rotation = quaternion_to_matrix(Quaternion.squeeze(1))
    center = Translation[0].clone()
    tfm_ori = compose_transform(scale=Scale, rotation=Rotation, translation=Translation)
    mesh, faces_idx, textures = get_mesh(Mesh, tfm_ori, device)

    # get mask and renderer
    mask, renderer = get_mask_renderer(Mask, min_size, Intrinsics, device)

    # check occlusion
    if check_occlusion(mask[0, 0].cpu().numpy(), Point_Map.cpu().numpy()):
        return (
            Quaternion,
            Translation,
            Scale,
            -1.0,
            False,
            False,
        )

    # Step 1: Manual Alignment
    source_points, target_points, center, tfm1, mesh, ori_iou, final_iou, flag_notgt = (
        run_alignment(
            Point_Map, mask, mesh, center, faces_idx, textures, renderer, device
        )
    )

    # return original layout if no target points. 
    if flag_notgt:
        return (
            Quaternion,
            Translation,
            Scale,
            -1.0,
            False,
            False,
        )

    # Step 2: Shape ICP
    if Enable_shape_ICP:
        Flag_ICP = True
        points_aligned_icp, transformation = run_ICP(
            mesh, source_points, target_points, threshold=0.05
        )
        mesh_ICP = Meshes(
            verts=[points_aligned_icp], faces=[faces_idx], textures=textures
        )
        rendered = renderer(mesh_ICP)
        ori_iou_shapeICP = compute_iou(
            rendered[..., 3][0][None, None], mask, threshold=0.5
        )
        # determine whether accept ICP
        if ori_iou_shapeICP > ori_iou:
            mesh = mesh_ICP
            final_iou = ori_iou_shapeICP.cpu().item()
            T_o3d = torch.tensor(transformation, dtype=torch.float32, device=device)
            T_o3d = T_o3d.T
            A = T_o3d[:3, :3]
            t = T_o3d[3, :3]
            scale = A.norm(dim=1)
            R = A / scale[:, None]
            center = ((center[None] * scale) @ R + t)[0]  # transform center
            tfm2 = (
                Transform3d(device=device)
                .scale(scale[None])
                .rotate(R[None])
                .translate(t[None])
            )
        else:
            Flag_ICP = False
            scale_2, translation_2 = torch.tensor(1).to(device), torch.zeros([3]).to(
                device
            )
            tfm2 = (
                Transform3d(device=device)
                .scale(scale_2.expand(3)[None])
                .translate(translation_2[None])
            )
    else:
        Flag_ICP = False
        scale_2, translation_2 = torch.tensor(1).to(device), torch.zeros([3]).to(device)
        tfm2 = (
            Transform3d(device=device)
            .scale(scale_2.expand(3)[None])
            .translate(translation_2[None])
        )

    # Step 3: Render-and-Compare
    if not Enable_rendering_optimization:
        Flag_optim = False
        tfm = tfm_ori.compose(tfm1).compose(tfm2)
    else:
        quat, translation, scale, R = run_render_compare(
            mesh, center, renderer, mask, device
        )
        with torch.no_grad():
            transformed = apply_transform(mesh, center, quat, translation, scale)
            rendered = renderer(transformed)
        optimized_iou = compute_iou(
            rendered[..., 3][0][None, None], mask, threshold=0.5
        )
        # Criterior to use layout optimization
        if optimized_iou < 0.5 or optimized_iou <= ori_iou:
            Flag_optim = False
            tfm = tfm_ori  # reject manual alignment and ICP as well.
            # tfm = tfm_ori.compose(tfm1).compose(tfm2)  # only reject render-compare but keep manual alignment and ICP.
        else:
            Flag_optim = True
            final_iou = optimized_iou.detach().cpu().item()
            tfm3 = (
                Transform3d(device=device)
                .translate(-center[None])  # move to center
                .scale(scale.expand(3)[None])
                .rotate(R.T[None])
                .translate(center[None])  # move back
                .translate(translation[None])
            )
            tfm = tfm_ori.compose(tfm1).compose(tfm2).compose(tfm3)

    M = tfm.get_matrix()[0]
    T_final = M[3, :3][None]
    A = M[:3, :3]
    scale_final = A.norm(dim=1)[None]
    R_final = A / scale_final[:, None]
    quat_final = matrix_to_quaternion(R_final)

    return (
        quat_final,
        T_final,
        scale_final,
        round(float(final_iou), 4),
        Flag_ICP,
        Flag_optim,
    )


def pose_decoder(
    pose_target_convention,
):
    def decode(model_output_dict, scene_scale=None, scene_shift=None):
        x = model_output_dict

        # BEGIN: copied from generative.py
        key_mapping = {
            "shape": "x_shape_latent",
            "quaternion": "x_instance_rotation",
            "6drotation": "x_instance_rotation_6d",
            "6drotation_normalized": "x_instance_rotation_6d_normalized",
            "translation": "x_instance_translation",
            "scale": "x_instance_scale",
            "translation_scale": "x_translation_scale",
        }

        # Decodes for metrics
        pose_target_dict = {}
        for k, v in x.items():
            pose_target_dict[key_mapping.get(k, k)] = v

        # TODO: Hao & Bowen please do clean this up!
        # Convert 6D rotation to quaternion if needed
        if (
            "x_instance_rotation_6d" in pose_target_dict
            or "x_instance_rotation_6d_normalized" in pose_target_dict
        ):
            # Extract the two 3D vectors
            if "x_instance_rotation_6d_normalized" in pose_target_dict:
                rot_6d = pose_target_dict[
                    "x_instance_rotation_6d_normalized"
                ] * ROTATION_6D_STD.to(
                    pose_target_dict["x_instance_rotation_6d_normalized"].device
                ) + ROTATION_6D_MEAN.to(
                    pose_target_dict["x_instance_rotation_6d_normalized"].device
                )
            else:
                rot_6d = pose_target_dict["x_instance_rotation_6d"]
            a1 = rot_6d[..., 0:3]
            a2 = rot_6d[..., 3:6]

            # Normalize first vector
            b1 = torch.nn.functional.normalize(a1, dim=-1)

            # Make second vector orthogonal to first
            b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
            b2 = torch.nn.functional.normalize(b2, dim=-1)

            # Compute third vector as cross product
            b3 = torch.cross(b1, b2, dim=-1)

            # Stack to create rotation matrix
            rotation_matrix = torch.stack([b1, b2, b3], dim=-1)

            # Convert to quaternion
            quaternion = matrix_to_quaternion(rotation_matrix)
            pose_target_dict["x_instance_rotation"] = quaternion

        if "x_instance_scale" in pose_target_dict:
            pose_target_dict["x_instance_scale"] = torch.exp(
                pose_target_dict["x_instance_scale"]
            )

        if "x_translation_scale" in pose_target_dict:
            pose_target_dict["x_translation_scale"] = torch.exp(
                pose_target_dict["x_translation_scale"]
            )

        pose_target_dict["pose_target_convention"] = [pose_target_convention] * x[
            "shape"
        ].shape[0]
        # END: copied from generative.py

        # Fake pointmap moments
        device = x["shape"].device
        _scene_scale = (
            scene_scale if scene_scale is not None else torch.tensor(1.0, device=device)
        )
        _scene_shift = (
            scene_shift
            if scene_shift is not None
            else torch.tensor([[0, 0, 0]], device=device)
        )
        pose_target_dict["x_scene_scale"] = _scene_scale
        pose_target_dict["x_scene_center"] = _scene_shift

        # Convert to instance pose
        pose_instance_dict = PoseTargetConverter.dicts_pose_target_to_instance_pose(
            pose_target_convention=pose_target_convention,
            x_instance_scale=pose_target_dict["x_instance_scale"],
            x_instance_translation=pose_target_dict["x_instance_translation"],
            x_instance_rotation=pose_target_dict["x_instance_rotation"],
            x_translation_scale=pose_target_dict["x_translation_scale"],
            x_scene_scale=pose_target_dict["x_scene_scale"],
            x_scene_center=pose_target_dict["x_scene_center"],
        )
        return {
            "translation": pose_instance_dict["instance_position_l2c"].squeeze(0),
            "rotation": pose_instance_dict["instance_quaternion_l2c"].squeeze(0),
            "scale": pose_instance_dict["instance_scale_l2c"].squeeze(0).mean(-1, keepdim=True).expand(1,3),
        }

    return decode

def zero_prediction_decoder():
    def decode(model_output_dict, scene_scale=None, scene_shift=None):
        import copy
        from loguru import logger
        _pose_decoder = pose_decoder("ScaleShiftInvariant")
        model_output_dict = copy.deepcopy(model_output_dict)
        logger.warning("Overwriting predictions to zero prediction")
        model_output_dict["translation"] = torch.zeros_like(model_output_dict["translation"])
        model_output_dict["translation_scale"] = torch.zeros_like(model_output_dict["translation_scale"])
        model_output_dict["scale"] = torch.zeros_like(model_output_dict["scale"]) + 1.337 # Empirical average on R3
        return _pose_decoder(model_output_dict, scene_scale, scene_shift)

    return decode


def get_default_pose_decoder():
    def decode(model_output_dict, **kwargs):
        return {}

    return decode


POSE_DECODERS = {
    "default": get_default_pose_decoder(),
    "ApparentSize": pose_decoder("ApparentSize"),
    "DisparitySpace": pose_decoder("DisparitySpace"),
    "ScaleShiftInvariant": pose_decoder("ScaleShiftInvariant"),
    "ZeroPredictionScaleShiftInvariant": zero_prediction_decoder(),
}


def get_pose_decoder(name):
    if name not in POSE_DECODERS:
        raise NotImplementedError

    return POSE_DECODERS[name]

def _gaussian_blur_binary_image(mask_2d: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma is None or sigma <= 0 or mask_2d.numel() == 0:
        return mask_2d
    kernel_size = int(2 * round(3 * float(sigma)) + 1)
    kernel_size = max(kernel_size, 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=mask_2d.device, dtype=mask_2d.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp(min=1e-6)
    kernel_2d = torch.matmul(kernel_1d[:, None], kernel_1d[None, :])
    kernel_2d = kernel_2d / kernel_2d.sum().clamp(min=1e-6)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    mask_2d = mask_2d.view(1, 1, *mask_2d.shape)
    blurred = F.conv2d(mask_2d, kernel_2d, padding=padding)
    return blurred.view(*mask_2d.shape[-2:])


def _ensure_batch_dim(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{name} cannot be None when computing visibility mask.")
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _transform_points_l2c(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scale_vec = scale.view(-1)
    if scale_vec.numel() == 1:
        scale_vec = scale_vec.expand(3)
    elif scale_vec.numel() != 3:
        scale_vec = scale_vec[:3]
    rot = rotation
    if rot.shape[-1] == 4:
        rot_mat = quaternion_to_matrix(rot.unsqueeze(0))[0]
    else:
        rot_mat = rot.view(3, 3)
    tfm = compose_transform(
        scale=scale_vec.view(1, 3),
        rotation=rot_mat.view(1, 3, 3),
        translation=translation.view(1, 3),
    )
    return tfm.transform_points(points.unsqueeze(0))[0]


def compute_geometry_visibility_mask(
    coords: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: Tuple[int, int],
    condition_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    # ------
    soft_falloff: float = 3.0,
    param_tolerance_scale: float = 1.5,
    param_dilate_scale: float = 1.5,
    # -------------------
    return_debug: bool = False,
) -> Optional[torch.Tensor]:
    """
    Build a visibility mask for structured latent tokens using Stage-1 geometry and camera pose.
    Author: Jiayin Zhu
    """
    if coords is None or coords.numel() == 0:
        return None
    if intrinsics is None:
        return None

    H, W = image_shape
    coords_cpu = coords.detach().cpu()
    translation_cpu = _ensure_batch_dim(translation.detach().cpu(), "translation")
    rotation_cpu = _ensure_batch_dim(rotation.detach().cpu(), "rotation")
    scale_cpu = _ensure_batch_dim(scale.detach().cpu(), "scale")

    intrinsics_cpu = intrinsics
    if not torch.is_tensor(intrinsics_cpu):
        intrinsics_cpu = torch.from_numpy(np.array(intrinsics_cpu))
    intrinsics_cpu = intrinsics_cpu.detach().cpu()
    if intrinsics_cpu.dim() == 2:
        intrinsics_cpu = intrinsics_cpu.unsqueeze(0)
    if intrinsics_cpu.shape[0] == 1 and translation_cpu.shape[0] > 1:
        intrinsics_cpu = intrinsics_cpu.expand(translation_cpu.shape[0], -1, -1)

    batch_size = translation_cpu.shape[0]
    total_tokens = coords_cpu.shape[0]
    mask_cpu = torch.zeros((batch_size, total_tokens, 1), dtype=torch.float32)
    raw_mask_cpu = torch.zeros_like(mask_cpu)

    pixel_masks_raw: List[torch.Tensor] = [] if return_debug else None
    pixel_masks_blurred: List[torch.Tensor] = [] if return_debug else None

    cond_mask_cpu = None
    if condition_mask is not None:
        cond_mask_cpu = condition_mask
        if not torch.is_tensor(cond_mask_cpu):
            cond_mask_cpu = torch.from_numpy(np.array(cond_mask_cpu))
        cond_mask_cpu = cond_mask_cpu.detach().cpu().float()
        if cond_mask_cpu.dim() == 4:
            if cond_mask_cpu.shape[1] in (1, 3, 4):
                cond_mask_cpu = cond_mask_cpu[:, -1]
            elif cond_mask_cpu.shape[-1] in (1, 3, 4):
                cond_mask_cpu = cond_mask_cpu[..., -1]
            else:
                cond_mask_cpu = cond_mask_cpu[:, 0]
        elif cond_mask_cpu.dim() == 3:
            if cond_mask_cpu.shape[-1] in (1, 3, 4) and cond_mask_cpu.shape[0] != batch_size:
                cond_mask_cpu = cond_mask_cpu[..., -1]
            elif cond_mask_cpu.shape[0] == 1 and cond_mask_cpu.shape[1] == H and cond_mask_cpu.shape[2] == W:
                cond_mask_cpu = cond_mask_cpu.squeeze(0)
        if cond_mask_cpu.dim() == 2:
            cond_mask_cpu = cond_mask_cpu.unsqueeze(0)
        if cond_mask_cpu.numel() > 0 and float(cond_mask_cpu.max()) > 1.0:
            cond_mask_cpu = cond_mask_cpu / 255.0
        cond_mask_cpu = cond_mask_cpu.clamp(0.0, 1.0)
        if cond_mask_cpu.shape[-2:] != (H, W):
            cond_mask_cpu = F.interpolate(
                cond_mask_cpu.unsqueeze(1),
                size=(H, W),
                mode="nearest",
            ).squeeze(1)
        if cond_mask_cpu.shape[0] == 1 and batch_size > 1:
            cond_mask_cpu = cond_mask_cpu.expand(batch_size, -1, -1)
        elif cond_mask_cpu.shape[0] != batch_size:
            cond_mask_cpu = cond_mask_cpu[:batch_size]

    coord_batches = coords_cpu[:, 0].long()
    coords_xyz = coords_cpu[:, 1:].to(torch.float32)

    for b in range(batch_size):
        sel_idx = torch.nonzero(coord_batches == b, as_tuple=False).squeeze(1)
        if sel_idx.numel() == 0:
            if return_debug:
                zero_img = torch.zeros((H, W), dtype=torch.float32)
                pixel_masks_raw.append(zero_img)
                pixel_masks_blurred.append(zero_img.clone())
            continue

        # --- 1. Adaptive Parameter Calculation ---
        obj_scale = scale_cpu[b].max().item() # Assume scale is uniform, take max for defense
        obj_z = translation_cpu[b, 2].item()
        if obj_z < 1e-3: obj_z = 1.0 # Prevent division by zero

        K = intrinsics_cpu[b if intrinsics_cpu.shape[0] > 1 else 0]
        fx, fy = K[0, 0].item(), K[1, 1].item()
        
        fx_pixel = fx if fx > 10 else fx * W
        fy_pixel = fy if fy > 10 else fy * H
        f_avg = (fx_pixel + fy_pixel) / 2.0

        # Calculate Depth Tolerance (3D distance)
        # Physical meaning: size of one Voxel * scale
        voxel_size_3d = obj_scale / 64.0
        depth_tolerance = param_tolerance_scale * voxel_size_3d

        # Calculate Dilate Kernel (2D pixels)
        # Physical meaning: size of Voxel projected to screen * scale
        # Formula: Size_2D = Size_3D * (f / z)
        projected_voxel_size = voxel_size_3d * (f_avg / obj_z)
        
        # Automatically calculate kernel, must be odd
        raw_kernel = param_dilate_scale * projected_voxel_size
        kernel_size = int(round(raw_kernel))
        if kernel_size < 1: kernel_size = 1
        if kernel_size % 2 == 0: kernel_size += 1
        

        # ---------------------------------------------------

        local_coords = coords_xyz[sel_idx]
        local_points = local_coords / 64.0 - 0.5

        # Transform to camera coordinate system
        cam_points = _transform_points_l2c(
            local_points,
            rotation_cpu[b],
            translation_cpu[b],
            scale_cpu[b],
        )

        z = cam_points[:, 2]
        valid_mask = z > 1e-4
        if not valid_mask.any():
            if return_debug:
                zero_img = torch.zeros((H, W), dtype=torch.float32)
                pixel_masks_raw.append(zero_img)
                pixel_masks_blurred.append(zero_img.clone())
            continue

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        cam_valid = cam_points[valid_mask]
        z_valid = cam_valid[:, 2]

        K = intrinsics_cpu[b if intrinsics_cpu.shape[0] > 1 else 0]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        fx_val = float(fx)
        fy_val = float(fy)
        cx_val = float(cx)
        cy_val = float(cy)
        is_normalized = (
            0.0 <= cx_val <= 1.5
            and 0.0 <= cy_val <= 1.5
            and fx_val < 10
            and fy_val < 10
        )
        if is_normalized:
            fx = fx * W
            fy = fy * H
            cx = cx * W
            cy = cy * H

        # PyTorch3D camera convention: x-left, y-up, z-in
        # Image convention: x-right, y-down => flip x and y.
        px = cx - fx * (cam_valid[:, 0] / z_valid)
        py = cy - fy * (cam_valid[:, 1] / z_valid)

        # Frustum Culling (inside the image)
        inside = (px >= 0) & (px <= (W - 1)) & (py >= 0) & (py <= (H - 1))
        if not inside.any():
            if return_debug:
                zero_img = torch.zeros((H, W), dtype=torch.float32)
                pixel_masks_raw.append(zero_img)
                pixel_masks_blurred.append(zero_img.clone())
            continue

        px = torch.floor(px[inside]).long().clamp(0, W - 1)
        py = torch.floor(py[inside]).long().clamp(0, H - 1)
        depth = z_valid[inside]

        flat_idx = py * W + px
        
        # 1. Build depth buffer (Z-Buffer)
        # Here we calculate the depth of the "surface" of the pixel
        depth_buffer = torch.full((H * W,), float("inf"), dtype=depth.dtype)

        if hasattr(torch, "scatter_reduce"):
            depth_buffer = torch.scatter_reduce(
                depth_buffer, 0, flat_idx, depth, reduce="amin", include_self=True
            )
        else:
            depth_np = depth_buffer.numpy()
            np.minimum.at(depth_np, flat_idx.numpy(), depth.numpy())
            depth_buffer = torch.from_numpy(depth_np)

        depth_map = depth_buffer.view(H, W)
        
        # 2. Dilate depth map (Dilate Surface)
        # Because the point cloud is sparse, we need to slightly expand the surface on the pixel plane
        # This way, even if a point projects on a neighboring pixel, the correct surface depth can be found, preventing penetration
        if kernel_size > 1:
            pad = kernel_size // 2
            # MinPool via MaxPool(-x)
            depth_dense = -F.max_pool2d(
                -depth_map.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size, stride=1, padding=pad
            ).squeeze()
        else:
            depth_dense = depth_map

        # 3. Core: Calculate Soft Visibility based on depth
        
        nearest_surface_depth = depth_dense.view(-1)[flat_idx]
        
        # Calculate the distance between the current point and the surface
        diff = depth - nearest_surface_depth
        
        sigma = depth_tolerance if depth_tolerance > 1e-6 else 1e-6
        
        # Gaussian Decay: exp( - (d/sigma)^2 )
        exponent = - soft_falloff * (diff.clamp(min=0) / sigma).pow(2)
        soft_weight = torch.exp(exponent) 

        
        # nearest = depth_dense.view(-1)[flat_idx]
        # # ### core computation of visible mask ### #
        # visible = depth <= (nearest + depth_tolerance)

        total_points_in_batch = sel_idx.numel()
        
        # 1. Points remaining after Frustum Culling (inside the image)
        points_in_frustum = inside.sum().item()
        
        # 2. Points remaining after Occlusion Culling (final visible)
        points_visible = (soft_weight > 0.1).float().sum().item()
        
        if total_points_in_batch > 0:
            frustum_kept_ratio = (points_in_frustum / total_points_in_batch) * 100
            occlusion_kept_ratio = 0.0
            if points_in_frustum > 0:
                occlusion_kept_ratio = (points_visible / points_in_frustum) * 100
            final_ratio = (points_visible / total_points_in_batch) * 100
            
        
        if cond_mask_cpu is not None:
            cond_vals = cond_mask_cpu[b].view(-1)[flat_idx]
            soft_weight = soft_weight * cond_vals
            
        inside_local_idx = valid_indices[inside]
        
        local_mask = torch.zeros(sel_idx.shape[0], dtype=torch.float32)
        local_mask[inside_local_idx] = soft_weight
        
        local_binary = torch.zeros(sel_idx.shape[0], dtype=torch.float32)
        local_binary[inside_local_idx] = (soft_weight > 0.1).float() # Threshold can be adjusted

        mask_cpu[b, sel_idx, 0] = local_mask
        raw_mask_cpu[b, sel_idx, 0] = local_binary
        
        if return_debug:
            vis_kernel_size = 5 # Dilate kernel size
            padding = vis_kernel_size // 2

            # --- 1. Visualize Soft Weights (Heatmap) ---
            
            heatmap_raw = torch.zeros((H * W,), dtype=torch.float32, device=coords.device)
            existence_raw = torch.zeros((H * W,), dtype=torch.float32, device=coords.device) # Record where there is a point
            
            if hasattr(torch, "scatter_reduce"):
                heatmap_raw = torch.scatter_reduce(
                    heatmap_raw, 0, flat_idx.to(coords.device), soft_weight.to(coords.device), reduce="amax", include_self=True
                )
                existence_raw = torch.scatter_reduce(
                    existence_raw, 0, flat_idx.to(coords.device), torch.ones_like(soft_weight).to(coords.device), reduce="amax", include_self=True
                )
            else:
                hm_np = heatmap_raw.cpu().numpy()
                ex_np = existence_raw.cpu().numpy()
                np.maximum.at(hm_np, flat_idx.cpu().numpy(), soft_weight.detach().cpu().numpy())
                np.maximum.at(ex_np, flat_idx.cpu().numpy(), 1.0)
                heatmap_raw = torch.from_numpy(hm_np).to(coords.device)
                existence_raw = torch.from_numpy(ex_np).to(coords.device)

            heatmap_2d = heatmap_raw.view(H, W)
            existence_2d = existence_raw.view(H, W)

            heatmap_dilated = F.max_pool2d(
                heatmap_2d.unsqueeze(0).unsqueeze(0), 
                kernel_size=vis_kernel_size, stride=1, padding=padding
            ).squeeze()
            
            mask_dilated = F.max_pool2d(
                existence_2d.unsqueeze(0).unsqueeze(0),
                kernel_size=vis_kernel_size, stride=1, padding=padding
            ).squeeze()

            heatmap_rgb = _apply_colormap(heatmap_dilated, mask=(mask_dilated > 0).float())
            pixel_masks_raw.append(heatmap_rgb.cpu())

            valid_depth_mask = depth_dense < float('inf')
            
            if valid_depth_mask.any():
                d_min = depth_dense[valid_depth_mask].min()
                d_max = depth_dense[valid_depth_mask].max()
                
                depth_norm = torch.zeros_like(depth_dense)
                if d_max - d_min > 1e-4:
                    depth_norm[valid_depth_mask] = 1.0 - (depth_dense[valid_depth_mask] - d_min) / (d_max - d_min)
                else:
                    depth_norm[valid_depth_mask] = 1.0 
                
                depth_vis = F.max_pool2d(
                    depth_norm.unsqueeze(0).unsqueeze(0),
                    kernel_size=vis_kernel_size, stride=1, padding=padding
                ).squeeze()
                
                depth_mask = (depth_vis > 0).float() # Simplified processing, non-0 after normalization is valid
                
                depth_rgb = _apply_colormap(depth_vis, mask=depth_mask)
            else:
                depth_rgb = torch.zeros((3, H, W), dtype=torch.float32, device=coords.device)

            pixel_masks_blurred.append(depth_rgb.cpu())



    mask = mask_cpu.to(coords.device)
    if not return_debug:
        return mask

    debug_payload = {
        "token_mask_binary": raw_mask_cpu,
        "pixel_masks_raw": pixel_masks_raw, # Now Heatmap
        "pixel_masks_blurred": pixel_masks_blurred, # Now Depth Map
    }
    return mask, debug_payload


def prune_sparse_structure(
    coord_batch,
    max_neighbor_axes_dist=1,
):
    # Guard: empty sparse structure (N=0) can happen for degenerate/failed branches.
    # Downstream code expects a (N,4) tensor; keep it empty rather than crashing.
    if coord_batch is None or coord_batch.numel() == 0 or coord_batch.shape[0] == 0:
        return coord_batch
    coords, batch = coord_batch[:, 1:], coord_batch[:, 0].unsqueeze(-1)
    device = coords.device
    # 1) shift coords so minimum is zero
    min_xyz = coords.min(0)[0]
    coords0 = coords - min_xyz
    # 2) build occupancy grid
    max_xyz = coords0.max(0)[0] + 1  # size in each dim
    D, H, W = max_xyz.tolist()
    # shape (1,1,D,H,W)
    occ = torch.zeros((1, 1, D, H, W), dtype=torch.uint8, device=device)
    x, y, z = coords0.unbind(1)
    occ[0, 0, x, y, z] = 1
    # 3) 3×3×3 convolution to count each voxel + neighbors
    kernel = torch.ones(
        (
            1,
            1,
            2 * max_neighbor_axes_dist + 1,
            2 * max_neighbor_axes_dist + 1,
            2 * max_neighbor_axes_dist + 1,
        ),
        dtype=torch.uint8,
        device=device,
    )
    # pad so output is same size
    pad = max_neighbor_axes_dist
    counts = torch.nn.functional.conv3d(occ.float(), kernel.float(), padding=pad)
    # interior voxels have count == (2*max_neighbor_axes_dist+1)**3
    full_count = (2 * max_neighbor_axes_dist + 1) ** 3
    # 4) lookup counts at each original coord
    counts_at_pts = counts[0, 0, x, y, z]  # (N,)
    is_surface = counts_at_pts < full_count
    # 5) return filtered batch+coords (shift back if you want original coords)
    kept = is_surface.nonzero(as_tuple=False).squeeze(1)
    out_batch = batch[kept]
    out_coords = coords[kept]
    coords = torch.cat([out_batch, out_coords], dim=1)

    return torch.cat([out_batch, out_coords], dim=1)

def _apply_colormap(img_2d: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Blue (Low) -> Red (High) linear mapping.
    Background (Mask=0) is black.
    """
    x = img_2d.clamp(0, 1)

    # Simple linear interpolation:
    # Red:   0.0 -> 1.0
    # Green: 0.0 (keep as 0, remove green interference)
    # Blue:  1.0 -> 0.0
    
    r = x
    g = torch.zeros_like(x)
    b = 1.0 - x

    rgb = torch.stack([r, g, b], dim=0) # (3, H, W)

    # Process background: must pass in mask, otherwise x=0 (invisible point) and x=0 (no data background) will be confused
    if mask is not None:
        rgb = rgb * mask.unsqueeze(0)
    else:
        # If no mask is passed, it is assumed that only values greater than 0 are data
        # But this will cause the point with weight=0 to turn black instead of blue
        rgb = rgb * (x > 0).float().unsqueeze(0)

    return rgb.permute(1, 2, 0) # (H, W, 3)

def downsample_sparse_structure(
    coord_batch,
    max_coords=42000,
    downsample_factor=2,
):
    """
    Downsample sparse structure coordinates when there are more than max_coords.

    Downsamples by rescaling coordinates, effectively shrinking the grid while preserving
    the structure. The downsampled grid is centered in the original space.

    Args:
        coord_batch: tensor of shape (N, 4) where [:, 0] is batch index and [:, 1:] are coords
        max_coords: maximum number of coordinates to keep
            42000 should be safe number. Calculation: max(int32) / (64*768) ~= 43691
            Only needed for mesh decoding.
        downsample_factor: factor by which to downsample (e.g., 2 means half resolution)

    Returns:
        Downsampled coord_batch with coordinates rescaled if downsampling is needed
    """
    if coord_batch.shape[0] <= max_coords:
        return coord_batch, 1

    # Extract coordinates and batch indices
    coords = coord_batch[:, 1:].float()  # Shape: (N, 3), convert to float for scaling
    batch_indices = coord_batch[:, 0:1]  # Shape: (N, 1)

    # Find the actual coordinate bounds
    coords_min = coords.min(dim=0)[0]  # Shape: (3,)
    coords_max = coords.max(dim=0)[0]  # Shape: (3,)
    original_size = coords_max - coords_min + 1  # Add 1 since coordinates are discrete

    # Calculate target size after downsampling
    target_size = original_size / downsample_factor

    # Calculate the offset to center the downsampled grid
    offset = (original_size - target_size) / 2
    target_min = coords_min + offset
    target_max = coords_min + offset + target_size - 1

    # Normalize coordinates to [0, 1] within their actual range
    coords_normalized = (coords - coords_min) / (coords_max - coords_min)

    # Scale to the target range
    coords_rescaled = coords_normalized * (target_size - 1) + target_min

    # Round to integers to get discrete grid coordinates
    coords_rescaled = torch.round(coords_rescaled).int()

    # Clamp to ensure we stay within bounds
    coords_rescaled = torch.clamp(coords_rescaled, target_min.int(), target_max.int())

    # Remove duplicates that may have been created by the downsampling
    # Concatenate batch and coords for duplicate removal
    combined = torch.cat([batch_indices, coords_rescaled], dim=1)
    unique_combined = torch.unique(combined, dim=0)

    # If still too many after deduplication, randomly subsample
    if unique_combined.shape[0] > max_coords:
        indices = torch.randperm(unique_combined.shape[0], device=coord_batch.device)[
            :max_coords
        ]
        unique_combined = unique_combined[indices]

    return unique_combined.int(), downsample_factor


def normalize_mesh_verts(verts):
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = (vmax + vmin) / 2.0
    extent = vmax - vmin  # largest side length
    max_extent = np.max(extent)
    if max_extent == 0:
        vertices = verts - center
        scale = 1
    else:
        scale = 1.0 / max_extent
        vertices = (verts - center) * scale
    return vertices, scale, center


def voxelize_mesh(mesh, resolution=64):
    verts = np.asarray(mesh.vertices)
    # rotate mesh (from z-up to y-up)
    verts = verts @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    # normalize vertices
    # skip vertices to avoid losing points, likely already normalized
    if np.abs(verts.min() + 0.5) < 1e-3 and np.abs(verts.max() - 0.5) < 1e-3:
        vertices, scale, center = verts, None, None
    else:
        vertices, scale, center = normalize_mesh_verts(verts)

    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    vertices = (vertices + 0.5) / 64 - 0.5
    coords = ((torch.tensor(vertices) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss, scale, center


def preprocess_mesh(mesh: trimesh.Trimesh):
    verts = mesh.vertices
    if np.abs(verts.min() + 0.5) < 1e-3 and np.abs(verts.max() - 0.5) < 1e-3:
        return mesh
    vertices, _, _ = normalize_mesh_verts(verts)
    mesh.vertices = vertices
    return mesh


def trimesh2o3d_mesh(trimesh_mesh):
    verts = np.asarray(trimesh_mesh.vertices)
    faces = np.asarray(trimesh_mesh.faces)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
    )


def update_layout(pred_t, pred_s, pred_quat, center, scale, to_halo=True):
    if center is None and not to_halo:
        return pred_t, pred_s, pred_quat
    pred_transform = compose_transform(
        pred_s, quaternion_to_matrix(pred_quat[0]), pred_t
    )
    if center is None:
        comb_transform = pred_transform
    else:
        norm_transform = compose_transform(
            scale * torch.ones_like(pred_t),
            torch.eye(3, dtype=pred_t.dtype).to(pred_t.device)[None],
            scale * -torch.tensor(center, dtype=pred_t.dtype).to(pred_t.device)[None],
        )
        comb_transform = norm_transform.compose(pred_transform)
    comb_transform = convert_to_halo(comb_transform, pred_t.device, pred_t.dtype)
    decomposed = decompose_transform(comb_transform)
    quat = matrix_to_quaternion(decomposed.rotation)
    return decomposed.translation, decomposed.scale, quat


def convert_to_halo(pred_transform, device, dtype):
    on_mesh_transform = Transform3d(dtype=dtype, device=device).rotate(
        torch.tensor(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ],
            dtype=dtype,
        )
    )
    on_pm_transform = Transform3d(dtype=dtype, device=device).rotate(
        torch.tensor(
            [
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ],
            dtype=dtype,
        )
    )
    return on_mesh_transform.compose(pred_transform).compose(on_pm_transform)


def quat_wxyz_to_euler_XYZ(q: torch.Tensor) -> torch.Tensor:
    """
    Convert PyTorch3D quaternions (w,x,y,z) to SciPy-style Euler angles
    with sequence 'XYZ' (extrinsic, radians). Works with batch dims.

    Args:
        q: (..., 4) tensor in w,x,y,z order. Doesn't need to be normalized.
    Returns:
        angles: (..., 3) tensor [alpha_X, beta_Y, gamma_Z] in radians.
    """
    q = q / q.norm(dim=-1, keepdim=True)  # normalize
    R = quaternion_to_matrix(q)  # (..., 3, 3)
    R = R.transpose(-1, -2)

    r00 = R[..., 0, 0]
    r10 = R[..., 1, 0]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    # For extrinsic XYZ (R = Rz(gamma) @ Ry(beta) @ Rx(alpha)):
    # beta = atan2(-r20, sqrt(r00^2 + r10^2))
    # alpha = atan2(r21, r22)
    # gamma = atan2(r10, r00)
    eps = torch.finfo(R.dtype).eps
    beta = torch.atan2(-r20, torch.clamp((r00 * r00 + r10 * r10).sqrt(), min=eps))
    alpha = torch.atan2(r21, r22)
    gamma = torch.atan2(r10, r00)

    return -torch.stack((alpha, beta, gamma), dim=-1)


def format_to_halo(layout_output):
    json_out = {}
    quaternion = layout_output["quaternion"][0, 0]
    translation = layout_output["translation"][0]
    scale = list(layout_output["scale"][0])

    euler = quat_wxyz_to_euler_XYZ(quaternion)
    json_out["roll"] = float(euler[0])
    json_out["pitch"] = float(euler[1])
    json_out["yaw"] = float(euler[2])
    json_out["pred_scale"] = [float(s) for s in scale]
    rot_matrix = quaternion_to_matrix(quaternion)
    pred_transform = torch.eye(4, dtype=quaternion.dtype).to(quaternion.device)
    pred_transform[:3, :3] = rot_matrix
    pred_transform[:3, 3] = translation
    pred_transform_list = [
        [float(t) for t in trans_row] for trans_row in pred_transform
    ]
    json_out["pred_transform"] = pred_transform_list
    return json_out


def json_to_halo_payloads(target_data):
    pred_transform = target_data["pred_transform"]
    pred_scale = target_data["pred_scale"]
    roll = target_data.get("roll", 0)
    pitch = target_data.get("pitch", 0)
    yaw = target_data.get("yaw", 0)
    # Update positions, rotation, and scale in the payload
    item_attachments = {}
    item_attachments["positions"] = {
        "x": pred_transform[0][3],
        "y": pred_transform[1][3],
        "z": pred_transform[2][3] - 1,  # Adjust for Halo design
    }
    item_attachments["rotation"] = {"x": roll, "y": pitch, "z": yaw}
    item_attachments["scale"] = {
        "x": pred_scale[0],
        "y": pred_scale[1],
        "z": pred_scale[2],
    }
    return item_attachments


def o3d_plane_estimation(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(0.02, 3, 1000)

    [a, b, c, d] = plane_model
    logger.info(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Get the inlier points from RANSAC
    inlier_points = np.asarray(pcd.points)[inliers]

    # Adaptive flying point removal based on Z-range
    z_range = np.max(inlier_points[:, 2]) - np.min(inlier_points[:, 2])
    if z_range > 6.0:       # Large range - likely flying points
        thresh = 0.90       # Remove 10%
    elif z_range > 2.0:     # Moderate range
        thresh = 0.93       # Remove 7%
    else:                   # Small range - clean
        thresh = 0.95       # Remove 5%

    depth_quantile = np.quantile(inlier_points[:, 2], thresh)
    clean_points = inlier_points[inlier_points[:, 2] <= depth_quantile]

    logger.info(f"Flying point removal: {len(inlier_points)} -> {len(clean_points)} points (z_range: {z_range:.2f}m, thresh: {thresh})")
    logger.info(f"Clean points Z range: [{clean_points[:, 2].min():.3f}, {clean_points[:, 2].max():.3f}]")

    # Get the normal vector of the plane
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Create two orthogonal vectors in the plane using camera-aware approach
    # Use Z-axis as primary tangent (depth direction in camera coords)
    # This helps align one plane axis with the camera's depth direction
    if abs(normal[2]) < 0.9:  # Use Z-axis if normal isn't too close to Z
        tangent = np.array([0, 0, 1])
    else:
        tangent = np.array([1, 0, 0])  # Use X-axis otherwise

    v1 = np.cross(normal, tangent)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)  # Explicit normalization for numerical stability

    # Ensure consistent right-handed coordinate system
    if np.dot(np.cross(v1, v2), normal) < 0:
        v2 = -v2

    logger.info(f"Plane basis vectors - v1: [{v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f}], v2: [{v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f}]")

    # Calculate centroid using bounding box center (more robust to density bias)
    min_vals = np.min(clean_points, axis=0)
    max_vals = np.max(clean_points, axis=0)
    centroid = (min_vals + max_vals) / 2
    logger.info(f"Bbox centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

    # Project clean points onto the plane's coordinate system
    relative_points = clean_points - centroid
    u_coords = np.dot(relative_points, v1)  # coordinates along v1 direction
    v_coords = np.dot(relative_points, v2)  # coordinates along v2 direction

    # Since flying points are already removed, use minimal percentile filtering [0, 99]
    u_min, u_max = np.percentile(u_coords, [0, 100])
    v_min, v_max = np.percentile(v_coords, [0, 100])

    # Calculate extents
    u_extent = u_max - u_min
    v_extent = v_max - v_min

    # Ensure minimum size
    u_extent = max(u_extent, 0.1)  # minimum 10cm
    v_extent = max(v_extent, 0.1)
    logger.info(f"Plane size: {u_extent:.3f}m x {v_extent:.3f}m")

    # Calculate direction away from camera center (at origin [0,0,0])
    camera_pos = np.array([0, 0, 0])  # Camera at origin
    camera_to_centroid = centroid - camera_pos  # Direction from camera to plane center
    camera_distance = np.linalg.norm(camera_to_centroid)
    away_direction = camera_to_centroid / camera_distance

    # Project away direction onto the plane (remove component normal to plane)
    away_in_plane = away_direction - np.dot(away_direction, normal) * normal
    away_in_plane_norm = np.linalg.norm(away_in_plane)

    # Create plane coordinate system based on camera direction
    if away_in_plane_norm > 1e-6:  # Only if there's a meaningful in-plane component
        # Define plane axes directly based on camera direction
        away_axis = away_in_plane / away_in_plane_norm  # Away from camera direction (in plane)
        perp_axis = np.cross(normal, away_axis)  # Perpendicular to away direction (in plane)
        perp_axis = perp_axis / np.linalg.norm(perp_axis)

        logger.info(f"Camera-based plane axes:")
        logger.info(f"  Away axis: [{away_axis[0]:.3f}, {away_axis[1]:.3f}, {away_axis[2]:.3f}]")
        logger.info(f"  Perp axis: [{perp_axis[0]:.3f}, {perp_axis[1]:.3f}, {perp_axis[2]:.3f}]")

        # Project all points onto this camera-aligned coordinate system
        relative_points = clean_points - centroid
        away_coords = np.dot(relative_points, away_axis)  # coordinates along away direction
        perp_coords = np.dot(relative_points, perp_axis)  # coordinates perpendicular to away

        # Calculate extents in camera-aligned system
        away_min, away_max = np.percentile(away_coords, [0, 100])
        perp_min, perp_max = np.percentile(perp_coords, [0, 100])

        away_extent = max(away_max - away_min, 0.1)
        perp_extent = max(perp_max - perp_min, 0.1)

        # Asymmetric extension: 10% towards camera, 50% away from camera, 20% perpendicular both sides
        away_extent_extended = away_extent * 1.6  # 60% larger in away direction (10% + 50%)
        perp_extent_extended = perp_extent * 1.4  # 40% larger in perpendicular direction (20% each side)

        logger.info(f"Original extents: away={away_extent:.3f}m, perp={perp_extent:.3f}m")
        logger.info(f"Extended extents: away={away_extent_extended:.3f}m, perp={perp_extent_extended:.3f}m")

        # Extension amounts for each direction
        away_extension_near = away_extent * 0.1   # 10% extension towards camera (near side)
        away_extension_far = away_extent * 0.5    # 50% extension away from camera (far side)
        perp_extension = perp_extent * 0.2        # 20% extension on each perpendicular side

        logger.info(f"Extensions: near={away_extension_near:.3f}m, far={away_extension_far:.3f}m, perp={perp_extension:.3f}m per side")
        logger.info(f"Extending plane asymmetrically: 10% towards camera, 50% away from camera, 20% perpendicular both sides")

        corners = []
        for da in [-1, 1]:
            for dp in [-1, 1]:
                # Asymmetric extension in away direction
                if da == 1:  # Away from camera side - extend by 50%
                    away_distance = away_extent/2 + away_extension_far
                else:  # Near camera side - extend by 10%
                    away_distance = da * (away_extent/2 + away_extension_near)

                # Extend perpendicular direction by 20% on both sides
                perp_distance = dp * (perp_extent/2 + perp_extension)

                corner = (centroid +
                         away_distance * away_axis +
                         perp_distance * perp_axis)
                corners.append(corner)
    else:
        # If plane is parallel to camera direction, use original v1/v2 system
        logger.info("Plane parallel to camera direction, using original coordinate system")
        corners = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                corner = centroid + dx * (u_extent/2) * v1 + dy * (v_extent/2) * v2
                corners.append(corner)
    corners = np.array(corners)
    # Create a quad mesh using trimesh
    # Define vertices (4 corners)
    vertices = corners
    # Define a single quad face (indices of the 4 vertices)
    # Make sure the order is correct for proper orientation
    faces = np.array([[0, 1, 3, 2]])  # quad face
    # Create trimesh with quad faces

    # rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False  # Important: prevents automatic triangulation
    )
    # Optional: set face colors
    mesh.visual.face_colors = [128, 128, 128, 255]  # gray color (RGBA)

    return mesh


def estimate_plane_area(mask):
    """
    Calculate the area covered by the mask's 2D bounding box as a fraction of total image area.
    """
    if mask.numel() == 0:
        return 0.0

    # Find coordinates where mask > 0.5 (valid mask pixels)
    valid_mask = mask > 0.5

    # If no valid pixels, return 0
    if not torch.any(valid_mask):
        return 0.0

    # Get mask dimensions
    H, W = mask.shape
    total_area = H * W

    # Find bounding box coordinates
    # Get row and column indices of valid pixels
    valid_coords = torch.nonzero(valid_mask, as_tuple=False)  # Returns [N, 2] array of [row, col]

    if valid_coords.size(0) == 0:
        return 0.0

    # Find min/max coordinates to form bounding box
    min_row = torch.min(valid_coords[:, 0]).item()
    max_row = torch.max(valid_coords[:, 0]).item()
    min_col = torch.min(valid_coords[:, 1]).item()
    max_col = torch.max(valid_coords[:, 1]).item()

    # Calculate bounding box dimensions
    bbox_height = max_row - min_row + 1
    bbox_width = max_col - min_col + 1
    bbox_area = bbox_height * bbox_width

    # Return ratio of bounding box area to total image area
    return bbox_area / total_area
