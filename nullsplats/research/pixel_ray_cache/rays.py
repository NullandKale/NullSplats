"""Vectorized camera -> ray math (COLMAP/OpenCV convention, no OpenGL flip)."""

from __future__ import annotations

import torch


def qvec_tvec_to_camtoworld(qvec: list[float], tvec: list[float], *, device: torch.device) -> torch.Tensor:
    """COLMAP world-to-camera (qvec, tvec) -> 4x4 camera-to-world matrix.

    qvec = [qw, qx, qy, qz], tvec = [tx, ty, tz], both world-to-camera per COLMAP convention.
    R (world-to-camera rotation) is built from qvec; c2w rotation is R.T, c2w translation is
    -R.T @ t (the camera center in world space). Computed in float64 for numerical stability,
    cast to float32 on return.
    """
    q = torch.tensor(qvec, dtype=torch.float64, device=device)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    r00 = 1 - 2 * qy * qy - 2 * qz * qz
    r01 = 2 * qx * qy - 2 * qz * qw
    r02 = 2 * qx * qz + 2 * qy * qw
    r10 = 2 * qx * qy + 2 * qz * qw
    r11 = 1 - 2 * qx * qx - 2 * qz * qz
    r12 = 2 * qy * qz - 2 * qx * qw
    r20 = 2 * qx * qz - 2 * qy * qw
    r21 = 2 * qy * qz + 2 * qx * qw
    r22 = 1 - 2 * qx * qx - 2 * qy * qy
    R = torch.stack(
        [
            torch.stack([r00, r01, r02]),
            torch.stack([r10, r11, r12]),
            torch.stack([r20, r21, r22]),
        ]
    )  # (3,3) world-to-camera rotation
    t = torch.tensor(tvec, dtype=torch.float64, device=device)  # (3,)

    c2w = torch.eye(4, dtype=torch.float64, device=device)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w.to(torch.float32)


def pixel_rays(
    K: torch.Tensor, camtoworld: torch.Tensor, height: int, width: int, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-resolution per-pixel ray grid for one frame.

    K: (3,3) [[fx,0,cx],[0,fy,cy],[0,0,1]]. camtoworld: (4,4) camera-to-world.
    Returns (origins, directions), each (H, W, 3) float32; directions are unit-length; origin
    is the same camera-center point broadcast to every pixel. No per-pixel Python loop.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )  # both (H, W); v = row index, u = col index
    dir_x = (u + 0.5 - cx) / fx
    dir_y = (v + 0.5 - cy) / fy
    dir_z = torch.ones_like(dir_x)
    dirs_cam = torch.stack([dir_x, dir_y, dir_z], dim=-1)  # (H, W, 3), OpenCV convention
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

    R = camtoworld[:3, :3]  # (3,3)
    dirs_world = torch.einsum("ij,hwj->hwi", R, dirs_cam)  # (H, W, 3)
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True)

    origin = camtoworld[:3, 3]  # (3,)
    origins_world = origin.view(1, 1, 3).expand(height, width, 3).contiguous()
    return origins_world, dirs_world
