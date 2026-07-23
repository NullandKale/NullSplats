"""Depth-to-normal derivation: default normal target source (no dedicated normal network)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def depth_to_normals(depth: torch.Tensor, origins: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """(H,W,1 or H,W) depth + (H,W,3) ray origins/directions -> (H,W,3) unit world-space normals.

    Unprojects each pixel to a world-space point P = origin + direction*depth, takes the cross
    product of the finite-differenced tangent vectors (dP/du, dP/dv), and normalizes. Border
    rows/columns reuse the adjacent interior gradient (no wraparound). Fully vectorized, no
    per-pixel Python loop.
    """
    if depth.dim() == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    points = origins + directions * depth.unsqueeze(-1)  # (H, W, 3)

    du = points[:, 1:, :] - points[:, :-1, :]  # (H, W-1, 3) tangent along u (columns)
    dv = points[1:, :, :] - points[:-1, :, :]  # (H-1, W, 3) tangent along v (rows)

    du = torch.cat([du, du[:, -1:, :]], dim=1)  # (H, W, 3), repeat last column
    dv = torch.cat([dv, dv[-1:, :, :]], dim=0)  # (H, W, 3), repeat last row

    normals = torch.cross(du, dv, dim=-1)
    return F.normalize(normals, dim=-1, eps=1e-8)
