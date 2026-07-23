"""Out-of-distribution ("no data") regularization for Pixel Ray Cache.

Supervises a randomly perturbed camera pose's rays as pure black each training step, so a
per-tile network explicitly learns "a ray unlike anything I was trained on -> black" instead of
implicitly regressing toward some blended/average color when queried from a pose far outside the
training trajectory -- e.g. flying the interactive viewer's camera away from the captured path,
which is exactly when this becomes visible (tile identity is a fixed screen-space index, so EVERY
tile is queried with novel rays simultaneously the moment the camera leaves the trained range --
this isn't a per-image "missing pixels" problem, it's an out-of-distribution one).

Off by default (config.oob_loss_weight == 0, zero extra compute in that case). Only applies to
the color target -- depth/normal have no sensible "no data" analog handled here. The perturbation
covers a random 3D direction each step (not a single fixed offset), since the real failure mode
is the camera moving away in an arbitrary direction, not a specific one.
"""

from __future__ import annotations

import math

import torch

from nullsplats.research.pixel_ray_cache.data import RayCacheFrame


def estimate_pose_scale(frames: list[RayCacheFrame]) -> float:
    """Largest pairwise distance between training camera positions -- used as the default
    out-of-distribution jitter magnitude when the user hasn't set one explicitly, so the jitter
    is proportional to how far the real capture actually moved (a tabletop scan and a room-scale
    walkthrough need very different absolute jitter distances to mean "clearly outside the
    trained range")."""
    positions = torch.stack([f.camtoworld[:3, 3] for f in frames]).cpu()
    if positions.shape[0] < 2:
        return 1.0
    diffs = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N,N,3)
    dists = diffs.norm(dim=-1)
    return float(dists.max().clamp_min(1e-6))


def random_perturbed_pose(
    base_camtoworld: torch.Tensor, *, translation_scale: float, device: torch.device
) -> torch.Tensor:
    """A random rigid perturbation of base_camtoworld: translation magnitude uniform in
    [1x, 3x] translation_scale in a random direction, plus a small random rotation (up to ~15
    degrees) so the negative sample isn't just a translated copy of the same view direction.
    """
    direction = torch.randn(3, device=device)
    direction = direction / direction.norm().clamp_min(1e-8)
    magnitude = translation_scale * (1.0 + 2.0 * torch.rand((), device=device).item())

    angle = (torch.rand((), device=device).item() - 0.5) * (math.pi / 6.0)  # +/- 15 degrees
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm().clamp_min(1e-8)
    zero = torch.zeros((), device=device)
    ax, ay, az = axis[0], axis[1], axis[2]
    skew = torch.stack(
        [
            torch.stack([zero, -az, ay]),
            torch.stack([az, zero, -ax]),
            torch.stack([-ay, ax, zero]),
        ]
    )
    # Rodrigues' rotation formula: R = I + sin(angle)*K + (1-cos(angle))*K^2
    r_jitter = torch.eye(3, device=device) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)

    perturbed = base_camtoworld.clone()
    perturbed[:3, :3] = r_jitter @ base_camtoworld[:3, :3]
    perturbed[:3, 3] = base_camtoworld[:3, 3] + direction * magnitude
    return perturbed
