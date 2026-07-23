"""Interactive Tkinter viewer + dataset-import + training control panel for Pixel Ray Cache.

A standalone window (this package's own -- decoupled from nullsplats/ui, since the production
Gaussian-splat OpenGL viewer cannot render this representation, see the package docstring) that:

  - Imports a new dataset (video -> extract_frames -> COLMAP SfM, same pipeline gpu_smoke_test.py
    uses) from the "Import" tab, without leaving the GUI.
  - Lets you pick, per the "Model" tab, ANY registered per-tile architecture (see
    tile_models/registry.py) with a hyperparameter panel built dynamically from that
    architecture's own HyperparamSpec schema -- add a new architecture to the registry and it
    shows up here with zero viewer.py changes.
  - Lets you pick, per the "Training" tab, optimizer / LR schedule / color+depth loss functions
    (see optim.py / losses.py), targets (color/depth/normal), and coarse-to-fine cascade
    settings, then Start Training on a background thread while flying a camera around and
    watching the model's current weights render live.

Because tile identity is a fixed screen-space index rather than a 3D location (the package
docstring's stated limitation), quality degrades quickly the further the camera strays from an
actual training pose -- this viewer is as much a live diagnostic for THAT behavior as it is a way
to look at a good reconstruction.

Run (from repo root):
    .venv\\Scripts\\python.exe -m nullsplats.research.pixel_ray_cache.viewer --scene <id> [--stage color_depth_normal]

If --scene doesn't exist yet (no extracted frames / no COLMAP output), the viewer still opens --
use the Import tab to create it. A checkpoint at the resolved location is loaded automatically if
one already exists.

Controls:
    W/A/S/D       move forward/back/strafe, relative to the current view direction
    Q/E           move down/up
    Left-drag     look around (yaw/pitch)
    Left/Right    jump to the previous/next training frame's exact pose (also resets drift)
    R             reset to the current training frame's exact pose
    1/2/3         switch display channel: color / depth / normal (only if the checkpoint has them)
    Esc           quit

Threading model: training and dataset import both run on background daemon threads, reporting
back to the GUI thread only via a shared queue.Queue of small tagged messages (never raw model/
Tk objects). Training's on_step hook renders a plain numpy array *on the training thread itself*
(safe: called right after optimizer.step(), before the next forward pass touches weights again)
using whatever camera pose/channel the GUI thread most recently set, and pushes only that array.
The GUI thread only ever touches its own `self.loaded` model when no training is active. A small
lock guards reads/writes of the shared camera state.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import torch
from PIL import Image, ImageTk
from tqdm import tqdm

from nullsplats.backend.io_cache import ScenePaths
from nullsplats.research.pixel_ray_cache import (
    cascade,
    cascade_train,
    data,
    optim as optim_registry,
    preview,
    tiles,
    train,
)
from nullsplats.research.pixel_ray_cache.config import PixelRayCacheConfig, target_spec_for
from nullsplats.research.pixel_ray_cache.data import RayCacheFrame
from nullsplats.research.pixel_ray_cache.losses import list_losses
from nullsplats.research.pixel_ray_cache.tile_models.base import HyperparamSpec
from nullsplats.research.pixel_ray_cache.tile_models.registry import (
    estimate_params_per_tile,
    get_hyperparams,
    get_tile_model_cls,
    list_tile_models,
)
from nullsplats.research.pixel_ray_cache.train import LoadedCheckpoint, load_checkpoint
from nullsplats.util.logging import get_logger, setup_logging

logger = get_logger("research.pixel_ray_cache.viewer")

MOVE_SPEED = 0.02  # world units per tick per held movement key
LOOK_SPEED = 0.005  # radians per pixel of mouse drag
PITCH_LIMIT = math.radians(89.0)
TICK_MS = 33  # ~30Hz input-polling/redraw schedule; actual cadence depends on render cost
LIVE_PREVIEW_INTERVAL_SEC = 0.3  # minimum gap between live-training preview renders
COLMAP_MATCHERS = ("exhaustive", "sequential")
COLMAP_CAMERA_MODELS = ("PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL")

# From a 15-trial Optuna sweep (see sweep.py) on scene prc_full_10000, 2026-07-22 (logged in
# PROGRESS.md), run on a small 64x64-tile CROP for search speed. The outright winner (resmlp,
# hidden_width=59, hidden_layers=8) is NOT used here: at that crop's ~4k tiles it's tiny, but
# scaled to a real full-grid scene (~230k+ tiles) it balloons to ~7.2 BILLION params (~27 GiB
# weights, ~80 GiB with Adam) -- an early version of this default caused exactly that OOM. Using
# trial 2 instead: the best-performing *mlp* trial (loss 0.037, vs resmlp's 0.011 crop-only
# score), whose small size (width 23, 1 layer, no encoding) stays safe at full-grid scale
# (~54M params, <1 GiB). See tile_models.registry.estimate_params_per_tile / the GUI's live
# "estimated memory" label (updated every tick, next to the Start button) before choosing
# anything bigger -- that's what should have caught this the first time.
DEFAULT_TILE_MODEL = "mlp"
DEFAULT_TILE_MODEL_KWARGS = {
    "hidden_width": 23,
    "hidden_layers": 1,
    "hidden_activation": "relu",
    "encoding": "none",
}
DEFAULT_LR = 0.0012325212368433337
DEFAULT_OPTIMIZER = "rmsprop"
DEFAULT_COLOR_LOSS = "mse"

# Conservative starting weight for oob.py's out-of-distribution black regularization. A
# controlled test (prc_pose_norm_and_oob_sanity.py) found weight=1.0 (equal to the implicit
# weight-1 primary photometric loss) visibly degrades in-distribution reconstruction quality --
# 0.15 held quality while still measurably darkening far-away predictions. Kept lower still here
# (0.05) since that test used a tiny toy scene; raise it from the Training tab if far-away
# extrapolation still looks too "averaged" at this weight.
DEFAULT_OOB_LOSS_WEIGHT = 0.05


def _rotation_from_yaw_pitch(yaw: float, pitch: float) -> torch.Tensor:
    """3x3 rotation, camera-local convention (+x right, +y down, +z forward, matching
    rays.pixel_rays/COLMAP): yaw rotates around the local Y axis, pitch around the
    (post-yaw) local X axis."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return ry @ rx


class _CameraState:
    """Position/orientation expressed as a delta on top of a chosen base training frame's own
    camtoworld. Resetting (R key, or jumping to a different frame) zeroes the delta, exactly
    reproducing that frame's real pose -- the one case guaranteed to be in-distribution."""

    def __init__(self, base_frame: RayCacheFrame) -> None:
        self.base_frame = base_frame
        self.translation = torch.zeros(3)
        self.yaw = 0.0
        self.pitch = 0.0

    def reset(self, base_frame: RayCacheFrame | None = None) -> None:
        if base_frame is not None:
            self.base_frame = base_frame
        self.translation = torch.zeros(3)
        self.yaw = 0.0
        self.pitch = 0.0

    def camtoworld(self) -> torch.Tensor:
        base_r = self.base_frame.camtoworld[:3, :3].cpu()
        base_t = self.base_frame.camtoworld[:3, 3].cpu()
        delta_r = _rotation_from_yaw_pitch(self.yaw, self.pitch)
        c2w = torch.eye(4)
        c2w[:3, :3] = base_r @ delta_r
        c2w[:3, 3] = base_t + base_r @ self.translation
        return c2w

    def move_local(self, dx: float, dy: float, dz: float) -> None:
        """Move by (dx,dy,dz) in the CURRENT view's local frame (already includes yaw/pitch)."""
        delta_r = _rotation_from_yaw_pitch(self.yaw, self.pitch)
        self.translation = self.translation + delta_r @ torch.tensor([dx, dy, dz])

    def look(self, d_yaw: float, d_pitch: float) -> None:
        self.yaw += d_yaw
        self.pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, self.pitch + d_pitch))


class _QueueStatusLogger:
    """Adapts the (logging.Logger-shaped) `logger` param that gpu_smoke_test._ensure_scene_ready
    expects into something that also pushes each message into the viewer's status queue, so
    dataset-import progress shows up in the GUI the same way training status does."""

    def __init__(self, message_queue: "queue.Queue[tuple]", real_logger) -> None:
        self._queue = message_queue
        self._real = real_logger

    def info(self, msg: str, *args) -> None:
        text = msg % args if args else msg
        self._real.info(text)
        self._queue.put(("status", text))


class PixelRayCacheViewer:
    def __init__(
        self,
        loaded: LoadedCheckpoint | None,
        frames: list[RayCacheFrame],
        *,
        base_config: PixelRayCacheConfig,
        device: torch.device,
        max_display_width: int = 1024,
    ) -> None:
        self.loaded = loaded
        self.frames = frames
        self.base_config = base_config
        self.device = device
        self.max_display_width = max_display_width
        self.frame_idx = 0
        self.camera: _CameraState | None = _CameraState(frames[0]) if frames else None
        self.available_channels = [name for name, _, _ in loaded.model.target_spec] if loaded else ["color"]
        self.channel = self.available_channels[0]
        self._keys_down: set[str] = set()
        self._drag_last: tuple[int, int] | None = None
        self._camera_lock = threading.Lock()

        # Training / import state.
        self.is_training = False
        self.is_importing = False
        self.stop_event: threading.Event | None = None
        self.message_queue: "queue.Queue[tuple]" = queue.Queue()
        self._last_preview_time = 0.0

        self.root = tk.Tk()
        self.root.title("Pixel Ray Cache viewer")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="both", expand=True)

        image_frame = tk.Frame(top_frame)
        image_frame.pack(side="left", fill="both", expand=True)
        self.image_label = tk.Label(image_frame)
        self.image_label.pack()

        frame_jump_row = tk.Frame(image_frame)
        frame_jump_row.pack(fill="x")
        tk.Label(frame_jump_row, text="Jump to frame:").pack(side="left")
        self.var_frame_index = tk.StringVar(value="1")
        self.frame_spinbox = tk.Spinbox(
            frame_jump_row, from_=1, to=max(1, len(frames)), textvariable=self.var_frame_index, width=6,
            command=self._on_frame_spinbox_changed,
        )
        self.frame_spinbox.pack(side="left")
        self.frame_spinbox.bind("<Return>", self._on_frame_spinbox_changed)
        self.frame_spinbox.bind("<FocusOut>", self._on_frame_spinbox_changed)
        self.var_frame_name = tk.StringVar(value=frames[0].name if frames else "")
        tk.Label(frame_jump_row, textvariable=self.var_frame_name, fg="gray30").pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar()
        tk.Label(image_frame, textvariable=self.status_var, anchor="w", justify="left").pack(fill="x")
        self.train_status_var = tk.StringVar(
            value="No scene loaded yet -- use the Import tab." if not frames else "No training run started yet."
        )
        tk.Label(image_frame, textvariable=self.train_status_var, anchor="w", justify="left", wraplength=480).pack(
            fill="x"
        )
        self._photo: ImageTk.PhotoImage | None = None

        self._build_controls(top_frame)

        self.root.bind("<KeyPress>", self._on_key_down)
        self.root.bind("<KeyRelease>", self._on_key_up)
        self.image_label.bind("<ButtonPress-1>", self._on_drag_start)
        self.image_label.bind("<B1-Motion>", self._on_drag_move)
        self.image_label.bind("<ButtonRelease-1>", self._on_drag_end)
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    # -- control panel ---------------------------------------------------------------------

    def _build_controls(self, parent: tk.Widget) -> None:
        outer = tk.Frame(parent, padx=10, pady=10)
        outer.pack(side="right", fill="y")

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        import_tab = tk.Frame(notebook, padx=8, pady=8)
        model_tab = tk.Frame(notebook, padx=8, pady=8)
        training_tab = tk.Frame(notebook, padx=8, pady=8)
        notebook.add(import_tab, text="Import")
        notebook.add(model_tab, text="Model")
        notebook.add(training_tab, text="Training")

        self._build_import_tab(import_tab)
        self._build_model_tab(model_tab)
        self._build_training_tab(training_tab)

        self.memory_estimate_var = tk.StringVar(value="")
        tk.Label(outer, textvariable=self.memory_estimate_var, anchor="w", justify="left", wraplength=260).pack(
            fill="x", pady=(8, 0)
        )
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(outer, variable=self.progress_var, maximum=100.0)
        self.progress_bar.pack(fill="x", pady=(10, 0))
        self.progress_text_var = tk.StringVar(value="")
        tk.Label(
            outer, textvariable=self.progress_text_var, anchor="w", justify="left", wraplength=260,
            font=("Consolas", 8),
        ).pack(fill="x")
        self.start_button = tk.Button(outer, text="Start Training", command=self._start_training, width=20)
        self.start_button.pack(pady=(10, 0))

    @staticmethod
    def _entry_row(parent: tk.Widget, row: int, label: str, var: tk.Variable, width: int = 12) -> None:
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        tk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="e")

    # -- Import tab ---------------------------------------------------------------------

    def _build_import_tab(self, parent: tk.Widget) -> None:
        self.var_import_input = tk.StringVar(value="assets/input.mp4")
        self.var_import_scene = tk.StringVar(value=self.base_config.scene)
        self.var_import_candidate = tk.StringVar(value="10000")
        self.var_import_target = tk.StringVar(value="10000")
        self.var_import_matcher = tk.StringVar(value=COLMAP_MATCHERS[0])
        self.var_import_camera_model = tk.StringVar(value=COLMAP_CAMERA_MODELS[0])

        row = 0
        tk.Label(parent, text="input video/folder").grid(row=row, column=0, sticky="w")
        input_row = tk.Frame(parent)
        tk.Entry(input_row, textvariable=self.var_import_input, width=18).pack(side="left")
        tk.Button(input_row, text="...", command=self._browse_input, width=3).pack(side="left")
        input_row.grid(row=row, column=1, sticky="e")
        row += 1
        self._entry_row(parent, row, "scene id", self.var_import_scene)
        row += 1
        self._entry_row(parent, row, "candidate frames", self.var_import_candidate)
        row += 1
        self._entry_row(parent, row, "target frames", self.var_import_target)
        row += 1
        tk.Label(parent, text="colmap matcher").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            parent, textvariable=self.var_import_matcher, values=list(COLMAP_MATCHERS), width=10, state="readonly"
        ).grid(row=row, column=1, sticky="e")
        row += 1
        tk.Label(parent, text="camera model").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            parent, textvariable=self.var_import_camera_model, values=list(COLMAP_CAMERA_MODELS), width=10,
            state="readonly",
        ).grid(row=row, column=1, sticky="e")
        row += 1
        self.var_normalize_poses = tk.BooleanVar(value=self.base_config.normalize_poses)
        tk.Checkbutton(parent, text="normalize camera poses to [-1,1]", variable=self.var_normalize_poses).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        self.var_import_depth = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="also compute depth (DA3) now", variable=self.var_import_depth).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        tk.Label(
            parent,
            text="Extracts every available frame by default (10000 is a large sentinel,\n"
            "clamped to the video's real length) and runs COLMAP once. Toggling pose\n"
            "normalization requires re-importing (even an already-processed scene --\n"
            "COLMAP itself is skipped/cheap, only frame loading changes) and isn't\n"
            "compatible with a checkpoint trained under the other setting. Depth is\n"
            "disk-cached per scene regardless of when it's first computed -- checking\n"
            "the depth box here just does that work up front (as part of dataset\n"
            "preparation) instead of deferring it to your first depth-enabled\n"
            "Start Training click.",
            justify="left", fg="gray30",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        row += 1
        self.import_button = tk.Button(parent, text="Import && Run COLMAP", command=self._start_import, width=20)
        self.import_button.grid(row=row, column=0, columnspan=2, pady=(8, 0))

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.var_import_input.set(path)

    def _start_import(self) -> None:
        if self.is_training or self.is_importing:
            return
        input_path = Path(self.var_import_input.get())
        scene_id = self.var_import_scene.get().strip()
        if not scene_id:
            self.train_status_var.set("Scene id is required for import.")
            return
        if not input_path.exists():
            self.train_status_var.set(f"Input not found: {input_path}")
            return
        try:
            candidate_count = int(self.var_import_candidate.get())
            target_count = int(self.var_import_target.get())
        except ValueError as exc:
            self.train_status_var.set(f"Invalid frame count: {exc}")
            return

        # Free any checkpoint currently resident on the GPU for live preview (e.g. the viewer was
        # launched with --stage <x> and never trained this session) before import starts. Import
        # is about to replace self.loaded via _reload_scene anyway once it finishes, but leaving
        # the OLD model on the GPU in the meantime competes for VRAM with whatever import itself
        # needs -- concretely, DA3's own model (several GiB) failing to fit alongside an
        # already-resident tile-model checkpoint is exactly the OOM this was added to prevent.
        # Safe to do here (GUI thread, before the background import thread starts) with no
        # cross-thread race on self.loaded.
        if self.loaded is not None:
            self.loaded.model.to("cpu")
            self.loaded = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.is_importing = True
        self.import_button.configure(state="disabled")
        self.train_status_var.set(f"importing {input_path} as scene={scene_id}...")

        thread = threading.Thread(
            target=self._import_worker,
            args=(
                input_path, scene_id, candidate_count, target_count,
                self.var_import_matcher.get(), self.var_import_camera_model.get(),
                self.var_normalize_poses.get(), self.var_import_depth.get(),
            ),
            daemon=True,
        )
        thread.start()

    def _import_worker(
        self, input_path: Path, scene_id: str, candidate_count: int, target_count: int, matcher: str,
        camera_model: str, normalize_poses: bool, compute_depth: bool,
    ) -> None:
        from nullsplats.research.pixel_ray_cache.gpu_smoke_test import _ensure_scene_ready

        adapter = _QueueStatusLogger(self.message_queue, logger)
        try:
            selected_count = _ensure_scene_ready(
                scene_id=scene_id,
                input_path=input_path,
                candidate_count=candidate_count,
                target_count=target_count,
                cache_root=self.base_config.cache_root,
                colmap_matcher=matcher,
                colmap_camera_model=camera_model,
                logger=adapter,
            )

            if compute_depth:
                # Depth generation as part of input processing, not deferred to the first
                # depth-enabled training click: computed here on the import thread (safe --
                # pure CPU/GPU tensor work, no Tk access) and disk-cached the same way
                # depth_targets.py always has, so any later training run (any architecture,
                # any hyperparameters) just finds it already there.
                self.message_queue.put(("status", f"computing depth (DA3) for scene={scene_id}..."))
                from nullsplats.research.pixel_ray_cache import data as data_mod
                from nullsplats.research.pixel_ray_cache.depth_targets import compute_or_load_depth_for_scene

                paths = ScenePaths(scene_id=scene_id, cache_root=self.base_config.cache_root)
                depth_frames = data_mod.load_scene_frames(paths, device=self.device, normalize_poses=normalize_poses)
                cache_dir = self.base_config.cache_root / "outputs" / scene_id / "pixel_ray_cache" / "depth_cache"
                compute_or_load_depth_for_scene(
                    depth_frames, cache_dir=cache_dir, pretrained_id=self.base_config.da3_pretrained,
                    device=self.device, chunk_size=self.base_config.da3_chunk_size,
                )
                self.message_queue.put(("status", f"depth ready for scene={scene_id} ({len(depth_frames)} frames)"))

            self.message_queue.put(("scene_ready", scene_id, selected_count))
        except Exception as exc:  # noqa: BLE001 -- report to the GUI instead of dying silently
            logger.exception("import/colmap failed")
            self.message_queue.put(("import_error", str(exc)))

    def _reload_scene(self, scene_id: str, selected_count: int) -> None:
        normalize_poses = self.var_normalize_poses.get()
        self.base_config = dataclasses.replace(
            self.base_config, scene=scene_id, out_dir=None, normalize_poses=normalize_poses
        )
        paths = ScenePaths(scene_id=scene_id, cache_root=self.base_config.cache_root)
        self.frames = data.load_scene_frames(paths, device=self.device, normalize_poses=normalize_poses)
        self.frame_spinbox.configure(to=max(1, len(self.frames)))
        self._set_frame_idx(0)
        try:
            self.loaded = load_checkpoint(self.base_config.resolve_out_dir(), device=self.device)
            self.available_channels = [name for name, _, _ in self.loaded.model.target_spec]
        except FileNotFoundError:
            self.loaded = None
            self.available_channels = ["color"]
        self.channel = self.available_channels[0]
        self.train_status_var.set(
            f"scene '{scene_id}' ready: {selected_count} frames. Configure hyperparameters and click Start Training."
        )

    # -- Model tab ------------------------------------------------------------------------

    def _build_model_tab(self, parent: tk.Widget) -> None:
        self.var_tile_model = tk.StringVar(value=self.base_config.tile_model)
        tk.Label(parent, text="architecture").grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(
            parent, textvariable=self.var_tile_model, values=list_tile_models(), width=12, state="readonly"
        )
        combo.grid(row=0, column=1, sticky="e")
        combo.bind("<<ComboboxSelected>>", self._rebuild_model_hyperparam_widgets)

        self.model_hyperparam_frame = tk.Frame(parent)
        self.model_hyperparam_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(8, 0))
        self.model_hyperparam_vars: dict[str, tuple[HyperparamSpec, tk.Variable]] = {}
        self._rebuild_model_hyperparam_widgets()

    def _rebuild_model_hyperparam_widgets(self, *_args) -> None:
        for widget in self.model_hyperparam_frame.winfo_children():
            widget.destroy()
        self.model_hyperparam_vars = {}
        model_name = self.var_tile_model.get()
        specs = get_hyperparams(model_name)
        # Seed from base_config.tile_model_kwargs when it matches the currently-selected
        # architecture (e.g. launched with --tile-model + --model-kwargs), else each spec's own
        # declared default.
        seed = self.base_config.tile_model_kwargs if self.base_config.tile_model == model_name else {}
        for i, spec in enumerate(specs):
            tk.Label(self.model_hyperparam_frame, text=spec.name).grid(row=i, column=0, sticky="w")
            default = seed.get(spec.name, spec.default)
            if spec.kind == "choice":
                var: tk.Variable = tk.StringVar(value=str(default))
                ttk.Combobox(
                    self.model_hyperparam_frame, textvariable=var, values=list(spec.choices), width=10, state="readonly"
                ).grid(row=i, column=1, sticky="e")
            elif spec.kind == "bool":
                var = tk.BooleanVar(value=bool(default))
                tk.Checkbutton(self.model_hyperparam_frame, variable=var).grid(row=i, column=1, sticky="e")
            else:  # "int" | "float"
                var = tk.StringVar(value=str(default))
                tk.Entry(self.model_hyperparam_frame, textvariable=var, width=12).grid(row=i, column=1, sticky="e")
            self.model_hyperparam_vars[spec.name] = (spec, var)

    # -- Training tab ---------------------------------------------------------------------

    def _build_training_tab(self, parent: tk.Widget) -> None:
        self.var_lr = tk.StringVar(value=str(self.base_config.lr))
        self.var_epochs = tk.StringVar(value=str(self.base_config.epochs))
        self.var_optimizer = tk.StringVar(value=self.base_config.optimizer)
        self.var_lr_schedule = tk.StringVar(value=self.base_config.lr_schedule)
        self.var_color_loss = tk.StringVar(value=self.base_config.color_loss)
        self.var_depth_loss_kind = tk.StringVar(value=self.base_config.depth_loss)
        self.var_depth = tk.BooleanVar(value="depth" in self.base_config.targets)
        self.var_normal = tk.BooleanVar(value="normal" in self.base_config.targets)
        self.var_depth_weight = tk.StringVar(value=str(self.base_config.depth_loss_weight))
        self.var_normal_weight = tk.StringVar(value=str(self.base_config.normal_loss_weight))
        self.var_cascade = tk.BooleanVar(value=self.base_config.cascade)
        self.var_cascade_levels = tk.StringVar(value=str(self.base_config.cascade_levels))
        self.var_oob_weight = tk.StringVar(value=str(self.base_config.oob_loss_weight))
        self.var_oob_scale = tk.StringVar(
            value="" if self.base_config.oob_translation_scale is None else str(self.base_config.oob_translation_scale)
        )
        self.var_images_per_step = tk.StringVar(value=str(self.base_config.images_per_step))

        row = 0

        def label_row(text: str) -> int:
            nonlocal row
            tk.Label(parent, text=text, font=("", 9, "bold")).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(6, 0)
            )
            row += 1
            return row

        def combo_row(label: str, var: tk.StringVar, values: list[str]) -> None:
            nonlocal row
            tk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
            ttk.Combobox(parent, textvariable=var, values=values, width=10, state="readonly").grid(
                row=row, column=1, sticky="e"
            )
            row += 1

        label_row("Optimization")
        self._entry_row(parent, row, "learning rate", self.var_lr)
        row += 1
        self._entry_row(parent, row, "epochs / level", self.var_epochs)
        row += 1
        self._entry_row(parent, row, "images per step", self.var_images_per_step)
        row += 1
        tk.Label(
            parent, text="(1 = original per-image loop; higher batches more images per\n"
            "forward/backward call for less Python overhead, at more VRAM)",
            justify="left", fg="gray30",
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        combo_row("optimizer", self.var_optimizer, optim_registry.list_optimizers())
        combo_row("lr schedule", self.var_lr_schedule, optim_registry.list_schedulers())
        combo_row("color loss", self.var_color_loss, list_losses())

        label_row("Targets")
        tk.Checkbutton(parent, text="depth (DA3)", variable=self.var_depth).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        tk.Checkbutton(parent, text="normal (needs depth)", variable=self.var_normal).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        combo_row("depth loss", self.var_depth_loss_kind, list_losses())
        self._entry_row(parent, row, "depth loss weight", self.var_depth_weight)
        row += 1
        self._entry_row(parent, row, "normal loss weight", self.var_normal_weight)
        row += 1

        label_row("Cascade")
        tk.Checkbutton(parent, text="coarse-to-fine cascade", variable=self.var_cascade).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        self._entry_row(parent, row, "cascade levels", self.var_cascade_levels)
        row += 1

        label_row("No-data (black) regularization")
        tk.Label(
            parent,
            text="Trains a randomly perturbed camera pose's rays as black each step, so tiles\n"
            "learn 'unlike anything I saw -> black' instead of an averaged color when the\n"
            "viewer camera strays outside the training trajectory. 0 = off.",
            justify="left", fg="gray30", wraplength=260,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self._entry_row(parent, row, "oob loss weight", self.var_oob_weight)
        row += 1
        self._entry_row(parent, row, "oob translation scale", self.var_oob_scale)
        row += 1
        tk.Label(parent, text="(blank = auto-estimate from scene)", fg="gray30").grid(
            row=row, column=0, columnspan=2, sticky="w"
        )

    # -- camera / input ---------------------------------------------------------------------

    def _on_key_down(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        self._keys_down.add(key)
        if not self.frames:
            return
        if key == "left":
            self._jump_frame(-1)
        elif key == "right":
            self._jump_frame(1)
        elif key == "r":
            with self._camera_lock:
                self.camera.reset(self.frames[self.frame_idx])
        elif key in ("1", "2", "3"):
            idx = int(key) - 1
            if idx < len(self.available_channels):
                self.channel = self.available_channels[idx]

    def _on_key_up(self, event: tk.Event) -> None:
        self._keys_down.discard(event.keysym.lower())

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_last = (event.x, event.y)

    def _on_drag_move(self, event: tk.Event) -> None:
        if self._drag_last is None or self.camera is None:
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self._drag_last = (event.x, event.y)
        with self._camera_lock:
            self.camera.look(dx * LOOK_SPEED, dy * LOOK_SPEED)

    def _on_drag_end(self, _event: tk.Event) -> None:
        self._drag_last = None

    def _set_frame_idx(self, idx: int) -> None:
        """Jumps the camera to frame `idx`'s exact pose (clamped in range) -- the one case
        guaranteed to be in-distribution, per _CameraState's docstring -- and keeps the jump-to
        spinbox/filename label in sync regardless of which UI path (arrow keys or the spinbox)
        triggered the jump."""
        if not self.frames:
            return
        idx = max(0, min(len(self.frames) - 1, idx))
        self.frame_idx = idx
        with self._camera_lock:
            self.camera.reset(self.frames[idx])
        self.var_frame_index.set(str(idx + 1))
        self.var_frame_name.set(self.frames[idx].name)

    def _jump_frame(self, delta: int) -> None:
        if not self.frames:
            return
        self._set_frame_idx((self.frame_idx + delta) % len(self.frames))

    def _on_frame_spinbox_changed(self, *_args) -> None:
        if not self.frames:
            return
        try:
            idx = int(self.var_frame_index.get()) - 1
        except ValueError:
            self.var_frame_index.set(str(self.frame_idx + 1))
            return
        self._set_frame_idx(idx)

    def _apply_movement(self) -> None:
        if self.camera is None:
            return
        dx = dy = dz = 0.0
        if "w" in self._keys_down:
            dz += MOVE_SPEED
        if "s" in self._keys_down:
            dz -= MOVE_SPEED
        if "d" in self._keys_down:
            dx += MOVE_SPEED
        if "a" in self._keys_down:
            dx -= MOVE_SPEED
        if "e" in self._keys_down:
            dy += MOVE_SPEED
        if "q" in self._keys_down:
            dy -= MOVE_SPEED
        if dx or dy or dz:
            with self._camera_lock:
                self.camera.move_local(dx, dy, dz)

    # -- static (not-training) rendering -----------------------------------------------------

    def _render(self) -> np.ndarray:
        assert self.loaded is not None and self.camera is not None
        base = self.frames[self.frame_idx]
        with self._camera_lock:
            camtoworld = self.camera.camtoworld()
        pose_frame = RayCacheFrame(
            name=base.name, image_path=base.image_path, camtoworld=camtoworld.to(self.device),
            K=base.K, width=base.width, height=base.height,
        )
        render_h = self.loaded.render_height or base.height
        render_w = self.loaded.render_width or base.width
        common = dict(
            device=self.device, tile=self.loaded.render_tile, render_height=render_h, render_width=render_w
        )
        if self.channel == "color":
            return preview.render_color_array(self.loaded.model, pose_frame, **common)
        if self.channel == "depth":
            gray = preview.render_depth_array(self.loaded.model, pose_frame, **common)
            return np.stack([gray, gray, gray], axis=-1)
        return preview.render_normal_array(self.loaded.model, pose_frame, **common)

    def _display(self, array: np.ndarray) -> None:
        image = Image.fromarray(array)
        if image.width > self.max_display_width:
            scale = self.max_display_width / image.width
            image = image.resize((self.max_display_width, max(1, int(image.height * scale))), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo)

    # -- training ---------------------------------------------------------------------------

    def _read_model_kwargs(self) -> dict:
        kwargs = {}
        for name, (spec, var) in self.model_hyperparam_vars.items():
            raw = var.get()
            if spec.kind == "int":
                kwargs[name] = int(raw)
            elif spec.kind == "float":
                kwargs[name] = float(raw)
            elif spec.kind == "bool":
                kwargs[name] = bool(raw)
            else:
                kwargs[name] = raw
        return kwargs

    def _start_training(self) -> None:
        if self.is_training:
            return
        if not self.frames:
            self.train_status_var.set("No scene loaded -- use the Import tab first.")
            return
        try:
            tile_model_kwargs = self._read_model_kwargs()
            lr = float(self.var_lr.get())
            epochs = int(self.var_epochs.get())
            depth_weight = float(self.var_depth_weight.get())
            normal_weight = float(self.var_normal_weight.get())
            cascade_levels = int(self.var_cascade_levels.get())
            oob_weight = float(self.var_oob_weight.get())
            oob_scale_raw = self.var_oob_scale.get().strip()
            oob_scale = float(oob_scale_raw) if oob_scale_raw else None
            images_per_step = int(self.var_images_per_step.get())
        except ValueError as exc:
            self.train_status_var.set(f"Invalid hyperparameter value: {exc}")
            return

        targets: list[str] = ["color"]
        if self.var_depth.get():
            targets.append("depth")
            if self.var_normal.get():
                targets.append("normal")

        try:
            config = dataclasses.replace(
                self.base_config,
                tile_model=self.var_tile_model.get(),
                tile_model_kwargs=tile_model_kwargs,
                lr=lr,
                optimizer=self.var_optimizer.get(),
                lr_schedule=self.var_lr_schedule.get(),
                color_loss=self.var_color_loss.get(),
                depth_loss=self.var_depth_loss_kind.get(),
                epochs=epochs,
                targets=tuple(targets),
                cascade=self.var_cascade.get(),
                cascade_levels=cascade_levels,
                depth_loss_weight=depth_weight,
                normal_loss_weight=normal_weight,
                oob_loss_weight=oob_weight,
                oob_translation_scale=oob_scale,
                images_per_step=images_per_step,
                tile_crop=None,
            )
        except (ValueError, TypeError) as exc:
            self.train_status_var.set(f"Invalid config: {exc}")
            return

        self.available_channels = list(targets)
        if self.channel not in self.available_channels:
            self.channel = self.available_channels[0]

        self.is_training = True
        self.stop_event = threading.Event()
        self.start_button.configure(text="Stop Training", command=self._stop_training)
        self.train_status_var.set("starting...")
        self.progress_var.set(0.0)
        self.progress_text_var.set("")

        thread = threading.Thread(target=self._training_worker, args=(config, self.stop_event), daemon=True)
        thread.start()

    def _stop_training(self) -> None:
        if not self.is_training or self.stop_event is None:
            return
        self.train_status_var.set("stopping (finishing current step)...")
        self.stop_event.set()
        self.start_button.configure(state="disabled")

    def _on_close(self) -> None:
        if self.is_training and self.stop_event is not None:
            self.stop_event.set()
        self.root.destroy()

    def _live_preview_on_step(self, model, tile, render_h, render_w) -> None:
        """Called by the training loop after every optimizer step (see train.OnStep). Renders at
        most once every LIVE_PREVIEW_INTERVAL_SEC, using whatever camera pose/channel the GUI
        thread most recently set -- runs entirely on the training thread, using the model in its
        current (consistent, just-stepped) state, and only pushes the resulting plain array
        across the thread boundary."""
        now = time.perf_counter()
        if now - self._last_preview_time < LIVE_PREVIEW_INTERVAL_SEC:
            return
        self._last_preview_time = now
        if self.camera is None:
            return
        with self._camera_lock:
            camtoworld = self.camera.camtoworld()
        frame_idx = self.frame_idx
        channel = self.channel
        base = self.frames[frame_idx]
        pose_frame = RayCacheFrame(
            name=base.name, image_path=base.image_path, camtoworld=camtoworld.to(self.device),
            K=base.K, width=base.width, height=base.height,
        )
        eff_h = render_h or base.height
        eff_w = render_w or base.width
        available = [name for name, _, _ in model.target_spec]
        use_channel = channel if channel in available else available[0]
        try:
            common = dict(device=self.device, tile=tile, render_height=eff_h, render_width=eff_w)
            if use_channel == "color":
                arr = preview.render_color_array(model, pose_frame, **common)
            elif use_channel == "depth":
                gray = preview.render_depth_array(model, pose_frame, **common)
                arr = np.stack([gray, gray, gray], axis=-1)
            else:
                arr = preview.render_normal_array(model, pose_frame, **common)
            self.message_queue.put(("preview", arr))
        except Exception:
            logger.exception("live preview render failed")

    def _training_worker(self, config: PixelRayCacheConfig, stop_event: threading.Event) -> None:
        def on_epoch_loss(*args) -> None:
            if len(args) == 2:
                epoch, loss = args
                self.message_queue.put(("status", f"epoch {epoch + 1}/{config.epochs}  loss={loss:.6f}"))
            else:
                level_idx, epoch, loss = args
                self.message_queue.put(
                    ("status", f"level {level_idx + 1}  epoch {epoch + 1}/{config.epochs}  loss={loss:.6f}")
                )

        def on_level_start(level_idx, spec) -> None:
            self.message_queue.put(("status", f"level {level_idx + 1} starting ({spec.n_groups} groups)"))

        n_batches_per_epoch = math.ceil(len(self.frames) / max(1, config.images_per_step))
        if config.cascade:
            base = self.frames[0]
            height_tiles, width_tiles = base.height // 2, base.width // 2
            plan = cascade.build_cascade_plan(height_tiles, width_tiles, max_levels=config.cascade_levels)
            total_steps = len(plan) * config.epochs * n_batches_per_epoch
        else:
            total_steps = config.epochs * n_batches_per_epoch

        # tqdm.format_meter reused directly here so the GUI's progress text matches the CLI's bar
        # formatting -- this IS the "add tqdm to the gui" ask, just rendered into a ttk widget
        # instead of a terminal instead of a second bespoke formatter.
        progress_start = time.perf_counter()
        completed = 0

        def on_batch_progress(*args) -> None:
            nonlocal completed
            completed += 1
            loss = args[-1]
            elapsed = time.perf_counter() - progress_start
            if len(args) == 4:
                epoch, batch_idx, n_batches, _ = args
                prefix = f"epoch {epoch + 1}/{config.epochs}"
            else:
                level_idx, epoch, batch_idx, n_batches, _ = args
                prefix = f"level {level_idx + 1}/{len(plan)} epoch {epoch + 1}/{config.epochs}"
            text = tqdm.format_meter(
                n=completed, total=total_steps, elapsed=elapsed, prefix=prefix, unit="batch",
                postfix=f"loss={loss:.4f}", ascii=True,
            )
            fraction = 100.0 * completed / max(1, total_steps)
            self.message_queue.put(("progress", fraction, text))

        try:
            if config.cascade:
                result = cascade_train.run_cascade_training(
                    config,
                    self.frames,
                    on_level_start=on_level_start,
                    on_epoch_loss=on_epoch_loss,
                    on_batch_progress=on_batch_progress,
                    on_step=self._live_preview_on_step,
                    stop_event=stop_event,
                )
            else:
                n_tiles = tiles.n_tiles_for(self.frames[0].height, self.frames[0].width, crop=config.tile_crop)
                model = get_tile_model_cls(config.tile_model)(n_tiles, config.target_spec, **config.tile_model_kwargs)
                result = train.run_training(
                    config,
                    self.frames,
                    model,
                    on_epoch_loss=on_epoch_loss,
                    on_batch_progress=on_batch_progress,
                    on_step=self._live_preview_on_step,
                    stop_event=stop_event,
                )
            loaded = load_checkpoint(config.resolve_out_dir(), device=self.device)
            self.message_queue.put(("done", loaded, result))
        except Exception as exc:  # noqa: BLE001 -- report to the GUI instead of dying silently
            logger.exception("training failed")
            self.message_queue.put(("error", str(exc)))

    def _drain_messages(self) -> np.ndarray | None:
        """Process all pending messages (training AND import) this tick; returns the latest
        live-preview array if one arrived, else None."""
        latest_preview: np.ndarray | None = None
        while True:
            try:
                msg = self.message_queue.get_nowait()
            except queue.Empty:
                break
            tag = msg[0]
            if tag == "preview":
                latest_preview = msg[1]
            elif tag == "status":
                self.train_status_var.set(msg[1])
            elif tag == "progress":
                _, fraction, text = msg
                self.progress_var.set(fraction)
                self.progress_text_var.set(text)
            elif tag == "done":
                _, loaded, result = msg
                self.loaded = loaded
                self.available_channels = [name for name, _, _ in loaded.model.target_spec]
                if self.channel not in self.available_channels:
                    self.channel = self.available_channels[0]
                self.is_training = False
                self.stop_event = None
                self.start_button.configure(text="Start Training", command=self._start_training, state="normal")
                self.train_status_var.set(
                    f"done: final_loss={result.final_loss:.6f}  checkpoint={result.checkpoint_path}"
                )
                self.progress_var.set(100.0)
            elif tag == "error":
                self.is_training = False
                self.stop_event = None
                self.start_button.configure(text="Start Training", command=self._start_training, state="normal")
                self.train_status_var.set(f"ERROR: {msg[1]}")
                self.progress_var.set(0.0)
                self.progress_text_var.set("")
            elif tag == "scene_ready":
                _, scene_id, selected_count = msg
                self._reload_scene(scene_id, selected_count)
                self.is_importing = False
                self.import_button.configure(state="normal")
            elif tag == "import_error":
                self.is_importing = False
                self.import_button.configure(state="normal")
                self.train_status_var.set(f"IMPORT ERROR: {msg[1]}")
        return latest_preview

    # -- main loop --------------------------------------------------------------------------

    def _update_memory_estimate(self) -> None:
        """Estimates full-grid parameter/memory cost for the CURRENTLY configured architecture
        (Model tab) at the current scene's tile count, without ever constructing the real
        N-tile model -- this is exactly the check that would have caught the oversized resmlp
        default before it OOM'd: estimate_params_per_tile only builds a throwaway n_tiles=1
        probe, so it's cheap to recompute every tick regardless of how large the real N is.
        """
        try:
            tile_model_kwargs = self._read_model_kwargs()
            targets: list[str] = ["color"]
            if self.var_depth.get():
                targets.append("depth")
                if self.var_normal.get():
                    targets.append("normal")
            spec = target_spec_for(tuple(targets))
            per_tile = estimate_params_per_tile(self.var_tile_model.get(), spec, **tile_model_kwargs)
        except Exception:
            self.memory_estimate_var.set("")
            return

        if not self.frames:
            self.memory_estimate_var.set(f"~{per_tile:,} params/tile (load a scene to estimate full-grid total)")
            return

        base = self.frames[0]
        n_tiles = tiles.n_tiles_for(base.height, base.width)
        total_params = per_tile * n_tiles
        weights_gib = total_params * 4 / (2**30)
        adam_gib = weights_gib * 3
        warning = "  [WARNING: likely too large for a consumer GPU]" if adam_gib > 10 else ""
        self.memory_estimate_var.set(
            f"~{per_tile:,} params/tile x {n_tiles:,} tiles = {total_params:,} params\n"
            f"~{weights_gib:.2f} GiB weights, ~{adam_gib:.2f} GiB w/ Adam optimizer state{warning}"
        )

    def _tick(self) -> None:
        self._apply_movement()
        latest_preview = self._drain_messages()

        if not self.is_training and not self.is_importing:
            self._update_memory_estimate()

        if self.is_training:
            if latest_preview is not None:
                self._display(latest_preview)
        elif self.loaded is not None and self.frames:
            start = time.perf_counter()
            array = self._render()
            render_ms = (time.perf_counter() - start) * 1000.0
            self._display(array)
            pos = [round(v, 3) for v in self.camera.translation.tolist()]
            self.status_var.set(
                f"frame {self.frame_idx + 1}/{len(self.frames)}  channel={self.channel}  "
                f"render={render_ms:.0f}ms  pos={pos}  "
                f"yaw={math.degrees(self.camera.yaw):.0f} pitch={math.degrees(self.camera.pitch):.0f}"
            )
        elif not self.frames:
            self.status_var.set("No scene loaded -- use the Import tab to extract frames and run COLMAP.")
        else:
            self.status_var.set("No checkpoint loaded yet -- configure hyperparameters and click Start Training.")

        self.root.after(TICK_MS, self._tick)

    def run(self) -> None:
        self.root.after(0, self._tick)
        self.root.mainloop()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="Scene id; doesn't need to exist yet -- use the Import tab.")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument(
        "--stage",
        default=None,
        help="Subfolder under pixel_ray_cache/, e.g. color_depth_normal -- matches "
        "gpu_smoke_test.py's per-stage output dirs. Omit if you trained via cli.py directly "
        "without --out-dir.",
    )
    parser.add_argument("--out-dir", default=None, help="Explicit checkpoint directory; overrides --stage.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-display-width", type=int, default=1024)
    parser.add_argument(
        "--normalize-poses",
        action="store_true",
        help="Recenter+isotropically rescale camera positions into [-1,1] when loading frames "
        "(see data.compute_pose_normalization). Also settable/changeable from the Import tab's "
        "checkbox after launch. Not compatible with a checkpoint trained under the other setting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _build_arg_parser().parse_args(argv)
    device = torch.device(args.device)
    cache_root = Path(args.cache_root)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.stage:
        out_dir = cache_root / "outputs" / args.scene / "pixel_ray_cache" / args.stage
    else:
        out_dir = None  # PixelRayCacheConfig.resolve_out_dir()'s default convention

    base_config = PixelRayCacheConfig(
        scene=args.scene,
        cache_root=cache_root,
        device=args.device,
        out_dir=out_dir,
        tile_model=DEFAULT_TILE_MODEL,
        tile_model_kwargs=dict(DEFAULT_TILE_MODEL_KWARGS),
        lr=DEFAULT_LR,
        optimizer=DEFAULT_OPTIMIZER,
        color_loss=DEFAULT_COLOR_LOSS,
        oob_loss_weight=DEFAULT_OOB_LOSS_WEIGHT,
        normalize_poses=args.normalize_poses,
    )
    resolved_out_dir = base_config.resolve_out_dir()

    loaded: LoadedCheckpoint | None = None
    try:
        loaded = load_checkpoint(resolved_out_dir, device=device)
        logger.info(
            "Loaded checkpoint from %s: n_tiles=%d target_spec=%s render_tile=%s render_hw=(%s,%s)",
            resolved_out_dir, loaded.model.n_tiles, loaded.model.target_spec, loaded.render_tile,
            loaded.render_height, loaded.render_width,
        )
    except FileNotFoundError:
        logger.info("No existing checkpoint at %s.", resolved_out_dir)

    frames: list[RayCacheFrame] = []
    try:
        paths = ScenePaths(scene_id=args.scene, cache_root=cache_root)
        frames = data.load_scene_frames(paths, device=device, normalize_poses=args.normalize_poses)
        logger.info("Loaded %d frames for camera poses / training (normalize_poses=%s)", len(frames), args.normalize_poses)
    except (FileNotFoundError, ValueError) as exc:
        logger.info("Scene '%s' not ready yet (%s) -- use the Import tab.", args.scene, exc)

    viewer = PixelRayCacheViewer(
        loaded, frames, base_config=base_config, device=device, max_display_width=args.max_display_width
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
