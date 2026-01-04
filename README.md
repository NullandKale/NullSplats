# NullSplats

A Tkinter + OpenGL desktop app for **training and viewing 3D Gaussian splats** from casual captures. NullSplats wraps **COLMAP** for camera poses, uses **PyTorch + gsplat** (plus optional monocular backends) for training/inference, and stores everything in a **reproducible cache tree** so you can resume scenes without redoing work.

Demo:
https://github.com/user-attachments/assets/6bcda99b-5d9d-4960-a759-5008b4bdc766

## Highlights

- Ingest a video or image folder, extract and score frames, and auto-select a subset.
- Run COLMAP SfM to produce camera poses and sparse points.
- Train Gaussian splats on the GPU with gsplat; export checkpoints as `.ply` or `.splat`.
- View splats in an embedded OpenGL viewer inside the app.
- Keep per-scene inputs/outputs under `cache/` for repeatable, resumable workflows.

## Splat creation methods

NullSplats supports three ways to create splats:

1. **COLMAP + [gsplat](https://github.com/nerfstudio-project/gsplat)** training (classic pipeline, best quality with enough views).
2. **[Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)** 3D Gaussian Estimation (fast, few-view friendly).
3. **[SHARP](https://github.com/apple/ml-sharp)** monocular view synthesis (surprisingly strong even from a single view).

## Sample splats

- [50 views — gsplat, 720p (iter 12k)](https://superspl.at/editor?load=https%3A%2F%2Fraw.githubusercontent.com%2FNullandKale%2FNullSplats%2Fmaster%2Fassets%2Fgsplat_iter_12000_50views_720p.splat)
- [5 views — Depth Anything 3, 720p](https://superspl.at/editor?load=https%3A%2F%2Fraw.githubusercontent.com%2FNullandKale%2FNullSplats%2Fmaster%2Fassets%2Fsplat_DA3_5views_720p.splat)
- [1 view — SHARP, 720p](https://superspl.at/editor?load=https%3A%2F%2Fraw.githubusercontent.com%2FNullandKale%2FNullSplats%2Fmaster%2Fassets%2Fsplat_SHARP_1views_720p.splat)

## Video examples

Gsplat 50-view:
https://github.com/user-attachments/assets/0a5431de-d146-4192-8305-6ec645161ba1

Gsplat closeup:
https://github.com/user-attachments/assets/01245b1e-0098-4ebb-ad6c-00e220ba4e85

DA3 5-view:
https://github.com/user-attachments/assets/c5cf7697-eb91-4ee9-9f46-6190622df765

DA3 closeup:
https://github.com/user-attachments/assets/b87fa2f9-23c9-4a3b-bba7-82734a069edf

SHARP 1-view:
https://github.com/user-attachments/assets/127c5651-a7da-471f-8c65-5c65db2283b2

SHARP closeup:
https://github.com/user-attachments/assets/c5599843-46d5-4d2f-ab3d-5881534f3c3e

Side-by-side comparison:
https://github.com/user-attachments/assets/00e8449d-0f26-4258-8469-4dc7424652ca

### Rough timings (RTX Pro 6000 Blackwell)

- **gsplat + COLMAP**: ~5 minutes end-to-end (incl. COLMAP)
- **Depth Anything 3**: ~3–4 minutes, ~16GB VRAM peak
- **SHARP**: ~2.5 minutes

Notes: SHARP’s single-view result is especially impressive. DA3 often nails geometry but can look “weird” or overly transparent; if that improves, it could be a standout few-view option.

## Community

Want to discuss development or get support? Open an Issue or join the Discord:
https://discord.gg/nP8BMtZ42C

## Repository layout

- `main.py` - app entrypoint (Tk root + tabs)
- `nullsplats/`
  - `ui/` - Tk UI, tabs, OpenGL viewers, shaders (`ui/shaders/*.vert|*.frag`)
  - `backend/` - frame extraction, COLMAP pipeline, splat training
  - `util/` - logging, config, threading helpers and tool path defaults
- `build.bat` - portable bundle builder
- `run.bat` - launcher used inside the portable bundle
- `requirements.txt` - Python dependencies

## Architecture (high level)

The app centers around a small core state object (`AppState`), four UI tabs (Inputs, COLMAP, Training, Exports), and a backend pipeline that handles frame extraction, COLMAP structure-from-motion, and training/inference.

### End-to-end flow

1) `main.py` sets up logging, creates `AppState`, and builds the Tk root in `ui/root.py`.
2) `ui/root.py` wires the four tabs and routes scene selection between them.
3) Inputs tab creates scenes, extracts frames, and persists selected/resized frames.
4) COLMAP tab runs SfM to generate camera poses and sparse points.
5) Training tab runs gsplat training and streams live previews.
6) Exports tab lists checkpoints and renders previews/turntables.

### Core state and caching

- `nullsplats/app_state.py` owns `AppState` (config + `SceneManager` + current scene)
- `nullsplats/backend/io_cache.py` defines `ScenePaths` and metadata read/write

Cache layout (per scene):

- `cache/inputs/<scene_id>/source` (original source copy)
- `cache/inputs/<scene_id>/frames_all`
- `cache/inputs/<scene_id>/frames_selected`
- `cache/inputs/<scene_id>/metadata.json`
- `cache/outputs/<scene_id>/sfm`
- `cache/outputs/<scene_id>/splats`
- `cache/outputs/<scene_id>/renders`

`nullsplats/backend/scene_manager.py` handles scene discovery, selection persistence, and thumbnail caching (`thumbnails.db`).

### UI tabs

#### Root + Tabs

- `ui/root.py` builds the `ttk.Notebook`, instantiates tabs, and coordinates tab changes
- Tabs:
  - Inputs: `ui/tab_inputs.py` + mixins
  - COLMAP: `ui/tab_colmap.py`
  - Training: `ui/tab_training.py` + layout/preview mixins
  - Exports: `ui/tab_exports.py`

Wizard flows:

- Inline wizard in Inputs: `ui/tab_inputs_wizard.py` (single popup with inputs + training preset + COLMAP options)
- Standalone wizard window: `ui/wizard.py`

#### Inputs tab

- Coordinator: `ui/tab_inputs.py`
- Scene sidebar: `ui/tab_inputs_scenes.py`
- Virtualized frame grid + thumbnails: `ui/tab_inputs_grid.py`
- Backend: `backend/video_frames.py`, `backend/scene_manager.py`

#### COLMAP tab

- `ui/tab_colmap.py` runs SfM and manages COLMAP settings/logs
- Backend: `backend/sfm_pipeline.py` (COLMAP CLI)

#### Training tab

- `ui/tab_training.py` orchestrates training runs, manages logging, and owns preview state
- Training method selectable (gsplat or DA3); live preview is gsplat-only
- UI layout: `ui/tab_training_layout.py`
- Preview polling/queue: `ui/tab_training_preview.py`
- Backend: `backend/splat_train.py` (gsplat training loop)
- DA3 backend: `backend/splat_backends/depth_anything3_trainer.py` (Depth Anything 3 inference + gs_ply export)

#### Exports tab

- `ui/tab_exports.py` lists checkpoints, opens a preview viewer, and renders turntables
- Uses `GLCanvas` to display `.ply` checkpoints and `imageio` for video output

### Rendering and viewer stack

- `ui/gl_canvas.py` is the main preview surface
  - Wraps `GaussianSplatViewer` for live OpenGL display
  - Uses `SplatRenderer` (gsplat rasterization) for offline renders and turntables
  - Supports in-memory previews via `PreviewPayload`
- `ui/gaussian_splat_viewer.py` is the OpenGL renderer (instanced quads + shaders)
- `ui/gaussian_splat_camera.py` contains camera math helpers
- Shaders: `ui/shaders/gaussian_splat.vert` and `ui/shaders/gaussian_splat.frag`
- Control panels:
  - `ui/render_controls.py` (basic controls)
  - `ui/advanced_render_controls.py` (debug/scale/camera)
  - `ui/colmap_camera_panel.py` (apply COLMAP poses)

### Threading and logging

- `util/threading.py` runs background tasks and marshals callbacks to the Tk thread
- `util/logging.py` sets a consistent console + file logger under `logs/app.log`
- `util/tooling_paths.py` resolves default COLMAP and CUDA paths

## Requirements

- Windows (primary target) or a Linux environment with matching binaries
- Python 3.10+ with pip/venv
- GPU with a CUDA-capable driver; PyTorch CUDA build installed
- `ffmpeg`/`ffprobe` on PATH (video extraction)
- COLMAP binaries (CUDA build recommended) under `tools/colmap` or user-provided path
- Depth Anything 3 backend (submodule under `tools/depth-anything-3`)
- SHARP backend (submodule under `tools/sharp`)

## Install (development)

Prereqs:

- Windows 10/11
- Python 3.10+ on PATH
- Visual Studio 2022 or newer (Desktop development with C++)
- CUDA Toolkit 12.8: https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network
- COLMAP under `tools/colmap` (or set in the UI)
- `ffmpeg`/`ffprobe` on PATH

From repo root (recommended):

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup.ps1
