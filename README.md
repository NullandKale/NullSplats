# NullSplats

Tkinter + OpenGL desktop app for training and viewing 3D Gaussian splats from casual captures. It wraps COLMAP for camera poses, uses PyTorch + gsplat for training, and stores everything in a reproducible cache tree so scenes can be resumed later.

## What it does

https://github.com/user-attachments/assets/6bcda99b-5d9d-4960-a759-5008b4bdc766

- Ingest a video or image folder, extract and score frames, and auto-select a subset.
- Run COLMAP SfM to produce camera poses and sparse points.
- Train Gaussian splats on the GPU with gsplat; export checkpoints as .ply or .splat.
- View splats in an embedded OpenGL viewer inside the app.
- Keep per-scene inputs/outputs under cache for repeatable workflows.

Nullsplats supports 3 methods of creating splats:

1. Traditional colmap + [gsplat](https://github.com/nerfstudio-project/gsplat) training.
2. [Depth Anything 3](github.com/ByteDance-Seed/depth-anything-3) 3D Gaussian Estimation.
3. [SHARP](https://github.com/apple/ml-sharp) Monocular View Synthesis.

Here are some sample splats trained using this program.
- [50 views — gsplat, 720p (iter 12k)](https://superspl.at/editor?load=https%3A%2F%2Fraw.githubusercontent.com%2FNullandKale%2FNullSplats%2Fmaster%2Fassets%2Fgsplat_iter_12000_50views_720p.splat)
- [5 views — Depth Anything 3, 720p](https://superspl.at/editor?load=https%3A%2F%2Fraw.githubusercontent.com%2FNullandKale%2FNullSplats%2Fmaster%2Fassets%2Fsplat_DA3_5views_720p.splat)
- [1 view — SHARP, 720p](https://superspl.at/editor?load=https%3A%2F%2Fraw.githubusercontent.com%2FNullandKale%2FNullSplats%2Fmaster%2Fassets%2Fsplat_SHARP_1views_720p.splat)

## Video Examples
Gsplat 50-view

https://github.com/user-attachments/assets/0a5431de-d146-4192-8305-6ec645161ba1

Gsplat closeup

https://github.com/user-attachments/assets/01245b1e-0098-4ebb-ad6c-00e220ba4e85

DA3 5-view

https://github.com/user-attachments/assets/c5cf7697-eb91-4ee9-9f46-6190622df765

DA3 closeup

https://github.com/user-attachments/assets/b87fa2f9-23c9-4a3b-bba7-82734a069edf

SHARP 1-view

https://github.com/user-attachments/assets/127c5651-a7da-471f-8c65-5c65db2283b2

SHARP closeup

https://github.com/user-attachments/assets/c5599843-46d5-4d2f-ab3d-5881534f3c3e

Side-by-side comparison

https://github.com/user-attachments/assets/00e8449d-0f26-4258-8469-4dc7424652ca

On my RTX pro 6000 Blackwell the gsplat trained in around 5 minutes including colmap time. Depth Anything 3 took around 3-4 minutes but used a significant 16GB of vram. SHARP produced a splat in around 2.5 minutes.

Overall each is pretty good considering the input. The single view SHARP splat is particularly impressive. If DA3 was less weird, was less transparent it would be significantly better. The geometry looks pretty good.

## Discord

Want to discuss development or get support? Open an Issue or checkout the [discord](https://discord.gg/nP8BMtZ42C).

## Repository layout

- main.py - app entrypoint (Tk root + tabs).
- nullsplats/
  - ui/ - Tk UI, tabs, OpenGL viewers, shaders (ui/shaders/*.vert|*.frag).
  - backend/ - frame extraction, COLMAP pipeline, splat training.
  - util/ - logging, config, threading helpers and tool path defaults.
- build.bat - portable bundle builder.
- run.bat - launcher used inside the portable bundle.
- requirements.txt - Python dependencies.

## Code Structure

### Overview
The app is organized around a small core state object (AppState), four UI tabs
(Inputs, COLMAP, Training, Exports), and a backend pipeline that handles frame
extraction, COLMAP structure-from-motion, and gsplat training.

### High-Level Flow
1) main.py sets up logging, creates AppState, and builds the Tk root in ui/root.py.
2) ui/root.py wires the four tabs and routes scene selection between them.
3) Inputs tab creates scenes, extracts frames, and persists selected/resized frames.
4) COLMAP tab runs SfM to generate camera poses and sparse points.
5) Training tab runs gsplat training and streams live previews.
6) Exports tab lists checkpoints and renders previews/turntables.

### Core State and Caching
- nullsplats/app_state.py owns AppState (config + SceneManager + current scene).
- nullsplats/backend/io_cache.py defines ScenePaths and metadata read/write.
- Cache layout (per scene):
  - cache/inputs/<scene_id>/source (original source copy)
  - cache/inputs/<scene_id>/frames_all
  - cache/inputs/<scene_id>/frames_selected
  - cache/inputs/<scene_id>/metadata.json
  - cache/outputs/<scene_id>/sfm
  - cache/outputs/<scene_id>/splats
  - cache/outputs/<scene_id>/renders
- nullsplats/backend/scene_manager.py handles scene discovery, selection persistence,
  and thumbnail caching (thumbnails.db).

### UI Architecture
#### Root + Tabs
- ui/root.py builds the ttk.Notebook, instantiates tabs, and coordinates tab changes.
- Tabs:
  - Inputs: ui/tab_inputs.py + mixins
  - COLMAP: ui/tab_colmap.py
  - Training: ui/tab_training.py + layout/preview mixins
  - Exports: ui/tab_exports.py
- Wizard flows:
  - Inline wizard in Inputs: ui/tab_inputs_wizard.py (single popup with inputs + training preset + COLMAP options).
  - Standalone wizard window: ui/wizard.py

#### Inputs Tab
- ui/tab_inputs.py is the coordinator for the Inputs workflow.
- ui/tab_inputs_scenes.py renders the scene sidebar and scene management.
- ui/tab_inputs_grid.py renders the virtualized frame grid and thumbnails.
- Key backend usage: backend/video_frames.py, backend/scene_manager.py.

#### COLMAP Tab
- ui/tab_colmap.py runs SfM and manages COLMAP settings/logs.
- Backend calls:
  - backend/sfm_pipeline.py (COLMAP CLI)

#### Training Tab
- ui/tab_training.py orchestrates training runs, manages logging, and owns preview state.
- Training method is selectable (gsplat or DA3). Live preview is gsplat-only.
- ui/tab_training_layout.py builds the UI widgets.
- ui/tab_training_preview.py handles preview polling + in-memory preview queue.
- Backend calls:
  - backend/splat_train.py (gsplat training loop)

#### Exports Tab
- ui/tab_exports.py lists checkpoints, opens a preview viewer, and renders turntables.
- Uses GLCanvas to display .ply checkpoints and imageio for video output.

### Backend Pipeline
#### Frame Extraction
- backend/video_frames.py handles:
  - ffmpeg/ffprobe extraction
  - sharpness/variance scoring
  - auto-select of best frames
  - cache persistence of selections

#### SfM (COLMAP)
- backend/sfm_pipeline.py runs COLMAP feature extraction, matching, mapping,
  and model conversion. Logs go to cache/outputs/<scene_id>/sfm/logs.

#### Training
- backend/splat_train.py is the training entry point.
- Supporting modules:
  - backend/splat_train_config.py (dataclasses and callbacks)
  - backend/splat_train_io.py (COLMAP text parsing + frame loading)
  - backend/splat_train_ops.py (CUDA config, optimizers, export helpers)
  - backend/gs_utils.py (camera/appearance optimization utilities)
- DA3 backend: backend/splat_backends/depth_anything3_trainer.py (Depth Anything 3 inference + gs_ply export)

### Rendering and Viewer Stack
- ui/gl_canvas.py is the main preview surface:
  - Wraps GaussianSplatViewer for live OpenGL display.
  - Uses SplatRenderer (gsplat rasterization) for offline renders and turntables.
  - Supports in-memory previews via PreviewPayload.
- ui/gaussian_splat_viewer.py is the OpenGL renderer (instanced quads + shaders).
- ui/gaussian_splat_camera.py contains camera math helpers.
- Shaders live in ui/shaders/gaussian_splat.vert and ui/shaders/gaussian_splat.frag.
- Control panels:
  - ui/render_controls.py (basic controls)
  - ui/advanced_render_controls.py (debug/scale/camera)
  - ui/colmap_camera_panel.py (apply COLMAP poses)

### Threading and Logging
- util/threading.py runs background tasks and marshals callbacks to the Tk thread.
- util/logging.py sets a consistent console + file logger under logs/app.log.
- util/tooling_paths.py resolves default COLMAP and CUDA paths.

### Tooling, Build, and Run
- requirements.txt contains core deps (torch, gsplat, PyOpenGL, etc.).
- build.bat creates a portable bundle with venv, COLMAP, and CUDA DLLs.
- run.bat is the launcher inside the portable bundle.

## Requirements

- Windows (primary target) or a Linux environment with matching binaries.
- Python 3.10+ with pip/venv.
- GPU with a CUDA-capable driver; PyTorch CUDA build installed.
- ffmpeg/ffprobe on PATH (for video extraction).
- COLMAP binaries (CUDA build recommended) under tools/colmap or user-provided path.
- Depth Anything 3 backend (submodule under tools/depth-anything-3).
- SHARP backend (submodule under tools/sharp).

## Install for development

Prereqs:

- Windows 10/11.
- Python 3.10+ on PATH.
- Visual Studio 2022 or newer (Desktop development with C++).
- CUDA Toolkit 12.8: https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network
- COLMAP under `tools/colmap` (or set in the UI).
- ffmpeg/ffprobe on PATH.

From repo root (recommended):

```
powershell -ExecutionPolicy Bypass -File tools\setup.ps1
```

The setup script initializes the VS build environment, enforces CUDA 12.8, updates submodules,
creates the venv, installs dependencies, and builds gsplat from source.

## Running the app

With the venv active:

```
python main.py
```

Logs go to logs and stdout; cache lives under cache/inputs/<scene_id> and cache/outputs/<scene_id>.

## UI workflow

- Inputs tab: choose or create a Scene ID, select video or image folder, set candidate/target frame counts, then Extract Frames. Frames and metadata land in cache/inputs/<scene>/.
- COLMAP tab: verify COLMAP path, matcher, and camera model, then run SfM. Outputs land in cache/outputs/<scene>/sfm.
- Training tab: configure CUDA device and training hyperparams, then run training. Outputs land in cache/outputs/<scene>/splats.
- Exports tab: browse checkpoints and preview them in the viewer.

## Portable bundle (Windows)

build.bat creates a self-contained bundle under build\NullSplats-portable and build\NullSplats-portable.zip.

- Prereq: .venv populated with all deps (including CUDA PyTorch/gsplat).
- Optional: set SKIP_CLEAN=1 to reuse an existing bundle; pass a CUDA path as the first arg to override CUDA_PATH/CUDA_HOME for DLL copy. Set REQUIRE_CUDA=0 if you intentionally want to skip bundling CUDA DLLs (otherwise the build fails when CUDA is missing). CUDA copy pulls DLLs from CUDA_SRC\bin (cud*/nv*).
- Optional: set SKIP_ZIP=1 to skip creating the zip (faster). If 7z is on PATH, zipping uses -mx=0 (store-only) for speed; otherwise falls back to PowerShell Compress-Archive -CompressionLevel Fastest.
- The builder prunes unused Python packages (tqdm, tyro, opencv-python, PyYAML) and copies only core CUDA DLLs; COLMAP is bundled, GLOMAP is not.
- If you need to debug CUDA bundling, build.bat prints source and destination paths plus directory listings for the copied DLLs. Set REQUIRE_CUDA=1 to fail fast when none are copied.
- Run from repo root:

```
build.bat
```

Inside the bundle, use run.bat to launch; it activates the bundled venv and prepends bundled CUDA and COLMAP paths.

## Known issues

1. Download is Windows + CUDA only.
2. COLMAP is not built during build process.
3. Download is large.
4. Thumbnails fail to load sometimes.
5. Splat rendering is slightly off.
6. Camera control is not great.
7. Does not include ffmpeg with the build.
8. May need CUDA SDK on the path, even though it is included.

## Paths and defaults

- COLMAP default: bundled tools/colmap/COLMAP.bat if present; otherwise user-specified.
- CUDA default: bundled cuda/ inside the bundle; otherwise CUDA_PATH/CUDA_HOME, then C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8.
- Cache: created on demand under cache/inputs and cache/outputs.
