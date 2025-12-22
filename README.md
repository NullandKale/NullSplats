# NullSplats

Tkinter + OpenGL desktop app for training and viewing 3D Gaussian splats from casual captures. It wraps COLMAP for camera poses, uses PyTorch + `gsplat` for training, and stores everything in a reproducible `cache/` tree so scenes can be resumed later.

## What it does

https://github.com/user-attachments/assets/6bcda99b-5d9d-4960-a759-5008b4bdc766

- Ingest a video or image folder, extract and score frames, and auto-select a subset.
- Run COLMAP SfM to produce camera poses and sparse points.
- Train Gaussian splats on the GPU with `gsplat` and PyTorch; export checkpoints as `.ply` or `.splat`.
- View splats in an embedded OpenGL viewer inside the app.
- Keep per-scene inputs/outputs under `cache/` for repeatable workflows.

## Repository layout

- `main.py` — app entrypoint (Tk root + tabs).
- `nullsplats/`
  - `ui/` — Tk UI, tabs, OpenGL viewers, shaders (`ui/shaders/*.vert|*.frag`).
  - `backend/` — frame extraction, COLMAP pipeline, splat training.
  - `util/` — logging, config, threading helpers.
- `tools/` — bundled COLMAP binaries and optional CUDA slices (if present).
- `build.bat` — portable bundle builder (app + venv + tools into `build/NullSplats-portable` and zip).
- `run.bat` — launcher used inside the portable bundle (uses bundled Python, CUDA, COLMAP).
- `requirements.txt` — Python dependencies.

## Requirements

- Windows (primary target) or a Linux environment with matching binaries.
- Python 3.10+ with pip/venv.
- GPU with a CUDA-capable driver; PyTorch CUDA build installed.
- ffmpeg/ffprobe on PATH (for video extraction).
- COLMAP binaries (CUDA build recommended) under `tools/colmap` or user-provided path.
- Optional: GLOMAP binaries under `tools/glomap` (future use).

## Install for development

From repo root:

```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

If you need to bootstrap CUDA-friendly PyTorch/gsplat, run `tools\setup_cuda_venv.bat` (optional helper if present).

## Running the app

With the venv active:

```
python main.py
```

Logs go to `logs/` and stdout; cache lives under `cache/inputs/<scene_id>` and `cache/outputs/<scene_id>`.

## UI workflow

- **Inputs tab**: choose or create a Scene ID, select video or image folder, set candidate/target frame counts, then Extract Frames. Frames and metadata land in `cache/inputs/<scene>/`.
- **Training tab**: point COLMAP to the bundled path (auto-filled if present under `tools/colmap`). Configure CUDA device and training hyperparams, then run SfM and training. Outputs land in `cache/outputs/<scene>/sfm` and `.../splats`.
- **Exports tab**: browse checkpoints and preview them in the viewer.

## Portable bundle (Windows)

`build.bat` creates a self-contained bundle under `build\NullSplats-portable` and `build\NullSplats-portable.zip`.

- Prereq: `.venv` populated with all deps (including CUDA PyTorch/gsplat).
- Optional: set `SKIP_CLEAN=1` to reuse an existing bundle; pass a CUDA path as the first arg to override `CUDA_PATH`/`CUDA_HOME` for DLL copy. Set `REQUIRE_CUDA=0` if you intentionally want to skip bundling CUDA DLLs (otherwise the build fails when CUDA is missing). CUDA copy pulls DLLs from `CUDA_SRC\bin` (cud*/nv*).
- Optional: set `SKIP_ZIP=1` to skip creating the .zip (faster). If `7z` is on PATH, zipping uses `-mx=0` (store-only) for speed; otherwise falls back to PowerShell `Compress-Archive -CompressionLevel Fastest`.
- The builder prunes unused Python packages (tqdm, tyro, opencv-python, PyYAML) and copies only core CUDA DLLs; COLMAP is bundled, GLOMAP is not.
- If you need to debug CUDA bundling, `build.bat` prints source and destination paths plus directory listings for the copied DLLs. Set `REQUIRE_CUDA=1` to fail fast when none are copied.
- Run from repo root:

```
build.bat
```

Inside the bundle, use `run.bat` to launch; it activates the bundled venv and prepends bundled CUDA and COLMAP paths.

## Known issues:

1. Download is windows + cuda only
2. Colmap is not built during build process
3. Download is giant.
4. Thumbnails fail to load sometimes.
5. Splat rendering is slightly off
6. Camera control is not great.
7. Doesn't include ffmpeg with the build
8. May need cuda sdk on the path, even though I include it.

## Paths and defaults

- COLMAP default: bundled `tools/colmap/COLMAP.bat` if present; otherwise user-specified.
- CUDA default: bundled `cuda/` inside the bundle; otherwise `CUDA_PATH`/`CUDA_HOME`, then `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`.
- Cache: created on demand under `cache/inputs` and `cache/outputs`.