# NullSplats (working title)

Tkinter and OpenGL desktop application for training and viewing 3D Gaussian splats from casual captures, using COLMAP and GLOMAP for camera poses and `gsplat` plus PyTorch for splat training and rendering.

This document targets a robot that operates on the repository and also serves as a structural overview for the human. Explanations focus on concrete behavior and verifiable outcomes.


PROJECT OVERVIEW
----------------

The application presents three main tabs:

- Inputs tab:
  - Ingest a video or a folder of images.
  - Extract many candidate frames.
  - Select a smaller subset of high-quality frames.
  - Cache extracted frames and metadata for reuse.

- Training tab:
  - Run camera pose estimation with COLMAP and GLOMAP.
  - Train Gaussian splats using `gsplat` and PyTorch based on the generated poses.
  - Optionally show live splat previews via an OpenGL viewport bound to the latest checkpoint.

- Exports tab:
  - Browse cached `.ply` splat checkpoints for each scene.
  - View a selected splat in a read-only OpenGL viewer.
  - Export chosen `.ply` files or derived turntable renders.

All datasets and outputs exist under a structured `./cache/` directory so that scenes can be revisited and refined without repeating every processing step.


PROJECT STRUCTURE
-----------------

Repository layout for NullSplats:

.
+- main.py                 Entry point for the Tkinter application.
+- nullsplats/             Python package.
   +- __init__.py
   +- app_state.py         Global application state and scene registry.
   +- ui/
      +- __init__.py
      +- root.py           Tk root creation, menubar, notebook tabs.
      +- tab_inputs.py     Inputs tab widgets plus callbacks.
      +- tab_training.py   Training tab widgets plus callbacks.
      +- tab_exports.py    Exports tab widgets plus callbacks.
      +- gl_canvas.py      OpenGL-backed canvas embedded in Tk.
   +- backend/
      +- io_cache.py       Cache abstraction for inputs and outputs.
      +- video_frames.py   Video frame extraction and scoring.
      +- sfm_pipeline.py   COLMAP and GLOMAP command-line integration.
      +- splat_train.py    gsplat-based training loop and checkpointing.
      +- splat_export.py   Export helpers and render routines.
   +- util/
      +- config.py         Paths, tunables, and defaults.
      +- threading.py      Helper for background tasks and progress.
      +- logging.py        Console and file logging setup.
+- tools/
   +- setup_cuda_venv.bat  PyTorch plus gsplat environment bootstrapper.
   +- build.bat            Nuitka onefile build script for Windows.
+- tests/
   +- run_checks.bat       One-shot script to run py_compile, imports, UI, and tests.
+- cache/
   +- inputs/
      +- <scene_id>/
         +- source/
            +- video.mp4          Original video or equivalent.
         +- frames_all/           Full set of extracted frames.
         +- frames_selected/      Subset selected for training.
         +- metadata.json         Scene and selection metadata.
   +- outputs/
      +- <scene_id>/
         +- sfm/                  COLMAP and GLOMAP outputs.
         +- splats/               Splat snapshots and training logs.
         +- renders/              Optional rendered flythroughs.
+- .github/
   +- workflows/
      +- build-windows.yml        Tag-triggered Nuitka build workflow.


SCENE MODEL AND CACHE LAYOUT
----------------------------

Scene identity:

- Each dataset uses a Scene ID such as `room1_001`.
- All scene-specific data lives in:

  - cache/inputs/<scene_id>/
  - cache/outputs/<scene_id>/

Input cache layout for a scene:

- cache/inputs/<scene_id>/source/
  - video.mp4 for video capture input.
  - Or copies or links to original images for image-folder input.
- cache/inputs/<scene_id>/frames_all/
  - Dense sequence of frames as `frame_XXXX.png`.
- cache/inputs/<scene_id>/frames_selected/
  - Curated subset of frames for training.
- cache/inputs/<scene_id>/metadata.json
  - JSON structure with:
    - scene_id
    - source_type
    - source_path
    - candidate_count
    - target_count
    - created_at or equivalent marker.
    - available_frames list.
    - selected_frame_indices or filenames.

Output cache layout for a scene:

- cache/outputs/<scene_id>/sfm/
  - database.db and related COLMAP assets.
  - images directories and sparse reconstructions.
  - logs for SfM runs.
- cache/outputs/<scene_id>/splats/
  - iter_00000.ply, iter_00500.ply, etc.
  - training_log.jsonl for metrics and events.
  - config.json for hyperparameters and training configuration.
- cache/outputs/<scene_id>/renders/
  - orbit.mp4, turntable.mp4, and still frames such as keyframe_XXXX.png.


ENVIRONMENT AND SETUP
---------------------

Expected environment for running NullSplats:

- Operating system:
  - Windows compatible with CUDA and the used dependencies.
- GPU:
  - NVIDIA GPU with a CUDA driver compatible with CUDA 12.x.
- PyTorch with CUDA:
  - Training is GPU-only; install a CUDA build of PyTorch and ensure `torch.cuda.is_available()` returns True.
- Python:
  - Version 3.10 or higher, installed and on PATH.
- CUDA toolkit:
  - Installed and accessible to PyTorch and gsplat.
  - gsplat JIT-compiles its CUDA extension at runtime using the system toolkit (currently targeting CUDA 12.8). When producing a shippable exe we must plan how to supply a matching toolkit or prebuilt extension; this is not bundled yet.
- COLMAP and GLOMAP:
  - Installed as separate tools and available to the application via configured paths.
  - Repo-local defaults: `tools\COLMAP-3.7-windows-cuda\bin\colmap.exe` is preferred so PATH changes are unnecessary; enter this full path in the Training tab if not already prefilled. GLOMAP is staged for TODO-7; keep a real binary under `tools\glomap\bin` for future integration even though the current UI only drives COLMAP.
  - COLMAP: CUDA build such as `C:\Repos\COLMAP-3.7-windows-cuda\bin` also works if you point the Training tab to the binary.
  - GLOMAP: download the release from https://github.com/MariusKM/Colmap_CeresS_withCuDSS/releases/tag/v.1.0 and place `glomap.exe` under a stable folder (for example, `tools\glomap\bin`) for the upcoming TODO-7 integration.
  - COLMAP bundle adjustment speed tips: see https://github.com/colmap/colmap/blob/main/doc/faq.rst#speedup-bundle-adjustemnt when tuning SfM performance.

Virtual environment setup with the supplied helper script:

- From the repository root, run:
  - python -m venv .venv
  - call .venv\Scripts\activate.bat
  - call tools\setup_cuda_venv.bat

Application execution:

- With the virtual environment active:
  - python main.py


BUILD AND RELEASE
-----------------

Nuitka build script:

- tools/build.bat encapsulates creation of a single-file Windows executable.
- The script is responsible for:
  - Activating `.venv`.
  - Running Nuitka for `main.py` using onefile mode.
  - Producing `dist/NullSplats.exe` or similar output.

Conceptual build steps for the script:

- call ..\.venv\Scripts\activate.bat
- python -m nuitka --onefile --enable-plugin=tk-inter --follow-imports --output-filename=NullSplats.exe ..\main.py

GitHub Actions:

- .github/workflows/build-windows.yml:
  - Listens for tags with the format `v0.0.0`.
  - Performs checkout and Python setup.
  - Prepares environment, including Nuitka.
  - Runs `tools/build.bat`.
  - Uploads the resulting executable as an artifact and attaches it to a release for the tag.


UI TABS AND USER FLOWS
----------------------

Inputs tab:

- Scene selection:
  - Scene field plus buttons for New Scene and Load Existing.
  - New Scene:
    - Prompt for a Scene ID using alphanumeric characters, underscores, or dashes.
    - Create the cache directories for inputs and outputs.
    - Initialize metadata file.
  - Delete Scene:
    - Remove cached inputs and outputs for the selected scene.
- Source type:
  - Radio buttons for:
    - Video file input.
    - Image folder input.
- Source paths:
  - File entry and browse button for the video path.
  - Folder entry and browse button for the image folder path.
- Frame parameters:
  - Candidate frames input, default 200.
  - Target frames input, default 40.
- Actions:
  - Extract Frames:
    - Copies or links original source into cache/inputs/<scene_id>/source.
    - Runs frame extraction using ffmpeg-based raw video decoding or direct image loading for folder input.
    - Stores frames in frames_all.
    - Computes quality scores for frames using a chosen method such as Laplacian-based sharpness.
  - Reuse Cached Frames:
    - Reads frames_all and frames_selected from disk.
    - Reconstructs the selection state in the UI.
- Frame selection grid:
  - Scrollable list with:
    - Thumbnail for each candidate frame.
    - Checkbox or toggle for selection.
  - Buttons:
    - Select All.
    - Select None.
    - Auto-Select Best N:
      - Uses quality scores to select the highest-ranked frames up to target_count.
    - Save Choice:
      - Copies selected frames to frames_selected.
      - Updates metadata with chosen frames and counts.

Training tab:

- Scene selection:
  - Scene dropdown linked to the global Scene Registry.
  - Status indicator for inputs (e.g., `Frames selected: 40`).
- Section for camera poses (SfM):
  - Text entry and browse button for the COLMAP binary path (GLOMAP integration is deferred to TODO-7).
  - Buttons:
    - Run COLMAP + Train:
      - Validates that frames_selected exists and contains images.
      - Invokes COLMAP feature extraction, matching, mapper, and model conversion to PLY.
      - Exports a text model (cameras.txt/images.txt) for training to consume.
      - Immediately starts training after COLMAP completes.
      - Writes outputs and logs under cache/outputs/<scene_id>/sfm (logs live under sfm/logs).
  - Status label reflecting SfM progress and completion.
- Section for splat training:
  - Controls:
    - Device selector (must be CUDA; CPU training is not supported).
    - Iterations count (default 3000) and snapshot interval (default 7000).
    - Max points per checkpoint (0 keeps all points) and export format (ply or splat).
    - Image downscale (default 4) and batch size.
    - SH degree with a scheduling interval, init scale, min/max scale, opacity bias, random background toggle.
    - Learning rates for means, scales, opacities, and SH coefficients.
  - Training starts automatically after COLMAP via the combined button above; a live log pane streams logger output.
  - Checkpoints land in cache/outputs/<scene_id>/splats as both .ply and .splat.
- Live preview:
  - OpenGL preview is a future TODO; the current tab focuses on correct SfM + gsplat training and live logs.

Exports tab:

- Scene selection:
  - Scene dropdown and a Refresh button.
- Checkpoint list:
  - List of `.ply` files under cache/outputs/<scene_id>/splats.
  - Sorted by iteration or filename.
- Splat preview:
  - Read-only OpenGL viewport rendering the selected checkpoint.
- Export controls:
  - Export folder entry and browse button.
  - Buttons:
    - Export Selected .ply:
      - Copies the selected checkpoint to the chosen export folder.
    - Export turntable.mp4:
      - Renders a camera orbit sequence over the loaded splat.
      - Produces an mp4 file stored under cache/outputs/<scene_id>/renders and optionally in the export folder.


robot EXECUTION RULES INSIDE README
-------------------------------------

This section is targeted directly at robots and aligns with llms.txt.

No environment variables:

- The robot must:
  - Avoid reading environment variables for configuration or toggles.
  - Avoid writing environment variables.
  - Use only explicit configuration sources such as:
    - Files like settings.json.
    - Command-line parameters.
    - Direct human input gathered through prompts or UI.

No mocks or fallbacks:

- The robot must:
  - Avoid crafting mock backends.
  - Avoid stubbed code paths with fake outputs.
  - Implement only real, explicit behavior.
- Inability to access required tooling must:
  - Surface as clear, hard failure.
  - Include guidance to the human about how to install or configure real tools.

Imports:

- Imports must:
  - Appear as plain import statements or from-import statements.
  - Not be suppressed through try/except wrappers, except in extremely specific, documented optional feature contexts.
- Any optional feature path must:
  - Use a clearly named configuration switch.
  - Explain behavior with the feature enabled or disabled.

Import validation:

- For each new or modified module, the robot must instruct the human to run:
  - python -m py_compile path/to/module.py
  - python -c "import module_name"

Logging for background loops:

- Each background loop must:
  - Log loop start with identifiers and summary of intent.
  - Log each iteration with:
    - Iteration index.
    - Relevant counts such as processed frames or samples.
    - Metric updates such as loss values or PSNR.
  - Log success or failure for operations in the iteration.
  - Log loop termination with counts and final metrics.

No truncation of stored text:

- Persistent storage:
  - Must write full content to files.
  - Must not truncate or elide text data.
- UI components:
  - May abbreviate logs or text for readability, but:
    - The underlying stored files must retain complete content.

Human-in-loop testing:

- For each task:
  - Provide a runnable command or test for the human.
  - Explicitly request that the human executes the command.
  - Ask for console output or logs from the human.
  - Adjust code or configuration until all reported issues are addressed.

Runnable requirement for TODOs:

- Each TODO entry must:
  - Describe at least one exact command or test path to run.
  - Include expectations for visible or measurable outcomes.
- If a TODO lacks such a path:
  - The robot must rewrite the TODO to include it before further work proceeds.


ACTIVE WORK SECTION
-------------------

This section describes the expected general workflow for a robot working inside the repository.

Active Work (single entry for robots):
- None (idle; choose the next TODO before starting new work).

Workflow for robot operations:

- Task selection:
  - Choose a TODO entry that yields a concrete runnable artifact.
- Active Work update:
  - Insert the Task ID and a short description into this Active Work section.
- Implementation:
  - Modify files according to the TODO description.
  - Use only real behavior and real tools.
  - Keep imports unsuppressed and visible.
- Testing:
  - Provide commands for syntax tests, import tests, and integration tests.
  - Instruct the human to run these commands.
  - Collect logs and error outputs from the human.
  - Fix all issues they report.
- Documentation:
  - Update the Project Structure, Components, and TODO sections.
  - Ensure new paths, modules, and behaviors are documented.
- Conceptual commit messages:
  - The robot’s reasoning should align with messages such as:
    - Checkpoint: Starting [Task ID]
    - feat: Complete [Task ID]: [description]


TODO AND ROADMAP
----------------

This TODO section is designed for robots. Each entry must yield a runnable artifact and a clear human-run step.

TODO-0 (completed): Global concepts and conventions

- Implement SceneId helper:
  - Small utility for validation and normalization of scene identifiers.
  - Runnable step:
    - python -m py_compile nullsplats/backend/io_cache.py
    - python -c "from nullsplats.backend.io_cache import SceneId"
- Implement io_cache.ScenePaths:
  - Central place for computing all paths for a given scene.
  - Runnable step:
    - python -c "from nullsplats.backend.io_cache import ScenePaths; print(ScenePaths('test_scene'))"
- Implement io_cache.ensure_scene_dirs(scene_id):
  - Creates directories for inputs and outputs for the scene.
  - Runnable step:
    - python -c "from nullsplats.backend.io_cache import ensure_scene_dirs; ensure_scene_dirs('test_scene')"
- Implement metadata helpers:
  - load_metadata(scene_id) and save_metadata(scene_id, data).
  - Runnable step:
    - python -c "from nullsplats.backend.io_cache import ensure_scene_dirs, save_metadata, load_metadata; ensure_scene_dirs('test_scene'); save_metadata('test_scene', {'scene_id':'test_scene'}); print(load_metadata('test_scene'))"
- Implement app_state.SceneRegistry:
  - Scans cache/inputs and cache/outputs for scenes.
  - Flags:
    - has_inputs
    - has_sfm
    - has_splats
    - has_renders
  - Runnable step:
    - python -c "from nullsplats.app_state import SceneRegistry; reg = SceneRegistry('cache'); print(reg.list_scenes())"
- Validation tests:
  - Runnable step:
    - python -m unittest discover -s tests

TODO-1 (completed): App skeleton and threading model

- Root UI creation:
  - Implement nullsplats/ui/root.py with:
    - Tk root.
    - Menubar with File, View, Help.
    - Notebook with Inputs, Training, Exports tabs.
    - Status bar.
  - Runnable step:
    - python main.py
- Global state wiring:
  - Implement AppState with:
    - current_scene_id
    - scene_registry
    - config
  - Runnable step:
    - python -c "from nullsplats.app_state import AppState; app = AppState()"
- Background task helper:
  - Implement util.threading.run_in_background:
    - Launches background work and uses Tk callbacks for UI updates.
    - Emits logs for loop-like behavior.
  - Runnable step:
    - python -c "from nullsplats.util.threading import run_in_background; print('thread helper import ok')"

TODO-2 (completed): Inputs tab

- Scene creation flow:
  - Implement New Scene dialog and directory creation via io_cache.
  - Runnable step:
    - python main.py
    - Human confirms ability to create a new scene and see it reflected in the UI.
- Frame extraction:
  - Implement video_frames.extract_frames for video and image folder modes using moviepy and standard library.
  - Runnable step:
    - python -c "from nullsplats.backend.video_frames import extract_frames; print('video_frames import ok')"
    - Human runs the Inputs tab extraction path on a real video.
- Frame scoring and selection:
  - Implement a simple sharpness metric and Auto-Select Best N.
  - Runnable step:
    - python main.py
    - Human observes automatic selection and verifies saved frames_selected and metadata.
- Frame selection grid:
  - Implement scrollable grid with thumbnails and checkboxes.
  - Runnable step:
    - python main.py
    - Human interacts with the grid and saves choices.

TODO-3 (completed): Training tab and SfM

- COLMAP integration (COLMAP-only path):
  - Implement sfm_pipeline.run_sfm with real subprocess calls and log streaming using COLMAP binaries.
  - Runnable step:
    - python -c "from nullsplats.backend.sfm_pipeline import run_sfm; print('sfm_pipeline import ok')"
    - Human triggers SfM in the UI using a real dataset and inspects sfm/logs to confirm COLMAP writes cache/outputs/<scene_id>/sfm/sparse.
- Training configuration:
  - Implement SplatTrainingConfig for iterations and snapshot interval.
  - Runnable step:
    - python -c "from nullsplats.backend.splat_train import SplatTrainingConfig; print('config import ok')"
- Training loop:
  - Implement splat_train.train_scene with:
    - Background execution.
    - Periodic checkpoint writing.
    - Metrics logging.
  - Runnable step:
    - python main.py
    - Human runs training for a scene and checks cache/outputs/<scene_id>/splats.

Notes:
- COLMAP feature extraction and matching run successfully on real scenes; GLOMAP is deferred to TODO-7.
- Training now mirrors tools/gsplat_examples/simple_trainer defaults (3k iterations, save every 7k, downscale 4, log-scale export) with densify/prune via DefaultStrategy and parity in SH scheduling, LR scaling/decay, and appearance/pose options.

TODO-4: OpenGL preview and Exports tab

- OpenGL canvas:
  - Implement gl_canvas with orbit, pan, and zoom.
  - Runnable step:
    - python -c "from nullsplats.ui.gl_canvas import GLCanvas; print('GLCanvas import ok')"
- Live training preview:
  - Implement polling of latest `.ply` file and redraw in the Training tab.
  - Runnable step:
    - python main.py
    - Human runs a short training session and observes updates.
- Exports tab:
  - Implement checkpoint list, viewer, and export actions.
  - Runnable step:
    - python main.py
    - Human selects a checkpoint and exports `.ply` and a turntable render.

TODO-5: Configuration, settings, and logging

- Settings management:
  - Implement util.config.Settings with load and save to settings.json.
  - Runnable step:
    - python -c "from nullsplats.util.config import Settings; s = Settings.load_or_default(); print(s)"
- Settings dialog:
  - Implement a dialog for paths and defaults accessible from the menu.
  - Runnable step:
    - python main.py
    - Human opens Settings and adjusts values.
- Logging:
  - Logging helper exists (util.logging.setup_logging writes to logs/app.log and console).
  - Ensure background loops use this helper.
  - Runnable step:
    - python main.py
    - Human triggers a background operation and inspects logs/app.log.

TODO-6: Tests and documentation

- Integration tests:
  - Add tests under tests/ for:
    - ScenePaths.
    - metadata helpers.
    - basic frame extraction and selection on small sample data.
  - Runnable step:
    - python -m pytest tests or equivalent command defined in the repo.
- Developer quickstart:
  - Create docs/dev_quickstart.md explaining:
    - Virtual environment setup.
    - Required real dependencies.
    - Core commands to run the app and tests.
  - Runnable step:
    - Human reads the quickstart and confirms that instructions match behavior.
- Contributing guidelines:
  - Add CONTRIBUTING.md with:
    - Coding style rules.
    - Expectation to follow llms.txt and this README.
  - Runnable step:
    - None at runtime, but human review ensures consistency.

TODO-7: GLOMAP integration and cuDSS alignment

- GLOMAP mapper (requires working cuDSS-backed glomap.exe):
  - Ensure glomap mapper runs end-to-end on a real scene and writes COLMAP-format outputs under cache/outputs/<scene_id>/sfm/glomap.
  - Runnable step:
    - glomap.exe mapper --database_path cache/outputs/<scene_id>/sfm/database.db --image_path cache/inputs/<scene_id>/frames_selected --output_path cache/outputs/<scene_id>/sfm/glomap --log_to_stderr 1 --log_level 0
    - Human confirms cameras.txt, images.txt, and points3D.txt (or binary equivalents) exist in the output folder and reports success.

End of README.
