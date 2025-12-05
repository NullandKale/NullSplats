@echo off
rem Run core checks for NullSplats UI scaffold.
pushd "%~dp0.."

echo [1/7] Dependency check for ffmpeg, numpy, pillow, colmap, torch (CUDA)
python tests\dependency_check.py
if errorlevel 1 exit /b 1

echo [2/7] Automated integration first: real frames -> COLMAP -> training
python -m tests.integration_sfm_training
if errorlevel 1 exit /b 1

echo [3/7] Syntax check via py_compile
python -m py_compile main.py nullsplats\app_state.py nullsplats\ui\root.py nullsplats\ui\tab_inputs.py nullsplats\ui\tab_training.py nullsplats\ui\tab_exports.py nullsplats\backend\io_cache.py nullsplats\backend\video_frames.py nullsplats\backend\sfm_pipeline.py nullsplats\backend\splat_train.py nullsplats\backend\gs_utils.py nullsplats\backend\__init__.py nullsplats\util\config.py nullsplats\util\threading.py nullsplats\util\logging.py nullsplats\util\scene_id.py nullsplats\util\__init__.py nullsplats\__init__.py tests\dependency_check.py tests\integration_sfm_training.py
if errorlevel 1 exit /b 1

echo [4/7] Import verification
python -c "import nullsplats, nullsplats.app_state, nullsplats.backend, nullsplats.backend.gs_utils, nullsplats.backend.video_frames, nullsplats.backend.sfm_pipeline, nullsplats.backend.splat_train, nullsplats.ui.root, nullsplats.util.threading, nullsplats.util.config, nullsplats.util.logging, tests.dependency_check, tests.integration_sfm_training"
if errorlevel 1 exit /b 1

echo [5/7] CLI smoke tests for SfM and training config
python -c "from nullsplats.backend.sfm_pipeline import run_sfm; print('sfm_pipeline import ok'); from nullsplats.backend.splat_train import SplatTrainingConfig; print(SplatTrainingConfig())"
if errorlevel 1 exit /b 1

echo [6/7] Launching NullSplats UI for manual check (close the window to continue)
echo  - In Training tab, confirm COLMAP path, run the combined "Run COLMAP + Train" button, and watch status/log updates.
echo  - Confirm training uses a CUDA device and checkpoints land in cache\outputs\<scene>\splats using the selected single export format (.ply or .splat).
echo  - Close the UI window after verifying to let this script finish.
python main.py
if errorlevel 1 exit /b 1

echo [7/7] Running unit tests (verbose)
python -m unittest discover -s tests -v

popd
