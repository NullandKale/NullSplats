"""Test runner for extraction + training with wizard-like defaults."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import shutil
import time
import sys
import textwrap
import traceback
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wizard-style extraction tests with UI (default) or headless backend runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python test.py
              python test.py --input assets/input.mp4 --targets 50,10,5,1
              python test.py --run-case --target 10 --input assets/input.mp4
            """
        ),
    )
    parser.add_argument("--input", default="assets/input.mp4", help="Video file to extract frames from.")
    parser.add_argument("--targets", default="50,10,5,1", help="Comma-separated target frame counts.")
    parser.add_argument("--cache-root", default="cache", help="Cache root for outputs.")
    parser.add_argument("--scene-prefix", default="wizard_test", help="Scene id prefix.")
    parser.add_argument("--candidate", type=int, default=100, help="Candidate frame count.")
    parser.add_argument("--preset", default="low", help="Training preset for gsplat/DA3 (low|medium|high).")
    parser.add_argument("--gsplat-iterations", type=int, default=0, help="Override gsplat iterations for tests.")
    parser.add_argument("--colmap-matcher", default="exhaustive", help="COLMAP matcher for gsplat runs.")
    parser.add_argument("--colmap-camera-model", default="PINHOLE", help="COLMAP camera model for gsplat runs.")
    parser.add_argument("--device", default="", help="Device override (gsplat: cuda:0, DA3: cuda, SHARP: default).")
    parser.add_argument("--da3-pretrained", default="depth-anything/DA3NESTED-GIANT-LARGE", help="DA3 pretrained id.")
    parser.add_argument("--center-image", default="assets/input.png", help="Image to force into 1/5 frame tests.")
    parser.add_argument("--headless", action="store_true", help="Run backend-only tests without the UI.")
    parser.add_argument("--test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-case", action="store_true", help="Run a single case inside this process.")
    parser.add_argument("--target", type=int, default=0, help="Target frame count for --run-case.")
    return parser.parse_args()


def _setup_test_logging() -> logging.Logger:
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "test.log"
    logger = logging.getLogger("nullsplats.test")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Test log initialized at %s", log_path)
    return logger


def _log(logger: logging.Logger, message: str) -> None:
    logger.info(message)
    print(message)


def _log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    logger.exception("%s: %s", message, exc)
    print(f"{message}: {exc}")


def _attach_ui_exception_logger(root, logger: logging.Logger) -> None:
    def _report_callback_exception(exc, val, tb) -> None:
        formatted = "".join(traceback.format_exception(exc, val, tb))
        logger.error("UI exception:\n%s", formatted)
        print(formatted)

    try:
        root.report_callback_exception = _report_callback_exception
    except Exception:
        pass


def _clear_scene_cache(cache_root: Path, scene_id: str) -> None:
    inputs_root = cache_root / "inputs" / scene_id
    outputs_root = cache_root / "outputs" / scene_id
    if inputs_root.exists():
        shutil.rmtree(inputs_root, ignore_errors=True)
    if outputs_root.exists():
        shutil.rmtree(outputs_root, ignore_errors=True)


def _build_scene_name(target: int, backend: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M")
    return f"{target}_splat_{backend}_{stamp}"


def _ensure_available_frames(scene_id: str, cache_root: Path, logger: logging.Logger) -> list[str]:
    from nullsplats.backend.io_cache import ScenePaths
    from nullsplats.backend.video_frames import load_metadata, save_metadata

    paths = ScenePaths(scene_id, cache_root=cache_root)
    try:
        metadata = load_metadata(scene_id, cache_root=cache_root)
    except FileNotFoundError:
        metadata = {}
    available = list(metadata.get("available_frames", []))
    try:
        frames = sorted([p.name for p in paths.frames_all_dir.iterdir() if p.is_file()])
    except Exception:
        frames = []
    if frames and len(frames) != len(available):
        metadata["available_frames"] = frames
        save_metadata(scene_id, metadata, cache_root=cache_root)
        _log(logger, f"Rebuilt metadata available_frames scene={scene_id} count={len(frames)}")
        return frames
    return available


def _list_frames_all(scene_id: str, cache_root: Path) -> list[str]:
    from nullsplats.backend.io_cache import ScenePaths

    paths = ScenePaths(scene_id, cache_root=cache_root)
    try:
        return sorted([p.name for p in paths.frames_all_dir.iterdir() if p.is_file()])
    except Exception:
        return []


def _infer_profile(selected_count: int) -> dict[str, Any]:
    if selected_count <= 1:
        return {
            "profile": "single",
            "backend": "sharp",
            "colmap": False,
            "resize": False,
        }
    if selected_count <= 15:
        return {
            "profile": "few",
            "backend": "depth_anything_3",
            "colmap": False,
            "resize": False,
        }
    return {
        "profile": "many",
        "backend": "gsplat",
        "colmap": True,
        "resize": True,
    }


def _build_gsplat_config(preset: str, device: str, override_iters: int) -> "SplatTrainingConfig":
    from nullsplats.backend.splat_train_config import SplatTrainingConfig

    preset_key = preset.strip().lower()
    if preset_key == "high":
        iterations = 30_000
        max_points = 2_500_000
        prune_opacity = 0.01
        prune_scale = 0.12
    elif preset_key == "medium":
        iterations = 12_000
        max_points = 1_000_000
        prune_opacity = 0.015
        prune_scale = 0.15
    else:
        iterations = 3000
        max_points = 500_000
        prune_opacity = 0.02
        prune_scale = 0.2

    if override_iters and override_iters > 0:
        iterations = int(override_iters)
    snapshot_interval = max(1, iterations // 5)
    return SplatTrainingConfig(
        iterations=iterations,
        snapshot_interval=snapshot_interval,
        max_points=max_points,
        densify_max_points=max_points,
        prune_opacity_threshold=prune_opacity,
        prune_scale_threshold=prune_scale,
        image_downscale=1,
        device=device or "cuda:0",
    )


def _build_da3_config(preset: str, device: str, pretrained_id: str) -> dict[str, Any]:
    preset_key = preset.strip().lower()
    if preset_key == "high":
        process_res = 720
        process_method = "lower_bound_resize"
        max_views = 50
    elif preset_key == "medium":
        process_res = 504
        process_method = "lower_bound_resize"
        max_views = 20
    else:
        process_res = 504
        process_method = "upper_bound_resize"
        max_views = 10
    return {
        "pretrained_id": pretrained_id,
        "device": device or "cuda",
        "use_input_resolution": False,
        "process_res": process_res,
        "process_res_method": process_method,
        "align_to_input_ext_scale": True,
        "infer_gs": True,
        "use_ray_pose": False,
        "ref_view_strategy": "saddle_balanced",
        "gs_views_interval": 1,
        "view_stride": 1,
        "max_views": max_views,
    }


def _build_sharp_config(device: str, use_colmap: bool) -> dict[str, Any]:
    return {
        "device": device or "default",
        "intrinsics_source": "colmap" if use_colmap else "exif",
        "image_index": 0,
        "use_compile": True,
    }


def _ensure_center_image(
    scene_id: str,
    cache_root: Path,
    center_image: Path,
) -> str | None:
    if not center_image.exists():
        print(f"Center image not found; skipping: {center_image}")
        return None
    from nullsplats.backend.io_cache import ScenePaths
    from nullsplats.backend.video_frames import load_metadata, save_metadata

    center_name = "center_input.png"
    paths = ScenePaths(scene_id, cache_root=cache_root)
    paths.frames_all_dir.mkdir(parents=True, exist_ok=True)
    target_path = paths.frames_all_dir / center_name
    target_path.write_bytes(center_image.read_bytes())

    try:
        metadata = load_metadata(scene_id, cache_root=cache_root)
    except FileNotFoundError:
        metadata = {}
    available = list(metadata.get("available_frames", []))
    if not available:
        try:
            available = sorted([p.name for p in paths.frames_all_dir.iterdir() if p.is_file()])
        except Exception:
            available = []
    if center_name not in available:
        available.append(center_name)
        metadata["available_frames"] = available
    scores = list(metadata.get("frame_scores", []))
    if not any(item.get("file") == center_name for item in scores if isinstance(item, dict)):
        scores.append(
            {
                "file": center_name,
                "score": 0.0,
                "sharpness": None,
                "variance": None,
                "fingerprint": None,
            }
        )
        metadata["frame_scores"] = scores
    save_metadata(scene_id, metadata, cache_root=cache_root)
    return center_name


class _UiTestRunner:
    def __init__(self, *, root, app_state, tabs: dict, args: argparse.Namespace) -> None:
        self.root = root
        self.app_state = app_state
        self.tabs = tabs
        self.args = args
        self.logger = _setup_test_logging()
        self._disable_autosave()
        self.targets = [int(item.strip()) for item in args.targets.split(",") if item.strip()]
        self.total_cases = len(self.targets)
        self.failures = 0
        self.current_target: int | None = None
        self.scene_id: str | None = None

    def start(self) -> None:
        self.root.after(1000, self._start_next_case)

    def _start_next_case(self) -> None:
        if not self.targets:
            self._log_summary()
            try:
                self.root.after(1000, self.root.destroy)
            except Exception:
                pass
            return
        self.current_target = self.targets.pop(0)
        target = int(self.current_target)
        profile = _infer_profile(target)
        scene_id = _build_scene_name(target, profile["backend"])
        _clear_scene_cache(Path(self.args.cache_root), scene_id)
        _log(self.logger, f"UI case start target={target} scene={scene_id} cache_cleared=True")
        inputs_tab = self.tabs["inputs"]
        notebook = self.tabs["notebook"]
        notebook.select(0)
        if inputs_tab.scene_entry is not None:
            try:
                inputs_tab.scene_entry.delete(0, "end")
                inputs_tab.scene_entry.insert(0, scene_id)
            except Exception:
                pass

        input_path = str(Path(self.args.input).expanduser())
        inputs_tab._apply_input_selection("Video file", input_path)
        inputs_tab.candidate_var.set(int(self.args.candidate))
        inputs_tab.target_var.set(target)
        _log(
            self.logger,
            f"Inputs vars scene={scene_id} candidate={inputs_tab.candidate_var.get()} target={inputs_tab.target_var.get()}",
        )
        profile = _infer_profile(target)
        if profile["resize"]:
            inputs_tab.training_resolution_var.set(1080)
        else:
            inputs_tab.training_resolution_var.set(0)
        inputs_tab.training_resample_var.set("lanczos")
        inputs_tab._on_resolution_change()
        inputs_tab._start_extraction()
        self.root.after(500, self._poll_extract)

    def _disable_autosave(self) -> None:
        inputs_tab = self.tabs.get("inputs")
        if inputs_tab is None:
            return
        if getattr(inputs_tab, "_test_autosave_disabled", False):
            return
        try:
            inputs_tab._original_schedule_autosave = inputs_tab._schedule_autosave
            inputs_tab._schedule_autosave = lambda *_args, **_kwargs: None
            inputs_tab._test_autosave_disabled = True
            _log(self.logger, "Autosave disabled for UI test run.")
        except Exception:
            pass

    def _poll_extract(self) -> None:
        inputs_tab = self.tabs["inputs"]
        if getattr(inputs_tab, "_extracting", False):
            self.root.after(500, self._poll_extract)
            return
        if inputs_tab.current_result is None:
            self._record_failure("Extraction did not produce results.")
            self.root.after(500, self._start_next_case)
            return
        self.scene_id = str(self.app_state.current_scene_id)
        result = inputs_tab.current_result
        if str(result.scene_id) != self.scene_id:
            try:
                result = self.app_state.scene_manager.load_cached_frames(self.scene_id)
                inputs_tab._render_result(result)
            except Exception as exc:
                _log_exception(self.logger, "Result reload failed", exc)
        _log(
            self.logger,
            "Extraction done scene={scene} target={target} available={available} selected={selected}".format(
                scene=self.scene_id,
                target=self.current_target,
                available=len(result.available_frames),
                selected=len(result.selected_frames),
            ),
        )
        try:
            if inputs_tab._autosave_job is not None:
                try:
                    inputs_tab.frame.after_cancel(inputs_tab._autosave_job)
                except Exception:
                    pass
                inputs_tab._autosave_job = None
            selected, target_px, resample = self._build_selected_frames()
            _log(
                self.logger,
                "Selection plan scene={scene} target={target} selected={count} target_px={target_px} resample={resample}".format(
                    scene=self.scene_id,
                    target=self.current_target,
                    count=len(selected),
                    target_px=target_px,
                    resample=resample,
                ),
            )
            result, summary = self.app_state.scene_manager.save_selection(
                self.scene_id,
                selected,
                target_px=target_px,
                resample=resample,
            )
            inputs_tab._render_result(result)
            inputs_tab._dirty_selection = False
            if inputs_tab._autosave_job is not None:
                try:
                    inputs_tab.frame.after_cancel(inputs_tab._autosave_job)
                except Exception:
                    pass
                inputs_tab._autosave_job = None
            self._log_frames_selected_state()
            _log(
                self.logger,
                "Selection saved scene={scene} target={target} processed={processed} skipped={skipped} deleted={deleted}".format(
                    scene=self.scene_id,
                    target=self.current_target,
                    processed=summary.processed,
                    skipped=summary.skipped,
                    deleted=summary.deleted,
                ),
            )
        except Exception as exc:
            _log_exception(self.logger, "Selection save failed", exc)
            self._record_failure("Selection save failed.")
        self._maybe_run_colmap()

    def _maybe_run_colmap(self) -> None:
        target = int(self.current_target or 0)
        profile = _infer_profile(target)
        if not profile["colmap"]:
            _log(self.logger, f"Skipping COLMAP scene={self.scene_id} target={self.current_target}")
            self._run_training()
            return
        notebook = self.tabs["notebook"]
        colmap_tab = self.tabs["colmap"]
        notebook.select(1)
        colmap_tab.matcher_var.set(self.args.colmap_matcher)
        colmap_tab.camera_model_var.set(self.args.colmap_camera_model)
        colmap_tab.run_sfm()
        _log(self.logger, f"COLMAP started scene={self.scene_id} target={self.current_target}")
        self.root.after(1000, self._poll_colmap)

    def _poll_colmap(self) -> None:
        colmap_tab = self.tabs["colmap"]
        if colmap_tab.is_working():
            self.root.after(1000, self._poll_colmap)
            return
        _log(self.logger, f"COLMAP finished scene={self.scene_id} target={self.current_target}")
        self._run_training()

    def _run_training(self) -> None:
        target = int(self.current_target or 0)
        profile = _infer_profile(target)
        notebook = self.tabs["notebook"]
        training_tab = self.tabs["training"]
        notebook.select(2)
        if profile["backend"] == "gsplat":
            training_tab.training_method_var.set("gsplat")
            if self.args.gsplat_iterations:
                training_tab.iterations_var.set(int(self.args.gsplat_iterations))
                training_tab.snapshot_var.set(max(1, int(self.args.gsplat_iterations) // 5))
        elif profile["backend"] == "depth_anything_3":
            training_tab.training_method_var.set("depth_anything_3")
            training_tab.da3_allow_unposed_var.set(True)
        else:
            training_tab.training_method_var.set("sharp")
            training_tab.sharp_intrinsics_source_var.set("exif")
        try:
            training_tab._apply_trainer_capabilities()
        except Exception:
            pass
        training_tab.apply_training_preset("low")
        training_tab.run_training()
        self._log_frames_selected_state()
        _log(self.logger, f"Training started scene={self.scene_id} target={self.current_target} backend={profile['backend']}")
        self.root.after(1000, self._poll_training)

    def _poll_training(self) -> None:
        training_tab = self.tabs["training"]
        if training_tab.is_working():
            self.root.after(1000, self._poll_training)
            return
        self._check_training_output()
        if self._wait_for_ply(timeout_s=20):
            notebook = self.tabs["notebook"]
            try:
                notebook.select(3)
            except Exception as exc:
                _log_exception(self.logger, "Exports tab select failed", exc)
        _log(self.logger, f"Training finished scene={self.scene_id} target={self.current_target}")
        self.root.after(1000, self._start_next_case)

    def _check_training_output(self) -> None:
        if self.scene_id is None:
            self._record_failure("Missing scene id after training.")
            return
        splats_dir = Path(self.args.cache_root) / "outputs" / self.scene_id / "splats"
        has_ply = bool(list(splats_dir.glob("*.ply")))
        if not has_ply:
            status_text = ""
            try:
                status_text = self.tabs["training"].status_var.get()
            except Exception:
                status_text = ""
            extra = f" status={status_text}" if status_text else ""
            self._record_failure(f"No .ply outputs found in {splats_dir}{extra}")
            self._log_frames_selected_state()

    def _wait_for_ply(self, timeout_s: int = 20) -> bool:
        if self.scene_id is None:
            return False
        splats_dir = Path(self.args.cache_root) / "outputs" / self.scene_id / "splats"
        deadline = time.time() + max(1, timeout_s)
        while time.time() < deadline:
            if list(splats_dir.glob("*.ply")):
                return True
            time.sleep(1.0)
        return False

    def _log_frames_selected_state(self) -> None:
        if self.scene_id is None:
            return
        cache_root = Path(self.args.cache_root)
        frames_dir = cache_root / "inputs" / self.scene_id / "frames_selected"
        try:
            files = sorted([p.name for p in frames_dir.iterdir() if p.is_file()])
        except Exception:
            files = []
        _log(
            self.logger,
            "frames_selected scene={scene} exists={exists} count={count} files={files}".format(
                scene=self.scene_id,
                exists=frames_dir.exists(),
                count=len(files),
                files=",".join(files[:8]) + ("..." if len(files) > 8 else ""),
            ),
        )
        try:
            from nullsplats.backend.video_frames import load_metadata

            metadata = load_metadata(self.scene_id, cache_root=cache_root)
            _log(
                self.logger,
                "metadata scene={scene} available={avail} selected={sel} source={source}".format(
                    scene=self.scene_id,
                    avail=len(metadata.get("available_frames", [])),
                    sel=len(metadata.get("selected_frames", [])),
                    source=metadata.get("source_path", ""),
                ),
            )
        except Exception as exc:
            _log_exception(self.logger, "Metadata read failed", exc)

    def _build_selected_frames(self) -> tuple[list[str], int, str]:
        if self.scene_id is None:
            return [], 0, "lanczos"
        target = int(self.current_target or 0)
        inputs_tab = self.tabs["inputs"]
        center_image = Path(self.args.center_image).expanduser()
        center_name = None
        if target in {1, 5}:
            center_name = _ensure_center_image(self.scene_id, Path(self.args.cache_root), center_image)
        cache_root = Path(self.args.cache_root)
        available = _list_frames_all(self.scene_id, cache_root)
        if not available:
            available = _ensure_available_frames(self.scene_id, cache_root, self.logger)
        selected = []
        if available:
            selected = available[:target] if target > 0 else list(available)
        if center_name:
            if target == 1:
                selected = [center_name]
            else:
                remainder = [name for name in selected if name != center_name]
                selected = [center_name] + remainder[: max(0, target - 1)]
                if len(selected) < target and available:
                    for name in available:
                        if name in selected:
                            continue
                        selected.append(name)
                        if len(selected) >= target:
                            break

        if inputs_tab.current_result:
            available = inputs_tab.current_result.available_frames
            inputs_tab.selection_state = {name: (name in selected) for name in available}
            inputs_tab._dirty_selection = True
            inputs_tab._update_visible_tiles(force=True)
            inputs_tab._sync_status()
        target_px = 1080 if _infer_profile(target)["resize"] else 0
        return selected, target_px, "lanczos"

    def _record_failure(self, message: str) -> None:
        self.failures += 1
        _log(self.logger, f"[error] {message}")

    def _log_summary(self) -> None:
        passed = self.total_cases - self.failures
        status = "PASS" if self.failures == 0 else "FAIL"
        _log(
            self.logger,
            "UI summary status={status} total={total} passed={passed} failed={failed}".format(
                status=status,
                total=self.total_cases,
                passed=passed,
                failed=self.failures,
            ),
        )


def _run_case(
    input_path: Path,
    target_count: int,
    candidate_count: int,
    cache_root: Path,
    scene_prefix: str,
    preset: str,
    colmap_matcher: str,
    colmap_camera_model: str,
    device: str,
    da3_pretrained: str,
    center_image: Path,
    gsplat_iterations: int,
) -> int:
    if target_count <= 0:
        raise ValueError("target_count must be positive.")
    if candidate_count < target_count:
        candidate_count = target_count

    from nullsplats.backend.scene_manager import SceneManager
    from nullsplats.backend.sfm_pipeline import SfmConfig, run_sfm
    from nullsplats.backend.splat_backends.dispatch import train_with_trainer
    from nullsplats.backend.video_frames import extract_frames  # local import for subprocess mode

    profile = _infer_profile(target_count)
    scene_id = _build_scene_name(target_count, profile["backend"])
    result = extract_frames(
        scene_id,
        input_path,
        source_type="video",
        candidate_count=candidate_count,
        target_count=target_count,
        cache_root=cache_root,
    )

    available = len(result.available_frames)
    selected = len(result.selected_frames)
    expected_selected = min(target_count, available)
    if selected != expected_selected:
        raise AssertionError(
            f"Selected frame count mismatch scene={scene_id} expected={expected_selected} got={selected}"
        )

    resize_enabled = profile["resize"]
    manager = SceneManager(cache_root=cache_root)
    target_px = 1080 if resize_enabled else 0
    resample = "lanczos"
    selected_frames = list(result.selected_frames)

    if target_count in {1, 5}:
        center_name = _ensure_center_image(scene_id, cache_root, center_image)
        if center_name:
            if target_count == 1:
                selected_frames = [center_name]
            else:
                remainder = [name for name in selected_frames if name != center_name]
                selected_frames = [center_name] + remainder[: max(0, target_count - 1)]
                if len(selected_frames) < target_count:
                    try:
                        from nullsplats.backend.video_frames import load_metadata

                        metadata = load_metadata(scene_id, cache_root=cache_root)
                        available = list(metadata.get("available_frames", []))
                        for name in available:
                            if name in selected_frames:
                                continue
                            selected_frames.append(name)
                            if len(selected_frames) >= target_count:
                                break
                    except Exception:
                        pass
    manager.save_selection(scene_id, selected_frames, target_px=target_px, resample=resample)

    frames_selected = result.paths.frames_selected_dir
    if not frames_selected.exists() or not any(frames_selected.iterdir()):
        raise AssertionError(f"No frames selected under {frames_selected}")

    colmap_enabled = profile["colmap"]
    if colmap_enabled:
        sfm_config = SfmConfig(
            colmap_path="",
            matcher=colmap_matcher,
            camera_model=colmap_camera_model,
        )
        run_sfm(scene_id, config=sfm_config, cache_root=cache_root)

    backend = profile["backend"]
    if backend == "gsplat":
        train_config = _build_gsplat_config("low", device, gsplat_iterations)
        allow_missing_colmap = False
    elif backend == "depth_anything_3":
        train_config = _build_da3_config("low", device, da3_pretrained)
        allow_missing_colmap = True
    else:
        train_config = _build_sharp_config(device, colmap_enabled)
        allow_missing_colmap = True

    training_output = train_with_trainer(
        scene_id,
        backend,
        train_config,
        cache_root=cache_root,
        allow_missing_colmap=allow_missing_colmap,
    )
    if not training_output.primary_path.exists():
        raise AssertionError(f"Training output missing: {training_output.primary_path}")

    print(
        "OK scene={scene} profile={profile} backend={backend} candidates={cand} available={avail} "
        "selected={sel} cache={cache} output={output}".format(
            scene=scene_id,
            profile=profile["profile"],
            backend=backend,
            cand=candidate_count,
            avail=available,
            sel=selected,
            cache=cache_root,
            output=training_output.primary_path,
        )
    )
    return 0


def _run_driver(args: argparse.Namespace) -> int:
    logger = _setup_test_logging()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    center_image = Path(args.center_image).expanduser()

    targets = [int(item.strip()) for item in args.targets.split(",") if item.strip()]
    if not targets:
        raise ValueError("No targets provided.")

    failures = 0
    for target in targets:
        scene_id = f"{args.scene_prefix}_{target:02d}"
        _clear_scene_cache(Path(args.cache_root), scene_id)
        _log(logger, f"Headless case start target={target} scene={scene_id} cache_cleared=True")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-case",
            "--input",
            str(input_path),
            "--target",
            str(target),
            "--cache-root",
            args.cache_root,
            "--scene-prefix",
            args.scene_prefix,
            "--preset",
            args.preset,
            "--colmap-matcher",
            args.colmap_matcher,
            "--colmap-camera-model",
            args.colmap_camera_model,
            "--device",
            args.device,
            "--da3-pretrained",
            args.da3_pretrained,
            "--gsplat-iterations",
            str(args.gsplat_iterations),
            "--center-image",
            str(center_image),
        ]
        cmd.extend(["--candidate", str(args.candidate)])
        _log(logger, f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            _log(logger, f"[error] Headless case failed target={target} code={result.returncode}")
        else:
            _log(logger, f"Headless case finished target={target}")

    if failures:
        _log(logger, f"{failures} test case(s) failed.")
        return 1
    _log(logger, "All test cases passed.")
    return 0


def main() -> int:
    args = _parse_args()
    if args.run_case:
        input_path = Path(args.input).expanduser()
        cache_root = Path(args.cache_root).expanduser()
        candidate = max(1, int(args.candidate))
        return _run_case(
            input_path,
            args.target,
            candidate,
            cache_root,
            args.scene_prefix,
            args.preset,
            args.colmap_matcher,
            args.colmap_camera_model,
            args.device,
            args.da3_pretrained,
            Path(args.center_image).expanduser(),
            args.gsplat_iterations,
        )
    if not args.headless:
        from nullsplats.app_state import AppState
        from nullsplats.ui.root import create_root

        app_state = AppState()
        root = create_root(app_state)
        tabs = getattr(root, "_nullsplats_tabs", None)
        if not tabs:
            raise RuntimeError("UI tabs not found; cannot run UI-driven tests.")
        runner = _UiTestRunner(root=root, app_state=app_state, tabs=tabs, args=args)
        _attach_ui_exception_logger(root, runner.logger)
        runner.start()
        root.mainloop()
        return 0
    return _run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
