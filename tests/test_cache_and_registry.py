from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nullsplats.backend.io_cache import (
    SceneId,
    ScenePaths,
    ensure_scene_dirs,
    load_metadata,
    save_metadata,
)
from nullsplats.app_state import SceneRegistry


class SceneCacheTests(unittest.TestCase):
    def test_scene_paths_and_dir_creation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache_root"
            paths = ensure_scene_dirs("demo_scene", cache_root=cache_root)
            expected_dirs = list(paths.iter_required_dirs())
            for directory in expected_dirs:
                self.assertTrue(directory.exists() and directory.is_dir())
            self.assertEqual(
                paths.metadata_path,
                cache_root / "inputs" / "demo_scene" / "metadata.json",
            )

    def test_metadata_round_trip(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache_root"
            data = {"scene_id": "demo_scene", "count": 3}
            save_metadata("demo_scene", data, cache_root=cache_root)
            loaded = load_metadata(SceneId("demo_scene"), cache_root=cache_root)
            self.assertEqual(loaded, data)


class SceneRegistryTests(unittest.TestCase):
    def test_scene_registry_flags(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache_root"
            paths = ensure_scene_dirs("demo_scene", cache_root=cache_root)

            # Inputs present
            sample_frame = paths.frames_all_dir / "frame_0001.png"
            sample_frame.write_text("png-bytes-placeholder", encoding="utf-8")
            paths.metadata_path.write_text('{"scene_id": "demo_scene"}', encoding="utf-8")

            # SfM, splats, and renders markers
            (paths.sfm_dir / "database.db").write_text("db", encoding="utf-8")
            (paths.splats_dir / "iter_00000.ply").write_text("ply", encoding="utf-8")
            (paths.renders_dir / "orbit.mp4").write_text("mp4", encoding="utf-8")

            registry = SceneRegistry(cache_root=cache_root)
            scenes = registry.list_scenes()
            self.assertEqual(len(scenes), 1)
            status = scenes[0]
            self.assertEqual(status.scene_id, SceneId("demo_scene"))
            self.assertTrue(status.has_inputs)
            self.assertTrue(status.has_sfm)
            self.assertTrue(status.has_splats)
            self.assertTrue(status.has_renders)


if __name__ == "__main__":
    unittest.main()
