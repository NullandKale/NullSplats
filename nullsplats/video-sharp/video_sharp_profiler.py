"""Profiling helpers for the video-to-SHARP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List


@dataclass
class FrameTiming:
    index: int
    image_path: str
    output_path: str
    decode_s: float
    load_s: float
    infer_s: float
    save_s: float
    total_s: float


class VideoSharpProfiler:
    """Collect per-frame timings and emit summary to tmp/."""

    def __init__(self, label: str, output_dir: Path | None = None) -> None:
        self.label = label
        self.output_dir = Path(output_dir or "tmp")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.perf_counter()
        self.frames: List[FrameTiming] = []

    def record(self, payload: Dict[str, Any]) -> None:
        self.frames.append(
            FrameTiming(
                index=int(payload["index"]),
                image_path=str(payload["image_path"]),
                output_path=str(payload["output_path"]),
                decode_s=float(payload.get("decode_s", 0.0)),
                load_s=float(payload["load_s"]),
                infer_s=float(payload["infer_s"]),
                save_s=float(payload["save_s"]),
                total_s=float(payload["total_s"]),
            )
        )

    def finalize(self) -> tuple[Dict[str, Any], Path, Path]:
        elapsed = time.perf_counter() - self.start_time
        summary = self._build_summary(elapsed)
        base = f"video_sharp_profile_{self.label}"
        json_path = self.output_dir / f"{base}.json"
        md_path = self.output_dir / f"{base}.md"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(self._render_markdown(summary), encoding="utf-8")
        return summary, json_path, md_path

    def _build_summary(self, elapsed: float) -> Dict[str, Any]:
        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
            values_sorted = sorted(values)
            return {
                "mean": float(statistics.mean(values)),
                "p50": float(values_sorted[int(0.5 * (len(values_sorted) - 1))]),
                "p95": float(values_sorted[int(0.95 * (len(values_sorted) - 1))]),
                "min": float(values_sorted[0]),
                "max": float(values_sorted[-1]),
            }

        decode_s = [f.decode_s for f in self.frames]
        load_s = [f.load_s for f in self.frames]
        infer_s = [f.infer_s for f in self.frames]
        save_s = [f.save_s for f in self.frames]
        total_s = [f.total_s for f in self.frames]
        return {
            "label": self.label,
            "elapsed_s": elapsed,
            "frames": len(self.frames),
            "stats": {
                "decode_s": _stats(decode_s),
                "load_s": _stats(load_s),
                "infer_s": _stats(infer_s),
                "save_s": _stats(save_s),
                "total_s": _stats(total_s),
            },
            "per_frame": [f.__dict__ for f in self.frames],
        }

    @staticmethod
    def _render_markdown(summary: Dict[str, Any]) -> str:
        stats = summary["stats"]
        return "\n".join(
            [
                "# Video SHARP Profile",
                "",
                f"- label: {summary['label']}",
                f"- frames: {summary['frames']}",
                f"- elapsed_s: {summary['elapsed_s']:.3f}",
                "",
                "## Timing stats (seconds)",
                "",
                f"- decode mean/p50/p95: {stats['decode_s']['mean']:.3f} / {stats['decode_s']['p50']:.3f} / {stats['decode_s']['p95']:.3f}",
                f"- load mean/p50/p95: {stats['load_s']['mean']:.3f} / {stats['load_s']['p50']:.3f} / {stats['load_s']['p95']:.3f}",
                f"- infer mean/p50/p95: {stats['infer_s']['mean']:.3f} / {stats['infer_s']['p50']:.3f} / {stats['infer_s']['p95']:.3f}",
                f"- save mean/p50/p95: {stats['save_s']['mean']:.3f} / {stats['save_s']['p50']:.3f} / {stats['save_s']['p95']:.3f}",
                f"- total mean/p50/p95: {stats['total_s']['mean']:.3f} / {stats['total_s']['p50']:.3f} / {stats['total_s']['p95']:.3f}",
                "",
            ]
        )


__all__ = ["VideoSharpProfiler"]
