"""Wizard flow and dialog helpers for InputsTab."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class InputsTabWizardMixin:
    def _start_inline_wizard(self) -> None:
        if self._wizard_running:
            return
        params = self._wizard_prompt_settings()
        if params is None:
            return
        self._wizard_running = True
        self._wizard_preset = params["preset"]
        self._wizard_colmap_matcher = params["colmap_matcher"]
        self._wizard_colmap_camera_model = params["colmap_camera_model"]
        self.source_type_var.set(params["source_type"])
        if params["source_type"] == "video":
            self.video_path_var.set(params["video_path"])
        else:
            self.image_dir_var.set(params["image_dir"])
        self.candidate_var.set(params["candidate"])
        self.target_var.set(params["target"])
        self.training_resolution_var.set(params["resolution"])
        self.training_resample_var.set(params["mode"])
        self._on_resolution_change()
        self._start_extraction()
        self.frame.after(500, self._wizard_wait_for_extract)

    def _wizard_wait_for_extract(self) -> None:
        if self._extracting:
            self.frame.after(500, self._wizard_wait_for_extract)
            return
        if self.current_result is None or not self.current_result.available_frames:
            messagebox.showerror("Wizard", "Extraction did not produce frames. Check the input and try again.", parent=self.frame.winfo_toplevel())
            self._wizard_running = False
            return
        self._persist_selection()
        self.frame.after(500, self._wizard_wait_for_save)

    def _wizard_wait_for_save(self) -> None:
        if self._saving:
            self.frame.after(500, self._wizard_wait_for_save)
            return
        if self.notebook is not None:
            try:
                self.notebook.select(1)
            except Exception:
                pass
        if self.colmap_tab is not None:
            try:
                if hasattr(self, "_wizard_colmap_matcher"):
                    self.colmap_tab.matcher_var.set(self._wizard_colmap_matcher)
                if hasattr(self, "_wizard_colmap_camera_model"):
                    self.colmap_tab.camera_model_var.set(self._wizard_colmap_camera_model)
                self.colmap_tab.run_sfm()
                self.frame.after(1000, self._wizard_wait_for_sfm)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Wizard", f"COLMAP failed: {exc}", parent=self.frame.winfo_toplevel())
                self._wizard_running = False
        else:
            self._wizard_start_training()

    def _wizard_wait_for_sfm(self) -> None:
        if self.colmap_tab is not None and self.colmap_tab.is_working():
            self.frame.after(1000, self._wizard_wait_for_sfm)
            return
        self._wizard_start_training()

    def _wizard_start_training(self) -> None:
        preset = getattr(self, "_wizard_preset", None)
        if not preset or self.training_tab is None:
            self._wizard_running = False
            return
        if self.notebook is not None:
            try:
                self.notebook.select(2)
            except Exception:
                pass
        try:
            self.training_tab.apply_training_preset(preset)
            self.training_tab.run_training()
            self.frame.after(1000, self._wizard_wait_for_training)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Wizard", f"Training failed: {exc}", parent=self.frame.winfo_toplevel())
            self._wizard_running = False

    def _wizard_wait_for_training(self) -> None:
        if self.training_tab is not None and self.training_tab.is_working():
            self.frame.after(1000, self._wizard_wait_for_training)
            return
        if self.exports_tab is not None:
            try:
                self.exports_tab._load_checkpoints()
            except Exception:
                pass
        if self.notebook is not None:
            try:
                self.notebook.select(3)
            except Exception:
                pass
        self._wizard_finish_exports()
        self._wizard_running = False

    def _wizard_prompt_settings(self) -> dict | None:
        dialog = tk.Toplevel(self.frame)
        dialog.title("Wizard: Inputs + Training preset")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        self._center_dialog(dialog)
        result: dict | None = None

        source_type = tk.StringVar(value=self.source_type_var.get())
        video_var = tk.StringVar(value=self.video_path_var.get())
        folder_var = tk.StringVar(value=self.image_dir_var.get())
        cand_var = tk.IntVar(value=self.candidate_var.get())
        target_var = tk.IntVar(value=self.target_var.get())
        res_var = tk.IntVar(value=self.training_resolution_var.get())
        mode_var = tk.StringVar(value=self.training_resample_var.get())
        preset_var = tk.StringVar(value="medium")
        matcher_var = tk.StringVar(value="exhaustive")
        camera_model_var = tk.StringVar(value="PINHOLE")

        dialog.columnconfigure(1, weight=1)
        ttk.Radiobutton(dialog, text="Video", variable=source_type, value="video").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Radiobutton(dialog, text="Image folder", variable=source_type, value="images").grid(row=0, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(dialog, text="Video file:").grid(row=1, column=0, sticky="w", padx=8)
        ttk.Entry(dialog, textvariable=video_var, width=50).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(dialog, text="Browse", command=lambda: self._wizard_browse_file(video_var)).grid(row=1, column=2, sticky="e", padx=8)
        ttk.Label(dialog, text="Image folder:").grid(row=2, column=0, sticky="w", padx=8, pady=(4, 0))
        ttk.Entry(dialog, textvariable=folder_var, width=50).grid(row=2, column=1, sticky="ew", padx=4, pady=(4, 0))
        ttk.Button(dialog, text="Browse", command=lambda: self._wizard_browse_folder(folder_var)).grid(row=2, column=2, sticky="e", padx=8, pady=(4, 0))

        row3 = ttk.Frame(dialog)
        row3.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(row3, text="Candidates").pack(side="left")
        ttk.Spinbox(row3, from_=1, to=10000, textvariable=cand_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Targets").pack(side="left")
        ttk.Spinbox(row3, from_=1, to=10000, textvariable=target_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Resolution (px, small side)").pack(side="left")
        ttk.Combobox(row3, values=[720, 1080, 2160], textvariable=res_var, state="readonly", width=10).pack(side="left", padx=(4, 6))
        ttk.Combobox(row3, values=["lanczos", "bicubic", "bilinear", "nearest"], textvariable=mode_var, state="readonly", width=10).pack(side="left")

        row4 = ttk.Frame(dialog)
        row4.grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(row4, text="Training preset").pack(side="left")
        ttk.Combobox(row4, values=["low", "medium", "high"], textvariable=preset_var, state="readonly", width=10).pack(
            side="left", padx=(4, 0)
        )

        row5 = ttk.Frame(dialog)
        row5.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(row5, text="COLMAP matcher").pack(side="left")
        ttk.Combobox(
            row5,
            values=["exhaustive", "sequential", "spatial"],
            textvariable=matcher_var,
            state="readonly",
            width=12,
        ).pack(side="left", padx=(4, 12))
        ttk.Label(row5, text="Camera model").pack(side="left")
        ttk.Combobox(
            row5,
            values=["PINHOLE", "SIMPLE_PINHOLE", "OPENCV"],
            textvariable=camera_model_var,
            state="readonly",
            width=14,
        ).pack(side="left", padx=(4, 0))

        def _ok() -> None:
            nonlocal result
            path = video_var.get().strip() if source_type.get() == "video" else folder_var.get().strip()
            if not path:
                messagebox.showerror("Wizard", "Provide a video file or image folder.", parent=dialog)
                return
            result = {
                "source_type": source_type.get(),
                "video_path": video_var.get().strip(),
                "image_dir": folder_var.get().strip(),
                "candidate": max(1, int(cand_var.get())),
                "target": max(1, int(target_var.get())),
                "resolution": max(1, int(res_var.get())),
                "mode": mode_var.get(),
                "preset": preset_var.get(),
                "colmap_matcher": matcher_var.get(),
                "colmap_camera_model": camera_model_var.get(),
            }
            dialog.destroy()

        def _cancel() -> None:
            dialog.destroy()

        btn_row = ttk.Frame(dialog)
        btn_row.grid(row=6, column=0, columnspan=3, sticky="e", padx=8, pady=(8, 8))
        ttk.Button(btn_row, text="Cancel", command=_cancel).pack(side="right", padx=(6, 0))
        ttk.Button(btn_row, text="OK", command=_ok).pack(side="right")

        dialog.wait_window()
        return result

    def _wizard_browse_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            parent=self.frame.winfo_toplevel(),
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if path:
            var.set(path)

    def _wizard_browse_folder(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(parent=self.frame.winfo_toplevel(), title="Select image folder")
        if path:
            var.set(path)

    def _center_dialog(self, dialog: tk.Toplevel) -> None:
        """Center a dialog over the main window."""
        try:
            dialog.update_idletasks()
            parent = self.frame.winfo_toplevel()
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            w = dialog.winfo_width()
            h = dialog.winfo_height()
            x = px + max(0, (pw - w) // 2)
            y = py + max(0, (ph - h) // 2)
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            return

    def _wizard_finish_exports(self) -> None:
        latest = None
        if self.exports_tab is not None and getattr(self.exports_tab, "checkpoint_paths", None):
            latest = self.exports_tab.checkpoint_paths[0] if self.exports_tab.checkpoint_paths else None
        msg = "Extraction and training complete."
        if latest:
            msg += f"\nLatest checkpoint: {latest.name}"
        if messagebox.askyesno("Wizard complete", msg + "\nOpen output folder?"):
            try:
                if latest:
                    Path(latest).parent.mkdir(parents=True, exist_ok=True)
                    import os
                    os.startfile(str(Path(latest).parent))
            except Exception:
                pass

