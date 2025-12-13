"""Wizard flow to guide users from inputs through training to exports."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Callable, List

from nullsplats.app_state import AppState


class WizardStep:
    """Single wizard step definition."""

    def __init__(self, title: str, description: str, require: Callable[[], bool], on_navigate: Callable[[], None]) -> None:
        self.title = title
        self.description = description
        self.require = require
        self.on_navigate = on_navigate
        self.status = "pending"


class WizardWindow(tk.Toplevel):
    """Guided flow overlay."""

    def __init__(self, root: tk.Tk, app_state: AppState, select_tab: Callable[[int], None]) -> None:
        super().__init__(root)
        self.title("NullSplats Wizard")
        self.geometry("900x600")
        self.app_state = app_state
        self.select_tab = select_tab
        self.steps: List[WizardStep] = []
        self.current_idx = 0
        self.status_var = tk.StringVar(value="")
        self._build_steps()
        self._build_ui()
        self._refresh_status()

    def _build_steps(self) -> None:
        def require_inputs() -> bool:
            scene = self.app_state.current_scene_id
            if scene is None:
                return False
            paths = self.app_state.scene_manager.get(scene).paths
            return paths.frames_selected_dir.exists() and any(paths.frames_selected_dir.iterdir())

        def require_training() -> bool:
            scene = self.app_state.current_scene_id
            if scene is None:
                return False
            paths = self.app_state.scene_manager.get(scene).paths
            has_sfm = paths.sfm_dir.exists() and any(paths.sfm_dir.iterdir())
            has_splats = paths.splats_dir.exists() and any(paths.splats_dir.iterdir())
            return has_sfm and has_splats

        def require_exports() -> bool:
            scene = self.app_state.current_scene_id
            if scene is None:
                return False
            paths = self.app_state.scene_manager.get(scene).paths
            return paths.splats_dir.exists() and any(paths.splats_dir.iterdir())

        self.steps = [
            WizardStep(
                "Step 1: Inputs",
                "Create/select a scene, choose video or images, extract frames, and save selected/resized images.",
                require_inputs,
                lambda: self._go_tab(0),
            ),
            WizardStep(
                "Step 2: Training",
                "Run COLMAP and training for the active scene. Watch preview while training.",
                require_training,
                lambda: self._go_tab(1),
            ),
            WizardStep(
                "Step 3: Exports",
                "Preview checkpoints and export .ply or renders for the active scene.",
                require_exports,
                lambda: self._go_tab(2),
            ),
        ]

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self)
        sidebar.grid(row=0, column=0, sticky="ns")
        self.listbox = tk.Listbox(sidebar, height=10)
        for step in self.steps:
            self.listbox.insert(tk.END, step.title)
        self.listbox.bind("<<ListboxSelect>>", self._on_step_select)
        self.listbox.pack(fill="both", expand=True, padx=8, pady=8)

        main = ttk.Frame(self)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        self.title_label = ttk.Label(main, text=self.steps[0].title, font=("Segoe UI", 12, "bold"))
        self.title_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        self.desc_label = ttk.Label(main, text=self.steps[0].description, wraplength=520, justify="left")
        self.desc_label.grid(row=1, column=0, sticky="nw", padx=10, pady=(0, 10))

        controls = ttk.Frame(main)
        controls.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls.columnconfigure(1, weight=1)

        self.status_display = ttk.Label(main, textvariable=self.status_var, foreground="#444")
        self.status_display.grid(row=3, column=0, sticky="w", padx=10, pady=(0, 8))

        btn_prev = ttk.Button(controls, text="Previous", command=self._prev_step)
        btn_prev.grid(row=0, column=0, sticky="w")
        btn_next = ttk.Button(controls, text="Next", command=self._next_step)
        btn_next.grid(row=0, column=2, sticky="e")
        btn_goto = ttk.Button(controls, text="Go to tab", command=self._open_current_tab)
        btn_goto.grid(row=0, column=1, sticky="e", padx=(0, 8))
        btn_refresh = ttk.Button(controls, text="Refresh status", command=self._refresh_status)
        btn_refresh.grid(row=0, column=3, sticky="e", padx=(8, 0))

        self.listbox.selection_set(0)

    def _open_current_tab(self) -> None:
        self.steps[self.current_idx].on_navigate()

    def _go_tab(self, index: int) -> None:
        try:
            self.select_tab(index)
        except Exception:
            pass

    def _on_step_select(self, _: tk.Event) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        self.current_idx = sel[0]
        self._refresh_status()

    def _next_step(self) -> None:
        if self.current_idx < len(self.steps) - 1:
            self.current_idx += 1
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.current_idx)
            self._refresh_status()
        else:
            messagebox.showinfo("Wizard", "You have reached the final step.")

    def _prev_step(self) -> None:
        if self.current_idx > 0:
            self.current_idx -= 1
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.current_idx)
            self._refresh_status()

    def _refresh_status(self) -> None:
        step = self.steps[self.current_idx]
        complete = step.require()
        step.status = "complete" if complete else "pending"
        self.title_label.config(text=step.title)
        self.desc_label.config(text=step.description)
        self.status_var.set(f"Status: {'Complete' if complete else 'Pending'} | Active scene: {self._scene_label()}")
        self._update_listbox_labels()

    def _scene_label(self) -> str:
        scene = self.app_state.current_scene_id
        return str(scene) if scene is not None else "(none)"

    def _update_listbox_labels(self) -> None:
        self.listbox.delete(0, tk.END)
        for idx, step in enumerate(self.steps):
            prefix = "✔ " if step.status == "complete" else "• "
            self.listbox.insert(tk.END, f"{prefix}{step.title}")
        self.listbox.selection_set(self.current_idx)


__all__ = ["WizardWindow", "GuidedWizard"]


class GuidedWizard(tk.Toplevel):
    """Actionable wizard that executes extract -> train -> export."""

    def __init__(
        self,
        root: tk.Misc,
        app_state: AppState,
        *,
        inputs_tab,
        training_tab,
        exports_tab,
        notebook,
    ) -> None:
        super().__init__(root)
        self.title("Wizard mode")
        self.geometry("820x640")
        self.app_state = app_state
        self.inputs_tab = inputs_tab
        self.training_tab = training_tab
        self.exports_tab = exports_tab
        self.notebook = notebook
        self.current_step = 0

        self.source_type_var = tk.StringVar(value="video")
        self.path_var = self.inputs_tab.video_path_var
        self.folder_var = self.inputs_tab.image_dir_var
        self.candidate_var = tk.IntVar(value=self.inputs_tab.candidate_var.get())
        self.target_var = tk.IntVar(value=self.inputs_tab.target_var.get())
        self.resolution_var = tk.IntVar(value=self.inputs_tab.training_resolution_var.get())
        self.mode_var = tk.StringVar(value=self.inputs_tab.training_resample_var.get())
        self.preset_var = tk.StringVar(value="low")

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        ttk.Label(header, text="Guided flow", font=("Segoe UI", 12, "bold")).pack(side="left")
        self.step_label = ttk.Label(header, text="Step 1 of 3")
        self.step_label.pack(side="right")

        self.card_frame = ttk.Frame(self)
        self.card_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        self.card_frame.columnconfigure(0, weight=1)

        footer = ttk.Frame(self)
        footer.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 10))
        ttk.Button(footer, text="Back", command=self._prev_step).pack(side="left")
        ttk.Button(footer, text="Next", command=self._next_step).pack(side="right")

        self.cards: list[ttk.Frame] = [self._build_step_inputs(), self._build_step_training(), self._build_step_exports()]
        for card in self.cards:
            card.grid(row=0, column=0, sticky="nsew")
        self._show_step(0)

    def _build_step_inputs(self) -> ttk.Frame:
        frame = ttk.LabelFrame(self.card_frame, text="Inputs")
        ttk.Radiobutton(frame, text="Video", variable=self.source_type_var, value="video").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(frame, text="Image folder", variable=self.source_type_var, value="images").grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(frame, text="Video file:").grid(row=1, column=0, sticky="w", padx=6)
        ttk.Entry(frame, textvariable=self.path_var, width=60).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 6))
        ttk.Label(frame, text="Image folder:").grid(row=2, column=0, sticky="w", padx=6)
        ttk.Entry(frame, textvariable=self.folder_var, width=60).grid(row=2, column=1, columnspan=2, sticky="ew", padx=(0, 6))

        row3 = ttk.Frame(frame)
        row3.grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(8, 4))
        ttk.Label(row3, text="Candidate frames:").pack(side="left")
        ttk.Spinbox(row3, from_=1, to=10000, textvariable=self.candidate_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Target frames:").pack(side="left")
        ttk.Spinbox(row3, from_=1, to=10000, textvariable=self.target_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Resolution (px, small side):").pack(side="left")
        ttk.Combobox(row3, values=[720, 1080, 2160], textvariable=self.resolution_var, state="readonly", width=12).pack(side="left", padx=(4, 8))
        ttk.Combobox(row3, values=["lanczos", "bicubic", "bilinear", "nearest"], textvariable=self.mode_var, width=10, state="readonly").pack(side="left")

        ttk.Button(frame, text="Extract and continue", command=self._extract_and_continue).grid(
            row=4, column=0, columnspan=3, sticky="w", padx=6, pady=(10, 6)
        )
        return frame

    def _build_step_training(self) -> ttk.Frame:
        frame = ttk.LabelFrame(self.card_frame, text="Training")
        row = ttk.Frame(frame)
        row.pack(fill="x", padx=6, pady=6)
        ttk.Label(row, text="Preset:").pack(side="left")
        ttk.Combobox(row, values=["low", "medium", "high"], textvariable=self.preset_var, state="readonly", width=10).pack(side="left", padx=(4, 8))
        ttk.Button(frame, text="Run training", command=self._train_and_continue).pack(anchor="w", padx=6, pady=(6, 6))
        ttk.Label(frame, text="Training runs on the active scene. Preview updates while training.", wraplength=540, justify="left").pack(anchor="w", padx=6, pady=(0, 6))
        return frame

    def _build_step_exports(self) -> ttk.Frame:
        frame = ttk.LabelFrame(self.card_frame, text="Exports")
        ttk.Label(
            frame,
            text="Latest checkpoint will be selected automatically. You can render or copy .ply from the Exports tab.",
            wraplength=540,
            justify="left",
        ).pack(anchor="w", padx=6, pady=6)
        ttk.Button(frame, text="Open Exports tab", command=lambda: self._select_tab(2)).pack(anchor="w", padx=6, pady=(4, 6))
        return frame

    def _show_step(self, idx: int) -> None:
        for i, card in enumerate(self.cards):
            card.grid_remove()
            if i == idx:
                card.grid()
        self.current_step = idx
        self.step_label.config(text=f"Step {idx + 1} of 3")

    def _next_step(self) -> None:
        if self.current_step < len(self.cards) - 1:
            self._show_step(self.current_step + 1)

    def _prev_step(self) -> None:
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _select_tab(self, index: int) -> None:
        if self.notebook is not None:
            try:
                self.notebook.select(index)
            except Exception:
                pass

    def _extract_and_continue(self) -> None:
        try:
            # Sync vars to inputs tab
            self.inputs_tab.source_type_var.set(self.source_type_var.get())
            if self.source_type_var.get() == "video":
                self.inputs_tab.video_path_var.set(self.path_var.get())
            else:
                self.inputs_tab.image_dir_var.set(self.folder_var.get())
            self.inputs_tab.candidate_var.set(self.candidate_var.get())
            self.inputs_tab.target_var.set(self.target_var.get())
            self.inputs_tab.training_resolution_var.set(self.resolution_var.get())
            self.inputs_tab.training_resample_var.set(self.mode_var.get())
            self.inputs_tab._on_resolution_change()
            self.inputs_tab._start_extraction()
            self.after(500, self._wait_for_extract)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Wizard", f"Extraction failed: {exc}", parent=self)

    def _wait_for_extract(self) -> None:
        if getattr(self.inputs_tab, "_extracting", False):
            self.after(500, self._wait_for_extract)
            return
        # Auto-save selection with resolution
        self.inputs_tab._persist_selection()
        self.after(500, self._wait_for_save)

    def _wait_for_save(self) -> None:
        if getattr(self.inputs_tab, "_saving", False):
            self.after(500, self._wait_for_save)
            return
        self._select_tab(1)
        self._show_step(1)

    def _train_and_continue(self) -> None:
        if self.training_tab is None:
            return
        try:
            self.training_tab.training_preset_var.set(self.preset_var.get())
            self.training_tab._apply_training_preset()
            self.training_tab._run_pipeline()
            self.after(1000, self._wait_for_training)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Wizard", f"Training failed: {exc}", parent=self)

    def _wait_for_training(self) -> None:
        if getattr(self.training_tab, "_working", False):
            self.after(1000, self._wait_for_training)
            return
        # Jump to exports and select latest
        if self.exports_tab is not None:
            try:
                self.exports_tab._load_checkpoints()
            except Exception:
                pass
        self._select_tab(2)
        self._show_step(2)
