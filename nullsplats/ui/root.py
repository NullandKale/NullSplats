"""Root Tkinter window for NullSplats."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from nullsplats.app_state import AppState
from nullsplats.ui.tab_exports import ExportsTab
from nullsplats.ui.wizard import WizardWindow
from nullsplats.ui.tab_inputs import InputsTab
from nullsplats.ui.tab_training import TrainingTab
from nullsplats.util.logging import get_logger


def create_root(app_state: AppState) -> tk.Tk:
    """Create the main Tk window with menus and notebook tabs."""
    root = tk.Tk()
    root.title(app_state.config.window_title)
    root.geometry("1600x900")
    root.minsize(1600, 900)

    status_var = tk.StringVar(value="Ready.")
    notebook = _build_tabs(root, app_state, status_var)
    _build_menubar(root, app_state, notebook)
    notebook.pack(fill="both", expand=True)

    status = ttk.Label(root, textvariable=status_var, anchor="w", relief="sunken")
    status.pack(fill="x", side="bottom")

    logger = get_logger("ui.root")

    def _report_callback_exception(exc_type, exc_value, exc_traceback) -> None:
        logger.exception("Tk callback exception", exc_info=(exc_type, exc_value, exc_traceback))

    root.report_callback_exception = _report_callback_exception  # type: ignore[attr-defined]
    return root


def _build_menubar(root: tk.Tk, app_state: AppState, notebook: ttk.Notebook) -> None:
    menubar = tk.Menu(root)

    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Exit", command=root.destroy)
    menubar.add_cascade(label="File", menu=file_menu)

    view_menu = tk.Menu(menubar, tearoff=0)
    view_menu.add_command(label="Reset Layout", command=lambda: _show_info("Layout reset has no effect yet."))
    view_menu.add_command(
        label="Wizard Mode",
        command=lambda: WizardWindow(root, app_state, lambda idx: notebook.select(idx)),
    )
    menubar.add_cascade(label="View", menu=view_menu)

    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="About", command=lambda: _show_info("NullSplats UI scaffold is running."))
    menubar.add_cascade(label="Help", menu=help_menu)

    root.config(menu=menubar)


def _build_tabs(root: tk.Tk, app_state: AppState, status_var: tk.StringVar) -> ttk.Notebook:
    notebook = ttk.Notebook(root)

    training_tab = TrainingTab(notebook, app_state)
    exports_tab = ExportsTab(notebook, app_state)
    # _on_scene_selected needs to be defined after inputs_tab is created; use a placeholder and update later.
    placeholder_callback = lambda scene_id: None
    inputs_tab = InputsTab(
        notebook,
        app_state,
        on_scene_selected=placeholder_callback,
        training_tab=training_tab,
        exports_tab=exports_tab,
        notebook=notebook,
    )

    def _on_scene_selected(scene_id: str) -> None:
        status_var.set(f"Active scene: {scene_id}")
        training_tab.on_scene_changed(scene_id)
        exports_tab.on_scene_changed(scene_id)
        try:
            inputs_tab._persist_selection()
        except Exception:
            pass

    inputs_tab.on_scene_selected = _on_scene_selected  # type: ignore[attr-defined]

    notebook.add(inputs_tab.frame, text="Inputs")
    notebook.add(training_tab.frame, text="Training")
    notebook.add(exports_tab.frame, text="Exports")

    last_idx = 0
    suppress = False

    def _on_tab_changed(event: tk.Event) -> None:
        nonlocal last_idx, suppress
        if suppress:
            suppress = False
            return
        try:
            current = event.widget.index("current")
        except Exception:
            current = last_idx
        # Prevent leaving Inputs while extraction/save is in-flight, and kick off save if dirty.
        if last_idx == 0 and current != 0 and not inputs_tab.can_leave_tab():
            suppress = True
            notebook.select(last_idx)
            return
        last_idx = current
        inputs_tab.on_tab_selected(current == 0)
        training_tab.on_tab_selected(current == 1)
        exports_tab.on_tab_selected(current == 2)
        if current == 2:
            # Ensure training preview is fully halted while Exports GL context is active.
            training_tab.deactivate_viewer()
        elif current == 1:
            # Likewise, halt the exports viewer while Training is active.
            exports_tab.deactivate_viewer()

    notebook.bind("<<NotebookTabChanged>>", _on_tab_changed)
    # Initialize tab selection state
    _on_tab_changed(type("e", (), {"widget": notebook}))
    return notebook


def _show_info(message: str) -> None:
    messagebox.showinfo("NullSplats", message)
