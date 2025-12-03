"""Root Tkinter window for NullSplats."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from nullsplats.app_state import AppState
from nullsplats.ui.tab_exports import ExportsTab
from nullsplats.ui.tab_inputs import InputsTab
from nullsplats.ui.tab_training import TrainingTab


def create_root(app_state: AppState) -> tk.Tk:
    """Create the main Tk window with menus and notebook tabs."""
    root = tk.Tk()
    root.title(app_state.config.window_title)
    root.geometry("900x700")

    status_var = tk.StringVar(value="Ready.")
    _build_menubar(root)
    notebook = _build_tabs(root, app_state, status_var)
    notebook.pack(fill="both", expand=True)

    status = ttk.Label(root, textvariable=status_var, anchor="w", relief="sunken")
    status.pack(fill="x", side="bottom")
    return root


def _build_menubar(root: tk.Tk) -> None:
    menubar = tk.Menu(root)

    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Exit", command=root.destroy)
    menubar.add_cascade(label="File", menu=file_menu)

    view_menu = tk.Menu(menubar, tearoff=0)
    view_menu.add_command(label="Reset Layout", command=lambda: _show_info("Layout reset has no effect yet."))
    menubar.add_cascade(label="View", menu=view_menu)

    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="About", command=lambda: _show_info("NullSplats UI scaffold is running."))
    menubar.add_cascade(label="Help", menu=help_menu)

    root.config(menu=menubar)


def _build_tabs(root: tk.Tk, app_state: AppState, status_var: tk.StringVar) -> ttk.Notebook:
    notebook = ttk.Notebook(root)

    def _on_scene_selected(scene_id: str) -> None:
        status_var.set(f"Active scene: {scene_id}")

    inputs_tab = InputsTab(notebook, app_state, on_scene_selected=_on_scene_selected)
    training_tab = TrainingTab(notebook, app_state)
    exports_tab = ExportsTab(notebook, app_state)

    notebook.add(inputs_tab.frame, text="Inputs")
    notebook.add(training_tab.frame, text="Training")
    notebook.add(exports_tab.frame, text="Exports")
    return notebook


def _show_info(message: str) -> None:
    messagebox.showinfo("NullSplats", message)
