"""VoxShield — polished desktop interface.

Run with: python desktop_app.py
Requires: numpy (for DSP modules) and Python's built-in tkinter.
"""

from __future__ import annotations

import os
import queue
import shutil
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from audio_io import AudioPlayer, Recorder, read_audio, write_output_wav
from phase_vocoder import (
    match_voice_character, reduce_background_estimate, StftConfig,
    anonymize_voice, naive_time_stretch, time_stretch,
)
from secure_audio import create_vault, unlock_vault


# ---------------------------------------------------------------------------
# Palette. Studio-dark with a warm brass accent. Every value is used somewhere;
# additions should have a purpose or the surface reads as "AI generic dark UI".
# ---------------------------------------------------------------------------
C = {
    "bg":         "#0a0a0a",   # window background — deeper than the panels
    "surface":    "#151515",   # cards, sidebar
    "surface_hi": "#1e1e1e",   # inputs, hovered rows
    "surface_lo": "#0f0f0f",   # sunken wells (readouts, code)
    "border":     "#2a2a2a",   # hairline dividers
    "border_hi":  "#3a3a3a",   # focused input borders
    "text":       "#efece3",   # primary text
    "text_dim":   "#a8a49a",   # secondary text
    "text_faint": "#6b6862",   # tertiary, hints
    "accent":     "#e8a34c",   # warm brass — active state, primary buttons
    "accent_hi":  "#f2b968",   # accent hover
    "accent_dim": "#7a5a2c",   # accent when disabled
    "danger":     "#e26a5c",   # destructive
    "danger_hi":  "#ec8478",
    "success":    "#7ac74f",   # completed/OK
    "info":       "#5ea9d6",   # informational chip
}

# Fonts. Segoe UI is a safe default on Windows; falls back cleanly on other OSes.
F = {
    "display": ("Segoe UI Semibold", 22),
    "h1":      ("Segoe UI Semibold", 17),
    "h2":      ("Segoe UI Semibold", 12),
    "body":    ("Segoe UI", 10),
    "body_b":  ("Segoe UI Semibold", 10),
    "small":   ("Segoe UI", 9),
    "eyebrow": ("Segoe UI Semibold", 8),
    "mono":    ("Consolas", 10),
    "mono_b":  ("Consolas", 10, "bold"),
    "icon":    ("Segoe UI Symbol", 12),
    "icon_lg": ("Segoe UI Symbol", 18),
}

PRESETS = {
    "Custom":            {"semitones": None, "description": "Use the pitch slider directly."},
    "Low pitch":         {"semitones": -4,   "description": "A lower, weightier pitch transformation."},
    "High pitch":        {"semitones": 4,    "description": "A brighter, higher pitch transformation."},
    "Feminine-style":    {"semitones": 3,    "description": "A higher-pitch creative style with an age character control."},
    "Masculine-style":   {"semitones": -3,   "description": "A lower-pitch creative style with an age character control."},
    "🔒 Cyber Oracle":   {"semitones": 7,    "description": "Premium preview: a dramatic high-register sci-fi voice.", "premium": True},
    "🔒 Deep Space":     {"semitones": -7,   "description": "Premium preview: a dramatic low-register sci-fi voice.",  "premium": True},
}

# Presets for the hybrid frame-size / hop inputs. The user can also type a
# custom value; the widget accepts any positive power-of-two.
FRAME_PRESETS = ("512", "1024", "2048", "4096", "8192")
HOP_PRESETS   = ("128", "256", "512", "1024", "2048")
SR_PRESETS    = ("22050", "32000", "44100", "48000", "96000")


# ===========================================================================
# Reusable widgets
# ===========================================================================

class Tooltip:
    """A subtle dark tooltip that fades in after a short hover delay."""

    _active: "Tooltip | None" = None

    def __init__(self, widget: tk.Widget, text: str, delay: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _e=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, _e=None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip or not self.widget.winfo_exists():
            return
        if Tooltip._active and Tooltip._active is not self:
            Tooltip._active._hide()
        Tooltip._active = self

        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6

        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tip.configure(bg=C["border_hi"])

        # Two-frame trick for a 1px border without ttk styling headaches
        inner = tk.Frame(tip, bg=C["surface_hi"])
        inner.pack(padx=1, pady=1)
        tk.Label(
            inner, text=self.text, justify="left", wraplength=280,
            bg=C["surface_hi"], fg=C["text"], font=F["small"],
            padx=10, pady=7,
        ).pack()
        self._tip = tip

    def _hide(self) -> None:
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None
        if Tooltip._active is self:
            Tooltip._active = None


class HoverButton(tk.Button):
    """A flat button with real hover/press color transitions and a tooltip."""

    def __init__(self, parent, text: str, command=None, *, kind: str = "ghost",
                 tooltip: str = "", icon: str = "", width: int | None = None):
        self.kind = kind
        self._enabled = True
        self._colors = self._palette(kind)

        display = f"{icon}  {text}" if icon else text
        super().__init__(
            parent, text=display, command=command,
            bd=0, relief="flat", cursor="hand2",
            font=F["body_b"], padx=16, pady=9,
            bg=self._colors["bg"], fg=self._colors["fg"],
            activebackground=self._colors["press"],
            activeforeground=self._colors["fg"],
            disabledforeground=C["text_faint"],
            highlightthickness=0,
        )
        if width:
            self.configure(width=width)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        if tooltip:
            Tooltip(self, tooltip)

    @staticmethod
    def _palette(kind: str) -> dict:
        if kind == "primary":
            return {"bg": C["accent"],    "hover": C["accent_hi"], "press": C["accent_hi"], "fg": "#111111"}
        if kind == "danger":
            return {"bg": C["danger"],    "hover": C["danger_hi"], "press": C["danger_hi"], "fg": "#111111"}
        if kind == "solid":
            return {"bg": C["surface_hi"], "hover": C["border_hi"], "press": C["border_hi"], "fg": C["text"]}
        # ghost
        return {"bg": C["surface"], "hover": C["surface_hi"], "press": C["border"], "fg": C["text"]}

    def _on_enter(self, _e=None) -> None:
        if self._enabled and str(self["state"]) != "disabled":
            self.configure(bg=self._colors["hover"])

    def _on_leave(self, _e=None) -> None:
        if self._enabled:
            self.configure(bg=self._colors["bg"])

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(state="normal" if enabled else "disabled",
                       bg=self._colors["bg"] if enabled else C["surface_lo"])


class NavButton(tk.Frame):
    """Sidebar entry with a left accent bar on the active page."""

    def __init__(self, parent, icon: str, label: str, command):
        super().__init__(parent, bg=C["surface"])
        self.command = command
        self._active = False

        self.indicator = tk.Frame(self, bg=C["surface"], width=3)
        self.indicator.pack(side="left", fill="y")

        self.button = tk.Label(
            self, text=f"  {icon}   {label}", anchor="w",
            bg=C["surface"], fg=C["text_dim"],
            font=F["body_b"], padx=20, pady=12, cursor="hand2",
        )
        self.button.pack(side="left", fill="both", expand=True)

        for w in (self, self.button, self.indicator):
            w.bind("<Button-1>", lambda _e: self.command())
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)

    def _on_enter(self, _e=None) -> None:
        if not self._active:
            self.button.configure(bg=C["surface_hi"], fg=C["text"])
            self.configure(bg=C["surface_hi"])

    def _on_leave(self, _e=None) -> None:
        if not self._active:
            self.button.configure(bg=C["surface"], fg=C["text_dim"])
            self.configure(bg=C["surface"])

    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self.indicator.configure(bg=C["accent"])
            self.button.configure(bg=C["surface_hi"], fg=C["text"])
            self.configure(bg=C["surface_hi"])
        else:
            self.indicator.configure(bg=C["surface"])
            self.button.configure(bg=C["surface"], fg=C["text_dim"])
            self.configure(bg=C["surface"])


class ScrollFrame(tk.Frame):
    """A scrollable container. Put your content inside `self.body`."""

    def __init__(self, parent, bg: str = None):
        bg = bg or C["bg"]
        super().__init__(parent, bg=bg)

        self.canvas = tk.Canvas(self, bg=bg, bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical",
                                       command=self.canvas.yview,
                                       style="Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.body = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.body,
                                                 anchor="nw")

        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Cross-platform mousewheel: bind on Enter, unbind on Leave, so pages
        # never fight each other for the scroll wheel.
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _on_body_configure(self, _e=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self, _e=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_wheel_linux)

    def _unbind_wheel(self, _e=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, e) -> None:
        self.canvas.yview_scroll(int(-e.delta / 120), "units")

    def _on_wheel_linux(self, e) -> None:
        self.canvas.yview_scroll(-1 if e.num == 4 else 1, "units")


class HybridInput(tk.Frame):
    """Combined entry + dropdown + step buttons for numeric parameters.

    Behaves like a combobox but the entry is always freely editable and there
    are +/- buttons for fine adjustment. Snappier than a Spinbox and matches
    the rest of the surface.
    """

    def __init__(self, parent, variable: tk.IntVar | tk.StringVar,
                 presets: tuple, step: int = 1, tooltip: str = ""):
        super().__init__(parent, bg=C["surface"])
        self.variable = variable
        self.step = step

        # Entry
        self.entry = tk.Entry(
            self, textvariable=variable, width=8,
            bg=C["surface_hi"], fg=C["text"],
            insertbackground=C["text"], relief="flat",
            font=F["mono"], justify="right",
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["accent"],
        )
        self.entry.pack(side="left", ipady=5, padx=(0, 4))

        # Minus / Plus
        for text, delta, tip in (("−", -step, "Decrease"), ("+", +step, "Increase")):
            btn = tk.Label(
                self, text=text, bg=C["surface_hi"], fg=C["text"],
                font=F["body_b"], padx=8, pady=3, cursor="hand2",
            )
            btn.pack(side="left", padx=(0, 4), ipady=2)
            btn.bind("<Button-1>", lambda _e, d=delta: self._step(d))
            btn.bind("<Enter>", lambda _e, b=btn: b.configure(bg=C["border_hi"]))
            btn.bind("<Leave>", lambda _e, b=btn: b.configure(bg=C["surface_hi"]))
            Tooltip(btn, f"{tip} by {step}")

        # Preset menu button
        self.menu_btn = tk.Label(
            self, text="▾ Presets", bg=C["surface_hi"], fg=C["text_dim"],
            font=F["small"], padx=10, pady=4, cursor="hand2",
        )
        self.menu_btn.pack(side="left", ipady=2)
        self.menu_btn.bind("<Button-1>", self._show_menu)
        self.menu_btn.bind("<Enter>", lambda _e: self.menu_btn.configure(bg=C["border_hi"], fg=C["text"]))
        self.menu_btn.bind("<Leave>", lambda _e: self.menu_btn.configure(bg=C["surface_hi"], fg=C["text_dim"]))

        self.presets = presets
        if tooltip:
            Tooltip(self.entry, tooltip)
            Tooltip(self.menu_btn, "Pick a common preset")

    def _step(self, delta: int) -> None:
        try:
            self.variable.set(max(1, int(self.variable.get()) + delta))
        except (tk.TclError, ValueError):
            self.variable.set(delta if delta > 0 else 1)

    def _show_menu(self, _e=None) -> None:
        menu = tk.Menu(
            self, tearoff=0, bg=C["surface_hi"], fg=C["text"],
            activebackground=C["accent"], activeforeground="#111111",
            bd=0, font=F["mono"],
        )
        for value in self.presets:
            menu.add_command(label=value, command=lambda v=value: self.variable.set(int(v)))
        menu.tk_popup(self.menu_btn.winfo_rootx(),
                      self.menu_btn.winfo_rooty() + self.menu_btn.winfo_height())


class ProgressChip(tk.Frame):
    """A small inline progress indicator with a spinning glyph."""

    _frames = ("◐", "◓", "◑", "◒")

    def __init__(self, parent):
        super().__init__(parent, bg=C["surface"])
        self.icon = tk.Label(self, text="", bg=C["surface"], fg=C["accent"], font=F["icon"])
        self.icon.pack(side="left", padx=(0, 8))
        self.label = tk.Label(self, text="", bg=C["surface"], fg=C["text_dim"], font=F["small"])
        self.label.pack(side="left")
        self._i = 0
        self._running = False

    def start(self, text: str) -> None:
        self.label.configure(text=text)
        self._running = True
        self._tick()

    def stop(self, text: str = "", ok: bool = True) -> None:
        self._running = False
        self.icon.configure(text="✓" if ok else "✕",
                            fg=C["success"] if ok else C["danger"])
        self.label.configure(text=text)

    def clear(self) -> None:
        self._running = False
        self.icon.configure(text="")
        self.label.configure(text="")

    def _tick(self) -> None:
        if not self._running or not self.winfo_exists():
            return
        self.icon.configure(text=self._frames[self._i % 4], fg=C["accent"])
        self._i += 1
        self.after(120, self._tick)


def hline(parent, pad_y: int = 0) -> tk.Frame:
    line = tk.Frame(parent, bg=C["border"], height=1)
    line.pack(fill="x", pady=pad_y)
    return line


def make_card(parent) -> tk.Frame:
    """Standard surface card with a hairline border."""
    border = tk.Frame(parent, bg=C["border"])
    inner = tk.Frame(border, bg=C["surface"])
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    inner.pack_propagate(False) if False else None  # no-op, kept for clarity
    return border, inner


# ===========================================================================
# Main application
# ===========================================================================

class VoiceLab(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VoxShield · Phase Vocoder Studio")
        self.geometry("1200x780")
        self.minsize(980, 640)
        self.configure(bg=C["bg"])
        try:
            self.iconphoto(True, tk.PhotoImage(file="dist/HackerIcon.png"))
        except Exception:
            pass

        # --- state ---
        self.result_queue: queue.Queue = queue.Queue()
        self.preview_queue: queue.Queue = queue.Queue()
        self.busy = False
        self.current_audio: tuple | None = None
        self.last_output: Path | None = None
        self.recorder: Recorder | None = None
        self.player: AudioPlayer | None = None
        self.preview_playing = False
        self.temp_recording: Path | None = None
        self.recording_paused = False

        self.temp_root = Path(tempfile.gettempdir()) / "phase_vocoder_voice_lab"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_output_dirs: list[Path] = []
        self.result_files: list[Path] = []

        # --- control variables ---
        self.volume = tk.DoubleVar(value=1.0)
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "outputs"))
        self.stretch = tk.DoubleVar(value=1.5)
        self.semitones = tk.DoubleVar(value=4.0)
        self.record_seconds = tk.IntVar(value=5)
        self.record_sample_rate = tk.IntVar(value=44100)
        self.normalize_output = tk.BooleanVar(value=False)
        self.preset_name = tk.StringVar(value="Custom")
        self.use_preset = tk.BooleanVar(value=False)
        self.premium_preview_enabled = False
        self.age = tk.DoubleVar(value=30)
        self.remove_background = tk.BooleanVar(value=False)
        self.vault_real_path = tk.StringVar()
        self.vault_decoy_path = tk.StringVar()
        self.vault_container_path = tk.StringVar()
        self.vault_passphrase = tk.StringVar()
        self.match_source_path = tk.StringVar()
        self.match_reference_path = tk.StringVar()
        self.write_naive = tk.BooleanVar(value=True)
        self.write_basic = tk.BooleanVar(value=True)
        self.write_locked = tk.BooleanVar(value=True)
        self.write_voice = tk.BooleanVar(value=True)
        self.frame_size = tk.IntVar(value=2048)
        self.hop_size = tk.IntVar(value=512)

        self.nav_buttons: dict[str, NavButton] = {}
        self.active_page: str | None = None
        self.status_label: tk.Label | None = None
        self.progress: ProgressChip | None = None
        self.process_button: HoverButton | None = None

        self._configure_style()
        self._build_shell()
        self.show_page("Home")
        self.after(100, self._poll_results)
        self.after(300, self._recover_temp_outputs)

    # ------------------------------------------------------------------
    # ttk styling
    # ------------------------------------------------------------------
    def _configure_style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure("TFrame", background=C["bg"])
        s.configure("Card.TFrame", background=C["surface"])
        s.configure("TLabel", background=C["bg"], foreground=C["text"], font=F["body"])
        s.configure("Muted.TLabel", background=C["bg"], foreground=C["text_dim"], font=F["body"])

        # Sliders
        s.configure("Horizontal.TScale",
                    background=C["surface"], troughcolor=C["surface_lo"],
                    bordercolor=C["surface"], lightcolor=C["accent"],
                    darkcolor=C["accent"])
        s.map("Horizontal.TScale",
              background=[("active", C["surface"])])

        # Checkbuttons
        s.configure("TCheckbutton",
                    background=C["surface"], foreground=C["text"],
                    font=F["body"], focuscolor=C["surface"],
                    indicatorcolor=C["surface_hi"],
                    indicatorbackground=C["surface_hi"])
        s.map("TCheckbutton",
              background=[("active", C["surface"])],
              foreground=[("active", C["text"])],
              indicatorcolor=[("selected", C["accent"])])

        # Combobox
        s.configure("TCombobox",
                    fieldbackground=C["surface_hi"],
                    background=C["surface_hi"],
                    foreground=C["text"],
                    arrowcolor=C["text"],
                    bordercolor=C["border"],
                    lightcolor=C["border"],
                    darkcolor=C["border"],
                    padding=6, relief="flat")
        s.map("TCombobox",
              fieldbackground=[("readonly", C["surface_hi"])],
              foreground=[("readonly", C["text"])],
              bordercolor=[("focus", C["accent"])])
        # dropdown list
        self.option_add("*TCombobox*Listbox.background", C["surface_hi"])
        self.option_add("*TCombobox*Listbox.foreground", C["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
        self.option_add("*TCombobox*Listbox.selectForeground", "#111111")
        self.option_add("*TCombobox*Listbox.font", F["mono"])
        self.option_add("*TCombobox*Listbox.borderWidth", 0)
        self.option_add("*TCombobox*Listbox.relief", "flat")

        # Entry
        s.configure("TEntry",
                    fieldbackground=C["surface_hi"], foreground=C["text"],
                    insertcolor=C["text"], bordercolor=C["border"],
                    lightcolor=C["border"], darkcolor=C["border"],
                    padding=7, relief="flat")
        s.map("TEntry", bordercolor=[("focus", C["accent"])])

        # Scrollbar
        s.configure("Vertical.TScrollbar",
                    background=C["surface_hi"], troughcolor=C["bg"],
                    bordercolor=C["bg"], arrowcolor=C["text_dim"],
                    gripcount=0, relief="flat")
        s.map("Vertical.TScrollbar",
              background=[("active", C["border_hi"])])

        # Progressbar (used in loading toasts)
        s.configure("Accent.Horizontal.TProgressbar",
                    background=C["accent"], troughcolor=C["surface_lo"],
                    bordercolor=C["surface_lo"], lightcolor=C["accent"],
                    darkcolor=C["accent"])

    # ------------------------------------------------------------------
    # Shell (sidebar + content region + status bar)
    # ------------------------------------------------------------------
    def _build_shell(self) -> None:
        # SIDEBAR
        sidebar = tk.Frame(self, bg=C["surface"], width=232)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # brand
        brand = tk.Frame(sidebar, bg=C["surface"])
        brand.pack(fill="x", pady=(24, 4))
        tk.Label(brand, text="◈", bg=C["surface"], fg=C["accent"],
                 font=("Segoe UI Symbol", 20)).pack(side="left", padx=(24, 10))
        wrap = tk.Frame(brand, bg=C["surface"])
        wrap.pack(side="left")
        tk.Label(wrap, text="VOXSHIELD", bg=C["surface"], fg=C["text"],
                 font=("Segoe UI Semibold", 13)).pack(anchor="w")
        tk.Label(wrap, text="Phase Vocoder Studio", bg=C["surface"],
                 fg=C["text_faint"], font=F["small"]).pack(anchor="w")

        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill="x", pady=(20, 12), padx=20)

        # nav
        for name, icon in [("Home", "⌂"), ("Transform", "✦"), ("Match", "≈"),
                           ("Results", "◫"), ("Vault", "⌁"), ("Compare", "≋"),
                           ("Learn", "◌"), ("Settings", "⚙")]:
            nav = NavButton(sidebar, icon, name, lambda n=name: self.show_page(n))
            nav.pack(fill="x", pady=1)
            self.nav_buttons[name] = nav

        # footer
        footer = tk.Frame(sidebar, bg=C["surface"])
        footer.pack(side="bottom", fill="x", pady=20, padx=20)
        tk.Frame(footer, bg=C["border"], height=1).pack(fill="x", pady=(0, 12))
        tk.Label(footer, text="🔒  Local processing",
                 bg=C["surface"], fg=C["text_dim"], font=F["small"]).pack(anchor="w")
        tk.Label(footer, text="Audio never leaves this computer.",
                 bg=C["surface"], fg=C["text_faint"], font=F["small"],
                 justify="left").pack(anchor="w", pady=(2, 0))

        # RIGHT COLUMN: content on top, persistent status bar underneath
        wrapper = tk.Frame(self, bg=C["bg"])
        wrapper.pack(side="left", fill="both", expand=True)

        status_bar = tk.Frame(wrapper, bg=C["surface"], height=36)
        status_bar.pack(side="bottom", fill="x")
        status_bar.pack_propagate(False)
        tk.Frame(status_bar, bg=C["border"], height=1).pack(fill="x", side="top")

        self.content = tk.Frame(wrapper, bg=C["bg"])
        self.content.pack(side="top", fill="both", expand=True)

        self.progress = ProgressChip(status_bar)
        self.progress.pack(side="left", padx=16, pady=8)

        self.status_label = tk.Label(
            status_bar, text="Ready. Load a WAV, or record from the mic.",
            bg=C["surface"], fg=C["text_dim"], font=F["small"],
        )
        self.status_label.pack(side="right", padx=16)

    # ------------------------------------------------------------------
    # Page routing
    # ------------------------------------------------------------------
    def show_page(self, name: str) -> None:
        self.active_page = name
        for child in self.content.winfo_children():
            child.destroy()
        for page_name, nav in self.nav_buttons.items():
            nav.set_active(page_name == name)

        builders = {
            "Home": self._home_page, "Transform": self._transform_page,
            "Match": self._match_page, "Results": self._results_page,
            "Vault": self._vault_page, "Compare": self._compare_page,
            "Learn": self._learn_page, "Settings": self._settings_page,
        }
        builders[name]()

    def _scroll_page(self) -> tk.Frame:
        """Every page mounts inside one of these — returns the body frame."""
        sf = ScrollFrame(self.content, bg=C["bg"])
        sf.pack(fill="both", expand=True)
        return sf.body

    def _page_header(self, parent, title: str, subtitle: str,
                     eyebrow: str = "") -> None:
        head = tk.Frame(parent, bg=C["bg"])
        head.pack(fill="x", padx=42, pady=(34, 22))
        if eyebrow:
            tk.Label(head, text=eyebrow, bg=C["bg"], fg=C["accent"],
                     font=F["eyebrow"]).pack(anchor="w", pady=(0, 6))
        tk.Label(head, text=title, bg=C["bg"], fg=C["text"],
                 font=F["display"]).pack(anchor="w")
        tk.Label(head, text=subtitle, bg=C["bg"], fg=C["text_dim"],
                 font=F["body"], justify="left").pack(anchor="w", pady=(6, 0))

    def _card_section(self, parent, title: str = "",
                      pad: tuple = (42, 0, 42, 16)) -> tk.Frame:
        border, inner = make_card(parent)
        border.pack(fill="x", padx=(pad[0], pad[2]), pady=(pad[1], pad[3]))
        if title:
            tk.Label(inner, text=title.upper(), bg=C["surface"],
                     fg=C["text_dim"], font=F["eyebrow"]).pack(
                         anchor="w", padx=22, pady=(18, 0))
            tk.Frame(inner, bg=C["border"], height=1).pack(
                fill="x", padx=22, pady=(10, 0))
        return inner

    # ==================================================================
    # HOME
    # ==================================================================
    def _home_page(self) -> None:
        body = self._scroll_page()
        self._page_header(
            body,
            title="Welcome to VoxShield",
            subtitle="Phase-vocoder pitch and time transformations, entirely on your computer.",
            eyebrow="OVERVIEW",
        )

        # HERO
        hero_border, hero = make_card(body)
        hero_border.pack(fill="x", padx=42, pady=(0, 20))
        hero.configure(padx=0, pady=0)
        row = tk.Frame(hero, bg=C["surface"])
        row.pack(fill="x", padx=28, pady=26)
        left = tk.Frame(row, bg=C["surface"])
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="Ready to make your first comparison?",
                 bg=C["surface"], fg=C["text"], font=F["h1"]).pack(anchor="w")
        tk.Label(left, text="Load a short spoken WAV in Transform, then generate\n"
                            "naive, phase-vocoder and phase-locked outputs side by side.",
                 bg=C["surface"], fg=C["text_dim"], font=F["body"],
                 justify="left").pack(anchor="w", pady=(6, 16))
        HoverButton(left, "Open Transform", command=lambda: self.show_page("Transform"),
                    kind="primary", icon="→",
                    tooltip="Set up input, pick outputs, and start processing."
                    ).pack(anchor="w")

        # STEPS
        steps = tk.Frame(body, bg=C["bg"])
        steps.pack(fill="x", padx=42, pady=(0, 20))
        for i, (title, body_text, tip) in enumerate([
            ("Choose a voice",  "Use a 16-bit PCM mono WAV. A short spoken sentence is ideal for comparison.",
             "Click Transform → Choose a file, or use the built-in recorder."),
            ("Generate variants", "Create naive, phase-vocoder and phase-locked transformations in one run.",
             "Uncheck any variant you don't need to save time on long clips."),
            ("Listen critically", "Compare sustained vowels, consonants, pitch, timing and audible artifacts.",
             "Use Results to A/B the outputs and Save the ones you want to keep."),
        ]):
            steps.columnconfigure(i, weight=1, uniform="step")
            border, card = make_card(steps)
            border.grid(row=0, column=i, sticky="nsew", padx=6)
            body_frame = tk.Frame(card, bg=C["surface"], padx=18, pady=16)
            body_frame.pack(fill="both", expand=True)

            step_num = tk.Label(body_frame, text=f"0{i+1}",
                                bg=C["surface"], fg=C["accent"], font=F["mono_b"])
            step_num.pack(anchor="w")
            tk.Label(body_frame, text=title, bg=C["surface"], fg=C["text"],
                     font=F["h2"]).pack(anchor="w", pady=(6, 6))
            tk.Label(body_frame, text=body_text, bg=C["surface"], fg=C["text_dim"],
                     font=F["small"], wraplength=240, justify="left").pack(anchor="w")
            Tooltip(border, tip)

    # ==================================================================
    # TRANSFORM
    # ==================================================================
    def _transform_page(self) -> None:
        body = self._scroll_page()
        self._page_header(
            body,
            title="Transform",
            subtitle="Create listening comparisons and a fixed-duration voice transformation.",
            eyebrow="STUDIO",
        )

        columns = tk.Frame(body, bg=C["bg"])
        columns.pack(fill="both", expand=True, padx=42, pady=(0, 20))
        columns.columnconfigure(0, weight=3, uniform="col")
        columns.columnconfigure(1, weight=2, uniform="col")

        # LEFT: input + parameters
        left_border, left = make_card(columns)
        left_border.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self._section_label(left, "INPUT AUDIO")
        self._path_row(left, self.input_path, self._choose_input,
                       "Path to any WAV, FLAC, OGG or MP3 you have locally.",
                       "Pick an audio file on your computer.")

        # Recording controls
        rec_row = tk.Frame(left, bg=C["surface"])
        rec_row.pack(fill="x", padx=22, pady=(6, 4))
        self.record_button = HoverButton(rec_row, "Record", command=self._start_recording,
                                         kind="primary", icon="●",
                                         tooltip="Record from your default microphone.")
        self.record_button.pack(side="left")
        self.record_pause_button = HoverButton(rec_row, "Pause", command=self._toggle_record_pause,
                                               icon="⏸", tooltip="Pause the current recording.")
        self.record_pause_button.pack(side="left", padx=(6, 0))
        self.record_stop_button = HoverButton(rec_row, "Stop", command=self._stop_recording,
                                              icon="■", tooltip="End the recording and keep the audio.")
        self.record_stop_button.pack(side="left", padx=(6, 0))

        time_row = tk.Frame(left, bg=C["surface"])
        time_row.pack(fill="x", padx=22, pady=(6, 12))
        tk.Label(time_row, text="Time limit", bg=C["surface"], fg=C["text_dim"],
                 font=F["small"]).pack(side="left")
        limit_entry = tk.Entry(time_row, textvariable=self.record_seconds, width=4,
                               bg=C["surface_hi"], fg=C["text"],
                               insertbackground=C["text"], relief="flat",
                               font=F["mono"], justify="right",
                               highlightthickness=1,
                               highlightbackground=C["border"],
                               highlightcolor=C["accent"])
        limit_entry.pack(side="left", padx=(8, 4), ipady=4)
        Tooltip(limit_entry, "Maximum seconds to record. Stop earlier at any time.")
        tk.Label(time_row, text="seconds", bg=C["surface"], fg=C["text_faint"],
                 font=F["small"]).pack(side="left")

        temp_row = tk.Frame(left, bg=C["surface"])
        temp_row.pack(fill="x", padx=22, pady=(0, 14))
        HoverButton(temp_row, "Play", command=self._play_input, icon="▶",
                    tooltip="Play the loaded or recorded audio.").pack(side="left")
        HoverButton(temp_row, "Pause", command=self._pause_playback, icon="⏸",
                    tooltip="Pause playback.").pack(side="left", padx=(6, 0))
        HoverButton(temp_row, "Save recording", command=self._save_temp_recording, icon="⤓",
                    tooltip="Save the current recording as a WAV.").pack(side="left", padx=(6, 0))
        HoverButton(temp_row, "Delete", command=self._delete_temp_recording, icon="🗑",
                    tooltip="Discard the current recording.").pack(side="left", padx=(6, 0))

        self._section_label(left, "OUTPUT FOLDER")
        self._path_row(left, self.output_dir, self._choose_output,
                       "Where saved files land when you promote them from Results.",
                       "Pick where saved WAVs go.")

        self._section_label(left, "TIME AND PITCH")

        stretch_row = tk.Frame(left, bg=C["surface"])
        stretch_row.pack(fill="x", padx=22, pady=(4, 2))
        tk.Label(stretch_row, text="Duration factor", bg=C["surface"], fg=C["text"],
                 font=F["body_b"]).pack(side="left")
        self.stretch_label = tk.Label(stretch_row, text="1.50×", bg=C["surface"],
                                      fg=C["accent"], font=F["mono_b"])
        self.stretch_label.pack(side="right")
        stretch_scale = ttk.Scale(left, from_=0.5, to=2.0, variable=self.stretch,
                                  command=lambda _=None: self._update_values())
        stretch_scale.pack(fill="x", padx=22)
        Tooltip(stretch_scale, "1.0 keeps the original length. Below 1.0 speeds it up; above 1.0 slows it down.")

        pitch_row = tk.Frame(left, bg=C["surface"])
        pitch_row.pack(fill="x", padx=22, pady=(14, 2))
        tk.Label(pitch_row, text="Pitch offset", bg=C["surface"], fg=C["text"],
                 font=F["body_b"]).pack(side="left")
        self.pitch_label = tk.Label(pitch_row, text="+4.0 st", bg=C["surface"],
                                    fg=C["accent"], font=F["mono_b"])
        self.pitch_label.pack(side="right")
        pitch_scale = ttk.Scale(left, from_=-8, to=8, variable=self.semitones,
                                command=lambda _=None: self._update_values())
        pitch_scale.pack(fill="x", padx=22, pady=(0, 18))
        Tooltip(pitch_scale, "How many semitones to shift the voice. ±12 is one octave.")

        self._update_values()

        # RIGHT: presets + outputs + action
        right_border, right = make_card(columns)
        right_border.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._section_label(right, "CREATIVE PRESET")
        preset_row = tk.Frame(right, bg=C["surface"])
        preset_row.pack(fill="x", padx=22, pady=(4, 8))
        self.preset_menu = ttk.Combobox(preset_row, textvariable=self.preset_name,
                                        values=list(PRESETS), state="readonly")
        self.preset_menu.pack(side="left", fill="x", expand=True)
        self.preset_menu.bind("<<ComboboxSelected>>", self._preset_changed)
        Tooltip(self.preset_menu, "Preset semitone offsets. 'Custom' uses the pitch slider on the left.")

        preview_row = tk.Frame(right, bg=C["surface"])
        preview_row.pack(fill="x", padx=22, pady=(0, 8))
        HoverButton(preview_row, "Preview", command=self._preview_preset, icon="▶",
                    tooltip="Hear the selected preset applied to the current audio."
                    ).pack(side="left")
        HoverButton(preview_row, "Pause", command=self._pause_playback, icon="⏸",
                    tooltip="Pause preview playback.").pack(side="left", padx=(6, 0))

        ttk.Checkbutton(right, text="Use this preset for the voice transformation output",
                        variable=self.use_preset).pack(anchor="w", padx=22, pady=(6, 2))

        self.preset_description = tk.Label(right, text=PRESETS["Custom"]["description"],
                                           bg=C["surface"], fg=C["text_dim"],
                                           font=F["small"], wraplength=340,
                                           justify="left")
        self.preset_description.pack(anchor="w", padx=22, pady=(4, 10))

        age_row = tk.Frame(right, bg=C["surface"])
        age_row.pack(fill="x", padx=22, pady=(0, 4))
        tk.Label(age_row, text="Style age", bg=C["surface"], fg=C["text_dim"],
                 font=F["small"]).pack(side="left")
        self.age_label = tk.Label(age_row, text="30 years", bg=C["surface"],
                                  fg=C["text"], font=F["mono"])
        self.age_label.pack(side="right")
        age_scale = ttk.Scale(right, from_=18, to=80, variable=self.age,
                              command=lambda _=None: self._update_values())
        age_scale.pack(fill="x", padx=22)
        Tooltip(age_scale, "Only affects Feminine-style and Masculine-style presets.")

        ttk.Checkbutton(right, text="Attempt background-music reduction (experimental)",
                        variable=self.remove_background).pack(anchor="w", padx=22, pady=(14, 8))

        self._section_label(right, "OUTPUTS TO WRITE")
        output_info = {
            self.write_naive:  ("Naive resampling baseline",
                                "Straight resample. Pitch and duration move together — the wrong-sounding reference."),
            self.write_basic:  ("Phase vocoder",
                                "STFT stretch with phase accumulation. Pitch preserved, may sound watery."),
            self.write_locked: ("Phase-locked vocoder",
                                "Bins around each spectral peak share the peak's phase correction. Cleaner sustains."),
            self.write_voice:  ("Voice transformation",
                                "Applies the pitch offset while keeping the original length."),
        }
        for variable, (label, tip) in output_info.items():
            row = tk.Frame(right, bg=C["surface"])
            row.pack(fill="x", padx=22, pady=3)
            cb = ttk.Checkbutton(row, text=label, variable=variable)
            cb.pack(side="left", anchor="w")
            Tooltip(cb, tip)

        ttk.Checkbutton(right, text="Normalize files (prevents clipping)",
                        variable=self.normalize_output).pack(anchor="w", padx=22, pady=(10, 12))

        # process button + inline chip
        action = tk.Frame(right, bg=C["surface"])
        action.pack(fill="x", padx=22, pady=(0, 20))
        self.process_button = HoverButton(action, "Generate audio files",
                                          command=self._start_processing,
                                          kind="primary", icon="✦",
                                          tooltip="Render all checked outputs into temporary files.")
        self.process_button.pack(side="left")
        self.transform_progress = ProgressChip(action)
        self.transform_progress.pack(side="left", padx=(14, 0))

    def _section_label(self, parent, text: str) -> None:
        wrap = tk.Frame(parent, bg=C["surface"])
        wrap.pack(fill="x", padx=22, pady=(16, 0))
        tk.Label(wrap, text=text, bg=C["surface"], fg=C["text_dim"],
                 font=F["eyebrow"]).pack(anchor="w")
        tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", padx=22, pady=(6, 8))

    def _path_row(self, parent, variable, command, description: str, browse_tip: str) -> None:
        row = tk.Frame(parent, bg=C["surface"])
        row.pack(fill="x", padx=22, pady=(0, 4))
        entry = tk.Entry(row, textvariable=variable, bg=C["surface_hi"], fg=C["text"],
                         insertbackground=C["text"], relief="flat", font=F["mono"],
                         highlightthickness=1, highlightbackground=C["border"],
                         highlightcolor=C["accent"])
        entry.pack(side="left", fill="x", expand=True, ipady=6)
        HoverButton(row, "Browse", command=command, kind="solid", icon="📂",
                    tooltip=browse_tip).pack(side="left", padx=(8, 0))
        tk.Label(parent, text=description, bg=C["surface"], fg=C["text_faint"],
                 font=F["small"], wraplength=560, justify="left").pack(
                     anchor="w", padx=22, pady=(4, 6))

    # ==================================================================
    # MATCH
    # ==================================================================
    def _match_page(self) -> None:
        body = self._scroll_page()
        self._page_header(
            body,
            title="Reference-guided match",
            subtitle="Match broad pitch and brightness traits from a permitted reference. This is not voice cloning.",
            eyebrow="MATCH",
        )

        card_border, card = make_card(body)
        card_border.pack(fill="x", padx=42, pady=(0, 20))

        self._section_label(card, "SOURCE VOICE")
        self._path_row(card, self.match_source_path,
                       lambda: self._choose_vault_file(self.match_source_path),
                       "The voice you want to modify. Recorded audio also works.",
                       "Pick the source voice file.")

        self._section_label(card, "REFERENCE VOICE")
        self._path_row(card, self.match_reference_path,
                       lambda: self._choose_vault_file(self.match_reference_path),
                       "Use only audio you have permission to use.",
                       "Pick the reference voice file.")

        tk.Label(card, text=(
            "The matcher estimates median pitch and broad spectral brightness, then "
            "applies a phase-locked pitch shift and gentle EQ tilt.\n"
            "It cannot copy vocal identity — only nudge the source toward the "
            "reference's general register and tone."
        ), bg=C["surface"], fg=C["text_dim"], font=F["small"],
        wraplength=760, justify="left").pack(anchor="w", padx=22, pady=(6, 14))

        action = tk.Frame(card, bg=C["surface"])
        action.pack(anchor="w", padx=22, pady=(0, 20))
        HoverButton(action, "Preview match", command=lambda: self._start_match(preview=True),
                    kind="solid", icon="▶",
                    tooltip="Render a quick preview and play it back.").pack(side="left")
        HoverButton(action, "Render to file", command=lambda: self._start_match(preview=False),
                    kind="primary", icon="✦",
                    tooltip="Render the match into a temporary WAV in Results."
                    ).pack(side="left", padx=(8, 0))
        self.match_progress = ProgressChip(action)
        self.match_progress.pack(side="left", padx=(14, 0))

    # ==================================================================
    # RESULTS
    # ==================================================================
    def _results_page(self) -> None:
        body = self._scroll_page()
        self._page_header(
            body,
            title="Temporary results",
            subtitle="Nothing here is final. Save only the versions you want to keep; delete the rest.",
            eyebrow="RESULTS",
        )

        # PLAYBACK CONTROLS
        ctrl_border, ctrl = make_card(body)
        ctrl_border.pack(fill="x", padx=42, pady=(0, 14))
        row = tk.Frame(ctrl, bg=C["surface"])
        row.pack(fill="x", padx=22, pady=16)
        tk.Label(row, text="Volume", bg=C["surface"], fg=C["text"],
                 font=F["body_b"]).pack(side="left")
        vol_scale = ttk.Scale(row, from_=0, to=2, variable=self.volume,
                              command=lambda _=None: self._set_volume())
        vol_scale.pack(side="left", fill="x", expand=True, padx=14)
        Tooltip(vol_scale, "Playback volume. Above 100% boosts; watch for clipping.")
        self.volume_label = tk.Label(row, text="100%", bg=C["surface"],
                                     fg=C["accent"], font=F["mono_b"], width=6)
        self.volume_label.pack(side="left")
        HoverButton(row, "Pause", command=self._pause_playback, icon="⏸",
                    tooltip="Pause the currently-playing clip.").pack(side="left", padx=(14, 0))
        HoverButton(row, "Stop", command=self._stop_playback, icon="■",
                    tooltip="Stop playback entirely.").pack(side="left", padx=(6, 0))

        # RESULT ROWS
        list_area = tk.Frame(body, bg=C["bg"])
        list_area.pack(fill="both", expand=True, padx=42, pady=(0, 30))
        if not self.result_files:
            empty_border, empty = make_card(list_area)
            empty_border.pack(fill="x")
            tk.Label(empty, text="◫", bg=C["surface"], fg=C["text_faint"],
                     font=F["icon_lg"]).pack(pady=(28, 8))
            tk.Label(empty, text="No temporary outputs yet",
                     bg=C["surface"], fg=C["text"], font=F["h2"]).pack()
            tk.Label(empty, text="Generate audio from the Transform page to see results here.",
                     bg=C["surface"], fg=C["text_dim"], font=F["small"]).pack(pady=(4, 28))
            return

        for path in list(self.result_files):
            if not path.is_file():
                continue
            row_border, row_card = make_card(list_area)
            row_border.pack(fill="x", pady=4)
            row_body = tk.Frame(row_card, bg=C["surface"])
            row_body.pack(fill="x", padx=18, pady=12)

            # icon
            tk.Label(row_body, text="♪", bg=C["surface"], fg=C["accent"],
                     font=F["icon_lg"]).pack(side="left", padx=(0, 14))

            # name + parent
            name_wrap = tk.Frame(row_body, bg=C["surface"])
            name_wrap.pack(side="left", fill="x", expand=True)
            tk.Label(name_wrap, text=path.name, bg=C["surface"], fg=C["text"],
                     font=F["body_b"]).pack(anchor="w")
            tk.Label(name_wrap, text=f"Temporary · {path.parent.name}",
                     bg=C["surface"], fg=C["text_faint"],
                     font=F["small"]).pack(anchor="w", pady=(2, 0))

            # actions
            HoverButton(row_body, "Save as…", command=lambda p=path: self._save_result(p),
                        kind="primary", icon="⤓",
                        tooltip="Save this file to a permanent location."
                        ).pack(side="right", padx=(6, 0))
            HoverButton(row_body, "Play", command=lambda p=path: self._play_result(p),
                        icon="▶", tooltip="Play this clip."
                        ).pack(side="right", padx=(6, 0))
            HoverButton(row_body, "Delete", command=lambda p=path: self._delete_result(p),
                        kind="danger", icon="🗑",
                        tooltip="Delete the temporary copy of this file."
                        ).pack(side="right", padx=(6, 0))

    # ==================================================================
    # VAULT
    # ==================================================================
    def _vault_page(self) -> None:
        body = self._scroll_page()
        self._page_header(
            body,
            title="Decoy vault",
            subtitle="Ordinary players hear the decoy; only VoxShield with the passphrase can unlock the hidden clip.",
            eyebrow="VAULT",
        )

        columns = tk.Frame(body, bg=C["bg"])
        columns.pack(fill="both", expand=True, padx=42, pady=(0, 30))
        columns.columnconfigure(0, weight=1, uniform="col")
        columns.columnconfigure(1, weight=1, uniform="col")

        # CREATE
        create_border, create = make_card(columns)
        create_border.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._section_label(create, "CREATE VAULT")

        tk.Label(create, text="Real clip (the hidden audio)", bg=C["surface"],
                 fg=C["text_dim"], font=F["small"]).pack(anchor="w", padx=22)
        self._path_row_light(create, self.vault_real_path,
                             lambda: self._choose_vault_file(self.vault_real_path))

        tk.Label(create, text="Decoy clip (what everyone else hears)",
                 bg=C["surface"], fg=C["text_dim"], font=F["small"]
                 ).pack(anchor="w", padx=22, pady=(6, 0))
        self._path_row_light(create, self.vault_decoy_path,
                             lambda: self._choose_vault_file(self.vault_decoy_path))

        tk.Label(create, text="Passphrase (8+ characters)", bg=C["surface"],
                 fg=C["text_dim"], font=F["small"]).pack(anchor="w", padx=22, pady=(6, 0))
        pass_entry_a = tk.Entry(create, textvariable=self.vault_passphrase, show="●",
                                bg=C["surface_hi"], fg=C["text"],
                                insertbackground=C["text"], relief="flat",
                                font=F["mono"], highlightthickness=1,
                                highlightbackground=C["border"],
                                highlightcolor=C["accent"])
        pass_entry_a.pack(fill="x", padx=22, pady=(3, 14), ipady=6)
        Tooltip(pass_entry_a, "Use a memorable but strong passphrase. It is never stored inside the vault.")

        HoverButton(create, "Create temporary vault",
                    command=self._create_vault, kind="primary", icon="⌁",
                    tooltip="Encode the real clip inside the decoy and save a temporary WAV."
                    ).pack(anchor="w", padx=22, pady=(0, 20))

        # UNLOCK
        unlock_border, unlock = make_card(columns)
        unlock_border.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._section_label(unlock, "UNLOCK VAULT")

        tk.Label(unlock, text="Vault WAV", bg=C["surface"], fg=C["text_dim"],
                 font=F["small"]).pack(anchor="w", padx=22)
        self._path_row_light(unlock, self.vault_container_path,
                             lambda: self._choose_vault_file(self.vault_container_path))

        tk.Label(unlock, text="Passphrase", bg=C["surface"], fg=C["text_dim"],
                 font=F["small"]).pack(anchor="w", padx=22, pady=(6, 0))
        pass_entry_b = tk.Entry(unlock, textvariable=self.vault_passphrase, show="●",
                                bg=C["surface_hi"], fg=C["text"],
                                insertbackground=C["text"], relief="flat",
                                font=F["mono"], highlightthickness=1,
                                highlightbackground=C["border"],
                                highlightcolor=C["accent"])
        pass_entry_b.pack(fill="x", padx=22, pady=(3, 14), ipady=6)
        Tooltip(pass_entry_b, "The same passphrase that was used to create the vault.")

        HoverButton(unlock, "Unlock to temporary audio",
                    command=self._unlock_vault, kind="primary", icon="🔓",
                    tooltip="Decode the hidden clip and add it to Results."
                    ).pack(anchor="w", padx=22, pady=(0, 12))
        tk.Label(unlock, text=(
            "The passphrase is never stored inside the vault. Keep it separately; "
            "the decoy is visible by design and is not the secret."
        ), bg=C["surface"], fg=C["text_faint"], font=F["small"],
        wraplength=340, justify="left").pack(anchor="w", padx=22, pady=(0, 20))

    def _path_row_light(self, parent, variable, command) -> None:
        row = tk.Frame(parent, bg=C["surface"])
        row.pack(fill="x", padx=22, pady=(3, 4))
        entry = tk.Entry(row, textvariable=variable, bg=C["surface_hi"], fg=C["text"],
                         insertbackground=C["text"], relief="flat", font=F["mono"],
                         highlightthickness=1, highlightbackground=C["border"],
                         highlightcolor=C["accent"])
        entry.pack(side="left", fill="x", expand=True, ipady=6)
        HoverButton(row, "Browse", command=command, kind="solid", icon="📂",
                    tooltip="Choose a file.").pack(side="left", padx=(8, 0))

    # ==================================================================
    # COMPARE
    # ==================================================================
    def _compare_page(self) -> None:
        body = self._scroll_page()
        self._page_header(
            body,
            title="Compare",
            subtitle="A listening guide for the files produced by Transform.",
            eyebrow="COMPARE",
        )

        entries = [
            ("Naive resampling", C["danger"],
             "Changes duration and pitch together. At 1.5× duration everything should sound noticeably lower — "
             "this is the reference for 'what we're arguing against'."),
            ("Phase vocoder", C["info"],
             "Changes duration while preserving pitch. Independent bins can sound watery or slightly phasey, "
             "especially on sustained vowels. That's the horizontal-only phase coherence showing up."),
            ("Phase locked", C["accent"],
             "Anchors neighboring bins to spectral peaks. Listen to long vowels for improved harmonic coherence "
             "and less of the 'chorus' effect."),
            ("Voice transformation", C["success"],
             "Applies a pitch offset, then restores the original duration. Compare several offsets on the same "
             "sentence to hear the formant shift artifacts."),
        ]

        for title, accent, text in entries:
            border, card = make_card(body)
            border.pack(fill="x", padx=42, pady=6)
            # colored top strip
            tk.Frame(card, bg=accent, height=2).pack(fill="x")
            wrap = tk.Frame(card, bg=C["surface"])
            wrap.pack(fill="x", padx=22, pady=16)
            tk.Label(wrap, text=title, bg=C["surface"], fg=C["text"],
                     font=F["h2"]).pack(anchor="w")
            tk.Label(wrap, text=text, bg=C["surface"], fg=C["text_dim"],
                     font=F["body"], wraplength=780, justify="left"
                     ).pack(anchor="w", pady=(6, 0))

        action_border, action = make_card(body)
        action_border.pack(fill="x", padx=42, pady=(14, 30))
        wrap = tk.Frame(action, bg=C["surface"])
        wrap.pack(fill="x", padx=22, pady=16)
        HoverButton(wrap, "Open output folder", command=self._open_output_folder,
                    kind="solid", icon="📂",
                    tooltip="Open the folder where saved outputs go."
                    ).pack(side="left")
        HoverButton(wrap, "Play last output", command=self._play_last_output,
                    kind="primary", icon="▶",
                    tooltip="Play the most recently rendered file."
                    ).pack(side="left", padx=(8, 0))

    # ==================================================================
    # LEARN — rewritten to walk through the actual Python implementation
    # ==================================================================
    def _learn_page(self) -> None:
        body = self._scroll_page()
        self._page_header(
            body,
            title="Learn",
            subtitle="How the code inside VoxShield turns raw samples into a pitch-shifted voice.",
            eyebrow="LEARN",
        )

        intro_border, intro = make_card(body)
        intro_border.pack(fill="x", padx=42, pady=(0, 18))
        wrap = tk.Frame(intro, bg=C["surface"])
        wrap.pack(fill="x", padx=22, pady=18)
        tk.Label(wrap, text="The pipeline in one paragraph", bg=C["surface"],
                 fg=C["text"], font=F["h2"]).pack(anchor="w")
        tk.Label(wrap, text=(
            "VoxShield reads an audio file into a 1-D NumPy array of floats, slices it into overlapping "
            "Hann-windowed frames, runs an FFT on each frame, and separates each bin into magnitude and phase. "
            "To time-stretch, it estimates the true instantaneous frequency in every bin from the phase change "
            "between frames, then re-integrates those frequencies at a new output hop so no phase jumps appear "
            "at frame boundaries. Inverse-FFT, apply a second window, add back with running normalization — done."
        ), bg=C["surface"], fg=C["text_dim"], font=F["body"], wraplength=800,
        justify="left").pack(anchor="w", pady=(6, 0))

        # Numbered topic cards, each with code
        topics = [
            ("01", "Reading and normalizing audio",
             "soundfile decodes the file into a float array. Stereo becomes mono by averaging channels — the DSP "
             "assumes one dimension. The sample rate stays as separate metadata; without it, bin index → Hz is "
             "meaningless.",
             """import soundfile as sf, numpy as np

data, sample_rate = sf.read(path, dtype="float64", always_2d=True)
mono = data.mean(axis=1)                    # (samples, ch) → (samples,)
if not np.all(np.isfinite(mono)):
    mono = np.nan_to_num(mono)              # one NaN would poison every FFT"""),

            ("02", "Framing with a Hann window",
             "The signal is sliced into frames of length N (default 2048) that overlap by 75%. Each frame gets "
             "multiplied by a Hann window before FFT — otherwise the rectangular slicing edge produces sinc-shaped "
             "leakage across every bin.",
             """N   = config.frame_size          # 2048
H_a = config.hop_size            # 512  →  75% overlap
window = np.hanning(N)

frames = [
    signal[i : i + N] * window
    for i in range(0, len(signal) - N, H_a)
]"""),

            ("03", "FFT into magnitude and phase",
             "np.fft.rfft returns N/2+1 complex bins for real input — the negative-frequency half is redundant. "
             "np.abs() and np.angle() split each bin into its two useful pieces: how loud, and where in its cycle.",
             """spectra   = np.stack([np.fft.rfft(frame) for frame in frames])
magnitude = np.abs(spectra)
phase     = np.angle(spectra)     # radians, wrapped into (-π, π]"""),

            ("04", "Estimating true frequency per bin",
             "A partial almost never sits exactly at a bin center. Between two frames its phase should have "
             "advanced by ω · H_a. Subtract the on-center expectation, wrap the leftover with princarg, and the "
             "deviation gives you the sub-bin correction — the phase vocoder's whole trick.",
             """def princarg(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

omega   = 2 * np.pi * np.arange(N // 2 + 1) / N      # bin centers
delta   = phase[1:] - phase[:-1] - omega * H_a       # deviation
delta   = princarg(delta)
true_w  = omega + delta / H_a                        # rad / sample"""),

            ("05", "Synthesis phase accumulation",
             "For a stretch factor α, output hop H_s = α · H_a. Re-integrate the true frequencies at that new hop, "
             "so successive output frames are phase-consistent by construction — no jumps at the seams.",
             """H_s = int(round(alpha * H_a))
psi = np.zeros_like(phase)
psi[0] = phase[0]
for m in range(1, len(true_w) + 1):
    psi[m] = psi[m - 1] + H_s * true_w[m - 1]

Y = magnitude * np.exp(1j * psi)                      # rebuilt spectrum"""),

            ("06", "Inverse FFT and overlap-add",
             "IFFT each spectrum, apply a synthesis window, and lay the frames down at H_s intervals. The "
             "denominator accumulates the sum of squared windows so any hop / window combination reconstructs "
             "cleanly — no COLA gymnastics needed.",
             """out_len = H_s * (len(Y) - 1) + N
out = np.zeros(out_len)
norm = np.zeros(out_len)
for m, spectrum in enumerate(Y):
    frame = np.fft.irfft(spectrum) * window
    start = m * H_s
    out[start : start + N]  += frame
    norm[start : start + N] += window ** 2
out /= np.maximum(norm, 1e-8)"""),

            ("07", "Pitch shift = stretch, then resample",
             "Pitch shifting is one call to the stretcher followed by np.interp. Stretch by β, then read the "
             "result back β× faster — the resample cancels the length change and scales all frequencies. "
             "np.interp does linear interpolation between samples.",
             """beta = 2 ** (semitones / 12)
stretched = time_stretch(signal, beta, config)         # β× longer
idx = np.arange(0, len(stretched), beta)
shifted = np.interp(idx, np.arange(len(stretched)),
                    stretched)                         # back to original length"""),

            ("08", "Naive resampling for comparison",
             "The baseline this whole project argues against. One line. Pitch and duration change together — "
             "and when you A/B it against the phase vocoder, you hear exactly what the phase math bought you.",
             """def naive_time_stretch(signal, alpha):
    idx = np.arange(0, len(signal), 1 / alpha)
    return np.interp(idx, np.arange(len(signal)), signal)"""),
        ]

        for num, title, text, code in topics:
            border, card = make_card(body)
            border.pack(fill="x", padx=42, pady=6)

            head = tk.Frame(card, bg=C["surface"])
            head.pack(fill="x", padx=22, pady=(18, 4))
            tk.Label(head, text=num, bg=C["surface"], fg=C["accent"],
                     font=F["mono_b"]).pack(side="left", padx=(0, 12))
            tk.Label(head, text=title, bg=C["surface"], fg=C["text"],
                     font=F["h2"]).pack(side="left")

            tk.Label(card, text=text, bg=C["surface"], fg=C["text_dim"],
                     font=F["body"], wraplength=800, justify="left"
                     ).pack(anchor="w", padx=22, pady=(4, 12))

            # code well
            code_bg = tk.Frame(card, bg=C["border"])
            code_bg.pack(fill="x", padx=22, pady=(0, 18))
            code_inner = tk.Frame(code_bg, bg=C["surface_lo"])
            code_inner.pack(fill="x", padx=1, pady=1)
            tk.Label(code_inner, text=code, bg=C["surface_lo"], fg=C["text"],
                     font=F["mono"], justify="left", anchor="w"
                     ).pack(fill="x", padx=16, pady=12)

        # Closing note
        note_border, note = make_card(body)
        note_border.pack(fill="x", padx=42, pady=(14, 30))
        wrap = tk.Frame(note, bg=C["surface"])
        wrap.pack(fill="x", padx=22, pady=18)
        tk.Label(wrap, text="Where the artifacts come from", bg=C["surface"],
                 fg=C["text"], font=F["h2"]).pack(anchor="w")
        tk.Label(wrap, text=(
            "The 'phasy' or 'watery' quality you sometimes hear is horizontal phase coherence only — "
            "we align each bin across time but destroy the relative phase between bins that share a partial. "
            "Phase locking fixes this by forcing bins near a peak to share the peak's correction. "
            "Transients smear because the 'slowly-varying sinusoid' model breaks down at drum hits and plosives — "
            "the fix is transient detection with a phase reset at those frames."
        ), bg=C["surface"], fg=C["text_dim"], font=F["body"], wraplength=800,
        justify="left").pack(anchor="w", pady=(6, 0))

    # ==================================================================
    # SETTINGS — now uses HybridInput (entry + presets + step buttons)
    # ==================================================================
    def _settings_page(self) -> None:
        body = self._scroll_page()
        self._page_header(
            body,
            title="Settings",
            subtitle="Advanced processing options. Defaults are chosen for spoken voice at 44.1 kHz.",
            eyebrow="SETTINGS",
        )

        # STFT parameters
        stft_border, stft = make_card(body)
        stft_border.pack(fill="x", padx=42, pady=(0, 14))
        self._section_label(stft, "STFT PARAMETERS")

        self._setting_row(
            stft, "Frame size",
            "Samples per FFT. Larger = finer frequency resolution but blurrier time resolution.",
            self.frame_size, FRAME_PRESETS, step=256,
            tooltip="Power-of-two samples per FFT. 2048 at 44.1 kHz ≈ 46 ms per frame.",
        )
        self._setting_row(
            stft, "Analysis hop",
            "Samples between successive frames. Smaller = more overlap and better phase tracking.",
            self.hop_size, HOP_PRESETS, step=64,
            tooltip="Should typically be frame_size / 4 (75% overlap) for phase vocoders.",
        )

        # Ratio badge
        ratio_wrap = tk.Frame(stft, bg=C["surface"])
        ratio_wrap.pack(fill="x", padx=22, pady=(4, 18))
        tk.Label(ratio_wrap, text="OVERLAP", bg=C["surface"], fg=C["text_faint"],
                 font=F["eyebrow"]).pack(side="left")
        self.overlap_label = tk.Label(ratio_wrap, text="—", bg=C["surface"],
                                      fg=C["accent"], font=F["mono_b"])
        self.overlap_label.pack(side="left", padx=(10, 0))
        self._update_overlap_label()
        self.frame_size.trace_add("write", lambda *_: self._update_overlap_label())
        self.hop_size.trace_add("write", lambda *_: self._update_overlap_label())

        # Recording
        rec_border, rec = make_card(body)
        rec_border.pack(fill="x", padx=42, pady=(0, 14))
        self._section_label(rec, "RECORDING")
        self._setting_row(
            rec, "Sample rate",
            "Recording sample rate in Hz. 44,100 is the standard for spoken voice.",
            self.record_sample_rate, SR_PRESETS, step=1000,
            tooltip="Higher rates capture more bandwidth but cost more disk and CPU.",
        )

        # Output preferences
        out_border, out = make_card(body)
        out_border.pack(fill="x", padx=42, pady=(0, 30))
        self._section_label(out, "OUTPUT")
        row = tk.Frame(out, bg=C["surface"])
        row.pack(fill="x", padx=22, pady=(4, 18))
        cb = ttk.Checkbutton(row, text="Normalize generated files by default",
                             variable=self.normalize_output)
        cb.pack(anchor="w")
        Tooltip(cb, "Scales each output so the loudest sample is just below 0 dBFS.")

    def _setting_row(self, parent, label: str, description: str,
                     variable, presets: tuple, step: int, tooltip: str) -> None:
        row = tk.Frame(parent, bg=C["surface"])
        row.pack(fill="x", padx=22, pady=(10, 4))

        left = tk.Frame(row, bg=C["surface"])
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=label, bg=C["surface"], fg=C["text"],
                 font=F["body_b"]).pack(anchor="w")
        tk.Label(left, text=description, bg=C["surface"], fg=C["text_dim"],
                 font=F["small"], wraplength=520, justify="left"
                 ).pack(anchor="w", pady=(2, 0))

        HybridInput(row, variable, presets, step=step, tooltip=tooltip
                    ).pack(side="right", padx=(14, 0))

    def _update_overlap_label(self) -> None:
        if not hasattr(self, "overlap_label") or not self.overlap_label.winfo_exists():
            return
        try:
            n = int(self.frame_size.get())
            h = int(self.hop_size.get())
            if n > 0 and h > 0:
                pct = (1 - h / n) * 100
                self.overlap_label.configure(text=f"{pct:.1f}%   ·   hop = frame ÷ {n / h:.2f}")
                return
        except (tk.TclError, ValueError, ZeroDivisionError):
            pass
        self.overlap_label.configure(text="—")

    # ==================================================================
    # BUSINESS LOGIC — unchanged from your original app, only surfaced
    # through the new widgets. Status and progress use the new bar.
    # ==================================================================

    def _set_status(self, text: str) -> None:
        if self.status_label and self.status_label.winfo_exists():
            self.status_label.configure(text=text)

    def _preset_changed(self, _event=None) -> None:
        info = PRESETS[self.preset_name.get()]
        if info.get("premium") and not self.premium_preview_enabled:
            if messagebox.askyesno(
                "Premium preset preview",
                "This is a locked premium preset. Enable a temporary local preview for this session?",
            ):
                self.premium_preview_enabled = True
            else:
                self.preset_name.set("Custom")
                info = PRESETS["Custom"]
        if hasattr(self, "preset_description") and self.preset_description.winfo_exists():
            self.preset_description.configure(text=info["description"])

    def _active_semitones(self) -> float:
        preset = PRESETS[self.preset_name.get()]
        if self.use_preset.get() and preset["semitones"] is not None:
            offset = float(preset["semitones"])
            if self.preset_name.get() in {"Feminine-style", "Masculine-style"}:
                offset -= (self.age.get() - 30) * 0.03
            return offset
        return self.semitones.get()

    def _preview_preset(self) -> None:
        if self.preview_playing and self.player is not None and self.player.paused:
            self.player.resume()
            self._set_status("Preset preview resumed.")
            return
        source = Path(self.input_path.get())
        if self.current_audio is None and not source.is_file():
            messagebox.showerror("Choose audio", "Select or record audio before previewing a preset.")
            return
        semitones = self._active_semitones()
        if semitones == 0:
            messagebox.showinfo("Choose a transformation",
                                "Choose a preset or a non-zero pitch offset to preview.")
            return
        try:
            config = StftConfig(frame_size=self.frame_size.get(),
                                hop_size=self.hop_size.get())
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        input_data = self.current_audio if self.current_audio is not None else source
        self._set_status("Rendering preset preview…")
        threading.Thread(
            target=self._preview_worker,
            args=(input_data, semitones, config, self.remove_background.get()),
            daemon=True).start()

    def _preview_worker(self, input_data, semitones, config, remove_background) -> None:
        try:
            if isinstance(input_data, Path):
                signal, sample_rate = read_audio(input_data)
            else:
                signal, sample_rate = input_data
            config = StftConfig(sample_rate=sample_rate,
                                frame_size=config.frame_size,
                                hop_size=config.hop_size)
            if remove_background:
                signal = reduce_background_estimate(signal, config)
            transformed = anonymize_voice(signal, semitones, config)
            self.preview_queue.put((True, (transformed, sample_rate)))
        except Exception as error:
            self.preview_queue.put((False, str(error)))

    def _start_processing(self) -> None:
        if self.busy:
            return
        source = Path(self.input_path.get())
        if self.current_audio is None and not source.is_file():
            messagebox.showerror("Choose an input",
                                 "Select a valid audio file or record from the microphone first.")
            return
        if not any(v.get() for v in (self.write_naive, self.write_basic,
                                     self.write_locked, self.write_voice)):
            messagebox.showerror("Choose an output", "Select at least one output type.")
            return
        active_semitones = self._active_semitones()
        if self.write_voice.get() and active_semitones == 0:
            messagebox.showerror("Choose a pitch offset",
                                 "Voice transformation needs a non-zero pitch offset.")
            return
        try:
            config = StftConfig(frame_size=self.frame_size.get(),
                                hop_size=self.hop_size.get())
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        self.busy = True
        if self.process_button:
            self.process_button.set_enabled(False)
        if hasattr(self, "transform_progress"):
            self.transform_progress.start("Processing locally…")
        self._set_status("Processing locally… this can take a moment.")
        input_data = self.current_audio if self.current_audio is not None else source
        session_dir = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root))
        self.temp_output_dirs.append(session_dir)
        options = (input_data, session_dir, self.stretch.get(), active_semitones,
                   config, self.write_naive.get(), self.write_basic.get(),
                   self.write_locked.get(), self.write_voice.get(),
                   self.normalize_output.get(), self.remove_background.get())
        threading.Thread(target=self._process_worker, args=options, daemon=True).start()

    def _process_worker(self, input_data, destination, stretch, semitones, config,
                        naive, basic, locked, voice, normalize, remove_background) -> None:
        try:
            if isinstance(input_data, Path):
                signal, sample_rate = read_audio(input_data)
                stem = input_data.stem
            else:
                signal, sample_rate = input_data
                stem = "recording"
            config = StftConfig(sample_rate=sample_rate,
                                frame_size=config.frame_size,
                                hop_size=config.hop_size)
            if remove_background:
                signal = reduce_background_estimate(signal, config)
            destination.mkdir(parents=True, exist_ok=True)
            written = []
            jobs = []
            if naive:  jobs.append((f"{stem}_naive_{stretch:g}x.wav",
                                    naive_time_stretch(signal, stretch)))
            if basic:  jobs.append((f"{stem}_phase_vocoder_{stretch:g}x.wav",
                                    time_stretch(signal, stretch, config)))
            if locked: jobs.append((f"{stem}_phase_locked_{stretch:g}x.wav",
                                    time_stretch(signal, stretch, config, phase_locking=True)))
            if voice:  jobs.append((f"{stem}_voice_transform_{semitones:+g}st.wav",
                                    anonymize_voice(signal, semitones, config)))
            for name, transformed in jobs:
                if normalize:
                    peak = float(abs(transformed).max())
                    if peak > 0:
                        transformed = transformed * (0.98 / peak)
                write_output_wav(destination / name, transformed, sample_rate)
                written.append(name)
            paths = [destination / name for name in written]
            self.result_queue.put((True, f"Created {len(written)} temporary file(s). Review and save the ones you want.", paths))
        except Exception as error:
            self.result_queue.put((False, str(error), []))

    # (Vault, match, playback, recording — same logic as your original file)
    def _start_match(self, preview: bool) -> None:
        reference = Path(self.match_reference_path.get())
        source_path = Path(self.match_source_path.get())
        if not reference.is_file() or (self.current_audio is None and not source_path.is_file()):
            messagebox.showerror("Choose voices",
                                 "Choose a reference and source, or record/select a source in Transform first.")
            return
        try:
            config = StftConfig(frame_size=self.frame_size.get(),
                                hop_size=self.hop_size.get())
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        source = self.current_audio if self.current_audio is not None else source_path
        if hasattr(self, "match_progress"):
            self.match_progress.start("Matching…")
        if preview:
            threading.Thread(target=self._match_worker,
                             args=(source, reference, config, None),
                             daemon=True).start()
        else:
            session = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root))
            self.temp_output_dirs.append(session)
            threading.Thread(target=self._match_worker,
                             args=(source, reference, config, session / "reference_guided_match.wav"),
                             daemon=True).start()

    def _match_worker(self, source, reference_path, config, destination) -> None:
        try:
            if isinstance(source, Path):
                source_samples, source_rate = read_audio(source)
            else:
                source_samples, source_rate = source
            reference_samples, reference_rate = read_audio(reference_path)
            config = StftConfig(sample_rate=source_rate,
                                frame_size=config.frame_size,
                                hop_size=config.hop_size)
            matched, profile = match_voice_character(
                source_samples, source_rate, reference_samples, reference_rate, config)
            shift = profile["pitch_shift_semitones"]
            message = f"Reference match used {shift:+.1f} semitones of pitch adjustment."
            if destination is None:
                self.preview_queue.put((True, (matched, source_rate)))
            else:
                write_output_wav(destination, matched, source_rate)
                self.result_queue.put((True, message, [destination]))
        except Exception as error:
            if destination is None:
                self.preview_queue.put((False, str(error)))
            else:
                self.result_queue.put((False, str(error), []))

    def _choose_vault_file(self, variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            title="Choose audio",
            filetypes=[("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a"),
                       ("All files", "*.*")])
        if selected:
            variable.set(selected)

    def _create_vault(self) -> None:
        decoy = Path(self.vault_decoy_path.get())
        real_path = Path(self.vault_real_path.get())
        if not decoy.is_file() or (self.current_audio is None and not real_path.is_file()):
            messagebox.showerror("Choose audio", "Choose a decoy and a real clip, or record/select audio in Transform first.")
            return
        if len(self.vault_passphrase.get()) < 8:
            messagebox.showerror("Passphrase required", "Use a passphrase of at least 8 characters.")
            return
        session = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root))
        self.temp_output_dirs.append(session)
        destination = session / "voice_lab_vault.wav"
        real_input = self.current_audio if self.current_audio is not None else real_path
        threading.Thread(target=self._vault_create_worker,
                         args=(real_input, decoy, self.vault_passphrase.get(), destination),
                         daemon=True).start()

    def _vault_create_worker(self, real_input, decoy, passphrase, destination) -> None:
        try:
            if isinstance(real_input, Path):
                samples, sample_rate = read_audio(real_input)
            else:
                samples, sample_rate = real_input
            create_vault(samples, sample_rate, decoy, passphrase, destination)
            self.result_queue.put((True, "Created a temporary decoy vault. Save it from Results when ready.", [destination]))
        except Exception as error:
            self.result_queue.put((False, str(error), []))

    def _unlock_vault(self) -> None:
        vault = Path(self.vault_container_path.get())
        if not vault.is_file() or len(self.vault_passphrase.get()) < 8:
            messagebox.showerror("Vault details required", "Choose a vault WAV and enter its passphrase.")
            return
        session = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root))
        self.temp_output_dirs.append(session)
        destination = session / "unlocked_audio.wav"
        threading.Thread(target=self._vault_unlock_worker,
                         args=(vault, self.vault_passphrase.get(), destination),
                         daemon=True).start()

    def _vault_unlock_worker(self, vault, passphrase, destination) -> None:
        try:
            samples, sample_rate = unlock_vault(vault, passphrase)
            write_output_wav(destination, samples, sample_rate)
            self.result_queue.put((True, "Unlocked audio to a temporary file. Review or save it from Results.", [destination]))
        except Exception as error:
            self.result_queue.put((False, str(error), []))

    # --- playback helpers -----------------------------------------------
    def _set_volume(self) -> None:
        if hasattr(self, "volume_label") and self.volume_label.winfo_exists():
            self.volume_label.configure(text=f"{self.volume.get() * 100:.0f}%")
        if self.player is not None:
            self.player.set_volume(self.volume.get())

    def _play_result(self, path: Path) -> None:
        try:
            self.preview_playing = False
            self._start_playback(*read_audio(path))
        except Exception as error:
            messagebox.showerror("Playback failed", str(error))

    def _save_result(self, path: Path) -> None:
        destination = filedialog.asksaveasfilename(
            title="Save generated audio", initialdir=self.output_dir.get(),
            initialfile=path.name, defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")])
        if destination:
            shutil.copy2(path, destination)
            messagebox.showinfo("Saved", f"Saved {path.name} to your selected location.")

    def _delete_result(self, path: Path) -> None:
        if not path.is_file():
            return
        if not messagebox.askyesno("Delete temporary file",
                                   f"Delete {path.name}? This only deletes the temporary copy."):
            return
        self._stop_playback()
        path.unlink()
        self.result_files = [item for item in self.result_files if item != path]
        self.show_page("Results")

    def _recover_temp_outputs(self) -> None:
        old_dirs = [p for p in self.temp_root.iterdir()
                    if p.is_dir() and p.name.startswith("session_")]
        old_files = [f for d in old_dirs for f in d.glob("*.wav")]
        if not old_files:
            return
        keep = messagebox.askyesno(
            "Temporary audio found",
            f"Found {len(old_files)} generated temporary file(s) from an earlier session. Keep them?")
        if keep:
            self.temp_output_dirs.extend(old_dirs)
            self.result_files.extend(old_files)
        else:
            for d in old_dirs:
                shutil.rmtree(d)

    def _update_values(self) -> None:
        if hasattr(self, "stretch_label") and self.stretch_label.winfo_exists():
            self.stretch_label.configure(text=f"{self.stretch.get():.2f}×")
        if hasattr(self, "pitch_label") and self.pitch_label.winfo_exists():
            self.pitch_label.configure(text=f"{self.semitones.get():+.1f} st")
        if hasattr(self, "age_label") and self.age_label.winfo_exists():
            self.age_label.configure(text=f"{self.age.get():.0f} years")

    def _choose_input(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose audio",
            filetypes=[("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a"),
                       ("All files", "*.*")])
        if selected:
            self.current_audio = None
            self.input_path.set(selected)

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Choose output folder")
        if selected:
            self.output_dir.set(selected)

    def _open_output_folder(self) -> None:
        path = Path(self.output_dir.get())
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except AttributeError:
            # Non-Windows fallback
            import subprocess, sys
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])

    def _play_last_output(self) -> None:
        if self.last_output and Path(self.last_output).is_file():
            self._play_result(Path(self.last_output))
        else:
            messagebox.showinfo("Nothing to play", "Generate audio first.")

    # --- recording ------------------------------------------------------
    def _start_recording(self) -> None:
        if self.busy or self.recorder is not None:
            return
        try:
            seconds = int(self.record_seconds.get())
            if seconds <= 0:
                raise ValueError("Recording duration must be positive")
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Invalid recording length", str(error))
            return
        try:
            self.recorder = Recorder(self.record_sample_rate.get())
            self.recorder.start()
            self.recording_paused = False
            self.record_button.set_enabled(False)
            self._set_status("Recording… press Pause, then Resume, or Stop to keep the audio.")
            self.after(seconds * 1000, self._auto_stop_recording)
        except Exception as error:
            self.recorder = None
            messagebox.showerror("Recording failed", str(error))

    def _auto_stop_recording(self) -> None:
        if self.recorder is not None:
            self._stop_recording()

    def _toggle_record_pause(self) -> None:
        if self.recorder is None:
            return
        if self.recording_paused:
            self.recorder.resume()
            self.recording_paused = False
            self._set_status("Recording resumed.")
        else:
            self.recorder.pause()
            self.recording_paused = True
            self._set_status("Recording paused.")

    def _stop_recording(self) -> None:
        if self.recorder is None:
            return
        try:
            samples, sample_rate = self.recorder.stop()
            self.current_audio = (samples, sample_rate)
            self.temp_recording = self.temp_root / "last_recording.wav"
            write_output_wav(self.temp_recording, samples, sample_rate)
            self.input_path.set(str(self.temp_recording))
            self._set_status("Recording captured. Ready to preview or process.")
        except Exception as error:
            messagebox.showerror("Recording failed", str(error))
        finally:
            self.recorder = None
            if hasattr(self, "record_button") and self.record_button.winfo_exists():
                self.record_button.set_enabled(True)

    def _save_temp_recording(self) -> None:
        if not self.temp_recording or not self.temp_recording.is_file():
            messagebox.showinfo("Nothing to save", "Record something first.")
            return
        destination = filedialog.asksaveasfilename(
            title="Save recording", defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")])
        if destination:
            shutil.copy2(self.temp_recording, destination)
            messagebox.showinfo("Saved", "Recording saved.")

    def _delete_temp_recording(self) -> None:
        if self.temp_recording and self.temp_recording.is_file():
            self.temp_recording.unlink()
        self.temp_recording = None
        self.current_audio = None
        self.input_path.set("")
        self._set_status("Recording deleted.")

    def _play_input(self) -> None:
        if self.player is not None and self.player.paused:
            self.player.resume()
            return
        if self.current_audio is not None:
            self._start_playback(*self.current_audio)
        elif Path(self.input_path.get()).is_file():
            try:
                self._start_playback(*read_audio(Path(self.input_path.get())))
            except Exception as error:
                messagebox.showerror("Playback failed", str(error))
        else:
            messagebox.showinfo("No audio", "Load or record audio first.")

    def _pause_playback(self) -> None:
        if self.player is not None:
            self.player.pause()

    def _stop_playback(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.player = None
        self.preview_playing = False

    def _start_playback(self, samples, sample_rate) -> None:
        self._stop_playback()
        try:
            self.player = AudioPlayer(samples, sample_rate, self.volume.get())
            self.player.play()
        except Exception as error:
            messagebox.showerror("Playback failed", str(error))

    # --- result polling -------------------------------------------------
    def _poll_results(self) -> None:
        try:
            while True:
                ok, payload = self.preview_queue.get_nowait()
                if ok:
                    samples, sample_rate = payload
                    self.preview_playing = True
                    self._start_playback(samples, sample_rate)
                    if hasattr(self, "match_progress") and self.match_progress.winfo_exists():
                        self.match_progress.stop("Preview ready.", ok=True)
                    self._set_status("Preview playing.")
                else:
                    if hasattr(self, "match_progress") and self.match_progress.winfo_exists():
                        self.match_progress.stop("Failed.", ok=False)
                    messagebox.showerror("Preview failed", str(payload))
        except queue.Empty:
            pass

        try:
            while True:
                ok, message, outputs = self.result_queue.get_nowait()
                self.busy = False
                if outputs:
                    self.result_files.extend(outputs)
                    self.last_output = outputs[-1]
                if self.process_button and self.process_button.winfo_exists():
                    self.process_button.set_enabled(True)
                if hasattr(self, "transform_progress") and self.transform_progress.winfo_exists():
                    self.transform_progress.stop("Done." if ok else "Failed.", ok=ok)
                if hasattr(self, "match_progress") and self.match_progress.winfo_exists():
                    self.match_progress.stop("Rendered." if ok else "Failed.", ok=ok)
                self._set_status(message)
                if not ok:
                    messagebox.showerror("Processing failed", message)
                elif outputs:
                    self.show_page("Results")
        except queue.Empty:
            pass

        self.after(100, self._poll_results)


if __name__ == "__main__":
    VoiceLab().mainloop()
