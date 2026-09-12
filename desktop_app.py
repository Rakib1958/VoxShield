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
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from audio_io import AudioPlayer, Recorder, read_audio, write_output_wav
from phase_vocoder import (
    match_voice_character, reduce_background_estimate, StftConfig, stft,
    anonymize_voice, naive_time_stretch, time_stretch,
)
from secure_audio import create_vault, unlock_vault
from realtime import LiveVoiceChanger, detect_virtual_cable, list_output_devices
from ml_voice import MLVoiceConverter, capability_report


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
    "Custom":            {"semitones": None, "formant_ratio": 1.0,
                          "description": "Use the pitch and formant sliders directly."},
    "Low pitch":         {"semitones": -4,   "formant_ratio": 1.0,
                          "description": "Pitch down only. Vocal-tract size (formants) stays put."},
    "High pitch":        {"semitones": 4,    "formant_ratio": 1.0,
                          "description": "Pitch up only. Vocal-tract size (formants) stays put."},
    "Feminine-style":    {"semitones": 4,    "formant_ratio": 1.18,
                          "description": "Higher pitch plus raised formants — a smaller-sounding vocal tract. Age control adjusts both."},
    "Masculine-style":   {"semitones": -4,   "formant_ratio": 0.85,
                          "description": "Lower pitch plus lowered formants — a larger-sounding vocal tract. Age control adjusts both."},
    "🔒 Cyber Oracle":   {"semitones": 7,    "formant_ratio": 0.72,
                          "description": "Premium preview: a high pitch over lowered formants — a deliberate register clash.", "premium": True},
    "🔒 Deep Space":     {"semitones": -7,   "formant_ratio": 0.65,
                          "description": "Premium preview: a cavernous low pitch and lowered formants.",  "premium": True},
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


class HoverButton(tk.Canvas):
    """A rounded button with a hover glow, a slight press-scale, and a tooltip.

    Canvas-drawn (rounded rectangles aren't available on a plain tk.Button)
    so it can grow a little on hover and shrink a touch on press — small
    motion is most of what makes a flat form feel like a live app instead of
    a printed page. The public surface (constructor kwargs, `.pack()`,
    `.set_enabled()`) is unchanged from the old tk.Button version, so every
    existing call site keeps working.
    """

    def __init__(self, parent, text: str, command=None, *, kind: str = "ghost",
                 tooltip: str = "", icon: str = "", width: int | None = None):
        self.kind = kind
        self._enabled = True
        self._hover = False
        self._pressed = False
        self._colors = self._palette(kind)
        self.command = command
        self._display = f"{icon}  {text}" if icon else text

        probe = tk.Label(parent, text=self._display, font=F["body_b"])
        probe.update_idletasks()
        text_w, text_h = probe.winfo_reqwidth(), probe.winfo_reqheight()
        probe.destroy()

        pad_x, pad_y = 18, 11
        min_w = width * 8 if width else 0
        self._btn_w = max(text_w + pad_x * 2, min_w)
        self._btn_h = text_h + pad_y * 2
        self._radius = min(14, self._btn_h // 2)

        try:
            parent_bg = parent.cget("bg")
        except tk.TclError:
            parent_bg = C["bg"]

        super().__init__(parent, width=self._btn_w, height=self._btn_h, bg=parent_bg,
                         bd=0, highlightthickness=0, cursor="hand2")
        self._draw()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
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

    @staticmethod
    def _rounded_points(x0: float, y0: float, x1: float, y1: float, r: float) -> list[float]:
        return [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
            x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
            x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
        ]

    def _draw(self, scale: float = 1.0, fill: str | None = None) -> None:
        if not self.winfo_exists():
            return
        self.delete("all")
        cx, cy = self._btn_w / 2, self._btn_h / 2
        w2, h2 = self._btn_w * scale, self._btn_h * scale
        x0, y0, x1, y1 = cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2
        color = fill or (self._colors["bg"] if self._enabled else C["surface_lo"])
        self.create_polygon(
            self._rounded_points(x0, y0, x1, y1, self._radius),
            smooth=True, splinesteps=12, fill=color, outline="",
        )
        fg = self._colors["fg"] if self._enabled else C["text_faint"]
        self.create_text(cx, cy, text=self._display, fill=fg, font=F["body_b"])

    def _on_enter(self, _e=None) -> None:
        if not self._enabled:
            return
        self._hover = True
        self._draw(scale=1.03, fill=self._colors["hover"])

    def _on_leave(self, _e=None) -> None:
        self._hover = False
        self._pressed = False
        if self._enabled:
            self._draw(scale=1.0, fill=self._colors["bg"])

    def _on_press(self, _e=None) -> None:
        if not self._enabled:
            return
        self._pressed = True
        self._draw(scale=0.97, fill=self._colors["press"])

    def _on_release(self, _e=None) -> None:
        if not self._enabled:
            return
        was_pressed = self._pressed
        self._pressed = False
        inside = self._hover
        self._draw(scale=1.03 if inside else 1.0,
                   fill=self._colors["hover"] if inside else self._colors["bg"])
        if was_pressed and inside and self.command:
            self.command()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw(scale=1.0, fill=self._colors["bg"] if enabled else C["surface_lo"])


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


class WaveformView(tk.Canvas):
    """A lightweight min/max envelope waveform display — no extra dependencies.

    Redraws from scratch on resize or a new `set_samples()` call; this isn't
    meant to animate, just to make "here is the audio you loaded" visible
    rather than a bare file path.
    """

    def __init__(self, parent, height: int = 64):
        super().__init__(parent, height=height, bg=C["surface_lo"], bd=0, highlightthickness=0)
        self._samples: np.ndarray | None = None
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_samples(self, samples) -> None:
        if samples is None or len(samples) == 0:
            self._samples = None
        else:
            self._samples = np.asarray(samples, dtype=np.float64)
        self._redraw()

    def clear(self) -> None:
        self.set_samples(None)

    def _redraw(self) -> None:
        if not self.winfo_exists():
            return
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 2 or height <= 2:
            return
        mid = height / 2
        self.create_line(0, mid, width, mid, fill=C["border"])
        if self._samples is None or len(self._samples) == 0:
            self.create_text(width / 2, mid, text="No audio loaded",
                             fill=C["text_faint"], font=F["small"])
            return
        samples = self._samples
        columns = max(1, int(width))
        bucket = max(1, len(samples) // columns)
        peak = float(np.max(np.abs(samples))) or 1.0
        usable = mid - 3
        for x in range(columns):
            chunk = samples[x * bucket : x * bucket + bucket]
            if len(chunk) == 0:
                continue
            lo, hi = float(chunk.min()), float(chunk.max())
            y0 = mid - (hi / peak) * usable
            y1 = mid - (lo / peak) * usable
            self.create_line(x, y0, x, max(y1, y0 + 1), fill=C["accent"])


class SpectrogramView(tk.Canvas):
    """A coarse spectrogram thumbnail — a grid of colored rectangles, no PIL needed.

    Frequency increases upward. Color runs from `surface_hi` (quiet) to
    `accent` (loud) over roughly a 60 dB window below the loudest cell, which
    is plenty of resolution to *see* formants and harmonics move without
    needing an image library the rest of this dependency-free project doesn't use.
    """

    def __init__(self, parent, height: int = 120, max_columns: int = 110, max_rows: int = 48):
        super().__init__(parent, height=height, bg=C["surface_lo"], bd=0, highlightthickness=0)
        self._magnitude_db: np.ndarray | None = None
        self._max_columns = max_columns
        self._max_rows = max_rows
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_samples(self, samples, sample_rate: int) -> None:
        if samples is None or len(samples) < 512:
            self._magnitude_db = None
            self._redraw()
            return
        try:
            config = StftConfig(sample_rate=sample_rate, frame_size=1024, hop_size=512)
            spectra, _ = stft(np.asarray(samples, dtype=np.float64), config)
            magnitude = np.abs(spectra).T  # (bins, frames), low bin first
            self._magnitude_db = 20 * np.log10(magnitude + 1e-6)
        except Exception:
            self._magnitude_db = None
        self._redraw()

    def clear(self) -> None:
        self._magnitude_db = None
        self._redraw()

    @staticmethod
    def _lerp_color(low_hex: str, high_hex: str, t: float) -> str:
        low = tuple(int(low_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        high = tuple(int(high_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        mixed = tuple(int(low[i] + (high[i] - low[i]) * t) for i in range(3))
        return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"

    def _redraw(self) -> None:
        if not self.winfo_exists():
            return
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 2 or height <= 2:
            return
        if self._magnitude_db is None:
            self.create_text(width / 2, height / 2, text="No audio loaded",
                             fill=C["text_faint"], font=F["small"])
            return
        db = self._magnitude_db
        bins, frames = db.shape
        cols = max(1, min(self._max_columns, frames))
        rows = max(1, min(self._max_rows, bins))
        col_edges = np.linspace(0, frames, cols + 1).astype(int)
        row_edges = np.linspace(0, bins, rows + 1).astype(int)
        vmax = float(np.percentile(db, 99))
        vmin = vmax - 60.0
        span = max(vmax - vmin, 1e-6)
        cell_w = width / cols
        cell_h = height / rows
        for ci in range(cols):
            c0, c1 = col_edges[ci], max(col_edges[ci + 1], col_edges[ci] + 1)
            for ri in range(rows):
                r0, r1 = row_edges[ri], max(row_edges[ri + 1], row_edges[ri] + 1)
                value = float(db[r0:r1, c0:c1].mean())
                t = float(np.clip((value - vmin) / span, 0.0, 1.0))
                color = self._lerp_color(C["surface_hi"], C["accent"], t)
                y = height - (ri + 1) * cell_h
                self.create_rectangle(ci * cell_w, y, (ci + 1) * cell_w + 1, y + cell_h + 1,
                                     fill=color, outline="")


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
        # Guards re-entrancy while a preset programmatically sets the pitch/
        # formant sliders, so that assignment isn't mistaken for the user
        # manually dragging a slider (which should fall back to "Custom").
        self._applying_preset = False
        self.premium_preview_enabled = False
        self.age = tk.DoubleVar(value=30)
        self.formant_ratio = tk.DoubleVar(value=1.0)
        self.remove_background = tk.BooleanVar(value=False)
        self.vault_real_path = tk.StringVar()
        self.vault_decoy_path = tk.StringVar()
        self.vault_container_path = tk.StringVar()
        # Separate variables per field on purpose: these used to share one
        # StringVar, so typing a passphrase to create a vault silently
        # overwrote whatever was typed in the unlock field (and vice versa).
        self.vault_create_passphrase = tk.StringVar()
        self.vault_unlock_passphrase = tk.StringVar()
        self.match_source_path = tk.StringVar()
        self.match_reference_path = tk.StringVar()
        self.use_ml_matching = tk.BooleanVar(value=False)
        self.write_naive = tk.BooleanVar(value=True)
        self.write_basic = tk.BooleanVar(value=True)
        self.write_locked = tk.BooleanVar(value=True)
        self.write_voice = tk.BooleanVar(value=True)
        self.frame_size = tk.IntVar(value=2048)
        self.hop_size = tk.IntVar(value=512)

        # Live / real-time state
        self.live_processor: LiveVoiceChanger | None = None
        self.live_running = False
        self.live_semitones = tk.DoubleVar(value=4.0)
        self.live_formant_ratio = tk.DoubleVar(value=1.0)
        self.live_output_device = tk.StringVar(value="Default output device")

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

        # brand — a slow "breathing" ring behind the glyph is the one purely
        # decorative animation in the app: small, constant, and confined to a
        # spot the eye already goes to, rather than something competing for
        # attention across the whole window.
        brand = tk.Frame(sidebar, bg=C["surface"])
        brand.pack(fill="x", pady=(24, 4))
        self._brand_canvas = tk.Canvas(brand, width=34, height=34, bg=C["surface"],
                                       bd=0, highlightthickness=0)
        self._brand_canvas.pack(side="left", padx=(24, 10))
        self._brand_pulse_phase = 0.0
        self._animate_brand_pulse()
        wrap = tk.Frame(brand, bg=C["surface"])
        wrap.pack(side="left")
        tk.Label(wrap, text="VOXSHIELD", bg=C["surface"], fg=C["text"],
                 font=("Segoe UI Semibold", 13)).pack(anchor="w")
        tk.Label(wrap, text="Phase Vocoder Studio", bg=C["surface"],
                 fg=C["text_faint"], font=F["small"]).pack(anchor="w")

        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill="x", pady=(20, 12), padx=20)

        # nav
        for name, icon in [("Home", "⌂"), ("Transform", "✦"), ("Match", "≈"),
                           ("Live", "◉"), ("Results", "◫"), ("Vault", "⌁"), ("Compare", "≋"),
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

    def _animate_brand_pulse(self) -> None:
        canvas = getattr(self, "_brand_canvas", None)
        if canvas is None or not canvas.winfo_exists():
            return
        import math
        self._brand_pulse_phase += 0.10
        t = (math.sin(self._brand_pulse_phase) + 1) / 2  # breathes between 0 and 1
        radius = 10 + t * 3
        ring_color = SpectrogramView._lerp_color(C["surface_hi"], C["accent_dim"], t)
        canvas.delete("all")
        canvas.create_oval(17 - radius, 17 - radius, 17 + radius, 17 + radius,
                           outline=ring_color, width=2)
        canvas.create_text(17, 17, text="◈", fill=C["accent"], font=("Segoe UI Symbol", 16))
        self.after(90, self._animate_brand_pulse)

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
            "Match": self._match_page, "Live": self._live_page,
            "Results": self._results_page,
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

        self._section_label(left, "WAVEFORM AND SPECTROGRAM")
        self.transform_waveform = WaveformView(left, height=56)
        self.transform_waveform.pack(fill="x", padx=22, pady=(0, 6))
        self.transform_spectrogram = SpectrogramView(left, height=110)
        self.transform_spectrogram.pack(fill="x", padx=22, pady=(0, 14))
        self._refresh_transform_visuals()

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
                                command=lambda _=None: self._on_manual_slider_change())
        pitch_scale.pack(fill="x", padx=22, pady=(0, 14))
        Tooltip(pitch_scale, "How many semitones to shift the voice. ±12 is one octave.")

        formant_row = tk.Frame(left, bg=C["surface"])
        formant_row.pack(fill="x", padx=22, pady=(0, 2))
        tk.Label(formant_row, text="Formant / character", bg=C["surface"], fg=C["text"],
                 font=F["body_b"]).pack(side="left")
        self.formant_label = tk.Label(formant_row, text="1.00×", bg=C["surface"],
                                      fg=C["accent"], font=F["mono_b"])
        self.formant_label.pack(side="right")
        formant_scale = ttk.Scale(left, from_=0.6, to=1.6, variable=self.formant_ratio,
                                  command=lambda _=None: self._on_manual_slider_change())
        formant_scale.pack(fill="x", padx=22, pady=(0, 18))
        Tooltip(formant_scale, "Moves formants independently of pitch: below 1.0 sounds like a "
                               "larger vocal tract, above 1.0 a smaller one. 1.0 leaves them untouched.")

        self._update_values()

        # RIGHT: presets + outputs + action
        right_border, right = make_card(columns)
        right_border.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._section_label(right, "CREATIVE PRESET")
        preset_row = tk.Frame(right, bg=C["surface"])
        preset_row.pack(fill="x", padx=22, pady=(4, 4))
        self.preset_menu = ttk.Combobox(preset_row, textvariable=self.preset_name,
                                        values=list(PRESETS), state="readonly")
        self.preset_menu.pack(side="left", fill="x", expand=True)
        self.preset_menu.bind("<<ComboboxSelected>>", self._preset_changed)
        Tooltip(self.preset_menu, "Picking a preset immediately sets the pitch and formant sliders on the "
                                  "left to its values — Preview and Generate always use whatever those "
                                  "sliders currently show. Moving a slider by hand switches back to Custom.")
        HoverButton(preset_row, "Reset all", command=self._reset_parameters, kind="solid", icon="↺",
                    tooltip="Reset every slider, preset, checkbox, and setting to its default."
                    ).pack(side="left", padx=(8, 0))

        preview_row = tk.Frame(right, bg=C["surface"])
        preview_row.pack(fill="x", padx=22, pady=(0, 8))
        HoverButton(preview_row, "Preview", command=self._preview_preset, icon="▶",
                    tooltip="Hear the current pitch/formant sliders applied to the loaded audio."
                    ).pack(side="left")
        HoverButton(preview_row, "Pause", command=self._pause_playback, icon="⏸",
                    tooltip="Pause preview playback.").pack(side="left", padx=(6, 0))

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
                              command=lambda _=None: self._on_age_changed())
        age_scale.pack(fill="x", padx=22)
        Tooltip(age_scale, "Only affects Feminine-style and Masculine-style presets — re-applies that "
                           "preset's pitch/formant at the new age.")

        ttk.Checkbutton(right, text="Attempt background-music reduction (experimental)",
                        variable=self.remove_background).pack(anchor="w", padx=22, pady=(14, 8))

        self._section_label(right, "OUTPUTS TO WRITE")
        output_info = [
            (self.write_naive,  "Naive resampling baseline",
             "Straight resample. Pitch and duration move together — the wrong-sounding reference."),
            (self.write_basic,  "Phase vocoder",
             "STFT stretch with phase accumulation. Pitch preserved, may sound watery."),
            (self.write_locked, "Phase-locked vocoder",
             "Bins around each spectral peak share the peak's phase correction. Cleaner sustains."),
            (self.write_voice,  "Voice transformation",
             "Applies the pitch and formant offsets, then the duration factor too if it isn't 1.0×."),
        ]
        for variable, label, tip in output_info:
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
            "The matcher estimates the reference's median pitch and its averaged LPC spectral "
            "envelope (a lightweight model of vocal-tract resonance), then applies a "
            "formant-preserving pitch shift and blends the source's own envelope shape toward "
            "the reference's.\nIt cannot copy vocal identity — only nudge the source toward the "
            "reference's general register and resonant character."
        ), bg=C["surface"], fg=C["text_dim"], font=F["small"],
        wraplength=760, justify="left").pack(anchor="w", padx=22, pady=(6, 14))

        self._section_label(card, "ML VOICE MATCHING (EXPERIMENTAL)")
        tk.Label(card, text=(
            "An optional neural alternative (kNN-VC) that sounds considerably closer to the reference "
            "than the LPC matcher above — at the cost of PyTorch, a one-time model download over the "
            "internet, and real compute per conversion. Check your device before turning it on."
        ), bg=C["surface"], fg=C["text_dim"], font=F["small"],
        wraplength=760, justify="left").pack(anchor="w", padx=22, pady=(4, 8))

        ml_row = tk.Frame(card, bg=C["surface"])
        ml_row.pack(fill="x", padx=22, pady=(0, 4))
        ttk.Checkbutton(ml_row, text="Use ML voice matching instead of LPC",
                        variable=self.use_ml_matching).pack(side="left")
        HoverButton(ml_row, "Check my device", command=self._check_ml_capability,
                    kind="solid", icon="🖥",
                    tooltip="Detect PyTorch/GPU and estimate how long a conversion would take here."
                    ).pack(side="left", padx=(10, 0))

        self.ml_capability_label = tk.Label(
            card, text="Not checked yet. Press \"Check my device\" before enabling this.",
            bg=C["surface"], fg=C["text_faint"], font=F["small"], wraplength=760, justify="left",
        )
        self.ml_capability_label.pack(anchor="w", padx=22, pady=(6, 14))

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

    def _check_ml_capability(self) -> None:
        duration_seconds = 8.0
        try:
            source = Path(self.match_source_path.get())
            if self.current_audio is not None:
                signal, rate = self.current_audio
                duration_seconds = len(signal) / rate
            elif source.is_file():
                signal, rate = read_audio(source)
                duration_seconds = len(signal) / rate
        except Exception:
            pass  # best-effort; the estimate just falls back to a generic clip length

        report = capability_report(duration_seconds)
        lines = [report.reason]
        if report.torch_available:
            ram_text = f"{report.ram_gb:.1f} GB RAM" if report.ram_gb else "RAM unknown"
            lines.append(f"{report.cpu_count} CPU cores, {ram_text}, GPU: {'yes' if report.gpu_available else 'no'}.")
            lines.append(f"Estimated time for ~{duration_seconds:.0f}s of audio: about {report.estimated_seconds:.0f}s "
                         f"(first run also downloads model weights — add a couple of minutes for that).")
        color = {"fast": C["success"], "workable": C["success"], "slow": C["accent"],
                 "unavailable": C["danger"]}.get(report.recommendation, C["text_dim"])
        if hasattr(self, "ml_capability_label") and self.ml_capability_label.winfo_exists():
            self.ml_capability_label.configure(text=" ".join(lines), fg=color)

    # ==================================================================
    # LIVE (near-real-time)
    # ==================================================================
    def _live_page(self) -> None:
        body = self._scroll_page()
        self._page_header(
            body,
            title="Live",
            subtitle="Apply pitch and formant shifting to your microphone in near real time.",
            eyebrow="LIVE · EXPERIMENTAL",
        )

        note_border, note = make_card(body)
        note_border.pack(fill="x", padx=42, pady=(0, 16))
        wrap = tk.Frame(note, bg=C["surface"])
        wrap.pack(fill="x", padx=22, pady=16)
        tk.Label(wrap, text="How this differs from Transform", bg=C["surface"],
                 fg=C["text"], font=F["h2"]).pack(anchor="w")
        tk.Label(wrap, text=(
            "Transform processes a whole recording at once. Live instead chops the microphone feed into short "
            "overlapping blocks, runs each one through the same pitch/formant pipeline, and crossfades them back "
            "together — trading a bit of latency (about one block, shown below) for something you can talk "
            "through live. To use this inside a call app (Zoom, Discord, Meet…), install a virtual audio cable "
            "(e.g. VB-CABLE on Windows), pick it as the output device below, then select that same cable as the "
            "microphone inside the call app."
        ), bg=C["surface"], fg=C["text_dim"], font=F["body"], wraplength=800,
        justify="left").pack(anchor="w", pady=(6, 0))

        call_border, call = make_card(body)
        call_border.pack(fill="x", padx=42, pady=(0, 16))
        self._section_label(call, "CALL ROUTING · WHATSAPP, MESSENGER, DISCORD, ZOOM…")
        self.call_status_label = tk.Label(
            call, text="Checking for a virtual audio cable…",
            bg=C["surface"], fg=C["text_dim"], font=F["body"], wraplength=800, justify="left",
        )
        self.call_status_label.pack(anchor="w", padx=22, pady=(4, 10))
        call_action = tk.Frame(call, bg=C["surface"])
        call_action.pack(fill="x", padx=22, pady=(0, 18))
        HoverButton(call_action, "Rescan", command=self._rescan_call_routing, kind="solid", icon="↻",
                    tooltip="Look again after installing a virtual audio cable."
                    ).pack(side="left")
        self.call_start_button = HoverButton(
            call_action, "Start call mode", command=self._start_call_mode, kind="primary", icon="📞",
            tooltip="Start Live processing routed straight into the detected virtual cable.",
        )
        self.call_start_button.pack(side="left", padx=(8, 0))

        columns = tk.Frame(body, bg=C["bg"])
        columns.pack(fill="both", expand=True, padx=42, pady=(0, 30))
        columns.columnconfigure(0, weight=1, uniform="col")
        columns.columnconfigure(1, weight=1, uniform="col")

        # LEFT: controls
        left_border, left = make_card(columns)
        left_border.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._section_label(left, "OUTPUT DEVICE")

        device_row = tk.Frame(left, bg=C["surface"])
        device_row.pack(fill="x", padx=22, pady=(4, 4))
        self._live_devices = list_output_devices()
        device_names = [name for _, name in self._live_devices] or ["No output devices found"]
        self.live_device_menu = ttk.Combobox(device_row, textvariable=self.live_output_device,
                                             values=device_names, state="readonly")
        self.live_device_menu.pack(side="left", fill="x", expand=True)
        if device_names and self.live_output_device.get() not in device_names:
            self.live_output_device.set(device_names[0])
        Tooltip(self.live_device_menu, "Pick your speakers/headphones to monitor yourself, or a virtual "
                                       "audio cable to feed a call app.")
        HoverButton(device_row, "Refresh", command=self._refresh_live_devices, kind="solid", icon="↻",
                    tooltip="Rescan audio devices.").pack(side="left", padx=(8, 0))

        self._section_label(left, "LIVE PITCH AND FORMANT")
        live_pitch_row = tk.Frame(left, bg=C["surface"])
        live_pitch_row.pack(fill="x", padx=22, pady=(4, 2))
        tk.Label(live_pitch_row, text="Pitch offset", bg=C["surface"], fg=C["text"],
                 font=F["body_b"]).pack(side="left")
        self.live_pitch_label = tk.Label(live_pitch_row, text="+4.0 st", bg=C["surface"],
                                         fg=C["accent"], font=F["mono_b"])
        self.live_pitch_label.pack(side="right")
        ttk.Scale(left, from_=-8, to=8, variable=self.live_semitones,
                  command=lambda _=None: self._update_live_values()).pack(fill="x", padx=22, pady=(0, 12))

        live_formant_row = tk.Frame(left, bg=C["surface"])
        live_formant_row.pack(fill="x", padx=22, pady=(0, 2))
        tk.Label(live_formant_row, text="Formant / character", bg=C["surface"], fg=C["text"],
                 font=F["body_b"]).pack(side="left")
        self.live_formant_label = tk.Label(live_formant_row, text="1.00×", bg=C["surface"],
                                           fg=C["accent"], font=F["mono_b"])
        self.live_formant_label.pack(side="right")
        ttk.Scale(left, from_=0.6, to=1.6, variable=self.live_formant_ratio,
                  command=lambda _=None: self._update_live_values()).pack(fill="x", padx=22, pady=(0, 18))

        action = tk.Frame(left, bg=C["surface"])
        action.pack(fill="x", padx=22, pady=(0, 20))
        self.live_start_button = HoverButton(action, "Start live processing", command=self._start_live,
                                             kind="primary", icon="◉",
                                             tooltip="Open the microphone and start shifting it live.")
        self.live_start_button.pack(side="left")
        self.live_stop_button = HoverButton(action, "Stop", command=self._stop_live, kind="danger", icon="■",
                                            tooltip="Stop live processing and close the audio devices.")
        self.live_stop_button.pack(side="left", padx=(8, 0))
        self.live_stop_button.set_enabled(False)

        # RIGHT: status
        right_border, right = make_card(columns)
        right_border.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._section_label(right, "STATUS")
        self.live_status_label = tk.Label(
            right, text="Idle. Choose a device and press Start.",
            bg=C["surface"], fg=C["text_dim"], font=F["body"], wraplength=340, justify="left",
        )
        self.live_status_label.pack(anchor="w", padx=22, pady=(4, 16))

        self._update_live_values()
        self._detect_call_routing()

    def _detect_call_routing(self) -> None:
        """Look for an installed virtual audio cable and update the Live page.

        This is what makes routing into WhatsApp/Messenger/etc. a single
        button instead of "go find the right device name in a dropdown":
        if a cable is already installed, it's auto-selected and the person
        just needs to pick its name inside the call app's mic settings.
        """
        cable = detect_virtual_cable()
        self._call_cable = cable
        has_label = hasattr(self, "call_status_label") and self.call_status_label.winfo_exists()
        has_button = hasattr(self, "call_start_button") and self.call_start_button.winfo_exists()
        if cable is not None:
            _, name = cable
            if has_label:
                self.call_status_label.configure(
                    text=(f"✓ Found a virtual audio cable: \"{name}\". Press \"Start call mode\", then in "
                          f"WhatsApp/Messenger/Discord's call settings, set the microphone to \"{name}\" — "
                          f"they'll hear the shifted voice instead of your real mic."),
                    fg=C["success"],
                )
            if has_button:
                self.call_start_button.set_enabled(True)
        else:
            if has_label:
                self.call_status_label.configure(
                    text=("No virtual audio cable detected. These apps can't accept audio from another "
                          "program directly — install a free one (search \"VB-CABLE\", vb-audio.com; or "
                          "VoiceMeeter for more routing options), then press Rescan. It installs a virtual "
                          "microphone/speaker pair: this app plays the shifted voice into it, and the call "
                          "app picks it up as if it were a real microphone."),
                    fg=C["text_dim"],
                )
            if has_button:
                self.call_start_button.set_enabled(False)

    def _rescan_call_routing(self) -> None:
        self._refresh_live_devices()
        self._detect_call_routing()

    def _start_call_mode(self) -> None:
        cable = getattr(self, "_call_cable", None)
        if cable is None:
            return
        _, name = cable
        if name in [n for _, n in getattr(self, "_live_devices", [])]:
            self.live_output_device.set(name)
        self._start_live()

    def _update_live_values(self) -> None:
        if hasattr(self, "live_pitch_label") and self.live_pitch_label.winfo_exists():
            self.live_pitch_label.configure(text=f"{self.live_semitones.get():+.1f} st")
        if hasattr(self, "live_formant_label") and self.live_formant_label.winfo_exists():
            self.live_formant_label.configure(text=f"{self.live_formant_ratio.get():.2f}×")
        if self.live_processor is not None:
            self.live_processor.set_parameters(self.live_semitones.get(), self.live_formant_ratio.get())

    def _refresh_live_devices(self) -> None:
        self._live_devices = list_output_devices()
        names = [name for _, name in self._live_devices] or ["No output devices found"]
        if hasattr(self, "live_device_menu") and self.live_device_menu.winfo_exists():
            self.live_device_menu.configure(values=names)
        if names:
            self.live_output_device.set(names[0])

    def _start_live(self) -> None:
        if self.live_running:
            return
        device_index = None
        for index, name in getattr(self, "_live_devices", []):
            if name == self.live_output_device.get():
                device_index = index
                break
        try:
            config = StftConfig(frame_size=1024, hop_size=256,
                                sample_rate=self.record_sample_rate.get())
            self.live_processor = LiveVoiceChanger(sample_rate=config.sample_rate, config=config)
            self.live_processor.set_parameters(self.live_semitones.get(), self.live_formant_ratio.get())
            self.live_processor.start(output_device=device_index)
        except Exception as error:
            self.live_processor = None
            messagebox.showerror("Live mode failed", str(error))
            return
        self.live_running = True
        if hasattr(self, "live_start_button") and self.live_start_button.winfo_exists():
            self.live_start_button.set_enabled(False)
        if hasattr(self, "live_stop_button") and self.live_stop_button.winfo_exists():
            self.live_stop_button.set_enabled(True)
        if hasattr(self, "live_status_label") and self.live_status_label.winfo_exists():
            latency_ms = self.live_processor.latency_seconds * 1000
            self.live_status_label.configure(
                text=f"Running. Approximate latency: {latency_ms:.0f} ms per block.",
                fg=C["success"],
            )

    def _stop_live(self) -> None:
        if self.live_processor is not None:
            self.live_processor.stop()
        self.live_processor = None
        self.live_running = False
        if hasattr(self, "live_start_button") and self.live_start_button.winfo_exists():
            self.live_start_button.set_enabled(True)
        if hasattr(self, "live_stop_button") and self.live_stop_button.winfo_exists():
            self.live_stop_button.set_enabled(False)
        if hasattr(self, "live_status_label") and self.live_status_label.winfo_exists():
            self.live_status_label.configure(text="Stopped.", fg=C["text_dim"])

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
        pass_entry_a = tk.Entry(create, textvariable=self.vault_create_passphrase, show="●",
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
        pass_entry_b = tk.Entry(unlock, textvariable=self.vault_unlock_passphrase, show="●",
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
             "Applies an independent pitch offset and formant offset (vocal-tract size), via LPC envelope "
             "warping, then the duration factor. Compare pitch-only vs. pitch+formant on the same sentence to "
             "hear how much of 'masculine/feminine' character actually comes from formants."),
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

            ("09", "LPC: separating the vocal tract from pitch",
             "Pitch shifting above moves the *whole* spectrum, so formants (the vocal tract's resonances) "
             "move with it — that's the 'chipmunk' artifact at large offsets. Linear Predictive Coding fits a "
             "small all-pole filter to each frame's autocorrelation (Levinson-Durbin); dividing the frame's "
             "spectrum by that filter's response leaves a flat 'residual' carrying only pitch harmonics. "
             "Warping the filter's response along frequency and multiplying it back in moves formants alone.",
             """r = autocorrelate(frame)[: order + 1]        # Yule-Walker equations
a, error = levinson_durbin(r, order)          # all-pole coefficients
envelope = sqrt(error) / abs(rfft([1, *a], n=frame_size))

residual = spectrum / envelope                # pitch harmonics, formants flattened
warped   = interp(bins / ratio, bins, envelope)   # ratio > 1 raises formants
new_spectrum = residual * warped"""),
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
        # Selecting a preset used to do nothing to Preview/Generate unless a
        # separate, easy-to-miss checkbox was also ticked — that's why
        # Masculine/Feminine/age looked broken. Now picking a preset writes
        # straight into the actual pitch/formant sliders, so whatever those
        # sliders show is always exactly what gets previewed or rendered.
        self._apply_preset_to_sliders()

    def _apply_preset_to_sliders(self) -> None:
        """Push the selected named preset's (age-adjusted) values onto the
        real pitch/formant sliders. A no-op for "Custom", which just means
        "use whatever the sliders currently hold."
        """
        preset = PRESETS[self.preset_name.get()]
        if preset["semitones"] is None:
            return
        semitones = float(preset["semitones"])
        formant_ratio = float(preset["formant_ratio"])
        if self.preset_name.get() in {"Feminine-style", "Masculine-style"}:
            age_delta = self.age.get() - 30
            semitones -= age_delta * 0.03
            # Formant ratio is multiplicative, so nudge it in log space.
            formant_ratio *= 2 ** (-age_delta * 0.006)
        self._applying_preset = True
        try:
            self.semitones.set(round(semitones, 2))
            self.formant_ratio.set(round(formant_ratio, 3))
        finally:
            self._applying_preset = False
        self._update_values()

    def _on_manual_slider_change(self) -> None:
        """Dragging pitch/formant by hand overrides whatever preset was picked."""
        if not self._applying_preset and self.preset_name.get() != "Custom":
            self.preset_name.set("Custom")
            if hasattr(self, "preset_description") and self.preset_description.winfo_exists():
                self.preset_description.configure(text=PRESETS["Custom"]["description"])
        self._update_values()

    def _on_age_changed(self) -> None:
        if self.preset_name.get() in {"Feminine-style", "Masculine-style"}:
            self._apply_preset_to_sliders()
        else:
            self._update_values()

    def _active_transform(self) -> tuple[float, float]:
        """The (semitones, formant_ratio) pair Preview/Generate use — always

        just whatever the pitch/formant sliders currently show, since picking
        a preset writes straight into them (see `_apply_preset_to_sliders`).
        """
        return self.semitones.get(), self.formant_ratio.get()

    def _active_semitones(self) -> float:
        return self._active_transform()[0]

    def _reset_parameters(self) -> None:
        """Return every adjustable control to its startup default."""
        self.stretch.set(1.5)
        self.preset_name.set("Custom")
        self.semitones.set(4.0)
        self.formant_ratio.set(1.0)
        self.age.set(30)
        if hasattr(self, "preset_description") and self.preset_description.winfo_exists():
            self.preset_description.configure(text=PRESETS["Custom"]["description"])
        self.remove_background.set(False)
        self.normalize_output.set(False)
        self.write_naive.set(True)
        self.write_basic.set(True)
        self.write_locked.set(True)
        self.write_voice.set(True)
        self.frame_size.set(2048)
        self.hop_size.set(512)
        self.record_sample_rate.set(44100)
        self.live_semitones.set(4.0)
        self.live_formant_ratio.set(1.0)
        self.use_ml_matching.set(False)
        if hasattr(self, "ml_capability_label") and self.ml_capability_label.winfo_exists():
            self.ml_capability_label.configure(
                text="Not checked yet. Press \"Check my device\" before enabling this.",
                fg=C["text_faint"],
            )
        self._update_values()
        self._update_live_values()
        self._set_status("All parameters reset to defaults.")

    def _preview_preset(self) -> None:
        if self.preview_playing and self.player is not None and self.player.paused:
            self.player.resume()
            self._set_status("Preset preview resumed.")
            return
        source = Path(self.input_path.get())
        if self.current_audio is None and not source.is_file():
            messagebox.showerror("Choose audio", "Select or record audio before previewing a preset.")
            return
        semitones, formant_ratio = self._active_transform()
        if semitones == 0 and abs(formant_ratio - 1.0) < 1e-6:
            messagebox.showinfo("Choose a transformation",
                                "Choose a preset, or a non-zero pitch/formant offset, to preview.")
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
            args=(input_data, semitones, formant_ratio, config, self.remove_background.get()),
            daemon=True).start()

    def _preview_worker(self, input_data, semitones, formant_ratio, config, remove_background) -> None:
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
            transformed = anonymize_voice(signal, semitones, config, formant_ratio=formant_ratio)
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
        active_semitones, active_formant_ratio = self._active_transform()
        if self.write_voice.get() and active_semitones == 0 and abs(active_formant_ratio - 1.0) < 1e-6:
            messagebox.showerror("Choose a transformation",
                                 "Voice transformation needs a non-zero pitch or formant offset.")
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
                   active_formant_ratio, config, self.write_naive.get(), self.write_basic.get(),
                   self.write_locked.get(), self.write_voice.get(),
                   self.normalize_output.get(), self.remove_background.get())
        threading.Thread(target=self._process_worker, args=options, daemon=True).start()

    def _process_worker(self, input_data, destination, stretch, semitones, formant_ratio, config,
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
            if voice:
                # The duration slider used to have no effect on this output —
                # it only reached the naive/vocoder/locked jobs above, while
                # the voice-transform path (anonymize_voice) always restored
                # the original duration internally. Compose the two: shift
                # pitch/formants first, then stretch the result if asked.
                voice_signal = anonymize_voice(signal, semitones, config, formant_ratio=formant_ratio)
                if abs(stretch - 1.0) > 1e-3:
                    voice_signal = time_stretch(voice_signal, stretch, config, phase_locking=True)
                jobs.append((f"{stem}_voice_transform_{semitones:+g}st_{stretch:g}x.wav", voice_signal))
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
        use_ml = self.use_ml_matching.get()
        if hasattr(self, "match_progress"):
            self.match_progress.start("Running ML voice matching… (this can take a while)" if use_ml else "Matching…")
        if preview:
            threading.Thread(target=self._match_worker,
                             args=(source, reference, config, None, use_ml),
                             daemon=True).start()
        else:
            session = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root))
            self.temp_output_dirs.append(session)
            threading.Thread(target=self._match_worker,
                             args=(source, reference, config, session / "reference_guided_match.wav", use_ml),
                             daemon=True).start()

    def _match_worker(self, source, reference_path, config, destination, use_ml=False) -> None:
        try:
            if isinstance(source, Path):
                source_samples, source_rate = read_audio(source)
            else:
                source_samples, source_rate = source
            reference_samples, reference_rate = read_audio(reference_path)

            if use_ml:
                # The neural path resamples internally to its own 16 kHz
                # working rate and returns audio at that rate — distinct from
                # the LPC path, which stays at the source's own sample rate.
                started = time.monotonic()
                matched, matched_rate = MLVoiceConverter().convert(
                    source_samples, source_rate, reference_samples, reference_rate)
                elapsed = time.monotonic() - started
                message = f"ML voice matching finished in {elapsed:.0f}s."
            else:
                config = StftConfig(sample_rate=source_rate,
                                    frame_size=config.frame_size,
                                    hop_size=config.hop_size)
                matched, profile = match_voice_character(
                    source_samples, source_rate, reference_samples, reference_rate, config)
                matched_rate = source_rate
                shift = profile["pitch_shift_semitones"]
                message = f"Reference match used {shift:+.1f} semitones of pitch adjustment (LPC)."

            if destination is None:
                self.preview_queue.put((True, (matched, matched_rate)))
            else:
                write_output_wav(destination, matched, matched_rate)
                self.result_queue.put((True, message, [destination]))
        except Exception as error:
            failure = str(error)
            if use_ml:
                failure += " (Uncheck \"Use ML voice matching\" to fall back to the always-available LPC matcher.)"
            if destination is None:
                self.preview_queue.put((False, failure))
            else:
                self.result_queue.put((False, failure, []))

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
        if len(self.vault_create_passphrase.get()) < 8:
            messagebox.showerror("Passphrase required", "Use a passphrase of at least 8 characters.")
            return
        session = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root))
        self.temp_output_dirs.append(session)
        destination = session / "voice_lab_vault.wav"
        real_input = self.current_audio if self.current_audio is not None else real_path
        threading.Thread(target=self._vault_create_worker,
                         args=(real_input, decoy, self.vault_create_passphrase.get(), destination),
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
        if not vault.is_file() or len(self.vault_unlock_passphrase.get()) < 8:
            messagebox.showerror("Vault details required", "Choose a vault WAV and enter its passphrase.")
            return
        session = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root))
        self.temp_output_dirs.append(session)
        destination = session / "unlocked_audio.wav"
        threading.Thread(target=self._vault_unlock_worker,
                         args=(vault, self.vault_unlock_passphrase.get(), destination),
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
        if hasattr(self, "formant_label") and self.formant_label.winfo_exists():
            self.formant_label.configure(text=f"{self.formant_ratio.get():.2f}×")
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
            self._refresh_transform_visuals()

    def _refresh_transform_visuals(self) -> None:
        """Update the Transform page's waveform/spectrogram from the current input.

        Best-effort and silent on failure — this is a visualization, not a
        step anything downstream depends on, so a bad path or unreadable
        file just clears the view instead of raising a dialog.
        """
        has_waveform = hasattr(self, "transform_waveform") and self.transform_waveform.winfo_exists()
        has_spectrogram = hasattr(self, "transform_spectrogram") and self.transform_spectrogram.winfo_exists()
        if not has_waveform and not has_spectrogram:
            return
        signal = sample_rate = None
        try:
            if self.current_audio is not None:
                signal, sample_rate = self.current_audio
            else:
                source = Path(self.input_path.get())
                if source.is_file():
                    signal, sample_rate = read_audio(source)
        except Exception:
            signal = None
        if has_waveform:
            self.transform_waveform.set_samples(signal)
        if has_spectrogram:
            self.transform_spectrogram.set_samples(signal, sample_rate or 44_100)

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
            self._refresh_transform_visuals()
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
        self._refresh_transform_visuals()

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
            self.player = AudioPlayer(samples, sample_rate)
            self.player.set_volume(self.volume.get())
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

    def _on_close(self) -> None:
        # Live mode owns a real microphone/speaker stream — leaving it open
        # after the window closes would be a real surprise, not just a
        # cosmetic bug, so it's stopped explicitly alongside playback.
        self._stop_live()
        self._stop_playback()
        if self.recorder is not None:
            try:
                self.recorder.stop()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = VoiceLab()
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.mainloop()