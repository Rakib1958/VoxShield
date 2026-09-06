"""Desktop interface for the VoxShield.

Run with: python desktop_app.py
No packages beyond NumPy are required; the UI uses Python's built-in tkinter.
"""

from __future__ import annotations

import queue
import tkinter as tk
import shutil
import tempfile
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from audio_io import AudioPlayer, Recorder, read_audio, write_output_wav
from phase_vocoder import match_voice_character, reduce_background_estimate, StftConfig, anonymize_voice, naive_time_stretch, time_stretch
from secure_audio import create_vault, unlock_vault


COLORS = {
    "background": "#0B0B0B", "panel": "#151515", "panel_alt": "#242424",
    "accent": "#DD933D", "accent_hover": "#96723D", "text": "#F5F3ED",
    "muted": "#A7A49D", "border": "#3B3B3B", "success": "#DFFF3F",
}

PRESETS = {
    "Custom": {"semitones": None, "description": "Use the pitch slider directly."},
    "Low pitch": {"semitones": -4, "description": "A lower, weightier pitch transformation."},
    "High pitch": {"semitones": 4, "description": "A brighter, higher pitch transformation."},
    "Feminine-style": {"semitones": 3, "description": "A higher-pitch creative style with an optional age character control."},
    "Masculine-style": {"semitones": -3, "description": "A lower-pitch creative style with an optional age character control."},
    "🔒 Cyber Oracle": {"semitones": 7, "description": "Premium preview: a dramatic high-register sci-fi voice.", "premium": True},
    "🔒 Deep Space": {"semitones": -7, "description": "Premium preview: a dramatic low-register sci-fi voice.", "premium": True},
}


class VoiceLab(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VoxShield")
        self.geometry("1120x720")
        self.minsize(900, 620)
        self.configure(bg=COLORS["background"])
        try:
            app_icon = tk.PhotoImage(file="HackerIcon.png")
            self.iconphoto(True, app_icon)
        except Exception as error:
            print("Icon not found")
        self.result_queue: queue.Queue[tuple[bool, str, list[Path]]] = queue.Queue()
        self.preview_queue: queue.Queue[tuple[bool, object]] = queue.Queue()
        self.busy = False
        self.current_audio: tuple[object, int] | None = None
        self.last_output: Path | None = None
        self.recorder: Recorder | None = None
        self.player: AudioPlayer | None = None
        self.preview_playing = False
        self.temp_recording: Path | None = None
        self.recording_paused = False
        self.volume = tk.DoubleVar(value=1.0)
        self.temp_root = Path(tempfile.gettempdir()) / "phase_vocoder_voice_lab"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_output_dirs: list[Path] = []
        self.result_files: list[Path] = []

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "outputs"))
        self.stretch = tk.DoubleVar(value=1.5)
        self.semitones = tk.DoubleVar(value=4.0)
        self.record_seconds = tk.IntVar(value=5)
        self.record_sample_rate = tk.IntVar(value=44_100)
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

        self._configure_style()
        self._build_shell()
        self.show_page("Home")
        self.after(100, self._poll_results)
        self.after(300, self._recover_temp_outputs)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["background"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["background"], foreground=COLORS["text"], font=("Arial", 10))
        style.configure("Muted.TLabel", foreground=COLORS["muted"], font=("Arial", 10))
        style.configure("Title.TLabel", foreground=COLORS["text"], font=("Arial", 25, "bold"))
        style.configure("Heading.TLabel", foreground=COLORS["text"], font=("Arial", 15, "bold"))
        style.configure("Card.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Arial", 12, "bold"))
        style.configure("CardMuted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=("Arial", 10))
        style.configure("TButton", background=COLORS["panel_alt"], foreground=COLORS["text"], borderwidth=1, padding=(14, 9), font=("Arial", 10, "bold"))
        style.map("TButton", background=[("active", COLORS["border"])], foreground=[("active", COLORS["text"])])
        style.configure("Accent.TButton", background=COLORS["accent"], foreground="#101010")
        style.map("Accent.TButton", background=[("active", COLORS["accent_hover"])])
        style.configure("TEntry", fieldbackground=COLORS["panel_alt"], foreground=COLORS["text"], insertcolor=COLORS["text"], bordercolor=COLORS["border"], padding=7)
        style.configure("TCheckbutton", background=COLORS["panel"], foreground=COLORS["text"], font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", COLORS["panel"])], foreground=[("active", COLORS["text"])])
        style.configure("Horizontal.TScale", background=COLORS["panel"], troughcolor=COLORS["border"])

    def _build_shell(self) -> None:
        sidebar = tk.Frame(self, bg=COLORS["panel"], width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="◈  VOICE LAB", bg=COLORS["panel"], fg=COLORS["text"], font=("Segoe UI Semibold", 16)).pack(anchor="w", padx=24, pady=(28, 5))
        tk.Label(sidebar, text="Phase vocoder studio", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 9)).pack(anchor="w", padx=24, pady=(0, 28))
        self.nav_buttons: dict[str, tk.Button] = {}
        for name, symbol in [("Home", "⌂"), ("Transform", "✦"), ("Match", "≈"), ("Results", "◫"), ("Vault", "⌁"), ("Compare", "≋"), ("Learn", "◌"), ("Settings", "⚙")]:
            button = tk.Button(sidebar, text=f"  {symbol}   {name}", anchor="w", command=lambda n=name: self.show_page(n), relief="flat", bd=0, bg=COLORS["panel"], activebackground=COLORS["panel_alt"], fg=COLORS["muted"], activeforeground=COLORS["text"], font=("Segoe UI Semibold", 10), padx=24, pady=11)
            button.pack(fill="x", pady=1)
            button.bind("<Enter>", lambda event, b=button, n=name: self._nav_hover(b, n, True))
            button.bind("<Leave>", lambda event, b=button, n=name: self._nav_hover(b, n, False))
            self.nav_buttons[name] = button
        tk.Label(sidebar, text="Local processing only\nWAV files stay on this computer.", justify="left", bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 9)).pack(side="bottom", anchor="w", padx=24, pady=25)
        self.content = tk.Frame(self, bg=COLORS["background"])
        self.content.pack(side="left", fill="both", expand=True)

    def _nav_hover(self, button: tk.Button, name: str, entered: bool) -> None:
        if name != getattr(self, "active_page", None):
            button.configure(bg=COLORS["panel_alt"] if entered else COLORS["panel"])

    def show_page(self, name: str) -> None:
        self.active_page = name
        for child in self.content.winfo_children():
            child.destroy()
        for page, button in self.nav_buttons.items():
            button.configure(bg=COLORS["accent"] if page == name else COLORS["panel"], fg="white" if page == name else COLORS["muted"])
        builders = {"Home": self._home_page, "Transform": self._transform_page, "Match": self._match_page, "Results": self._results_page, "Vault": self._vault_page, "Compare": self._compare_page, "Learn": self._learn_page, "Settings": self._settings_page}
        builders[name]()

    def _page_header(self, title: str, subtitle: str) -> tk.Frame:
        header = tk.Frame(self.content, bg=COLORS["background"])
        header.pack(fill="x", padx=42, pady=(36, 20))
        ttk.Label(header, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text=subtitle, style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        return header

    def _card(self, parent: tk.Misc) -> tk.Frame:
        return tk.Frame(parent, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)

    def _home_page(self) -> None:
        self._page_header("Welcome to Voice Lab", "Explore phase-vocoder transformations without leaving your computer.")
        cards = tk.Frame(self.content, bg=COLORS["background"])
        cards.pack(fill="x", padx=42)
        for title, body in [
            ("1  Choose a voice", "Use a 16-bit PCM mono WAV. A short spoken sentence is perfect for comparison."),
            ("2  Generate variants", "Create naive, phase-vocoder, and phase-locked transformations in one run."),
            ("3  Listen critically", "Compare sustained vowels, consonants, pitch, timing, and audible artifacts."),
        ]:
            card = self._card(cards)
            card.pack(side="left", fill="both", expand=True, padx=6, ipady=14)
            ttk.Label(card, text=title, style="Card.TLabel").pack(anchor="w", padx=18, pady=(12, 8))
            ttk.Label(card, text=body, style="CardMuted.TLabel", wraplength=205, justify="left").pack(anchor="w", padx=18, pady=(0, 12))
        hero = self._card(self.content)
        hero.pack(fill="x", padx=48, pady=30)
        ttk.Label(hero, text="Ready to make your first comparison?", style="Card.TLabel").pack(anchor="w", padx=24, pady=(22, 6))
        ttk.Label(hero, text="Start in Transform to select a WAV file and choose your outputs.", style="CardMuted.TLabel").pack(anchor="w", padx=24)
        ttk.Button(hero, text="Open Transform  →", style="Accent.TButton", command=lambda: self.show_page("Transform")).pack(anchor="w", padx=24, pady=20)

    def _path_row(self, parent: tk.Misc, label: str, variable: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").pack(anchor="w", padx=20, pady=(18, 6))
        row = tk.Frame(parent, bg=COLORS["panel"])
        row.pack(fill="x", padx=20, pady=(0, 10))
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse", command=command).pack(side="left", padx=(8, 0))

    def _transform_page(self) -> None:
        self._page_header("Transform", "Create listening comparisons and a fixed-duration voice transformation.")
        canvas = tk.Frame(self.content, bg=COLORS["background"])
        canvas.pack(fill="both", expand=True, padx=42)
        left, right = self._card(canvas), self._card(canvas)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._path_row(left, "Input audio", self.input_path, self._choose_input)
        input_actions = tk.Frame(left, bg=COLORS["panel"])
        input_actions.pack(fill="x", padx=20, pady=(0, 8))
        self.record_button = ttk.Button(input_actions, text="● Record", style="Accent.TButton", command=self._start_recording)
        self.record_button.pack(side="left")
        self.record_pause_button = ttk.Button(input_actions, text="Ⅱ Pause", command=self._toggle_record_pause)
        self.record_pause_button.pack(side="left", padx=(7, 0))
        self.record_stop_button = ttk.Button(input_actions, text="■ Stop", command=self._stop_recording)
        self.record_stop_button.pack(side="left", padx=(7, 0))
        time_row = tk.Frame(left, bg=COLORS["panel"]); time_row.pack(fill="x", padx=20, pady=(0, 10))
        ttk.Label(time_row, text="Recording limit", style="CardMuted.TLabel").pack(side="left")
        ttk.Entry(time_row, textvariable=self.record_seconds, width=4).pack(side="left", padx=(6, 3))
        ttk.Label(time_row, text="seconds (stop earlier anytime)", style="CardMuted.TLabel").pack(side="left")
        temp_actions = tk.Frame(left, bg=COLORS["panel"]); temp_actions.pack(fill="x", padx=20, pady=(0, 12))
        ttk.Button(temp_actions, text="▶ Play / Resume", command=self._play_input).pack(side="left")
        ttk.Button(temp_actions, text="Ⅱ Pause playback", command=self._pause_playback).pack(side="left", padx=(7, 0))
        ttk.Button(temp_actions, text="Save recording as…", command=self._save_temp_recording).pack(side="left", padx=(7, 0))
        ttk.Button(temp_actions, text="Delete temp", command=self._delete_temp_recording).pack(side="left", padx=(7, 0))
        self._path_row(left, "Preferred save folder", self.output_dir, self._choose_output)
        ttk.Label(left, text="Duration factor", style="Card.TLabel").pack(anchor="w", padx=20, pady=(12, 3))
        stretch_row = tk.Frame(left, bg=COLORS["panel"]); stretch_row.pack(fill="x", padx=20)
        ttk.Scale(stretch_row, from_=0.5, to=2.0, variable=self.stretch, command=lambda _: self._update_values()).pack(side="left", fill="x", expand=True)
        self.stretch_label = ttk.Label(stretch_row, text="", style="CardMuted.TLabel"); self.stretch_label.pack(side="left", padx=(12, 0))
        ttk.Label(left, text="Voice pitch offset", style="Card.TLabel").pack(anchor="w", padx=20, pady=(18, 3))
        pitch_row = tk.Frame(left, bg=COLORS["panel"]); pitch_row.pack(fill="x", padx=20, pady=(0, 20))
        ttk.Scale(pitch_row, from_=-8, to=8, variable=self.semitones, command=lambda _: self._update_values()).pack(side="left", fill="x", expand=True)
        self.pitch_label = ttk.Label(pitch_row, text="", style="CardMuted.TLabel"); self.pitch_label.pack(side="left", padx=(12, 0))
        self._update_values()
        ttk.Label(right, text="Creative preset", style="Card.TLabel").pack(anchor="w", padx=20, pady=(18, 6))
        preset_row = tk.Frame(right, bg=COLORS["panel"]); preset_row.pack(fill="x", padx=20)
        self.preset_menu = ttk.Combobox(preset_row, textvariable=self.preset_name, values=list(PRESETS), state="readonly")
        self.preset_menu.pack(side="left", fill="x", expand=True)
        self.preset_menu.bind("<<ComboboxSelected>>", self._preset_changed)
        ttk.Button(preset_row, text="▶ Preview / Resume", command=self._preview_preset).pack(side="left", padx=(8, 0))
        ttk.Button(preset_row, text="Ⅱ Pause", command=self._pause_playback).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(right, text="Use this preset for voice transformation", variable=self.use_preset).pack(anchor="w", padx=20, pady=(8, 2))
        self.preset_description = ttk.Label(right, text=PRESETS["Custom"]["description"], style="CardMuted.TLabel", wraplength=340, justify="left")
        self.preset_description.pack(anchor="w", padx=20, pady=(0, 12))
        age_row = tk.Frame(right, bg=COLORS["panel"]); age_row.pack(fill="x", padx=20, pady=(0, 8))
        ttk.Label(age_row, text="Style age", style="CardMuted.TLabel").pack(side="left")
        ttk.Scale(age_row, from_=18, to=80, variable=self.age, command=lambda _: self._update_values()).pack(side="left", fill="x", expand=True, padx=9)
        self.age_label = ttk.Label(age_row, text="", style="CardMuted.TLabel"); self.age_label.pack(side="left")
        ttk.Checkbutton(right, text="Attempt background-music reduction first (experimental)", variable=self.remove_background).pack(anchor="w", padx=20, pady=(0, 10))
        self._update_values()
        ttk.Label(right, text="Outputs to write", style="Card.TLabel").pack(anchor="w", padx=20, pady=(6, 8))
        for label, variable in [
            ("Naive resampling baseline", self.write_naive),
            ("Phase-vocoder stretch", self.write_basic),
            ("Phase-locked stretch", self.write_locked),
            ("Voice transformation", self.write_voice),
        ]:
            ttk.Checkbutton(right, text=label, variable=variable).pack(anchor="w", padx=20, pady=5)
        ttk.Label(right, text="The voice transformation keeps the original duration while applying the selected pitch offset. Input audio is decoded or recorded into a NumPy array before DSP begins.", style="CardMuted.TLabel", wraplength=340, justify="left").pack(anchor="w", padx=20, pady=(18, 8))
        ttk.Checkbutton(right, text="Normalize generated files (prevents clipping)", variable=self.normalize_output).pack(anchor="w", padx=20, pady=(0, 8))
        self.process_button = ttk.Button(right, text="Generate audio files", style="Accent.TButton", command=self._start_processing)
        self.process_button.pack(anchor="w", padx=20)
        self.status_label = ttk.Label(right, text="Choose a file to begin.", style="CardMuted.TLabel", wraplength=340)
        self.status_label.pack(anchor="w", padx=20, pady=16)

    def _compare_page(self) -> None:
        self._page_header("Compare", "A listening guide for the files produced by Transform.")
        card = self._card(self.content); card.pack(fill="both", expand=True, padx=42, pady=(0, 36))
        text = (
            "Naive resampling\nChanges duration and pitch together. At 1.5× it should sound lower.\n\n"
            "Phase vocoder\nChanges duration while preserving pitch, but independent bins can sound watery.\n\n"
            "Phase locked\nAnchors neighboring bins to spectral peaks. Listen to sustained vowels for improved coherence.\n\n"
            "Voice transformation\nApplies a pitch offset, then restores duration. Compare several offsets on the same sentence."
        )
        ttk.Label(card, text=text, style="CardMuted.TLabel", justify="left", wraplength=720).pack(anchor="nw", padx=26, pady=25)
        actions = tk.Frame(card, bg=COLORS["panel"]); actions.pack(anchor="w", padx=26, pady=(0, 22))
        ttk.Button(actions, text="Open output folder", command=self._open_output_folder).pack(side="left")
        ttk.Button(actions, text="Play last output", command=self._play_last_output).pack(side="left", padx=(8, 0))

    def _results_page(self) -> None:
        self._page_header("Temporary results", "Nothing here is final. Save only the versions you want to keep.")
        controls = self._card(self.content)
        controls.pack(fill="x", padx=42, pady=(0, 14))
        row = tk.Frame(controls, bg=COLORS["panel"]); row.pack(fill="x", padx=20, pady=14)
        ttk.Label(row, text="Playback volume", style="Card.TLabel").pack(side="left")
        ttk.Scale(row, from_=0, to=2, variable=self.volume, command=lambda _: self._set_volume()).pack(side="left", fill="x", expand=True, padx=15)
        self.volume_label = ttk.Label(row, text="100%", style="CardMuted.TLabel"); self.volume_label.pack(side="left")
        ttk.Button(row, text="Pause", command=self._pause_playback).pack(side="left", padx=(15, 0))
        ttk.Button(row, text="Stop", command=self._stop_playback).pack(side="left", padx=(7, 0))
        list_area = tk.Frame(self.content, bg=COLORS["background"])
        list_area.pack(fill="both", expand=True, padx=42, pady=(0, 25))
        if not self.result_files:
            ttk.Label(list_area, text="No temporary outputs yet. Generate audio from the Transform page.", style="Muted.TLabel").pack(anchor="w", pady=20)
            return
        for path in list(self.result_files):
            if path.is_file():
                card = self._card(list_area); card.pack(fill="x", pady=5)
                ttk.Label(card, text=path.name, style="Card.TLabel").pack(side="left", padx=16, pady=13)
                ttk.Label(card, text=f"Temporary • {path.parent.name}", style="CardMuted.TLabel").pack(side="left", padx=(0, 12))
                ttk.Button(card, text="▶ Play", command=lambda p=path: self._play_result(p)).pack(side="right", padx=(5, 14), pady=8)
                ttk.Button(card, text="Delete", command=lambda p=path: self._delete_result(p)).pack(side="right", padx=5, pady=8)
                ttk.Button(card, text="Save as…", style="Accent.TButton", command=lambda p=path: self._save_result(p)).pack(side="right", padx=5, pady=8)

    def _vault_page(self) -> None:
        self._page_header("Decoy vault", "Ordinary players hear the decoy; only Voice Lab plus the passphrase can unlock the hidden clip.")
        columns = tk.Frame(self.content, bg=COLORS["background"]); columns.pack(fill="both", expand=True, padx=42)
        encrypt, decrypt = self._card(columns), self._card(columns)
        encrypt.pack(side="left", fill="both", expand=True, padx=(0, 7))
        decrypt.pack(side="left", fill="both", expand=True, padx=(7, 0))
        ttk.Label(encrypt, text="Create vault", style="Card.TLabel").pack(anchor="w", padx=20, pady=(18, 8))
        ttk.Label(encrypt, text="Real clip", style="CardMuted.TLabel").pack(anchor="w", padx=20)
        real_row = tk.Frame(encrypt, bg=COLORS["panel"]); real_row.pack(fill="x", padx=20, pady=(3, 8))
        ttk.Entry(real_row, textvariable=self.vault_real_path).pack(side="left", fill="x", expand=True)
        ttk.Button(real_row, text="Browse", command=lambda: self._choose_vault_file(self.vault_real_path)).pack(side="left", padx=(7, 0))
        ttk.Label(encrypt, text="Decoy clip (what other players hear)", style="CardMuted.TLabel").pack(anchor="w", padx=20)
        decoy_row = tk.Frame(encrypt, bg=COLORS["panel"]); decoy_row.pack(fill="x", padx=20, pady=(3, 8))
        ttk.Entry(decoy_row, textvariable=self.vault_decoy_path).pack(side="left", fill="x", expand=True)
        ttk.Button(decoy_row, text="Browse", command=lambda: self._choose_vault_file(self.vault_decoy_path)).pack(side="left", padx=(7, 0))
        ttk.Label(encrypt, text="Passphrase (8+ characters)", style="CardMuted.TLabel").pack(anchor="w", padx=20)
        ttk.Entry(encrypt, textvariable=self.vault_passphrase, show="●").pack(fill="x", padx=20, pady=(3, 12))
        ttk.Button(encrypt, text="Create temporary vault", style="Accent.TButton", command=self._create_vault).pack(anchor="w", padx=20, pady=(0, 18))
        ttk.Label(decrypt, text="Unlock vault", style="Card.TLabel").pack(anchor="w", padx=20, pady=(18, 8))
        ttk.Label(decrypt, text="Vault WAV", style="CardMuted.TLabel").pack(anchor="w", padx=20)
        container_row = tk.Frame(decrypt, bg=COLORS["panel"]); container_row.pack(fill="x", padx=20, pady=(3, 8))
        ttk.Entry(container_row, textvariable=self.vault_container_path).pack(side="left", fill="x", expand=True)
        ttk.Button(container_row, text="Browse", command=lambda: self._choose_vault_file(self.vault_container_path)).pack(side="left", padx=(7, 0))
        ttk.Label(decrypt, text="Passphrase", style="CardMuted.TLabel").pack(anchor="w", padx=20)
        ttk.Entry(decrypt, textvariable=self.vault_passphrase, show="●").pack(fill="x", padx=20, pady=(3, 12))
        ttk.Button(decrypt, text="Unlock to temporary audio", style="Accent.TButton", command=self._unlock_vault).pack(anchor="w", padx=20, pady=(0, 12))
        ttk.Label(decrypt, text="The passphrase is never stored in the vault. Keep it separately; the decoy is visible by design and is not the secret.", style="CardMuted.TLabel", wraplength=330, justify="left").pack(anchor="w", padx=20, pady=(0, 18))

    def _match_page(self) -> None:
        self._page_header("Reference-guided voice match", "Match broad pitch and brightness traits from a permitted reference—this is not voice cloning.")
        card = self._card(self.content); card.pack(fill="x", padx=42, pady=(0, 18))
        ttk.Label(card, text="Source voice", style="Card.TLabel").pack(anchor="w", padx=22, pady=(18, 4))
        source_row = tk.Frame(card, bg=COLORS["panel"]); source_row.pack(fill="x", padx=22, pady=(0, 10))
        ttk.Entry(source_row, textvariable=self.match_source_path).pack(side="left", fill="x", expand=True)
        ttk.Button(source_row, text="Browse", command=lambda: self._choose_vault_file(self.match_source_path)).pack(side="left", padx=(7, 0))
        ttk.Label(card, text="Reference voice (use only audio you have permission to use)", style="Card.TLabel").pack(anchor="w", padx=22, pady=(6, 4))
        reference_row = tk.Frame(card, bg=COLORS["panel"]); reference_row.pack(fill="x", padx=22, pady=(0, 10))
        ttk.Entry(reference_row, textvariable=self.match_reference_path).pack(side="left", fill="x", expand=True)
        ttk.Button(reference_row, text="Browse", command=lambda: self._choose_vault_file(self.match_reference_path)).pack(side="left", padx=(7, 0))
        ttk.Label(card, text="The matcher estimates median pitch and broad spectral brightness, then applies a phase-locked pitch shift and gentle EQ tilt. It cannot copy vocal identity.", style="CardMuted.TLabel", wraplength=720, justify="left").pack(anchor="w", padx=22, pady=(4, 12))
        action_row = tk.Frame(card, bg=COLORS["panel"]); action_row.pack(anchor="w", padx=22, pady=(0, 20))
        ttk.Button(action_row, text="Preview match", command=lambda: self._start_match(preview=True)).pack(side="left")
        ttk.Button(action_row, text="Create temporary match", style="Accent.TButton", command=lambda: self._start_match(preview=False)).pack(side="left", padx=(8, 0))

    def _start_match(self, preview: bool) -> None:
        reference = Path(self.match_reference_path.get())
        source_path = Path(self.match_source_path.get())
        if not reference.is_file() or (self.current_audio is None and not source_path.is_file()):
            messagebox.showerror("Choose voices", "Choose a reference and source, or record/select a source voice in Transform first.")
            return
        try:
            config = StftConfig(frame_size=self.frame_size.get(), hop_size=self.hop_size.get())
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        source: object = self.current_audio if self.current_audio is not None else source_path
        if preview:
            threading.Thread(target=self._match_worker, args=(source, reference, config, None), daemon=True).start()
        else:
            session = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root)); self.temp_output_dirs.append(session)
            threading.Thread(target=self._match_worker, args=(source, reference, config, session / "reference_guided_match.wav"), daemon=True).start()

    def _match_worker(self, source: object, reference_path: Path, config: StftConfig, destination: Path | None) -> None:
        try:
            if isinstance(source, Path):
                source_samples, source_rate = read_audio(source)
            else:
                source_samples, source_rate = source
            reference_samples, reference_rate = read_audio(reference_path)
            config = StftConfig(sample_rate=source_rate, frame_size=config.frame_size, hop_size=config.hop_size)
            matched, profile = match_voice_character(source_samples, source_rate, reference_samples, reference_rate, config)
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
        selected = filedialog.askopenfilename(title="Choose audio", filetypes=[("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a"), ("All files", "*.*")])
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
        session = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root)); self.temp_output_dirs.append(session)
        destination = session / "voice_lab_vault.wav"
        real_input: object = self.current_audio if self.current_audio is not None else real_path
        threading.Thread(target=self._vault_create_worker, args=(real_input, decoy, self.vault_passphrase.get(), destination), daemon=True).start()

    def _vault_create_worker(self, real_input: object, decoy: Path, passphrase: str, destination: Path) -> None:
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
        session = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root)); self.temp_output_dirs.append(session)
        destination = session / "unlocked_audio.wav"
        threading.Thread(target=self._vault_unlock_worker, args=(vault, self.vault_passphrase.get(), destination), daemon=True).start()

    def _vault_unlock_worker(self, vault: Path, passphrase: str, destination: Path) -> None:
        try:
            samples, sample_rate = unlock_vault(vault, passphrase)
            write_output_wav(destination, samples, sample_rate)
            self.result_queue.put((True, "Unlocked audio to a temporary file. Review or save it from Results.", [destination]))
        except Exception as error:
            self.result_queue.put((False, str(error), []))

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
            title="Save generated audio", initialdir=self.output_dir.get(), initialfile=path.name,
            defaultextension=".wav", filetypes=[("WAV audio", "*.wav")],
        )
        if destination:
            shutil.copy2(path, destination)
            messagebox.showinfo("Saved", f"Saved {path.name} to your selected location.")

    def _delete_result(self, path: Path) -> None:
        if not path.is_file():
            return
        if not messagebox.askyesno("Delete temporary file", f"Delete {path.name}? This only deletes the temporary copy."):
            return
        self._stop_playback()
        path.unlink()
        self.result_files = [item for item in self.result_files if item != path]
        self.show_page("Results")

    def _recover_temp_outputs(self) -> None:
        old_dirs = [path for path in self.temp_root.iterdir() if path.is_dir() and path.name.startswith("session_")]
        old_files = [file for directory in old_dirs for file in directory.glob("*.wav")]
        if not old_files:
            return
        keep = messagebox.askyesno(
            "Temporary audio found",
            f"Found {len(old_files)} generated temporary file(s) from an earlier session. Keep them for review?",
        )
        if keep:
            self.temp_output_dirs.extend(old_dirs)
            self.result_files.extend(old_files)
        else:
            for directory in old_dirs:
                shutil.rmtree(directory)

    def _learn_page(self) -> None:
        self._page_header("Learn", "The signal-processing ideas behind each output.")
        card = self._card(self.content); card.pack(fill="both", expand=True, padx=42, pady=(0, 36))
        text = (
            "STFT  •  A recording is split into overlapping, windowed frames. Each frame gets its own FFT.\n\n"
            "Magnitude and phase  •  Magnitude tells us how much of a frequency is present; phase tracks its timing.\n\n"
            "Phase vocoder  •  When output frames use a different hop size, phase must be advanced by the measured instantaneous frequency.\n\n"
            "Identity phase locking  •  Nearby bins retain their phase relationship to a spectral peak, improving harmonic coherence."
        )
        ttk.Label(card, text=text, style="CardMuted.TLabel", justify="left", wraplength=750).pack(anchor="nw", padx=26, pady=25)

    def _settings_page(self) -> None:
        self._page_header("Settings", "Advanced processing and studio options. Keep the defaults while learning.")
        card = self._card(self.content); card.pack(fill="x", padx=42)
        for label, variable in [("Frame size", self.frame_size), ("Analysis hop", self.hop_size), ("Recording sample rate", self.record_sample_rate)]:
            row = tk.Frame(card, bg=COLORS["panel"]); row.pack(fill="x", padx=22, pady=12)
            ttk.Label(row, text=label, style="Card.TLabel").pack(side="left")
            ttk.Entry(row, textvariable=variable, width=12).pack(side="right")
        ttk.Checkbutton(card, text="Normalize generated files by default", variable=self.normalize_output).pack(anchor="w", padx=22, pady=(6, 8))
        ttk.Label(card, text="A 2048-sample frame and 512-sample hop give 75% overlap. Larger frames improve frequency detail; shorter frames follow quick changes more closely. 44,100 Hz is a good default recording rate for voice.", style="CardMuted.TLabel", wraplength=650, justify="left").pack(anchor="w", padx=22, pady=(8, 22))

    def _update_values(self) -> None:
        if hasattr(self, "stretch_label"):
            self.stretch_label.configure(text=f"{self.stretch.get():.2f}×")
            self.pitch_label.configure(text=f"{self.semitones.get():+.1f} st")
            if hasattr(self, "age_label"):
                self.age_label.configure(text=f"{self.age.get():.0f} years")

    def _choose_input(self) -> None:
        selected = filedialog.askopenfilename(title="Choose audio", filetypes=[("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a"), ("All files", "*.*")])
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
        import os
        os.startfile(path)  # type: ignore[attr-defined]  # Windows desktop app

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
            self.record_button.configure(state="disabled")
            self.record_pause_button.configure(text="Ⅱ Pause")
            self.status_label.configure(text="Recording… press Pause, then Resume, or Stop to create a temporary WAV.")
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
            self.record_pause_button.configure(text="Ⅱ Pause")
            self.status_label.configure(text="Recording resumed.")
        else:
            self.recorder.pause()
            self.recording_paused = True
            self.record_pause_button.configure(text="▶ Resume")
            self.status_label.configure(text="Recording paused. No audio is being added.")

    def _stop_recording(self) -> None:
        if self.recorder is None:
            return
        try:
            samples, sample_rate = self.recorder.stop()
            temp_dir = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root))
            self.temp_output_dirs.append(temp_dir)
            self.temp_recording = temp_dir / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            write_output_wav(self.temp_recording, samples, sample_rate)
            self.current_audio = (samples, sample_rate)
            self.input_path.set(f"Temporary recording — {len(samples) / sample_rate:.1f}s (not saved)")
            self.status_label.configure(text="Temporary recording ready. Play it, save it explicitly, delete it, or transform it.")
        except Exception as error:
            messagebox.showerror("Recording failed", str(error))
        finally:
            self.recorder = None
            self.recording_paused = False
            if hasattr(self, "record_button") and self.record_button.winfo_exists():
                self.record_button.configure(state="normal")

    def _save_temp_recording(self) -> None:
        if self.temp_recording is None or not self.temp_recording.is_file():
            messagebox.showinfo("No temporary recording", "Record and stop audio before saving it.")
            return
        destination = filedialog.asksaveasfilename(title="Save recording", defaultextension=".wav", filetypes=[("WAV audio", "*.wav")])
        if destination:
            shutil.copy2(self.temp_recording, destination)
            self.status_label.configure(text=f"Saved recording to {Path(destination).name}.")

    def _delete_temp_recording(self) -> None:
        if self.temp_recording is None:
            return
        self._stop_playback()
        try:
            if self.temp_recording.exists():
                self.temp_recording.unlink()
            self.temp_recording = None
            self.current_audio = None
            self.input_path.set("")
            self.status_label.configure(text="Temporary recording deleted.")
        except OSError as error:
            messagebox.showerror("Could not delete recording", str(error))

    def _play_input(self) -> None:
        if self.player is not None and self.player.paused:
            self.player.resume()
            self.status_label.configure(text="Playback resumed.")
            return
        if self.current_audio is not None:
            samples, sample_rate = self.current_audio
            self.preview_playing = False
            self._start_playback(samples, sample_rate)
            return
        path = Path(self.input_path.get())
        if not path.is_file():
            messagebox.showerror("Choose audio", "Select or record audio first.")
            return
        try:
            self.preview_playing = False
            self._start_playback(*read_audio(path))
        except Exception as error:
            messagebox.showerror("Playback failed", str(error))

    def _play_last_output(self) -> None:
        if self.last_output is None or not self.last_output.is_file():
            messagebox.showinfo("No output yet", "Generate audio files first.")
            return
        try:
            self.preview_playing = False
            self._start_playback(*read_audio(self.last_output))
        except Exception as error:
            messagebox.showerror("Playback failed", str(error))

    def _start_playback(self, samples: object, sample_rate: int) -> None:
        try:
            self._stop_playback()
            self.player = AudioPlayer(samples, sample_rate)  # type: ignore[arg-type]
            self.player.play()
            if hasattr(self, "status_label") and self.status_label.winfo_exists():
                self.status_label.configure(text="Playing. Use Pause playback, then Play / Resume to continue.")
        except Exception as error:
            messagebox.showerror("Playback failed", str(error))

    def _pause_playback(self) -> None:
        if self.player is not None:
            self.player.pause()
            if hasattr(self, "status_label") and self.status_label.winfo_exists():
                self.status_label.configure(text="Playback paused.")

    def _stop_playback(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.player = None

    def _preset_changed(self, event=None) -> None:
        preset = PRESETS[self.preset_name.get()]
        if preset.get("premium") and not self.premium_preview_enabled:
            enable = messagebox.askyesno(
                "Premium preset",
                "This sci-fi preset is behind the future upgrade check. No payment is implemented yet. Enable a local preview session?",
            )
            if not enable:
                self.preset_name.set("Custom")
                preset = PRESETS["Custom"]
            else:
                self.premium_preview_enabled = True
        self.preset_description.configure(text=preset["description"])
        if preset["semitones"] is not None:
            self.semitones.set(float(preset["semitones"]))
            self.use_preset.set(True)
            self._update_values()

    def _active_semitones(self) -> float:
        preset = PRESETS[self.preset_name.get()]
        if self.use_preset.get() and preset["semitones"] is not None:
            offset = float(preset["semitones"])
            if self.preset_name.get() in {"Feminine-style", "Masculine-style"}:
                # Creative age-character control: older positions gently lower
                # the pitch, younger positions gently raise it.
                offset -= (self.age.get() - 30) * 0.03
            return offset
        return self.semitones.get()

    def _preview_preset(self) -> None:
        if self.preview_playing and self.player is not None and self.player.paused:
            self.player.resume()
            self.status_label.configure(text="Preset preview resumed.")
            return
        source = Path(self.input_path.get())
        if self.current_audio is None and not source.is_file():
            messagebox.showerror("Choose audio", "Select or record audio before previewing a preset.")
            return
        semitones = self._active_semitones()
        if semitones == 0:
            messagebox.showinfo("Choose a transformation", "Choose a preset or a non-zero pitch offset to preview.")
            return
        try:
            config = StftConfig(frame_size=self.frame_size.get(), hop_size=self.hop_size.get())
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        input_data: object = self.current_audio if self.current_audio is not None else source
        self.status_label.configure(text="Rendering preset preview…")
        threading.Thread(target=self._preview_worker, args=(input_data, semitones, config, self.remove_background.get()), daemon=True).start()

    def _preview_worker(self, input_data: object, semitones: float, config: StftConfig, remove_background: bool) -> None:
        try:
            if isinstance(input_data, Path):
                signal, sample_rate = read_audio(input_data)
            else:
                signal, sample_rate = input_data
            config = StftConfig(sample_rate=sample_rate, frame_size=config.frame_size, hop_size=config.hop_size)
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
            messagebox.showerror("Choose an input", "Select a valid audio file or record from the microphone first.")
            return
        if not any(v.get() for v in (self.write_naive, self.write_basic, self.write_locked, self.write_voice)):
            messagebox.showerror("Choose an output", "Select at least one output type.")
            return
        active_semitones = self._active_semitones()
        if self.write_voice.get() and active_semitones == 0:
            messagebox.showerror("Choose a pitch offset", "Voice transformation needs a non-zero pitch offset.")
            return
        try:
            config = StftConfig(frame_size=self.frame_size.get(), hop_size=self.hop_size.get())
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        self.busy = True
        self.process_button.configure(state="disabled")
        self.status_label.configure(text="Processing locally… this can take a moment.")
        input_data: object = self.current_audio if self.current_audio is not None else source
        session_dir = Path(tempfile.mkdtemp(prefix="session_", dir=self.temp_root))
        self.temp_output_dirs.append(session_dir)
        options = (input_data, session_dir, self.stretch.get(), active_semitones, config, self.write_naive.get(), self.write_basic.get(), self.write_locked.get(), self.write_voice.get(), self.normalize_output.get(), self.remove_background.get())
        threading.Thread(target=self._process_worker, args=options, daemon=True).start()

    def _process_worker(self, input_data: object, destination: Path, stretch: float, semitones: float, config: StftConfig, naive: bool, basic: bool, locked: bool, voice: bool, normalize: bool, remove_background: bool) -> None:
        try:
            if isinstance(input_data, Path):
                signal, sample_rate = read_audio(input_data)
                stem = input_data.stem
            else:
                signal, sample_rate = input_data  # Recorded arrays skip file decoding entirely.
                stem = "recording"
            config = StftConfig(sample_rate=sample_rate, frame_size=config.frame_size, hop_size=config.hop_size)
            if remove_background:
                signal = reduce_background_estimate(signal, config)
            destination.mkdir(parents=True, exist_ok=True)
            written: list[str] = []
            jobs = []
            if naive: jobs.append((f"{stem}_naive_{stretch:g}x.wav", naive_time_stretch(signal, stretch)))
            if basic: jobs.append((f"{stem}_phase_vocoder_{stretch:g}x.wav", time_stretch(signal, stretch, config)))
            if locked: jobs.append((f"{stem}_phase_locked_{stretch:g}x.wav", time_stretch(signal, stretch, config, phase_locking=True)))
            if voice: jobs.append((f"{stem}_voice_transform_{semitones:+g}st.wav", anonymize_voice(signal, semitones, config)))
            for name, transformed in jobs:
                if normalize:
                    peak = float(abs(transformed).max())
                    if peak > 0:
                        transformed = transformed * (0.98 / peak)
                write_output_wav(destination / name, transformed, sample_rate)
                written.append(name)
            paths = [destination / name for name in written]
            self.result_queue.put((True, f"Created {len(written)} temporary file(s). Review and save the ones you want.", paths))
        except Exception as error:  # Surface input-format errors in the UI thread.
            self.result_queue.put((False, str(error), []))

    def _poll_results(self) -> None:
        try:
            while True:
                ok, payload = self.preview_queue.get_nowait()
                if ok:
                    samples, sample_rate = payload  # type: ignore[misc]
                    self.preview_playing = True
                    self._start_playback(samples, sample_rate)
                    if hasattr(self, "status_label") and self.status_label.winfo_exists():
                        self.status_label.configure(text="Preset preview playing. Adjust volume or pause it from the controls.")
                else:
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
                if hasattr(self, "process_button") and self.process_button.winfo_exists():
                    self.process_button.configure(state="normal")
                    if self.status_label.winfo_exists():
                        self.status_label.configure(text=message, style="CardMuted.TLabel" if ok else "Card.TLabel")
                if not ok:
                    messagebox.showerror("Processing failed", message)
                elif outputs:
                    self.show_page("Results")
        except queue.Empty:
            pass
        self.after(100, self._poll_results)


if __name__ == "__main__":
    VoiceLab().mainloop()
