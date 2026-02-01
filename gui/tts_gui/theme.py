#!/usr/bin/env python3

from tts_gui.common import ttk


class TTSGuiThemeMixin:
    def setup_theme(self):
        """Configure the dark theme for ttk widgets"""
        style = ttk.Style()

        # Configure the theme
        style.theme_use("clam")

        # Configure Frame styles
        style.configure(
            "Dark.TFrame",
            background=self.colors["bg_secondary"],
            borderwidth=1,
            relief="solid",
            bordercolor=self.colors["border"],
        )

        style.configure(
            "Card.TFrame",
            background=self.colors["bg_tertiary"],
            borderwidth=1,
            relief="solid",
            bordercolor=self.colors["border_light"],
        )

        # Configure LabelFrame styles
        style.configure(
            "Dark.TLabelframe",
            background=self.colors["bg_secondary"],
            borderwidth=2,
            relief="solid",
            bordercolor=self.colors["border"],
        )

        style.configure(
            "Dark.TLabelframe.Label",
            background=self.colors["bg_accent"],
            foreground=self.colors["fg_primary"],
            font=("Segoe UI", 12, "bold"),
            borderwidth=0,
            relief="flat",
        )

        # Configure Label styles
        style.configure(
            "Dark.TLabel",
            background=self.colors["bg_tertiary"],
            foreground=self.colors["fg_primary"],
            font=("Segoe UI", 12),
        )

        style.configure(
            "Time.TLabel",
            background=self.colors["bg_tertiary"],
            foreground=self.colors["accent_cyan"],
            font=("Consolas", 11, "bold"),
        )

        # Configure Button styles with Dracula colors
        style.configure(
            "Primary.TButton",
            background=self.colors["accent_pink"],
            foreground=self.colors["bg_primary"],
            borderwidth=0,
            focuscolor="none",
            font=("Segoe UI", 11, "bold"),
            padding=(15, 8),
        )

        style.map(
            "Primary.TButton",
            background=[
                ("active", self.colors["accent_pink_hover"]),
                ("pressed", "#ff66c4"),
                ("disabled", self.colors["bg_accent"]),
            ],
        )

        style.configure(
            "Success.TButton",
            background=self.colors["accent_green"],
            foreground=self.colors["bg_primary"],
            borderwidth=0,
            focuscolor="none",
            font=("Segoe UI", 11),
            padding=(12, 6),
        )

        style.map(
            "Success.TButton",
            background=[
                ("active", "#45e070"),
                ("pressed", "#3dd164"),
                ("disabled", self.colors["bg_accent"]),
            ],
        )

        style.configure(
            "Warning.TButton",
            background=self.colors["accent_orange"],
            foreground=self.colors["bg_primary"],
            borderwidth=0,
            focuscolor="none",
            font=("Segoe UI", 11),
            padding=(12, 6),
        )

        style.map(
            "Warning.TButton",
            background=[
                ("active", "#ffad5c"),
                ("pressed", "#ffa04d"),
                ("disabled", self.colors["bg_accent"]),
            ],
        )

        style.configure(
            "Danger.TButton",
            background=self.colors["accent_red"],
            foreground=self.colors["bg_primary"],
            borderwidth=0,
            focuscolor="none",
            font=("Segoe UI", 11),
            padding=(12, 6),
        )

        style.map(
            "Danger.TButton",
            background=[
                ("active", "#ff4444"),
                ("pressed", "#ff3333"),
                ("disabled", self.colors["bg_accent"]),
            ],
        )

        style.configure(
            "Dark.TButton",
            background=self.colors["bg_accent"],
            foreground=self.colors["fg_primary"],
            borderwidth=0,
            focuscolor="none",
            font=("Segoe UI", 11),
            padding=(12, 6),
        )

        style.map(
            "Dark.TButton",
            background=[
                ("active", self.colors["bg_tertiary"]),
                ("pressed", self.colors["border"]),
                ("disabled", self.colors["bg_secondary"]),
            ],
        )

        # Configure Utility button style (for Import/Export buttons)
        style.configure(
            "Utility.TButton",
            background=self.colors["accent_cyan"],
            foreground=self.colors["bg_primary"],
            borderwidth=0,
            focuscolor="none",
            font=("Segoe UI", 11, "bold"),
            padding=(12, 6),
        )

        style.map(
            "Utility.TButton",
            background=[
                ("active", "#7de8f5"),
                ("pressed", "#6dd9e8"),
                ("disabled", self.colors["bg_accent"]),
            ],
        )

        # Configure Radiobutton styles
        style.configure(
            "Dark.TRadiobutton",
            background=self.colors["bg_secondary"],
            foreground=self.colors["fg_primary"],
            focuscolor="none",
            font=("Segoe UI", 11),
        )

        style.map(
            "Dark.TRadiobutton", background=[("active", self.colors["bg_accent"])]
        )

        # Configure Scale styles with Dracula colors
        style.configure(
            "Dark.Horizontal.TScale",
            background=self.colors["bg_secondary"],
            troughcolor=self.colors["bg_accent"],
            borderwidth=0,
            lightcolor=self.colors["accent_purple"],
            darkcolor=self.colors["accent_purple"],
        )

        # Configure Spinbox styles
        style.configure(
            "Dark.TSpinbox",
            fieldbackground=self.colors["bg_tertiary"],
            background=self.colors["bg_secondary"],
            foreground=self.colors["fg_primary"],
            bordercolor=self.colors["border"],
            arrowcolor=self.colors["fg_secondary"],
            font=("Segoe UI", 10),
        )

        # Configure Progressbar styles with Dracula colors
        style.configure(
            "Dark.Horizontal.TProgressbar",
            background=self.colors["accent_pink"],
            troughcolor=self.colors["bg_accent"],
            borderwidth=0,
            lightcolor=self.colors["accent_pink"],
            darkcolor=self.colors["accent_pink"],
        )
