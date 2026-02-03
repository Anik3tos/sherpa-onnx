#!/usr/bin/env python3
"""
Theme configuration for the TTS GUI using PySide6 (Qt).
Implements the Dracula color scheme.
"""

from tts_gui.common import QPalette, QColor, QFont


class TTSGuiThemeMixin:
    """Mixin class providing theme and styling functionality."""

    def setup_theme(self):
        """Configure the dark theme (Dracula) for Qt widgets using stylesheets."""

        # Build comprehensive stylesheet for all widgets
        stylesheet = f"""
            /* Main Window and Frames */
            QMainWindow, QWidget {{
                background-color: {self.colors['bg_secondary']};
                color: {self.colors['fg_primary']};
                font-family: "Segoe UI";
                font-size: 14px;
            }}
            
            QFrame {{
                background-color: {self.colors['bg_secondary']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
            }}
            
            QFrame#cardFrame {{
                background-color: {self.colors['bg_tertiary']};
                border: 1px solid {self.colors['border_light']};
            }}
            
            /* Group Boxes (replacing LabelFrames) */
            QGroupBox {{
                background-color: {self.colors['bg_secondary']};
                border: 2px solid {self.colors['border']};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
                font-size: 15px;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                background-color: {self.colors['bg_accent']};
                color: {self.colors['fg_primary']};
                border-radius: 4px;
                left: 10px;
            }}
            
            /* Labels */
            QLabel {{
                background-color: transparent;
                color: {self.colors['fg_primary']};
                font-size: 15px;
                border: none;
            }}
            
            QLabel#timeLabel {{
                background-color: transparent;
                color: {self.colors['accent_cyan']};
                font-family: "Consolas";
                font-size: 14px;
                font-weight: bold;
            }}
            
            QLabel#mutedLabel {{
                color: {self.colors['fg_muted']};
            }}
            
            /* Primary Button (Pink) */
            QPushButton#primaryButton {{
                background-color: {self.colors['accent_pink']};
                color: {self.colors['bg_primary']};
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                font-size: 14px;
            }}
            
            QPushButton#primaryButton:hover {{
                background-color: {self.colors['accent_pink_hover']};
            }}
            
            QPushButton#primaryButton:pressed {{
                background-color: #ff66c4;
            }}
            
            QPushButton#primaryButton:disabled {{
                background-color: {self.colors['bg_accent']};
                color: {self.colors['fg_muted']};
            }}
            
            /* Success Button (Green) */
            QPushButton#successButton {{
                background-color: {self.colors['accent_green']};
                color: {self.colors['bg_primary']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 14px;
            }}
            
            QPushButton#successButton:hover {{
                background-color: #45e070;
            }}
            
            QPushButton#successButton:pressed {{
                background-color: #3dd164;
            }}
            
            QPushButton#successButton:disabled {{
                background-color: {self.colors['bg_accent']};
                color: {self.colors['fg_muted']};
            }}
            
            /* Warning Button (Orange) */
            QPushButton#warningButton {{
                background-color: {self.colors['accent_orange']};
                color: {self.colors['bg_primary']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 14px;
            }}
            
            QPushButton#warningButton:hover {{
                background-color: #ffad5c;
            }}
            
            QPushButton#warningButton:pressed {{
                background-color: #ffa04d;
            }}
            
            QPushButton#warningButton:disabled {{
                background-color: {self.colors['bg_accent']};
                color: {self.colors['fg_muted']};
            }}
            
            /* Danger Button (Red) */
            QPushButton#dangerButton {{
                background-color: {self.colors['accent_red']};
                color: {self.colors['bg_primary']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 14px;
            }}
            
            QPushButton#dangerButton:hover {{
                background-color: #ff4444;
            }}
            
            QPushButton#dangerButton:pressed {{
                background-color: #ff3333;
            }}
            
            QPushButton#dangerButton:disabled {{
                background-color: {self.colors['bg_accent']};
                color: {self.colors['fg_muted']};
            }}
            
            /* Dark Button (Default) */
            QPushButton {{
                background-color: {self.colors['bg_accent']};
                color: {self.colors['fg_primary']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 14px;
            }}
            
            QPushButton:hover {{
                background-color: {self.colors['bg_tertiary']};
            }}
            
            QPushButton:pressed {{
                background-color: {self.colors['border']};
            }}
            
            QPushButton:disabled {{
                background-color: {self.colors['bg_secondary']};
                color: {self.colors['fg_muted']};
            }}
            
            /* Utility Button (Cyan) */
            QPushButton#utilityButton {{
                background-color: {self.colors['accent_cyan']};
                color: {self.colors['bg_primary']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 14px;
            }}
            
            QPushButton#utilityButton:hover {{
                background-color: #7de8f5;
            }}
            
            QPushButton#utilityButton:pressed {{
                background-color: #6dd9e8;
            }}
            
            QPushButton#utilityButton:disabled {{
                background-color: {self.colors['bg_accent']};
                color: {self.colors['fg_muted']};
            }}
            
            /* ComboBox */
            QComboBox {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 13px;
            }}
            
            QComboBox:hover {{
                border-color: {self.colors['accent_purple']};
            }}
            
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid {self.colors['border']};
            }}
            
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
            }}
            
            QComboBox QAbstractItemView {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                selection-background-color: {self.colors['selection']};
                selection-color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['border']};
            }}
            
            /* Checkboxes and Radio Buttons */
            QCheckBox, QRadioButton {{
                background-color: transparent;
                color: {self.colors['fg_primary']};
                font-size: 14px;
                spacing: 8px;
            }}
            
            QCheckBox:hover, QRadioButton:hover {{
                color: {self.colors['accent_cyan']};
            }}
            
            QCheckBox::indicator, QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {self.colors['border']};
                background-color: {self.colors['bg_primary']};
            }}
            
            QCheckBox::indicator {{
                border-radius: 4px;
            }}
            
            QRadioButton::indicator {{
                border-radius: 9px;
            }}
            
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
                background-color: {self.colors['accent_pink']};
                border-color: {self.colors['accent_pink']};
            }}
            
            /* Sliders */
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {self.colors['bg_accent']};
                border-radius: 3px;
            }}
            
            QSlider::handle:horizontal {{
                background: {self.colors['accent_purple']};
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            
            QSlider::handle:horizontal:hover {{
                background: {self.colors['accent_pink']};
            }}
            
            QSlider::sub-page:horizontal {{
                background: {self.colors['accent_purple']};
                border-radius: 3px;
            }}
            
            /* Progress Bar */
            QProgressBar {{
                border: none;
                background-color: {self.colors['bg_accent']};
                border-radius: 4px;
                text-align: center;
                color: {self.colors['fg_primary']};
                height: 20px;
            }}
            
            QProgressBar::chunk {{
                background-color: {self.colors['accent_pink']};
                border-radius: 4px;
            }}
            
            /* Text Edit / ScrolledText */
            QTextEdit, QPlainTextEdit {{
                background-color: {self.colors['bg_primary']};
                color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                font-family: "Segoe UI";
                font-size: 14px;
                selection-background-color: {self.colors['selection']};
                selection-color: {self.colors['fg_primary']};
            }}
            
            QTextEdit#statusText {{
                font-family: "Consolas";
                font-size: 12px;
                color: {self.colors['fg_secondary']};
            }}
            
            /* Scroll Bars */
            QScrollBar:vertical {{
                background-color: {self.colors['bg_secondary']};
                width: 12px;
                border: none;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {self.colors['bg_accent']};
                border-radius: 6px;
                min-height: 30px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {self.colors['border']};
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            
            QScrollBar:horizontal {{
                background-color: {self.colors['bg_secondary']};
                height: 12px;
                border: none;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:horizontal {{
                background-color: {self.colors['bg_accent']};
                border-radius: 6px;
                min-width: 30px;
            }}
            
            QScrollBar::handle:horizontal:hover {{
                background-color: {self.colors['border']};
            }}
            
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            
            /* Spin Box */
            QSpinBox, QDoubleSpinBox {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                background-color: {self.colors['bg_accent']};
                border: none;
                width: 16px;
            }}
            
            /* Line Edit */
            QLineEdit {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 5px;
                selection-background-color: {self.colors['selection']};
            }}
            
            QLineEdit:focus {{
                border-color: {self.colors['accent_purple']};
            }}
            
            /* List Widget */
            QListWidget {{
                background-color: {self.colors['bg_primary']};
                color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
            }}
            
            QListWidget::item {{
                padding: 5px;
            }}
            
            QListWidget::item:selected {{
                background-color: {self.colors['selection']};
                color: {self.colors['fg_primary']};
            }}
            
            QListWidget::item:hover {{
                background-color: {self.colors['bg_accent']};
            }}
            
            /* Dialog */
            QDialog {{
                background-color: {self.colors['bg_primary']};
            }}
            
            /* Message Box */
            QMessageBox {{
                background-color: {self.colors['bg_primary']};
            }}
            
            QMessageBox QLabel {{
                color: {self.colors['fg_primary']};
            }}
        """

        # Apply stylesheet to the main window
        self.main_window.setStyleSheet(stylesheet)
