"""Reading and writing config.ini, the file that holds the system's defaults.

Every other module asks this one for its settings, so a value is changed in
one place only. Relative paths in config.ini are relative to the project
folder (the folder that holds config.ini), not to the current directory, so
the program works no matter where it is started from.
"""

from __future__ import annotations

import configparser
import os

# The project folder: the parent of the tbrgs package.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config.ini")


class Config:
    """A thin wrapper around configparser with typed getters and path handling."""

    def __init__(self, path: str = DEFAULT_CONFIG):
        self.path = path
        self.parser = configparser.ConfigParser()
        if not self.parser.read(path, encoding="utf-8"):
            raise FileNotFoundError(f"Cannot read the configuration file '{path}'")

    # ----- typed getters -------------------------------------------------
    def get(self, section: str, key: str, fallback: str | None = None) -> str:
        value = self.parser.get(section, key, fallback=fallback)
        if value is None:
            raise KeyError(f"[{section}] {key} is missing from {self.path}")
        return value

    def getint(self, section: str, key: str, fallback: int | None = None) -> int:
        return int(self.get(section, key, None if fallback is None else str(fallback)))

    def getfloat(self, section: str, key: str, fallback: float | None = None) -> float:
        return float(self.get(section, key, None if fallback is None else str(fallback)))

    def path_of(self, section: str, key: str) -> str:
        """Return a path setting as an absolute path inside the project folder."""
        value = self.get(section, key)
        return value if os.path.isabs(value) else os.path.join(PROJECT_DIR, value)

    def section(self, name: str) -> dict:
        """Return a whole section as a plain dict of strings."""
        return dict(self.parser.items(name)) if self.parser.has_section(name) else {}

    # ----- writing ---------------------------------------------------------
    def set(self, section: str, key: str, value) -> None:
        if not self.parser.has_section(section):
            self.parser.add_section(section)
        self.parser.set(section, key, str(value))

    def save_values(self, section: str, values: dict) -> None:
        """Change some values in config.ini and keep everything else (comments included).

        configparser drops comments when it writes a file, so this edits the
        matching "key = value" lines in place instead.
        """
        for key, value in values.items():
            self.set(section, key, value)
        with open(self.path, encoding="utf-8") as file:
            lines = file.read().splitlines()

        current = None
        remaining = {str(k).lower(): str(v) for k, v in values.items()}
        output = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if current == section.lower() and remaining:
                    output.extend(f"{k} = {v}" for k, v in remaining.items())
                    remaining = {}
                current = stripped[1:-1].strip().lower()
            elif current == section.lower() and "=" in stripped and not stripped.startswith((";", "#")):
                key = stripped.split("=", 1)[0].strip().lower()
                if key in remaining:
                    line = f"{key} = {remaining.pop(key)}"
            output.append(line)
        if remaining:  # the section was missing, or was the last one in the file
            if current != section.lower():
                output.extend(["", f"[{section}]"])
            output.extend(f"{k} = {v}" for k, v in remaining.items())
        with open(self.path, "w", encoding="utf-8") as file:
            file.write("\n".join(output) + "\n")


def load_config(path: str | None = None) -> Config:
    """Load config.ini (or another file given by path)."""
    return Config(path or DEFAULT_CONFIG)
