#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", ".ipynb_checkpoints"}
BINARY_SUFFIXES = {".joblib", ".png", ".jpg", ".jpeg", ".gif", ".pdf"}
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PRIVATE_PATHS = re.compile(
    re.escape("/" + "Users" + "/")
    + "|"
    + re.escape("C:" + "\\\\" + "Users" + "\\\\")
    + "|"
    + re.escape("/" + "home" + "/")
    + r"[^/]+/"
)


def tracked_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
    ]


def main() -> None:
    errors = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size >= 95 * 1024 * 1024:
            errors.append(f"File exceeds GitHub's safe size limit: {relative}")
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if HAN.search(text):
            errors.append(f"Han characters found: {relative}")
        if PRIVATE_PATHS.search(text):
            errors.append(f"Private absolute path found: {relative}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Repository validation passed for {len(tracked_files())} files.")


if __name__ == "__main__":
    main()
