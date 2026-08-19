#!/usr/bin/env python3
"""Лёгкий линт Markdown для CI: битые локальные ссылки уже в validate_repository.

Проверяет:
- заголовки без пробела после #;
- наличие trailing whitespace (предупреждение);
- fenced code blocks без закрытия;
- пустые ссылки []( ) / [text]().
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}


def md_files():
    for path in REPO_ROOT.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"{path.relative_to(REPO_ROOT)}: не удалось прочитать: {exc}"]

    rel = path.relative_to(REPO_ROOT)
    lines = text.splitlines()

    # Незакрытые fenced blocks
    fence_count = len(re.findall(r"^```", text, re.MULTILINE))
    if fence_count % 2 != 0:
        errors.append(f"{rel}: незакрытый fenced code block (```)")

    heading_bad = re.compile(r"^#+[^#\s]")
    empty_link = re.compile(r"\[[^\]]*\]\(\s*\)")
    for i, line in enumerate(lines, 1):
        if heading_bad.match(line):
            errors.append(f"{rel}:{i}: заголовок без пробела после #: {line[:60]!r}")
        if empty_link.search(line):
            errors.append(f"{rel}:{i}: пустая Markdown-ссылка")
        if line.rstrip("\n") != line.rstrip():
            # trailing spaces — только предупреждение, не fail CI
            warnings.append(f"{rel}:{i}: trailing whitespace")

    # Слишком много trailing whitespace не роняем CI
    _ = warnings
    return errors


def main() -> int:
    print("--- Markdown lint (tools/lint_markdown.py) ---")
    all_errors: list[str] = []
    count = 0
    for path in sorted(md_files()):
        count += 1
        all_errors.extend(check_file(path))

    if all_errors:
        for err in all_errors:
            print(f"  ОШИБКА: {err}")
        print(f"  Проверено файлов: {count}, ошибок: {len(all_errors)}")
        return 1

    print(f"  {count} Markdown-файлов, ошибок линта нет")
    return 0


if __name__ == "__main__":
    sys.exit(main())
