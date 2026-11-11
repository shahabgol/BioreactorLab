"""Reject tracked study inputs and generated artifacts before CI or a push."""

import subprocess
from pathlib import PurePosixPath

ROOT_FILES = {
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    ".python-version",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
}
DOCS = {"docs/WORKBOOK_FORMAT.md", "docs/source/.gitkeep"}
EMPTY_DIRS = {
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
    "outputs/.gitkeep",
}
TEXT_EXTENSIONS = {".css", ".md", ".ps1", ".py", ".svg", ".txt", ".yaml", ".yml"}
SOURCE_DIRS = {".github", "environment", "main", "scripts", "tests"}


def is_public(name):
    path = PurePosixPath(name)
    if name == "tests/integration/test_local_sources.py":
        return False
    if name in ROOT_FILES | DOCS | EMPTY_DIRS:
        return True
    return path.parts[0] in SOURCE_DIRS and (
        path.suffix.lower() in TEXT_EXTENSIONS or path.name == ".gitkeep"
    )


def main():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], text=True, encoding="utf-8")
    blocked = sorted(name for name in tracked.split("\0") if name and not is_public(name))
    if blocked:
        raise SystemExit("Remove these files from the Git index:\n" + "\n".join(blocked))
    print("Public tree check passed.")


if __name__ == "__main__":
    main()
