#!/usr/bin/env python3
"""Create a verified tensor-md release from a clean Git checkout.

The script increments the project version, runs the same checks as CI, commits
the version change, creates an annotated ``vX.Y.Z`` tag, and pushes the commit
and tag.  Pushing the tag starts the trusted PyPI publishing workflow.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERSION_PATTERN = re.compile(r'^(version\s*=\s*")(?P<version>\d+\.\d+\.\d+)(")\s*$', re.MULTILINE)


def run(*command: str) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def output(*command: str) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def project_version() -> str:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def bump(version: str, kind: str) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def replace_version(text: str, new_version: str) -> str:
    updated, replacements = VERSION_PATTERN.subn(
        lambda match: f'{match.group(1)}{new_version}{match.group(3)}', text, count=1
    )
    if replacements != 1:
        raise RuntimeError("Could not find one PEP 621 version field in pyproject.toml.")
    return updated


def require_clean_checkout() -> None:
    changes = output("git", "status", "--porcelain")
    if changes:
        raise RuntimeError(
            "The checkout has uncommitted changes. Commit or stash them before releasing."
        )


def ensure_release_tools() -> None:
    """Install the local test and distribution tools when the active Python lacks them."""

    required_modules = ("pytest", "build", "twine")
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        print("Installing release tools for the active Python environment:", ", ".join(missing))
        run(sys.executable, "-m", "pip", "install", ".[test]", "build", "twine")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, tag, and publish a tensor-md release.")
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument("--major", action="store_const", const="major", dest="bump")
    version_group.add_argument("--minor", action="store_const", const="minor", dest="bump")
    version_group.add_argument("--patch", action="store_const", const="patch", dest="bump")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the next version without changing files, committing, tagging, or pushing.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Create the validated commit and tag locally without pushing them.",
    )
    args = parser.parse_args()
    bump_kind = args.bump or "patch"

    require_clean_checkout()
    current = project_version()
    next_version = bump(current, bump_kind)
    tag = f"v{next_version}"
    if output("git", "tag", "-l", tag):
        raise RuntimeError(f"Git tag {tag} already exists.")

    print(f"Preparing tensor-md {next_version} ({bump_kind} release).")
    if args.dry_run:
        return

    original_pyproject = PYPROJECT.read_text(encoding="utf-8")
    PYPROJECT.write_text(replace_version(original_pyproject, next_version), encoding="utf-8")
    committed = False
    try:
        ensure_release_tools()
        run(sys.executable, "-m", "pytest", "-q")
        with tempfile.TemporaryDirectory(prefix="tensor-md-release-") as temp_dir:
            run(sys.executable, "-m", "build", "--outdir", temp_dir)
            distributions = sorted(str(path) for path in Path(temp_dir).iterdir())
            if not distributions:
                raise RuntimeError("The build produced no distribution files.")
            run(sys.executable, "-m", "twine", "check", *distributions)

        run("git", "add", "pyproject.toml")
        run("git", "commit", "-m", f"Release v{next_version}")
        committed = True
        run("git", "tag", "-a", tag, "-m", f"Release {tag}")
        if not args.no_push:
            run("git", "push", "origin", "HEAD")
            run("git", "push", "origin", tag)
            print(f"Published {tag}: GitHub Actions will now build and upload it to PyPI.")
        else:
            print(f"Created {tag} locally. Push it with: git push origin HEAD {tag}")
    except Exception:
        if not committed:
            PYPROJECT.write_text(original_pyproject, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
