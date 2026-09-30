"""Immutable experiment directories and JSON records, without importing TensorFlow."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import hashlib
import importlib.metadata
import json
import os
import platform

from course_utils import OUTPUT, ROOT, repository_path


def write_record(path, data):
    """Create once: rerunning a save cell must never replace previous evidence."""
    content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(content)


def new_experiment(name, config):
    parent = OUTPUT / "experiments" / name
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    directory.mkdir()
    versions = {}
    for package in ["numpy", "scikit-learn", "tensorflow", "tensorflow-cpu", "keras", "Pillow"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    # Hash course source files so records remain useful even with uncommitted edits.
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / "Code").glob("*.py"))}
    write_record(directory / "config.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                 "python": platform.python_version(), "packages": versions, "source_sha256": sources,
                 "config": config})
    print("Experiment:", directory)
    return directory


def selected_pet_manifest():
    explicit = os.environ.get("COURSE_PET_MANIFEST")
    if explicit:
        path = repository_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"COURSE_PET_MANIFEST does not exist: {path}")
        return path
    candidates = sorted((OUTPUT / "experiments" / "day40").glob("*/pets_manifest.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("Run Day 40 first, or set COURSE_PET_MANIFEST to an existing manifest.")
    raise ValueError("Multiple data splits found. Set COURSE_PET_MANIFEST explicitly; "
                     "no latest split is selected automatically.\n" + "\n".join(map(str, candidates)))
