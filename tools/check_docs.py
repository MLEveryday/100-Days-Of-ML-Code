"""Execute standalone theory examples and check active local Markdown links."""
from pathlib import Path
import ast
import os
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    lessons = sorted((ROOT / "docs/lessons").glob("day-*.md"))
    for path in lessons:
        examples = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
        if not examples:
            errors.append(f"Missing example: {path}")
        for example in examples:
            ast.parse(example)
            with tempfile.TemporaryDirectory(prefix="ml-theory-") as directory:
                result = subprocess.run([sys.executable, "-c", example], cwd=directory,
                                        capture_output=True, text=True, timeout=30,
                                        env={**os.environ, "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1"})
                if path.stem in {"day-51", "day-52", "day-53"}:
                    image = Path(directory) / (path.stem.replace("-", "") + "_example.png")
                    if not image.is_file() or image.stat().st_size == 0:
                        errors.append(f"No plot produced: {path}")
            if result.returncode:
                errors.append(f"{path}: {result.stderr}")
        print(f"Checked theory Day {path.stem[-2:]}", flush=True)
    solution_blocks = 0
    # Worked solutions contain optional standalone numerical checks.
    for path in sorted((ROOT / "docs/solutions").glob("day-*.md")):
        for example in re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S):
            solution_blocks += 1
            ast.parse(example)
            with tempfile.TemporaryDirectory(prefix="ml-solution-") as directory:
                result = subprocess.run([sys.executable, "-c", example], cwd=directory,
                                        capture_output=True, text=True, timeout=30,
                                        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
            if result.returncode:
                errors.append(f"{path}: {result.stderr}")
    paths = [ROOT / "README.md", ROOT / "FAQ.MD", ROOT / "datasets/readme.md"]
    paths += list((ROOT / "Code").glob("Day*.md")) + list((ROOT / "docs").rglob("*.md"))
    historical = {"day-by-day-review.md", "original-roadmap.md"}
    for path in paths:
        if path.name in historical:
            continue
        for target in re.findall(r"\]\(([^\s)]+)\)", path.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            if not (path.parent / unquote(parsed.path)).exists():
                errors.append(f"Broken local link in {path.relative_to(ROOT)}: {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Passed {len(lessons)} theory lessons, {solution_blocks} solution code blocks and active local links.")


if __name__ == "__main__":
    main()
