"""Execute course scripts and notebooks in clean processes; no Kafka scripts."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def fixtures(root):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(42)
    for label, category in enumerate(["Dog", "Cat"]):
        directory = root / category
        directory.mkdir(parents=True)
        for i in range(24):
            pixels = rng.integers(0, 80, (32, 32, 3), dtype=np.uint8)
            pixels[:, :, label] += 160
            Image.fromarray(pixels).save(directory / f"{i}.png")
        (directory / "broken.jpg").write_bytes(b"not an image")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true", help="Only Days 39-42, using small synthetic image fixtures")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "tools/sync_lessons.py"), "--check"], check=True)
    paths = sorted((ROOT / "Code").glob("Day*.py"), key=lambda p: int(re.search(r"Day (\d+)", p.name)[1]))
    paths = [p for p in paths if (39 <= int(re.search(r"Day (\d+)", p.name)[1]) <= 42) == args.deep]
    failures = []
    with tempfile.TemporaryDirectory(prefix="ml-course-check-") as temporary:
        temporary = Path(temporary)
        env = {**os.environ, "MPLBACKEND": "Agg", "COURSE_SHOW_PLOTS": "0",
               "COURSE_SMOKE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
        if args.deep:
            fixtures(temporary / "PetImages")
            env["COURSE_PET_IMAGES"] = str(temporary / "PetImages")
            env.pop("COURSE_PET_MANIFEST", None)  # Do not inherit a user-selected real dataset.
        # A temporary kernelspec selects precisely the Python running this checker.
        kernel = temporary / "kernels" / "course-check"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Course check", "language": "python"}))
        env["JUPYTER_PATH"] = str(temporary)
        for mode in ["script", "notebook"]:
            env["COURSE_OUTPUT_DIR"] = str(temporary / mode)
            for index, path in enumerate(paths):
                cwd = ROOT if index % 2 == 0 else ROOT / "Code"
                if mode == "script":
                    command = [sys.executable, str(path)]
                else:
                    nbpath = path.with_name(path.stem.replace("Day 11_k-NN", "Day 11_K-NN") + ".ipynb")
                    program = "import nbformat,sys; from nbclient import NotebookClient; nb=nbformat.read(sys.argv[1],as_version=4); NotebookClient(nb,timeout=180,kernel_name='course-check',resources={'metadata':{'path':sys.argv[2]}}).execute()"
                    command = [sys.executable, "-c", program, str(nbpath), str(cwd)]
                try:
                    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
                    ok = result.returncode == 0
                    if not ok:
                        print(result.stdout[-3000:], result.stderr[-6000:])
                except subprocess.TimeoutExpired:
                    ok = False
                    print("Timed out")
                if ok and args.deep:
                    day = int(re.search(r"Day (\d+)", path.name)[1])
                    runs = list((Path(env["COURSE_OUTPUT_DIR"]) / "experiments" / f"day{day}").glob("*"))
                    expected = {
                        39: ["config.json", "history.json", "metrics.json", "day39_mnist.keras", "completed.json", "architecture.json"],
                        40: ["config.json", "pets_manifest.json"],
                        41: ["config.json", "manifest.json", "history.json", "metrics.json", "classification.json", "day41_cnn.keras", "completed.json", "architecture.json"],
                        42: ["config.json", "manifest.json", "results.json", "width_8/best.keras", "width_16/best.keras"],
                    }[day]
                    if len(runs) != 1 or any(not (runs[0] / item).is_file() for item in expected):
                        print(f"Missing or ambiguous experiment artifacts for Day {day}")
                        ok = False
                print(f"{'PASS' if ok else 'FAIL'} {mode}: {path.stem}", flush=True)
                if not ok:
                    failures.append((mode, path.name))
    if failures:
        raise SystemExit(f"Failed lessons: {failures}")
    print(f"Passed {len(paths)} scripts and {len(paths)} notebooks.")


if __name__ == "__main__":
    main()
