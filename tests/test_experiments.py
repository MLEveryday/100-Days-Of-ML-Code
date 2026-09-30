"""History must survive reruns; split selection must never silently switch."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
import experiments


class ExperimentHistory(unittest.TestCase):
    def test_unique_directories_and_write_once_records(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(experiments, "OUTPUT", Path(directory)):
            first = experiments.new_experiment("day39", {"seed": 42})
            experiments.write_record(first / "metrics.json", {"accuracy": 0.5})
            before = {p.name: p.read_bytes() for p in first.iterdir()}
            second = experiments.new_experiment("day39", {"seed": 42})
            self.assertNotEqual(first, second)
            with self.assertRaises(FileExistsError):
                experiments.write_record(first / "metrics.json", {"accuracy": 1.0})
            self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
            config = json.loads((first / "config.json").read_text())
            self.assertIn("Code/Day 39.py", config["source_sha256"])

    def test_invalid_json_does_not_leave_a_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            with self.assertRaises(ValueError):
                experiments.write_record(path, {"loss": float("nan")})
            self.assertFalse(path.exists())

    def test_manifest_ambiguity_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(experiments, "OUTPUT", Path(directory)), patch.dict(os.environ):
            os.environ.pop("COURSE_PET_MANIFEST", None)
            with self.assertRaises(FileNotFoundError):
                experiments.selected_pet_manifest()
            first = experiments.new_experiment("day40", {}) / "pets_manifest.json"
            experiments.write_record(first, {"example": 1})
            self.assertEqual(experiments.selected_pet_manifest(), first)
            second = experiments.new_experiment("day40", {}) / "pets_manifest.json"
            experiments.write_record(second, {"example": 2})
            with self.assertRaisesRegex(ValueError, "Multiple"):
                experiments.selected_pet_manifest()
            os.environ["COURSE_PET_MANIFEST"] = str(first)
            self.assertEqual(experiments.selected_pet_manifest(), first)
            os.environ["COURSE_PET_MANIFEST"] = str(first.parent / "missing.json")
            with self.assertRaises(FileNotFoundError):
                experiments.selected_pet_manifest()
