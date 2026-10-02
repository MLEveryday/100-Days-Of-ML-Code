"""Image preparation must not leak exact duplicates across evaluation splits."""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "optional deep-learning dependencies")
class PetSplits(unittest.TestCase):
    def test_reproducible_disjoint_splits_and_bad_image_reporting(self):
        from deep_utils import prepare_pets
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(9)
            for category in ["Dog", "Cat"]:
                directory = root / category
                directory.mkdir()
                for i in range(12):
                    Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(directory / f"{i}.png")
                shutil.copyfile(directory / "0.png", directory / "duplicate.png")
                (directory / "bad.jpg").write_bytes(b"bad image")
            first = prepare_pets(root)
            second = prepare_pets(root)
            self.assertEqual(first, second)
            self.assertEqual(len(first["rejected"]), 4)
            hashes = {}
            for split in ["train", "validation", "test"]:
                records = [r for r in first["records"] if r["split"] == split]
                self.assertEqual({r["label"] for r in records}, {0, 1})
                hashes[split] = {r["sha256"] for r in records}
            self.assertFalse(hashes["train"] & hashes["validation"])
            self.assertFalse(hashes["train"] & hashes["test"])
            self.assertFalse(hashes["validation"] & hashes["test"])
            shutil.copyfile(root / "Dog/0.png", root / "Cat/conflict.png")
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                prepare_pets(root)

    def test_png_with_bad_checksum_is_rejected_not_fatal(self):
        from deep_utils import prepare_pets
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(3)
            for category in ["Dog", "Cat"]:
                directory = root / category
                directory.mkdir()
                for i in range(10):
                    Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(directory / f"{i}.png")
            damaged = bytearray((root / "Dog/0.png").read_bytes())
            damaged[-20] ^= 0xFF  # inside the IDAT checksum; Pillow raises SyntaxError for this
            (root / "Dog/corrupt.png").write_bytes(bytes(damaged))
            result = prepare_pets(root)
            self.assertEqual([r["path"] for r in result["rejected"]], ["Dog/corrupt.png"])
            self.assertEqual(result["counts"], {"Dog": 10, "Cat": 10})

    def test_runtime_can_be_configured_again_after_tensorflow_started(self):
        import tensorflow as tf
        from deep_utils import configure
        _ = tf.constant([1.0]) + 1
        configure()
        first = tf.random.uniform((4,)).numpy()
        configure()
        np.testing.assert_array_equal(first, tf.random.uniform((4,)).numpy())

    def test_full_shuffle_and_non_square_image_shape(self):
        from deep_utils import pet_dataset
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # More consecutive Dog entries than the old 512-image buffer.
            for label in [0, 1]:
                Image.fromarray(np.full((8, 8, 3), label * 255, dtype=np.uint8)).save(root / f"{label}.png")
            manifest = {"root": str(root), "records": [
                {"path": f"{label}.png", "label": label, "split": "train"}
                for label in [0, 1] for _ in range(600)]}
            first_batch = next(iter(pet_dataset(manifest, "train", training=True, image_size=(12, 20))))
            images, labels = first_batch
            self.assertEqual(tuple(images.shape), (32, 12, 20, 3))
            self.assertEqual(set(labels.numpy()), {0.0, 1.0})
            again = next(iter(pet_dataset(manifest, "train", training=True, image_size=(12, 20))))
            np.testing.assert_array_equal(labels.numpy(), again[1].numpy())
            np.testing.assert_allclose(images.numpy()[:, 0, 0, 0], labels.numpy())

    def test_manifest_rejects_data_changes_and_overlap(self):
        from deep_utils import prepare_pets, validate_manifest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(4)
            for category in ["Dog", "Cat"]:
                (root / category).mkdir()
                for i in range(10):
                    Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(root / category / f"{i}.png")
            manifest = prepare_pets(root)
            validate_manifest(manifest)
            manifest["records"].append(manifest["records"][0].copy())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_manifest(manifest)
            manifest["records"].pop()
            first = root / manifest["records"][0]["path"]
            first.write_bytes(b"changed data")
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_manifest(manifest)

    def test_saving_manifest_twice_keeps_both_snapshots(self):
        from unittest.mock import patch
        import experiments
        from deep_utils import save_manifest
        with tempfile.TemporaryDirectory() as directory, patch.object(experiments, "OUTPUT", Path(directory)):
            original = {"counts": {"Dog": 10, "Cat": 10}, "rejected": [], "records": []}
            first = save_manifest(original)
            before = first.read_bytes()
            changed = {**original, "counts": {"Dog": 20, "Cat": 20}}
            second = save_manifest(changed)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), before)
            self.assertNotEqual(first.read_bytes(), second.read_bytes())

    def test_missing_classes_fail_clearly(self):
        from deep_utils import prepare_pets
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Day 40"):
                prepare_pets(directory)


if __name__ == "__main__":
    unittest.main()
