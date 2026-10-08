"""Data loading shared by Days 39-42; never downloads or unpickles data."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import tensorflow as tf
from course_utils import DATA, repository_path

SEED = 42


def configure():
    tf.keras.utils.set_random_seed(SEED)
    # Global thread settings cannot change after TensorFlow starts; notebook reruns
    # only reset seeds. Dataset thread limits are applied in bounded_dataset().


def bounded_dataset(dataset):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).prefetch(1)


def array_dataset(X, y, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        dataset = dataset.shuffle(len(X), seed=SEED)
    return bounded_dataset(dataset.batch(32))


def pet_root():
    return repository_path(os.environ.get("COURSE_PET_IMAGES", DATA / "PetImages"))


def prepare_pets(root, limit_per_class=None):
    """Check images, deduplicate by content and split 60/20/20 per class.

    Selected byte-identical files cannot cross splits. Related photos need group metadata
    for stronger leakage prevention; this dataset does not provide it.
    Scanning stops at limit_per_class: rejected describes inspected files only.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError
    if limit_per_class is not None and (not isinstance(limit_per_class, int) or limit_per_class < 10):
        raise ValueError("limit_per_class must be an integer >= 10 or None")
    root = Path(root).resolve()
    records, rejected, seen = [], [], {}
    rng = np.random.default_rng(SEED)
    counts = {}
    for label, category in enumerate(["Dog", "Cat"]):
        directory = root / category
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {directory}. See docs/setup.md, then run Day 40.")
        paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng.shuffle(paths)
        valid = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                # Decode pixels and apply the same orientation handling as pet_dataset.
                with Image.open(path) as image:
                    ImageOps.exif_transpose(image).convert("RGB").load()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # Pillow reports some damaged PNGs (bad chunk checksum) as SyntaxError.
            except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as exc:
                rejected.append({"path": str(path.relative_to(root)), "reason": str(exc)})
                continue
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError(f"Identical image has conflicting labels: {path}")
                rejected.append({"path": str(path.relative_to(root)), "reason": "duplicate content"})
                continue
            seen[digest] = label
            valid.append({"path": str(path.relative_to(root)), "label": label, "sha256": digest})
            if limit_per_class is not None and len(valid) >= limit_per_class:
                break
        if len(valid) < 10:
            raise ValueError(f"Need at least 10 valid distinct images per class: {category} has {len(valid)}")
        n_train, n_valid = int(len(valid) * 0.6), int(len(valid) * 0.2)
        for i, record in enumerate(valid):
            record["split"] = "train" if i < n_train else "validation" if i < n_train + n_valid else "test"
        records.extend(valid)
        counts[category] = len(valid)
    result = {"seed": SEED, "root": str(root), "class_names": ["Dog", "Cat"],
              "counts": counts, "records": records, "rejected": rejected}
    return result


def save_manifest(manifest, config=None):
    from experiments import new_experiment, write_record
    directory = new_experiment("day40", config or {"seed": SEED, "counts": manifest["counts"]})
    path = directory / "pets_manifest.json"
    write_record(path, manifest)
    print("Manifest:", path, "counts:", manifest["counts"], "rejected:", len(manifest["rejected"]))
    return path


def load_manifest():
    from experiments import selected_pet_manifest
    path = selected_pet_manifest()
    print("Using data split:", path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    """Refuse changed files or overlapping splits before reporting model quality."""
    root = Path(manifest["root"])
    seen_paths, seen_hashes = set(), set()
    groups = {(split, label): 0 for split in ("train", "validation", "test") for label in (0, 1)}
    if manifest.get("class_names") != ["Dog", "Cat"]:
        raise ValueError("Expected class order Dog=0, Cat=1")
    for record in manifest["records"]:
        group = record["split"], record["label"]
        if group not in groups:
            raise ValueError(f"Invalid split/label: {group}")
        path = (root / record["path"]).resolve()
        if path in seen_paths or record["sha256"] in seen_hashes:
            raise ValueError("Duplicate file/content in manifest; recreate splits with Day 40")
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Image changed since Day 40: {path}; recreate and record a new split")
        seen_paths.add(path)
        seen_hashes.add(record["sha256"])
        groups[group] += 1
    if not all(groups.values()):
        raise ValueError("Every split must contain both Dog and Cat")


def pet_dataset(manifest, split, training=False, image_size=(64, 64)):
    from PIL import Image, ImageOps
    records = [r for r in manifest["records"] if r["split"] == split]
    if not records:
        raise ValueError(f"Empty split: {split}")
    root = Path(manifest["root"])

    height, width = image_size
    # Shuffle small integer indices BEFORE decoding: full permutation, low memory,
    # and no bias from a buffer initially filled with a single class.
    indices = tf.data.Dataset.range(len(records))
    if training:
        indices = indices.shuffle(len(records), seed=SEED, reshuffle_each_iteration=True)

    def decode(index):
        record = records[int(index.numpy())]
        with Image.open(root / record["path"]) as image:
            pixels = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize(
                (width, height), resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return pixels, np.float32(record["label"])

    def load(index):
        pixels, label = tf.py_function(decode, [index], (tf.float32, tf.float32))
        pixels.set_shape((height, width, 3))
        label.set_shape(())
        return pixels, label

    dataset = indices.map(load, num_parallel_calls=1)
    return bounded_dataset(dataset.batch(32))
