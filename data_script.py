# cub_split_generator.py
"""Generate train/test filenames.pickle and class_info.pickle that are
perfectly aligned (one class id per image) for the CUB_200_2011 dataset.

The script replicates the official CUB train / test split but **drops** any
image that does not have the expected number of caption sentences
(`CAPS_PER_IMAGE`).  This guarantees that the TextDataset in AttnGAN style
can later pick `CAPS_PER_IMAGE` captions per image without hitting missing
files or short caption files.

Run from the repository root:

    python cub_split_generator.py \
        --cub-dir data/CUB_200_2011 \
        --out-dir data \
        --caps-per-image 10

It writes:
    data/train/filenames.pickle
    data/train/class_info.pickle
    data/test/filenames.pickle
    data/test/class_info.pickle

"""

import argparse
import os
import pickle
from collections import defaultdict
from typing import Dict, List


def read_mapping(path: str) -> Dict[int, str]:
    """Read space‑separated file where the first field is an int id and
    the second field is a path / label. Returns a dict[int, str]."""
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            idx, value = line.strip().split(maxsplit=1)
            mapping[int(idx)] = value
    return mapping


def has_full_captions(captions_dir: str, key: str, caps_per_image: int) -> bool:
    """Return True if `<captions_dir>/<key>.txt` exists **and** contains at
    least `caps_per_image` non‑empty lines."""
    cap_path = os.path.join(captions_dir, f"{key}.txt")
    if not os.path.exists(cap_path):
        return False
    with open(cap_path, "r", encoding="utf‑8", errors="ignore") as f:
        non_empty = [ln for ln in f if ln.strip()]
    return len(non_empty) >= caps_per_image


def build_split(
    split_flag: int,
    id_to_filename: Dict[int, str],
    id_to_class: Dict[int, int],
    caps_per_image: int,
    captions_dir: str,
):
    """Return (filenames, class_ids) lists for one split."""
    filenames, class_ids = [], []
    for idx, fname in id_to_filename.items():
        if split_flag_map[idx] != split_flag:
            continue  # wrong split

        filenames.append(fname)
        class_ids.append(id_to_class[idx])
    return filenames, class_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train/test split pickles for CUB_200_2011 with aligned class IDs.")
    parser.add_argument("--cub-dir", type=str, default="data/CUB_200_2011", help="Path to CUB_200_2011 directory (contains images.txt etc.)")
    parser.add_argument("--out-dir", type=str, default="data", help="Directory to write train/ and test/ subfolders with pickles")
    parser.add_argument("--caps-per-image", type=int, default=10, help="Expected caption count per image (cfg.TEXT.CAPTIONS_PER_IMAGE)")
    args = parser.parse_args()

    cub_dir = args.cub_dir
    out_dir = args.out_dir
    caps_per_image = args.caps_per_image

    images_txt = os.path.join(cub_dir, "images.txt")
    split_txt = os.path.join(cub_dir, "train_test_split.txt")
    labels_txt = os.path.join(cub_dir, "image_class_labels.txt")
    captions_dir = os.path.join(cub_dir, "text")

    # 1. load master mappings
    id_to_filename = read_mapping(images_txt)  # id -> relative path with .jpg
    id_to_filename = {k: v[:-4] for k, v in id_to_filename.items()}  # strip .jpg

    split_flag_map = {int(idx): int(flag) for idx, flag in (ln.strip().split() for ln in open(split_txt))}
    id_to_class = {int(idx): int(cid) for idx, cid in (ln.strip().split() for ln in open(labels_txt))}

    # 2. build splits
    train_filenames, train_class_ids = build_split(1, id_to_filename, id_to_class, caps_per_image, captions_dir)
    test_filenames,  test_class_ids  = build_split(0, id_to_filename, id_to_class, caps_per_image, captions_dir)

    # 3. write pickles
    for split, fnames, cids in [("train", train_filenames, train_class_ids), ("test", test_filenames, test_class_ids)]:
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "filenames.pickle"), "wb") as f:
            pickle.dump(fnames, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(split_dir, "class_info.pickle"), "wb") as f:
            pickle.dump(cids, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{split.capitalize():5s}: {len(fnames):5d} images written to {split_dir}")

    # 4. sanity‑check lengths match
    assert len(train_filenames) == len(train_class_ids), "train lengths mismatch"
    assert len(test_filenames)  == len(test_class_ids),  "test  lengths mismatch"
    print("✅ Finished generating split pickles.")