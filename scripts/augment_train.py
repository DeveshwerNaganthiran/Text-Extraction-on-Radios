import argparse
from pathlib import Path
import cv2
import numpy as np


def _apply_brightness_contrast(img, alpha: float, beta: float):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _apply_gamma(img, gamma: float):
    if gamma <= 0:
        return img
    inv = 1.0 / gamma
    table = (np.array([(i / 255.0) ** inv * 255 for i in range(256)])).astype("uint8")
    return cv2.LUT(img, table)


def _apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _apply_blur(img, k: int = 3):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def _apply_noise(img, sigma: float = 8.0):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_train(
    train_images_dir: Path,
    train_labels_dir: Path,
    copies_per_image: int,
    skip_if_name_contains: tuple[str, ...],
):
    if not train_images_dir.exists() or not train_labels_dir.exists():
        raise FileNotFoundError(
            f"Train dirs not found: {train_images_dir} / {train_labels_dir}. Run scripts/split_data.py first."
        )

    images = list(train_images_dir.glob("*.jpg"))
    images.extend(list(train_images_dir.glob("*.png")))
    images.extend(list(train_images_dir.glob("*.jpeg")))

    transforms = [
        ("aug_bright", lambda im: _apply_brightness_contrast(im, alpha=1.15, beta=18)),
        ("aug_dark", lambda im: _apply_brightness_contrast(im, alpha=0.85, beta=-12)),
        ("aug_contrast", lambda im: _apply_brightness_contrast(im, alpha=1.25, beta=0)),
        ("aug_gamma_hi", lambda im: _apply_gamma(im, gamma=1.35)),
        ("aug_gamma_lo", lambda im: _apply_gamma(im, gamma=0.80)),
        ("aug_clahe", lambda im: _apply_clahe(im)),
        ("aug_blur", lambda im: _apply_blur(im, k=3)),
        ("aug_noise", lambda im: _apply_noise(im, sigma=8.0)),
    ]

    created = 0
    skipped = 0
    missing_labels = 0

    for img_path in images:
        name_lower = img_path.name.lower()
        if "_aug_" in name_lower or "aug_" in name_lower:
            skipped += 1
            continue

        if any(token.lower() in name_lower for token in skip_if_name_contains):
            skipped += 1
            continue

        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            skipped += 1
            continue

        selected = transforms[: max(0, min(copies_per_image, len(transforms)))]
        for suffix, fn in selected:
            out_name = f"{img_path.stem}_{suffix}{img_path.suffix}"
            out_img_path = train_images_dir / out_name
            out_lbl_path = train_labels_dir / f"{Path(out_name).stem}.txt"

            if out_img_path.exists() and out_lbl_path.exists():
                continue

            out_img = fn(img)
            cv2.imwrite(str(out_img_path), out_img)
            out_lbl_path.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")
            created += 1

    return {
        "total_images": len(images),
        "created": created,
        "skipped": skipped,
        "missing_labels": missing_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", default="data/train/images")
    parser.add_argument("--train-labels", default="data/train/labels")
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument(
        "--skip-if-name-contains",
        nargs="*",
        default=["different_lighting"],
    )

    args = parser.parse_args()

    stats = augment_train(
        train_images_dir=Path(args.train_images),
        train_labels_dir=Path(args.train_labels),
        copies_per_image=args.copies,
        skip_if_name_contains=tuple(args.skip_if_name_contains),
    )

    print("🧪 Train augmentation complete")
    print(f" Total base images scanned: {stats['total_images']}")
    print(f" Augmented images created: {stats['created']}")
    print(f" Skipped (already augmented / filtered / unreadable): {stats['skipped']}")
    print(f" Missing labels: {stats['missing_labels']}")


if __name__ == "__main__":
    main()
