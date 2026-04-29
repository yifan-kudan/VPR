from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


IMAGE_NAME_RE = re.compile(r"^IMG_(\d+)\.(jpe?g)$", re.IGNORECASE)


def get_existing_image_numbers(image_dir: Path) -> set[int]:
    image_numbers = set()

    for image_path in image_dir.iterdir():
        match = IMAGE_NAME_RE.match(image_path.name)
        if match:
            image_numbers.add(int(match.group(1)))

    return image_numbers


def expand_image_spec(image_spec: str) -> list[int]:
    image_numbers = []

    for part in image_spec.split(";"):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)

            if end < start:
                raise ValueError(
                    f"Invalid descending range: {part}. "
                    "Use ';' to separate multiple ranges, and make each range "
                    "increase from start to end, for example: 3529-3531;3598-3602."
                )

            image_numbers.extend(range(start, end + 1))
        else:
            image_numbers.append(int(part))

    return image_numbers


def expand_label_csv(
    input_csv: str | Path,
    image_dir: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    input_csv = Path(input_csv)
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)

    labels_df = pd.read_csv(input_csv)
    existing_image_numbers = get_existing_image_numbers(image_dir)

    expanded_rows = []
    missing_images = []

    for _, row in labels_df.iterrows():
        image_numbers = expand_image_spec(str(row["image"]))

        for image_number in image_numbers:
            if image_number not in existing_image_numbers:
                missing_images.append(image_number)
                continue

            expanded_row = row.copy()
            expanded_row["image"] = image_number
            expanded_rows.append(expanded_row)

    expanded_df = pd.DataFrame(expanded_rows, columns=labels_df.columns)
    expanded_df.to_csv(output_csv, index=False)

    print(f"Read labels from: {input_csv}")
    print(f"Checked images in: {image_dir}")
    print(f"Wrote expanded labels to: {output_csv}")
    print(f"Rows written: {len(expanded_df)}")

    if missing_images:
        missing_text = ", ".join(str(number) for number in sorted(set(missing_images)))
        print(f"Skipped missing images: {missing_text}")

    return expanded_df


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_image_dir = project_root / "data" / "images" / "converted_jpeg"

    parser = argparse.ArgumentParser(
        description="Expand ranged image labels into one CSV row per existing image."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_image_dir / "label_init.csv",
        help="Initial label CSV with ranges in the image column.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Directory containing files such as IMG_3426.jpg.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_image_dir / "labels.csv",
        help="Output CSV with one row per existing image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expand_label_csv(args.input, args.image_dir, args.output)


if __name__ == "__main__":
    main()
