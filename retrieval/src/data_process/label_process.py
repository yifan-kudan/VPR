import argparse
from pathlib import Path

import pandas as pd

# define all columns of the output CSV
OUTPUT_COLUMNS = [
    "image",
    "direction",
    "light",
    "weather",
    "indoor",
    "construction",
    "place",
]

# seperate multiple values in a column by ";", for example "day;night", multiple image groups "3848-3857;3858-3867", and the different view of an image.
def split_groups(value: object) -> list[str]:
    return [part.strip() for part in str(value).split(";")]

# expand the image group "3848-3857" into a list of image numbers [3848, 3849, ..., 3857]
def expand_range(range_text: str) -> list[int]:
    range_text = range_text.strip()

    # check if the row is a single image or a range of images
    if "-" not in range_text:
        return [int(range_text)]

    # create the list of image numbers from the range
    start_text, end_text = range_text.split("-", maxsplit=1)
    start = int(start_text)
    end = int(end_text)

    if end < start:
        raise ValueError(f"Invalid descending image range: {range_text}")

    return list(range(start, end + 1))

# check the image directory for existing images, make sure each image set has 10 images
# if range is larger than 10, but there's only 10 images, then it's likely that 
# some images are deleted when selecting images. then jump to the next image number
def image_path_for_number(image_dir: Path, image_number: int) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        image_path = image_dir / f"IMG_{image_number}{suffix}"
        if image_path.exists():
            return image_path

    return None

# expand the image group into list, link the number with image file, and list the existing images of one place
def expand_existing_images(image_dir: Path, image_group: str) -> list[tuple[int, Path]]:
    existing_images = []

    # expand the image groups into list
    for image_number in expand_range(image_group):
        # check if image file exists for the image number, 
        # if exists, add to the list of existing images
        image_path = image_path_for_number(image_dir, image_number)
        if image_path is not None:
            existing_images.append((image_number, image_path))

    return existing_images


def project_relative_path(image_path: Path, project_root: Path) -> str:
    try:
        return str(Path(project_root.name) / image_path.relative_to(project_root))
    except ValueError:
        return str(image_path)

# for columns with multiple values, for example "day;night"
# this is used for split the lines that includes multiple image groups
def value_for_group(row: pd.Series, column: str, group_index: int, group_count: int) -> str:
    values = split_groups(row[column])

    if len(values) == 1:
        return values[0]

    if len(values) == group_count:
        return values[group_index]

    raise ValueError(
        f"Column '{column}' has {len(values)} values for {group_count} image groups "
        f"in row with image='{row['image']}'. Use either one value or one value per group."
    )

# The main function to rebuild the label CSV
def refine_labels(
    input_csv: str | Path,
    image_dir: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    input_csv = Path(input_csv)
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)
    project_root = Path(__file__).resolve().parents[3]

    # Read the initial labels with image groups
    labels_df = pd.read_csv(input_csv)
    refined_rows = []
    errors = []

    # iterate through each row of the label CSV
    for place_index, row in labels_df.iterrows():

        # split the image groups and view points into list
        image_groups = split_groups(row["image"])
        direction_sequence = split_groups(row["direction"])

        # identify how many image groups are in one row, it indicate 
        # how many images are there for one place
        group_count = len(image_groups)

        # split the image group
        for group_index, image_group in enumerate(image_groups):
            existing_images = expand_existing_images(image_dir, image_group)

            if len(existing_images) != len(direction_sequence):
                errors.append(
                    f"row {place_index + 2}: image group '{image_group}' has "
                    f"{len(existing_images)} existing images, but direction has "
                    f"{len(direction_sequence)} values. Check the files in {image_dir}"
                )
                continue
            
            # split the view point and link with the image
            for (_, image_path), direction in zip(existing_images, direction_sequence):
                try:
                    refined_rows.append(
                        {
                            "image": project_relative_path(image_path, project_root),
                            "direction": direction,

                            # split the columns with multiple values
                            # especially for the day;night
                            # TODO: Considering a restriction that the length of 
                            # image groups, light, weather, indoor, and construction 
                            # should be the same, to prevent the mismatch of labels,
                            # if add more scenarios later
                            "light": value_for_group(row, "light", group_index, group_count),
                            "weather": value_for_group(row, "weather", group_index, group_count),
                            "indoor": value_for_group(row, "indoor", group_index, group_count),
                            "construction": value_for_group(
                                row, "construction", group_index, group_count
                            ),
                            "place": place_index,
                        }
                    )
                except ValueError as error:
                    errors.append(f"row {place_index + 2}: {error}")

    if errors:
        error_text = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Could not refine labels:\n{error_text}")

    refined_df = pd.DataFrame(refined_rows, columns=OUTPUT_COLUMNS)
    refined_df.to_csv(output_csv, index=False)

    print(f"Read grouped labels from: {input_csv}")
    print(f"Checked images in: {image_dir}")
    print(f"Wrote refined labels to: {output_csv}")
    print(f"Rows written: {len(refined_df)}")
    print(f"Places written: {labels_df.shape[0]}")

    return refined_df


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_image_dir = project_root / "retrieval" / "data" / "images" / "converted_jpeg"

    parser = argparse.ArgumentParser(
        description="Expand place-level image labels into one row per image."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_image_dir / "labels.csv",
        help="Grouped label CSV with image ranges.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Directory containing files such as IMG_3735.jpg.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_image_dir / "labels_refined.csv",
        help="Output CSV with one row per image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refine_labels(args.input, args.image_dir, args.output)


if __name__ == "__main__":
    main()
