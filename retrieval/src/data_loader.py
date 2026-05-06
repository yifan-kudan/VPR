from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# define the columns
REQUIRED_COLUMNS = [
    "image",
    "direction",
    "light",
    "weather",
    "indoor",
    "construction",
    "place",
]

# define the attribute of each image
@dataclass(frozen=True)
class ImageRecord:
    image: Path
    place: int
    direction: str
    light: str
    weather: str
    indoor: str
    construction: str

# define the dataset for retrieval
@dataclass(frozen=True)
class RetrievalDataset:
    references: list[ImageRecord]
    queries: list[ImageRecord]

    @property
    def reference_images(self) -> list[Path]:
        return [record.image for record in self.references]

    @property
    def reference_places(self) -> list[int]:
        return [record.place for record in self.references]

    @property
    def query_images(self) -> list[Path]:
        return [record.image for record in self.queries]

    @property
    def query_places(self) -> list[int]:
        return [record.place for record in self.queries]

# make sure the image path is absolute
def resolve_image_path(image_path: str, project_root: Path) -> Path:
    path = Path(image_path)

    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == project_root.name:
        return project_root.parent / path

    return project_root / path

# format the image record
def row_to_record(row: pd.Series, project_root: Path) -> ImageRecord:
    return ImageRecord(
        image=resolve_image_path(str(row["image"]), project_root),
        place=int(row["place"]),
        direction=str(row["direction"]),
        light=str(row["light"]),
        weather=str(row["weather"]),
        indoor=str(row["indoor"]),
        construction=str(row["construction"]),
    )

# make sure the columns are correct
def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{csv_path} is missing required columns: {missing_text}")

# make sure the corresponding image files are exist
def validate_image_files(records: list[ImageRecord]) -> None:
    missing_images = [record.image for record in records if not record.image.exists()]

    if missing_images:
        missing_text = "\n".join(f"- {image}" for image in missing_images[:20])
        extra_count = len(missing_images) - 20

        if extra_count > 0:
            missing_text += f"\n- ... and {extra_count} more"

        raise FileNotFoundError(f"Some images listed in the CSV do not exist:\n{missing_text}")


def load_retrieval_dataset(
    csv_path: str | Path,
    project_root: str | Path | None = None,
    validate_files: bool = True,
    n_references: int = 1,
) -> RetrievalDataset:
    csv_path = Path(csv_path)

    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    else:
        project_root = Path(project_root)

    df = pd.read_csv(csv_path)
    validate_columns(df, csv_path)

    records = [row_to_record(row, project_root) for _, row in df.iterrows()]

    if validate_files:
        validate_image_files(records)

    references = []
    queries = []

    # add logic to allow change the number of reference images per place,
    # the rest of images will be used as queries
    for _, place_df in df.groupby("place", sort=True):
        place_records = [row_to_record(row, project_root) for _, row in place_df.iterrows()]

        if len(place_records) <= n_references:
            references.extend(place_records)
        else:
            references.extend(place_records[:n_references])
            queries.extend(place_records[n_references:])

    return RetrievalDataset(references=references, queries=queries)


# def main() -> None:
#     project_root = Path(__file__).resolve().parents[2]
#     csv_path = project_root / "retrieval" / "data" / "images" / "converted_jpeg" / "labels_refined.csv"
#     dataset = load_retrieval_dataset(csv_path, project_root=project_root)

#     print(f"References: {len(dataset.references)}")
#     print("dataset references:", dataset.references)
#     print(f"Queries: {len(dataset.queries)}")
#     print(f"Places: {len(set(dataset.reference_places))}")


# if __name__ == "__main__":
#     main()
