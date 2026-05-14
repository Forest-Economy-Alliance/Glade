"""
Extract original images from duplicate groups based on earliest capture time.
Uses EXIF metadata, with fallback to provided metadata CSV file.
"""
import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".avif", ".gif"}


def _get_exif_datetime(image_path: str) -> Optional[str]:
    """Extract datetime from EXIF metadata. Returns ISO format string or None."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name in ("DateTime", "DateTimeOriginal", "DateTimeDigitized"):
                # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                # Convert to ISO format: "YYYY-MM-DD HH:MM:SS"
                return value.replace(":", "-", 2)
        return None
    except Exception:
        return None


def _load_metadata_csv(
    csv_path: str,
    image_name_column: str,
    datetime_column: str,
) -> Dict[str, str]:
    """Load metadata CSV and return mapping: image_name -> datetime_str."""
    try:
        df = pd.read_csv(csv_path)
        if image_name_column not in df.columns or datetime_column not in df.columns:
            raise ValueError(
                f"CSV must contain columns: '{image_name_column}' and '{datetime_column}'"
            )
        # Create mapping, converting datetime to string if needed
        mapping = {}
        for _, row in df.iterrows():
            img_name = str(row[image_name_column]).strip()
            dt_value = str(row[datetime_column]).strip()
            if img_name and dt_value:
                mapping[img_name] = dt_value
        return mapping
    except Exception as e:
        print(f"Warning: Could not load metadata CSV: {e}")
        return {}


def _parse_datetime_safe(dt_str: str) -> tuple:
    """Parse datetime string to comparable tuple. Returns (year, month, day, hour, minute, second)."""
    try:
        # Handle formats: "YYYY-MM-DD HH:MM:SS" or "YYYY:MM:DD HH:MM:SS"
        dt_str = dt_str.replace(":", "-", 2)  # Replace first 2 colons with dashes
        if "T" in dt_str:
            date_part, time_part = dt_str.split("T")
            date_part = date_part.replace("-", "-")
            time_part = time_part.replace(":", ":")
        else:
            parts = dt_str.split(" ")
            if len(parts) >= 2:
                date_part = parts[0]
                time_part = parts[1]
            else:
                date_part = parts[0]
                time_part = "00:00:00"

        date_components = tuple(int(x) for x in date_part.split("-"))
        time_components = tuple(int(x) for x in time_part.split(":"))
        return date_components + time_components
    except Exception:
        return (9999, 99, 99, 99, 99, 99)  # Return far future as fallback


def find_original_per_group(
    merged_groups_root: Path,
    metadata_csv: Optional[str] = None,
    image_name_column: str = "image_name",
    datetime_column: str = "hh_information-datetime",
) -> pd.DataFrame:
    """
    For each merged group folder, identify the original image (earliest capture time).
    Returns DataFrame with columns: group_id, original_image, capture_time
    """
    if not merged_groups_root.exists():
        raise FileNotFoundError(f"Merged groups root not found: {merged_groups_root}")

    # Load metadata CSV if provided
    metadata_mapping: Dict[str, str] = {}
    if metadata_csv and os.path.exists(metadata_csv):
        metadata_mapping = _load_metadata_csv(
            csv_path=metadata_csv,
            image_name_column=image_name_column,
            datetime_column=datetime_column,
        )
        print(f"Loaded {len(metadata_mapping)} entries from metadata CSV")

    results = []

    for group_folder in sorted(merged_groups_root.iterdir()):
        if not group_folder.is_dir():
            continue

        group_id = group_folder.name
        images_with_times: list[Tuple[str, tuple]] = []

        # Collect all images in this group with their capture times
        for root, _, files in os.walk(group_folder):
            for file_name in files:
                if Path(file_name).suffix.lower() not in IMAGE_EXTS:
                    continue

                file_path = Path(root) / file_name
                capture_time = None

                # Try EXIF first
                capture_time_str = _get_exif_datetime(str(file_path))
                if capture_time_str:
                    capture_time = _parse_datetime_safe(capture_time_str)

                # Fall back to metadata CSV
                if not capture_time and file_name in metadata_mapping:
                    capture_time = _parse_datetime_safe(metadata_mapping[file_name])

                # If still no time, use file modification time as last resort
                if not capture_time:
                    mtime = os.path.getmtime(file_path)
                    from datetime import datetime
                    dt = datetime.fromtimestamp(mtime)
                    capture_time = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

                images_with_times.append((file_name, capture_time))

        if not images_with_times:
            continue

        # Find earliest (original)
        original_image, original_time = min(images_with_times, key=lambda x: x[1])
        results.append(
            {
                "group_id": group_id,
                "original_image": original_image,
                "capture_time": original_time,
            }
        )

    return pd.DataFrame(results)


def extract_originals_to_folder(
    merged_groups_root: Path,
    originals_folder: Path,
    metadata_csv: Optional[str] = None,
    image_name_column: str = "image_name",
    datetime_column: str = "hh_information-datetime",
) -> Tuple[pd.DataFrame, int]:
    """
    Extract original images from each group and copy to originals folder.
    Returns (DataFrame with extraction info, total copied count).
    """
    originals_folder.mkdir(parents=True, exist_ok=True)

    originals_df = find_original_per_group(
        merged_groups_root=merged_groups_root,
        metadata_csv=metadata_csv,
        image_name_column=image_name_column,
        datetime_column=datetime_column,
    )

    copied_count = 0
    copied_names = []

    for _, row in originals_df.iterrows():
        group_id = row["group_id"]
        original_image = row["original_image"]

        src_path = merged_groups_root / group_id / original_image
        if src_path.exists():
            prefixed_name = f"{group_id}_{original_image}"
            dst_path = originals_folder / prefixed_name
            if dst_path.exists():
                stem = Path(prefixed_name).stem
                suffix = Path(prefixed_name).suffix
                counter = 1
                while dst_path.exists():
                    dst_path = originals_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            copied_names.append(dst_path.name)
        else:
            copied_names.append(None)

    originals_df = originals_df.copy()
    originals_df["copied_image_name"] = copied_names

    return originals_df, copied_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract original images from merged duplicate groups based on capture time."
    )
    parser.add_argument(
        "--merged-groups-root",
        required=True,
        help="Path to exact_duplicates_merged folder containing merged_group_* folders.",
    )
    parser.add_argument(
        "--originals-folder",
        required=True,
        help="Output folder where original images will be copied.",
    )
    parser.add_argument(
        "--metadata-csv",
        required=False,
        help="Optional CSV file with image metadata and datetime column.",
    )
    parser.add_argument(
        "--image-name-column",
        default="image_name",
        help="Column name in metadata CSV that contains image filenames.",
    )
    parser.add_argument(
        "--datetime-column",
        default="hh_information-datetime",
        help="Column name in metadata CSV that contains capture datetime.",
    )
    args = parser.parse_args()

    merged_groups_root = Path(args.merged_groups_root).resolve()
    originals_folder = Path(args.originals_folder).resolve()

    originals_df, copied_count = extract_originals_to_folder(
        merged_groups_root=merged_groups_root,
        originals_folder=originals_folder,
        metadata_csv=args.metadata_csv,
        image_name_column=args.image_name_column,
        datetime_column=args.datetime_column,
    )

    # Save summary CSV
    summary_csv = originals_folder.parent / "originals_extraction_summary.csv"
    originals_df.to_csv(summary_csv, index=False)

    print(f"✅ Extracted {copied_count} original images to: {originals_folder}")
    print(f"✅ Summary CSV saved to: {summary_csv}")
    print(f"✅ Total groups processed: {len(originals_df)}")


if __name__ == "__main__":
    main()
