import os
import shutil
import pandas as pd
from pathlib import Path


def save_duplicate_groups(
    groups: dict,
    image_dir: str,
    output_dir: str,
    csv_name: str = "duplicate_groups.csv",
    copy_files: bool = True
):
    """
    Save duplicate image groups into folders.

    Parameters
    ----------
    groups : dict
        {group_id: [image1, image2, ...]}
    image_dir : str
        Folder where original images exist
    output_dir : str
        Folder to store grouped duplicates
    csv_name : str
        CSV filename
    copy_files : bool
        If True, copy images into folders

    Returns
    -------
    pandas.DataFrame
    """

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for group_id, images in groups.items():

        if len(images) < 2:
            continue

        group_folder = f"{group_id}"
        group_dir = output_dir / group_folder

        if copy_files:
            group_dir.mkdir(parents=True, exist_ok=True)

        for img in images:

            src = image_dir / img
            dst = group_dir / img

            records.append({
                "image_name": img,
                "group_id": group_id
            })

            if copy_files and src.exists():
                shutil.copy2(src, dst)
            elif not src.exists():
                print(f"⚠️ Missing file: {src}")

    df = pd.DataFrame(records)

    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved {len(df)} duplicate images")
    print(f"📄 CSV saved to {csv_path}")

    return df
