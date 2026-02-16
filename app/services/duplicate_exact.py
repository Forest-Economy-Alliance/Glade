import os
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from app.utils.image_hash import compute_file_hash

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_exact_duplicates(
    image_dir: str,
    base_output_dir: str,
    copy_files: bool = True,
    csv_name: str = "exact_duplicates.csv"
):
    """
    Detect exact duplicate images using pHash.

    Parameters
    ----------
    image_dir : str
        Folder containing raw images
    base_output_dir : str
        Base output folder from config
    copy_files : bool
        Copy duplicate images into grouped folders
    csv_name : str
        CSV filename to save results

    Returns
    -------
    pandas.DataFrame
        DataFrame of duplicate images
    """

    image_dir = Path(image_dir)
    base_output_dir = Path(base_output_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    exact_dir = base_output_dir / "exact_duplicates"
    exact_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔍 Computing pHashes...")

    records = []

    # ----------------------------
    # 1️⃣ Compute hashes
    # ----------------------------
    for file in tqdm(list(image_dir.iterdir())):

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTS:
            continue

        phash = compute_file_hash(str(file))


        if phash:
            records.append({
                "image_name": file.name,
                "phash": phash
            })

    df = pd.DataFrame(records)

    if df.empty:
        print("⚠️ No images processed.")
        return df

    print(f"✅ Processed {len(df)} images")

    # ----------------------------
    # 2️⃣ Group by hash
    # ----------------------------
    phash_groups = defaultdict(list)

    for _, row in df.iterrows():
        phash_groups[row["phash"]].append(row["image_name"])

    exact_dups_records = []

    print("\n📂 Saving duplicate groups...")

    for phash_value, images in phash_groups.items():

        if len(images) < 2:
            continue

        group_folder = f"phash_{phash_value}"
        group_dir = exact_dir / group_folder

        if copy_files:
            group_dir.mkdir(parents=True, exist_ok=True)

        for img_name in images:
            src = image_dir / img_name
            dst = group_dir / img_name

            exact_dups_records.append({
                "image_name": img_name,
                "phash": phash_value,
                "group_folder": group_folder
            })

            if copy_files and src.exists():
                shutil.copy2(src, dst)

    exact_df = pd.DataFrame(exact_dups_records)

    # ----------------------------
    # 3️⃣ Save CSV
    # ----------------------------
    csv_path = exact_dir / csv_name
    exact_df.to_csv(csv_path, index=False)

    print(f"\n✅ Found {len(exact_df)} duplicate images")
    print(f"📄 CSV saved to {csv_path}")

    return exact_df,df
