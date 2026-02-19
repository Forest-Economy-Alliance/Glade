import os
import pandas as pd


def groups_folder_to_csv(groups_folder: str, out_csv: str) -> pd.DataFrame:
    """
    Convert a folder of grouped images into CSV for LLM verification.

    Folder structure:
        groups_folder/
            group_0001/
                img1.jpg
                img2.jpg
            group_0002/
                img3.jpg

    Returns DataFrame and saves CSV.
    """

    if not os.path.exists(groups_folder):
        raise FileNotFoundError(groups_folder)

    rows = []

    # Sort folders for deterministic output
    group_dirs = sorted([
        d for d in os.listdir(groups_folder)
        if os.path.isdir(os.path.join(groups_folder, d))
    ])

    if not group_dirs:
        raise ValueError("No group folders found")

    for gid in group_dirs:
        folder_path = os.path.join(groups_folder, gid)

        images = sorted([
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ])

        if len(images) < 2:
            # Skip groups with only 1 image
            continue

        for img in images:
            rows.append({
                "group_id": gid,
                "image_name": img
            })

    if not rows:
        raise ValueError("No valid groups found")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"✅ CSV saved → {out_csv}")
    print(f"Groups: {df['group_id'].nunique()} | Images: {len(df)}")

    return df
