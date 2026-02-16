import os
import shutil
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import cv2
except Exception:
    cv2 = None


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Xn = X / norms
    # Cosine similarity = dot of normalized vectors
    return Xn @ Xn.T


def _load_image_cv(path: str, image_size: Tuple[int, int]):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


def _hsv_hist_feature(img, hist_bins: Tuple[int, int, int], ranges: Tuple[int, int, int, int, int, int]):
    hist = cv2.calcHist([img], channels=[0, 1, 2], mask=None, histSize=list(hist_bins), ranges=list(ranges))
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def find_near_duplicates_hsv_cosine(
    df: pd.DataFrame,
    image_dir: str,
    base_output_dir: str,
    image_size: Tuple[int, int] = (256, 256),
    hist_bins: Tuple[int, int, int] = (8, 8, 8),
    ranges: Tuple[int, int, int, int, int, int] = (0, 180, 0, 256, 0, 256),
    similarity_threshold: float = 0.90,
    output_subfolder: str = "hsv_cosine",
    csv_name: str = "near_duplicate_groups_hsv_cosine.csv",
    copy_files: bool = True,
) -> pd.DataFrame:
    """
    Cluster near-duplicates using HSV histogram features and cosine similarity.
    Returns a DataFrame of grouped near-duplicates and copies images into grouped folders.
    """

    if cv2 is None:
        raise ImportError("OpenCV is required. Install: pip install opencv-python")

    df = df.copy()
    df = df[df["image_name"].notna()].copy()

    image_names: List[str] = df["image_name"].astype(str).tolist()
    image_paths: List[str] = [os.path.join(image_dir, name) for name in image_names]

    out_dir = os.path.join(base_output_dir, output_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    features = {}
    valid_paths = []

    for name, path in tqdm(list(zip(image_names, image_paths)), total=len(image_paths)):
        if not os.path.exists(path):
            continue
        img = _load_image_cv(path, image_size)
        if img is None:
            continue
        feat = _hsv_hist_feature(img, hist_bins, ranges)
        features[name] = feat
        valid_paths.append(path)

    valid_names = list(features.keys())
    if not valid_names:
        return pd.DataFrame(columns=["group_id", "image_name", "src_path"])  # nothing to group

    feature_matrix = np.array([features[name] for name in valid_names])
    sim_matrix = _cosine_similarity_matrix(feature_matrix)

    visited = set()
    groups = []
    for i in range(len(valid_names)):
        if valid_names[i] in visited:
            continue
        group = [valid_names[i]]
        visited.add(valid_names[i])
        for j in range(i + 1, len(valid_names)):
            if sim_matrix[i, j] >= similarity_threshold:
                group.append(valid_names[j])
                visited.add(valid_names[j])
        groups.append(group)

    final_groups = [g for g in groups if len(g) > 1]

    rows = []
    group_id = 1
    # Map names to paths for copying
    name_to_path = {os.path.basename(p): p for p in valid_paths}

    for g in final_groups:
        group_folder = os.path.join(out_dir, f"group_{group_id:04d}")
        os.makedirs(group_folder, exist_ok=True)
        for name in g:
            src = name_to_path.get(name, os.path.join(image_dir, name))
            dst = os.path.join(group_folder, name)
            if copy_files:
                if os.path.exists(dst):
                    base, ext = os.path.splitext(name)
                    dst = os.path.join(group_folder, f"{base}_dup{ext}")
                shutil.copy2(src, dst)
            rows.append({
                "group_id": f"group_{group_id:04d}",
                "image_name": name,
                "src_path": src,
            })
        group_id += 1

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, csv_name)
    out_df.to_csv(out_csv, index=False)
    return out_df
