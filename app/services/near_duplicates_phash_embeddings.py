import os
import shutil
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import imagehash
import torch
import torchvision.transforms as T
from torchvision import models


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


# ================================
# Model Init
# ================================
def init_embedding_model(model_name: str = "resnet50", device: str = "auto"):

    dev = (
        "cuda" if device == "cuda"
        else "cpu" if device == "cpu"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if model_name.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.fc = torch.nn.Identity()
    model.eval().to(dev)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, transform, dev


def get_embedding(img_path, model, transform, device):
    try:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(x).cpu().numpy().flatten()

        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb
    except Exception:
        return None


def get_phash(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        return imagehash.phash(img)
    except Exception:
        return None


# ================================
# Union Find
# ================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ================================
# MAIN FUNCTION
# ================================
def find_near_duplicates_phash_embeddings(
    df: pd.DataFrame,
    image_dir: str,
    base_output_dir: str,
    phash_max_distance: int = 20,
    embed_similarity_threshold: float = 0.75,
    model_name: str = "resnet50",
    device: str = "auto",
    output_subfolder: str = "near_duplicates_phash_embeddings",
    csv_name: str = "near_duplicate_groups.csv",
    copy_files: bool = True,
) -> pd.DataFrame:

    print("\n🔍 Near-duplicate detection (pHash + embeddings)")

    image_dir = Path(image_dir)
    out_dir = Path(base_output_dir) / output_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df[df["image_name"].notna()].copy()
    image_names = df["image_name"].astype(str).tolist()
    image_paths = [image_dir / n for n in image_names]

    print(f"Images from df: {len(image_names)}")

    # Init model
    model, transform, dev = init_embedding_model(model_name=model_name, device=device)

    valid_names, valid_paths, embeddings, phashes = [], [], [], []

    print("Extracting embeddings + phash...")

    for name, path in tqdm(list(zip(image_names, image_paths))):

        if not path.exists():
            continue

        emb = get_embedding(str(path), model, transform, dev)
        ph = get_phash(str(path))

        if emb is None or ph is None:
            continue

        valid_names.append(name)
        valid_paths.append(path)
        embeddings.append(emb)
        phashes.append(ph)

    n = len(valid_names)
    print("Valid images:", n)

    if n < 2:
        return pd.DataFrame()

    embeddings = np.array(embeddings)

    # ================================
    # Pairwise compare
    # ================================
    uf = UnionFind(n)

    print("Pairwise comparisons...")

    for i in tqdm(range(n)):
        for j in range(i + 1, n):

            ph_dist = phashes[i] - phashes[j]
            if ph_dist > phash_max_distance:
                continue

            emb_sim = float(np.dot(embeddings[i], embeddings[j]))

            if emb_sim >= embed_similarity_threshold:
                uf.union(i, j)

    # ================================
    # Build groups
    # ================================
    groups = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    final_groups = [g for g in groups.values() if len(g) > 1]
    print("Near-duplicate groups found:", len(final_groups))

    # ================================
    # Save groups
    # ================================
    rows = []
    gid = 1

    for g in final_groups:
        group_name = f"group_{gid:04d}"
        group_folder = out_dir / group_name
        group_folder.mkdir(exist_ok=True)

        for idx in g:
            src = valid_paths[idx]
            dst = group_folder / valid_names[idx]

            if copy_files:
                if dst.exists():
                    base, ext = os.path.splitext(valid_names[idx])
                    dst = group_folder / f"{base}_dup{ext}"
                shutil.copy2(src, dst)

            rows.append({
                "group_id": group_name,
                "image_name": valid_names[idx],
                "src_path": str(src)
            })

        gid += 1

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / csv_name, index=False)

    print("CSV saved to:", out_dir / csv_name)

    return out_df
