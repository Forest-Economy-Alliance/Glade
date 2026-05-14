import argparse
import os
import shutil
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
    ".avif",
    ".gif",
}


def _list_group_folders(duplicates_root: Path) -> list[Path]:
    if not duplicates_root.exists() or not duplicates_root.is_dir():
        raise FileNotFoundError(f"duplicates root not found: {duplicates_root}")
    return sorted([p for p in duplicates_root.iterdir() if p.is_dir()])


def _build_membership(group_folders: list[Path]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    folder_to_images: dict[str, set[str]] = {}
    image_to_folders: dict[str, set[str]] = defaultdict(set)

    for folder in group_folders:
        folder_name = folder.name
        images: set[str] = set()
        for root, _, files in os.walk(folder):
            for file_name in files:
                if Path(file_name).suffix.lower() in IMAGE_EXTS:
                    images.add(file_name)
                    image_to_folders[file_name].add(folder_name)
        folder_to_images[folder_name] = images

    return folder_to_images, image_to_folders


def _build_folder_graph(
    folder_to_images: dict[str, set[str]],
    image_to_folders: dict[str, set[str]],
) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {folder: set() for folder in folder_to_images}
    for folders in image_to_folders.values():
        if len(folders) <= 1:
            continue
        items = sorted(folders)
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    return graph


def _connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    visited: set[str] = set()
    components: list[set[str]] = []

    for node in sorted(graph):
        if node in visited:
            continue
        comp: set[str] = set()
        queue: deque[str] = deque([node])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            for nb in sorted(graph[cur]):
                if nb not in visited:
                    queue.append(nb)
        components.append(comp)

    return components


def _copy_cluster(
    cluster_folders: set[str],
    duplicates_root: Path,
    merged_dir: Path,
) -> tuple[int, int]:
    merged_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    copied_count = 0
    skipped_count = 0

    for folder_name in sorted(cluster_folders):
        src_folder = duplicates_root / folder_name
        if not src_folder.is_dir():
            continue
        for root, _, files in os.walk(src_folder):
            rel = os.path.relpath(root, src_folder)
            target_root = merged_dir / rel
            target_root.mkdir(parents=True, exist_ok=True)

            for file_name in files:
                if Path(file_name).suffix.lower() not in IMAGE_EXTS:
                    continue
                if file_name in seen_names:
                    skipped_count += 1
                    continue
                src_file = Path(root) / file_name
                dst_file = target_root / file_name
                shutil.copy2(src_file, dst_file)
                seen_names.add(file_name)
                copied_count += 1

    return copied_count, skipped_count


def merge_duplicate_folders(
    duplicates_root: Path,
    merged_root: Path,
    summary_csv_path: Path | None = None,
) -> pd.DataFrame:
    group_folders = _list_group_folders(duplicates_root)
    if not group_folders:
        raise ValueError(f"no folders found under: {duplicates_root}")

    folder_to_images, image_to_folders = _build_membership(group_folders)
    graph = _build_folder_graph(folder_to_images, image_to_folders)
    components = _connected_components(graph)

    merged_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for idx, comp in enumerate(components, start=1):
        merged_name = f"merged_group_{idx:04d}"
        merged_dir = merged_root / merged_name

        if merged_dir.exists():
            shutil.rmtree(merged_dir)

        copied_count, skipped_count = _copy_cluster(comp, duplicates_root, merged_dir)
        summary_rows.append(
            {
                "merged_group": merged_name,
                "source_folders": "|".join(sorted(comp)),
                "source_folder_count": len(comp),
                "copied_unique_images": copied_count,
                "skipped_duplicate_names": skipped_count,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = summary_csv_path if summary_csv_path else (merged_root / "merged_groups_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge duplicate group folders by shared image names."
    )
    parser.add_argument(
        "--duplicates-root",
        required=True,
        help="Path to the exact_duplicates folder containing phash_* and llm_* folders.",
    )
    parser.add_argument(
        "--merged-root",
        required=False,
        help="Output folder for merged groups. Default: <duplicates_root>/merged_groups",
    )
    args = parser.parse_args()

    duplicates_root = Path(args.duplicates_root).resolve()
    merged_root = Path(args.merged_root).resolve() if args.merged_root else duplicates_root / "merged_groups"

    summary_csv_path = duplicates_root.parent / "merged_groups_summary.csv"
    summary_df = merge_duplicate_folders(
        duplicates_root=duplicates_root,
        merged_root=merged_root,
        summary_csv_path=summary_csv_path,
    )
    print(f"Merged groups written to: {merged_root}")
    print(f"Merged groups summary CSV: {summary_csv_path}")
    print(f"Total merged groups: {len(summary_df)}")
    print(f"Total copied unique images: {int(summary_df['copied_unique_images'].sum())}")


if __name__ == "__main__":
    main()
