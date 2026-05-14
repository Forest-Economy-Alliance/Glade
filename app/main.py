from fastapi import FastAPI, BackgroundTasks
from .schemas import RunRequest
from .config import Config
import os
import shutil
from pathlib import Path
import pandas as pd
from .services.duplicate_exact import find_exact_duplicates
from .services.near_duplicates_runner import run_near_duplicates
from .services.llm_verify_runner import run as llm_verify_run
from .utils.groups_folder_to_csv import groups_folder_to_csv
from .utils.merge_duplicate_groups import merge_duplicate_folders

app = FastAPI(title="Image Cleaner Pipeline")

@app.get("/")
def health():
    return {"status": "ok"} 


def _normalize_candidate_name(name: str, input_names: set[str]) -> str:
    """Map generated near-duplicate filenames (e.g. *_dup.jpg) back to input names when possible."""
    if name in input_names:
        return name
    stem, ext = os.path.splitext(name)
    if stem.endswith("_dup"):
        candidate = f"{stem[:-4]}{ext}"
        if candidate in input_names:
            return candidate
    return name

@app.post("/run")
def run_pipeline():
    # Load defaults from YAML config
    print("Loading configuration...")
    cfg = Config()

    # Resolve inputs with request overrides where available
    image_dir = cfg.get("input.image_folder")
    output_dir = cfg.get("output.base_folder")
    print(f"Running pipeline with image_dir={image_dir}, output_dir={output_dir}")

    # Run exact duplicate detection (pHash-based)
    exact_dups_df, df = find_exact_duplicates(
        image_dir=image_dir,
        base_output_dir=output_dir,
        copy_files=True,
        csv_name="exact_duplicates.csv",
    )

    all_input_names = set(df["image_name"].astype(str).tolist()) if not df.empty else set()

    duplicate_mask = df["phash"].duplicated(keep="first")  
    removed_df = df[duplicate_mask].copy()
    kept_df = df[~duplicate_mask].copy()

    print("")

    
    # Near-duplicate clustering based on active method in config
    near_counts = run_near_duplicates(kept_df=kept_df, image_dir=image_dir, output_dir=output_dir, cfg=cfg)

    groups_folder = os.path.join(output_dir, cfg.get("near_duplicates.active_method"))
    near_csv = os.path.join(output_dir, "near_duplicates.csv")
    near_groups_df = pd.DataFrame(columns=["group_id", "image_name"])
    near_csv_ready = False

    if near_counts.get("error"):
        print(f"Near-duplicate step failed: {near_counts['error']}")
    elif near_counts.get("groups", 0) <= 0:
        print("No near-duplicate groups found; skipping CSV conversion")
    elif not os.path.exists(groups_folder):
        print(f"Near-duplicate output folder not found: {groups_folder}; skipping CSV conversion")
    else:
        try:
            near_groups_df = groups_folder_to_csv(
                groups_folder=groups_folder,
                out_csv=near_csv
            )
            near_csv_ready = True
        except (FileNotFoundError, ValueError) as exc:
            print(f"Near-duplicate CSV conversion skipped: {exc}")

    verified_csv = None
    confirmed = set()
    rejected = set()
    failed = set()
    # Run LLM verification only if enabled in config
    llm_cfg = cfg.get("similarity_check_llm") or cfg.get("llm") or {}
    if llm_cfg.get("enabled"):
        verified_csv = os.path.join(output_dir, "near_duplicates_llm_verified.csv")
        if near_csv_ready and os.path.exists(near_csv):
            try:
                provider = llm_cfg.get("provider")
                print(f"Running LLM verification on {near_csv} (provider={provider})")
                confirmed, rejected, failed, verified_csv = llm_verify_run(
                    csv_path=near_csv,
                    image_dir=groups_folder,
                    provider=provider,
                    out_csv=verified_csv,
                )
                print(f"LLM verification completed: {len(confirmed)} confirmed duplicates, {len(rejected)} rejected, {len(failed)} failed checks")
            
                
  
            except Exception as e:
                print(f"LLM verification failed: {e}")
                verified_csv = None
        else:
            print("Near-duplicates CSV not generated in this run; skipping LLM verification")
    else:
        print("LLM verification disabled in config; skipping")

    exact_duplicate_names = set(exact_dups_df["image_name"].astype(str).tolist()) if not exact_dups_df.empty else set()
    near_llm_confirmed_names = {
        _normalize_candidate_name(name, all_input_names)
        for name in confirmed
    }
    near_llm_confirmed_names = {name for name in near_llm_confirmed_names if name in all_input_names}
    final_duplicate_names = exact_duplicate_names | near_llm_confirmed_names
    unique_image_names = sorted(all_input_names - final_duplicate_names)

    exact_groups_csv = os.path.join(output_dir, "exact_duplicate_groups.csv")
    exact_duplicates_root_dir = os.path.join(output_dir, "exact_duplicates")
    os.makedirs(exact_duplicates_root_dir, exist_ok=True)

    llm_confirmed_group_ids = set()
    if verified_csv and os.path.exists(verified_csv):
        try:
            verified_df = pd.read_csv(verified_csv)
            if {"group_id", "is_duplicate_group"}.issubset(verified_df.columns):
                llm_confirmed_group_ids = {
                    str(gid)
                    for gid in verified_df.loc[
                        verified_df["is_duplicate_group"] == True, "group_id"
                    ].astype(str).tolist()
                }
        except Exception as exc:
            print(f"Could not parse verified CSV for group copy: {exc}")

    if not llm_confirmed_group_ids and confirmed and not near_groups_df.empty:
        normalized_confirmed = {
            _normalize_candidate_name(name, all_input_names)
            for name in confirmed
        }
        normalized_confirmed = {
            name for name in normalized_confirmed if name in all_input_names
        }
        fallback_group_ids = set()
        for _, row in near_groups_df.iterrows():
            raw_name = str(row["image_name"])
            normalized_name = _normalize_candidate_name(raw_name, all_input_names)
            if normalized_name in normalized_confirmed:
                fallback_group_ids.add(str(row["group_id"]))
        llm_confirmed_group_ids = fallback_group_ids

    if llm_confirmed_group_ids and os.path.exists(groups_folder):
        for gid in sorted(llm_confirmed_group_ids):
            src_group_dir = os.path.join(groups_folder, gid)
            dst_group_dir = os.path.join(exact_duplicates_root_dir, f"llm_{gid}")
            if not os.path.isdir(src_group_dir):
                continue
            if os.path.exists(dst_group_dir):
                shutil.rmtree(dst_group_dir)
            shutil.copytree(src_group_dir, dst_group_dir)

    if not exact_dups_df.empty:
        exact_groups_df = exact_dups_df[["group_folder", "image_name"]].rename(
            columns={"group_folder": "group_id"}
        )
    else:
        exact_groups_df = pd.DataFrame(columns=["group_id", "image_name"])

    if llm_confirmed_group_ids and not near_groups_df.empty:
        near_confirmed_df = near_groups_df[
            near_groups_df["group_id"].astype(str).isin(llm_confirmed_group_ids)
        ].copy()
        if not near_confirmed_df.empty:
            near_confirmed_df["image_name"] = near_confirmed_df["image_name"].astype(str).map(
                lambda name: _normalize_candidate_name(name, all_input_names)
            )
            near_confirmed_df = near_confirmed_df[
                near_confirmed_df["image_name"].isin(all_input_names)
            ]
            if not near_confirmed_df.empty:
                near_confirmed_df["group_id"] = "llm_" + near_confirmed_df["group_id"].astype(str)
                exact_groups_df = pd.concat(
                    [exact_groups_df, near_confirmed_df[["group_id", "image_name"]]],
                    ignore_index=True,
                )

    if not exact_groups_df.empty:
        exact_groups_df = exact_groups_df.drop_duplicates(subset=["group_id", "image_name"])
    exact_groups_df.to_csv(exact_groups_csv, index=False)

    merged_duplicates_dir = os.path.join(output_dir, "exact_duplicates_merged")
    merged_groups_summary_csv = os.path.join(output_dir, "merged_groups_summary.csv")
    merged_duplicate_groups = 0
    try:
        merged_summary_df = merge_duplicate_folders(
            duplicates_root=Path(exact_duplicates_root_dir),
            merged_root=Path(merged_duplicates_dir),
            summary_csv_path=Path(merged_groups_summary_csv),
        )
        merged_duplicate_groups = int(len(merged_summary_df))
        print(
            f"Merged duplicate groups created: {merged_duplicate_groups} "
            f"(output={merged_duplicates_dir})"
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Skipping merged duplicate group build: {exc}")
    except Exception as exc:
        print(f"Merged duplicate group build failed: {exc}")

    unique_csv = os.path.join(output_dir, "unique_images.csv")
    unique_df = pd.DataFrame({"image_name": unique_image_names})
    unique_df.to_csv(unique_csv, index=False)

    unique_images_dir = os.path.join(output_dir, "unique_images")
    os.makedirs(unique_images_dir, exist_ok=True)
    for name in unique_image_names:
        src = os.path.join(image_dir, name)
        dst = os.path.join(unique_images_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    print(
        f"Total input images: {len(all_input_names)} | "
        f"Exact duplicate images: {len(exact_duplicate_names)} | "
        f"Near-duplicate confirmed by LLM: {len(near_llm_confirmed_names)} | "
        f"Final duplicate images: {len(final_duplicate_names)} | "
        f"Unique images: {len(unique_image_names)}"
    )
    # Return a simple summary
    return {
        "message": "Pipeline completed",
        "counts": {
            "duplicates_removed": int(len(removed_df)),
            "images_remaining": int(len(kept_df)),
            "all_input_images": int(len(all_input_names)),
            "exact_duplicate_images": int(len(exact_duplicate_names)),
            "near_duplicate_images_confirmed": int(len(near_llm_confirmed_names)),
            "final_duplicate_images": int(len(final_duplicate_names)),
            "unique_images": int(len(unique_image_names)),
            "merged_duplicate_groups": int(merged_duplicate_groups),
        },
        "near_duplicates": near_counts,
        "output_dir": output_dir,
        "verified_csv": verified_csv,
        "exact_duplicate_groups_csv": exact_groups_csv,
        "merged_duplicate_groups_dir": merged_duplicates_dir,
        "merged_duplicate_groups_summary_csv": merged_groups_summary_csv,
        "unique_images_csv": unique_csv,
        "unique_images_dir": unique_images_dir,
        "llm_confirmed_groups_dir": exact_duplicates_root_dir,
    }