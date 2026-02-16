from fastapi import FastAPI, BackgroundTasks
from .schemas import RunRequest
from .config import Config
import os
from .services.duplicate_exact import find_exact_duplicates
from .utils.dedup import split_duplicates_by_phash
from .services.near_duplicates_runner import run_near_duplicates
from .utils.dedup import split_duplicates_by_phash

app = FastAPI(title="Image Cleaner Pipeline")

@app.get("/")
def health():
    return {"status": "ok"}

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
    exact_dups_df,df = find_exact_duplicates(
        image_dir=image_dir,
        base_output_dir=output_dir,
        copy_files=True,
        csv_name="exact_duplicates.csv",
    )
    duplicate_mask = df["phash"].duplicated(keep="first")  
    removed_df = df[duplicate_mask].copy()
    kept_df = df[~duplicate_mask].copy()

    print(f"Kept {len(kept_df)} unique images after exact deduplication")
    
    # Near-duplicate clustering based on active method in config
    near_counts = run_near_duplicates(kept_df=kept_df, image_dir=image_dir, output_dir=output_dir, cfg=cfg)
    # Return a simple summary
    return {
        "message": "Exact duplicate detection completed",
        "counts": {
            "duplicates": int(len(kept_df)) if kept_df is not None else 0,
            "removed": int(len(removed_df)) if removed_df is not None else 0,
        },
        "near_duplicates": near_counts,
        "output_dir": output_dir,
    }