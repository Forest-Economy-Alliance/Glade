from fastapi import FastAPI, BackgroundTasks
from .schemas import RunRequest
from .config import Config
import os
from .services.duplicate_exact import find_exact_duplicates
from .utils.dedup import split_duplicates_by_phash
from .services.near_duplicates_runner import run_near_duplicates
from .utils.dedup import split_duplicates_by_phash
from .services.llm_verify_runner import run as llm_verify_run
from .utils.groups_folder_to_csv import groups_folder_to_csv

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

    
    # Near-duplicate clustering based on active method in config
    near_counts = run_near_duplicates(kept_df=kept_df, image_dir=image_dir, output_dir=output_dir, cfg=cfg)
    
    groups_folder = os.path.join(output_dir,cfg.get("near_duplicates.active_method"))
    near_csv = os.path.join(output_dir, "near_duplicates.csv")
    df = groups_folder_to_csv(
    groups_folder=groups_folder,
    out_csv=near_csv
    )

    verified_csv = None
    # Run LLM verification only if enabled in config
    llm_cfg = cfg.get("similarity_check_llm") or cfg.get("llm") or {}
    if llm_cfg.get("enabled"):
        verified_csv = os.path.join(output_dir, "near_duplicates_llm_verified.csv")
        if os.path.exists(near_csv):
            try:
                provider = llm_cfg.get("provider")
                print(f"Running LLM verification on {near_csv} (provider={provider})")
                llm_verify_run(
                    csv_path=near_csv,
                    image_dir=groups_folder,
                    provider=provider,
                    out_csv=verified_csv,
                )
            except Exception as e:
                print(f"LLM verification failed: {e}")
                verified_csv = None
        else:
            print(f"Near-duplicates CSV not found: {near_csv}; skipping LLM verification")
    else:
        print("LLM verification disabled in config; skipping")
    # Return a simple summary
    return {
        "message": "Pipeline completed",
        "counts": {
            "duplicates_removed": int(len(removed_df)),
            "images_remaining": int(len(kept_df)),
        },
        "near_duplicates": near_counts,
        "output_dir": output_dir,
        "verified_csv": verified_csv
    }