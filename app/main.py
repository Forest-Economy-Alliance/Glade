from fastapi import FastAPI, BackgroundTasks
from .schemas import RunRequest
from .config import Config
import os
from .services.duplicate_exact import find_exact_duplicates
from .utils.dedup import split_duplicates_by_phash
from .services.near_duplicates_phash_embeddings import find_near_duplicates_phash_embeddings
from .services.near_duplicates_hsv_cosine import find_near_duplicates_hsv_cosine
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
    near_method = cfg.get("near_duplicates.active_method")
    near_counts = {"groups": 0, "method": near_method, "csv": None}
    try:
        if near_method == "phash_embeddings" and cfg.get("near_duplicates.methods.phash_embeddings.enabled"):
            phash_max_distance = cfg.get("near_duplicates.methods.phash_embeddings.phash_max_distance")
            embed_sim_threshold = cfg.get("near_duplicates.methods.phash_embeddings.embed_similarity_threshold")
            device = cfg.get("near_duplicates.methods.phash_embeddings.device")
            out_sub = cfg.get("near_duplicates.methods.phash_embeddings.output_subfolder")
            csv_name = cfg.get("near_duplicates.methods.phash_embeddings.csv_name")

            near_df = find_near_duplicates_phash_embeddings(
                df=kept_df,
                image_dir=image_dir,
                base_output_dir=output_dir,
                phash_max_distance=phash_max_distance,
                embed_similarity_threshold=embed_sim_threshold,
                model_name="resnet50",
                device=device,
                output_subfolder=out_sub,
                csv_name=csv_name,
                copy_files=True,
            )
            near_counts["groups"] = int(near_df["group_id"].nunique()) if not near_df.empty else 0
            near_counts["csv"] = os.path.join(output_dir, out_sub, csv_name)

        elif near_method == "hsv_cosine" and cfg.get("near_duplicates.methods.hsv_cosine.enabled"):
            image_size = tuple(cfg.get("near_duplicates.methods.hsv_cosine.image_size"))
            hist_bins = tuple(cfg.get("near_duplicates.methods.hsv_cosine.hist_bins"))
            ranges = tuple(cfg.get("near_duplicates.methods.hsv_cosine.ranges"))
            sim_threshold = cfg.get("near_duplicates.methods.hsv_cosine.similarity_threshold")
            out_sub = cfg.get("near_duplicates.methods.hsv_cosine.output_subfolder")
            csv_name = cfg.get("near_duplicates.methods.hsv_cosine.csv_name")

            near_df = find_near_duplicates_hsv_cosine(
                df=kept_df,
                image_dir=image_dir,
                base_output_dir=output_dir,
                image_size=image_size,
                hist_bins=hist_bins,
                ranges=ranges,
                similarity_threshold=sim_threshold,
                output_subfolder=out_sub,
                csv_name=csv_name,
                copy_files=True,
            )
            near_counts["groups"] = int(near_df["group_id"].nunique()) if not near_df.empty else 0
            near_counts["csv"] = os.path.join(output_dir, out_sub, csv_name)
    except ImportError as e:
        near_counts["error"] = str(e)
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