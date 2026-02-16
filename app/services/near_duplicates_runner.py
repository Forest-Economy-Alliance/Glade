import os
from typing import Dict
import pandas as pd

from .near_duplicates_phash_embeddings import find_near_duplicates_phash_embeddings
from .near_duplicates_hsv_cosine import find_near_duplicates_hsv_cosine


def run_near_duplicates(
    kept_df: pd.DataFrame,
    image_dir: str,
    output_dir: str,
    cfg,
) -> Dict[str, object]:
    """
    Run the active near-duplicate method as configured and return summary info.

    Returns a dict: {"groups": int, "method": str, "csv": str | None, "error": str | None}
    """
    near_method = cfg.get("near_duplicates.active_method")
    result = {"groups": 0, "method": near_method, "csv": None}

    try:
        if near_method == "phash_embeddings" and cfg.get("near_duplicates.methods.phash_embeddings.enabled"):
            phash_max_distance = cfg.get("near_duplicates.methods.phash_embeddings.phash_max_distance")
            embed_sim_threshold = cfg.get("near_duplicates.methods.phash_embeddings.embed_similarity_threshold")
            model_name = cfg.get("near_duplicates.methods.phash_embeddings.model")
            device = cfg.get("near_duplicates.methods.phash_embeddings.device")
            out_sub = cfg.get("near_duplicates.methods.phash_embeddings.output_subfolder")
            csv_name = cfg.get("near_duplicates.methods.phash_embeddings.csv_name")

            near_df = find_near_duplicates_phash_embeddings(
                df=kept_df,
                image_dir=image_dir,
                base_output_dir=output_dir,
                phash_max_distance=phash_max_distance,
                embed_similarity_threshold=embed_sim_threshold,
                model_name=model_name,
                device=device,
                output_subfolder=out_sub,
                csv_name=csv_name,
                copy_files=True,
            )
            result["groups"] = int(near_df["group_id"].nunique()) if not near_df.empty else 0
            result["csv"] = os.path.join(output_dir, out_sub, csv_name)

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
            result["groups"] = int(near_df["group_id"].nunique()) if not near_df.empty else 0
            result["csv"] = os.path.join(output_dir, out_sub, csv_name)

        else:
            # Method disabled or unknown
            result["error"] = f"Method '{near_method}' disabled or not recognized"

    except ImportError as e:
        result["error"] = str(e)

    return result
