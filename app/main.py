from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from .schemas import RunRequest
from .config import Config
import json
import os
import shutil
from pathlib import Path
import pandas as pd
from .services.duplicate_exact import find_exact_duplicates
from .services.near_duplicates_runner import run_near_duplicates
from .services.llm_verify_runner import run as llm_verify_run
from .utils.groups_folder_to_csv import groups_folder_to_csv
from .utils.merge_duplicate_groups import merge_duplicate_folders
from .utils.extract_originals import extract_originals_to_folder
from .services.phase2_validation import run_phase2_validation
import threading
import uuid

app = FastAPI(title="Vision Data Image Cleaner")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount(
    "/ui/media",
    StaticFiles(directory=str(Path(__file__).parent / "templates" / "media")),
    name="ui-media",
)
LAST_RUN_RESULT = None
RUN_JOBS = {}


def _append_job_log(job_id: str, message: str):
    job = RUN_JOBS.get(job_id)
    if not job:
        return
    logs = job.setdefault("logs", [])
    logs.append(str(message))
    if len(logs) > 300:
        del logs[:-300]

@app.get("/")
def health():
    return {"status": "ok"} 


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: str | None) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _as_json_dict(value: str | None, default: dict) -> dict:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return default


@app.get("/ui", response_class=HTMLResponse)
def ui_home(request: Request):
    cfg = Config()
    return templates.TemplateResponse(
        request,
        "config_ui.html",
        {
            "cfg": cfg.data,
            "last_run": LAST_RUN_RESULT,
        },
    )


@app.get("/ui/docs", response_class=HTMLResponse)
def ui_docs(request: Request):
    return templates.TemplateResponse(request, "methods_docs.html", {})


@app.post("/ui/save")
async def ui_save_config(request: Request):
    form = await request.form()
    cfg = Config()

    cfg.set("input.image_folder", str(form.get("input_image_folder") or ""))
    cfg.set("output.base_folder", str(form.get("output_base_folder") or ""))

    cfg.set("similarity_check_llm.enabled", _as_bool(form.get("llm_enabled")))
    cfg.set("similarity_check_llm.provider", str(form.get("llm_provider") or "openai"))
    cfg.set("similarity_check_llm.model", str(form.get("llm_model") or "gpt-4o-mini"))
    cfg.set("similarity_check_llm.api_key_env", str(form.get("llm_api_key_env") or "OPENAI_API_KEY"))
    cfg.set("similarity_check_llm.temperature", _as_float(form.get("llm_temperature"), 0.0))

    active_method = str(form.get("near_active_method") or "hsv_cosine")
    cfg.set("near_duplicates.active_method", active_method)
    cfg.set("near_duplicates.methods.hsv_cosine.enabled", _as_bool(form.get("hsv_enabled")))
    cfg.set("near_duplicates.methods.hsv_cosine.similarity_threshold", _as_float(form.get("hsv_similarity_threshold"), 0.90))
    cfg.set("near_duplicates.methods.hsv_cosine.csv_name", str(form.get("hsv_csv_name") or "near_duplicate_groups_hsv_cosine.csv"))

    cfg.set("near_duplicates.methods.phash_embeddings.enabled", _as_bool(form.get("phash_enabled")))
    cfg.set("near_duplicates.methods.phash_embeddings.phash_max_distance", _as_int(form.get("phash_max_distance"), 20))
    cfg.set("near_duplicates.methods.phash_embeddings.embed_similarity_threshold", _as_float(form.get("phash_embed_similarity_threshold"), 0.75))
    cfg.set("near_duplicates.methods.phash_embeddings.csv_name", str(form.get("phash_csv_name") or "near_duplicate_groups.csv"))

    cfg.set("metadata.csv_path", str(form.get("metadata_csv_path") or ""))
    cfg.set("metadata.image_name_column", str(form.get("metadata_image_name_column") or "image_name"))
    cfg.set("metadata.datetime_column", str(form.get("metadata_datetime_column") or "hh_information-datetime"))

    cfg.set("filenames.exact_duplicates_csv", str(form.get("exact_duplicates_csv") or "exact_duplicates.csv"))
    cfg.set("filenames.near_duplicates_csv", str(form.get("near_duplicates_csv") or "near_duplicates.csv"))
    cfg.set("filenames.near_duplicates_llm_verified_csv", str(form.get("near_duplicates_llm_verified_csv") or "near_duplicates_llm_verified.csv"))
    cfg.set("filenames.exact_duplicate_groups_csv", str(form.get("exact_duplicate_groups_csv") or "exact_duplicate_groups.csv"))
    cfg.set("filenames.merged_groups_summary_csv", str(form.get("merged_groups_summary_csv") or "merged_groups_summary.csv"))
    cfg.set("filenames.originals_extraction_summary_csv", str(form.get("originals_extraction_summary_csv") or "originals_extraction_summary.csv"))
    cfg.set("filenames.unique_images_csv", str(form.get("unique_images_csv") or "unique_images.csv"))
    cfg.set("filenames.phase2_validation_report_csv", str(form.get("phase2_validation_report_csv") or "phase2_validation_report.csv"))

    cfg.set("phase2.enabled", _as_bool(form.get("phase2_enabled")))
    cfg.set("phase2.output_keys", _as_list(form.get("phase2_output_keys")))
    cfg.set("phase2.retry_max", _as_int(form.get("phase2_retry_max"), 3))
    cfg.set("phase2.throttle_seconds", _as_float(form.get("phase2_throttle_seconds"), 0.0))
    cfg.set("phase2.input_folder", str(form.get("phase2_input_folder") or ""))

    providers = _as_list(form.get("phase2_llm_providers"))
    if not providers:
        providers = [str(form.get("phase2_llm_provider") or "openai")]
    cfg.set("phase2.llm.providers", providers)
    cfg.set("phase2.llm.models", _as_json_dict(form.get("phase2_llm_models"), {}))
    cfg.set("phase2.llm.api_key_envs", _as_json_dict(form.get("phase2_llm_api_key_envs"), {}))
    cfg.set("phase2.llm.provider", providers[0])
    cfg.set("phase2.llm.model", str(form.get("phase2_llm_model") or "gpt-4o-mini"))
    cfg.set("phase2.llm.api_key_env", str(form.get("phase2_llm_api_key_env") or "OPENAI_API_KEY"))
    cfg.set("phase2.llm.temperature", _as_float(form.get("phase2_llm_temperature"), 0.0))
    cfg.set("phase2.llm.max_tokens", _as_int(form.get("phase2_llm_max_tokens"), 300))
    cfg.set("phase2.llm.system_prompt", str(form.get("phase2_system_prompt") or ""))
    cfg.set("phase2.llm.user_prompt", str(form.get("phase2_user_prompt") or ""))

    cfg.save()
    return RedirectResponse(url="/ui", status_code=303)


@app.post("/ui/run")
def ui_run_pipeline():
    global LAST_RUN_RESULT
    LAST_RUN_RESULT = run_pipeline(run_phase2=False)
    return RedirectResponse(url="/ui", status_code=303)


def _run_pipeline_job(job_id: str):
    try:
        RUN_JOBS[job_id]["status"] = "running"
        _append_job_log(job_id, "Phase 1 started")

        def log_fn(msg: str):
            _append_job_log(job_id, msg)

        result = run_pipeline(run_phase2=False, log_fn=log_fn)
        RUN_JOBS[job_id]["status"] = "completed"
        RUN_JOBS[job_id]["result"] = result
        _append_job_log(job_id, "Phase 1 completed")
    except Exception as exc:
        RUN_JOBS[job_id]["status"] = "failed"
        RUN_JOBS[job_id]["error"] = str(exc)
        _append_job_log(job_id, f"Phase 1 failed: {exc}")


def _run_phase2_job(job_id: str):
    try:
        RUN_JOBS[job_id]["status"] = "running"
        _append_job_log(job_id, "Phase 2 started")

        def log_fn(msg: str):
            _append_job_log(job_id, msg)

        result = run_phase2_only(log_fn=log_fn)
        RUN_JOBS[job_id]["status"] = "completed"
        RUN_JOBS[job_id]["result"] = result
        _append_job_log(job_id, "Phase 2 completed")
    except Exception as exc:
        RUN_JOBS[job_id]["status"] = "failed"
        RUN_JOBS[job_id]["error"] = str(exc)
        _append_job_log(job_id, f"Phase 2 failed: {exc}")


@app.post("/ui/run/start")
def ui_run_start():
    job_id = str(uuid.uuid4())
    RUN_JOBS[job_id] = {"status": "queued", "result": None, "error": None, "phase": "phase1", "logs": []}
    thread = threading.Thread(target=_run_pipeline_job, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/ui/run/phase2/start")
def ui_run_phase2_start():
    job_id = str(uuid.uuid4())
    RUN_JOBS[job_id] = {"status": "queued", "result": None, "error": None, "phase": "phase2", "logs": []}
    thread = threading.Thread(target=_run_phase2_job, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/ui/run/status/{job_id}")
def ui_run_status(job_id: str):
    job = RUN_JOBS.get(job_id)
    if not job:
        return {"status": "not_found", "result": None, "error": "job not found"}
    return {
        "status": job.get("status"),
        "result": job.get("result"),
        "error": job.get("error"),
        "phase": job.get("phase"),
        "logs": job.get("logs", []),
    }


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
def run_pipeline(run_phase2: bool = False, log_fn=None):
    global LAST_RUN_RESULT
    # Load defaults from YAML config
    def _log(message: str):
        print(message)
        if callable(log_fn):
            log_fn(message)

    _log("Loading configuration...")
    cfg = Config()

    # Resolve inputs with request overrides where available
    image_dir = cfg.get("input.image_folder")
    output_dir = cfg.get("output.base_folder")
    exact_duplicates_csv_name = cfg.get("filenames.exact_duplicates_csv", "exact_duplicates.csv")
    near_duplicates_csv_name = cfg.get("filenames.near_duplicates_csv", "near_duplicates.csv")
    verified_csv_name = cfg.get("filenames.near_duplicates_llm_verified_csv", "near_duplicates_llm_verified.csv")
    exact_duplicate_groups_csv_name = cfg.get("filenames.exact_duplicate_groups_csv", "exact_duplicate_groups.csv")
    merged_groups_summary_csv_name = cfg.get("filenames.merged_groups_summary_csv", "merged_groups_summary.csv")
    originals_extraction_summary_csv_name = cfg.get("filenames.originals_extraction_summary_csv", "originals_extraction_summary.csv")
    unique_images_csv_name = cfg.get("filenames.unique_images_csv", "unique_images.csv")
    _log(f"Running pipeline with image_dir={image_dir}, output_dir={output_dir}")

    # Run exact duplicate detection (pHash-based)
    exact_dups_df, df = find_exact_duplicates(
        image_dir=image_dir,
        base_output_dir=output_dir,
        copy_files=True,
        csv_name=exact_duplicates_csv_name,
    )

    all_input_names = set(df["image_name"].astype(str).tolist()) if not df.empty else set()

    duplicate_mask = df["phash"].duplicated(keep="first")  
    removed_df = df[duplicate_mask].copy()
    kept_df = df[~duplicate_mask].copy()

    _log("Running near-duplicate clustering...")

    
    # Near-duplicate clustering based on active method in config
    near_counts = run_near_duplicates(kept_df=kept_df, image_dir=image_dir, output_dir=output_dir, cfg=cfg)

    groups_folder = os.path.join(output_dir, cfg.get("near_duplicates.active_method"))
    near_csv = os.path.join(output_dir, near_duplicates_csv_name)
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
            _log(f"Near-duplicate CSV conversion skipped: {exc}")

    verified_csv = None
    confirmed = set()
    rejected = set()
    failed = set()
    # Run LLM verification only if enabled in config
    llm_cfg = cfg.get("similarity_check_llm") or cfg.get("llm") or {}
    if llm_cfg.get("enabled"):
        verified_csv = os.path.join(output_dir, verified_csv_name)
        if near_csv_ready and os.path.exists(near_csv):
            try:
                provider = llm_cfg.get("provider")
                _log(f"Running LLM verification on {near_csv} (provider={provider})")
                confirmed, rejected, failed, verified_csv = llm_verify_run(
                    csv_path=near_csv,
                    image_dir=groups_folder,
                    provider=provider,
                    out_csv=verified_csv,
                )
                _log(f"LLM verification completed: {len(confirmed)} confirmed duplicates, {len(rejected)} rejected, {len(failed)} failed checks")
            
                
  
            except Exception as e:
                _log(f"LLM verification failed: {e}")
                verified_csv = None
        else:
            _log("Near-duplicates CSV not generated in this run; skipping LLM verification")
    else:
        _log("LLM verification disabled in config; skipping")

    exact_duplicate_names = set(exact_dups_df["image_name"].astype(str).tolist()) if not exact_dups_df.empty else set()
    near_llm_confirmed_names = {
        _normalize_candidate_name(name, all_input_names)
        for name in confirmed
    }
    near_llm_confirmed_names = {name for name in near_llm_confirmed_names if name in all_input_names}
    final_duplicate_names = exact_duplicate_names | near_llm_confirmed_names
    unique_image_names = sorted(all_input_names - final_duplicate_names)

    exact_groups_csv = os.path.join(output_dir, exact_duplicate_groups_csv_name)
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
            _log(f"Could not parse verified CSV for group copy: {exc}")

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
    merged_groups_summary_csv = os.path.join(output_dir, merged_groups_summary_csv_name)
    merged_duplicate_groups = 0
    try:
        merged_summary_df = merge_duplicate_folders(
            duplicates_root=Path(exact_duplicates_root_dir),
            merged_root=Path(merged_duplicates_dir),
            summary_csv_path=Path(merged_groups_summary_csv),
        )
        merged_duplicate_groups = int(len(merged_summary_df))
        _log(
            f"Merged duplicate groups created: {merged_duplicate_groups} "
            f"(output={merged_duplicates_dir})"
        )
    except (FileNotFoundError, ValueError) as exc:
        _log(f"Skipping merged duplicate group build: {exc}")
    except Exception as exc:
        _log(f"Merged duplicate group build failed: {exc}")

    originals_dir = os.path.join(output_dir, "duplicates_original")
    originals_extraction_summary_csv = None
    originals_extracted_count = 0
    try:
        metadata_csv_path = cfg.get("metadata.csv_path")  # Optional: path to metadata CSV
        image_name_col = cfg.get("metadata.image_name_column") or "image_name"
        datetime_col = cfg.get("metadata.datetime_column") or "hh_information-datetime"
        
        originals_df, copied_count = extract_originals_to_folder(
            merged_groups_root=Path(merged_duplicates_dir),
            originals_folder=Path(originals_dir),
            metadata_csv=metadata_csv_path,
            image_name_column=image_name_col,
            datetime_column=datetime_col,
        )
        originals_extracted_count = int(copied_count)
        originals_extraction_summary_csv = os.path.join(output_dir, originals_extraction_summary_csv_name)
        originals_df.to_csv(originals_extraction_summary_csv, index=False)
        _log(f"Extracted {copied_count} original images to: {originals_dir}")
    except (FileNotFoundError, ValueError) as exc:
        _log(f"Skipping original extraction: {exc}")
    except Exception as exc:
        _log(f"Original extraction failed: {exc}")

    unique_csv = os.path.join(output_dir, unique_images_csv_name)
    unique_df = pd.DataFrame({"image_name": unique_image_names})
    unique_df.to_csv(unique_csv, index=False)

    unique_images_dir = os.path.join(output_dir, "unique_images")
    os.makedirs(unique_images_dir, exist_ok=True)
    for name in unique_image_names:
        src = os.path.join(image_dir, name)
        dst = os.path.join(unique_images_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    phase2_result = None
    if run_phase2:
        try:
            phase2_result = run_phase2_only()
        except Exception as exc:
            phase2_result = {
                "enabled": True,
                "error": str(exc),
            }
            _log(f"Phase 2 failed after Phase 1: {exc}")

    _log(
        f"Total input images: {len(all_input_names)} | "
        f"Exact duplicate images: {len(exact_duplicate_names)} | "
        f"Near-duplicate confirmed by LLM: {len(near_llm_confirmed_names)} | "
        f"Final duplicate images: {len(final_duplicate_names)} | "
        f"Unique images: {len(unique_image_names)}"
    )
    # Return a simple summary
    result = {
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
            "original_images_extracted": int(originals_extracted_count),
        },
        "near_duplicates": near_counts,
        "output_dir": output_dir,
        "verified_csv": verified_csv,
        "exact_duplicate_groups_csv": exact_groups_csv,
        "merged_duplicate_groups_dir": merged_duplicates_dir,
        "merged_duplicate_groups_summary_csv": merged_groups_summary_csv,
        "originals_dir": originals_dir,
        "originals_extraction_summary_csv": originals_extraction_summary_csv,
        "unique_images_csv": unique_csv,
        "unique_images_dir": unique_images_dir,
        "llm_confirmed_groups_dir": exact_duplicates_root_dir,
        "phase2": phase2_result,
    }
    LAST_RUN_RESULT = result
    return result


@app.post("/run/phase2")
def run_phase2_only(log_fn=None):
    global LAST_RUN_RESULT
    cfg = Config()

    output_dir = cfg.get("output.base_folder")
    unique_images_csv_name = cfg.get("filenames.unique_images_csv", "unique_images.csv")
    unique_csv = os.path.join(output_dir, unique_images_csv_name)

    configured_phase2_folder = str(cfg.get("phase2.input_folder") or "").strip()
    unique_images_dir = configured_phase2_folder or os.path.join(output_dir, "unique_images")

    if callable(log_fn):
        log_fn(f"Phase 2 input folder: {unique_images_dir}")

    phase2_result = run_phase2_validation(
        cfg=cfg,
        image_dir=unique_images_dir,
        output_dir=output_dir,
        unique_images_csv_path=unique_csv,
        unique_images_dir_path=unique_images_dir,
        log_fn=log_fn,
    )

    result = {
        "message": "Phase 2 completed",
        "phase2_input_folder": unique_images_dir,
        "output_dir": output_dir,
        "phase2": phase2_result,
    }
    LAST_RUN_RESULT = result
    return result