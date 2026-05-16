from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _read_unique_images(unique_images_csv: Path, unique_images_dir: Path) -> list[str]:
    if unique_images_csv.exists():
        df = pd.read_csv(unique_images_csv)
        if "image_name" in df.columns:
            return sorted(df["image_name"].astype(str).tolist())

    if unique_images_dir.exists():
        return sorted([p.name for p in unique_images_dir.iterdir() if p.is_file()])

    return []


def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/jpeg"


def _encode_image_base64(path: str) -> str:
    import base64

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _encode_image_data_url(path: str) -> str:
    return f"data:{_guess_mime(path)};base64,{_encode_image_base64(path)}"


def _parse_json_loose(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError(f"Model did not return JSON: {text}")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _call_openai(image_path: str, llm_cfg: dict[str, Any]) -> dict[str, Any]:
    from openai import OpenAI

    api_key_env = llm_cfg.get("api_key_env") or "OPENAI_API_KEY"
    if not os.getenv(api_key_env):
        raise EnvironmentError(f"OpenAI API key not found in env '{api_key_env}'")

    client = OpenAI(api_key=os.getenv(api_key_env))
    model = llm_cfg.get("model") or "gpt-4o-mini"
    system_prompt = llm_cfg.get("system_prompt") or ""
    user_prompt = llm_cfg.get("user_prompt") or "Return JSON only."

    resp = client.chat.completions.create(
        model=model,
        temperature=float(llm_cfg.get("temperature", 0.0)),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image_data_url(image_path)}},
                ],
            },
        ],
    )
    return _parse_json_loose(resp.choices[0].message.content)


def _call_gemini(image_path: str, llm_cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        import importlib

        genai = importlib.import_module("google.generativeai")
    except ModuleNotFoundError:
        raise EnvironmentError(
            "Gemini client not installed. Install with: pip install google-generativeai"
        )

    api_key_env = llm_cfg.get("api_key_env") or "GEMINI_API_KEY"
    key = os.getenv(api_key_env)
    if not key:
        raise EnvironmentError(f"Gemini API key not found in env '{api_key_env}'")

    genai.configure(api_key=key)
    model = genai.GenerativeModel(llm_cfg.get("model") or "models/gemini-2.0-flash")

    system_prompt = llm_cfg.get("system_prompt") or ""
    user_prompt = llm_cfg.get("user_prompt") or "Return JSON only."

    resp = model.generate_content(
        [
            {
                "role": "user",
                "parts": [
                    {"text": system_prompt + "\n\n" + user_prompt},
                    {
                        "inline_data": {
                            "mime_type": _guess_mime(image_path),
                            "data": _encode_image_base64(image_path),
                        }
                    },
                ],
            }
        ],
        generation_config={"temperature": float(llm_cfg.get("temperature", 0.0))},
    )
    return _parse_json_loose(resp.text)


def _call_claude(image_path: str, llm_cfg: dict[str, Any]) -> dict[str, Any]:
    import anthropic

    api_key_env = llm_cfg.get("api_key_env") or "ANTHROPIC_API_KEY"
    key = os.getenv(api_key_env)
    if not key:
        raise EnvironmentError(f"Anthropic API key not found in env '{api_key_env}'")

    client = anthropic.Anthropic(api_key=key)
    model = llm_cfg.get("model") or "claude-3-5-sonnet-latest"
    system_prompt = llm_cfg.get("system_prompt") or ""
    user_prompt = llm_cfg.get("user_prompt") or "Return JSON only."

    resp = client.messages.create(
        model=model,
        max_tokens=int(llm_cfg.get("max_tokens", 300)),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _guess_mime(image_path),
                            "data": _encode_image_base64(image_path),
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    text = resp.content[0].text if resp.content else "{}"
    return _parse_json_loose(text)


def _call_provider(image_path: str, llm_cfg: dict[str, Any]) -> dict[str, Any]:
    provider = str(llm_cfg.get("provider") or "openai").strip().lower()
    if provider == "openai":
        return _call_openai(image_path, llm_cfg)
    if provider == "gemini":
        return _call_gemini(image_path, llm_cfg)
    if provider in {"claude", "anthropic"}:
        return _call_claude(image_path, llm_cfg)
    raise ValueError(f"Unsupported phase2 provider: {provider}")


def _resolve_bool_expectations(expected_values: dict[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for key, value in (expected_values or {}).items():
        if isinstance(key, str) and key.strip():
            out[key.strip()] = _to_bool(value)
    return out


def run_phase2_validation(
    *,
    cfg,
    image_dir: str,
    output_dir: str,
    unique_images_csv_path: str,
    unique_images_dir_path: str,
) -> dict[str, Any]:
    phase2_cfg = cfg.get("phase2") or {}
    llm_cfg = phase2_cfg.get("llm") or {}
    filenames_cfg = cfg.get("filenames") or {}

    unique_images_csv = Path(unique_images_csv_path)
    unique_images_dir = Path(unique_images_dir_path)
    source_images_dir = Path(image_dir)
    image_names = _read_unique_images(unique_images_csv, unique_images_dir)

    output_keys = [str(k).strip() for k in (phase2_cfg.get("output_keys") or []) if str(k).strip()]
    expected_values = _resolve_bool_expectations(phase2_cfg.get("expected_values") or {})
    retries = int(phase2_cfg.get("retry_max") or 3)
    throttle_seconds = float(phase2_cfg.get("throttle_seconds") or 0.0)

    rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for image_name in image_names:
        image_path = unique_images_dir / image_name
        if not image_path.exists():
            image_path = source_images_dir / image_name

        base_row: dict[str, Any] = {
            "image_name": image_name,
            "image_path": str(image_path),
            "llm_provider": llm_cfg.get("provider"),
            "model": llm_cfg.get("model"),
        }

        if not image_path.exists() or not image_path.is_file():
            row = {
                **base_row,
                "overall_valid": False,
                "validation_error": "file_not_found",
                "raw_json": "{}",
            }
            for key in output_keys:
                row[key] = None
            rows.append(row)
            failed_rows.append(row)
            continue

        result_obj: dict[str, Any] | None = None
        last_error = None
        for _ in range(retries):
            try:
                result_obj = _call_provider(str(image_path), llm_cfg)
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                if throttle_seconds > 0:
                    time.sleep(throttle_seconds)

        if result_obj is None:
            row = {
                **base_row,
                "overall_valid": False,
                "validation_error": last_error or "llm_failed",
                "raw_json": "{}",
            }
            for key in output_keys:
                row[key] = None
            rows.append(row)
            failed_rows.append(row)
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
            continue

        selected_values: dict[str, Any] = {}
        for key in output_keys:
            selected_values[key] = _to_bool(result_obj.get(key))

        mismatches: list[str] = []
        for key, expected in expected_values.items():
            actual = _to_bool(result_obj.get(key))
            if actual != expected:
                mismatches.append(key)

        overall_valid = len(mismatches) == 0

        row = {
            **base_row,
            **selected_values,
            "overall_valid": bool(overall_valid),
            "failed_checks": ";".join(mismatches),
            "validation_error": "",
            "raw_json": json.dumps(result_obj, ensure_ascii=True),
        }
        rows.append(row)
        if not overall_valid:
            failed_rows.append(row)

        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    results_df = pd.DataFrame(rows)
    failed_df = pd.DataFrame(failed_rows)

    phase2_report_name = filenames_cfg.get("phase2_validation_report_csv") or "phase2_validation_report.csv"
    phase2_failed_name = filenames_cfg.get("phase2_failed_images_csv") or "phase2_failed_images.csv"
    phase2_summary_name = filenames_cfg.get("phase2_summary_csv") or "phase2_summary.csv"

    phase2_report_csv = Path(output_dir) / str(phase2_report_name)
    phase2_failed_csv = Path(output_dir) / str(phase2_failed_name)
    phase2_summary_csv = Path(output_dir) / str(phase2_summary_name)

    results_df.to_csv(phase2_report_csv, index=False)
    failed_df.to_csv(phase2_failed_csv, index=False)

    summary = {
        "total_images_checked": int(len(results_df)),
        "overall_valid_images": int(results_df["overall_valid"].sum()) if not results_df.empty else 0,
        "failed_images": int(len(failed_df)),
        "provider": str(llm_cfg.get("provider") or "openai"),
        "model": str(llm_cfg.get("model") or ""),
        "output_keys": "|".join(output_keys),
        "expected_values": json.dumps(expected_values, ensure_ascii=True),
    }
    pd.DataFrame([summary]).to_csv(phase2_summary_csv, index=False)

    return {
        "enabled": True,
        "summary": summary,
        "report_csv": str(phase2_report_csv),
        "failed_images_csv": str(phase2_failed_csv),
        "summary_csv": str(phase2_summary_csv),
    }