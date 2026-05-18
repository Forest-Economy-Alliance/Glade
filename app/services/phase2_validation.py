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


def _default_model_for_provider(provider: str) -> str:
    p = str(provider).strip().lower()
    if p == "openai":
        return "gpt-4o-mini"
    if p == "gemini":
        return "models/gemini-2.0-flash"
    return "claude-3-5-sonnet-latest"


def _default_api_env_for_provider(provider: str) -> str:
    p = str(provider).strip().lower()
    if p == "openai":
        return "OPENAI_API_KEY"
    if p == "gemini":
        return "GEMINI_API_KEY"
    return "ANTHROPIC_API_KEY"


def _resolve_api_key_env(provider: str, llm_cfg: dict[str, Any], configured_envs: dict[str, Any]) -> str:
    provider_key = str(provider).strip().lower()
    provider_specific = configured_envs.get(provider_key)
    if provider_specific and str(provider_specific).strip():
        return str(provider_specific).strip()
    fallback = llm_cfg.get("api_key_env")
    if fallback and str(fallback).strip():
        return str(fallback).strip()
    return _default_api_env_for_provider(provider_key)


def run_phase2_validation(
    *,
    cfg,
    image_dir: str,
    output_dir: str,
    unique_images_csv_path: str,
    unique_images_dir_path: str,
    log_fn=None,
) -> dict[str, Any]:
    phase2_cfg = cfg.get("phase2") or {}
    llm_cfg = phase2_cfg.get("llm") or {}
    filenames_cfg = cfg.get("filenames") or {}

    unique_images_csv = Path(unique_images_csv_path)
    unique_images_dir = Path(unique_images_dir_path)
    source_images_dir = Path(image_dir)
    image_names = _read_unique_images(unique_images_csv, unique_images_dir)

    output_keys = [str(k).strip() for k in (phase2_cfg.get("output_keys") or []) if str(k).strip()]
    retries = int(phase2_cfg.get("retry_max") or 3)
    throttle_seconds = float(phase2_cfg.get("throttle_seconds") or 0.0)

    providers = [
        str(p).strip().lower()
        for p in (llm_cfg.get("providers") or [llm_cfg.get("provider") or "openai"])
        if str(p).strip()
    ]
    if not providers:
        providers = ["openai"]

    configured_models = llm_cfg.get("models") or {}
    if not isinstance(configured_models, dict):
        configured_models = {}
    configured_api_key_envs = llm_cfg.get("api_key_envs") or {}
    if not isinstance(configured_api_key_envs, dict):
        configured_api_key_envs = {}

    phase2_report_name = filenames_cfg.get("phase2_validation_report_csv") or "phase2_validation_report.csv"
    base_stem = Path(str(phase2_report_name)).stem
    validation_dir = Path(output_dir) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    provider_csv_files: dict[str, str] = {}
    provider_counts: dict[str, int] = {}

    if callable(log_fn):
        log_fn(
            "Phase 2 started in consensus mode: "
            f"providers={','.join(providers)} images={len(image_names)}"
        )

    for provider in providers:
        provider_rows: list[dict[str, Any]] = []
        model_name = str(configured_models.get(provider) or llm_cfg.get("model") or _default_model_for_provider(provider))
        provider_cfg = dict(llm_cfg)
        provider_cfg["provider"] = provider
        provider_cfg["model"] = model_name
        provider_cfg["api_key_env"] = _resolve_api_key_env(provider, llm_cfg, configured_api_key_envs)

        if callable(log_fn):
            log_fn(f"Starting provider={provider} model={model_name}")

        for idx, image_name in enumerate(image_names, start=1):
            image_path = unique_images_dir / image_name
            if not image_path.exists():
                image_path = source_images_dir / image_name

            if callable(log_fn):
                log_fn(f"[{provider}] [{idx}/{len(image_names)}] {image_name}")

            base_row: dict[str, Any] = {
                "image_name": image_name,
                "image_path": str(image_path),
                "llm_provider": provider,
                "model": model_name,
            }

            if not image_path.exists() or not image_path.is_file():
                row = {
                    **base_row,
                    "validation_error": "file_not_found",
                    "raw_json": "{}",
                }
                for key in output_keys:
                    row[key] = None
                provider_rows.append(row)
                continue

            result_obj: dict[str, Any] | None = None
            last_error = None
            for _ in range(retries):
                try:
                    result_obj = _call_provider(str(image_path), provider_cfg)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if throttle_seconds > 0:
                        time.sleep(throttle_seconds)

            if result_obj is None:
                row = {
                    **base_row,
                    "validation_error": last_error or "llm_failed",
                    "raw_json": "{}",
                }
                for key in output_keys:
                    row[key] = None
                provider_rows.append(row)
                if throttle_seconds > 0:
                    time.sleep(throttle_seconds)
                continue

            extracted_values: dict[str, Any] = {}
            for key in output_keys:
                extracted_values[key] = result_obj.get(key)

            row = {
                **base_row,
                **extracted_values,
                "validation_error": "",
                "raw_json": json.dumps(result_obj, ensure_ascii=True),
            }
            provider_rows.append(row)

            if throttle_seconds > 0:
                time.sleep(throttle_seconds)

        provider_df = pd.DataFrame(provider_rows)
        provider_slug = provider.replace("/", "_").replace(" ", "_")
        provider_csv = validation_dir / f"{base_stem}_{provider_slug}.csv"
        provider_df.to_csv(provider_csv, index=False)
        provider_csv_files[provider] = str(provider_csv)
        provider_counts[provider] = int(len(provider_df))

        if callable(log_fn):
            log_fn(f"Completed provider={provider} rows={len(provider_df)} output={provider_csv}")

    if callable(log_fn):
        log_fn("Phase 2 finished for all selected providers")

    return {
        "enabled": True,
        "mode": "consensus" if len(providers) > 1 else "single_provider",
        "validation_dir": str(validation_dir),
        "providers": providers,
        "rows_per_provider": provider_counts,
        "csv_files": provider_csv_files,
    }