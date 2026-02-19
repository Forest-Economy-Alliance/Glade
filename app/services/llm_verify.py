import os
import json
import mimetypes
import base64
from typing import List, Dict, Literal
# google.generativeai (Gemini) is imported lazily inside the Gemini provider
from openai import OpenAI
import anthropic
from ..config import Config
# ============================================================
from dotenv import load_dotenv

# Load .env into the process environment for local development
load_dotenv()
# Helpers
# ============================================================

def guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def encode_image_data_url(path: str) -> str:
    return f"data:{guess_mime(path)};base64,{encode_image_base64(path)}"


def parse_json_loose(content: str) -> dict:
    # Accept already-parsed dicts
    if isinstance(content, dict):
        return content
    content = (content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        s, e = content.find("{"), content.rfind("}")
        if s != -1 and e != -1:
            return json.loads(content[s:e+1])
    raise ValueError(f"Model did not return JSON: {content}")


def normalize_output(obj: dict) -> dict:
    def to_bool(v):
        if isinstance(v, bool): return v
        if isinstance(v, str): return v.lower() in {"true","yes","1"}
        return bool(v)

    def to_float(v):
        try: return float(v)
        except: return 0.0

    return {
        "is_duplicate_group": to_bool(obj.get("is_duplicate_group", False)),
        "confidence": max(0, min(1, to_float(obj.get("confidence", 0)))),
        "reason": str(obj.get("reason", "")),
    }



# ============================================================
# Provider Implementations
# ============================================================

def _verify_openai(image_paths: List[str], llm_cfg: dict) -> dict:
    """Call OpenAI chat completions using the configured model/texts."""

    api_key_env = llm_cfg.get("api_key_env")
    print("openai env var for key:", api_key_env)
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(f"OpenAI API key not found in env '{api_key_env}'")

    client = OpenAI()
    model = llm_cfg.get("model") or os.getenv("VISION_MODEL", "gpt-4o-mini")

    system_text = llm_cfg.get("system_text") or ""
    user_text = llm_cfg.get("user_text") or ""

    image_parts = [{
        "type": "image_url",
        "image_url": {"url": encode_image_data_url(p)}
    } for p in image_paths]

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [{"type": "text", "text": user_text}, *image_parts]}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=llm_cfg.get("temperature", 0),
        response_format={"type": "json_object"},
    )

    # The SDK returns structured content in different shapes; normalize to string
    choice = resp.choices[0]
    # Normalize several possible shapes into a string or dict
    content = None
    try:
        # structured SDK: choice.message.content
        content = choice.message.content
    except Exception:
        pass
    if content is None:
        # try dict-like
        try:
            content = choice.message.get("content")
        except Exception:
            pass
    if content is None:
        # fallback to raw text / string representation
        content = getattr(choice, "text", None) or str(choice)

    return parse_json_loose(content)


def _verify_gemini(image_paths: List[str], llm_cfg: dict) -> dict:
    # Import the Gemini client lazily so the module is optional unless used
    try:
        import google.generativeai as genai
    except ModuleNotFoundError:
        raise EnvironmentError("google-genai (google.generativeai) is not installed. Install with: pip install google-genai")

    model_name = llm_cfg.get("model") or os.getenv("GEMINI_VISION_MODEL", "models/gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)

    system_text = llm_cfg.get("system_text") or ""
    user_text = llm_cfg.get("user_text") or ""

    image_parts = [{
        "inline_data": {
            "mime_type": guess_mime(p),
            "data": encode_image_base64(p)
        }
    } for p in image_paths]

    resp = model.generate_content(
        [{
            "role": "user",
            "parts": [{"text": system_text + "\n\n" + user_text}, *image_parts]
        }],
        generation_config={"temperature": llm_cfg.get("temperature", 0)}
    )

    return parse_json_loose(resp.text)


def _verify_anthropic(image_paths: List[str], llm_cfg: dict) -> dict:
    """Call Anthropic / Claude with inline base64 images and configured prompts."""

    api_key_env = llm_cfg.get("api_key_env") or "ANTHROPIC_API_KEY"
    key = os.getenv(api_key_env)
    if not key:
        raise EnvironmentError(f"Anthropic API key not found in env '{api_key_env}'")

    # instantiate client (support a couple of SDK variants)
    try:
        client = anthropic.Anthropic(api_key=key)
    except Exception:
        client = anthropic.Client(api_key=key)

    system_text = llm_cfg.get("system_text") or ""
    user_text = llm_cfg.get("user_text") or ""
    model = llm_cfg.get("model") or "claude-3-5-sonnet-latest"

    image_contents = []
    for p in image_paths:
        mime = guess_mime(p)
        b64 = encode_image_base64(p)
        image_contents.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": b64,
            }
        })

    resp = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=llm_cfg.get("temperature", 0),
        system=system_text,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    *image_contents,
                ],
            }
        ],
    )

    # Anthropic may return content in different shapes
    content = None
    try:
        content = resp.content[0].text
    except Exception:
        try:
            content = resp["content"][0]["text"]
        except Exception:
            content = str(resp)

    return parse_json_loose(content)


# ============================================================
# Main General Function
# ============================================================

def verify_duplicate_group(
    image_names: List[str],
    image_dir: str,
    provider: Literal["openai", "gemini", "anthropic"] = None,
) -> Dict:
    """
    image_names: list of file names
    image_dir: folder containing images
    provider: openai / gemini
    """

    assert len(image_names) >= 2, "Need at least 2 images"

    image_paths = []
    for name in image_names:
        p = os.path.join(image_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        image_paths.append(p)

    # Load config and LLM settings
    cfg = Config()
    # support either `similarity_check_llm` (new) or `llm` (legacy)
    llm_cfg = cfg.get("similarity_check_llm") or cfg.get("llm") or {}
    # If provider not explicitly passed, use config
    provider = provider or llm_cfg.get("provider", "openai")

    if provider == "openai":
        raw = _verify_openai(image_paths, llm_cfg)

    elif provider == "gemini":
        raw = _verify_gemini(image_paths, llm_cfg)

    elif provider == "anthropic":
        raw = _verify_anthropic(image_paths, llm_cfg)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return normalize_output(raw)
