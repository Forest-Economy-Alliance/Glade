import os
import argparse
from typing import List
import pandas as pd

from .llm_verify import verify_duplicate_group
from ..config import Config


def _group_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    groups = {}
    if "group_id" not in df.columns or "image_name" not in df.columns:
        raise ValueError("CSV must contain 'group_id' and 'image_name' columns")
    for gid, sub in df.groupby("group_id"):
        groups[gid] = sub["image_name"].astype(str).tolist()
    return groups


def run(csv_path: str, image_dir: str = None, provider: str = None, out_csv: str = None):
    cfg = Config()
    if image_dir is None:
        image_dir = cfg.get("input.image_folder") or os.getcwd()

    groups = _group_from_csv(csv_path)

    confirmed_duplicate_names = set()
    llm_rejected_names = set()
    failed_similarity_checks = set()

    rows = []

    for idx, (gid, group) in enumerate(groups.items()):
        print(f"\nGroup {idx+1} ({gid}): {group}")
        # prefix each image name with the group subfolder so verifier looks in group folders
        group_prefixed = [os.path.join(str(gid), name) for name in group]
        try:
            result = verify_duplicate_group(group_prefixed, image_dir, provider=provider)
        except Exception as e:
            print(f"  ⚠️ Verification failed: {e}")
            failed_similarity_checks.add(gid)
            rows.append({
                "group_id": gid,
                "image_names": "|".join(group_prefixed),
                "is_duplicate_group": False,
                "confidence": 0.0,
                "reason": f"error: {e}",
            })
            continue

        if result.get("is_duplicate_group") is True:
            confirmed_duplicate_names.update(group)
            print(f"  ✅ Duplicates confirmed (confidence={result['confidence']:.2f})")
        else:
            failed_similarity_checks.add(gid)
            llm_rejected_names.update(group)
            print(f"  ❌ Not duplicates (confidence={result['confidence']:.2f})")
            print(f"  Reason: {result['reason']}")

        rows.append({
            "group_id": gid,
            "image_names": "|".join(group_prefixed),
            "is_duplicate_group": bool(result.get("is_duplicate_group", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": result.get("reason", ""),
        })

    verified_count = len(groups) - len(failed_similarity_checks)
    print(f"\n✅ Verified {verified_count} groups as duplicates")

    out_df = pd.DataFrame(rows)
    if out_csv is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_csv = os.path.join(os.path.dirname(csv_path), f"{base}_llm_verified.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote results to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM verification for near-duplicate groups")
    parser.add_argument("csv", help="Path to near-duplicate groups CSV (group_id,image_name)")
    parser.add_argument("--image-dir", help="Image directory (overrides config)")
    parser.add_argument("--provider", choices=["openai", "gemini", "anthropic"], help="LLM provider to use (overrides config)")
    parser.add_argument("--out", help="Output CSV path (optional)")
    args = parser.parse_args()

    run(args.csv, image_dir=args.image_dir, provider=args.provider, out_csv=args.out)


if __name__ == "__main__":
    main()
