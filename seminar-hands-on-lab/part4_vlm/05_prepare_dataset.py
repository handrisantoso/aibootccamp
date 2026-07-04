"""
Part 5 - Prepare a fine-tuning dataset.

Turns a raw image->instruction->answer dataset into clean train/val/test
splits, while enforcing the best practices that make or break a fine-tune:

  - verifies every referenced image exists
  - hashes image bytes to detect DUPLICATES ACROSS SPLITS (leakage kills eval)
  - checks assistant answers parse as JSON when they look like JSON
  - warns if you have too few examples or no "hard"/negative cases
  - writes JSONL splits ready for 06_finetune_qwen2vl_lora.py

Usage:
    python 05_prepare_dataset.py --in ../data/sample_dataset.json --outdir ../data/prepared
"""
import argparse
import hashlib
import json
import os
import random
import sys


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="skip the image-exists check (for a dry run without images)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(args.inp))
    with open(args.inp, encoding="utf-8") as f:
        rows = json.load(f)

    problems, hashes, cleaned = [], {}, []
    for i, r in enumerate(rows):
        img_rel = r.get("image")
        img_abs = os.path.join(base, img_rel) if img_rel else None

        if not img_rel:
            problems.append(f"row {i}: missing 'image'")
            continue

        if os.path.exists(img_abs):
            digest = sha(img_abs)
            if digest in hashes:
                problems.append(f"row {i}: DUPLICATE image of row {hashes[digest]} (leakage risk)")
            hashes[digest] = i
        elif not args.allow_missing_images:
            problems.append(f"row {i}: image not found -> {img_rel}")
            continue

        # if the assistant answer looks like JSON, confirm it parses
        conv = r.get("conversations", [])
        ans = next((c["content"] for c in conv if c.get("role") == "assistant"), "")
        if ans.strip().startswith("{"):
            try:
                json.loads(ans)
            except json.JSONDecodeError as e:
                problems.append(f"row {i}: assistant answer is not valid JSON ({e})")
        cleaned.append(r)

    # quality warnings (non-fatal)
    warnings = []
    if len(cleaned) < 50:
        warnings.append(f"only {len(cleaned)} examples - fine for a demo, too few for production.")
    if not any("null" in json.dumps(r) or "not a receipt" in json.dumps(r) for r in cleaned):
        warnings.append("no negative/edge cases detected - add blurry/empty/'none' examples.")

    if problems:
        print("BLOCKING PROBLEMS:")
        for p in problems:
            print("  - " + p)
        sys.exit("\nFix the problems above (or pass --allow-missing-images for a dry run).")

    for w in warnings:
        print("WARNING: " + w)

    # deterministic split
    random.seed(args.seed)
    random.shuffle(cleaned)
    n = len(cleaned)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    test, val, train = cleaned[:n_test], cleaned[n_test:n_test + n_val], cleaned[n_test + n_val:]

    os.makedirs(args.outdir, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.outdir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(split):>4} -> {path}")

    print("\nDone. No image leakage across splits. Ready for Part 6.")


if __name__ == "__main__":
    main()
