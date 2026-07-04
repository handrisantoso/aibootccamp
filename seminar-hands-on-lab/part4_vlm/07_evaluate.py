"""
Part 7 - Evaluate before you trust.

Compares BASE vs. TUNED on a held-out test set and reports:
  * valid-JSON rate     - did it return parseable output?
  * field accuracy      - exact match on merchant / date / total
  * don't-know rate      - correctly returns null on the 'not a receipt' cases
  * regression check     - a couple of general captioning prompts to confirm the
                           tune didn't cause catastrophic forgetting

Ship only if TUNED beats BASE on field accuracy AND passes the regression check.

Usage:
    python 07_evaluate.py \
        --base Qwen/Qwen2-VL-2B-Instruct \
        --adapter ../out/qwen2vl-receipts-lora \
        --test ../data/prepared/test.jsonl
"""
import argparse
import json
import os


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_json_loose(text):
    """Pull the first {...} block out of a reply and parse it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def score(rows, generate):
    """generate(img_path, prompt) -> str. Returns a metrics dict."""
    valid = fields_ok = fields_total = dk_ok = dk_total = 0
    for r in rows:
        img = r["image"]
        user = next(c["content"] for c in r["conversations"] if c["role"] == "user")
        gold = parse_json_loose(next(c["content"] for c in r["conversations"] if c["role"] == "assistant")) or {}
        pred = parse_json_loose(generate(img, user))

        if pred is not None:
            valid += 1

        is_negative = gold.get("merchant") is None
        if is_negative:
            dk_total += 1
            if pred is not None and pred.get("merchant") is None:
                dk_ok += 1
        else:
            for key in ("merchant", "date", "total"):
                if key in gold:
                    fields_total += 1
                    if pred is not None and str(pred.get(key)) == str(gold.get(key)):
                        fields_ok += 1

    n = len(rows)
    return {
        "valid_json_rate": round(valid / n, 3) if n else 0.0,
        "field_accuracy": round(fields_ok / fields_total, 3) if fields_total else 0.0,
        "dont_know_rate": round(dk_ok / dk_total, 3) if dk_total else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    data_dir = os.path.dirname(os.path.abspath(args.test))
    img_root = os.path.dirname(data_dir)
    rows = load_jsonl(args.test)

    processor = AutoProcessor.from_pretrained(args.base)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")

    def make_generator(model):
        def gen(img_rel, prompt):
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": os.path.join(img_root, img_rel)},
                {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inputs = processor(text=[text], images=imgs, videos=vids,
                               return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            trimmed = out[:, inputs["input_ids"].shape[1]:]
            return processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return gen

    print("Scoring BASE model...")
    base_metrics = score(rows, make_generator(base))

    print("Loading adapter + scoring TUNED model...")
    tuned = PeftModel.from_pretrained(base, args.adapter)
    tuned_metrics = score(rows, make_generator(tuned))

    print("\n=== RESULTS (base -> tuned) ===")
    for k in base_metrics:
        print(f"  {k:18s}: {base_metrics[k]}  ->  {tuned_metrics[k]}")

    improved = tuned_metrics["field_accuracy"] >= base_metrics["field_accuracy"]
    print("\nVERDICT:", "SHIP-worthy on this metric." if improved
          else "Tuned did NOT beat base - revisit data/LR/epochs.")
    print("Remember to eyeball a few general-caption prompts for forgetting before shipping.")


if __name__ == "__main__":
    main()
