"""
Part 2.1 - Structured output you can actually parse.

Free text is hard to use downstream. Here we:
  - force JSON with Ollama's `format="json"` mode
  - give a strict schema in the prompt
  - validate the returned JSON against that schema
  - retry ONCE with the error fed back if validation fails

This retry-on-invalid pattern is what you want in production.

Usage:
    python 03_structured_output.py path/to/receipt.jpg
"""
import argparse
import json
import sys

import ollama

# The shape we require back. Keep it small and typed.
REQUIRED_FIELDS = {
    "merchant": (str, type(None)),
    "date": (str, type(None)),
    "total": (int, float, type(None)),
    "currency": (str, type(None)),
    "line_items": (list,),
}

PROMPT = (
    "Extract this receipt as JSON with EXACTLY these keys: "
    "merchant (string), date (YYYY-MM-DD string), total (number), "
    "currency (string), line_items (array of {name, price}). "
    "Use null for any field you cannot read. Return JSON only, no prose."
)


def validate(obj: dict) -> list:
    """Return a list of human-readable problems (empty list == valid)."""
    problems = []
    if not isinstance(obj, dict):
        return ["top-level value is not an object"]
    for key, types in REQUIRED_FIELDS.items():
        if key not in obj:
            problems.append(f"missing key '{key}'")
        elif not isinstance(obj[key], types):
            problems.append(f"key '{key}' has wrong type {type(obj[key]).__name__}")
    return problems


def call(model, image, prompt):
    resp = ollama.chat(
        model=model,
        format="json",  # ask Ollama to constrain output to valid JSON
        options={"temperature": 0.0},
        messages=[{"role": "user", "content": prompt, "images": [image]}],
    )
    return resp["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="llava:7b")
    args = ap.parse_args()

    prompt = PROMPT
    for attempt in (1, 2):
        try:
            raw = call(args.model, args.image, prompt)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Ollama call failed: {e}")

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[attempt {attempt}] not valid JSON: {e}")
            prompt = PROMPT + f"\nYour previous reply was not valid JSON ({e}). Return JSON only."
            continue

        problems = validate(obj)
        if not problems:
            print(f"[attempt {attempt}] VALID\n")
            print(json.dumps(obj, indent=2, ensure_ascii=False))
            return

        print(f"[attempt {attempt}] schema problems: {problems}")
        prompt = PROMPT + "\nFix these problems: " + "; ".join(problems)

    sys.exit("\nStill invalid after retry. In production: log it, fall back, or flag for review.")


if __name__ == "__main__":
    main()
