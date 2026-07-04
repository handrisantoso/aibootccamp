"""
Part 2 - Prompt it well.

Runs the SAME image through four prompting strategies so you can see, side by
side, how much the wording changes the answer:

  1. Vague          - "What's in this image?"
  2. Specific task  - a precise instruction
  3. Role + context - system persona primes vocabulary and tone
  4. Few-shot       - one worked example steers the format

Also demonstrates the `temperature` knob: low for extraction, higher for
creative captions.

Usage:
    python 02_prompting.py path/to/receipt.jpg
"""
import argparse
import sys

import ollama


def ask(model, prompt, image, system=None, temperature=0.2):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt, "images": [image]})
    resp = ollama.chat(model=model, messages=messages, options={"temperature": temperature})
    return resp["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="llava:7b")
    args = ap.parse_args()

    demos = []

    # 1. Vague -- generic, hard to use
    demos.append(("1. VAGUE", lambda: ask(
        args.model, "What's in this image?", args.image)))

    # 2. Specific task -- scoped, actionable
    demos.append(("2. SPECIFIC TASK", lambda: ask(
        args.model,
        "List every text field you can read on this receipt as bullet points. "
        "If you cannot read a value, write '[unreadable]'.",
        args.image)))

    # 3. Role + context -- persona sharpens domain language
    demos.append(("3. ROLE + CONTEXT", lambda: ask(
        args.model,
        "Summarise this receipt for expense reporting: merchant, date, total, category.",
        args.image,
        system="You are a meticulous accounting assistant. Be concise and never invent values.")))

    # 4. Few-shot -- one example locks in the format
    demos.append(("4. FEW-SHOT (format locked)", lambda: ask(
        args.model,
        "Extract as: MERCHANT | DATE | TOTAL\n"
        "Example -> 'Warung Ibu | 2026-01-02 | 45000'\n"
        "Now do this image:",
        args.image)))

    try:
        for label, fn in demos:
            print("\n" + "=" * 60)
            print(label)
            print("=" * 60)
            print(fn())
    except ollama.ResponseError as e:
        sys.exit(f"Ollama error: {e}")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Failed (is Ollama running, model pulled?): {e}")

    print("\n\nTakeaway: same image, very different usefulness. Specificity + a "
          "format example do most of the work.")


if __name__ == "__main__":
    main()
