"""
Part 3.0 - Image captioning with a VLM (the gentlest possible start).

Where Parts 1-2 needed a separate model per task, ONE VLM captions any image -
and you steer it in plain language. This script shows three caption styles on
the same image so the contrast with classical CV is obvious.

Prereq:
    ollama pull llava:7b      # or moondream (tiny), llama3.2-vision

Run:
    python 00_image_captioning.py path/to/photo.jpg
    python 00_image_captioning.py path/to/photo.jpg --model moondream
"""
import argparse
import sys

import ollama


def caption(model, image, prompt, temperature=0.4):
    resp = ollama.chat(
        model=model,
        options={"temperature": temperature},
        messages=[{"role": "user", "content": prompt, "images": [image]}],
    )
    return resp["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="llava:7b")
    args = ap.parse_args()

    styles = [
        ("ONE-LINE ALT TEXT", "Write a single concise alt-text caption for this image.", 0.2),
        ("DETAILED DESCRIPTION", "Describe this image in 3-4 sentences: scene, objects, mood.", 0.5),
        ("COUNT & LIST", "How many people are in this image? List what each is doing.", 0.1),
    ]

    try:
        for label, prompt, temp in styles:
            print("\n" + "=" * 60 + f"\n{label}\n" + "=" * 60)
            print(caption(args.model, args.image, prompt, temp))
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Failed (is Ollama running and the model pulled?): {e}")

    print("\n\nSame model, three tasks, zero retraining - just different prompts. "
          "That flexibility is the whole point of VLMs vs. classical CV.")


if __name__ == "__main__":
    main()
