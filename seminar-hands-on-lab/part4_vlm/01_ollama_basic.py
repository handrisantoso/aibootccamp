"""
Part 1 - Just use it.

The simplest possible VLM call: hand Ollama an image and a question.
Shows two ways to do it:
  (A) the official `ollama` Python client
  (B) a raw HTTP POST (handy on servers without the client library)

Usage:
    python 01_ollama_basic.py path/to/image.jpg
    python 01_ollama_basic.py path/to/image.jpg --model moondream
"""
import argparse
import base64
import json
import sys

import requests

OLLAMA_URL = "http://localhost:11434"


def via_client(image_path: str, model: str, prompt: str) -> str:
    """Approach A: official client. `pip install ollama`."""
    import ollama  # imported here so approach B works even without the package

    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [image_path]}],
    )
    return resp["message"]["content"]


def via_http(image_path: str, model: str, prompt: str) -> str:
    """Approach B: raw HTTP. Images are base64-encoded in the JSON body."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--model", default="llava:7b")
    ap.add_argument("--prompt", default="Describe this image in two sentences.")
    ap.add_argument("--http", action="store_true", help="use raw HTTP instead of the client")
    args = ap.parse_args()

    try:
        if args.http:
            answer = via_http(args.image, args.model, args.prompt)
        else:
            answer = via_client(args.image, args.model, args.prompt)
    except requests.exceptions.ConnectionError:
        sys.exit("Cannot reach Ollama on :11434 - is it running? Try `ollama serve`.")
    except FileNotFoundError:
        sys.exit(f"Image not found: {args.image}")

    print(f"\nMODEL: {args.model}")
    print(f"PROMPT: {args.prompt}\n")
    print(answer)


if __name__ == "__main__":
    main()
