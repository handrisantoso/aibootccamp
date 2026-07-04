"""
Part 3 - Ground it (RAG).

A VLM knows general facts, not YOUR facts. This tiny, dependency-free RAG loop:

  1. VLM extracts key entities/description from the image
  2. embed that text (Ollama's nomic-embed-text) and retrieve the most similar
     entries from a small JSON knowledge base
  3. re-ask the VLM WITH the retrieved policy/context injected
  4. print the grounded answer + the sources used (for citation)

Swap the in-memory store for Chroma / Qdrant / pgvector in production.

Prereq:
    ollama pull nomic-embed-text

Usage:
    python 04_rag_grounding.py path/to/product.jpg "Is this compliant with our packaging policy?"
"""
import argparse
import json
import os
import sys

import numpy as np
import ollama

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")


def embed(text: str) -> np.ndarray:
    resp = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(resp["embedding"], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def retrieve(query_vec, kb, k=2):
    scored = [(cosine(query_vec, np.array(item["_vec"], dtype=np.float32)), item) for item in kb]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("question")
    ap.add_argument("--model", default="llava:7b")
    args = ap.parse_args()

    if not os.path.exists(KB_PATH):
        sys.exit(f"Knowledge base not found: {KB_PATH}")
    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)

    try:
        # pre-embed the KB once (in production you'd cache these vectors)
        for item in kb:
            item["_vec"] = embed(item["text"]).tolist()

        # 1. describe the image
        desc = ollama.chat(
            model=args.model,
            options={"temperature": 0.1},
            messages=[{"role": "user",
                       "content": "Describe this product's packaging: materials, labels, symbols, text.",
                       "images": [args.image]}],
        )["message"]["content"]
        print("\n--- image description ---\n" + desc)

        # 2. retrieve relevant policy
        hits = retrieve(embed(desc + " " + args.question), kb, k=2)
        context = "\n".join(f"[{i+1}] {h[1]['title']}: {h[1]['text']}" for i, h in enumerate(hits))
        print("\n--- retrieved context ---\n" + context)

        # 3. re-ask, grounded
        grounded = ollama.chat(
            model=args.model,
            options={"temperature": 0.1},
            messages=[{
                "role": "user",
                "content": (
                    f"Using ONLY the policy context below, answer the question. "
                    f"Cite the [n] you relied on. If the policy doesn't cover it, say so.\n\n"
                    f"POLICY CONTEXT:\n{context}\n\nQUESTION: {args.question}"
                ),
                "images": [args.image],
            }],
        )["message"]["content"]

        print("\n--- grounded answer ---\n" + grounded)
        print("\nSources: " + ", ".join(h[1]["title"] for h in hits))

    except Exception as e:  # noqa: BLE001
        sys.exit(f"Failed (did you `ollama pull nomic-embed-text`?): {e}")


if __name__ == "__main__":
    main()
