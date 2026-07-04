"""
Part 8 - Export the tuned model and serve it from Ollama.

Step 1 (this script): merge the LoRA adapter back into the base weights and save
a standalone HF model, then emit an Ollama Modelfile.

Step 2 (manual, printed below): convert to GGUF with llama.cpp and register the
model with Ollama so you can `ollama run` your tuned VLM like any other.

Usage:
    python 08_export_to_ollama.py \
        --base Qwen/Qwen2-VL-2B-Instruct \
        --adapter ../out/qwen2vl-receipts-lora \
        --outdir ../out/merged
"""
import argparse
import os
import textwrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading base + adapter, merging weights...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.outdir, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.outdir)
    print(f"Merged model saved to {args.outdir}")

    # Emit a Modelfile that points at a (to-be-created) GGUF.
    modelfile = textwrap.dedent("""\
        # Auto-generated. After converting to GGUF (see steps below), this
        # registers your tuned receipt reader with Ollama.
        FROM ./model.gguf
        PARAMETER temperature 0.0
        SYSTEM \"\"\"
        You are a receipt-extraction assistant. Return ONLY valid JSON with keys
        merchant, date (YYYY-MM-DD), total (number), currency. Use null if unreadable.
        \"\"\"
    """)
    mf_path = os.path.join(args.outdir, "Modelfile")
    with open(mf_path, "w", encoding="utf-8") as f:
        f.write(modelfile)

    print("\nNEXT STEPS (run in a shell):")
    print(textwrap.dedent(f"""\
        # 1. get llama.cpp and convert the merged HF model to GGUF
        git clone https://github.com/ggerganov/llama.cpp
        python llama.cpp/convert_hf_to_gguf.py {args.outdir} \\
               --outfile {args.outdir}/model.gguf --outtype q8_0

        # 2. register with Ollama and run
        cd {args.outdir}
        ollama create receipt-reader-tuned -f Modelfile
        ollama run receipt-reader-tuned ./../../data/images/receipt.jpg

        # NOTE: GGUF vision support depends on your llama.cpp/Ollama version.
        # If image input isn't supported for this arch yet, serve the adapter
        # on a GPU with vLLM instead (see Part 8b in LAB.md).
    """))


if __name__ == "__main__":
    main()
