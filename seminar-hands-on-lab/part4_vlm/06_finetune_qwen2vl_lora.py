"""
Part 6 - LoRA / QLoRA fine-tune of Qwen2-VL.

NEEDS A GPU. Qwen2-VL-2B fits on ~16 GB with 4-bit QLoRA; the 7B needs 24 GB+.
No local GPU? Run this exact script on Colab / RunPod / Lambda / Vast.ai.

Best practices baked in:
  * freeze the vision encoder AND the base LLM; train only LoRA adapters
  * 4-bit base weights (QLoRA) + gradient checkpointing to fit memory
  * gradient accumulation to simulate a larger batch
  * small LR (1e-4) and few epochs (1-3): small data overfits fast
  * evaluate on a val split each epoch

Usage:
    python 06_finetune_qwen2vl_lora.py \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --data ../data/prepared \
        --output ../out/qwen2vl-receipts-lora \
        --epochs 2 --batch-size 1 --grad-accum 8 --lr 1e-4 --lora-r 16
"""
import argparse
import json
import os


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--data", required=True, help="dir with train.jsonl / val.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-4bit", action="store_true", help="disable QLoRA 4-bit (needs more VRAM)")
    args = ap.parse_args()

    # Heavy imports live inside main() so `python -c` / --help work without a GPU stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
    from qwen_vl_utils import process_vision_info

    data_dir = os.path.abspath(args.data)
    img_root = os.path.dirname(data_dir)  # images/ paths are relative to the data folder's parent

    # ---- model + processor (4-bit base for QLoRA) ----
    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ---- freeze everything, then attach LoRA to the LLM attention projections ----
    for p in model.parameters():
        p.requires_grad = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # attention projections of the language model; leave the vision tower frozen
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()  # should be ~1% of total

    def to_messages(row):
        img_path = os.path.join(img_root, row["image"])
        user_txt = next(c["content"] for c in row["conversations"] if c["role"] == "user")
        asst_txt = next(c["content"] for c in row["conversations"] if c["role"] == "assistant")
        return img_path, user_txt, asst_txt

    def collate(batch):
        texts, image_inputs = [], []
        for row in batch:
            img_path, user_txt, asst_txt = to_messages(row)
            msgs = [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_txt},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": asst_txt}]},
            ]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            imgs, _ = process_vision_info(msgs)
            image_inputs.append(imgs)

        batch_imgs = [im for sub in image_inputs for im in sub]
        model_inputs = processor(
            text=texts, images=batch_imgs, return_tensors="pt", padding=True,
        )
        labels = model_inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  # ignore padding in loss
        model_inputs["labels"] = labels
        return model_inputs

    train = Dataset.from_list(load_jsonl(os.path.join(data_dir, "train.jsonl")))
    val_path = os.path.join(data_dir, "val.jsonl")
    val = Dataset.from_list(load_jsonl(val_path)) if os.path.exists(val_path) else None

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if val is not None else "no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collate,
    )
    trainer.train()

    model.save_pretrained(args.output)          # saves ONLY the LoRA adapter (tens of MB)
    processor.save_pretrained(args.output)
    print(f"\nLoRA adapter saved to {args.output}")
    print("Next: python 07_evaluate.py  (compare base vs. tuned before you trust it)")


if __name__ == "__main__":
    main()
