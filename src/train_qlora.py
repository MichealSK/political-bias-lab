from __future__ import annotations

import os
from pathlib import Path


def train_qlora(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    train_file: str | Path = "data/train_anonymized.jsonl",
    output_dir: str | Path = "results/qlora_adapter",
    seed: int = 42,
) -> None:
    """Optional zero-budget QLoRA experiment for Kaggle GPU.

    This is intentionally separate from the main benchmark. Do not train on test items.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train_file = str(train_file)
    output_dir = str(output_dir)
    dataset = load_dataset("json", data_files={"train": train_file})["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_example(example):
        return f"### Prompt\n{example['prompt']}\n\n### Completion\n{example['completion']}"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        max_length=1024,
        packing=False,
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_example,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train_qlora(
        model_id=os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"),
        train_file=os.getenv("TRAIN_FILE", "data/train_anonymized.jsonl"),
        output_dir=os.getenv("OUTPUT_DIR", "results/qlora_adapter"),
        seed=int(os.getenv("GLOBAL_SEED", "42")),
    )
