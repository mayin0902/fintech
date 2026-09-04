#!/usr/bin/env python3
"""DPO recipe for execution-derived chosen/rejected SQL pairs."""

from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="artifacts/dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=5e-7,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
