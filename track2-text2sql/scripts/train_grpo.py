#!/usr/bin/env python3
"""Online GRPO recipe using PostgreSQL execution as a verifiable reward."""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from text2sql_feedback.executor import PostgresExecutor
from text2sql_feedback.trl_adapter import TRLExecutionReward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="artifacts/grpo")
    parser.add_argument("--num-generations", type=int, default=8)
    args = parser.parse_args()

    dsn = os.environ["TEXT2SQL_DSN"]
    executor = PostgresExecutor(
        dsn=dsn,
        timezone=os.getenv("TEXT2SQL_TIMEZONE", "Asia/Shanghai"),
        statement_timeout_ms=int(os.getenv("TEXT2SQL_STATEMENT_TIMEOUT_MS", "5000")),
    )
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=1e-6,
        num_generations=args.num_generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_completion_length=512,
        temperature=1.0,
        beta=0.02,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=TRLExecutionReward(executor),
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
