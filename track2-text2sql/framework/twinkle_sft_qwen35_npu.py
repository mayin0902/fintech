"""Qwen3.5 LoRA SFT recipe for verified-only Text-to-SQL data on Ascend NPU.

Project adapter by mayin0902. It calls the Apache-2.0 ModelScope Twinkle
framework and intentionally contains no dataset, checkpoint, or machine-specific
absolute path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import twinkle
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoConfig
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import Message, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.kernel import kernelize
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor
from twinkle.utils.framework import Torch

LOGGER = get_logger()
VALID_ROLES = {"system", "user", "assistant", "tool"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def normalize_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return []
        role, content = item.get("role"), item.get("content")
        if role not in VALID_ROLES or not isinstance(content, str) or not content.strip():
            return []
        messages.append({"role": role, "content": content})
    return messages


def is_valid_sft_record(row: dict[str, Any]) -> bool:
    messages = normalize_messages(row.get("messages"))
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return False
    answer = messages[-1]["content"].strip()
    return answer.startswith("```sql") and answer.endswith("```")


class SQLSFTProcessor(Preprocessor):
    def __call__(self, rows: Any) -> Any:
        row_list = self.map_col_to_row(rows)
        trajectories = [self.preprocess(row) for row in row_list]
        return self.map_row_to_col(trajectories)

    @staticmethod
    def preprocess(row: dict[str, Any]) -> Trajectory:
        return Trajectory(
            messages=[
                Message(role=message["role"], content=message["content"])
                for message in normalize_messages(row.get("messages"))
            ]
        )


def validate_jsonl(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Training data does not exist: {path}")
    accepted = 0
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
            accepted += int(is_valid_sft_record(row))
    if not total:
        raise ValueError("Training dataset is empty")
    if accepted != total:
        raise ValueError(f"Format gate rejected {total - accepted}/{total} records")


def build_dataset(path: Path, model_id: str, max_length: int) -> Dataset:
    dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=str(path)))
    dataset.filter(is_valid_sft_record)
    try:
        dataset.set_template(
            "Qwen3_5Template",
            model_id=model_id,
            max_length=max_length,
            enable_thinking=False,
        )
    except TypeError:
        dataset.set_template("Qwen3_5Template", model_id=model_id, max_length=max_length)
    dataset.map(SQLSFTProcessor)
    dataset.encode(batched=False)
    return dataset


def build_model(model_id: str, device_mesh: DeviceMesh) -> TransformersModel:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    if hasattr(text_config, "use_cache"):
        text_config.use_cache = False
    model = TransformersModel(
        model_id=model_id,
        config=config,
        device_mesh=device_mesh,
        strategy=os.getenv("MODEL_STRATEGY", "accelerate"),
    )
    decoder_layers = {
        type(module).__name__
        for module in model.model.modules()
        if type(module).__name__.endswith("DecoderLayer")
    }
    model.model._no_split_modules = list(decoder_layers) or ["Qwen3_5DecoderLayer"]
    return kernelize(model)


def main() -> None:
    if not Torch.is_npu_available():
        raise RuntimeError("NPU unavailable: load the CANN and torch_npu environment first")

    model_id = os.getenv("MODEL_ID", "ms://Qwen/Qwen3.5-2B")
    train_path = Path(os.environ["TRAIN_DATASET_ID"]).expanduser().resolve()
    output_dir = Path(os.getenv("OUTPUT_DIR", "artifacts/qwen35-text2sql-lora"))
    num_npus = env_int("NUM_NPUS", 2)
    fsdp_size = env_int("FSDP_SIZE", num_npus)
    dp_size = env_int("DP_SIZE", max(num_npus // fsdp_size, 1))
    batch_size = env_int("BATCH_SIZE", num_npus)
    accumulation = env_int("GRADIENT_ACCUMULATION_STEPS", 8)
    max_length = env_int("MAX_LENGTH", 4096)
    save_interval = env_int("SAVE_INTERVAL", 100)
    log_interval = env_int("LOG_INTERVAL", 10)
    adapter_name = "text2sql_lora"

    if fsdp_size * dp_size != num_npus:
        raise ValueError("FSDP_SIZE * DP_SIZE must equal NUM_NPUS")
    if batch_size < num_npus:
        raise ValueError("BATCH_SIZE must be at least NUM_NPUS")
    validate_jsonl(train_path)

    mesh = DeviceMesh.from_sizes(
        fsdp_size=fsdp_size,
        dp_size=dp_size,
        device_type="npu",
    )
    twinkle.initialize(
        mode="local",
        nproc_per_node=num_npus,
        global_device_mesh=mesh,
    )
    dataset = build_dataset(train_path, model_id, max_length)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, device_mesh=mesh)
    model = build_model(model_id, mesh)
    model.add_adapter_to_model(
        adapter_name,
        LoraConfig(
            r=env_int("LORA_R", 8),
            lora_alpha=env_int("LORA_ALPHA", 32),
            lora_dropout=env_float("LORA_DROPOUT", 0.05),
            target_modules="all-linear",
        ),
        gradient_accumulation_steps=accumulation,
    )
    model.set_optimizer(
        "AdamW",
        lr=env_float("LEARNING_RATE", 1e-4),
        weight_decay=env_float("WEIGHT_DECAY", 0.0),
        foreach=False,
        adapter_name=adapter_name,
    )
    model.set_lr_scheduler(
        scheduler_cls="CosineWarmupScheduler",
        num_warmup_steps=env_int("WARMUP_STEPS", 5),
        num_training_steps=len(dataloader),
        adapter_name=adapter_name,
    )

    LOGGER.info(get_device_placement())
    LOGGER.info(
        "training=%s steps=%s fsdp=%s dp=%s batch=%s",
        train_path,
        len(dataloader),
        fsdp_size,
        dp_size,
        batch_size,
    )
    optimizer = model.optimizer_group[adapter_name]
    for batch in tqdm(dataloader, desc="text2sql-sft"):
        model.forward_backward(inputs=batch, adapter_name=adapter_name)
        model.clip_grad_and_step(adapter_name=adapter_name)
        step = optimizer.cur_step
        if step and step % log_interval == 0:
            LOGGER.info("step=%s metrics=%s", step, model.calculate_metric(True, adapter_name))
        if step and step % save_interval == 0:
            model.save(
                f"checkpoint-{step}",
                output_dir=str(output_dir),
                adapter_name=adapter_name,
                save_optimizer=True,
                consumed_train_samples=dataloader.get_state()["consumed_train_samples"],
            )

    model.save(
        "last-checkpoint",
        output_dir=str(output_dir),
        adapter_name=adapter_name,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()["consumed_train_samples"],
    )


if __name__ == "__main__":
    main()
