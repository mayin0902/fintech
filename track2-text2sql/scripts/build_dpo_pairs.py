#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from text2sql_feedback.post_training import build_preference_pairs


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO pairs from scored SQL rollouts")
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-margin", type=float, default=0.25)
    parser.add_argument("--min-chosen-reward", type=float, default=0.9)
    args = parser.parse_args()

    pairs = build_preference_pairs(
        read_jsonl(args.rollouts),
        min_margin=args.min_margin,
        min_chosen_reward=args.min_chosen_reward,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
