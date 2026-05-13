import csv
import sys
from pathlib import Path


def read_rows(run_dir):
    rows = []
    for path in Path(run_dir).glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("task") == "benchmark" and row.get("passed") == "True":
                    rows.append(row)
    return rows


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python summarize_search.py <run_dir>")

    run_dir = Path(sys.argv[1])
    rows = read_rows(run_dir)
    print(f"# Track 1 Search Summary\n")
    print(f"Run directory: `{run_dir}`\n")
    if not rows:
        print("No successful benchmark rows yet.")
        return

    by_seq = {}
    for row in rows:
        seq_len = int(row["seq_len"])
        by_seq.setdefault(seq_len, []).append(row)

    for seq_len in sorted(by_seq):
        candidates = by_seq[seq_len]
        best_mem = min(candidates, key=lambda r: to_float(r["peak_mb"]) or float("inf"))
        best_speed = min(candidates, key=lambda r: to_float(r["avg_ms"]) or float("inf"))
        print(f"## SeqLen {seq_len}\n")
        print("| Objective | Variant | GPU | Peak MB | Avg ms | Shared | Expert | Checkpoint |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        for label, row in [("lowest_memory", best_mem), ("fastest", best_speed)]:
            print(
                f"| {label} | {row['variant']} | {row['gpu']} | "
                f"{row['peak_mb']} | {row['avg_ms']} | {row['shared_chunk']} | "
                f"{row['expert_chunk']} | {row['checkpoint']} |"
            )
        print()


if __name__ == "__main__":
    main()

