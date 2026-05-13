import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = [
    {"name": "ckpt_s2048_e2048", "shared": 2048, "expert": 2048, "checkpoint": 1},
    {"name": "ckpt_s4096_e2048", "shared": 4096, "expert": 2048, "checkpoint": 1},
    {"name": "ckpt_s8192_e2048", "shared": 8192, "expert": 2048, "checkpoint": 1},
    {"name": "ckpt_s4096_e4096", "shared": 4096, "expert": 4096, "checkpoint": 1},
    {"name": "ckpt_s8192_e4096", "shared": 8192, "expert": 4096, "checkpoint": 1},
    {"name": "ckpt_s16384_e4096", "shared": 16384, "expert": 4096, "checkpoint": 1},
    {"name": "ckpt_s8192_e8192", "shared": 8192, "expert": 8192, "checkpoint": 1},
    {"name": "nocp_s2048_e2048", "shared": 2048, "expert": 2048, "checkpoint": 0},
    {"name": "nocp_s4096_e2048", "shared": 4096, "expert": 2048, "checkpoint": 0},
    {"name": "nocp_s4096_e4096", "shared": 4096, "expert": 4096, "checkpoint": 0},
    {"name": "nocp_s8192_e4096", "shared": 8192, "expert": 4096, "checkpoint": 0},
    {"name": "nocp_s8192_e8192", "shared": 8192, "expert": 8192, "checkpoint": 0},
    {"name": "ckpt_s1024_e1024", "shared": 1024, "expert": 1024, "checkpoint": 1},
    {"name": "ckpt_s2048_e1024", "shared": 2048, "expert": 1024, "checkpoint": 1},
    {"name": "ckpt_s4096_e1024", "shared": 4096, "expert": 1024, "checkpoint": 1},
]


BENCH_ROW_RE = re.compile(
    r"^\s*(?P<seq>\d+)\s+\|\s+"
    r"(?P<peak>OOM|[0-9.]+)\s+\|\s+"
    r"(?P<avg>--|[0-9.]+)\s+\|\s+"
    r"(?P<min>--|[0-9.]+)\s+\|\s+"
    r"(?P<max>--|[0-9.]+)"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "worker",
                "gpu",
                "variant",
                "task",
                "seq_len",
                "shared_chunk",
                "expert_chunk",
                "checkpoint",
                "returncode",
                "passed",
                "peak_mb",
                "avg_ms",
                "min_ms",
                "max_ms",
                "elapsed_s",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_command(args, env, cwd, timeout_s):
    start = time.time()
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout, time.time() - start


def run_with_timeout(args, env, cwd, timeout_s):
    try:
        return run_command(args, env, cwd, timeout_s)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return 124, output + f"\n[TIMEOUT after {timeout_s}s]\n", timeout_s


def parse_benchmark(stdout):
    rows = []
    for line in stdout.splitlines():
        match = BENCH_ROW_RE.match(line)
        if not match:
            continue
        data = match.groupdict()
        row = {"seq_len": int(data["seq"])}
        if data["peak"] == "OOM":
            row.update({"peak_mb": None, "avg_ms": None, "min_ms": None, "max_ms": None})
        else:
            row.update(
                {
                    "peak_mb": float(data["peak"]),
                    "avg_ms": float(data["avg"]),
                    "min_ms": float(data["min"]),
                    "max_ms": float(data["max"]),
                }
            )
        rows.append(row)
    return rows


def command_plan(profile):
    if profile == "correctness":
        return [
            ("correctness", ["2048"], 600),
            ("correctness", ["8192"], 900),
            ("benchmark", ["8192", "32768"], 1200),
            ("benchmark", ["65536"], 1500),
        ]
    if profile == "benchmark":
        return [
            ("correctness", ["2048"], 600),
            ("benchmark", ["2048", "8192", "32768"], 1200),
            ("benchmark", ["65536"], 1500),
            ("benchmark", ["131072"], 2400),
        ]
    raise ValueError(f"unknown profile: {profile}")


def build_env(base_env, gpu, variant):
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MOE_SHARED_CHUNK_SIZE"] = str(variant["shared"])
    env["MOE_EXPERT_CHUNK_SIZE"] = str(variant["expert"])
    env["MOE_USE_CHECKPOINT"] = str(variant["checkpoint"])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--profile", choices=["correctness", "benchmark"], required=True)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    official_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"{args.worker}.jsonl"
    csv_path = out_dir / f"{args.worker}.csv"
    deadline = time.time() + args.hours * 3600
    variants = list(VARIANTS)
    if args.profile == "benchmark":
        variants = list(reversed(variants))

    append_jsonl(
        jsonl_path,
        {
            "timestamp": utc_now(),
            "event": "worker_start",
            "worker": args.worker,
            "gpu": args.gpu,
            "profile": args.profile,
            "hours": args.hours,
            "pid": os.getpid(),
        },
    )

    cycle = 0
    while True:
        for variant in variants:
            if time.time() >= deadline:
                append_jsonl(
                    jsonl_path,
                    {
                        "timestamp": utc_now(),
                        "event": "worker_stop",
                        "worker": args.worker,
                        "reason": "deadline",
                        "cycle": cycle,
                    },
                )
                return

            env = build_env(os.environ, args.gpu, variant)
            variant_passed = True
            for task, seq_lens, base_timeout in command_plan(args.profile):
                remaining = deadline - time.time()
                if remaining < 300:
                    return
                timeout_s = int(min(base_timeout, max(300, remaining - 30)))
                if task == "correctness":
                    cmd = [
                        args.python,
                        "correctness_check.py",
                        "--solution",
                        "solution.py",
                        "--seq-len",
                        seq_lens[0],
                    ]
                else:
                    if not variant_passed:
                        break
                    cmd = [
                        args.python,
                        "benchmark.py",
                        "--solution",
                        "solution.py",
                        "--seq-lens",
                        ",".join(seq_lens),
                        "--warmup",
                        "1",
                        "--measure",
                        "1",
                    ]

                started = utc_now()
                returncode, stdout, elapsed_s = run_with_timeout(
                    cmd,
                    env=env,
                    cwd=official_dir,
                    timeout_s=timeout_s,
                )
                passed = returncode == 0 and (
                    task == "benchmark" or "总结: 所有检查项通过" in stdout
                )
                if task == "correctness" and not passed:
                    variant_passed = False

                record = {
                    "timestamp": started,
                    "event": "task_done",
                    "worker": args.worker,
                    "gpu": args.gpu,
                    "profile": args.profile,
                    "cycle": cycle,
                    "variant": variant,
                    "task": task,
                    "seq_lens": seq_lens,
                    "cmd": cmd,
                    "returncode": returncode,
                    "passed": passed,
                    "elapsed_s": elapsed_s,
                    "stdout_tail": stdout[-6000:],
                }
                append_jsonl(jsonl_path, record)

                if task == "benchmark":
                    rows = parse_benchmark(stdout)
                    if not rows:
                        append_csv(
                            csv_path,
                            {
                                "timestamp": started,
                                "worker": args.worker,
                                "gpu": args.gpu,
                                "variant": variant["name"],
                                "task": task,
                                "seq_len": ",".join(seq_lens),
                                "shared_chunk": variant["shared"],
                                "expert_chunk": variant["expert"],
                                "checkpoint": variant["checkpoint"],
                                "returncode": returncode,
                                "passed": passed,
                                "peak_mb": "",
                                "avg_ms": "",
                                "min_ms": "",
                                "max_ms": "",
                                "elapsed_s": f"{elapsed_s:.3f}",
                            },
                        )
                    for bench_row in rows:
                        append_csv(
                            csv_path,
                            {
                                "timestamp": started,
                                "worker": args.worker,
                                "gpu": args.gpu,
                                "variant": variant["name"],
                                "task": task,
                                "seq_len": bench_row["seq_len"],
                                "shared_chunk": variant["shared"],
                                "expert_chunk": variant["expert"],
                                "checkpoint": variant["checkpoint"],
                                "returncode": returncode,
                                "passed": passed and bench_row["peak_mb"] is not None,
                                "peak_mb": "" if bench_row["peak_mb"] is None else bench_row["peak_mb"],
                                "avg_ms": "" if bench_row["avg_ms"] is None else bench_row["avg_ms"],
                                "min_ms": "" if bench_row["min_ms"] is None else bench_row["min_ms"],
                                "max_ms": "" if bench_row["max_ms"] is None else bench_row["max_ms"],
                                "elapsed_s": f"{elapsed_s:.3f}",
                            },
                        )
                else:
                    append_csv(
                        csv_path,
                        {
                            "timestamp": started,
                            "worker": args.worker,
                            "gpu": args.gpu,
                            "variant": variant["name"],
                            "task": task,
                            "seq_len": seq_lens[0],
                            "shared_chunk": variant["shared"],
                            "expert_chunk": variant["expert"],
                            "checkpoint": variant["checkpoint"],
                            "returncode": returncode,
                            "passed": passed,
                            "peak_mb": "",
                            "avg_ms": "",
                            "min_ms": "",
                            "max_ms": "",
                            "elapsed_s": f"{elapsed_s:.3f}",
                        },
                    )
        cycle += 1


if __name__ == "__main__":
    main()

