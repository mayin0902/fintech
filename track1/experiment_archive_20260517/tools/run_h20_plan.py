import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


BENCH_ROW_RE = re.compile(
    r"^\s*(?P<seq>\d+)\s+\|\s+"
    r"(?P<peak>OOM|[0-9.]+)\s+\|\s+"
    r"(?P<avg>--|[0-9.]+)\s+\|\s+"
    r"(?P<min>--|[0-9.]+)\s+\|\s+"
    r"(?P<max>--|[0-9.]+)"
)


H20_MAIN_VARIANTS = [
    {"name": "nocp_s32768_e32768", "shared": 32768, "expert": 32768, "checkpoint": 0},
    {"name": "nocp_s65536_e65536", "shared": 65536, "expert": 65536, "checkpoint": 0},
    {"name": "nocp_s131072_e131072", "shared": 131072, "expert": 131072, "checkpoint": 0},
    {"name": "nocp_s196608_e196608", "shared": 196608, "expert": 196608, "checkpoint": 0},
    {"name": "nocp_s262144_e262144", "shared": 262144, "expert": 262144, "checkpoint": 0},
    {"name": "ckpt_s8192_e8192", "shared": 8192, "expert": 8192, "checkpoint": 1},
    {"name": "ckpt_s16384_e16384", "shared": 16384, "expert": 16384, "checkpoint": 1},
    {"name": "ckpt_s32768_e32768", "shared": 32768, "expert": 32768, "checkpoint": 1},
    {"name": "ckpt_s65536_e65536", "shared": 65536, "expert": 65536, "checkpoint": 1},
    {"name": "ckpt_s65536_e131072", "shared": 65536, "expert": 131072, "checkpoint": 1},
    {"name": "ckpt_s131072_e131072", "shared": 131072, "expert": 131072, "checkpoint": 1},
]


CSV_FIELDS = [
    "timestamp",
    "phase",
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
]


@dataclass(frozen=True)
class CommandSpec:
    task: str
    seq_lens: list[int]
    warmup: int = 0
    measure: int = 0
    timeout_s: int = 1800


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_seq_lens(raw):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_variant_name(name):
    checkpoint = 1 if name.startswith("ckpt_") else 0
    match = re.search(r"_s(?P<shared>\d+)_e(?P<expert>\d+)$", name)
    if not match:
        raise ValueError(f"cannot parse variant name: {name}")
    return {
        "name": name,
        "shared": int(match.group("shared")),
        "expert": int(match.group("expert")),
        "checkpoint": checkpoint,
    }


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


def run_with_timeout(cmd, env, cwd, timeout_s):
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout, time.time() - start
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return 124, output + f"\n[TIMEOUT after {timeout_s}s]\n", time.time() - start


def build_env(base_env, gpu, variant):
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MOE_SHARED_CHUNK_SIZE"] = str(variant["shared"])
    env["MOE_EXPERT_CHUNK_SIZE"] = str(variant["expert"])
    env["MOE_USE_CHECKPOINT"] = str(variant["checkpoint"])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def append_jsonl(path, record, lock):
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(path, row, lock):
    with lock:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def write_task_rows(csv_path, lock, phase, worker, gpu, variant, spec, started, returncode, passed, elapsed_s, stdout):
    if spec.task == "benchmark":
        bench_rows = parse_benchmark(stdout)
        if not bench_rows:
            append_csv(
                csv_path,
                base_csv_row(phase, worker, gpu, variant, spec, started, returncode, False, elapsed_s),
                lock,
            )
            return
        for bench_row in bench_rows:
            row = base_csv_row(
                phase,
                worker,
                gpu,
                variant,
                spec,
                started,
                returncode,
                passed and bench_row["peak_mb"] is not None,
                elapsed_s,
            )
            row.update(
                {
                    "seq_len": bench_row["seq_len"],
                    "peak_mb": "" if bench_row["peak_mb"] is None else bench_row["peak_mb"],
                    "avg_ms": "" if bench_row["avg_ms"] is None else bench_row["avg_ms"],
                    "min_ms": "" if bench_row["min_ms"] is None else bench_row["min_ms"],
                    "max_ms": "" if bench_row["max_ms"] is None else bench_row["max_ms"],
                }
            )
            append_csv(csv_path, row, lock)
        return

    row = base_csv_row(phase, worker, gpu, variant, spec, started, returncode, passed, elapsed_s)
    row["seq_len"] = spec.seq_lens[0]
    append_csv(csv_path, row, lock)


def base_csv_row(phase, worker, gpu, variant, spec, timestamp, returncode, passed, elapsed_s):
    return {
        "timestamp": timestamp,
        "phase": phase,
        "worker": worker,
        "gpu": gpu,
        "variant": variant["name"],
        "task": spec.task,
        "seq_len": ",".join(str(seq_len) for seq_len in spec.seq_lens),
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
    }


def command_for_spec(python, spec):
    if spec.task == "correctness":
        return [
            python,
            "correctness_check.py",
            "--solution",
            "solution.py",
            "--seq-len",
            str(spec.seq_lens[0]),
        ]
    return [
        python,
        "benchmark.py",
        "--solution",
        "solution.py",
        "--seq-lens",
        ",".join(str(seq_len) for seq_len in spec.seq_lens),
        "--warmup",
        str(spec.warmup),
        "--measure",
        str(spec.measure),
    ]


def run_variant(official_dir, out_dir, python, phase, gpu, worker, variant, specs, lock):
    jsonl_path = out_dir / f"{phase}.jsonl"
    csv_path = out_dir / f"{phase}.csv"
    env = build_env(os.environ, gpu, variant)
    append_jsonl(
        jsonl_path,
        {
            "timestamp": utc_now(),
            "event": "variant_start",
            "phase": phase,
            "worker": worker,
            "gpu": gpu,
            "variant": variant,
        },
        lock,
    )

    variant_passed = True
    for spec in specs:
        if spec.task == "benchmark" and not variant_passed:
            break

        cmd = command_for_spec(python, spec)
        started = utc_now()
        returncode, stdout, elapsed_s = run_with_timeout(
            cmd,
            env=env,
            cwd=official_dir,
            timeout_s=spec.timeout_s,
        )
        passed = returncode == 0 and (
            spec.task == "benchmark" or "总结: 所有检查项通过" in stdout
        )
        if spec.task == "correctness" and not passed:
            variant_passed = False

        append_jsonl(
            jsonl_path,
            {
                "timestamp": started,
                "event": "task_done",
                "phase": phase,
                "worker": worker,
                "gpu": gpu,
                "variant": variant,
                "task": spec.task,
                "seq_lens": spec.seq_lens,
                "cmd": cmd,
                "returncode": returncode,
                "passed": passed,
                "elapsed_s": elapsed_s,
                "stdout_tail": stdout[-6000:],
            },
            lock,
        )
        write_task_rows(
            csv_path,
            lock,
            phase,
            worker,
            gpu,
            variant,
            spec,
            started,
            returncode,
            passed,
            elapsed_s,
            stdout,
        )

    append_jsonl(
        jsonl_path,
        {
            "timestamp": utc_now(),
            "event": "variant_stop",
            "phase": phase,
            "worker": worker,
            "gpu": gpu,
            "variant": variant,
            "passed": variant_passed,
        },
        lock,
    )


def run_phase(official_dir, out_dir, python, phase, gpus, variants, specs):
    lock = threading.Lock()
    tasks = queue.Queue()
    for variant in variants:
        tasks.put(variant)

    def worker(gpu):
        worker_name = f"gpu{gpu}"
        while True:
            try:
                variant = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                run_variant(official_dir, out_dir, python, phase, gpu, worker_name, variant, specs, lock)
            finally:
                tasks.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def read_benchmark_rows(csv_path, phase=None, seq_len=None):
    rows = []
    if not csv_path.exists():
        return rows
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("task") != "benchmark" or row.get("passed") != "True":
                continue
            if phase is not None and row.get("phase") != phase:
                continue
            if seq_len is not None and int(row["seq_len"]) != seq_len:
                continue
            rows.append(row)
    return rows


def is_dominated(row, other):
    row_mem = float(row["peak_mb"])
    row_time = float(row["avg_ms"])
    other_mem = float(other["peak_mb"])
    other_time = float(other["avg_ms"])
    return (
        other_mem <= row_mem
        and other_time <= row_time
        and (other_mem < row_mem or other_time < row_time)
    )


def pareto_rows(rows):
    front = []
    for row in rows:
        if any(is_dominated(row, other) for other in rows if other is not row):
            continue
        front.append(row)
    return sorted(front, key=lambda row: (float(row["avg_ms"]), float(row["peak_mb"])))


def stress_selection(rows, limit=2):
    front = pareto_rows(rows)
    selected = []
    seen = set()
    if front:
        fastest = min(front, key=lambda row: float(row["avg_ms"]))
        lowest_memory = min(front, key=lambda row: float(row["peak_mb"]))
        for row in (fastest, lowest_memory):
            if row["variant"] not in seen:
                selected.append(row)
                seen.add(row["variant"])
    for row in front:
        if len(selected) >= limit:
            break
        if row["variant"] not in seen:
            selected.append(row)
            seen.add(row["variant"])
    return selected


def write_summary(out_dir, main_rows, refine_rows, stress_rows):
    summary_path = out_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# H20 Plan Summary\n\n")
        f.write(f"Run directory: `{out_dir}`\n\n")
        for title, rows in [
            ("128K Main", main_rows),
            ("Pareto Refine", refine_rows),
            ("Stress", stress_rows),
        ]:
            f.write(f"## {title}\n\n")
            if not rows:
                f.write("No successful benchmark rows.\n\n")
                continue
            f.write("| Variant | SeqLen | GPU | Peak MB | Avg ms | Min ms | Max ms | Shared | Expert | Checkpoint |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in sorted(rows, key=lambda r: (int(r["seq_len"]), float(r["avg_ms"]), float(r["peak_mb"]))):
                f.write(
                    f"| {row['variant']} | {row['seq_len']} | {row['gpu']} | "
                    f"{row['peak_mb']} | {row['avg_ms']} | {row['min_ms']} | {row['max_ms']} | "
                    f"{row['shared_chunk']} | {row['expert_chunk']} | {row['checkpoint']} |\n"
                )
            f.write("\n")

        front = pareto_rows([row for row in main_rows if int(row["seq_len"]) == 131072])
        f.write("## 128K Pareto Front\n\n")
        if not front:
            f.write("No Pareto front available.\n")
        else:
            f.write("| Variant | Peak MB | Avg ms | Shared | Expert | Checkpoint |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for row in front:
                f.write(
                    f"| {row['variant']} | {row['peak_mb']} | {row['avg_ms']} | "
                    f"{row['shared_chunk']} | {row['expert_chunk']} | {row['checkpoint']} |\n"
                )


def main():
    parser = argparse.ArgumentParser(description="Run the H20/A800 final experiment plan.")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU ids.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to track1/h20_runs/<timestamp>.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--refine-limit", type=int, default=5)
    args = parser.parse_args()

    official_dir = Path(__file__).resolve().parent
    repo_track_dir = official_dir.parent
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = repo_track_dir / "h20_runs" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_h20_plan")
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus = parse_seq_lens(args.gpus)
    env_path = out_dir / "environment.txt"
    env_cmd = [
        args.python,
        "-c",
        (
            "import sys, torch\n"
            "print('python', sys.version)\n"
            "print('torch', torch.__version__)\n"
            "print('cuda', torch.version.cuda)\n"
            "print('available', torch.cuda.is_available())\n"
            "print('device_count', torch.cuda.device_count())\n"
            "import subprocess\n"
            "print(subprocess.check_output(['nvidia-smi'], text=True))\n"
        ),
    ]
    returncode, stdout, _ = run_with_timeout(env_cmd, os.environ, official_dir, 120)
    env_path.write_text(stdout if returncode == 0 else f"returncode={returncode}\n{stdout}", encoding="utf-8")

    main_specs = [
        CommandSpec("correctness", [8192], timeout_s=1200),
        CommandSpec("benchmark", [131072], warmup=2, measure=3, timeout_s=3600),
    ]
    run_phase(official_dir, out_dir, args.python, "main_128k", gpus, H20_MAIN_VARIANTS, main_specs)

    main_rows = read_benchmark_rows(out_dir / "main_128k.csv", phase="main_128k", seq_len=131072)
    front = pareto_rows(main_rows)
    refine_variants = [parse_variant_name(row["variant"]) for row in front[: args.refine_limit]]

    refine_rows = []
    stress_rows = []
    if not args.main_only and refine_variants:
        refine_specs = [
            CommandSpec("correctness", [8192], timeout_s=1200),
            CommandSpec("benchmark", [8192, 32768, 65536, 131072], warmup=5, measure=10, timeout_s=7200),
        ]
        run_phase(official_dir, out_dir, args.python, "pareto_refine", gpus, refine_variants, refine_specs)
        refine_rows = read_benchmark_rows(out_dir / "pareto_refine.csv", phase="pareto_refine")

        if not args.skip_stress:
            refined_128k = [row for row in refine_rows if int(row["seq_len"]) == 131072]
            stress_variants = [parse_variant_name(row["variant"]) for row in stress_selection(refined_128k)]
            if stress_variants:
                stress_specs = [
                    CommandSpec("benchmark", [196608, 262144], warmup=1, measure=3, timeout_s=7200),
                ]
                run_phase(official_dir, out_dir, args.python, "stress_long", gpus, stress_variants, stress_specs)
                stress_rows = read_benchmark_rows(out_dir / "stress_long.csv", phase="stress_long")

    write_summary(out_dir, main_rows, refine_rows, stress_rows)
    print(out_dir)


if __name__ == "__main__":
    main()
