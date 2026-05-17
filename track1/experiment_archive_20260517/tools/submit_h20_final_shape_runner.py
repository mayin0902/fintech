import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


BENCH_RE = re.compile(
    r"^\s*(?P<seq>\d+)\s+\|\s+(?P<peak>OOM|[0-9.]+)\s+\|\s+"
    r"(?P<avg>--|[0-9.]+)\s+\|\s+(?P<min>--|[0-9.]+)\s+\|\s+(?P<max>--|[0-9.]+)"
)

CSV_FIELDS = [
    "timestamp",
    "phase",
    "cycle",
    "job",
    "gpu",
    "candidate",
    "solution",
    "seq_len",
    "returncode",
    "passed",
    "peak_mb",
    "avg_ms",
    "min_ms",
    "max_ms",
    "elapsed_s",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_benchmark(stdout):
    rows = []
    for line in stdout.splitlines():
        match = BENCH_RE.match(line)
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


def run_cmd(cmd, env, cwd, timeout_s):
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


def env_for(base_env, gpu, candidate, cache_dir):
    env = dict(base_env)
    for key in [
        "MOE_SHARED_CHUNK_SIZE",
        "MOE_EXPERT_CHUNK_SIZE",
        "MOE_USE_CHECKPOINT",
        "MOE_COMPILE_MODE",
        "MOE_COMPILE_TORCH_MODE",
        "MOE_DISABLE_COMPILE",
        "MOE_COMPILE_DYNAMIC",
        "MOE_COMPILE_FULLGRAPH",
    ]:
        if candidate.get("unset_env", False):
            env.pop(key, None)
    env.update({k: str(v) for k, v in candidate.get("env", {}).items()})
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir / candidate["name"])
    return env


def append_jsonl(path, record, lock):
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(path, rows, lock):
    with lock:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)


def run_correctness(official_dir, out_dir, py, base_env, cache_dir, lock, phase, job, gpu, candidate):
    cmd = [
        py,
        "correctness_check.py",
        "--solution",
        candidate["solution"],
        "--seq-len",
        str(job["seq_len"]),
    ]
    started = utc_now()
    rc, stdout, elapsed_s = run_cmd(
        cmd,
        env_for(base_env, gpu, candidate, cache_dir),
        official_dir,
        timeout_s=int(job.get("timeout_s", 2400)),
    )
    passed = rc == 0 and "所有检查项通过" in stdout
    append_jsonl(
        out_dir / f"{phase}_{job['name']}.jsonl",
        {
            "timestamp": started,
            "phase": phase,
            "job": job,
            "gpu": gpu,
            "candidate": candidate,
            "cmd": cmd,
            "returncode": rc,
            "passed": passed,
            "elapsed_s": elapsed_s,
            "stdout_tail": stdout[-12000:],
        },
        lock,
    )
    return passed


def run_benchmark(official_dir, out_dir, py, base_env, cache_dir, lock, phase, cycle, job, gpu, candidate):
    cmd = [
        py,
        "benchmark.py",
        "--solution",
        candidate["solution"],
        "--seq-lens",
        str(job["seq_lens"]),
        "--warmup",
        str(job["warmup"]),
        "--measure",
        str(job["measure"]),
    ]
    started = utc_now()
    rc, stdout, elapsed_s = run_cmd(
        cmd,
        env_for(base_env, gpu, candidate, cache_dir),
        official_dir,
        timeout_s=int(job.get("timeout_s", 3600)),
    )
    parsed = parse_benchmark(stdout)
    append_jsonl(
        out_dir / f"{phase}_{job['name']}.jsonl",
        {
            "timestamp": started,
            "phase": phase,
            "cycle": cycle,
            "job": job,
            "gpu": gpu,
            "candidate": candidate,
            "cmd": cmd,
            "returncode": rc,
            "elapsed_s": elapsed_s,
            "stdout_tail": stdout[-12000:],
        },
        lock,
    )
    if not parsed:
        parsed = [
            {
                "seq_len": str(job["seq_lens"]),
                "peak_mb": None,
                "avg_ms": None,
                "min_ms": None,
                "max_ms": None,
            }
        ]
    rows = []
    for row in parsed:
        passed = rc == 0 and row["peak_mb"] is not None and row["avg_ms"] is not None
        rows.append(
            {
                "timestamp": started,
                "phase": phase,
                "cycle": cycle,
                "job": job["name"],
                "gpu": gpu,
                "candidate": candidate["name"],
                "solution": candidate["solution"],
                "seq_len": row["seq_len"],
                "returncode": rc,
                "passed": passed,
                "peak_mb": "" if row["peak_mb"] is None else row["peak_mb"],
                "avg_ms": "" if row["avg_ms"] is None else row["avg_ms"],
                "min_ms": "" if row["min_ms"] is None else row["min_ms"],
                "max_ms": "" if row["max_ms"] is None else row["max_ms"],
                "elapsed_s": f"{elapsed_s:.3f}",
            }
        )
    append_csv(out_dir / f"{phase}_{job['name']}.csv", rows, lock)


def read_phase_rows(out_dir, phase):
    rows = []
    for path in out_dir.glob(f"{phase}_*.csv"):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def write_summary(out_dir, phase, cycle):
    rows = [
        row
        for row in read_phase_rows(out_dir, phase)
        if row.get("passed") == "True" and row.get("avg_ms")
    ]
    path = out_dir / f"{phase}_summary.md"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {phase} Summary\n\n")
        f.write(f"Updated: `{utc_now()}`\n\n")
        f.write(f"Cycles completed: `{cycle}`\n\n")
        if not rows:
            f.write("No successful benchmark rows yet.\n")
            return
        grouped = {}
        for row in rows:
            key = (row["job"], str(row["seq_len"]), row["candidate"])
            grouped.setdefault(key, []).append(row)
        current = None
        for (job, seq_len, candidate), group in sorted(
            grouped.items(),
            key=lambda item: (
                item[0][0],
                int(item[0][1]) if item[0][1].isdigit() else 10**12,
                statistics.mean(float(r["avg_ms"]) for r in item[1]),
            ),
        ):
            section = (job, seq_len)
            if section != current:
                if current is not None:
                    f.write("\n")
                current = section
                f.write(f"## {job} seq_len {seq_len}\n\n")
                f.write("| Candidate | Runs | Peak max | Avg mean | Avg std | Avg min | Avg max |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            avgs = [float(r["avg_ms"]) for r in group]
            peaks = [float(r["peak_mb"]) for r in group]
            f.write(
                f"| {candidate} | {len(group)} | {max(peaks):.2f} | "
                f"{statistics.mean(avgs):.3f} | "
                f"{(statistics.pstdev(avgs) if len(avgs) > 1 else 0.0):.3f} | "
                f"{min(avgs):.3f} | {max(avgs):.3f} |\n"
            )


def run_benchmark_job(official_dir, out_dir, py, base_env, cache_dir, lock, phase, cycle, job, candidates, gpu):
    for name in job["candidates"]:
        run_benchmark(
            official_dir,
            out_dir,
            py,
            base_env,
            cache_dir,
            lock,
            phase,
            cycle,
            job,
            gpu,
            candidates[name],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--phase-name", required=True)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--duration-minutes", type=float, default=0.0)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    official_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "torchinductor_cache"
    cache_dir.mkdir(exist_ok=True)
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    base_env = dict(os.environ)
    lock = threading.Lock()

    with Path(args.plan_json).open(encoding="utf-8") as f:
        plan = json.load(f)
    candidates = {item["name"]: item for item in plan["candidates"]}
    (out_dir / f"{args.phase_name}_config.json").write_text(
        json.dumps({"created_at": utc_now(), "phase": args.phase_name, "plan": plan}, indent=2),
        encoding="utf-8",
    )

    for job in plan.get("correctness_jobs", []):
        gpu = gpus[int(job.get("gpu_slot", 0)) % len(gpus)]
        for name in job["candidates"]:
            run_correctness(
                official_dir,
                out_dir,
                args.python,
                base_env,
                cache_dir,
                lock,
                args.phase_name,
                job,
                gpu,
                candidates[name],
            )

    bench_jobs = plan.get("benchmark_jobs", [])
    cycle = 0
    if bench_jobs:
        deadline = time.time() + args.duration_minutes * 60
        while cycle == 0 or (args.duration_minutes > 0 and time.time() < deadline):
            threads = []
            for job in bench_jobs:
                gpu = gpus[int(job.get("gpu_slot", 0)) % len(gpus)]
                threads.append(
                    threading.Thread(
                        target=run_benchmark_job,
                        args=(
                            official_dir,
                            out_dir,
                            args.python,
                            base_env,
                            cache_dir,
                            lock,
                            args.phase_name,
                            cycle,
                            job,
                            candidates,
                            gpu,
                        ),
                    )
                )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            cycle += 1
            write_summary(out_dir, args.phase_name, cycle)

    write_summary(out_dir, args.phase_name, cycle)
    print(json.dumps({"out_dir": str(out_dir), "phase": args.phase_name, "cycles": cycle}, ensure_ascii=False))


if __name__ == "__main__":
    main()
