"""Durable local supervisor: four parallel 2-GPU runs, then a validation gate.

Use --root for a unique suite. No retries or tuning after failed evidence.
Reinvocation may resume committed checkpoints of the SAME experiments.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASET = Path("/home/ka/CKM/data/raw/49ca1dcebe2caa2b2112e6c862132243a992b00a/Dataset")
MANIFESTS = REPO / "runs/width32_b128_20260908T052359Z/manifests"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def snapshot(root):
    receipt = root / "source_snapshot.json"
    destination = root / "code"
    if receipt.exists():
        records = json.loads(receipt.read_text())
        for name, digest in records.items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Frozen source changed: {name}")
        return destination
    if destination.exists():
        raise RuntimeError("Incomplete source snapshot; use a new suite root")
    records = {}
    for directory, dirs, files in os.walk(REPO):
        dirs[:] = sorted(d for d in dirs if d not in {"runs", "work", "figures", ".git", "__pycache__", ".venv", ".pytest_cache"})
        for name in sorted(files):
            source = Path(directory) / name
            if source.suffix not in {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".sh"}:
                continue
            relative = source.relative_to(REPO)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic(receipt, records)
    return destination


class Supervisor:
    def __init__(self, root, code):
        self.root, self.code = root, code
        self.lock = threading.Lock()
        self.state = {"status": "preparing", "started_at": now(), "pid": os.getpid(), "jobs": {}}

    def update(self, job, phase, **extra):
        with self.lock:
            self.state["jobs"][job] = {"phase": phase, "updated_at": now(), **extra}
            self.state["updated_at"] = now()
            atomic(self.root / "suite_status.json", self.state)
            print(json.dumps({"job": job, "phase": phase, **extra}), flush=True)

    def command(self, array, variant):
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                str(self.code / "run_matern_ablation.py"), "--dataset-root", str(DATASET),
                "--manifest-path", str(self.root / "manifests" / f"manifest_samefreq_6.7ghz_{array}_0deg.jsonl"),
                "--height-stats-path", str(self.root / "manifests/height_stats_train.json"),
                "--run-root", str(self.root / array / variant), "--array-size", array, "--variant", variant]

    def execute(self, array, variant, gpus, phase, extra):
        directory = self.root / array / variant
        directory.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "OMP_NUM_THREADS": "2",
               "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        env.pop("PYTHONPATH", None)
        command = self.command(array, variant) + extra
        with (directory / f"{phase}.log").open("a") as log:
            process = subprocess.Popen(command, cwd=self.code, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.update(f"{array}/{variant}", phase, pid=process.pid, gpus=gpus, command=command)
            status = process.wait()
        if status:
            self.update(f"{array}/{variant}", "failed", failed_phase=phase, exit_code=status)
            raise RuntimeError(f"{array}/{variant}: {phase} exited with {status}")

    def smoke(self, array, variant, gpus):
        receipt = self.root / array / variant / "_smoke/smoke_verified.json"
        if not receipt.exists():
            self.execute(array, variant, gpus, "smoke", ["--smoke-optimizer-steps", "2"])
        self.update(f"{array}/{variant}", "smoke_passed", gpus=gpus)

    def run(self, array, variant, gpus):
        runtime = self.root / array / variant / "training_runtime.json"
        if not runtime.exists() or json.loads(runtime.read_text())["status"] != "complete":
            self.execute(array, variant, gpus, "training", ["--resume", "auto"])
        for split in ("val", "test"):
            self.execute(array, variant, gpus, f"evaluate_{split}", ["--evaluate", split])
        self.update(f"{array}/{variant}", "complete", gpus=gpus)

    def parallel(self, action):
        failures = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(action, "8x8", variant, [2 * i, 2 * i + 1]): variant
                       for i, variant in enumerate(("B2", "M1", "M2", "M3"))}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures.append(str(error))
        if failures:
            raise RuntimeError("; ".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detach", action="store_true", help="Launch a durable local supervisor and return its PID")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (root / "supervisor.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root)],
                cwd=REPO, start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(supervisor_pid=process.pid, root=str(root))), flush=True)
        return
    with (root / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = snapshot(root)
        manifest_root = root / "manifests"
        manifest_root.mkdir(exist_ok=True)
        for name in ("manifest_samefreq_6.7ghz_8x8_0deg.jsonl", "manifest_samefreq_6.7ghz_16x16_0deg.jsonl",
                     "height_stats_train.json", "scene_split_seed42.json"):
            source, target = MANIFESTS / name, manifest_root / name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise RuntimeError("Pinned manifest mismatch")
            if not target.exists():
                shutil.copy2(source, target)
        protocol = code / "docs/MATERN_ENCODER_PILOT.md"
        if not (root / "protocol.md").exists():
            shutil.copy2(protocol, root / "protocol.md")
        atomic(root / "environment.json", dict(python=sys.executable, captured_at=now(),
            packages=subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
            gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used", "--format=csv"], text=True),
            protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest()))
        suite = Supervisor(root, code)
        try:
            suite.state["status"] = "smoke_testing"
            suite.parallel(suite.smoke)
            suite.state["status"] = "running_8x8"
            suite.parallel(suite.run)
            results = {name: json.loads((root / "8x8" / name / "val_diagnostics.json").read_text())
                       for name in ("B2", "M1", "M2", "M3")}
            # Import gate implementation from the immutable snapshot, not mutable working code.
            import importlib.util
            spec = importlib.util.spec_from_file_location("frozen_gate", code / "evaluation/matern_ablation_metrics.py")
            gate_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gate_module)
            gate = gate_module.validation_gate(results)
            atomic(root / "gate.json", {**gate, "evaluated_at": now(), "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()})
            if gate["launch_16x16_m3"]:
                suite.state["status"] = "running_conditional_16x16"
                marker = root / "16x16_M3_authorized.json"
                if not marker.exists():
                    atomic(marker, dict(authorized_at=now(), training_seed=42, distinct_runs_allowed=1,
                                       reason="8x8_validation_gate_passed"))
                suite.smoke("16x16", "M3", [0, 1])
                suite.run("16x16", "M3", [0, 1])
            else:
                suite.update("16x16/M3", "skipped", reason="8x8_validation_did_not_support_extension")
            suite.state["status"] = "complete"
            suite.update("suite", "complete")
        except Exception:
            suite.state["status"] = "failed_no_further_launch"
            suite.update("suite", "failed", traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
