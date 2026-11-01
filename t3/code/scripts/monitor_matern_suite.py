"""Read-only training monitor; writes only a separate monitoring directory.

No CUDA/PyTorch import, model reload, training restart, or gate modification.
Run under a lingering user systemd service, not a client terminal.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_matches(pid, root):
    try:
        directory = Path("/proc") / str(int(pid))
        stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return stat[0] != "Z" and str(root).encode() in (directory / "cmdline").read_bytes()
    except (OSError, ValueError, TypeError):
        return False


def committed_history(directory):
    marker = directory / "committed_epoch.json"
    if not marker.exists():
        return [], None
    before = marker.read_bytes()
    committed = int(json.loads(before)["completed_epochs"])
    raw = (directory / "metrics.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    if marker.read_bytes() != before:
        raise RuntimeError("Checkpoint commit changed during statistics read; retry next poll")
    selected = []
    for row in rows:
        epoch = int(row["epoch"])
        if epoch > committed:
            continue
        if any(value is None for value in row.values()):
            raise RuntimeError("Partially written metrics row; retry next poll")
        numeric = {key: float(value) for key, value in row.items()}
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("Nonfinite committed metric")
        numeric["epoch"] = epoch
        selected.append(numeric)
    if [r["epoch"] for r in selected] != list(range(1, committed + 1)):
        raise RuntimeError("Committed history is incomplete; retry next poll")
    return selected, hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def summarize(rows, epoch, interval=100):
    prefix = [r for r in rows if r["epoch"] <= epoch]
    if not prefix or prefix[-1]["epoch"] != epoch:
        raise ValueError("Cannot report an epoch that was not observed")
    block = prefix[max(0, epoch - interval):]
    best = min(prefix, key=lambda r: r["val_db_rmse"])
    start = block[0]
    end = prefix[-1]
    return dict(epoch=epoch, block_first_epoch=start["epoch"], block_last_epoch=epoch,
                last=end, best_val_db_rmse=best["val_db_rmse"], best_epoch=best["epoch"],
                block_mean_train_loss=statistics.mean(r["train_loss"] for r in block),
                block_mean_val_db_rmse=statistics.mean(r["val_db_rmse"] for r in block),
                block_val_rmse_change_db=end["val_db_rmse"] - start["val_db_rmse"],
                block_val_ssim_change=end["val_ssim"] - start["val_ssim"],
                fine_wavelet_and_feedback_gate="not_evaluated_at_intermediate_epochs",
                uncertainty="one_training_seed; no independent-seed confidence interval")


def table(items):
    lines = ["| Variant | Epoch | Train loss | Val RMSE (dB) | Val MAE (dB) | Val SSIM | Best RMSE (epoch) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in sorted(items.items()):
        row = item["last"]
        lines.append(f"| {name} | {item['epoch']} | {row['train_loss']:.6f} | {row['val_db_rmse']:.4f} | "
                     f"{row['val_db_mae']:.4f} | {row['val_ssim']:.6f} | {item['best_val_db_rmse']:.4f} ({item['best_epoch']}) |")
    return "\n".join(lines)


class Monitor:
    def __init__(self, root, interval=100, stall_seconds=900):
        self.root = root.resolve()
        self.output = self.root / "monitoring"
        self.output.mkdir(exist_ok=True)
        self.interval, self.stall_seconds = interval, stall_seconds
        self.events = self.output / "events"
        self.events.mkdir(exist_ok=True)
        self.previous_alerts = set()
        existing = self.output / "health.json"
        if existing.exists():
            self.previous_alerts = set(json.loads(existing.read_text()).get("alerts", []))

    def event(self, key, payload):
        target = self.events / f"{key}.json"
        if target.exists():
            return
        record = dict(recorded_at=utc(), **payload)
        atomic(target, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def poll(self):
        suite = json.loads((self.root / "suite_status.json").read_text())
        terminal = suite["status"] in {"complete", "failed_no_further_launch"}
        alerts, latest, observed, rows_by_job = [], {}, {}, {}
        if not terminal and not process_matches(suite.get("pid"), self.root):
            alerts.append("suite_supervisor_missing_or_pid_reused")
        if suite["status"] == "failed_no_further_launch":
            alerts.append("suite_failed_no_further_launch")
        for name, job in suite["jobs"].items():
            if name == "suite":
                continue
            directory = self.root / name
            phase = job["phase"]
            info = dict(phase=phase, pid=job.get("pid"), gpus=job.get("gpus"))
            if phase in {"training", "smoke", "evaluate_val", "evaluate_test"}:
                info["process_alive"] = process_matches(job.get("pid"), self.root)
                if not info["process_alive"] and not terminal:
                    alerts.append(f"{name}:process_missing")
                progress_path = directory / ("committed_epoch.json" if phase == "training" else phase + ".log")
                if progress_path.exists():
                    age = max(0., time.time() - progress_path.stat().st_mtime)
                    info["seconds_since_progress"] = round(age, 1)
                    if age > self.stall_seconds and not terminal:
                        alerts.append(f"{name}:no_progress_for_{self.stall_seconds}s")
            if phase == "failed":
                alerts.append(f"{name}:failed_{job.get('failed_phase', 'unknown')}")
            try:
                rows, digest = committed_history(directory)
                if rows:
                    rows_by_job[name] = rows
                    info["committed_epoch"] = rows[-1]["epoch"]
                    latest[name] = summarize(rows, rows[-1]["epoch"], self.interval)
                    for epoch in range(self.interval, rows[-1]["epoch"] + 1, self.interval):
                        report = self.output / name / f"epoch_{epoch:04d}.json"
                        if report.exists():
                            continue
                        summary = summarize(rows, epoch, self.interval)
                        # Prefix hash, not the current checkpoint: no epoch-N checkpoint is fabricated.
                        prefix_hash = hashlib.sha256(json.dumps(rows[:epoch], sort_keys=True).encode()).hexdigest()
                        config = json.loads((directory / "config.json").read_text())
                        record = dict(job=name, recorded_at=utc(), config_sha256=config["config_sha256"],
                                      history_prefix_sha256=prefix_hash, **summary)
                        atomic(report, record)
                        atomic(report.with_suffix(".md"), f"# {name} — Epoch {epoch}\n\n" + table({name: summary}) +
                            f"\n\nMean train loss in this interval: {summary['block_mean_train_loss']:.6f}; "
                            f"Val RMSE change: {summary['block_val_rmse_change_db']:+.4f} dB.\n\n"
                            "These are statistics from existing logs, not new validation or test results. Intermediate RMSE does not determine the 16x16 extension.\n")
                        self.event(f"{name.replace('/', '_')}_epoch_{epoch:04d}", dict(kind="epoch_summary", job=name, epoch=epoch, report=str(report)))
                    if phase == "complete":
                        final = self.output / name / "final_training_summary.json"
                        if not final.exists():
                            atomic(final, dict(job=name, recorded_at=utc(), history_sha256=digest, **latest[name]))
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                info["read_warning"] = str(error)
                alerts.append(f"{name}:history_read_warning:{type(error).__name__}")
            observed[name] = info
        # A cross-variant table is emitted only when all four reached the SAME milestone.
        names = [f"8x8/{v}" for v in ("B2", "M1", "M2", "M3")]
        if all(name in rows_by_job for name in names):
            common = min(rows_by_job[name][-1]["epoch"] for name in names)
            for epoch in range(self.interval, common + 1, self.interval):
                path = self.output / "comparisons" / f"8x8_epoch_{epoch:04d}.md"
                if path.exists():
                    continue
                items = {name: summarize(rows_by_job[name], epoch, self.interval) for name in names}
                content = f"# Matched-epoch 8x8 comparison: {epoch}\n\n" + table(items)
                content += "\n\nPairwise differences (second minus first; lower RMSE is better):\n\n"
                for a, b in (("B2", "M1"), ("M1", "M2"), ("M2", "M3")):
                    delta = items[f"8x8/{b}"]["last"]["val_db_rmse"] - items[f"8x8/{a}"]["last"]["val_db_rmse"]
                    content += f"- {a} -> {b}: {delta:+.4f} dB.\n"
                content += "\nThese are validation metrics from one training seed only; conclusions about detail recovery, feedback, or extension await the full preregistered evaluation.\n"
                atomic(path.with_suffix(".json"), dict(epoch=epoch, recorded_at=utc(), experiments=items))
                atomic(path, content)
                self.event(f"comparison_8x8_epoch_{epoch:04d}", dict(kind="matched_epoch_comparison", epoch=epoch, report=str(path)))
        disk = shutil.disk_usage(self.root)
        if disk.free < 20 * 1024 ** 3:
            alerts.append("disk_free_below_20GiB")
        try:
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv"], text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as error:
            gpu = str(error)
            alerts.append("gpu_telemetry_unavailable")
        current_alerts = set(alerts)
        if current_alerts != self.previous_alerts:
            key = f"alerts_{time.time_ns()}"
            self.event(key, dict(kind="health_change", alerts=sorted(current_alerts), resolved=sorted(self.previous_alerts - current_alerts)))
        self.previous_alerts = current_alerts
        health = dict(updated_at=utc(), monitor_pid=os.getpid(), suite_status=suite["status"],
                      jobs=observed, alerts=sorted(current_alerts), free_disk_gib=round(disk.free / 1024 ** 3, 2),
                      gpu=gpu, interval_epochs=self.interval, stall_seconds=self.stall_seconds,
                      notification_delivery="server_files_and_systemd_journal_only", training_mutations="none")
        atomic(self.output / "health.json", health)
        report = f"# Matérn experiment monitor\n\nUpdated: {health['updated_at']}\n\nSuite status: {suite['status']}\n\n"
        report += table(latest) if latest else "No committed epoch yet."
        report += "\n\nAlerts: " + ("; ".join(health["alerts"]) if alerts else "None")
        report += "\n\nStatistics are saved every 100 epochs per group; a comparison is generated when all four reach the same checkpoint. Latest metrics from different epochs are not a fair ranking."
        report += "\n\nOnly the monitoring directory is written; training, checkpoints, and the 16x16 gate are unchanged. Results remain on the server if the client is offline; chat delivery is not guaranteed.\n"
        atomic(self.output / "latest.md", report)
        if terminal:
            self.event("suite_terminal", dict(kind="suite_terminal", suite_status=suite["status"], alerts=health["alerts"]))
        return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    monitor = Monitor(args.root)
    with (monitor.output / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            terminal = monitor.poll()
            if terminal or args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
