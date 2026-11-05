"""Recreate the paired Beam response summary from the published results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [
    ("angle_seen", "Seen angle"),
    ("angle_interpolation", "Angle interpolation"),
    ("angle_extrapolation", "Angle extrapolation"),
    ("array_seen_angle", "Array, seen angle"),
    ("array_unseen_angle", "Array, unseen angle"),
]

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "legend.fontsize": 8, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.bbox": "tight",
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    summary = json.loads((ROOT / "results" / "beam_response" / "summary.json")
                         .read_text(encoding="utf-8"))
    model_key = ",".join(map(str, summary["seeds"]))
    by_group = {r["group"]: r for r in summary["groups"] if r["models"] == model_key}
    values = [by_group[key] for key, _ in GROUPS]
    labels = [label for _, label in GROUPS]
    y = np.arange(len(labels))
    height = 0.24
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    ax1.barh(y - height / 2, [r["truth_abs_change_db"] for r in values], height,
             color="#27638c", label="Ground truth")
    ax1.barh(y + height / 2, [r["prediction_abs_change_db"] for r in values], height,
             color="#e4a543", label="A prediction")
    ax1.set_xlabel("Mean absolute change (dB)")
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_title("(a) Magnitude of configuration response")
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    offsets = (-height, 0, height)
    names = ("A prediction", "Zero change", "Beam difference")
    metrics = ("response_mae_db", "zero_mae_db", "beam_mae_db")
    colors = ("#27638c", "#929da5", "#e4a543")
    for offset, name, metric, color in zip(offsets, names, metrics, colors):
        ax2.barh(y + offset, [r[metric] for r in values], height, label=name, color=color)
    mean = np.array([r["response_mae_db"] for r in values])
    lo = np.array([r["response_mae_ci95_db"][0] for r in values])
    hi = np.array([r["response_mae_ci95_db"][1] for r in values])
    ax2.errorbar(mean, y - height, xerr=np.vstack((mean - lo, hi - mean)), fmt="none",
                 ecolor="#142d3b", capsize=2, lw=0.8)
    ax2.set_xlabel("Response MAE (dB; lower is better)")
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_title("(b) Accuracy of the predicted change")
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    for axis in (ax1, ax2):
        axis.grid(axis="x", alpha=0.18, lw=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / f"beam_response_summary.{ext}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
