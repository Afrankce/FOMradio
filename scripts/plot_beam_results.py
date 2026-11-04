"""Rebuild the descriptive figure from the checked-in experiment summaries."""
from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "beam_matern"
summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
paired = json.loads((DATA / "paired_stats.json").read_text(encoding="utf-8"))
groups = ["seen_config", "array", "angle_interpolation", "angle_extrapolation", "joint"]
labels = ["Seen configuration", "Array transfer", "Angle interpolation", "Angle extrapolation", "Array + angle"]
models = ["A", "uniform_fixed", "beam_fixed", "uniform_learned", "beam_learned"]
model_labels = ["A", "Uniform / fixed", "Beam / fixed", "Uniform / learned", "Beam / learned"]
colors = ["#337b91", "#ce7854"]

mpl.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig = plt.figure(figsize=(12, 5.8))
grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], left=.16, right=.98,
                        top=.88, bottom=.27, wspace=.46)
ax = fig.add_subplot(grid[0])
mae = np.array([[summary["groups"][g]["models"][m]["mae_db"] for m in models] for g in groups])
delta = mae[:, 1:] - mae[:, [0]]
# Color encodes deviation within each task; labels always show absolute MAE.
im = ax.imshow(np.column_stack([np.zeros(len(groups)), delta]), cmap="BrBG_r",
               vmin=-.15, vmax=.15, aspect="auto")
for row in range(len(groups)):
    for col in range(len(models)):
        ax.text(col, row, f"{mae[row, col]:.3f}", ha="center", va="center", fontsize=10)
ax.set_xticks(range(len(models)), model_labels, rotation=30, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(groups)), labels)
ax.tick_params(length=0, pad=6)
ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(groups), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.8)
ax.tick_params(which="minor", bottom=False, left=False)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("a  Mean absolute error (dB, lower is better)", loc="left", fontweight="bold", pad=12)
cax = fig.add_axes([.20, .09, .28, .024])
cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-.15, 0, .15])
cbar.set_label("Color: change relative to A (dB)", labelpad=3)
cbar.outline.set_visible(False)

ax = fig.add_subplot(grid[1])
for i, (key, legend) in enumerate([("beam_fixed-minus-uniform_fixed", "Fixed scales"),
                                    ("beam_learned-minus-uniform_learned", "Learned scales")]):
    y = np.arange(len(groups)) + (-.12 if i == 0 else .12)
    dat = [paired["groups"][g][key] for g in groups]
    value = np.array([x["mae_delta_db"] for x in dat])
    limits = np.array([x["scene_bootstrap_95_interval_db"] for x in dat])
    ax.errorbar(value, y, xerr=np.vstack((value-limits[:, 0], limits[:, 1]-value)),
                marker="o" if i == 0 else "s", markersize=5, capsize=3, linewidth=1.2,
                color=colors[i], label=legend, linestyle="none")
ax.axvline(0, color="#676767", linewidth=.8, linestyle="--")
ax.set_yticks(range(len(groups)), labels)
ax.invert_yaxis()
ax.set_xlabel("Beam sampling − uniform sampling, MAE (dB)")
ax.set_title("b  Paired scene-level contrasts", loc="left", fontweight="bold", pad=12)
ax.grid(axis="x", color="#dddddd", linewidth=.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.19), ncol=2, fontsize=9)
fig.text(.59, .06, "Intervals: 95% scene bootstrap; one trained seed, four draws",
         ha="left", va="top", fontsize=8.5, color="#4b4b4b")
for ext in ("png", "pdf", "svg"):
    output = ROOT / "assets" / f"beam_matern_results.{ext}"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    if ext == "svg":
        # Matplotlib SVG paths can contain trailing spaces and Windows CRLFs.
        svg = output.read_text(encoding="utf-8")
        output.write_bytes(("\n".join(line.rstrip() for line in svg.splitlines()) + "\n").encode("utf-8"))
plt.close(fig)
