"""Gaya visual konsisten untuk seluruh figur laporan.

Palet kategorikal tervalidasi colorblind-safe (urutan slot tetap, tidak
di-cycle), sequential satu hue (biru), diverging biru <-> merah dengan titik
tengah abu-abu. Teks memakai warna tinta, bukan warna seri.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b"}
# Warna kelas: identitas tetap di seluruh laporan.
CLASS_COLORS = {0: SERIES[0], 1: SERIES[1]}
CLASS_NAMES = {0: "Normal", 1: "Failure"}

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a8292a"])


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 22,
        "axes.labelcolor": INK["secondary"], "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.prop_cycle": plt.cycler(color=SERIES),
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        "text.color": INK["primary"], "legend.frameon": False,
        "lines.linewidth": 2.0, "patch.linewidth": 0,
    })


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def subtitle(ax, text: str) -> None:
    """Subjudul (kesimpulan utama grafik) di bawah judul."""
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9,
            color=INK["secondary"], va="bottom")

# Warna tetap per model (warna mengikuti entitas, bukan peringkat).
MODEL_COLORS = {
    "random_forest": SERIES[0], "hist_gb": SERIES[1], "xgboost": SERIES[2],
    "logreg_spline": SERIES[3], "logreg": SERIES[4], "ensemble": SERIES[6],
}
