"""Membuat paket `mfp` dapat di-import tanpa `pip install -e .`."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", category=FutureWarning)


def to_markdown(df, path, floatfmt=".4f", index=False):
    """Simpan DataFrame sebagai CSV + tabel Markdown (tanpa dependensi tabulate)."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=index)
    d = df.reset_index() if index else df.copy()

    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "|".join("---" for _ in d.columns) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.values) + " |")
    md = "\n".join(lines) + "\n"
    path.with_suffix(".md").write_text(md, encoding="utf-8")
    return md


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
