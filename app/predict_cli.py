"""Batch scoring dari CSV (format kolom asli AI4I maupun snake_case).

    python app/predict_cli.py --input data/raw/machine_failure.csv --output preds.csv
    python app/predict_cli.py --input new_batch.csv --output preds.csv --drift-report drift.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mfp.data import COLUMN_MAP, SchemaError  # noqa: E402
from mfp.inference import FEATURE_COLUMNS, ModelBundle  # noqa: E402
from mfp.monitoring import drift_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=str(ROOT / "models" / "model_bundle.joblib"))
    ap.add_argument("--drift-report", help="simpan laporan PSI vs data training")
    args = ap.parse_args()

    df = pd.read_csv(args.input).rename(columns=COLUMN_MAP)
    bundle = ModelBundle.load(args.model)
    try:
        preds = bundle.predict(df)
    except SchemaError as e:
        print(f"[ERROR] Data tidak valid: {e}", file=sys.stderr)
        return 2

    keep = [c for c in ("udi", "product_id") if c in df.columns]
    result = df[keep + FEATURE_COLUMNS].join(preds)
    result.to_csv(args.output, index=False)
    counts = result["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    print(f"{len(result):,} unit diskor -> {args.output}")
    print("  " + " | ".join(f"{k}: {v:,}" for k, v in counts.items()))

    if args.drift_report:
        ref = pd.read_csv(Path(args.model).parent / "reference_data.csv")
        rep = drift_report(ref, df, FEATURE_COLUMNS[1:], ["type"])
        rep.to_csv(args.drift_report, index=False)
        flagged = rep[rep["status"] != "stabil"]
        print(f"Drift: {len(flagged)} fitur tidak stabil" +
              (f" -> {', '.join(flagged.feature)}" if len(flagged) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
