import pytest

from mfp.data import SchemaError, audit_quality, split_data, validate_features


def test_columns_renamed(raw_df):
    assert {"air_temp_k", "torque_nm", "machine_failure", "type"} <= set(raw_df.columns)
    assert not any("[" in c for c in raw_df.columns)


def test_raw_data_passes_schema(raw_df):
    validate_features(raw_df, require_target=True)


@pytest.mark.parametrize("col,value", [("torque_nm", -5.0), ("air_temp_k", 1000.0),
                                       ("type", "X")])
def test_schema_rejects_invalid(reading, col, value):
    bad = reading.copy()
    bad[col] = value
    with pytest.raises(SchemaError):
        validate_features(bad)


def test_schema_rejects_missing_column(reading):
    with pytest.raises(SchemaError, match="Kolom wajib"):
        validate_features(reading.drop(columns="torque_nm"))


def test_quality_audit_finds_known_label_anomalies(raw_df, cfg):
    q = audit_quality(raw_df, cfg)
    assert q.n_rows == 10_000 and q.n_missing == 0
    assert len(q.failure_without_mode) == 9
    assert q.mode_without_failure == {"RNF": 18}


def test_split_is_stratified_and_disjoint(raw_df, cfg):
    tr, te = split_data(raw_df, cfg)
    assert len(tr) + len(te) == len(raw_df)
    assert set(tr.udi).isdisjoint(te.udi)
    assert abs(tr.machine_failure.mean() - te.machine_failure.mean()) < 0.002
