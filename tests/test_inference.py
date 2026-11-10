import numpy as np
import pytest

from mfp.data import SchemaError
from mfp.inference import ModelBundle


def test_predict_output_contract(small_bundle, raw_df):
    out = small_bundle.predict(raw_df.head(200))
    expected = {"failure_probability", "predicted_failure", "risk_level",
                "recommended_action", "likely_failure_mode", "p_TWF", "p_OSF"}
    assert expected <= set(out.columns)
    assert out.failure_probability.between(0, 1).all()
    assert set(out.risk_level) <= {"LOW", "MEDIUM", "HIGH"}
    # unit yang di-flag tidak boleh berlabel LOW
    assert (out.loc[out.predicted_failure == 1, "risk_level"] != "LOW").all()


def test_save_load_roundtrip(small_bundle, raw_df, tmp_path):
    path = small_bundle.save(tmp_path / "b.joblib")
    loaded = ModelBundle.load(path)
    a = small_bundle.predict(raw_df.head(50)).failure_probability
    b = loaded.predict(raw_df.head(50)).failure_probability
    assert np.allclose(a, b)


def test_high_risk_scenario_scores_higher(small_bundle, reading):
    normal = small_bundle.predict(reading).failure_probability[0]
    # daya sangat tinggi + keausan tinggi -> PWF/OSF
    risky = reading.assign(rot_speed_rpm=1250, torque_nm=72.0, tool_wear_min=220)
    assert small_bundle.predict(risky).failure_probability[0] > normal


def test_invalid_input_rejected(small_bundle, reading):
    with pytest.raises(SchemaError):
        small_bundle.predict(reading.assign(rot_speed_rpm=-1))
