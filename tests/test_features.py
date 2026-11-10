import numpy as np

from mfp.features import ENGINEERED, PhysicsFeatures
from mfp.models import build_pipeline


def test_physics_values(reading):
    out = PhysicsFeatures().fit_transform(reading)
    assert np.isclose(out.temp_diff_k[0], 10.0)
    assert np.isclose(out.power_w[0], 40.0 * 1500 * 2 * np.pi / 60)   # ~6283 W
    assert np.isclose(out.strain_minnm[0], 4000.0)
    assert np.isclose(out.strain_ratio[0], 4000.0 / 11000.0)          # batas varian L


def test_overstrain_limit_depends_on_type(reading):
    h = reading.assign(type="H")
    assert PhysicsFeatures().fit_transform(h).strain_ratio[0] < \
        PhysicsFeatures().fit_transform(reading).strain_ratio[0]


def test_disabled_is_identity(reading):
    out = PhysicsFeatures(enabled=False).fit_transform(reading)
    assert list(out.columns) == list(reading.columns)


def test_feature_names_out(reading):
    pf = PhysicsFeatures().fit(reading)
    assert list(pf.get_feature_names_out())[-len(ENGINEERED):] == ENGINEERED


def test_no_leakage_columns_in_model_inputs(cfg, raw_df):
    from mfp.data import xy
    X, y = xy(raw_df.head(500), cfg)
    assert not set(cfg["data"]["failure_modes"]) & set(X.columns)
    pipe = build_pipeline("logreg", cfg).fit(X, y)
    names = set(pipe[:-1].get_feature_names_out())
    assert not names & {"TWF", "HDF", "PWF", "OSF", "RNF", "udi", "product_id"}
