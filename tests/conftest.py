import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfp.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_df(cfg):
    from mfp.data import load_raw
    return load_raw(cfg.path("raw_data"))


@pytest.fixture
def reading():
    return pd.DataFrame([{
        "type": "L", "air_temp_k": 300.0, "process_temp_k": 310.0,
        "rot_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100,
    }])


@pytest.fixture(scope="session")
def small_bundle(cfg, raw_df):
    """Bundle kecil yang dilatih cepat — tes tidak bergantung pada artefak `make all`."""
    from sklearn.ensemble import RandomForestClassifier
    from mfp.data import xy
    from mfp.features import PhysicsFeatures, build_preprocessor, feature_lists
    from mfp.inference import ModelBundle
    from mfp.models import build_pipeline
    from sklearn.pipeline import Pipeline

    df = raw_df.sample(3000, random_state=0)
    X, y = xy(df, cfg)
    pipe = build_pipeline("random_forest", cfg, engineered=True)
    pipe.set_params(model__n_estimators=50)
    pipe.fit(X, y)
    num, cat = feature_lists(cfg, True)
    modes = ["TWF", "HDF", "PWF", "OSF"]
    mm = Pipeline([("physics", PhysicsFeatures()), ("prep", build_preprocessor(num, cat)),
                   ("model", RandomForestClassifier(n_estimators=30, random_state=0))])
    mm.fit(X, df[modes].to_numpy())
    return ModelBundle(pipeline=pipe, threshold=0.2, risk_bands={"low": 0.1, "high": 0.5},
                       mode_model=mm, mode_labels=modes, metadata={"model_version": "test"})
