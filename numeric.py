import json
import sys
from pathlib import Path

import joblib
import numpy.core as numpy_core
import numpy.core.numeric as numpy_core_numeric
import numpy.random._pickle as numpy_random_pickle
import pandas as pd
from numpy.random import BitGenerator
from sklearn.compose import _column_transformer
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
METADATA_PATH = BASE_DIR / "house_price_metadata.json"

DEFAULT_NUMERIC_FEATURES = [
    "number of bedrooms",
    "number of bathrooms",
    "living area",
    "lot area",
    "number of floors",
    "number of views",
    "Area of the house(excluding basement)",
    "Area of the basement",
    "Built Year",
    "Renovation Year",
    "Lattitude",
    "Longitude",
    "living_area_renov",
    "lot_area_renov",
    "Number of schools nearby",
    "Distance from the airport",
    "Sale_Year",
    "Sale_Month",
    "waterfront present",
    "condition of the house",
    "grade of the house",
]
DEFAULT_CATEGORICAL_FEATURES = ["Postal Code"]

DEFAULT_VALUES = {
    "number of bedrooms": 3,
    "number of bathrooms": 2,
    "living area": 1800,
    "lot area": 5000,
    "number of floors": 2,
    "number of views": 0,
    "Area of the house(excluding basement)": 1600,
    "Area of the basement": 200,
    "Built Year": 2005,
    "Renovation Year": 0,
    "Lattitude": 47.5,
    "Longitude": -122.2,
    "living_area_renov": 1800,
    "lot_area_renov": 5000,
    "Number of schools nearby": 3,
    "Distance from the airport": 20,
    "Sale_Year": 2014,
    "Sale_Month": 5,
    "waterfront present": 0,
    "condition of the house": 3,
    "grade of the house": 7,
    "Postal Code": "98178",
}


def enable_pickle_compatibility():
    """
    Patch the Python module registry so that pickles produced by NumPy 1.x /
    older scikit-learn builds can be deserialised under NumPy 2.x.

    NumPy 2 moved its private C-extension modules from ``numpy.core.*`` to
    ``numpy._core.*``.  Joblib/pickle resolves module paths recorded in the
    pickle stream at load time; if the old path no longer exists as a real
    module the unpickling fails with an ImportError or an AttributeError about
    a missing ``MT19937`` state.

    We register thin aliases so every old path resolves to its NumPy-2
    equivalent without importing anything that doesn't already exist.
    """
    import numpy as np
    import numpy.core as _np_core
    import numpy._core as _np__core  # noqa: F401 – exists in NumPy ≥ 1.25

    # ------------------------------------------------------------------ #
    # 1. Alias every numpy.core sub-module that may appear in a pickle.   #
    # ------------------------------------------------------------------ #
    _core_submodules = [
        "multiarray", "numeric", "umath", "fromnumeric",
        "function_base", "shape_base", "arrayprint", "defchararray",
        "records", "memmap", "numerictypes", "getlimits",
        "overrides", "_methods", "_exceptions", "_add_newdocs",
        "_add_newdocs_scalars", "_dtype", "_dtype_ctypes", "_internal",
        "_multiarray_umath",
    ]
    for _name in _core_submodules:
        _old_key = f"numpy.core.{_name}"
        _new_key = f"numpy._core.{_name}"
        if _old_key not in sys.modules:
            try:
                import importlib as _il
                _mod = _il.import_module(_new_key)
            except (ImportError, ModuleNotFoundError):
                try:
                    _mod = _il.import_module(_old_key)
                except (ImportError, ModuleNotFoundError):
                    continue
            sys.modules.setdefault(_old_key, _mod)
            sys.modules.setdefault(_new_key, _mod)

    # Top-level aliases used by very old pickles.
    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
    sys.modules.setdefault("numpy.core._multiarray_umath",
                            np.core._multiarray_umath)

    # ------------------------------------------------------------------ #
    # 2. numpy.random aliases (MT19937 / BitGenerator pickle paths).      #
    # ------------------------------------------------------------------ #
    _rand_submodules = [
        "_pickle", "mtrand", "bit_generator",
        "_common", "_bounded_integers", "_generator",
    ]
    for _name in _rand_submodules:
        for _prefix in ("numpy.random", "numpy._core.random"):
            _key = f"{_prefix}.{_name}"
            if _key not in sys.modules:
                try:
                    import importlib as _il
                    sys.modules.setdefault(_key, _il.import_module(f"numpy.random.{_name}"))
                except (ImportError, ModuleNotFoundError):
                    pass

    # ------------------------------------------------------------------ #
    # 3. Fix the BitGenerator constructor used when unpickling RNGs.      #
    # ------------------------------------------------------------------ #
    _original_ctor = numpy_random_pickle.__bit_generator_ctor

    def _compat_bit_generator_ctor(value="MT19937"):
        # NumPy 2 passes the class itself; NumPy 1 passed a string name.
        if isinstance(value, type) and issubclass(value, BitGenerator):
            return value()
        if isinstance(value, str):
            return _original_ctor(value)
        if hasattr(value, "__name__"):
            return _original_ctor(value.__name__)
        return _original_ctor(str(value))

    numpy_random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor

    # ------------------------------------------------------------------ #
    # 4. sklearn internal class missing in newer sklearn builds.          #
    # ------------------------------------------------------------------ #
    if not hasattr(_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _column_transformer._RemainderColsList = _RemainderColsList


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing .pkl model at {MODEL_PATH}. Save your trained sklearn pipeline there first."
        )
    if MODEL_PATH.stat().st_size == 0:
        raise ValueError(
            f"Model file is empty: {MODEL_PATH}. Re-export the trained pipeline from your notebook."
        )

    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    try:
        enable_pickle_compatibility()
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        raise ValueError(f"Could not load pickle model: {error}") from error
    numeric_features = metadata.get("numeric_features", DEFAULT_NUMERIC_FEATURES)
    categorical_features = metadata.get("categorical_features", DEFAULT_CATEGORICAL_FEATURES)
    target_name = metadata.get("target_name", "Price")

    return {
        "model": model,
        "metadata": metadata,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_name": target_name,
    }


def build_input_frame(numeric_features, categorical_features):
    values = {}

    st.subheader("Property Details")
    col_left, col_right = st.columns(2)

    for index, feature in enumerate(numeric_features):
        default_value = float(DEFAULT_VALUES.get(feature, 0.0))
        target_col = col_left if index % 2 == 0 else col_right
        values[feature] = target_col.number_input(
            feature,
            value=default_value,
            step=1.0,
            format="%.4f",
        )

    for feature in categorical_features:
        values[feature] = st.text_input(feature, value=str(DEFAULT_VALUES.get(feature, "")))

    return pd.DataFrame([values])


def prepare_features(input_df, artifacts):
    ordered_columns = artifacts["numeric_features"] + artifacts["categorical_features"]
    return input_df[ordered_columns].copy()


def predict_price(input_df, artifacts):
    prepared = prepare_features(input_df, artifacts)
    prediction = artifacts["model"].predict(prepared)
    return float(prediction.ravel()[0])


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("House Price Predictor")
st.caption("Streamlit UI for the house-price model saved as a `.pkl` pipeline.")

try:
    artifacts = load_artifacts()
except Exception as error:
    st.error(f"Failed to load model artifacts: {error}")
    st.stop()

with st.sidebar:
    st.subheader("Model Info")
    st.write(f"Model file: `{MODEL_PATH.name}`")
    st.write(f"Target: `{artifacts['target_name']}`")
    if artifacts["metadata"].get("notes"):
        st.info(artifacts["metadata"]["notes"])
    else:
        st.info(
            "This app expects the saved sklearn pipeline in `house_price_model.pkl`."
        )

input_df = build_input_frame(
    artifacts["numeric_features"],
    artifacts["categorical_features"],
)

predict_clicked = st.button("Predict Price", use_container_width=True, type="primary")

if predict_clicked:
    predicted_price = predict_price(input_df, artifacts)

    st.subheader("Prediction")
    st.metric("Estimated Price", f"{predicted_price:,.2f}")

    with st.expander("Input summary"):
        st.dataframe(input_df, use_container_width=True)
