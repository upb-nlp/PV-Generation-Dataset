"""Gradient-boosted tree forecasters and the model registry."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

Predictor = Callable[..., np.ndarray]
ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], Predictor]

XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}

LIGHTGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "objective": "regression",
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

WARM_START_ESTIMATORS = 100


def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def _named(X: np.ndarray):
    """LightGBM records feature names at fit time and warns when predict gets a
    bare array, so give it the same named frame on both sides."""
    import pandas as pd

    values = np.asarray(X)
    return pd.DataFrame(values, columns=[f"f{i}" for i in range(values.shape[1])])


def xgboost(**overrides: Any) -> ModelFn:
    params = {**XGBOOST_PARAMS, **overrides}

    def model_fn(X_train, y_train, w_train):
        from xgboost import XGBRegressor

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train)

        def predict_fn(X_test, timestamps=None):
            return _clip(model.predict(X_test))

        return predict_fn

    return model_fn


def lightgbm(**overrides: Any) -> ModelFn:
    params = {**LIGHTGBM_PARAMS, **overrides}

    def model_fn(X_train, y_train, w_train):
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(**params)
        model.fit(_named(X_train), y_train, sample_weight=w_train)

        def predict_fn(X_test, timestamps=None):
            return _clip(model.predict(_named(X_test)))

        return predict_fn

    return model_fn


def xgboost_warm(update_estimators: int = WARM_START_ESTIMATORS, **overrides: Any) -> ModelFn:
    params = {**XGBOOST_PARAMS, **overrides}
    booster: list = [None]

    def model_fn(X_train, y_train, w_train):
        from xgboost import XGBRegressor

        started = booster[0] is not None
        model = XGBRegressor(
            **{**params, "n_estimators": update_estimators if started else params["n_estimators"]}
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            **({"xgb_model": booster[0]} if started else {}),
        )
        booster[0] = model.get_booster()

        def predict_fn(X_test, timestamps=None):
            return _clip(model.predict(X_test))

        return predict_fn

    return model_fn


def lightgbm_warm(update_estimators: int = WARM_START_ESTIMATORS, **overrides: Any) -> ModelFn:
    params = {**LIGHTGBM_PARAMS, **overrides}
    booster: list = [None]

    def model_fn(X_train, y_train, w_train):
        from lightgbm import LGBMRegressor

        started = booster[0] is not None
        model = LGBMRegressor(
            **{**params, "n_estimators": update_estimators if started else params["n_estimators"]}
        )
        model.fit(_named(X_train), y_train, sample_weight=w_train, init_model=booster[0])
        booster[0] = model.booster_

        def predict_fn(X_test, timestamps=None):
            return _clip(model.predict(_named(X_test)))

        return predict_fn

    return model_fn


TREE_MODELS: dict[str, Callable[..., ModelFn]] = {
    "xgboost": xgboost,
    "lightgbm": lightgbm,
    "xgboost_warm": xgboost_warm,
    "lightgbm_warm": lightgbm_warm,
}

DEEP_MODELS = (
    "vanilla_lstm",
    "cnn_lstm",
    "attention_lstm",
    "transformer",
    "nbeats",
    "transformer_warm",
)

MODEL_NAMES = tuple(TREE_MODELS) + DEEP_MODELS

DEFAULT_MODELS = (
    "xgboost",
    "lightgbm",
    "vanilla_lstm",
    "cnn_lstm",
    "attention_lstm",
    "transformer",
    "nbeats",
)


def get_model(name: str, **kwargs: Any) -> ModelFn:
    if name in TREE_MODELS:
        return TREE_MODELS[name](**kwargs)
    if name in DEEP_MODELS:
        try:
            from . import deep
        except ImportError as exc:  # torch is optional
            raise ImportError(
                f"model '{name}' needs PyTorch. Install it with 'pip install torch', "
                f"or use one of {sorted(TREE_MODELS)}."
            ) from exc

        return deep.get_model(name, **kwargs)
    raise KeyError(f"unknown model '{name}'; expected one of {sorted(MODEL_NAMES)}")
