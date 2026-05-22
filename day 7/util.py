from functools import lru_cache
from pathlib import Path
import pickle

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_pickle.pkl"
DATASET_PATH = BASE_DIR / "NarrateIQ_ML_Dataset.xlsx"
TARGET_COLUMN = "Learning_Outcome"
ID_COLUMN = "Student_ID"

CATEGORICAL_COLUMNS = [
    "Age_Group",
    "Disability_Type",
    "Severity_Level",
    "State",
    "School_Type",
    "Language",
    "Teacher_Experience",
    "Input_Mode",
    "Content_Format",
    "Device_Type",
]

BINARY_COLUMNS = ["ISL_Used", "Offline_Mode_Used", "Parental_Involvement"]
NUMERIC_COLUMNS = [
    "Session_Duration_Min",
    "Weekly_Sessions",
    "Months_on_Platform",
    "Internet_Mbps",
    "Engagement_Score",
    "Attention_Span_Min",
    "Completion_Rate_Pct",
    "Comprehension_Score",
    "Teacher_Satisfaction",
    "Improvement_Score_Pct",
]

FIELD_SECTIONS = [
    {
        "title": "Student profile",
        "description": "Demographics and school context used by the saved model.",
        "fields": [
            {"name": "Age_Group", "label": "Age Group", "kind": "select"},
            {"name": "Disability_Type", "label": "Disability Type", "kind": "select"},
            {"name": "Severity_Level", "label": "Severity Level", "kind": "select"},
            {"name": "State", "label": "State", "kind": "select"},
            {"name": "School_Type", "label": "School Type", "kind": "select"},
            {"name": "Language", "label": "Language", "kind": "select"},
            {"name": "Teacher_Experience", "label": "Teacher Experience", "kind": "select"},
        ],
    },
    {
        "title": "Delivery setup",
        "description": "How the content is delivered to the student.",
        "fields": [
            {"name": "Input_Mode", "label": "Input Mode", "kind": "select"},
            {"name": "Content_Format", "label": "Content Format", "kind": "select"},
            {"name": "Device_Type", "label": "Device Type", "kind": "select"},
            {"name": "ISL_Used", "label": "ISL Used", "kind": "binary"},
            {"name": "Offline_Mode_Used", "label": "Offline Mode Used", "kind": "binary"},
        ],
    },
    {
        "title": "Learning signals",
        "description": "Usage and progress values used for prediction.",
        "fields": [
            {"name": "Session_Duration_Min", "label": "Session Duration (Min)", "kind": "number", "step": 1},
            {"name": "Weekly_Sessions", "label": "Weekly Sessions", "kind": "number", "step": 1},
            {"name": "Months_on_Platform", "label": "Months on Platform", "kind": "number", "step": 1},
            {"name": "Internet_Mbps", "label": "Internet Speed (Mbps)", "kind": "number", "step": 0.1},
            {"name": "Parental_Involvement", "label": "Parental Involvement", "kind": "binary"},
            {"name": "Engagement_Score", "label": "Engagement Score", "kind": "number", "step": 0.1},
            {"name": "Attention_Span_Min", "label": "Attention Span (Min)", "kind": "number", "step": 0.1},
            {"name": "Completion_Rate_Pct", "label": "Completion Rate (%)", "kind": "number", "step": 0.1},
            {"name": "Comprehension_Score", "label": "Comprehension Score", "kind": "number", "step": 0.1},
            {"name": "Teacher_Satisfaction", "label": "Teacher Satisfaction", "kind": "number", "step": 0.1},
            {"name": "Improvement_Score_Pct", "label": "Improvement Score (%)", "kind": "number", "step": 0.1},
        ],
    },
]


@lru_cache(maxsize=1)
def load_training_frame() -> pd.DataFrame:
    return pd.read_excel(DATASET_PATH)


@lru_cache(maxsize=1)
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def get_feature_columns() -> list[str]:
    model = load_model()
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    frame = load_training_frame()
    raw_features = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    encoded = pd.get_dummies(raw_features, columns=CATEGORICAL_COLUMNS, drop_first=True)
    return encoded.columns.tolist()


@lru_cache(maxsize=1)
def get_category_levels() -> dict[str, list[str]]:
    frame = load_training_frame()
    return {
        column: sorted(frame[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }


@lru_cache(maxsize=1)
def get_default_values() -> dict[str, float | int | str]:
    frame = load_training_frame()
    defaults: dict[str, float | int | str] = {}

    for column in CATEGORICAL_COLUMNS:
        mode = frame[column].mode(dropna=True)
        defaults[column] = str(mode.iloc[0]) if not mode.empty else get_category_levels()[column][0]

    for column in BINARY_COLUMNS:
        mode = pd.to_numeric(frame[column], errors="coerce").mode(dropna=True)
        defaults[column] = int(mode.iloc[0]) if not mode.empty else 0

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        defaults[column] = float(values.median()) if not values.empty else 0.0

    return defaults


def get_model_name() -> str:
    return type(load_model()).__name__


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_binary(value, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)

    if value is None:
        return default

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return 1
    if normalized in {"0", "false", "no", "n", "off"}:
        return 0

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_sample_frame(payload: dict) -> pd.DataFrame:
    defaults = get_default_values()
    record: dict[str, float | int | str] = {}

    for column in CATEGORICAL_COLUMNS:
        record[column] = str(payload.get(column, defaults[column]))

    for column in BINARY_COLUMNS:
        record[column] = _to_binary(payload.get(column, defaults[column]), int(defaults[column]))

    for column in NUMERIC_COLUMNS:
        record[column] = _to_float(payload.get(column, defaults[column]), float(defaults[column]))

    sample = pd.DataFrame([record])

    for column in CATEGORICAL_COLUMNS:
        sample[column] = pd.Categorical(sample[column].astype(str), categories=get_category_levels()[column])

    encoded = pd.get_dummies(sample, columns=CATEGORICAL_COLUMNS, drop_first=True)
    return encoded.reindex(columns=get_feature_columns(), fill_value=0)


def predict_learning_outcome(payload: dict) -> dict:
    model = load_model()
    sample = build_sample_frame(payload)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    class_probabilities = {
        str(class_name): float(probability)
        for class_name, probability in zip(model.classes_, probabilities, strict=True)
    }

    ranked = sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)
    return {
        "prediction": str(prediction),
        "probabilities": dict(ranked),
        "confidence": float(ranked[0][1]) if ranked else 0.0,
    }