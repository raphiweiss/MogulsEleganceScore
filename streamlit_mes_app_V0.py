from __future__ import annotations

import re
import textwrap
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter


# ============================================================
# App-Konfiguration
# ============================================================
st.set_page_config(
    page_title="MES Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = APP_DIR / "model_results" / "yolov8n_HD"
DATA_DIR = RESULTS_DIR / "data"
VIDEO_DIR = RESULTS_DIR / "videos"

POSES_PATH = DATA_DIR / "all_poses.csv"
SCORES_PATH = DATA_DIR / "mes_scores.csv"
FIS_SCORES_PATH = DATA_DIR / "fis_scores.csv"

DEFAULT_FPS = 25
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".m4v"}

METRICS = {
    "rhythmus": {"label": "Rhythmus", "y": "x (px)"},
    "symmetrie": {"label": "Symmetrie", "y": "Y"},
    "stabilitaet": {"label": "Stabilität", "y": "θ (°)"},
    "smoothness": {"label": "Smoothness", "y": "Jerk"},
    "kompaktheit": {"label": "Kompaktheit", "y": "K"},
    "line_integrity": {"label": "Ausrichtung", "y": "φ (°)"},
}

MES_COMPONENTS = [
    ("Rhythmus", "R"),
    ("Symmetrie", "Y"),
    ("Stabilität", "S"),
    ("Smoothness", "M"),
    ("Ausrichtung", "L"),
    ("Kompaktheit", "C"),
]

COMPONENT_MAP = {label: key for label, key in MES_COMPONENTS}


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        min-width: 280px;
    }

    .block-container {
        padding-top: 2.8rem !important;
        padding-bottom: 0.25rem;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100%;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0.25rem !important;
    }

    h1, h2, h3, h4 {
        line-height: 1.2 !important;
        margin-top: 0 !important;
        margin-bottom: 0.35rem !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }

    .stPlotlyChart {
        margin-bottom: 0 !important;
    }

    section[data-testid="stSidebar"] .stButton button {
        padding: 0.22rem 0.4rem;
        font-size: 0.8rem;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stToggle {
        font-size: 0.9rem !important;
    }

    div[data-testid="stSlider"] {
        transform: scale(0.93);
        transform-origin: left;
    }

    .mes-card {
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        padding: 0.35rem 0.55rem;
        background: white;
        min-height: 230px;
    }

    .mes-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
        padding: 0.08rem 0;
        font-size: 0.84rem;
    }

    .mes-label {
        color: #1f2937;
    }

    .mes-value {
        color: #111827;
        font-variant-numeric: tabular-nums;
        text-align: right;
        min-width: 58px;
    }

    .mes-divider {
        border-top: 1px solid #e6e9ef;
        margin: 0.18rem 0 0.08rem 0;
    }

    .mes-total {
        font-weight: 700;
        padding-top: 0.2rem;
    }

    .mes-official {
        color: #374151;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Allgemeine Hilfsfunktionen
# ============================================================
def normalize_name(value: str) -> str:
    value = str(value).strip()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"\.(csv|mp4|mov|webm|m4v)$", "", value)
    value = re.sub(r"_web$", "", value)
    value = re.sub(r"_poses$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def make_display_label(value: str) -> str:
    value = Path(str(value).strip()).name
    value = re.sub(r"\.(csv|mp4|mov|webm|m4v)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"_poses$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"_web$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^moguls[-_ ]*", "", value, flags=re.IGNORECASE)
    return value


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def moving_average(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def get_gender_color(summary: dict[str, Any] | pd.Series) -> str:
    gender = str(summary.get("Gender", "")).strip().upper()
    if gender == "W":
        return "#1f77b4"
    if gender == "M":
        return "#2ca02c"
    return "#4b5563"


def visibility_for_gender(gender: str, selected_gender: str | None):
    if selected_gender not in {"W", "M"}:
        return True
    return True if gender == selected_gender else "legendonly"


# ============================================================
# Video-Hilfsfunktionen
# ============================================================
def get_video_fps(video_path: Path | str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else DEFAULT_FPS


def get_video_frame(video_path: Path | str, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = max(0, min(frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@st.cache_data
def discover_video_files(video_dir: Path) -> pd.DataFrame:
    if not video_dir.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for path in video_dir.iterdir():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        stem = path.stem
        is_web = stem.endswith("_web")
        base_name = stem[:-4] if is_web else stem

        rows.append(
            {
                "video_path": path,
                "video_name": stem,
                "video_key": normalize_name(base_name),
                "is_web": is_web,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["video_key", "is_web"], ascending=[True, False])
    return df.drop_duplicates(subset=["video_key"], keep="first").reset_index(drop=True)


# ============================================================
# Daten laden
# ============================================================
@st.cache_data
def load_scores() -> pd.DataFrame:
    if not SCORES_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(SCORES_PATH, encoding="utf-8")
    df["video_key"] = df["file"].astype(str).map(normalize_name)
    return df


@st.cache_data
def load_poses() -> pd.DataFrame:
    if not POSES_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(POSES_PATH, encoding="utf-8")
    df["video_key"] = df["video"].astype(str).map(normalize_name)
    return df


@st.cache_data
def load_fis_scores() -> pd.DataFrame:
    if not FIS_SCORES_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(FIS_SCORES_PATH)
    df.columns = [c.replace("\r", " ").replace("\n", " ").strip() for c in df.columns]
    df["official_score"] = pd.to_numeric(df["Total"], errors="coerce")
    df["Bib"] = pd.to_numeric(df["Bib"], errors="coerce")
    df["Gender"] = df["Gender"].astype(str).str.strip()
    df["Competition"] = df["Competition"].astype(str).str.strip()
    return df


# ============================================================
# Datenaufbereitung
# ============================================================
def rolling_symmetry_from_hip_center(hip_cx: pd.Series, window: int = 30) -> pd.Series:
    hip_cx = pd.Series(hip_cx).astype(float)
    values = np.full(len(hip_cx), np.nan)

    for i in range(len(hip_cx)):
        start = max(0, i - window + 1)
        seg = hip_cx.iloc[start:i + 1].dropna()

        if len(seg) < 5:
            continue

        seg = seg - seg.median()
        left_vals = np.abs(seg[seg < 0])
        right_vals = seg[seg > 0]

        left_amp = float(np.median(left_vals)) if len(left_vals) else 0.0
        right_amp = float(np.median(right_vals)) if len(right_vals) else 0.0

        total = left_amp + right_amp
        values[i] = 1.0 if total < 1e-6 else float(np.clip(1 - abs(left_amp - right_amp) / total, 0, 1))

    return pd.Series(values, index=hip_cx.index)


def savgol_smooth(series: pd.Series, window: int = 9, polyorder: int = 3) -> pd.Series:
    series = pd.Series(series, dtype=float)

    if series.isna().all():
        return series

    series = series.interpolate(method="linear", limit_direction="both")

    if window > len(series):
        window = len(series) if len(series) % 2 == 1 else len(series) - 1

    if window < 5 or window <= polyorder:
        return series

    return pd.Series(
        savgol_filter(series.to_numpy(), window_length=window, polyorder=polyorder),
        index=series.index,
    )


def compute_plot_timeseries(poses_df: pd.DataFrame, fps: float = DEFAULT_FPS) -> pd.DataFrame:
    df = poses_df.sort_values("frame").copy()
    frame = pd.to_numeric(df["frame"], errors="coerce").fillna(0).astype(int)

    left_hip_x = safe_series(df, "left_hip_x")
    right_hip_x = safe_series(df, "right_hip_x")
    left_hip_y = safe_series(df, "left_hip_y")
    right_hip_y = safe_series(df, "right_hip_y")

    left_shoulder_x = safe_series(df, "left_shoulder_x")
    right_shoulder_x = safe_series(df, "right_shoulder_x")
    left_shoulder_y = safe_series(df, "left_shoulder_y")
    right_shoulder_y = safe_series(df, "right_shoulder_y")

    left_knee_x = safe_series(df, "left_knee_x")
    right_knee_x = safe_series(df, "right_knee_x")
    left_knee_y = safe_series(df, "left_knee_y")
    right_knee_y = safe_series(df, "right_knee_y")

    left_ankle_x = safe_series(df, "left_ankle_x")
    right_ankle_x = safe_series(df, "right_ankle_x")
    left_ankle_y = safe_series(df, "left_ankle_y")
    right_ankle_y = safe_series(df, "right_ankle_y")

    hip_cx = (left_hip_x + right_hip_x) / 2
    hip_cx_smooth = moving_average(hip_cx, window=7)

    symmetrie = rolling_symmetry_from_hip_center(hip_cx_smooth, window=15)
    center = hip_cx_smooth.median()
    hip_dx = hip_cx_smooth - center
    sym_left = hip_dx.clip(upper=0)
    sym_right = hip_dx.clip(lower=0)

    shoulder_dx = right_shoulder_x - left_shoulder_x
    shoulder_dy = right_shoulder_y - left_shoulder_y
    theta = np.degrees(np.arctan2(shoulder_dy, shoulder_dx.abs() + 1e-6))

    hip_cx_sg = savgol_smooth(hip_cx)
    jerk = hip_cx_sg.diff().diff().diff().abs().fillna(0)

    d_leg = np.sqrt((left_knee_x - right_knee_x) ** 2 + (left_knee_y - right_knee_y) ** 2)
    hip_width = np.sqrt((left_hip_x - right_hip_x) ** 2 + (left_hip_y - right_hip_y) ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        d_norm = pd.Series(np.where(hip_width > 1e-3, d_leg / hip_width, np.nan), index=df.index)

    d_min, d_max = 0.6, 1.2
    kompaktheit = pd.Series(np.clip((d_max - d_norm) / (d_max - d_min), 0, 1), index=df.index)

    ankle_dx = right_ankle_x - left_ankle_x
    ankle_dy = right_ankle_y - left_ankle_y
    phi = np.degrees(np.arctan2(ankle_dy.abs(), ankle_dx.abs() + 1e-6))

    return pd.DataFrame(
        {
            "frame": frame,
            "timestamp_sec": frame / fps,
            "rhythmus": hip_cx_smooth,
            "symmetrie": symmetrie,
            "sym_left": sym_left,
            "sym_right": sym_right,
            "sym_dx": hip_dx,
            "stabilitaet": theta,
            "smoothness": jerk,
            "kompaktheit": kompaktheit,
            "line_integrity": phi,
        }
    )


def parse_video_meta(video_name: str) -> dict[str, Any]:
    norm = normalize_name(video_name)

    gender = None
    if "frauen" in norm:
        gender = "W"
    elif "manner" in norm or "maenner" in norm:
        gender = "M"

    match = re.search(r"final_(\d)_(\d{1,3})(?:_hd)?$", norm)

    competition = f"Final {match.group(1)}" if match else None
    bib = int(match.group(2)) if match else None

    return {"Gender": gender, "Competition": competition, "Bib": bib}


def build_run_index(scores_df: pd.DataFrame, poses_df: pd.DataFrame, videos_df: pd.DataFrame) -> pd.DataFrame:
    keys = set(scores_df.get("video_key", pd.Series(dtype=str)).dropna())
    keys.update(poses_df.get("video_key", pd.Series(dtype=str)).dropna())
    keys.update(videos_df.get("video_key", pd.Series(dtype=str)).dropna())

    rows: list[dict[str, Any]] = []

    for key in sorted(keys):
        score_match = scores_df[scores_df["video_key"] == key]
        pose_match = poses_df[poses_df["video_key"] == key]
        video_match = videos_df[videos_df["video_key"] == key] if not videos_df.empty else pd.DataFrame()

        raw_label = key
        if not pose_match.empty:
            raw_label = str(pose_match["video"].iloc[0])
        elif not score_match.empty:
            raw_label = str(score_match["file"].iloc[0])
        elif not video_match.empty:
            raw_label = str(video_match["video_name"].iloc[0])

        rows.append(
            {
                "video_key": key,
                "label": make_display_label(raw_label),
                "has_scores": not score_match.empty,
                "has_poses": not pose_match.empty,
                "video_path": None if video_match.empty else video_match["video_path"].iloc[0],
            }
        )

    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


@st.cache_data
def build_scatter_df(scores_df: pd.DataFrame, fis_df: pd.DataFrame) -> pd.DataFrame:
    if scores_df.empty or fis_df.empty:
        return pd.DataFrame()

    df = scores_df.copy()
    df["video_label"] = df["file"].astype(str).map(make_display_label)
    df["video_key"] = df["file"].astype(str).map(normalize_name)
    df["MES_60"] = pd.to_numeric(df["MES_60"], errors="coerce")

    meta_df = pd.DataFrame(df["video_key"].apply(parse_video_meta).tolist())
    df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    df["Bib"] = pd.to_numeric(df["Bib"], errors="coerce")
    df["Gender"] = df["Gender"].astype(str).str.strip()
    df["Competition"] = df["Competition"].astype(str).str.strip()

    fis_df = fis_df.copy()
    fis_df["Bib"] = pd.to_numeric(fis_df["Bib"], errors="coerce")
    fis_df["Gender"] = fis_df["Gender"].astype(str).str.strip()
    fis_df["Competition"] = fis_df["Competition"].astype(str).str.strip()

    return df.merge(
        fis_df[["Bib", "Gender", "Competition", "official_score", "Name"]],
        on=["Bib", "Gender", "Competition"],
        how="left",
    )


def get_summary_for_key(scores_df: pd.DataFrame, video_key: str, scatter_df: pd.DataFrame) -> dict[str, Any]:
    match = scores_df[scores_df["video_key"] == video_key]

    if match.empty:
        return {
            "R": np.nan,
            "Y": np.nan,
            "S": np.nan,
            "M": np.nan,
            "L": np.nan,
            "C": np.nan,
            "MES_60": np.nan,
            "official_score": np.nan,
            "Gender": None,
        }

    row = match.iloc[0].to_dict()
    row["official_score"] = np.nan
    row["Gender"] = None

    sc_match = scatter_df[scatter_df["video_key"] == video_key]
    if not sc_match.empty:
        row["official_score"] = sc_match.iloc[0]["official_score"]
        row["Gender"] = sc_match.iloc[0].get("Gender")

    return row


def get_pose_subset(poses_df: pd.DataFrame, video_key: str) -> pd.DataFrame:
    match = poses_df[poses_df["video_key"] == video_key].copy()
    if match.empty:
        return pd.DataFrame()
    return match.sort_values("frame").reset_index(drop=True)


# ============================================================
# Plot-Helfer
# ============================================================
def make_metric_figure(
    df: pd.DataFrame,
    metric_name: str,
    current_x: float,
    x_mode: str = "timestamp_sec",
    height: int = 220,
    line_color: str | None = None,
) -> go.Figure:
    cfg = METRICS.get(metric_name, {"label": metric_name, "y": "Wert"})

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_mode],
            y=df[metric_name],
            mode="lines",
            name=cfg["label"],
            line=dict(width=2, color=line_color) if line_color else dict(width=2),
        )
    )

    fig.add_vline(x=current_x, line_width=3, line_dash="dash", line_color="red")

    fig.update_layout(
        title=cfg["label"],
        xaxis_title="Zeit (s)" if x_mode == "timestamp_sec" else "Frame",
        yaxis_title=cfg["y"],
        margin=dict(l=8, r=8, t=40, b=8),
        height=height,
        showlegend=False,
    )

    if metric_name == "symmetrie":
        fig.update_yaxes(range=[0, 1])

    return fig


def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return df

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return df[(df[col] >= lower) & (df[col] <= upper)]


def build_regression_target(
    scatter_df: pd.DataFrame,
    mode: str,
    selected_components: list[str],
) -> tuple[pd.DataFrame, str]:
    df = scatter_df.copy()

    if mode == "MES_60":
        cols = ["R", "Y", "S", "M", "C", "L"]
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["regression_score"] = df[cols].sum(axis=1) * 10
        return df, "MES Total"

    cols = [COMPONENT_MAP[c] for c in selected_components if c in COMPONENT_MAP]

    if not cols:
        df["regression_score"] = np.nan
        return df, "Ausgewählte Kategorien"

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["regression_score"] = df[cols].sum(axis=1) * 10
    return df, "Ausgewählte Kategorien"


def make_scatter_figure(
    scatter_df: pd.DataFrame,
    selected_key: str,
    focus_main_cluster: bool = False,
    regression_mode: str = "MES_60",
    selected_components: list[str] | None = None,
    selected_gender: str | None = None,
) -> go.Figure:
    selected_components = selected_components or []
    fig = go.Figure()

    scatter_df, y_axis_label = build_regression_target(
        scatter_df=scatter_df,
        mode=regression_mode,
        selected_components=selected_components,
    )

    base_df = scatter_df.dropna(subset=["official_score", "regression_score"]).copy()

    women_df = base_df[base_df["Gender"] == "W"].copy()
    men_df = base_df[base_df["Gender"] == "M"].copy()

    if focus_main_cluster:
        women_df = remove_outliers_iqr(remove_outliers_iqr(women_df, "official_score"), "regression_score")
        men_df = remove_outliers_iqr(remove_outliers_iqr(men_df, "official_score"), "regression_score")

    base_df = pd.concat([women_df, men_df], ignore_index=True)
    women_df = base_df[base_df["Gender"] == "W"].copy()
    men_df = base_df[base_df["Gender"] == "M"].copy()

    def add_points(df: pd.DataFrame, name: str, color: str, gender: str) -> None:
        if df.empty:
            return

        fig.add_trace(
            go.Scatter(
                x=df["official_score"],
                y=df["regression_score"],
                mode="markers",
                name=name,
                visible=visibility_for_gender(gender, selected_gender),
                text=df["video_label"],
                customdata=np.stack(
                    [
                        df["Name"].fillna(""),
                        df["Competition"].fillna(""),
                        df["Bib"].fillna(""),
                    ],
                    axis=1,
                ),
                marker=dict(size=8, color=color, opacity=0.8),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Athlet: %{customdata[0]}<br>"
                    "Competition: %{customdata[1]}<br>"
                    "Startnummer: %{customdata[2]}<br>"
                    "Offizieller Score: %{x:.2f}<br>"
                    f"{y_axis_label}: "
                    "%{y:.2f}<extra></extra>"
                ),
            )
        )

    def add_regression(df: pd.DataFrame, color: str, label: str, gender: str) -> None:
        reg_df = df.dropna(subset=["official_score", "regression_score"]).copy()
        if len(reg_df) < 2:
            return

        x = reg_df["official_score"].to_numpy(dtype=float)
        y = reg_df["regression_score"].to_numpy(dtype=float)

        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(reg_df["official_score"].min(), reg_df["official_score"].max(), 100)
        y_line = m * x_line + b

        r = np.corrcoef(x, y)[0, 1]
        r2 = r**2

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"{label} Trend (R²={r2:.2f})",
                visible=visibility_for_gender(gender, selected_gender),
                line=dict(color=color, width=2),
                hoverinfo="skip",
            )
        )

    add_points(women_df, "Frauen", "#1f77b4", "W")
    add_points(men_df, "Männer", "#2ca02c", "M")
    add_regression(women_df, "#1f77b4", "Frauen", "W")
    add_regression(men_df, "#2ca02c", "Männer", "M")

    selected_df = base_df[base_df["video_key"] == selected_key]
    if not selected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_df["official_score"],
                y=selected_df["regression_score"],
                mode="markers",
                name="Auswahl",
                text=selected_df["video_label"],
                marker=dict(size=14, color="red", line=dict(width=2, color="darkred")),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Offizieller Score: %{x:.2f}<br>"
                    f"{y_axis_label}: "
                    "%{y:.2f}<extra></extra>"
                ),
            )
        )

    title = "MES vs. offizieller Score" if regression_mode == "MES_60" else f"{y_axis_label} vs. offizieller Score"

    fig.update_layout(
        title=title,
        xaxis_title="Offizieller Score",
        yaxis_title=y_axis_label,
        margin=dict(l=8, r=8, t=36, b=8),
        height=245,
        legend=dict(
            orientation="v",
            y=0.5,
            yanchor="middle",
            x=1.02,
            xanchor="left",
        ),
    )

    if not base_df.empty:
        y_min = base_df["regression_score"].min()
        y_max = base_df["regression_score"].max()
        fig.update_yaxes(range=[y_min - 0.5, y_max + 0.5], autorange=False)

    return fig


def make_radar_figure(summary: dict[str, Any], color: str | None = None) -> go.Figure:
    labels = ["R", "Y", "S", "M", "L", "C"]
    label_names = {
        "R": "Rhythmus",
        "Y": "Symmetrie",
        "S": "Stabilität",
        "M": "Smoothness",
        "L": "Ausrichtung",
        "C": "Kompaktheit",
    }

    values = [
        float(summary.get(k, 0)) * 10 if pd.notna(summary.get(k, np.nan)) else 0
        for k in labels
    ]

    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    hover_text = [f"{label_names[label]}: {value:.2f}" for label, value in zip(labels, values)]
    hover_text_closed = hover_text + [hover_text[0]]

    trace_kwargs = {}
    if color:
        trace_kwargs = {
            "line": dict(color=color, width=2),
            "fillcolor": color,
            "opacity": 0.35,
        }

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            hovertext=hover_text_closed,
            hovertemplate="%{hovertext}<extra></extra>",
            **trace_kwargs,
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=240,
    )

    return fig


# ============================================================
# Session State
# ============================================================
def init_session_state(selected_key: str, max_index: int) -> None:
    st.session_state.setdefault("current_idx", 0)
    st.session_state.setdefault("last_video_key", selected_key)
    st.session_state.setdefault("frame_mode", False)
    st.session_state.setdefault("frame_slider", 0)

    if st.session_state.last_video_key != selected_key:
        st.session_state.current_idx = 0
        st.session_state.frame_slider = 0
        st.session_state.last_video_key = selected_key
        st.session_state.frame_mode = False

    st.session_state.current_idx = min(st.session_state.current_idx, max_index)


# ============================================================
# UI-Hilfsfunktionen
# ============================================================
def render_mes_card(summary: dict[str, Any], selected_color: str) -> None:
    mes_items = [
        (
            label,
            summary.get(key, np.nan) * 10 if pd.notna(summary.get(key, np.nan)) else np.nan,
        )
        for label, key in MES_COMPONENTS
    ]

    total_mes = summary.get("MES_60", np.nan)
    official_score = summary.get("official_score", np.nan)

    rows_html = "".join(
        (
            f'<div class="mes-row">'
            f'<span class="mes-label">{label}</span>'
            f'<span class="mes-value">{"–" if pd.isna(value) else f"{value:.3f}"}</span>'
            f"</div>"
        )
        for label, value in mes_items
    )

    total_str = f"{total_mes:.3f}" if pd.notna(total_mes) else "–"
    official_str = f"{official_score:.1f}" if pd.notna(official_score) else "–"

    html = textwrap.dedent(
        f"""
        <div class="mes-card" style="border-left: 5px solid {selected_color};">
            {rows_html}
            <div class="mes-divider"></div>
            <div class="mes-row mes-total">
                <span class="mes-label">Total</span>
                <span class="mes-value">{total_str}</span>
            </div>
            <div class="mes-row mes-official">
                <span class="mes-label">Offizieller FIS-Score</span>
                <span class="mes-value">{official_str}</span>
            </div>
        </div>
        """
    )

    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# Daten laden
# ============================================================
scores_df = load_scores()
poses_df = load_poses()
fis_df = load_fis_scores()
videos_df = discover_video_files(VIDEO_DIR)

run_index = build_run_index(scores_df, poses_df, videos_df)
scatter_df = build_scatter_df(scores_df, fis_df)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    if run_index.empty:
        st.error("Keine Daten gefunden. Lege all_poses.csv und mes_scores.csv unter ./daten ab.")
        st.stop()

    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #1f2937, #111827);
            color: white;
            padding:12px 16px;
            border-radius:12px;
            margin-bottom:16px;
        ">
            <div style="font-size:18px; font-weight:600; line-height:1.25;">
                Moguls Elegance Score<br>
                Demotool mit YOLOv8n
            </div>
            <div style="font-size:13px; opacity:0.8; margin-top:4px;">
                Full HD (1920×1080) • 25.25 fps
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Videoanalyse")

    selected_label = st.selectbox("Sequenz", run_index["label"].tolist())
    selected_row = run_index.loc[run_index["label"] == selected_label].iloc[0]
    selected_key = selected_row["video_key"]

    summary = get_summary_for_key(scores_df, selected_key, scatter_df)
    selected_color = get_gender_color(summary)
    pose_subset = get_pose_subset(poses_df, selected_key)

    if pose_subset.empty:
        st.error("Für diese Videosequenz wurden keine Pose-Daten gefunden.")
        st.stop()

    fps = get_video_fps(selected_row["video_path"]) if selected_row["video_path"] else DEFAULT_FPS
    ts_df = compute_plot_timeseries(pose_subset, fps=fps)
    max_index = max(len(ts_df) - 1, 0)

    init_session_state(selected_key, max_index)

    st.toggle("Einzelbilder", key="frame_mode")

    if st.session_state.frame_mode:
        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("◀ -1", use_container_width=True):
                st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
                st.session_state.frame_slider = st.session_state.current_idx

        with c2:
            if st.button("↺", use_container_width=True):
                st.session_state.current_idx = 0
                st.session_state.frame_slider = 0

        with c3:
            if st.button("+1 ▶", use_container_width=True):
                st.session_state.current_idx = min(max_index, st.session_state.current_idx + 1)
                st.session_state.frame_slider = st.session_state.current_idx

        st.slider("Frame", 0, max_index, step=1, key="frame_slider")
        st.session_state.current_idx = st.session_state.frame_slider

    x_mode = st.radio(
        "X-Achse",
        ["timestamp_sec", "frame"],
        horizontal=True,
        format_func=lambda x: "Zeit" if x == "timestamp_sec" else "Frame",
    )

    metric_options = [metric for metric in METRICS if metric in ts_df.columns]

    metric_1 = st.selectbox(
        "Metrik 1",
        metric_options,
        index=0,
        format_func=lambda x: METRICS[x]["label"],
    )
    metric_2 = st.selectbox(
        "Metrik 2",
        metric_options,
        index=min(1, len(metric_options) - 1),
        format_func=lambda x: METRICS[x]["label"],
    )

    st.markdown("#### Regressionsbasis")

    regression_mode_label = st.radio(
        "Regressionsbasis",
        ["MES Total", "Ausgewählte Kategorien"],
        horizontal=False,
        label_visibility="collapsed",
    )

    selected_regression_components = st.multiselect(
        "Kategorien auswählen",
        options=list(COMPONENT_MAP.keys()),
        default=["Stabilität", "Kompaktheit"] if regression_mode_label == "Ausgewählte Kategorien" else [],
        disabled=(regression_mode_label != "Ausgewählte Kategorien"),
    )

    regression_mode = "MES_60" if regression_mode_label == "MES Total" else "selected_sum"

    st.markdown("#### Info")
    st.info("folgt")


# ============================================================
# Abgeleitete UI-Werte
# ============================================================
current_idx = min(st.session_state.current_idx, max_index)
current_row = ts_df.iloc[current_idx]
current_x = float(current_row[x_mode]) if st.session_state.frame_mode else 0


# ============================================================
# Hauptlayout
# ============================================================
top_left, top_right = st.columns([0.9, 1.1], gap="medium")

with top_left:
    video_path = selected_row["video_path"]

    if st.session_state.frame_mode:
        st.markdown("### Einzelbild")

        if video_path and Path(video_path).exists():
            frame_number = int(ts_df.iloc[current_idx]["frame"])
            frame_img = get_video_frame(video_path, frame_number)

            if frame_img is not None:
                st.image(frame_img, use_container_width=True)
                st.caption(f"Frame {frame_number} | t = {ts_df.iloc[current_idx]['timestamp_sec']:.2f} s")
            else:
                st.warning(f"Frame {frame_number} konnte nicht geladen werden.")
        else:
            st.info("Kein passendes Video im Video-Ordner gefunden.")
    else:
        st.markdown("#### Wiedergabe")
        if video_path and Path(video_path).exists():
            st.video(str(video_path))
        else:
            st.info("Kein passendes Video im Video-Ordner gefunden.")

with top_right:
    st.markdown("#### Metriken")

    fig1 = make_metric_figure(
        ts_df,
        metric_1,
        current_x=current_x,
        x_mode=x_mode,
        height=120,
        line_color=selected_color,
    )
    fig2 = make_metric_figure(
        ts_df,
        metric_2,
        current_x=current_x,
        x_mode=x_mode,
        height=120,
        line_color=selected_color,
    )

    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

bottom_left, bottom_right = st.columns([0.9, 1.1], gap="medium")

with bottom_left:
    radar_col, mes_col = st.columns([1.3, 0.9], gap="medium")

    with radar_col:
        st.markdown("#### Radar")
        st.plotly_chart(
            make_radar_figure(summary, color=selected_color),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with mes_col:
        st.markdown("#### MES")
        render_mes_card(summary, selected_color)

with bottom_right:
    score_header_left, score_header_right = st.columns([1.2, 1], vertical_alignment="center")

    with score_header_left:
        st.markdown("#### Score")

    with score_header_right:
        focus_main_cluster = st.toggle("Hauptcluster fokussieren", value=False)

    scatter_fig = make_scatter_figure(
        scatter_df,
        selected_key=selected_key,
        focus_main_cluster=focus_main_cluster,
        regression_mode=regression_mode,
        selected_components=selected_regression_components,
        selected_gender=summary.get("Gender"),
    )
    st.plotly_chart(scatter_fig, use_container_width=True, config={"displayModeBar": False})
