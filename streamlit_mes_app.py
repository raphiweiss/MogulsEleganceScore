from __future__ import annotations

import base64
import json
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
import streamlit.components.v1 as components
from scipy.signal import savgol_filter

from methodik import render_methodik
from ui import inject_styles, render_sidebar_header, render_sidebar_methodik_button

# ============================================================
# App-Konfiguration
# ============================================================
st.set_page_config(
    page_title="Moguls Elegance Score",
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
    "stabilitaet": {"label": "Stabilität", "y": "θ (°)"},
    "kompaktheit": {"label": "Kompaktheit", "y": "K"},
    "symmetrie": {"label": "Symmetrie", "y": "Y"},
    "smoothness": {"label": "Smoothness", "y": "Jerk"},
    "line_integrity": {"label": "Line-Integrity", "y": "φ (°)"},
    

}

MES_COMPONENTS = [
    ("Rhythmus", "R"),
    ("Stabilität", "S"),
    ("Kompaktheit", "C"),
    ("Symmetrie", "Y"),
    ("Smoothness", "M"),
    ("Line-Integrity", "L"),
]

COMPONENT_MAP = {label: key for label, key in MES_COMPONENTS}

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


def get_gender_color(summary: dict[str, Any] | pd.Series | str) -> str:
    if isinstance(summary, str):
        gender = summary.strip().upper()
    else:
        gender = str(summary.get("Gender", "")).strip().upper()

    if gender == "W":
        return "#6366f1"  # Indigo
    if gender == "M":
        return "#14b8a6"  # Teal
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


def encode_video_base64(video_path: Path | str) -> tuple[str | None, str | None]:
    path = Path(video_path)
    if not path.exists():
        return None, None

    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".m4v": "video/mp4",
    }
    mime_type = mime_map.get(path.suffix.lower(), "video/mp4")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return data, mime_type


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
            "stabilitaet": theta,
            "smoothness": jerk,
            "line_integrity": phi,
            "kompaktheit": kompaktheit,
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
            "S": np.nan,
            "C": np.nan,
            "Y": np.nan,
            "M": np.nan,
            "L": np.nan,

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
        return df, "Ausgewählte Metriken"

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["regression_score"] = df[cols].sum(axis=1) * 10
    return df, "Ausgewählte Metriken"


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

        colors = np.where(df["video_key"] == selected_key, "red", color)

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
                marker=dict(size=8, color=colors, opacity=0.8),
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

    add_points(women_df, "Frauen", get_gender_color("W"), "W")
    add_points(men_df, "Männer", get_gender_color("M"), "M")
    
    add_regression(women_df, get_gender_color("W"), "Frauen", "W")
    add_regression(men_df, get_gender_color("M"), "Männer", "M")

    title = "MES vs. offizieller Score" if regression_mode == "MES_60" else f"{y_axis_label} vs. offizieller Score"

    fig.add_trace(
        go.Scatter(
            x=[None],  # kein echter Punkt
            y=[None],
            mode="markers",
            name="Aktuelles Video",
            marker=dict(size=8, color="red"),
            showlegend=True,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Offizieller Score",
        yaxis_title=y_axis_label,
        margin=dict(l=4, r=8, t=36, b=8),
        height=230,
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
        "S": "Stabilität",
        "C": "Kompaktheit",
        "Y": "Symmetrie",
        "M": "Smoothness",
        "L": "Line-Integrity",

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
        height=220,
    )

    return fig


# ============================================================
# HTML-Komponente für Video + synchrone Marker + Klick-Sync
# ============================================================
def build_synced_metric_html(
    video_path: Path | str,
    ts_df: pd.DataFrame,
    metric_1: str,
    metric_2: str,
    selected_color: str,
    x_mode: str = "timestamp_sec",
) -> str:
    video_b64, mime_type = encode_video_base64(video_path)
    if not video_b64 or not mime_type:
        return "<p>Video konnte nicht geladen werden.</p>"

    metric_cfg_1 = METRICS.get(metric_1, {"label": metric_1, "y": "Wert"})
    metric_cfg_2 = METRICS.get(metric_2, {"label": metric_2, "y": "Wert"})

    x_values = pd.to_numeric(ts_df[x_mode], errors="coerce").fillna(0).tolist()
    y1 = pd.to_numeric(ts_df[metric_1], errors="coerce").fillna(0).tolist()
    y2 = pd.to_numeric(ts_df[metric_2], errors="coerce").fillna(0).tolist()
    frames = pd.to_numeric(ts_df["frame"], errors="coerce").fillna(0).astype(int).tolist()
    times = pd.to_numeric(ts_df["timestamp_sec"], errors="coerce").fillna(0).tolist()

    initial_x = 0.0 if x_mode == "timestamp_sec" else 0
    x_title = "Zeit (s)" if x_mode == "timestamp_sec" else "Frame"

    payload = {
        "mime_type": mime_type,
        "video_b64": video_b64,
        "x_values": x_values,
        "y1": y1,
        "y2": y2,
        "frames": frames,
        "times": times,
        "metric_1": metric_1,
        "metric_2": metric_2,
        "metric_label_1": metric_cfg_1["label"],
        "metric_label_2": metric_cfg_2["label"],
        "metric_y_1": metric_cfg_1["y"],
        "metric_y_2": metric_cfg_2["y"],
        "selected_color": selected_color,
        "x_mode": x_mode,
        "x_title": x_title,
        "initial_x": initial_x,
    }

    return f"""
    <div id="mes-layout" style="
        display:grid;
        grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.05fr);
        gap:24px;
        align-items:start;
        font-family:Inter, ui-sans-serif, system-ui, sans-serif;
        width:100%;
    ">
      <div style="min-width:0;">
        <video
          id="mes-video"
          controls
          style="
            width:100%;
            height:280px;
            object-fit:cover;
            border-radius:12px;
            display:block;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
          "
        >
          <source src="data:{mime_type};base64,{video_b64}" type="{mime_type}">
        </video>

        <div id="mes-readout" style="margin-top:4px; font-size:12px; color:#6b7280;">
          t = 0.00 s | Frame = 0
        </div>
      </div>

      <div id="metrics-layout" style="min-width:0;">
        <div id="chart1" style="width:100%; height:140px;"></div>
        <div id="chart2" style="width:100%; height:140px; margin-top:10px;"></div>
      </div>
    </div>

    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
      const data = {json.dumps(payload)};
      const video = document.getElementById("mes-video");
      const readout = document.getElementById("mes-readout");
      const chart1 = document.getElementById("chart1");
      const chart2 = document.getElementById("chart2");
      const layoutWrapper = document.getElementById("mes-layout");

      const markerShape = {{
        type: "line",
        x0: data.initial_x,
        x1: data.initial_x,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: {{
          color: "#ef4444",
          width: 2,
          dash: "dot"
        }}
      }};

      function buildLayout(title, yTitle) {{
        return {{
          margin: {{ l: 20, r: 6, t: 44, b: 42 }},
          title: {{
            text: "<b>" + title + "</b>",
            x: 0.01,
            xanchor: "left",
            y: 0.98,
            yanchor: "top",
            font: {{
              size: 14,
              color: "#111827"
            }}
          }},
          xaxis: {{
            title: {{
              text: data.x_title,
              standoff: 8,
              font: {{ size: 10 }}
            }},
            tickfont: {{ size: 10 }},
            automargin: true
          }},
          yaxis: {{
            title: {{
              text: yTitle,
              standoff: 6,
              font: {{ size: 10 }}
            }},
            tickfont: {{ size: 10 }},
            automargin: true
          }},
          shapes: [{{ ...markerShape }}],
          showlegend: false,
          plot_bgcolor: "white",
          paper_bgcolor: "white",
          hovermode: "x"
        }};
      }}

      function nearestFrameIndex(currentTime) {{
        if (!data.times.length) return 0;

        let bestIdx = 0;
        let bestDiff = Math.abs(data.times[0] - currentTime);

        for (let i = 1; i < data.times.length; i++) {{
          const diff = Math.abs(data.times[i] - currentTime);
          if (diff < bestDiff) {{
            bestDiff = diff;
            bestIdx = i;
          }}
        }}
        return bestIdx;
      }}

      function markerXFromVideoTime(currentTime) {{
        if (data.x_mode === "timestamp_sec") {{
          return currentTime;
        }}
        const idx = nearestFrameIndex(currentTime);
        return data.frames[idx] ?? 0;
      }}

      function timeFromChartX(xVal) {{
        if (data.x_mode === "timestamp_sec") {{
          return Math.max(0, Number(xVal) || 0);
        }}

        if (!data.frames.length || !data.times.length) return 0;

        let bestIdx = 0;
        let bestDiff = Math.abs(data.frames[0] - xVal);

        for (let i = 1; i < data.frames.length; i++) {{
          const diff = Math.abs(data.frames[i] - xVal);
          if (diff < bestDiff) {{
            bestDiff = diff;
            bestIdx = i;
          }}
        }}
        return data.times[bestIdx] ?? 0;
      }}

      function updateMarker() {{
        if (!video) return;

        const t = video.currentTime || 0;
        const idx = nearestFrameIndex(t);
        const frame = data.frames[idx] ?? 0;
        const markerX = markerXFromVideoTime(t);

        if (readout) {{
          readout.textContent = `t = ${{t.toFixed(2)}} s | Frame = ${{frame}}`;
        }}

        Plotly.relayout(chart1, {{
          "shapes[0].x0": markerX,
          "shapes[0].x1": markerX
        }});

        Plotly.relayout(chart2, {{
          "shapes[0].x0": markerX,
          "shapes[0].x1": markerX
        }});
      }}

      function seekVideoFromClick(eventData) {{
        if (!video || !eventData || !eventData.points || !eventData.points.length) return;

        const clickedX = eventData.points[0].x;
        const targetTime = timeFromChartX(clickedX);

        video.currentTime = targetTime;
        updateMarker();

        if (!video.paused) {{
          const playPromise = video.play();
          if (playPromise !== undefined) {{
            playPromise.catch(() => null);
          }}
        }}
      }}

      Plotly.newPlot(
        chart1,
        [{{
          x: data.x_values,
          y: data.y1,
          type: "scatter",
          mode: "lines",
          line: {{ width: 2, color: data.selected_color }}
        }}],
        buildLayout(data.metric_label_1, data.metric_y_1),
        {{ displayModeBar: false, responsive: true }}
      ).then(() => updateMarker());

      Plotly.newPlot(
        chart2,
        [{{
          x: data.x_values,
          y: data.y2,
          type: "scatter",
          mode: "lines",
          line: {{ width: 2, color: data.selected_color }}
        }}],
        buildLayout(data.metric_label_2, data.metric_y_2),
        {{ displayModeBar: false, responsive: true }}
      ).then(() => updateMarker());

      if (data.metric_1 === "symmetrie") {{
        Plotly.relayout(chart1, {{ "yaxis.range": [0, 1] }});
      }}
      if (data.metric_2 === "symmetrie") {{
        Plotly.relayout(chart2, {{ "yaxis.range": [0, 1] }});
      }}

      chart1.on("plotly_click", seekVideoFromClick);
      chart2.on("plotly_click", seekVideoFromClick);

      let rafId = null;

      function loop() {{
        updateMarker();
        if (!video.paused && !video.ended) {{
          rafId = requestAnimationFrame(loop);
        }}
      }}

      video.addEventListener("play", () => {{
        if (rafId) cancelAnimationFrame(rafId);
        loop();
      }});

      video.addEventListener("pause", () => {{
        updateMarker();
        if (rafId) cancelAnimationFrame(rafId);
      }});

      video.addEventListener("seeked", updateMarker);
      video.addEventListener("timeupdate", updateMarker);
      video.addEventListener("loadedmetadata", updateMarker);

      const resizeObserver = new ResizeObserver(() => {{
        Plotly.Plots.resize(chart1);
        Plotly.Plots.resize(chart2);
      }});

      if (layoutWrapper) {{
        resizeObserver.observe(layoutWrapper);
      }}

      setTimeout(updateMarker, 100);
    </script>
    """

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
            f'<span class="mes-value">{"–" if pd.isna(value) else f"{value:.1f}"}</span>'
            f"</div>"
        )
        for label, value in mes_items
    )

    total_str = f"{total_mes:.1f}" if pd.notna(total_mes) else "–"
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
                <span class="mes-label">Offizieller Score</span>
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
# Styles
# ============================================================
inject_styles()

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    if run_index.empty:
        st.error("Keine Daten gefunden.")
        st.stop()

    render_sidebar_header()
    open_methodik = render_sidebar_methodik_button()

    st.markdown(
        "<div class='sidebar-section'>Videoanalyse</div>",
        unsafe_allow_html=True,
    )

    selected_label = st.selectbox("Sequenz", run_index["label"].tolist())

    x_mode = st.segmented_control(
        "X-Achse",
        options=["timestamp_sec", "frame"],
        default="timestamp_sec",
        format_func=lambda x: "Zeit" if x == "timestamp_sec" else "Frame",
    )

    # ============================================================
    # Auswahl berechnen
    # ============================================================
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

    st.markdown(
        "<div class='sidebar-section'>Regressionsanalyse</div>",
        unsafe_allow_html=True,
    )

    regression_mode = st.segmented_control(
        "Basis",
        options=["MES_60", "selected_sum"],
        default="MES_60",
        format_func=lambda x: "MES Total" if x == "MES_60" else "Auswahl",
    )

    selected_regression_components = st.multiselect(
        "Kategorien",
        options=list(COMPONENT_MAP.keys()),
        default=["Stabilität", "Kompaktheit"] if regression_mode == "selected_sum" else [],
        disabled=(regression_mode != "selected_sum"),
    )

# ============================================================
# Methodik Dialog
# ============================================================
@st.dialog("Methodik & Hintergrund", width="large")
def show_methodik_dialog():
    render_methodik()

if open_methodik:
    show_methodik_dialog()

# ============================================================
# Header + Main View
# ============================================================
header_left, header_right = st.columns([0.92, 1.08], gap="large")

with header_left:
    st.markdown("#### Video-basierte Pose Estimation")
with header_right:
    st.markdown("#### Metriken")

video_path = selected_row["video_path"]
if video_path and Path(video_path).exists():
    html = build_synced_metric_html(
        video_path=video_path,
        ts_df=ts_df,
        metric_1=metric_1,
        metric_2=metric_2,
        selected_color=selected_color,
        x_mode=x_mode,
    )
    components.html(html, height=315, scrolling=False)
else:
    st.info("Kein passendes Video im Video-Ordner gefunden.")

# ============================================================
# Unterer Bereich
# ============================================================
bottom_left, bottom_right = st.columns([0.92, 1.08], gap="large")

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
    score_header_left, score_header_right = st.columns([3, 2], vertical_alignment="center")

    with score_header_left:
        st.markdown("#### Scorevergleich")

    with score_header_right:
        focus_main_cluster = st.toggle("Fokus auf Hauptcluster", value=False)

    scatter_fig = make_scatter_figure(
        scatter_df,
        selected_key=selected_key,
        focus_main_cluster=focus_main_cluster,
        regression_mode=regression_mode,
        selected_components=selected_regression_components,
        selected_gender=summary.get("Gender"),
    )
    st.plotly_chart(
        scatter_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )