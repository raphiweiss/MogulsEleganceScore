from __future__ import annotations

from pathlib import Path
from typing import Dict
import unicodedata
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import cv2
import textwrap
import time

from streamlit_autorefresh import st_autorefresh

# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------
st.set_page_config(
    page_title="MES Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = APP_DIR / "model_results" / "yolov8n" / "data"
VIDEO_DIR = APP_DIR / "model_results" / "yolov8n" / "videos"
POSES_PATH = DATA_DIR / "all_poses.csv"
SCORES_PATH = DATA_DIR / "mes_scores.csv"
DEFAULT_FPS = 25


# ------------------------------------------------------------
# Kompaktes Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>

    section[data-testid="stSidebar"] {
        min-width: 280px;
    }
    

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0.25rem;
        max-width: 100%;
    }

    h1, h2 {
        line-height: 1.25 !important;
        margin-top: 0 !important;
        margin-bottom: 0.4rem !important;
        overflow: visible !important;
        white-space: normal !important;
    }
    h2 {
        margin-top: 0.3rem;
    }
    }

    h3 {
        font-size: 1.05rem !important;
        line-height: 1.15 !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
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


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------
def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("_poses", "")
    value = value.replace(".csv", "").replace(".mp4", "").replace(".mov", "").replace(".webm", "")
    value = "".join(ch if ch.isalnum() else "_" for ch in value)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def moving_average(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def compute_plot_timeseries(poses_df: pd.DataFrame, fps: int = DEFAULT_FPS) -> pd.DataFrame:
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

    shoulder_dx = right_shoulder_x - left_shoulder_x
    shoulder_dy = right_shoulder_y - left_shoulder_y
    theta = np.degrees(np.arctan2(shoulder_dy, shoulder_dx.abs() + 1e-6))

    jerk = hip_cx_smooth.diff().diff().diff().abs().fillna(0)

    d_leg = np.sqrt((left_knee_x - right_knee_x) ** 2 + (left_knee_y - right_knee_y) ** 2)
    hip_width = np.sqrt((left_hip_x - right_hip_x) ** 2 + (left_hip_y - right_hip_y) ** 2)
    d_norm = d_leg / (hip_width.replace(0, np.nan))

    ankle_dx = right_ankle_x - left_ankle_x
    ankle_dy = right_ankle_y - left_ankle_y
    phi = np.degrees(np.arctan2(ankle_dy.abs(), ankle_dx.abs() + 1e-6))

    center = hip_cx_smooth.median()
    left_amp = (hip_cx_smooth - center).clip(lower=0)
    right_amp = (center - hip_cx_smooth).clip(lower=0)
    symmetrie = 1 - (left_amp - right_amp).abs() / (left_amp + right_amp + 1e-6)
    symmetrie = symmetrie.clip(lower=0, upper=1)

    return pd.DataFrame(
        {
            "frame": frame,
            "timestamp_sec": frame / fps,
            "rhythmus": hip_cx_smooth,
            "stabilitaet": theta,
            "smoothness": jerk,
            "kompaktheit": d_norm,
            "line_integrity": phi,
            "symmetrie": symmetrie,
        }
    )


# ------------------------------------------------------------
# Laden der Daten
# ------------------------------------------------------------
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
def discover_video_files(video_dir: Path) -> pd.DataFrame:
    if not video_dir.exists():
        return pd.DataFrame()

    candidates = []
    for path in video_dir.iterdir():
        if path.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v"}:
            stem = path.stem
            is_web = stem.endswith("_web")
            base_name = stem[:-4] if is_web else stem

            candidates.append(
                {
                    "video_path": path,
                    "video_name": stem,
                    "video_key": normalize_name(base_name),
                    "is_web": is_web,
                }
            )

    if not candidates:
        return pd.DataFrame()

    df = pd.DataFrame(candidates)
    df = df.sort_values(["video_key", "is_web"], ascending=[True, False])
    df = df.drop_duplicates(subset=["video_key"], keep="first").reset_index(drop=True)
    return df


def get_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = max(0, min(frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def build_run_index(scores_df: pd.DataFrame, poses_df: pd.DataFrame, videos_df: pd.DataFrame) -> pd.DataFrame:
    keys = set(scores_df.get("video_key", pd.Series(dtype=str)).dropna().tolist())
    keys.update(poses_df.get("video_key", pd.Series(dtype=str)).dropna().tolist())
    keys.update(videos_df.get("video_key", pd.Series(dtype=str)).dropna().tolist())

    rows = []
    for key in sorted(keys):
        score_match = scores_df[scores_df["video_key"] == key]
        pose_match = poses_df[poses_df["video_key"] == key]
        video_match = videos_df[videos_df["video_key"] == key] if not videos_df.empty else pd.DataFrame()

        label = key
        if not pose_match.empty:
            label = str(pose_match["video"].iloc[0])
        elif not score_match.empty:
            label = str(score_match["file"].iloc[0]).replace("_poses.csv", "")
        elif not video_match.empty:
            label = str(video_match["video_name"].iloc[0])

        rows.append(
            {
                "video_key": key,
                "label": label,
                "has_scores": not score_match.empty,
                "has_poses": not pose_match.empty,
                "video_path": None if video_match.empty else video_match["video_path"].iloc[0],
            }
        )

    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def get_summary_for_key(scores_df: pd.DataFrame, video_key: str, scatter_df: pd.DataFrame) -> Dict[str, float]:
    match = scores_df[scores_df["video_key"] == video_key]
    if match.empty:
        return {
            "R": np.nan,
            "S": np.nan,
            "M": np.nan,
            "Y": np.nan,
            "C": np.nan,
            "L": np.nan,
            "MES_60": np.nan,
            "official_score": np.nan,
        }

    row = match.iloc[0].to_dict()
    row["official_score"] = np.nan

    sc_match = scatter_df[scatter_df["video_key"] == video_key]
    if not sc_match.empty:
        row["official_score"] = sc_match.iloc[0]["official_score"]

    return row


def get_pose_subset(poses_df: pd.DataFrame, video_key: str) -> pd.DataFrame:
    match = poses_df[poses_df["video_key"] == video_key].copy()
    if match.empty:
        return pd.DataFrame()
    return match.sort_values("frame").reset_index(drop=True)


@st.cache_data
def load_fis_scores() -> pd.DataFrame:
    fis_path = DATA_DIR / "fis_scores.csv"
    if not fis_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(fis_path)
    df.columns = [c.replace("\r", " ").replace("\n", " ").strip() for c in df.columns]

    df["official_score"] = pd.to_numeric(df["Total"], errors="coerce")
    df["Bib"] = pd.to_numeric(df["Bib"], errors="coerce")
    df["Gender"] = df["Gender"].astype(str).str.strip()
    df["Competition"] = df["Competition"].astype(str).str.strip()

    return df


def parse_video_meta(video_name: str) -> dict:
    name = str(video_name)

    gender = None
    if "Frauen" in name:
        gender = "W"
    elif "Männer" in name or "Maenner" in name or "Manner" in normalize_name(name):
        gender = "M"

    comp_match = re.search(r"Final[_ ]?(\d)", name)
    competition = f"Final {comp_match.group(1)}" if comp_match else None

    bib_match = re.search(r"_(\d{2})$", name)
    bib = int(bib_match.group(1)) if bib_match else None

    return {
        "Gender": gender,
        "Competition": competition,
        "Bib": bib,
    }


@st.cache_data
def build_scatter_df(scores_df: pd.DataFrame, fis_df: pd.DataFrame) -> pd.DataFrame:
    if scores_df.empty or fis_df.empty:
        return pd.DataFrame()

    tmp = scores_df.copy()
    tmp["video_label"] = tmp["file"].astype(str).str.replace("_poses.csv", "", regex=False)
    tmp["video_key"] = tmp["video_label"].map(normalize_name)
    tmp["MES_60"] = pd.to_numeric(tmp["MES_60"], errors="coerce")

    meta = tmp["video_label"].apply(parse_video_meta)
    meta_df = pd.DataFrame(meta.tolist())
    tmp = pd.concat([tmp.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    merged = tmp.merge(
        fis_df[["Bib", "Gender", "Competition", "official_score", "Name"]],
        on=["Bib", "Gender", "Competition"],
        how="left",
    )

    return merged


# ------------------------------------------------------------
# Plot-Helfer
# ------------------------------------------------------------
def metric_config() -> Dict[str, Dict[str, str]]:
    return {
        "rhythmus": {"label": "Rhythmus", "y": "x (px)"},
        "symmetrie": {"label": "Symmetrie", "y": "Y"},
        "stabilitaet": {"label": "Stabilität", "y": "θ (°)"},
        "smoothness": {"label": "Smoothness", "y": "Jerk"},
        "kompaktheit": {"label": "Kompaktheit", "y": "d_norm"},
        "line_integrity": {"label": "Line Integrity", "y": "φ (°)"},
    }


def make_metric_figure(
    df: pd.DataFrame,
    metric_name: str,
    current_x: float,
    x_mode: str = "timestamp_sec",
    height: int = 150,
) -> go.Figure:
    cfg = metric_config().get(metric_name, {"label": metric_name, "y": "Wert"})
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[x_mode],
            y=df[metric_name],
            mode="lines",
            name=cfg["label"],
            line=dict(width=2),
        )
    )
    fig.add_vline(x=current_x, line_width=3, line_dash="dash", line_color="red")

    fig.update_layout(
        margin=dict(l=8, r=8, t=40, b=8),
        height=height,
        title=cfg["label"],
        xaxis_title="Zeit (s)" if x_mode == "timestamp_sec" else "Frame",
        yaxis_title=cfg["y"],
        showlegend=False,
    )
    return fig


def make_scatter_figure(scatter_df: pd.DataFrame, selected_key: str) -> go.Figure:
    fig = go.Figure()

    base_df = scatter_df.dropna(subset=["official_score", "MES_60"]).copy()
    base_df = base_df[base_df["official_score"] >= 40]

    women_df = base_df[base_df["Gender"] == "W"].copy()
    men_df = base_df[base_df["Gender"] == "M"].copy()

    def add_points(df: pd.DataFrame, name: str, color: str):
        if df.empty:
            return

        fig.add_trace(
            go.Scatter(
                x=df["official_score"],
                y=df["MES_60"],
                mode="markers",
                name=name,
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
                    "MES: %{y:.2f}<extra></extra>"
                ),
            )
        )

    def add_regression(df: pd.DataFrame, color: str, label: str):
        reg_df = df.dropna(subset=["official_score", "MES_60"]).copy()

        if len(reg_df) < 2:
            return None

        x = reg_df["official_score"].to_numpy(dtype=float)
        y = reg_df["MES_60"].to_numpy(dtype=float)

        m, b = np.polyfit(x, y, 1)

        x_line = np.linspace(reg_df["official_score"].min(), reg_df["official_score"].max(), 100)
        y_line = m * x_line + b

        r = np.corrcoef(x, y)[0, 1]
        r2 = r ** 2

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{label} Trend (R²={r2:.2f})",
                hoverinfo="skip",
            )
        )
        return r2

    add_points(women_df, "Frauen", "#1f77b4")
    add_points(men_df, "Männer", "#2ca02c")

    r2_w = add_regression(women_df, "#1f77b4", "Frauen")
    r2_m = add_regression(men_df, "#2ca02c", "Männer")

    selected_df = base_df[base_df["video_key"] == selected_key]
    if not selected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_df["official_score"],
                y=selected_df["MES_60"],
                mode="markers",
                name="Auswahl",
                text=selected_df["video_label"],
                marker=dict(size=14, color="red", line=dict(width=2, color="darkred")),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Offizieller Score: %{x:.2f}<br>"
                    "MES: %{y:.2f}<extra></extra>"
                ),
            )
        )

    title = "MES vs. offizieller Score"
    parts = []
    if r2_w is not None:
        parts.append(f"Frauen R²={r2_w:.2f}")
    if r2_m is not None:
        parts.append(f"Männer R²={r2_m:.2f}")
    if parts:
        title += " (" + " | ".join(parts) + ")"

    fig.update_layout(
        title=title,
        xaxis_title="Offizieller Score",
        yaxis_title="MES_60",
        margin=dict(l=8, r=8, t=36, b=8),
        height=260,
        legend=dict(
            orientation="v",
            y=0.5,
            yanchor="middle",
            x=1.02,
            xanchor="left",
        ),
    )

    if not base_df.empty:
        y_min = base_df["MES_60"].min()
        y_max = base_df["MES_60"].max()
        fig.update_yaxes(range=[y_min - 0.5, y_max + 0.5], autorange=False)

    return fig


def make_radar_figure(summary: Dict[str, float]) -> go.Figure:
    labels = ["R", "Y", "S", "M", "C", "L"]

    label_names = {
        "R": "Rhythmus",
        "Y": "Symmetrie",
        "S": "Stabilität",
        "M": "Smoothness",
        "C": "Kompaktheit",
        "L": "Line Integrity",
    }

    values = [
        float(summary.get(k, 0)) * 10 if pd.notna(summary.get(k, np.nan)) else 0
        for k in labels
    ]

    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    hover_text = [f"{label_names[l]}: {v:.2f}" for l, v in zip(labels, values)]
    hover_text_closed = hover_text + [hover_text[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            hovertext=hover_text_closed,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        margin=dict(l=5, r=5, t=5, b=5),
        height=220,
    )

    return fig


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown("## Moguls Elegance Score Dashboard")

scores_df = load_scores()
poses_df = load_poses()
fis_df = load_fis_scores()
videos_df = discover_video_files(VIDEO_DIR)

run_index = build_run_index(scores_df, poses_df, videos_df)
scatter_df = build_scatter_df(scores_df, fis_df)

with st.sidebar:
    st.header("Steuerung")

    if run_index.empty:
        st.error("Keine Daten gefunden. Lege all_poses.csv und mes_scores.csv unter ./daten ab.")
        st.stop()

    st.info("zusätziche Auswahl: Analysedaten Subsequent / YOLOv8")
    selected_label = st.selectbox("Videosequenz", run_index["label"].tolist())
    selected_row = run_index[run_index["label"] == selected_label].iloc[0]
    selected_key = selected_row["video_key"]

    summary = get_summary_for_key(scores_df, selected_key, scatter_df)
    pose_subset = get_pose_subset(poses_df, selected_key)

    if pose_subset.empty:
        st.error("Für diese Videosequenz wurden keine Pose-Daten gefunden.")
        st.stop()

    ts_df = compute_plot_timeseries(pose_subset)
    max_index = max(len(ts_df) - 1, 0)

    if "current_idx" not in st.session_state:
        st.session_state.current_idx = min(0, max_index)
    if "last_video_key" not in st.session_state:
        st.session_state.last_video_key = selected_key
    if "frame_mode" not in st.session_state:
        st.session_state.frame_mode = False

    if st.session_state.last_video_key != selected_key:
        start_idx = min(0, max_index)
        st.session_state.current_idx = start_idx
        st.session_state.frame_slider = start_idx
        st.session_state.last_video_key = selected_key
        st.session_state.frame_mode = False

    if st.session_state.current_idx > max_index:
        st.session_state.current_idx = max_index

    st.toggle("Einzelbilder", key="frame_mode")
    
    if "frame_slider" not in st.session_state:
        st.session_state.frame_slider = st.session_state.current_idx
    
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
    
        st.slider(
            "Frame",
            min_value=0,
            max_value=max_index,
            step=1,
            key="frame_slider",
        )
    
        st.session_state.current_idx = st.session_state.frame_slider

    x_mode = st.radio(
        "X-Achse",
        ["timestamp_sec", "frame"],
        horizontal=True,
        format_func=lambda x: "Zeit" if x == "timestamp_sec" else "Frame",
    )

    metric_options = [c for c in metric_config().keys() if c in ts_df.columns]

    metric_1 = st.selectbox(
        "Metrik 1",
        metric_options,
        index=0,
        format_func=lambda x: metric_config()[x]["label"],
    )

    metric_2 = st.selectbox(
        "Metrik 2",
        metric_options,
        index=min(1, len(metric_options) - 1),
        format_func=lambda x: metric_config()[x]["label"],
    )

    st.markdown("### Info")
    st.info("folgt von Simon")

current_idx = min(st.session_state.current_idx, max_index)
current_row = ts_df.iloc[current_idx]

# rote Linie nur im Einzelbildmodus synchronisieren
if st.session_state.frame_mode:
    current_x = float(current_row[x_mode])
else:
    current_x = 0


# ------------------------------------------------------------
# Layout: stabiles Raster
# ------------------------------------------------------------
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
                st.caption(
                    f"Frame {frame_number} | "
                    f"t = {ts_df.iloc[current_idx]['timestamp_sec']:.2f} s"
                )
            else:
                st.warning(f"Frame {frame_number} konnte nicht geladen werden.")
        else:
            st.info("Kein passendes Video im Video-Ordner gefunden.")

    else:
        st.markdown("### Wiedergabe")

        if video_path and Path(video_path).exists():
            st.video(str(video_path))
        else:
            st.info("Kein passendes Video im Video-Ordner gefunden.")

with top_right:
    st.markdown("### Metriken")
    fig1 = make_metric_figure(ts_df, metric_1, current_x=current_x, x_mode=x_mode, height=120)
    fig2 = make_metric_figure(ts_df, metric_2, current_x=current_x, x_mode=x_mode, height=120)

    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

bottom_left, bottom_right = st.columns([0.9, 1.1], gap="medium")

with bottom_left:
    radar_col, mes_col = st.columns([1, 1.25], gap="medium")

    with radar_col:
        st.markdown("### Radar")
        st.plotly_chart(
            make_radar_figure(summary),
            use_container_width=True,
            config={"displayModeBar": False},
        )

with mes_col:
    st.markdown("### MES")

    mes_items = [
        ("Rhythmus", summary.get("R", np.nan) * 10 if pd.notna(summary.get("R", np.nan)) else np.nan),
        ("Symmetrie", summary.get("Y", np.nan) * 10 if pd.notna(summary.get("Y", np.nan)) else np.nan),
        ("Stabilität", summary.get("S", np.nan) * 10 if pd.notna(summary.get("S", np.nan)) else np.nan),
        ("Smoothness", summary.get("M", np.nan) * 10 if pd.notna(summary.get("M", np.nan)) else np.nan),
        ("Kompaktheit", summary.get("C", np.nan) * 10 if pd.notna(summary.get("C", np.nan)) else np.nan),
        ("Line Integrity", summary.get("L", np.nan) * 10 if pd.notna(summary.get("L", np.nan)) else np.nan),
    ]

    total_mes = summary.get("MES_60", np.nan)
    official_score = summary.get("official_score", np.nan)

    rows_html = ""
    for label, value in mes_items:
        value_str = f"{value:.3f}" if pd.notna(value) else "–"
        rows_html += (
            f'<div class="mes-row">'
            f'<span class="mes-label">{label}</span>'
            f'<span class="mes-value">{value_str}</span>'
            f'</div>'
        )

    total_str = f"{total_mes:.3f}" if pd.notna(total_mes) else "–"
    official_str = f"{official_score:.1f}" if pd.notna(official_score) else "–"

    mes_html = textwrap.dedent(f"""
    <div class="mes-card">
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
    """)

    st.markdown(mes_html, unsafe_allow_html=True)
    
with bottom_right:
    st.markdown("### Score")
    scatter_fig = make_scatter_figure(scatter_df, selected_key=selected_key)
    st.plotly_chart(scatter_fig, use_container_width=True, config={"displayModeBar": False})

