"""
Moguls Pose Extraction Pipeline
================================
Verarbeitet alle Videos in videos/all/ mit YOLOv8 Pose Estimation.
CSV und annotiertes Video werden im selben Durchlauf erstellt –
Track-IDs sind dadurch garantiert synchron.

Output pro Run (runs/<timestamp>_<model>_conf<xx>/):
  data/
    metadata.json
    all_poses.csv
    <videoname>_poses.csv  (je Video)
  videos/
    <videoname>.mp4        (annotiert mit Bounding Box + Keypoints)

Keypoints (COCO-Format, 17 Punkte):
  0=Nose, 1=Left Eye, 2=Right Eye, 3=Left Ear, 4=Right Ear,
  5=Left Shoulder, 6=Right Shoulder, 7=Left Elbow, 8=Right Elbow,
  9=Left Wrist, 10=Right Wrist, 11=Left Hip, 12=Right Hip,
  13=Left Knee, 14=Right Knee, 15=Left Ankle, 16=Right Ankle
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ── Konfiguration ──────────────────────────────────────────────────────────────

VIDEO_DIR  = Path("videos/all")   # relativer Pfad; Skript vom Projektordner starten
RUNS_DIR   = Path("runs")
MODEL_PATH = "yolov8n-pose.pt"    # Alternativen: yolov8s-pose.pt, yolov8m-pose.pt

CONF     = 0.20   # Konfidenz-Schwelle
MAX_DET  = 1      # Nur 1 Person pro Frame (Moguls = Einzelläufer)
IMG_SIZE = 1280   # Höhere Auflösung = bessere Keypoints, aber langsamer
TRACKER  = "bytetrack.yaml"

# ── Device Setup ───────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device   = "0"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU gefunden: {gpu_name}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device   = "cpu"
    gpu_name = "cpu"
    print("⚠ Keine GPU gefunden, verwende CPU (wird deutlich langsamer sein)")

# ── Keypoint-Namen ─────────────────────────────────────────────────────────────

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def build_header():
    header = ["video", "frame", "track_id",
              "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_conf"]
    for name in KEYPOINT_NAMES:
        header += [f"{name}_x", f"{name}_y", f"{name}_conf"]
    return header


def process_video(video_path: Path, model: YOLO, writer, video_name: str, video_out_path: Path):
    """
    Einziger Durchlauf pro Video:
      - Keypoints werden pro Frame in den CSV-Writer geschrieben
      - Dasselbe Frame wird via r.plot() annotiert und ins Output-Video geschrieben
    Track-IDs in CSV und Video sind dadurch garantiert identisch.
    """
    # Video-Eigenschaften auslesen für den Writer
    cap = cv2.VideoCapture(str(video_path))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(video_out_path), fourcc, fps_in, (w, h))

    results = model.track(
        source=str(video_path),
        conf=CONF,
        max_det=MAX_DET,
        imgsz=IMG_SIZE,
        tracker=TRACKER,
        device=device,
        stream=True,
        save=False,    # wir schreiben das Video selbst via r.plot()
        verbose=False,
    )

    frame_count = 0

    for frame_idx, r in enumerate(results):
        frame_count += 1

        # Annotiertes Frame rendern und ins Video schreiben
        annotated = r.plot()  # numpy array (BGR), identisch mit YOLOs save=True Output
        out.write(annotated)

        # Keypoints extrahieren
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            continue

        ids       = r.boxes.id.cpu().numpy()
        keypoints = r.keypoints.data.cpu().numpy()  # shape: (N, 17, 3)
        boxes     = r.boxes.xyxy.cpu().numpy()       # shape: (N, 4)
        box_confs = r.boxes.conf.cpu().numpy()       # shape: (N,)

        for i in range(len(ids)):
            row = [
                video_name,
                frame_idx,
                int(ids[i]),
                *[round(float(v), 2) for v in boxes[i]],
                round(float(box_confs[i]), 4),
            ]
            for kp in keypoints[i]:
                row += [round(float(kp[0]), 2),
                        round(float(kp[1]), 2),
                        round(float(kp[2]), 4)]
            writer.writerow(row)

    out.release()
    return frame_count

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Run-Ordner anlegen
    run_start = datetime.now()
    model_tag = MODEL_PATH.replace(".pt", "").replace("yolov8", "v8")
    run_name  = f"{run_start.strftime('%Y-%m-%d_%H%M%S')}_{model_tag}_conf{int(CONF*100):02d}"
    run_dir   = RUNS_DIR / run_name
    data_dir  = run_dir / "data"
    video_dir = run_dir / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun: {run_name}")
    print(f"  data/   → {data_dir.resolve()}")
    print(f"  videos/ → {video_dir.resolve()}")

    # Videos finden
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos = sorted([f for f in VIDEO_DIR.iterdir()
                     if f.suffix.lower() in video_extensions])

    if not videos:
        print(f"\nKeine Videos gefunden in: {VIDEO_DIR}")
        return

    print(f"\nGefundene Videos: {len(videos)}")
    for v in videos:
        print(f"  • {v.name}")

    # Modell einmal laden
    print(f"\nLade Modell: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✓ Modell geladen\n")

    # Verarbeitung
    combined_csv = data_dir / "all_poses.csv"
    header       = build_header()
    total_frames = 0
    video_stats  = []
    t_start      = time.time()

    with open(combined_csv, "w", newline="") as f_all:
        writer_all = csv.writer(f_all)
        writer_all.writerow(header)

        for video_path in videos:
            video_name    = video_path.stem
            video_out     = video_dir / f"{video_name}.mp4"
            t_video       = time.time()
            print(f"▶ {video_name}")

            per_video_csv = data_dir / f"{video_name}_poses.csv"
            with open(per_video_csv, "w", newline="") as f_vid:
                writer_vid = csv.writer(f_vid)
                writer_vid.writerow(header)

                class DualWriter:
                    def writerow(self, row):
                        writer_vid.writerow(row)
                        writer_all.writerow(row)

                frames = process_video(video_path, model, DualWriter(), video_name, video_out)

            elapsed = time.time() - t_video
            fps     = frames / elapsed if elapsed > 0 else 0
            total_frames += frames
            video_stats.append({"video": video_name, "frames": frames,
                                 "duration_s": round(elapsed, 2), "fps": round(fps, 1)})
            print(f"  ✓ {frames} Frames in {elapsed:.1f}s ({fps:.1f} fps)")

    total_time = time.time() - t_start

    # Metadata schreiben
    metadata = {
        "run_name":         run_name,
        "timestamp":        run_start.strftime("%Y-%m-%d %H:%M:%S"),
        "model":            MODEL_PATH,
        "conf":             CONF,
        "max_det":          MAX_DET,
        "img_size":         IMG_SIZE,
        "tracker":          TRACKER,
        "device":           gpu_name,
        "cuda_available":   torch.cuda.is_available(),
        "videos_processed": len(videos),
        "total_frames":     total_frames,
        "duration_seconds": round(total_time, 2),
        "avg_fps":          round(total_frames / total_time, 1) if total_time > 0 else 0,
        "per_video":        video_stats,
    }
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Zusammenfassung
    print(f"\n{'─'*50}")
    print(f"Fertig! {len(videos)} Videos, {total_frames} Frames total")
    print(f"Gesamtzeit: {total_time:.1f}s ({total_frames/total_time:.1f} fps gesamt)")
    print(f"\nOutputs:")
    print(f"  {data_dir}/")
    print(f"    metadata.json")
    print(f"    all_poses.csv")
    print(f"    <videoname>_poses.csv")
    print(f"  {video_dir}/")
    print(f"    <videoname>.mp4")
    print(f"\nTipp für Pandas:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv(r'{combined_csv}')")


if __name__ == "__main__":
    main()
