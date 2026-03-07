"""
Playground – Parameter-Experimente
====================================
Für schnelle qualitative Tests auf wenigen Videos.
Outputs landen in playground_out/<experiment_name>/

Typischer Workflow:
  1. EXPERIMENT_NAME und TEST_VIDEOS anpassen
  2. Parameter verändern (CONF, IMG_SIZE, MODEL_PATH, ...)
  3. Skript starten → annotierte Videos + CSV anschauen
  4. Repeat
"""

from pathlib import Path
from ultralytics import YOLO

# ── Was willst du testen? ──────────────────────────────────────────────────────

EXPERIMENT_NAME = "conf035_medium_model"   # ← sprechenden Namen wählen!

TEST_VIDEOS = [                            # haben mehrere ID's in 2026-03-02_220528_v8n-pose_conf20
    "videos/all/Moguls_Frauen_Final_1_07.mp4", 
    "videos/all/Moguls_Männer_Final_1 - 11.mp4",
    "videos/all/Moguls_Männer_Final_1 - 23.mp4",
]

# ── Parameter zum Experimentieren ─────────────────────────────────────────────

MODEL_PATH = "yolov8n-pose.pt"   # Alternativen: yolov8s-pose.pt, yolov8m-pose.pt
CONF       = 0.35                # ausprobieren: 0.20 / 0.35 / 0.50
MAX_DET    = 1
IMG_SIZE   = 1280                # ausprobieren: 640 / 960 / 1280
TRACKER    = "bytetrack.yaml"    # Alternative: botsort.yaml

"""
ByteTrack wird als YAML-Datei bei der Installation von Ultralytics mitgeliefert und liegt typischerweise unter:
~/.../yolo_env/lib/python3.12/site-packages/ultralytics/cfg/trackers/bytetrack.yaml
find / -name "bytetrack.yaml" 2>/dev/null

Die Datei sollte aber nicht direkt bearbeitet werden, sondern eine Kopie in den Projektordner angelegt werden:
cp <pfad>/bytetrack.yaml .

Und dann im Skript auf die lokale Kopie zeigen:
TRACKER = "bytetrack.yaml"  # bleibt gleich, solange sie im Projektordner liegt
Varianten wie z.B. bytetrack_highbuffer.yaml bieten sich ebenfalls an.
"""

# ── Output ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("playground_out") / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Run ───────────────────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)

for video_path in TEST_VIDEOS:
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"⚠ Video nicht gefunden: {video_path}")
        continue

    print(f"▶ {video_path.name}  [conf={CONF}, imgsz={IMG_SIZE}, model={MODEL_PATH}]")

    model.track(
        source=str(video_path),
        conf=CONF,
        max_det=MAX_DET,
        imgsz=IMG_SIZE,
        tracker=TRACKER,
        device="0",
        save=True,                        # annotiertes Video speichern
        save_dir=str(OUTPUT_DIR),         # direkt in den Experiment-Ordner
        name=video_path.stem,             # Unterordner pro Video
        exist_ok=True,
        verbose=False,
    )
    print(f"  ✓ Video gespeichert in: {OUTPUT_DIR / video_path.stem}")

print(f"\nAlle Videos in: {OUTPUT_DIR.resolve()}")
print("Jetzt die annotierten Videos anschauen und Parameter beurteilen.")
