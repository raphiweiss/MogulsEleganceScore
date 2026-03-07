# CViz_FS26_Moguls_TurnAnalysis
Skeleton-based tracking to evaluate competitors’ turn performance in moguls, aligned with the FIS Freestyle Skiing Judging Handbook (JH 6204.1 Turns, JH 6204.2 Deductions). Supports objective analysis of turn quality and rule-compliant deductions.

## Setup Videos
Die Videos sind nicht im Repository. Lade sie von OneDrive:
[\[Link zu OneDrive\]](https://fhgraubuenden-my.sharepoint.com/my?id=%2Fpersonal%2Fgitzaaron%5Ffhgr%5Fch%2FDocuments%2FComputer%5FVisualization%5FFS2026%2FVideos&viewid=e3043235%2D4e6e%2D419b%2Dabf0%2D7e5e0b1328ad)

Dann erstelle einen Symlink im Projektverzeichnis:
```bash
ln -s "/pfad/zu/deinen/videos" videos
```

## Installation YOLOv8
Die folgenden Schritte beziehen sich auf Linux bzw. Ubuntu. Windows oder Mac spezifische Befehle können ergänzt werden. 

1. Virtual Environment erstellen:
```bash
python3 -m venv yolo_env
```

2. Virtual Environment aktivieren:
```bash
source ~/venvs/yolo_env/bin/activate
```

3. YOLOv8 installieren:
```bash
pip install ultralytics
```

## Tests
YOLO
```bash
yolo version
```
Person Detection
```bash
yolo predict model=yolov8n.pt source="videos/Einzelclips Frauen/Final_2/Moguls_Frauen_Final_2_01.mp4" save=True
```
Pose Estimation
```bash
yolo pose predict model=yolov8n-pose.pt source="videos/Einzelclips Frauen/Final_2/Moguls_Frauen_Final_2_01.mp4" save=True
```
Tracking
```bash
yolo track model=yolov8n.pt source="videos/Einzelclips Frauen/Final_2/Moguls_Frauen_Final_2_01.mp4" save=True
```
## Moguls Elegance Score (MES)
moguls_elegance_score.ipynb implementiert einen objektiven Eleganz-Score auf Basis der extrahierten Pose-Keypoints. Das Modell operationalisiert die Turn-Kriterien aus JH 6204.1 (Turns — 60%) in sechs Komponenten:

| Kürzel | Komponente | FIS-Bezug |
|--------|-----------|-----------|
| R | Rhythmus | Rhythmic changes in direction of travel |
| S | Stabilität | Upper body / Breaks in balance |
| M | Smoothness | No skidding / lateral sliding |
| Y | Symmetrie | Movements symmetrical and equal side to side |
| C | Bein-Kompaktheit | Legs should be together |
| L | Line Integrity | Skiing in the fall line |

Gesamtformel: MES_60 = 10·R + 10·S + 10·M + 10·Y + 10·C + 10·L (max. 60 Punkte)
