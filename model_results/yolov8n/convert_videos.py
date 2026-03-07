from pathlib import Path
import os

video_dir = Path("videos")

for video in video_dir.glob("*.mp4"):
    
    if "_web" in video.stem:
        continue

    output = video.with_name(video.stem + "_web.mp4")

    cmd = (
        f'ffmpeg -y -i "{video}" '
        f'-c:v h264_mf '
        f'-b:v 8M '
        f'-pix_fmt yuv420p '
        f'-movflags +faststart '
        f'"{output}"'
    )

    print("Konvertiere:", video.name)
    os.system(cmd)

print("Fertig.")


