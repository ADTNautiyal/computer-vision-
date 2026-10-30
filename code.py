import cv2
import numpy as np
from ultralytics import YOLO

# Updated model and correct video path to fix the FileNotFoundError
model = YOLO("yolov8s-pose.pt")
video_path = "/content/Aditi.mp4"
output_path = "/content/output_pose.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_width = width - (width % 2)
out_height = height - (height % 2)

if out_width <= 0 or out_height <= 0:
    cap.release()
    raise ValueError("Invalid input video dimensions.")

out = None
for codec in ["mp4v", "avc1", "H264"]:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (out_width, out_height)
    )
    if writer.isOpened():
        out = writer
        print(f"Using video codec: {codec}")
        break
    writer.release()

if out is None:
    cap.release()
    raise RuntimeError("Could not create MP4 output.")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame.shape[1] != out_width or frame.shape[0] != out_height:
            frame = cv2.resize(frame, (out_width, out_height))

        result = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )[0]

        black_frame = np.zeros_like(frame)

        result.orig_img = black_frame
        annotated_frame = result.plot(
            img=black_frame.copy(),
            boxes=False,
            kpt_line=True,
            kpt_radius=4,
            line_width=5,
        )

        out.write(annotated_frame)
        frame_count += 1

finally:
    cap.release()
    out.release()

print(f"Processed {frame_count} frames.")
print(f"Output saved to: {output_path}")