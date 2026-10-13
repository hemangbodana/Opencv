import numpy as np
import cv2 as cv
import os

# -----------------------------------------
# Step 1: Load the video
# -----------------------------------------
video_path = r"Photos\detect2.mp4"

# cap = cv.VideoCapture(video_path)
cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise FileNotFoundError("Video not found or could not be opened.")

# -----------------------------------------
# Step 2: Load the pretrained YOLOv8 model
# -----------------------------------------
model_path = r"models\yolov8n.onnx"

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

net = cv.dnn.readNetFromONNX(model_path)

if net.empty():
    raise RuntimeError("Failed to load YOLOv8 ONNX model.")

print("YOLOv8 Model Loaded")

# -----------------------------------------
# Step 3: Define class names and thresholds
# -----------------------------------------
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

confidence_threshold = 0.25
nms_threshold = 0.45
input_size = 640


# -----------------------------------------
# Frame skipping configuration
# -----------------------------------------
frame_skip = 10
# 0 = process every frame
# 1 = skip 1 frame between detections
# 2 = skip 2 frames between detections

frame_count = 0
processed_frames = 0
skipped_frames = 0

# -----------------------------------------
# Latency and FPS statistics
# -----------------------------------------
total_inference_latency = 0.0
last_latency_ms = 0.0

import time

start_time = time.perf_counter()

# -----------------------------------------
# Step 4: Process each video frame
# -----------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # -----------------------------------------
    # Step 1: Frame skipping
    # -----------------------------------------
    if frame_skip > 0 and (frame_count - 1) % (frame_skip + 1) != 0:
        skipped_frames += 1
        continue

    original_height, original_width = frame.shape[:2]

    # -----------------------------------------
    # Step 2: Start latency timer
    # -----------------------------------------
    inference_start = time.perf_counter()

    # -----------------------------------------
    # Step 3: Preprocess frame
    # -----------------------------------------
    blob = cv.dnn.blobFromImage(
        image=frame,
        scalefactor=1.0 / 255.0,
        size=(640, 640),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    # -----------------------------------------
    # Step 4: YOLO inference
    # -----------------------------------------
    net.setInput(blob)
    outputs = net.forward()

    # -----------------------------------------
    # Step 5: Decode predictions
    # -----------------------------------------
    predictions = outputs[0].T

    boxes = []
    confidences = []
    class_ids = []

    confidence_threshold = 0.25
    nms_threshold = 0.45

    for prediction in predictions:

        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < confidence_threshold:
            continue

        if not 0 <= class_id < len(class_names):
            continue

        cx, cy, w, h = prediction[:4]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1 *= original_width / 640
        x2 *= original_width / 640
        y1 *= original_height / 640
        y2 *= original_height / 640

        x1 = max(0, min(int(x1), original_width - 1))
        y1 = max(0, min(int(y1), original_height - 1))
        x2 = max(0, min(int(x2), original_width - 1))
        y2 = max(0, min(int(y2), original_height - 1))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            continue

        boxes.append([x1, y1, box_width, box_height])
        confidences.append(confidence)
        class_ids.append(class_id)

    # -----------------------------------------
    # Step 6: Apply NMS
    # -----------------------------------------
    indices = cv.dnn.NMSBoxes(
        boxes,
        confidences,
        confidence_threshold,
        nms_threshold
    )

    if len(indices) > 0:
        indices = np.array(indices).flatten().tolist()
    else:
        indices = []

    # -----------------------------------------
    # Step 7: Calculate detection latency
    # -----------------------------------------
    last_latency_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    total_inference_latency += last_latency_ms
    processed_frames += 1

    avg_latency_ms = (
        total_inference_latency / processed_frames
    )

    elapsed_time = time.perf_counter() - start_time
    effective_fps = processed_frames / elapsed_time

    # -----------------------------------------
    # Step 8: Draw detections
    # -----------------------------------------
    output_frame = frame.copy()

    for i in indices:

        x, y, w, h = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]

        label = (
            f"{class_names[class_id]} "
            f"{confidence * 100:.1f}%"
        )

        cv.rectangle(
            output_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv.putText(
            output_frame,
            label,
            (x, max(y - 10, 20)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -----------------------------------------
    # Step 9: Display performance statistics
    # -----------------------------------------
    cv.putText(
        output_frame,
        f"Frame: {frame_count}",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    cv.putText(
        output_frame,
        f"Latency: {last_latency_ms:.1f} ms",
        (20, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    cv.putText(
        output_frame,
        f"Avg latency: {avg_latency_ms:.1f} ms",
        (20, 90),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    cv.putText(
        output_frame,
        f"Effective FPS: {effective_fps:.1f}",
        (20, 120),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    cv.putText(
        output_frame,
        f"Skipped: {skipped_frames}",
        (20, 150),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    cv.imshow("YOLOv8 Video Detection", output_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------------------
# Step 10: Final statistics
# -----------------------------------------
elapsed_time = time.perf_counter() - start_time

print("\n----- Performance Summary -----")
print("Total frames read:", frame_count)
print("Frames processed:", processed_frames)
print("Frames skipped:", skipped_frames)

if processed_frames > 0:
    print(f"Average latency: {avg_latency_ms:.2f} ms")
    print(
        f"Average inference FPS: "
        f"{1000 / avg_latency_ms:.2f}"
        if avg_latency_ms > 0 else "Average inference FPS: N/A"
    )

print(f"Effective processing FPS: {processed_frames / elapsed_time:.2f}")

cap.release()
cv.destroyAllWindows()
