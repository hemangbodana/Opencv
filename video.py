
import numpy as np
import cv2 as cv
import os

# -----------------------------------------
# Step 1: Load the video
# -----------------------------------------
video_path = r"C:\Users\Hemang.bodana\Downloads\ObjectDetection_Yolo\ObjectDetection_Yolo\Photos\detect1.mp4"

cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("Video not found or could not be opened.")

# -----------------------------------------
# Step 2: Load the pretrained YOLOv8 model
# -----------------------------------------
model_path = r"C:\Users\Hemang.bodana\Downloads\ObjectDetection_Yolo\ObjectDetection_Yolo\models\yolov8n.onnx"

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
# Step 4: Process each video frame
# -----------------------------------------
while True:

    # Read the next frame
    ret, frame = cap.read()

    # Stop when the video ends
    if not ret:
        break

    original_height, original_width = frame.shape[:2]

    # -----------------------------------------
    # Step 5: Preprocess the frame
    # -----------------------------------------
    blob = cv.dnn.blobFromImage(
        image=frame,
        scalefactor=1.0 / 255.0,
        size=(input_size, input_size),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    # -----------------------------------------
    # Step 6: Run YOLO inference
    # -----------------------------------------
    net.setInput(blob)
    outputs = net.forward()

    # YOLOv8n output: (1, 84, 8400)
    # Transpose to (8400, 84)
    predictions = outputs[0].T

    boxes = []
    confidences = []
    class_ids = []

    # -----------------------------------------
    # Step 7: Decode and filter predictions
    # -----------------------------------------
    for prediction in predictions:

        # Get the 80 class scores
        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < confidence_threshold:
            continue

        if not 0 <= class_id < len(class_names):
            continue

        # Bounding box center and dimensions
        cx, cy, w, h = prediction[:4]

        # Convert center coordinates to corners
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Scale coordinates to the original frame
        x1 *= original_width / input_size
        x2 *= original_width / input_size
        y1 *= original_height / input_size
        y2 *= original_height / input_size

        # Clamp coordinates to frame boundaries
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
    # Step 8: Apply Non-Maximum Suppression
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
    # Step 9: Draw final detections
    # -----------------------------------------
    output_frame = frame.copy()

    for i in indices:

        x, y, w, h = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]

        label_name = class_names[class_id]
        label = f"{label_name} {confidence * 100:.1f}%"

        # Draw bounding box
        cv.rectangle(
            output_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Calculate label dimensions
        (text_w, text_h), baseline = cv.getTextSize(
            label,
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        label_y = max(y, text_h + baseline)

        # Filled label background
        cv.rectangle(
            output_frame,
            (x, label_y - text_h - baseline),
            (x + text_w, label_y),
            (0, 255, 0),
            cv.FILLED
        )

        # Draw label
        cv.putText(
            output_frame,
            label,
            (x, label_y - baseline),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv.LINE_AA
        )

    # Display detection count
    cv.putText(
        output_frame,
        f"Objects detected: {len(indices)}",
        (20, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # Display video
    cv.imshow("YOLOv8 Video Object Detection", output_frame)

    # Press q to exit
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------------------
# Step 10: Release resources
# -----------------------------------------
cap.release()
cv.destroyAllWindows()

print("Video processing completed.")
