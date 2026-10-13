import numpy as np
import cv2 as cv
import os

image = cv.imread(r"Photos\img2.jpg")

if image is None:
    raise FileNotFoundError("Image not found.")

print("Imgae shape: ",image.shape)

model_path = r"C:\Users\Hemang.bodana\Downloads\ObjectDetection_Yolo\ObjectDetection_Yolo\models\yolov8n.onnx"
net = cv.dnn.readNetFromONNX(model_path)

if net.empty():
    raise RuntimeWarning("Failed to load")

print("Yolo Model Loaded ")


blob = cv.dnn.blobFromImage(
    image=image,
    scalefactor=1.0/255.0,
    size=(640,640),
    mean=(0,0,0),
    swapRB=True,
    crop=False
)

print("Original image shape:", image.shape)
print("Blob shape:", blob.shape)
print("Blob data type:", blob.dtype)
print("Blob minimum:", blob.min())
print("Blob maximum:", blob.max())

net.setInput(blob)
outputs = net.forward()

# Inspect the output
print("Blob shape:", blob.shape)
print("Output type:", type(outputs))
print("Number of output tensors:", len(outputs))
print("Output shape:", outputs.shape)


print("First candidate prediction:")
print(outputs[0, :, 0])



# Step 5: Decode YOLO predictions

# Convert output from (1, 84, 8400) to (8400, 84)
predictions = outputs[0].T

# Set the confidence threshold
confidence_threshold = 0.25

# Original image dimensions
original_height, original_width = image.shape[:2]

# Lists to store valid detections
boxes = []
confidences = []
class_ids = []

# Process each candidate prediction
for prediction in predictions:

    # Extract the 80 class scores
    class_scores = prediction[4:]

    # Find the class with the highest score
    class_id = int(np.argmax(class_scores))
    confidence = float(class_scores[class_id])

    # Ignore low-confidence predictions
    if confidence < confidence_threshold:
        continue

    # Extract bounding-box center and dimensions
    cx, cy, w, h = prediction[:4]

    # Convert center coordinates to corner coordinates
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Map coordinates from 640x640 to the original image
    x1 *= original_width / 640
    x2 *= original_width / 640
    y1 *= original_height / 640
    y2 *= original_height / 640

    # Keep coordinates inside the image
    x1 = max(0, min(int(x1), original_width - 1))
    y1 = max(0, min(int(y1), original_height - 1))
    x2 = max(0, min(int(x2), original_width - 1))
    y2 = max(0, min(int(y2), original_height - 1))

    # OpenCV NMSBoxes expects [x, y, width, height]
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        continue

    # Save the valid detection
    boxes.append([x1, y1, box_width, box_height])
    confidences.append(confidence)
    class_ids.append(class_id)

print("Total candidate predictions:", len(predictions))
print("Detections after confidence filtering:", len(boxes))
print("First few boxes:", boxes[:5])
print("First few confidence scores:", confidences[:5])
print("First few class IDs:", class_ids[:5])



# Step 6: Non-Maximum Suppression

# Confidence threshold for NMS
confidence_threshold = 0.25

# IoU threshold for suppressing overlapping boxes
nms_threshold = 0.45

# Apply NMS
indices = cv.dnn.NMSBoxes(
    boxes,
    confidences,
    confidence_threshold,
    nms_threshold
)

# Convert returned indices into a simple list
if len(indices) > 0:
    indices = np.array(indices).flatten().tolist()
else:
    indices = []

print("Detections before NMS:", len(boxes))
print("Detections after NMS:", len(indices))

# Store only the detections selected by NMS
final_boxes = [boxes[i] for i in indices]
final_confidences = [confidences[i] for i in indices]
final_class_ids = [class_ids[i] for i in indices]

print("Final boxes:", final_boxes)
print("Final confidence scores:", final_confidences)
print("Final class IDs:", final_class_ids)



# Step 7: Draw bounding boxes and labels

# COCO class names for YOLOv8n
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

# Create a copy so the original image remains unchanged
output_image = image.copy()

# Draw every detection that survived NMS
for box, confidence, class_id in zip(
    final_boxes, final_confidences, final_class_ids
):
    # Extract bounding-box coordinates
    x, y, w, h = box

    # Get the object's class name
    label_name = class_names[class_id]

    # Format the label with confidence percentage
    label = f"{label_name} {confidence * 100:.1f}%"

    # Draw the bounding box
    cv.rectangle(
        output_image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Calculate label text dimensions
    (text_w, text_h), baseline = cv.getTextSize(
        label,
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    # Position label above the box where possible
    label_y = max(y, text_h + baseline)

    # Draw a filled background behind the label
    cv.rectangle(
        output_image,
        (x, label_y - text_h - baseline),
        (x + text_w, label_y),
        (0, 255, 0),
        cv.FILLED
    )

    # Draw the label text
    cv.putText(
        output_image,
        label,
        (x, label_y - baseline),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv.LINE_AA
    )

print("Objects drawn:", len(final_boxes))



# Step 8: Display and save the final result

# Display the detection result
cv.imshow("YOLOv8 Object Detection", output_image)

# Save the detection result
# output_path = "Photos/yolo_detection_result.jpg"
# saved = cv.imwrite(output_path, output_image)

# if saved:
#     print(f"Detection result saved successfully: {output_path}")
# else:
#     print("Error: Could not save the detection result.")

# Wait until a key is pressed
cv.waitKey(0)

# Close all OpenCV windows
cv.destroyAllWindows()
