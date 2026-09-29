

**Setup:** `pip install opencv-python numpy` · **Verify:** `import cv2; print(cv2.__version__)`

---

## 1. What OpenCV Is and Why It Matters

OpenCV is a fast, open-source library for image and video processing, used in object tracking, motion detection, document scanning and inspection. Its C++ core is called from Python through `cv2`.

**Key idea:** an image is a NumPy array of shape (height, width, channels), with values from 0 to 255. Colors are stored as **BGR**, not RGB.

**Task:** Load an image and print its shape, dtype, and the pixel value at (50, 50).

## 2. Basic Image Operations

- Read, show, save: `imread`, `imshow`, `imwrite`, `waitKey`
- Color and size: `cvtColor`, `resize`, crop with `img[y1:y2, x1:x2]`
- Drawing: `rectangle`, `circle`, `putText`
- Filtering: `GaussianBlur`, `threshold`, `adaptiveThreshold`
- Edges and shapes: `Canny`, `findContours`, `drawContours`

**Task:** Photograph objects on a plain background. Apply grayscale → blur → edges → contours, then draw a box around each object and show the count.

## 3. Basic Video Operations

Video is a sequence of frames, so every image operation works per frame.

```python
cap = cv2.VideoCapture(0)          # 0 = webcam, or a file path
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
```

- Save video with `VideoWriter`
- Motion detection with frame differencing or `createBackgroundSubtractorMOG2`

**Task:** Show the live webcam feed with a timestamp. Press **e** to toggle an edge view, and save the recording to `output.avi`.

## 4. Good Practices

- Always check inputs: `imread` returns `None` on a bad path, and `cap.read()` can fail.
- Remember BGR. Convert with `cvtColor` before using Matplotlib or other libraries.
- Release resources: call `cap.release()` and `destroyAllWindows()` when done.
- Work on copies (`img.copy()`) so drawing doesn't alter the original.
- Use NumPy operations instead of Python loops over pixels.
- Keep small, clearly named functions and test on a few sample images first.

## 5. Intro to Detection with OpenCV (ML)

Detection means finding **what** is in a frame and **where**, returned as bounding boxes with confidence scores.

- **People detection:** OpenCV has a built-in pretrained HOG + SVM people detector (`HOGDescriptor_getDefaultPeopleDetector`). It is simple and needs no downloads, but it is slower and less accurate than deep learning models.
- **Object detection:** the `cv2.dnn` module runs pretrained deep learning models such as YOLO (ONNX) or MobileNet-SSD. The flow is `readNet` → `blobFromImage` → `forward` → filter by confidence → remove overlapping boxes with `NMSBoxes`.

Detection is heavy, so video pipelines usually trade some accuracy for speed:

- **Frame skipping:** run detection on every Nth frame and reuse the last boxes in between.
- **Resizing:** shrink frames before detection (for example to 640 px wide), then scale the boxes back.
- **Region of interest:** crop to the area that matters (a door, a desk) instead of the full frame.

python

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap, n, boxes = cv2.VideoCapture(0), 0, []
while True:
    ok, frame = cap.read()
    if not ok: break
    small = cv2.resize(frame, (640, 360))            # resize for speed
    if n % 5 == 0:                                   # detect every 5th frame
        boxes, _ = hog.detectMultiScale(small, winStride=(8, 8))
    for (x, y, w, h) in boxes:
        cv2.rectangle(small, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("People", small); n += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
```
**Task:** Run the snippet above, then compare the FPS with skipping set to 1, 5 and 10, and with and without resizing. Note how speed and accuracy change.

---

**Reference:** [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)