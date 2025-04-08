import cv2
import numpy as np
import os

VIDEO_PATH = "video.mp4"
OUTPUT_FOLDER = "output"
FRAME_INTERVAL = 3
TARGET_SIZE = (600, 600)
PADDING = 60  # pour éviter que la voiture soit coupée

def center_object_with_padding(img, output_size, padding=60):
    # Prétraitement doux
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 100)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(img, output_size)

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Agrandir la box pour être sûr de ne pas couper la voiture
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, img.shape[1] - x)
    h = min(h + 2 * padding, img.shape[0] - y)

    center_x = x + w // 2
    center_y = y + h // 2
    half_w, half_h = output_size[0] // 2, output_size[1] // 2

    start_x = max(center_x - half_w, 0)
    start_y = max(center_y - half_h, 0)
    end_x = min(start_x + output_size[0], img.shape[1])
    end_y = min(start_y + output_size[1], img.shape[0])

    cropped = img[start_y:end_y, start_x:end_x]
    final = cv2.resize(cropped, output_size)
    return final

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_INTERVAL == 0:
        centered = center_object_with_padding(frame, TARGET_SIZE, padding=PADDING)
        filename = f"frame_{saved_count:03d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), centered)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ {saved_count} images centrées enregistrées dans : {OUTPUT_FOLDER}")
