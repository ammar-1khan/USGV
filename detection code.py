import cv2
import numpy as np
import datetime
import time
import sys
import torch
import serial

# Add YOLOv5 path to system path
yolov5_path = 'yolov5'
sys.path.insert(0, yolov5_path)

# Import necessary modules from YOLOv5
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# Serial port configuration (adjust according to your setup)
ser = serial.Serial('/dev/ttyUSB0', 9600)
# ser = serial.Serial('COM3', 9600)

print("Serial Port connected")

# Define grid parameters
grid_width = 3
grid_height = 3
grid_labels = [
    "Up_Left", "Up", "Up_Right",
    "Left", "Center", "Right",
    "Down_Left", "Down", "Down_Right"
]

# Function to preprocess image for YOLOv5
def preprocess_image(img, img_size, stride, auto=True):
    img = letterbox(img, img_size, stride=stride, auto=auto)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return img

# Function to run inference with YOLOv5 model
def run_inference_yolov5(model, img, device, imgsz, stride):
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    return pred

# Function to display results
def display_results(img, pred, names, detected_objects):
    weapon_detected = False
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                class_name = names[int(cls)].lower()  # Get the class name using the detected class ID
                if class_name in detected_objects and conf >= 0.70:  # Check if it's a desired object with high confidence
                    label = f'{class_name} {conf:.2f}'
                    print(f"Weapon detected: {label}")
                    weapon_detected = True
                    multiplier = 4
                    cv2.rectangle(img, (int(xyxy[0]*multiplier), int(xyxy[1]*multiplier)), (int(xyxy[2]*multiplier), int(xyxy[3]*multiplier)), (0, 0, 255), 2)
                    cv2.putText(img, label, (int(xyxy[0]*multiplier), int((xyxy[1]-10))*multiplier), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                elif class_name in detected_objects:
                    # Detected a weapon but with lower confidence, optional handling here
                    pass
    return img, weapon_detected

# Function to load YOLO model
def load_yolo_model(weights_path, device='cpu'):
    model = DetectMultiBackend(weights_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (160, 160)
    return model, stride, names, imgsz

# Main function
def main():
    device = select_device()  # Automatically select available CUDA device or fallback to CPU
    if "cuda" not in device.type:
        print("Warning: CUDA is not available, running on CPU...")
    weights_path_yolo = 'latest.pt'  # Adjust the path to your YOLO weights file

    # Load YOLO model
    model_yolo, stride_yolo, names_yolo, imgsz_yolo = load_yolo_model(weights_path_yolo, device=device)
    print("Loaded Model")

    camera = cv2.VideoCapture(0)
    detected_objects = ['knife', 'pistol', 'rifle']

    prev_x, prev_y = None, None
    tracking = False  # Initialize tracking flag
    weapon_counter = 0

    while True:
        ret, frame = camera.read()
        if frame is None:
            break

        # Resize frame to 128x128 (if needed)
        # frame = cv2.resize(frame, (128, 128))

        # YOLOv5 Weapon Detection
        if tracking == 0:
            textcolor = (0,0,255)
            cv2.putText(frame, "Weapon Detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textcolor, 1)

            img = preprocess_image(frame, imgsz_yolo, stride_yolo)
            pred = run_inference_yolov5(model_yolo, img, device, imgsz_yolo, stride_yolo)
            frame, weapon_detected = display_results(frame, pred, names_yolo, detected_objects)
            if weapon_detected == 1:
                weapon_counter += 1

            if weapon_counter >= 10:
                tracking = True

        # Face Tracking
        if tracking:
            textcolor = (0,0,255)
            cv2.putText(frame, "Face Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textcolor, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Draw green square around face
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

            if len(faces) > 0:
                (fx, fy, fw, fh) = faces[0]
                face_center = (fx + fw // 2, fy + fh // 2)
                # direction = detect_movement(prev_x, prev_y, face_center[0], face_center[1])
                prev_x, prev_y = face_center[0], face_center[1]

                # Print direction if significant movement detected
                # if direction:
                #     print("Direction:", direction)

                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

                # Calculate region of the face
                region_x = face_center[0] // (frame.shape[1] // grid_width)
                region_y = face_center[1] // (frame.shape[0] // grid_height)

                # Calculate the region number
                region_number = region_y * grid_width + region_x

                # Print the region name
                # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Face detected in region:", grid_labels[region_number])

                # If face is not in the center region, send command to Arduino
                if region_number != 4:
                    ser.write(str(region_number).encode())  # Send region number to Arduino

        # Draw grid lines and labels
        grid_color = (255, 255, 255)  # White color for grid lines
        line_thickness = 1
        line_type = cv2.LINE_AA

        # Draw vertical grid lines
        for i in range(1, grid_width):
            x = i * (frame.shape[1] // grid_width)
            cv2.line(frame, (x, 0), (x, frame.shape[0]), grid_color, line_thickness, line_type)

        # Draw horizontal grid lines
        for i in range(1, grid_height):
            y = i * (frame.shape[0] // grid_height)
            cv2.line(frame, (0, y), (frame.shape[1], y), grid_color, line_thickness, line_type)

        # Draw smaller grid lines for the "Center" region
        calib = 0
        center_region_width = frame.shape[1] // grid_width
        center_region_height = frame.shape[0] // grid_height
        center_x_start = ((frame.shape[1] - center_region_width) + calib) // 2
        center_y_start = ((frame.shape[0] - center_region_height) + calib) // 2
        center_x_end = (center_x_start + center_region_width) - calib
        center_y_end = (center_y_start + center_region_height) - calib
        cv2.rectangle(frame, (center_x_start, center_y_start), (center_x_end, center_y_end), grid_color, line_thickness, line_type)

        # Draw grid labels
        for i in range(len(grid_labels)):
            x = (i % grid_width) * (frame.shape[1] // grid_width) + 10
            y = (i // grid_height) * (frame.shape[0] // grid_height) + 20
            cv2.putText(frame, grid_labels[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)

        # Display frame
        cv2.imshow("Security Feed", frame)

        # Check for key press to exit
        key = cv2.waitKey(1) & 0xFF

        if key == ord('x'):
            tracking = 0
            weapon_counter = 0

        if key == ord('q'):
            break

    # Release resources
    ser.close()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
