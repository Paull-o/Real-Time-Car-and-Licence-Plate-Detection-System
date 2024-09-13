from ultralytics import YOLO
import cv2
import os
import glob
import re
import csv
from sort.sort import *
from functions import get_car, read_license_plate, write_csv
from tqdm import tqdm

def get_next_csv_filename():
    # Get a list of all CSV files in the directory
    csv_files = glob.glob('./*.csv')

    # Extract numbers from the existing CSV filenames
    numbers = [int(re.search(r'\d+', filename).group()) for filename in csv_files if re.search(r'\d+', filename)]

    # Increment the maximum number found or start with 1 if no CSV files exist
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1

    return f'./test{next_number}.csv'

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./data/videos/sample.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Progress bar for console
progress_bar_console = tqdm(total=total_frames, desc="Processing frames", unit="frame", position=0)

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
        progress_bar_console.update(1)

# write results
csv_filename = get_next_csv_filename()
write_csv(results, csv_filename)

# Close the progress bar for the console
progress_bar_console.close()
