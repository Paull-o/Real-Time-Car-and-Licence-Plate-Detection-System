import cv2
import numpy as np
import pandas as pd
import ast
import argparse
import os

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def run_visual(csv_file, video_file):
    results = pd.read_csv(csv_file)

    # Load video
    cap = cv2.VideoCapture(video_file)

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('Tracked_video.mp4', fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])

        license_plate[car_id] = {'license_crop': None,
                                 'license_plate_number': results[(results['car_id'] == car_id) &
                                                                 (results['license_number_score'] == max_score)]['license_number'].iloc[0]}

        cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                 (results['license_number_score'] == max_score)]['frame_nmr'].iloc[0])
        ret, frame = cap.read()

        x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                                  (results['license_number_score'] == max_score)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        # Resize license plate image to a smaller size
        license_crop = cv2.resize(license_crop, (200, 100))

        license_plate[car_id]['license_crop'] = license_crop

    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                # Draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                # Draw license plate
                x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                # Crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                H, W, _ = license_crop.shape

                try:
                    frame[int(car_y1) - H - 100:int(car_y1) - 100,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                    # Add license plate number text
                    plate_text = license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
                    text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
                    text_x = int((car_x1 + car_x2 - text_size[0]) / 2)
                    text_y = int(car_y1 - H - 150)
                    cv2.putText(frame, plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

                except:
                    pass

            out.write(frame)
            frame = cv2.resize(frame, (1280, 720))

    out.release()
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="CSV file containing car and license plate data")
    parser.add_argument("video_file", help="Video file to be processed")
    args = parser.parse_args()

    run_visual(args.csv_file, args.video_file)
