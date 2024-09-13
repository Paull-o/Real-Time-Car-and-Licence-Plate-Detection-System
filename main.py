import tkinter as tk
from PIL import Image, ImageTk
import shutil
from ultralytics import YOLO
import cv2
from tkinter import filedialog
import os
from tqdm import tqdm


# Set the working directory to the directory where main.py is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Function to run detector_write.py
def run_detector_write():
    # Open file dialog to select the video file
    filepath = filedialog.askopenfilename()
    if filepath:
        os.system(f"start cmd /k venv\\Scripts\\activate && python detector_write.py {filepath}")


# Function to run implement_data.py with the most recent CSV file
def run_data_interpolation():
    # Find the most recent CSV file created by the license plate reader
    csv_files = [f for f in os.listdir('.') if f.startswith('test') and f.endswith('.csv')]
    if csv_files:
        latest_csv_file = max(csv_files, key=os.path.getctime)
        latest_csv_file = max(csv_files, key=os.path.getctime)
        # Run implement_data.py with the latest CSV file
        os.system(f"start cmd /k venv\\Scripts\\activate && python implement_data.py {latest_csv_file}")
    else:
        print("No CSV files found.")


# Function to run visual.py with the specified CSV file and video file
def run_visual_with_progress(csv_file, video_file):
    if video_file and csv_file:
        # Quote the paths to handle spaces in file names
        csv_file_quoted = f'"{csv_file}"'
        video_file_quoted = f'"{video_file}"'

        # Get total number of frames in the video file
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Run visual.py with the quoted file paths
        os.system(f"start cmd /k venv\\Scripts\\activate && python visual.py {csv_file_quoted} {video_file_quoted}")

        # Create progress bar
        progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

        # Wait until the tracking video is generated
        while not os.path.exists('finished_vid.mp4'):
            pass

        # Display the tracking video
        cap = cv2.VideoCapture('finished_vid.mp4')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Tracking Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            progress_bar.update(1)  # Update progress bar
        cap.release()
        cv2.destroyAllWindows()

        # Close the progress bar
        progress_bar.close()

        # Generate the interpolated CSV file path
        interpolated_csv_file = csv_file.replace('.csv', '_interpolated.csv')

        # Check if the interpolated CSV file exists
        if os.path.exists(interpolated_csv_file):
            # Generate the destination file path for the interpolated CSV file
            destination_file = os.path.join('tests', os.path.basename(interpolated_csv_file))

            # Move the generated interpolated CSV file to the 'tests' folder
            try:
                shutil.move(interpolated_csv_file, destination_file)
            except FileNotFoundError:
                print(f"Interpolated CSV file '{interpolated_csv_file}' not found.")
        else:
            print(f"Interpolated CSV file '{interpolated_csv_file}' not generated.")

        # Open the tracking video for the user to watch
        os.system(f"start finished_vid.mp4")
    else:
        print("Video file path or CSV file path is missing.")


# Function to run visual.py with the most recent CSV file
def run_visual_with_latest():
    # Find the most recent interpolated CSV file
    interpolated_csv_files = [f for f in os.listdir('.') if f.endswith('_interpolated.csv')]
    if interpolated_csv_files:
        latest_interpolated_csv_file = max(interpolated_csv_files, key=os.path.getctime)
        interpolated_csv_file_path = os.path.abspath(latest_interpolated_csv_file)
        video_file = filedialog.askopenfilename()
        if video_file:
            run_visual_with_progress(interpolated_csv_file_path, video_file)
    else:
        print("Interpolated CSV file not found.")


# Function to open file dialog and get file path
def browse_file():
    filename = filedialog.askopenfilename()
    return filename


# Create main window
root = tk.Tk()
root.title("Enhancing Visual Understanding")


# Load and resize logo image
logo_image = Image.open("logo.png")
logo_image = logo_image.resize((500, 300))  # Resize logo image
logo_photo = ImageTk.PhotoImage(logo_image)


# Function to draw bounding boxes around detected objects
def draw_boxes(image, boxes):
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box[:4])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image


# Function to identify cars in an image or video
def identify_cars():
    # Specify the path to the checkpoint file
    checkpoint_path = os.path.join('.', 'runs', 'detect', 'train8', 'weights', 'last.pt')

    # Browse file and get file path
    filepath = browse_file()

    # Check if filepath is not empty
    if filepath:
        # Check if file is an image or video
        if filepath.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Load YOLO model with the checkpoint file
            model = YOLO(checkpoint_path)

            # Process image
            image = cv2.imread(filepath)
            results = model(image)
            for result in results:
                if hasattr(result, 'boxes') and result.boxes.xyxy.shape[0] > 0:
                    image_with_boxes = draw_boxes(image.copy(), result.boxes.xyxy.tolist())
                    cv2.imshow('Image with Boxes', image_with_boxes)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("No cars detected in the image.")
        elif filepath.endswith(('.mp4', '.avi', '.mov')):
            # Load YOLO model with the checkpoint file
            model = YOLO(checkpoint_path)

            # Process video
            cap = cv2.VideoCapture(filepath)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes.xyxy.shape[0] > 0:
                        image_with_boxes = draw_boxes(frame.copy(), result.boxes.xyxy.tolist())
                        cv2.imshow('Video', image_with_boxes)
                    else:
                        print("No cars detected in the frame.")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


# Create subtitles
car_detection_subtitle = tk.Label(root, text="Car Detection", font=("Helvetica", 16))
license_plate_subtitle = tk.Label(root, text="License Plate Detection", font=("Helvetica", 16))

# Position subtitles
car_detection_subtitle.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
license_plate_subtitle.grid(row=0, column=2, padx=20, pady=(20, 10), sticky="w")

# Position logo on the window
logo_label = tk.Label(root, image=logo_photo)
logo_label.grid(row=5, column=0, padx=50, pady=(0, 10), sticky="nsew", columnspan=3)

# Position buttons for car detection
image_button_desc = "Upload Image\n(Click to detect cars in an image)"
video_button_desc = "Upload Video\n(Click to detect cars in a video)"
image_button = tk.Button(root, text=image_button_desc, command=identify_cars, font=("Helvetica", 12), padx=20,
                         pady=10, bg="gold")
video_button = tk.Button(root, text=video_button_desc, command=identify_cars, font=("Helvetica", 12), padx=20,
                         pady=10, bg="gold")
image_button.grid(row=2, column=0, padx=20, pady=10, sticky="w")
video_button.grid(row=3, column=0, padx=20, pady=10, sticky="w")

# Position buttons for license plate detection
detector_write_button_desc = "Step 1 - License Plate Reader\n(Gather data from video onto CSV File)"
data_interpolation_button_desc = "Step 2 - Data Cleanup\n(Interpolate missing frames in file.)"
visualization_button_desc = "Step 3 - Generate Tracking Video\n(Create video with tracked License Plates)"
detector_write_button = tk.Button(root, text=detector_write_button_desc, command=run_detector_write,
                                  font=("Helvetica", 12), padx=20, pady=10, bg="gold")
data_interpolation_button = tk.Button(root, text=data_interpolation_button_desc, command=run_data_interpolation,
                                      font=("Helvetica", 12), padx=20, pady=10, bg="gold")
visualization_button = tk.Button(root, text=visualization_button_desc, command=run_visual_with_latest,
                                 font=("Helvetica", 12), padx=20, pady=10, bg="gold")

detector_write_button.grid(row=2, column=2, padx=20, pady=10, sticky="w")
data_interpolation_button.grid(row=3, column=2, padx=20, pady=10, sticky="w")
visualization_button.grid(row=4, column=2, padx=20, pady=10, sticky="w")

# Run the application
root.mainloop()
