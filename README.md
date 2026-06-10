# Pothole-Computer-Vision-Project

A simple Flask-based pothole detection demo using a YOLO model and OpenCV video streaming.

## Prerequisites

- Python 3.8 or newer
- `pip` package manager
- A working camera or video file named `demo.mp4` in the project root

## Repository Contents

- `app.py` - Flask application and video processing pipeline
- `best.pt` - YOLO model checkpoint used for pothole detection
- `demo.mp4` - example video input for the demo
- `templates/index.html` - web UI served by Flask

## Installation

1. Open a terminal in the project folder:

```bash
cd "c:\Users\buing\Downloads\Pothole-Computer-Vision-Project-main\Pothole-Computer-Vision-Project-main"
```

2. Install required Python packages:

```bash
pip install flask opencv-python ultralytics supervision pyresearch
```

> If you already have a virtual environment, activate it first before installing dependencies.

## Running the Application

1. Ensure `demo.mp4` is present in the project folder.
2. Start the Flask app:

```bash
python app.py
```

3. Open your browser and go to:

```text
http://127.0.0.1:5000/
```

The app will stream the processed video and display live detection results.

## Notes

- The app reads from `demo.mp4` by default. To use a different file or camera input, edit the `cv2.VideoCapture("demo.mp4")` line in `app.py`.
- The model file `best.pt` must remain in the project root for the YOLO model to load successfully.
- The detection count is exposed at `/detection_count` and LED-style status at `/led_state`.

## Troubleshooting

- If Flask fails to start, confirm the required packages are installed and that Python is on your PATH.
- If the video does not load, verify that `demo.mp4` exists and is a valid video file.
- If the model fails to load, confirm `best.pt` is not corrupted and matches the expected YOLO/Ultralytics version.
