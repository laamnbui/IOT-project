# Edge AI Pothole Tracker

## Part 1: About The Project

This is an IoT project that uses a Raspberry Pi to find and map potholes on the road in real-time. We mount a Raspberry Pi, a camera, and a GPS module on a vehicle. As the vehicle moves, the system watches the road, detects potholes, and saves their locations on a web map.

### How it works:

Camera: The Pi records video using rpicam-vid for smooth performance.

AI Detection: It takes frames from the video and sends them to a Roboflow API (YOLO model). The AI checks if there is a pothole in the picture.

GPS Tracking: A NEO-6M GPS module runs in the background to get the exact latitude and longitude.

Data Saving: If a pothole is found, the system saves the image to the local SD card and uploads the GPS data and image link to MongoDB.

Web Map: Users can open a web browser to see the live video stream and a map with red markers showing all the bad road spots.

### Tech Stack

Hardware: Raspberry Pi 4, Pi Camera, Ublox NEO-6M GPS.

Backend: Python, Flask.

AI & Vision: Roboflow API, OpenCV, Supervision.

Database: MongoDB Atlas.

Frontend: HTML, Leaflet.js (for OpenStreetMap).

## Part 2: How to Run on Terminal (Raspberry Pi)

Before you start, make sure your hardware is connected correctly and you have your MONGO_URI and ROBOFLOW_API_KEY inside the app_pi.py file.

### Step 1: Connect to your Pi

Open your terminal (or PowerShell on Windows) and SSH into your Raspberry Pi:

ssh admin@<your-pi-ip-address>


### Step 2: Go to the project folder

cd ~/pothole_project

### Step 3: Activate the Virtual Environment

You must turn on the Python virtual environment before running the code so it can load OpenCV and Flask.

source venv/bin/activate

### Step 4: Run the System

Start the main Python file:

python3 app_pi.py

## Note
If it runs successfully, you will see text saying the Database and AI are connected. Now, open your web browser and go to http://<your-pi-ip-address>:5001 to see the live feed and map.

### Troubleshooting (Camera Stuck Error)
Sometimes, if the code crashed before, the camera hardware gets stuck. If you see a StopIteration or HTTP 500 error, kill the hidden camera processes by running this command, then try Step 4 again:

sudo killall rpicam-vid rpicam-hello

