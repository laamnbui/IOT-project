from flask import Flask, Response, jsonify
import subprocess
import itertools
import time
import cv2
import numpy as np
import supervision as sv
from roboflow import Roboflow
from datetime import datetime
import threading
import serial
import pynmea2
import os

try:
    from pymongo import MongoClient
    _pymongo_available = True
except Exception:
    MongoClient = None
    _pymongo_available = False

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MONGO_URI = "SERVER_URI"
ROBOFLOW_API_KEY = "ROBOFLOW_API_KEY"

if _pymongo_available:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info() 
        db = client["pothole_mapping"]
        collection = db["pothole_locations"]
        print("[DATABASE] Connected to MongoDB successfully!")
    except Exception as e:
        print('[DB ERROR] Failed to connect to MongoDB:', e)
        collection = None
else:
    collection = None

current_gps = {"latitude": 0.0, "longitude": 0.0, "status": "V", "timestamp": None}

def gps_worker():
    global current_gps
    while True:
        try:
            ser = serial.Serial('/dev/serial0', 9600, timeout=1)
            print("[GPS SYSTEM] Started listening on port /dev/serial0...")
            while True:
                line = ser.readline().decode('ascii', errors='replace')
                if line.startswith('$GNRMC') or line.startswith('$GPRMC'):
                    msg = pynmea2.parse(line)
                    current_gps['status'] = msg.status
                    if msg.status == 'A':
                        current_gps['latitude'] = msg.latitude
                        current_gps['longitude'] = msg.longitude
                        current_gps['timestamp'] = datetime.now().isoformat()
        except Exception:
            time.sleep(2)

gps_thread = threading.Thread(target=gps_worker, daemon=True)
gps_thread.start()

try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("road-defect-detection").project("road_defect_detection_uni")
    model = project.version(11).model
    print("[AI SYSTEM] Roboflow model initialized successfully!")
except Exception as e:
    print("[AI ERROR] Failed to connect to Roboflow API:", e)
    model = None

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def generate_frames():
    global model, current_gps, collection
    
    cmd = [
        "rpicam-vid", "-t", "0", "--width", "640", "--height", "480", 
        "--framerate", "30", "--codec", "mjpeg", "-o", "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1024*64)

    boundary = b'\xff\xd8'
    buffer = b''

    last_inference_time = 0.0
    inference_interval = 1.0 
    
    last_save_time = 0.0
    save_interval = 3.0 
    detections = sv.Detections.empty()

    try:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk: break
            buffer += chunk

            while boundary in buffer:
                start_idx = buffer.find(boundary)
                end_idx = buffer.find(boundary, start_idx + 2)

                if end_idx != -1:
                    jpg_frame = buffer[start_idx:end_idx]
                    buffer = buffer[end_idx:]

                    if model is not None:
                        current_time = time.time()
                        if current_time - last_inference_time >= inference_interval:
                            nparr = np.frombuffer(jpg_frame, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            if frame is not None and frame.size > 0:
                                try:
                                    prediction = model.predict(frame, confidence=40, overlap=30).json()
                                    detections = sv.Detections.from_inference(prediction)
                                    last_inference_time = current_time
                                    
                                    if len(detections) > 0 and current_gps['status'] == 'A' and current_gps['latitude'] != 0.0:
                                        if current_time - last_save_time >= save_interval:
                                            save_frame = label_annotator.annotate(
                                                scene=box_annotator.annotate(scene=frame.copy(), detections=detections),
                                                detections=detections
                                            )
                                            
                                            filename = f"pothole_{int(current_time)}.jpg"
                                            filepath = os.path.join(UPLOAD_FOLDER, filename)
                                            cv2.imwrite(filepath, save_frame)
                                            image_url = f"/static/uploads/{filename}"

                                            if collection is not None:
                                                doc = {
                                                    "location": {"type": "Point", "coordinates": [current_gps['longitude'], current_gps['latitude']]},
                                                    "timestamp": current_gps['timestamp'],
                                                    "objects_count": len(detections),
                                                    "image_url": image_url
                                                }
                                                collection.insert_one(doc)
                                                print(f"[DATABASE] Saved pothole! Image: {filename}")
                                            
                                            last_save_time = current_time
                                except Exception as ai_err:
                                    print('[AI ERROR]', ai_err)

                    if len(detections) > 0:
                        try:
                            nparr = np.frombuffer(jpg_frame, np.uint8)
                            frame_to_draw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame_to_draw is not None:
                                annotated_frame = label_annotator.annotate(
                                    scene=box_annotator.annotate(scene=frame_to_draw, detections=detections),
                                    detections=detections
                                )
                                _, encoded_buffer = cv2.imencode('.jpg', annotated_frame)
                                jpg_frame = encoded_buffer.tobytes()
                        except: pass

                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg_frame + b'\r\n')
                else:
                    break
    except Exception as e:
        print('[ERROR]', e)
    finally:
        process.terminate()

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Pothole Pi Cam Live</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" />
        </head>
        <body style="background:#121212; color:white; text-align:center; font-family:sans-serif; padding-top:20px;">
            <h1><i class="fas fa-video"></i> Pothole Edge AI (Raspberry Pi)</h1>
            <div style="margin-bottom: 20px;">
                <a href="/api/status" style="color:#00ffcc; margin-right: 15px;">Check GPS Status</a>
                <a href="/map" style="background: #00ffcc; color: black; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">
                    <i class="fas fa-map-marked-alt"></i> Open Pothole Map
                </a>
            </div>
            <img src="/video_feed" width="640" height="480" style="border:5px solid #333; border-radius:8px;" />
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    gen = generate_frames()
    first_chunk = next(gen)
    chained = itertools.chain([first_chunk], gen)
    return Response(chained, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def system_status():
    global current_gps
    return jsonify({
        "gps_status": "Active" if current_gps['status'] == 'A' else "Void",
        "latitude": current_gps['latitude'],
        "longitude": current_gps['longitude']
    })

@app.route('/api/potholes')
def get_potholes():
    if collection is None: return jsonify([])
    docs = list(collection.find().sort("_id", -1).limit(100))
    for doc in docs: doc['_id'] = str(doc['_id'])
    return jsonify(docs)

@app.route('/map')
def map_dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pothole Map (Pi Server)</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" />
        <style>
            body { margin: 0; padding: 0; font-family: sans-serif; }
            #map { height: 100vh; width: 100vw; }
            .pothole-popup img { width: 100%; border-radius: 8px; margin-top: 10px; border: 2px solid #ff4444; }
            .pothole-popup h3 { margin: 0 0 5px 0; color: #d32f2f; font-size: 16px; font-weight: 800; }
            .pothole-popup p { margin: 0; font-size: 13px; color: #555; }
            .back-btn {
                position: absolute; top: 15px; left: 60px; z-index: 1000;
                background: white; padding: 10px 15px; border-radius: 5px;
                text-decoration: none; color: black; font-weight: bold;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3); border: 1px solid #ccc;
            }
            .back-btn:hover { background: #eee; }
        </style>
    </head>
    <body>
        <a href="/" class="back-btn"><i class="fas fa-arrow-left"></i> Back to Camera</a>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
        <script>
            var map = L.map('map').setView([10.762622, 106.660172], 13);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(map);
            var alertIcon = L.divIcon({
                html: '<i class="fas fa-exclamation-triangle" style="color: #d32f2f; font-size: 28px; filter: drop-shadow(2px 4px 6px rgba(0,0,0,0.5));"></i>',
                className: 'custom-div-icon', iconSize: [28, 28], iconAnchor: [14, 28], popupAnchor: [0, -28]
            });
            fetch('/api/potholes').then(res => res.json()).then(data => {
                if (data.length > 0) map.setView([data[0].location.coordinates[1], data[0].location.coordinates[0]], 15);
                data.forEach(pothole => {
                    var lng = pothole.location.coordinates[0];
                    var lat = pothole.location.coordinates[1];
                    var marker = L.marker([lat, lng], {icon: alertIcon}).addTo(map);
                    var date = new Date(pothole.timestamp).toLocaleString('en-US');
                    var popupContent = `<div class="pothole-popup"><h3>Detected ${pothole.objects_count || pothole.objects} pothole(s)!</h3><p>${date}</p><p>${lat.toFixed(5)}, ${lng.toFixed(5)}</p>`;
                    if (pothole.image_url) popupContent += `<img src="${pothole.image_url}" alt="Pothole Image" />`;
                    popupContent += `</div>`;
                    marker.bindPopup(popupContent, { minWidth: 250 });
                });
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)