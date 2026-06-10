from flask import Flask, render_template, Response, jsonify
import cv2
import supervision as sv
from ultralytics import YOLO
from ultralytics.nn.modules.block import AAttn
import time
import pyresearch 

# Compatibility patch for YOLOv12 attention layouts from older checkpoint state dicts.
def _aattn_forward(self, x):
    B, _, H, W = x.shape
    N = H * W

    if hasattr(self, 'qkv') and hasattr(self, 'qkv', '__call__'):
        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
    else:
        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, self.all_head_dim * 2)
            v = v.reshape(B * self.area, N // self.area, self.all_head_dim)
            B, N, _ = qk.shape
        q, k = (
            qk.view(B, N, self.num_heads, self.head_dim * 2)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim], dim=2)
        )
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

    attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
    attn = attn.softmax(dim=-1)
    x = v @ attn.transpose(-2, -1)
    x = x.permute(0, 3, 1, 2)
    v = v.permute(0, 3, 1, 2)

    if self.area > 1:
        x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
        v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
        B, N, _ = x.shape

    x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
    v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()

    x = x + self.pe(v)
    return self.proj(x)

AAttn.forward = _aattn_forward

# Flask App Initialization
app = Flask(__name__)

# PyResearch Configuration Constants
PR_MODEL_PATH = "best.pt"
PR_DISPLAY_CONFIG = {
    'window_title': "PyResearch - Pothole Computer Vision Project",
    'window_size': (1280, 720),
    'color_scheme': "PR_DARK_BLUE",
    'fps_display': True
}

# Global variable to store detection count
detection_count = 0

class PyResearchVisualizer:
    """PyResearch Standard Visualization Engine"""
    
    def __init__(self):
        self.model = YOLO(PR_MODEL_PATH)
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex("#0055FF")
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.7,
            text_thickness=1,
            text_color=sv.Color.WHITE,
            text_padding=10
        )
        
    def process_frame(self, frame):
        """PyResearch Standard Processing Pipeline"""
        global detection_count
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Update detection count
        detection_count = len(detections)  # Count the number of detections in the current frame
        
        # Apply PyResearch Visualization Standards
        annotated_frame = self.box_annotator.annotate(
            scene=frame,
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        return annotated_frame

def generate_frames():
    visualizer = PyResearchVisualizer()
    cap = cv2.VideoCapture("demo.mp4")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        try:
            output_frame = visualizer.process_frame(frame)
        except Exception as e:
            print(f'Error processing frame: {e}', flush=True)
            break
        
        _, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_count')
def get_detection_count():
    return jsonify({'detections': detection_count})

@app.route('/led_state')
def get_led_state():
    return jsonify({'led_on': detection_count > 0})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')