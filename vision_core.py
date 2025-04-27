"""
CareBotix Vision + Sensor-Fusion Service  v3.1
==============================================
✓ YOLOv8-Pose maths  ✓ Velocity  ✓ Bed-zone IoU
✓ WebSocket push     ✓ Static dashboard served from /dashboard
"""

# ─── Imports & monkey-patch (order matters) ──────────────────────
from __future__ import annotations          # must be first

import eventlet
eventlet.monkey_patch()                     # before ANY std/3rd-party IO

import os, cv2, time, threading, warnings, math, collections
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Deque, Optional

import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from pymongo import MongoClient, errors as mongo_errors

# ─── Config (env-overridable) ─────────────────────────────────────
YOLO_MODEL        = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_POSE_MODEL   = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")
MONGO_URI         = os.getenv("MONGO_URI") or "mongodb://localhost:27017"
MONGO_DB          = os.getenv("MONGO_DB", "morganhack")
MONGO_COL         = os.getenv("MONGO_COL", "readings")
PATIENT_ID        = os.getenv("PATIENT_ID", "00000001")
CAM_INDEX         = int(os.getenv("CAM_INDEX", 0))

HIGH_TEMP_F       = float(os.getenv("HIGH_TEMP_F", 100.4))
LOW_TEMP_F        = float(os.getenv("LOW_TEMP_F", 95.0))
DIST_THRESHOLD_CM = float(os.getenv("DIST_THRESHOLD_CM", 80))
POSE_FALL_ANGLE   = float(os.getenv("POSE_FALL_ANGLE", 40))     # deg
BED_IOU_THRESH    = float(os.getenv("BED_IOU_THRESH", 0.10))

# Calibrated bed polygon (edit these four points!)
BED_ZONE = np.array([[200,140], [1050,140],
                     [1050,520], [200,520]], dtype=np.int32)
BED_POLY = Polygon(BED_ZONE)

# ─── Vision module ────────────────────────────────────────────────
class VisionModule:
    def __init__(self):
        print("[Vision] loading bbox model →", YOLO_MODEL)
        self.detector = YOLO(YOLO_MODEL)
        print("[Vision] loading pose model →", YOLO_POSE_MODEL)
        self.pose = YOLO(YOLO_POSE_MODEL)
        self.track: Dict[int, Deque[Tuple[float,float,float]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=20))

    @staticmethod
    def bbox_aspect(box):                       # h / w
        x1,y1,x2,y2 = box; return (y2-y1)/max(1,(x2-x1))

    @staticmethod
    def iou_poly(bbox:np.ndarray, poly:Polygon)->float:
        x1,y1,x2,y2 = bbox
        bb_poly = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
        inter = bb_poly.intersection(poly).area
        union = bb_poly.area + poly.area - inter
        return inter/union if union else 0.0

    @staticmethod
    def torso_angle(kpts:np.ndarray)->Optional[float]:
        if kpts.shape[0] < 12: return None
        sh_x,sh_y = kpts[5][:2]; hp_x,hp_y = kpts[11][:2]
        dx,dy = hp_x-sh_x, hp_y-sh_y
        if math.hypot(dx,dy) < 1: return None
        return abs(math.degrees(math.atan2(dy,dx)))   # 0=flat, 90=vertical

    def analyze(self, frame:np.ndarray)->Tuple[Dict[str,Any],np.ndarray]:
        out:Dict[str,Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}
        vis = frame.copy()

        # 1) detect person bbox
        det = self.detector(frame, verbose=False)[0]
        box = next((b.xyxy[0].cpu().numpy() for b in det.boxes if int(b.cls)==0), None)
        out["person_present"] = box is not None

        angle = None
        pose_res = self.pose(frame, verbose=False)[0]
        best_k = None
        if box is not None and pose_res.keypoints:
            for kpts in pose_res.keypoints:
                k = kpts.xy.cpu().numpy()
                if k.shape[0] >= 12:
                    cx, cy = k[5][0], k[5][1]
                    x1, y1, x2, y2 = box.astype(int)
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        best_k = k
                        break
            if best_k is None and len(pose_res.keypoints) > 0:
                k = pose_res.keypoints[0].xy.cpu().numpy()
                if k.shape[0] >= 12:
                    best_k = k
        elif pose_res.keypoints and len(pose_res.keypoints) > 0:
            k = pose_res.keypoints[0].xy.cpu().numpy()
            if k.shape[0] >= 12:
                best_k = k

        if best_k is not None and best_k.shape[0] >= 12:
            angle = self.torso_angle(best_k)
            vis = pose_res.plot()
        else:
            print("[DEBUG] No valid keypoints for posture analysis.")
        out["pose_angle_deg"] = angle

        # posture / fall flags
        fallen_aspect = box is not None and self.bbox_aspect(box) < 0.6
        fallen_pose   = angle is not None and angle < POSE_FALL_ANGLE
        lying_flat    = angle is not None and angle < 25
        bad_posture   = angle is not None and angle > 70

        out.update({
            "fallen": fallen_aspect or fallen_pose,
            "lying_flat": lying_flat,
            "bad_posture": bad_posture
        })
        if not out["person_present"]:
            out["posture_label"]="None"
        elif lying_flat:
            out["posture_label"]="Lying"
        elif bad_posture:
            out["posture_label"]="Bent"
        else:
            out["posture_label"]="Normal"

        # --- DEBUG: Print vector math for posture ---
        print(f"[DEBUG] Posture: angle={angle}, label={out['posture_label']}, fallen={out['fallen']}, lying_flat={lying_flat}, bad_posture={bad_posture}")

        # bed IoU
        if box is not None:
            iou = self.iou_poly(box, BED_POLY)
            out["bed_iou"] = round(iou,3)
            out["out_of_bed"] = iou < BED_IOU_THRESH
        else:
            out["bed_iou"] = None; out["out_of_bed"] = False

        # velocity
        if box is not None:
            cx,cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            t=time.time(); tr=self.track[0]; tr.append((t,cx,cy))
            if len(tr)>=2:
                dt=tr[-1][0]-tr[-2][0]; dist=math.hypot(tr[-1][1]-tr[-2][1], tr[-1][2]-tr[-2][2])
                out["velocity_px_s"] = dist/dt if dt else 0.0
                if len(tr)>=5:
                    dists=[math.hypot(tr[i][1]-tr[i-1][1], tr[i][2]-tr[i-1][2]) for i in range(1,len(tr))]
                    dts=[tr[i][0]-tr[i-1][0] for i in range(1,len(tr))]
                    out["velocity_avg_px_s"]=sum(dists)/max(1e-3,sum(dts))
        else:
            out["velocity_px_s"]=out["velocity_avg_px_s"]=None

        # draw overlays
        if box is not None:
            col=(0,0,255) if out["fallen"] else (0,255,0)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
            cv2.putText(vis,f'{angle:.0f}' if angle else 'person',(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
            cv2.putText(vis,f'IoU {out["bed_iou"]:.2f}',(x1,y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
        cv2.polylines(vis,[BED_ZONE],True,(255,255,0),1)
        return out, vis

# ─── Resilient sensor DB wrapper ───────────────────────────────────
class SensorDB:
    def __init__(self, uri, db, col):
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.col = self.client[db][col]
            self.up = True; print(f"[Mongo] connected → {uri}")
        except mongo_errors.ServerSelectionTimeoutError:
            warnings.warn("MongoDB unreachable – vision-only mode"); self.up=False
            self.col=None
    def latest(self,pid):                       # latest doc or {}
        if not self.up: return {}
        try: return self.col.find_one({"patient_id":pid}, sort=[("timestamp",-1)]) or {}
        except Exception as e:
            warnings.warn(f"[Mongo] query failed: {e}"); self.up=False; return {}

# ─── Flask / SocketIO setup ───────────────────────────────────────
app = Flask(__name__, static_folder='dashboard', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

vision   = VisionModule()
_sensors = SensorDB(MONGO_URI, MONGO_DB, MONGO_COL)
LATEST_STATUS:Dict[str,Any]={}
LATEST_FRAME:bytes|None=None

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/video')
def video():
    def gen():
        boundary=b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            if LATEST_FRAME:
                yield boundary+LATEST_FRAME+b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    # Ensure ObjectId is always converted to string before returning
    status_copy = dict(LATEST_STATUS) if LATEST_STATUS else {}
    # Recursively convert ObjectId to string in all nested dicts
    def convert_objid(val):
        if isinstance(val, dict):
            return {k: convert_objid(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [convert_objid(v) for v in val]
        elif hasattr(val, '__class__') and val.__class__.__name__ == 'ObjectId':
            return str(val)
        else:
            return val
    status_copy = convert_objid(status_copy)
    return jsonify(status_copy)

@app.route('/config')
def cfg():
    return jsonify({
        "HIGH_TEMP_F":HIGH_TEMP_F,"LOW_TEMP_F":LOW_TEMP_F,
        "DIST_THRESHOLD_CM":DIST_THRESHOLD_CM,
        "POSE_FALL_ANGLE":POSE_FALL_ANGLE,"BED_IOU_THRESH":BED_IOU_THRESH
    })

# ─── Capture / fusion thread ───────────────────────────────────────
def capture_loop():
    global LATEST_STATUS, LATEST_FRAME
    cap=cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        warnings.warn("Camera not found – demo black frame")
    while True:
        ok,frame = cap.read() if cap.isOpened() else (True, np.zeros((480,640,3),np.uint8))
        if not ok: time.sleep(0.05); continue
        vs,annot = vision.analyze(frame)
        sens = _sensors.latest(PATIENT_ID)
        if sens and '_id' in sens:
            sens['_id'] = str(sens['_id'])
        vs.update(sens)
        # --- DEBUG: Print MongoDB sensor data ---
        print(f"[DEBUG] MongoDB sensor data: {sens}")
        if sens:
            vs["ultrasonic_bed_empty"] = sens.get("distance", 0) > DIST_THRESHOLD_CM
            t = sens.get("temperature")
            if t is not None:
                vs["fever"] = t > HIGH_TEMP_F; vs["hypothermia"] = t < LOW_TEMP_F
        _, jpg = cv2.imencode('.jpg', annot, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        # --- DEBUG: Print merged status dictionary ---
        print(f"[DEBUG] LATEST_STATUS: {vs}")
        LATEST_FRAME = jpg.tobytes(); LATEST_STATUS = vs
        socketio.emit('status', vs)
        time.sleep(5)  # Update every 5 seconds

threading.Thread(target=capture_loop, daemon=True).start()

# ─── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    port=int(os.getenv('PORT',8000))
    print(f"[Server] http://localhost:{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
