# app.py

from flask import Flask, jsonify
from flask_cors import CORS
import threading
import serial
import json
import time

# === Arduino Bluetooth Serial Settings ===
# Example ports:
# Mac: '/dev/cu.HC-05-DevB'  (HC-05 device)
# Linux: '/dev/rfcomm0'
# Windows: 'COM6' or 'COM7'  (after Bluetooth pairing)
BLUETOOTH_PORT = "/dev/cu.HC-06"
BAUD_RATE = 9600

# === Initialize Serial Connection ===
ser = serial.Serial(BLUETOOTH_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
ser.reset_input_buffer()

# === Shared Data ===
latest_sensor_data = {
    "temperature": None,
    "humidity": None,
    "sound": None,
    "distance": None,
    "alerts": []
}

# Thresholds
FEVER_TEMP_THRESHOLD = 38.0
SOUND_THRESHOLD = 48
FALL_DISTANCE_CHANGE = 50

last_distance = None

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

# === Background Serial Reader ===
def read_serial():
    global latest_sensor_data, last_distance
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                continue

            if not line.startswith("{") or not line.endswith("}"):
                print(f"Skipping incomplete line: {line}")
                continue

            print(f"Arduino: {line}")

            data = json.loads(line)

            if not isinstance(data, dict):
                print(f"Skipping non-dict data: {data}")
                continue

            latest_sensor_data["temperature"] = data.get("temperature")
            latest_sensor_data["humidity"] = data.get("humidity")
            latest_sensor_data["sound"] = data.get("sound")
            latest_sensor_data["distance"] = data.get("distance")

            alerts = []

            if latest_sensor_data["temperature"] is not None and latest_sensor_data["temperature"] > FEVER_TEMP_THRESHOLD:
                alerts.append("Fever Detected")

            if latest_sensor_data["sound"] is not None and latest_sensor_data["sound"] > SOUND_THRESHOLD:
                alerts.append("Loud Sound Detected")

            if last_distance is not None and latest_sensor_data["distance"] is not None and abs(latest_sensor_data["distance"] - last_distance) > FALL_DISTANCE_CHANGE:
                alerts.append("Fall Detected")

            latest_sensor_data["alerts"] = alerts
            last_distance = latest_sensor_data["distance"]

        except json.JSONDecodeError:
            print("Warning: Bad JSON skipped.")
        except Exception as e:
            print(f"Unexpected error: {e}")

# === API Route ===
@app.route('/latest_status', methods=['GET'])
def get_latest_status():
    return jsonify(latest_sensor_data)

# === Start Server ===
if __name__ == '__main__':
    threading.Thread(target=read_serial, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
