# app.py

from flask import Flask, jsonify
from flask_cors import CORS
import threading
import serial
import json
import time
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# === Load environment variables ===
load_dotenv()

# === Arduino Serial Settings ===
ARDUINO_PORT = '/dev/cu.usbmodem101'  # Adjust USB/Bluetooth port
BAUD_RATE = 9600

# === Initialize Serial Connection ===
ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
ser.reset_input_buffer()

# === MongoDB Atlas Settings ===
MONGO_URI = os.getenv('MONGO_URI')  # From .env
PATIENT_ID = os.getenv('PATIENT_ID')  # From .env

mongo_client = MongoClient(MONGO_URI)
db = mongo_client['morganhack']    # Database name
collection = db['readings']         # Collection name

# === Shared Data ===
latest_sensor_data = {
    "temperature": None,
    "humidity": None,
    "sound": None,
    "distance": None
}

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

# === Background Serial Reader ===
def read_serial():
    global latest_sensor_data
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

            # === Insert into MongoDB ===
            record = {
                "timestamp": time.time(),  # UNIX timestamp
                "patient_id": PATIENT_ID,  # Loaded from .env
                "temperature": latest_sensor_data["temperature"],
                "humidity": latest_sensor_data["humidity"],
                "sound": latest_sensor_data["sound"],
                "distance": latest_sensor_data["distance"]
            }
            collection.insert_one(record)
            print(f"Inserted into MongoDB: {record}")

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
