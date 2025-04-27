from pymongo import MongoClient
from dotenv import load_dotenv
import os
import sys
from datetime import datetime

# Load environment variables
load_dotenv()

def get_database_connection():
    try:
        # Get MongoDB URI from environment
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            raise ValueError("MONGO_URI not found in environment variables")

        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        
        # Test the connection
        client.admin.command('ping')
        
        # Get database and collection
        db = client['morganhack']
        collection = db['readings']
        
        return client, db, collection
    
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}", file=sys.stderr)
        return None, None, None

def get_latest_reading(patient_id="00000001"):
    """Get the most recent sensor reading for a patient"""
    try:
        client, db, collection = get_database_connection()
        if all(x is None for x in (client, db, collection)):
            return None
        
        # Get the most recent reading
        reading = collection.find_one(
            {"patient_id": patient_id},
            sort=[("timestamp", -1)]
        )
        
        if reading:
            # Convert ObjectId to string
            reading['_id'] = str(reading['_id'])
            
            # Format the data
            formatted_reading = {
                'timestamp': datetime.fromtimestamp(reading['timestamp']).isoformat(),
                'temperature': round(float(reading['temperature']), 1),
                'humidity': round(float(reading['humidity']), 1),
                'sound': int(reading['sound']),
                'distance': round(float(reading['distance']), 2),
                'patient_id': reading['patient_id']
            }
            return formatted_reading
        
        return None
    
    except Exception as e:
        print(f"Error fetching latest reading: {e}", file=sys.stderr)
        return None

def get_patient_readings(patient_id="00000001", limit=10):
    """Get recent sensor readings for a patient"""
    try:
        client, db, collection = get_database_connection()
        if all(x is None for x in (client, db, collection)):
            return []
        
        # Get recent readings
        readings = list(collection.find(
            {"patient_id": patient_id},
            sort=[("timestamp", -1)],
            limit=limit
        ))
        
        # Format the readings
        formatted_readings = []
        for reading in readings:
            reading['_id'] = str(reading['_id'])
            formatted_reading = {
                'timestamp': datetime.fromtimestamp(reading['timestamp']).isoformat(),
                'temperature': round(float(reading['temperature']), 1),
                'humidity': round(float(reading['humidity']), 1),
                'sound': int(reading['sound']),
                'distance': round(float(reading['distance']), 2),
                'patient_id': reading['patient_id']
            }
            formatted_readings.append(formatted_reading)
        
        return formatted_readings
    
    except Exception as e:
        print(f"Error fetching readings: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    # Test the connection and fetch readings
    latest = get_latest_reading()
    if latest:
        print("\nLatest reading:")
        print(f"Temperature: {latest['temperature']}°C")
        print(f"Humidity: {latest['humidity']}%")
        print(f"Sound Level: {latest['sound']} dB")
        print(f"Distance: {latest['distance']} cm")
        print(f"Timestamp: {latest['timestamp']}")
    
    print("\nRecent readings:")
    readings = get_patient_readings(limit=5)
    for reading in readings:
        print(f"- {reading['timestamp']}: Temp={reading['temperature']}°C, Dist={reading['distance']}cm") 