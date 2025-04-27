import sys
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os
from pymongo import MongoClient, DESCENDING
import pyqtgraph as pg
from ultralytics import YOLO
import google.generativeai as genai
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGraphicsView, QGraphicsScene, QTextEdit, QPushButton,
    QScrollArea
)
from PyQt6.QtCore import (
    Qt, QTimer, QRectF, QPointF, QPoint
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QBrush, QPen,
    QLinearGradient, QTextCursor
)
from PyQt6.QtCharts import QChart, QChartView, QPieSeries, QLineSeries, QBarSeries, QBarSet, QValueAxis, QBarCategoryAxis
from collections import deque
from pose_estimation import PoseEstimator
from anomaly_detection import AnomalyDetector
from sms_alert import send_sms_alert

class DonutChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.setMinimumSize(150, 150)

    def setValue(self, value):
        self.value = min(max(value, 0), 100)  # Clamp between 0 and 100
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate dimensions
        width = self.width()
        height = self.height()
        center = QPointF(width / 2, height / 2)
        radius = min(width, height) / 2 - 10

        # Draw background circle
        painter.setPen(QPen(QColor(200, 200, 200), 10))
        painter.drawEllipse(center, radius, radius)

        # Draw value arc
        painter.setPen(QPen(QColor(0, 150, 255), 10))
        angle = int(-self.value * 360 / 100)  # Convert to integer angle
        rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        painter.drawArc(rect, 90 * 16, angle * 16)  # Qt uses 16th of a degree

        # Draw inner circle and text
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(QColor(255, 255, 255))
        inner_radius = radius * 0.7
        painter.drawEllipse(center, inner_radius, inner_radius)

        # Draw text
        painter.setPen(QColor(0, 0, 0))
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        text = f"{int(self.value)}%"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        painter.end()

class BarChart(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.value = 0
        self.setMinimumSize(100, 200)
        
    def setValue(self, value):
        self.value = min(max(value, 0), 100)  # Clamp between 0 and 100
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate dimensions
        width = self.width()
        height = self.height()
        padding = 20
        bar_width = width - 2*padding
        bar_height = height - 3*padding  # Extra padding for title
        
        # Draw background
        painter.setPen(QPen(QColor("#e0e0e0"), 1))
        painter.setBrush(QBrush(QColor("#f5f5f5")))
        painter.drawRect(padding, padding + padding, bar_width, bar_height)
        
        # Draw bar
        if self.value > 0:
            gradient = QLinearGradient(0, 0, 0, height)
            gradient.setColorAt(0, QColor("#4CAF50"))
            gradient.setColorAt(1, QColor("#2E7D32"))
            painter.setBrush(gradient)
            
            value_height = int((self.value / 100.0) * bar_height)
            painter.drawRect(padding,
                           height - padding - value_height,
                           bar_width,
                           value_height)
        
        # Draw scale lines
        painter.setPen(QPen(QColor("#757575"), 1))
        for i in range(0, 101, 20):
            y = int(height - padding - (i / 100.0) * bar_height)
            p1 = QPoint(padding - 5, y)
            p2 = QPoint(padding, y)
            painter.drawLine(p1, p2)
            painter.drawText(2, y + 5, str(i))
        
        # Draw title
        painter.setPen(QColor("black"))
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(QRectF(0, 0, width, padding),
                        Qt.AlignmentFlag.AlignCenter, self.title)
        
        # Draw value
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(QRectF(0, height - 2*padding, width, padding),
                        Qt.AlignmentFlag.AlignCenter, f"{self.value}")

class CaretakerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CareBotix - Smart Patient Monitoring System")
        self.setGeometry(100, 100, 1600, 800)  # Made window wider for chat panel

        # Initialize fall alert tracking
        self.fall_alert_start_time = None
        self.sms_sent = False

        # Initialize components
        self.init_camera()
        self.init_pose_estimation()
        self.init_data_storage()
        self.init_mongodb()
        self.init_gemini()
        self.init_ui()

        # Start timers
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(30)  # 30ms = ~33 fps

        self.sensor_timer = QTimer()
        self.sensor_timer.timeout.connect(self.update_sensor_data)
        self.sensor_timer.start(1000)  # Update every second

    def init_camera(self):
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            sys.exit()
        print("Camera initialized successfully")

    def init_pose_estimation(self):
        print("Initializing pose estimation...")
        try:
            self.pose_estimator = PoseEstimator()
            self.anomaly_detector = AnomalyDetector()
            self.pose_confidence_data = deque(maxlen=100)
            self.pose_status = "No person detected"
            print("Pose estimation initialized successfully")
        except Exception as e:
            print(f"Error initializing pose estimation: {e}")
            sys.exit()

    def init_data_storage(self):
        self.temperature_data = deque(maxlen=10)
        self.humidity_data = deque(maxlen=10)
        self.distance_data = deque(maxlen=10)

    def init_mongodb(self):
        load_dotenv()
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            print("Error: MONGO_URI not found in environment variables")
            mongo_uri = 'mongodb://localhost:27017'
        
        try:
            self.client = MongoClient(mongo_uri)
            self.client.admin.command('ping')  # Test connection
            self.db = self.client['morganhack']
            self.collection = self.db['readings']
            print("Successfully connected to MongoDB")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None

    def init_gemini(self):
        """Initialize Gemini chatbot"""
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.gemini_model = genai.GenerativeModel("gemini-1.5-pro")
            print("Gemini chatbot initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini chatbot: {e}")
            self.gemini_model = None

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel for video feed and pose graph
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video feed
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)
        
        # Pose confidence graph
        self.pose_graph = pg.PlotWidget()
        self.pose_graph.setBackground('w')
        self.pose_graph.setTitle("Pose Confidence")
        self.pose_graph.setLabel('left', 'Confidence')
        self.pose_graph.setLabel('bottom', 'Time')
        self.pose_graph.setYRange(0, 1)
        self.pose_curve = self.pose_graph.plot(pen='g')
        left_layout.addWidget(self.pose_graph)
        
        # Status and Alert Labels
        status_alert_widget = QWidget()
        status_alert_layout = QVBoxLayout(status_alert_widget)
        
        self.pose_status_label = QLabel("Status: Monitoring")
        self.pose_status_label.setStyleSheet("QLabel { color: blue; font-size: 14pt; }")
        status_alert_layout.addWidget(self.pose_status_label)
        
        self.alert_label = QLabel("")
        self.alert_label.setStyleSheet("QLabel { color: red; font-size: 14pt; font-weight: bold; }")
        status_alert_layout.addWidget(self.alert_label)
        
        left_layout.addWidget(status_alert_widget)
        layout.addWidget(left_panel)

        # Middle panel for sensor data
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        
        # Temperature chart
        temp_widget = QWidget()
        temp_layout = QVBoxLayout(temp_widget)
        temp_label = QLabel("Temperature (°C)")
        temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        temp_label.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; }")
        temp_layout.addWidget(temp_label)
        self.temp_chart = DonutChart()
        temp_layout.addWidget(self.temp_chart)
        self.temp_value_label = QLabel("-- °C")
        self.temp_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.temp_value_label.setStyleSheet("QLabel { font-size: 11pt; }")
        temp_layout.addWidget(self.temp_value_label)
        middle_layout.addWidget(temp_widget)
        
        # Humidity chart
        humidity_widget = QWidget()
        humidity_layout = QVBoxLayout(humidity_widget)
        humidity_label = QLabel("Humidity (%)")
        humidity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        humidity_label.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; }")
        humidity_layout.addWidget(humidity_label)
        self.humidity_chart = DonutChart()
        humidity_layout.addWidget(self.humidity_chart)
        self.humidity_value_label = QLabel("-- %")
        self.humidity_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.humidity_value_label.setStyleSheet("QLabel { font-size: 11pt; }")
        humidity_layout.addWidget(self.humidity_value_label)
        middle_layout.addWidget(humidity_widget)
        
        # Sound Level chart
        sound_widget = QWidget()
        sound_layout = QVBoxLayout(sound_widget)
        sound_label = QLabel("Sound Level (dB)")
        sound_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sound_label.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; }")
        sound_layout.addWidget(sound_label)
        self.sound_chart = DonutChart()
        sound_layout.addWidget(self.sound_chart)
        self.sound_value_label = QLabel("-- dB")
        self.sound_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sound_value_label.setStyleSheet("QLabel { font-size: 11pt; }")
        sound_layout.addWidget(self.sound_value_label)
        middle_layout.addWidget(sound_widget)

        # Distance bar chart
        distance_widget = QWidget()
        distance_layout = QVBoxLayout(distance_widget)
        distance_label = QLabel("Distance from Bed (cm)")
        distance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        distance_label.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; }")
        distance_layout.addWidget(distance_label)
        self.distance_chart = BarChart("Distance")
        distance_layout.addWidget(self.distance_chart)
        self.distance_value_label = QLabel("-- cm")
        self.distance_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.distance_value_label.setStyleSheet("QLabel { font-size: 11pt; }")
        distance_layout.addWidget(self.distance_value_label)
        middle_layout.addWidget(distance_widget)
        
        layout.addWidget(middle_panel)

        # Right panel for chat
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        
        # Chat title
        chat_title = QLabel("CareBotix Assistant")
        chat_title.setStyleSheet("QLabel { font-size: 14pt; font-weight: bold; color: #2196F3; }")
        chat_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chat_layout.addWidget(chat_title)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                font-size: 11pt;
            }
        """)
        chat_layout.addWidget(self.chat_display)
        
        # Input area
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(60)
        self.chat_input.setPlaceholderText("Type your message here...")
        self.chat_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                font-size: 11pt;
            }
        """)
        input_layout.addWidget(self.chat_input)
        
        send_button = QPushButton("Send")
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)
        
        chat_layout.addWidget(input_widget)

        # FAQ buttons
        faq_widget = QWidget()
        faq_layout = QVBoxLayout(faq_widget)
        faq_layout.setSpacing(5)
        
        faq_title = QLabel("Quick Questions")
        faq_title.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; color: #424242; }")
        faq_layout.addWidget(faq_title)
        
        faq_questions = [
            "How is the patient's temperature?",
            "Is the patient's posture normal?",
            "What are the current sensor readings?",
            "Any alerts in the last hour?",
            "Summarize patient's condition"
        ]
        
        for question in faq_questions:
            btn = QPushButton(question)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 5px;
                    text-align: left;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)
            btn.clicked.connect(lambda checked, q=question: self.send_faq(q))
            faq_layout.addWidget(btn)
        
        chat_layout.addWidget(faq_widget)
        
        layout.addWidget(chat_panel)

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return

            # Process frame with pose estimation and get annotated frame
            annotated_frame, results, pose_status = self.pose_estimator.estimate(frame)
            confidence = results['confidence']

            # Update pose status label with color coding
            status_color = {
                'standing': 'green',
                'sitting': 'orange',
                'lying': 'red',
                'no_person': 'gray',
                'unknown': 'gray',
                'error': 'red'
            }.get(pose_status, 'gray')

            status_text = {
                'standing': 'Person is standing',
                'sitting': 'Person is sitting',
                'lying': 'ALERT: Person is lying down!',
                'no_person': 'No person detected',
                'unknown': 'Pose unknown',
                'error': 'Error in pose detection'
            }.get(pose_status, 'Unknown status')

            self.pose_status_label.setText(status_text)
            self.pose_status_label.setStyleSheet(f'color: {status_color}; font-weight: bold;')

            # Update confidence graph
            self.pose_confidence_data.append(confidence)
            self.pose_curve.setData(list(self.pose_confidence_data))

            # Check for lying down pose and update alerts
            if pose_status == 'lying':
                # Start tracking fall alert time if not already tracking
                if self.fall_alert_start_time is None:
                    self.fall_alert_start_time = datetime.now()
                    self.sms_sent = False
                    print("Fall detected - starting timer")
                    # Immediately show alert in red
                    self.alert_label.setText('⚠️ ALERT: Person may have fallen!')
                    self.alert_label.setStyleSheet('color: red; font-size: 14pt; font-weight: bold;')

                # Calculate how long the alert has been active
                alert_duration = (datetime.now() - self.fall_alert_start_time).total_seconds()
                print(f"Fall alert duration: {alert_duration:.1f} seconds")

                # Send SMS if alert persists for 5 seconds and hasn't been sent yet
                if alert_duration >= 5 and not self.sms_sent:
                    try:
                        # Get latest sensor data for the message
                        latest_reading = None
                        if self.collection is not None:  # Fixed collection check
                            try:
                                latest_reading = self.collection.find_one(
                                    {"patient_id": os.getenv('PATIENT_ID', '00000001')},
                                    sort=[('timestamp', -1)]
                                )
                            except Exception as e:
                                print(f"Error fetching sensor data: {str(e)}")

                        # Create detailed message
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        details = []
                        if latest_reading:
                            details.extend([
                                f"Temperature: {latest_reading.get('temperature', 'N/A')}°C",
                                f"Distance from bed: {latest_reading.get('distance', 'N/A')}cm"
                            ])
                        
                        message = (
                            f"[{timestamp}] URGENT: Patient Fall Detected\n"
                            f"Alert Duration: {int(alert_duration)}s\n"
                            f"Details: {', '.join(details)}\n"
                            "Immediate caregiver attention required!"
                        )
                        
                        print("Sending SMS alert...")
                        send_sms_alert(message)
                        self.sms_sent = True
                        print("SMS alert sent successfully")
                        
                        # Update alert label with SMS notification
                        self.alert_label.setText('⚠️ URGENT: Fall Detected - SMS Alert Sent to Caregiver!')
                        self.alert_label.setStyleSheet('color: red; font-size: 14pt; font-weight: bold;')
                    except Exception as e:
                        print(f"Failed to send SMS alert: {str(e)}")
                        import traceback
                        traceback.print_exc()
            else:
                # Reset fall tracking when pose is no longer 'lying'
                if self.fall_alert_start_time is not None:
                    print("Fall alert cleared - person no longer lying down")
                self.fall_alert_start_time = None
                self.sms_sent = False
                
                # Reset alert label
                self.alert_label.setText('Status: Normal')
                self.alert_label.setStyleSheet('color: green; font-size: 14pt;')

            # Convert annotated frame to QImage and display
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error in update_frame: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_sensor_data(self):
        if self.client is None or self.db is None or self.collection is None:
            print("MongoDB connection not available")
            return
            
        try:
            # Get the 10 most recent readings
            recent_readings = list(self.collection.find().sort('timestamp', -1).limit(10))
            
            if recent_readings:
                latest = recent_readings[0]
                print(f"Latest sensor reading: {latest}")
                
                # Update temperature
                temperature = min(float(latest.get('temperature', 0)), 100)
                self.temp_chart.setValue(temperature)
                self.temp_value_label.setText(f"{temperature:.1f} °C")
                
                # Update humidity
                humidity = min(float(latest.get('humidity', 0)), 100)
                self.humidity_chart.setValue(humidity)
                self.humidity_value_label.setText(f"{humidity:.1f} %")
                
                # Update sound level
                sound = min(float(latest.get('sound', 0)), 100)
                self.sound_chart.setValue(sound)
                self.sound_value_label.setText(f"{sound:.0f} dB")

                # Update distance with better error handling and scaling
                try:
                    distance = float(latest.get('distance', 0))
                    # Ensure distance is positive and cap at 100cm for visualization
                    distance = max(0, min(distance, 100))
                    
                    # Scale distance for bar chart (0-100cm maps to 0-100%)
                    scaled_distance = (distance / 100) * 100
                    
                    self.distance_chart.setValue(scaled_distance)
                    self.distance_value_label.setText(f"{distance:.1f} cm")
                    
                    # Update distance alert threshold
                    if distance < 30:  # Alert if closer than 30cm
                        self.alert_label.setText("⚠️ Patient too close to bed!")
                        self.alert_label.setStyleSheet("color: red; font-size: 14pt; font-weight: bold;")
                except (ValueError, TypeError) as e:
                    print(f"Error processing distance value: {e}")
                
                # Check for environmental alerts
                alerts = []
                if temperature > 30:
                    alerts.append("⚠️ High Temperature!")
                if humidity > 70:
                    alerts.append("⚠️ High Humidity!")
                if sound > 80:
                    alerts.append("⚠️ High Noise Level!")
                
                if alerts:
                    current_alert = self.alert_label.text()
                    if not current_alert.startswith("⚠️ Patient too close"):  # Don't override distance alert
                        self.alert_label.setText("\n".join(alerts))
                        self.alert_label.setStyleSheet("color: red; font-size: 14pt; font-weight: bold;")
                
                print(f"Updated values - Temp: {temperature}, Humidity: {humidity}, Sound: {sound}, Distance: {distance:.1f}")
            else:
                print("No sensor readings found")

        except Exception as e:
            print(f"Error updating sensor data: {e}")
            import traceback
            traceback.print_exc()

    def get_chat_response(self, user_input):
        """Get response from Gemini chatbot"""
        try:
            # Get recent sensor readings
            sensor_logs = list(
                self.collection.find({"patient_id": os.getenv('PATIENT_ID', '00000001')})
                .sort("timestamp", DESCENDING)
                .limit(20)
            )

            # Format sensor summaries
            sensor_summary = "\n".join(
                f"{i+1}. {datetime.fromtimestamp(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} — " +
                f"Temperature: {log.get('temperature', 'N/A')}°C, " +
                f"Humidity: {log.get('humidity', 'N/A')}%, " +
                f"Sound: {log.get('sound', 'N/A')}, " +
                f"Distance: {log.get('distance', 'N/A')} cm"
                for i, log in enumerate(sensor_logs)
            ) or "No sensor data available."

            # Get latest reading for quick reference
            latest_reading = sensor_logs[0] if sensor_logs else None
            current_temp = latest_reading.get('temperature', 'N/A') if latest_reading else 'N/A'
            current_humidity = latest_reading.get('humidity', 'N/A') if latest_reading else 'N/A'
            current_sound = latest_reading.get('sound', 'N/A') if latest_reading else 'N/A'

            # Build prompt with temperature interpretation guide
            prompt = f"""You are CareBotix, an empathetic and supportive AI health assistant monitoring a patient's environment.

Current Readings:
- Temperature: {current_temp}°C
- Humidity: {current_humidity}%
- Sound Level: {current_sound} dB
- Patient Status: {self.pose_status_label.text()}
- Alerts: {self.alert_label.text()}

Temperature Guide:
- Normal room temperature: 20-25°C
- Comfortable for patients: 21-24°C
- Too warm: >26°C
- Too cold: <20°C

Recent History:
{sensor_summary}

Question: {user_input}

Please provide a clear, informative response that:
1. Directly answers the question
2. Interprets the values in a medical context
3. Suggests any relevant actions if needed
4. Uses a caring, professional tone"""

            # Get response from Gemini
            response = self.gemini_model.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"Error getting chat response: {e}")
            import traceback
            traceback.print_exc()
            return "I apologize, but I'm having trouble processing your request at the moment. Please try again later."

    def send_message(self):
        """Send a message to the chatbot"""
        user_input = self.chat_input.toPlainText().strip()
        if not user_input:
            return

        # Add user message to chat display
        self.chat_display.append(f"<p style='color: #2196F3;'><b>You:</b> {user_input}</p>")
        self.chat_input.clear()

        # Get and display bot response
        response = self.get_chat_response(user_input)
        
        # Format the response with proper HTML
        formatted_response = response.replace('\n', '<br>')
        self.chat_display.append(f"<p style='color: #4CAF50;'><b>CareBotix:</b> {formatted_response}</p>")
        
        # Scroll to bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def send_faq(self, question):
        """Send a FAQ question to the chatbot"""
        self.chat_input.setPlainText(question)
        self.send_message()

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Return and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Send message on Enter (but allow Shift+Enter for new line)
            if self.chat_input.hasFocus():
                self.send_message()
                event.accept()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.cap.release()
        self.client.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the GUI
    gui = CaretakerGUI()
    gui.show()
    
    sys.exit(app.exec()) 