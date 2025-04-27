class AnomalyDetector:
    def __init__(self, angle_threshold=40):
        self.angle_threshold = angle_threshold

    def detect(self, results):
        """Detect anomalies in pose data"""
        try:
            if not results or not isinstance(results, dict):
                return None
                
            # Check if person is lying down based on pose status
            pose_status = results.get('pose_status', 'unknown')
            if pose_status == 'lying':
                return "Lying down detected"
                
            # Check confidence level
            confidence = results.get('confidence', 0)
            if confidence < 0.5:  # Low confidence in pose detection
                return "Low pose confidence"
                
            return None
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return None 