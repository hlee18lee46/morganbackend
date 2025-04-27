from ultralytics import YOLO
import cv2
import numpy as np
import torch

class PoseEstimator:
    def __init__(self, model_path='yolov8n-pose.pt'):
        """Initialize the pose estimator with the YOLO model"""
        self.model = YOLO(model_path)

    def process_frame(self, frame):
        try:
            # Run YOLO pose estimation
            results = self.model(frame, verbose=False)[0]
            
            if not results.keypoints or len(results.keypoints) == 0:
                return {'pose_status': 'no_person', 'confidence': 0.0}

            # Get keypoints and confidence
            keypoints = results.keypoints[0]  # Get first person's keypoints
            confidence = float(results.boxes.conf[0]) if len(results.boxes.conf) > 0 else 0.0

            # Get relevant keypoints for pose detection
            # Assuming COCO format: [nose, neck, ..., right ankle, right knee, right hip]
            if len(keypoints) >= 17:  # COCO format has 17 keypoints
                # Calculate angle between hip, knee and ankle
                right_hip = keypoints[12]
                right_knee = keypoints[13]
                right_ankle = keypoints[14]
                
                # Calculate vectors
                hip_to_knee = right_knee - right_hip
                knee_to_ankle = right_ankle - right_knee
                
                # Calculate angle
                angle = np.degrees(np.arccos(
                    np.dot(hip_to_knee, knee_to_ankle) / 
                    (np.linalg.norm(hip_to_knee) * np.linalg.norm(knee_to_ankle))
                ))

                # Determine pose based on angle
                if angle < 100:  # Person is likely standing
                    pose_status = 'standing'
                elif angle > 160:  # Person is likely lying down
                    pose_status = 'lying_down'
                else:  # Person is likely sitting
                    pose_status = 'sitting'
            else:
                pose_status = 'unknown'

            return {
                'pose_status': pose_status,
                'confidence': confidence
            }

        except Exception as e:
            print(f"Error in pose estimation: {str(e)}")
            return {'pose_status': 'error', 'confidence': 0.0}

    def estimate(self, frame):
        """Estimate pose in the frame"""
        try:
            # Process frame with YOLO
            results = self.model(frame, verbose=False)[0]
            
            # Initialize default values
            annotated = frame.copy()
            pose_status = 'unknown'
            confidence = 0.0
            
            # Process results if available
            if results.keypoints is not None and len(results.keypoints) > 0:
                # Get keypoints and confidence
                keypoints = results.keypoints[0].data[0].cpu().numpy()  # Get first person's keypoints
                if len(results.boxes) > 0:
                    confidence = float(results.boxes[0].conf.cpu().numpy())
                
                # Draw keypoints and skeleton
                if len(keypoints) > 0:
                    # Define skeleton pairs (YOLO format)
                    skeleton = [
                        [5, 7], [7, 9],  # Left arm
                        [6, 8], [8, 10],  # Right arm
                        [5, 6],  # Shoulders
                        [5, 11], [6, 12],  # Spine to hips
                        [11, 13], [13, 15],  # Left leg
                        [12, 14], [14, 16]  # Right leg
                    ]
                    
                    # Draw keypoints
                    for kpt in keypoints:
                        x, y = int(kpt[0]), int(kpt[1])
                        if x > 0 and y > 0:  # Only draw if point is valid
                            cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)  # Inner circle
                            cv2.circle(annotated, (x, y), 6, (255, 255, 255), 2)  # Outer circle
                    
                    # Draw skeleton
                    for pair in skeleton:
                        if (pair[0] < len(keypoints) and pair[1] < len(keypoints)):
                            pt1 = keypoints[pair[0]]
                            pt2 = keypoints[pair[1]]
                            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                pt1 = (int(pt1[0]), int(pt1[1]))
                                pt2 = (int(pt2[0]), int(pt2[1]))
                                cv2.line(annotated, pt1, pt2, (0, 255, 0), 3)  # Thick green line
                                cv2.line(annotated, pt1, pt2, (255, 255, 255), 1)  # Thin white overlay
                    
                    # Determine pose status
                    if len(keypoints) >= 17:  # Full body detection
                        left_shoulder = keypoints[5]
                        right_shoulder = keypoints[6]
                        left_hip = keypoints[11]
                        right_hip = keypoints[12]
                        
                        # Check if key points are detected
                        if (all(p[0] > 0 and p[1] > 0 for p in [left_shoulder, right_shoulder, left_hip, right_hip])):
                            # Calculate spine vector
                            mid_shoulder = np.array([(left_shoulder[0] + right_shoulder[0])/2,
                                                   (left_shoulder[1] + right_shoulder[1])/2])
                            mid_hip = np.array([(left_hip[0] + right_hip[0])/2,
                                              (left_hip[1] + right_hip[1])/2])
                            spine = mid_hip - mid_shoulder
                            
                            # Calculate angle with vertical
                            angle = abs(np.degrees(np.arctan2(spine[0], -spine[1])))
                            
                            # Classify pose
                            if angle < 30:
                                pose_status = 'standing'
                            elif angle > 60:
                                pose_status = 'lying'
                            else:
                                pose_status = 'sitting'
                            
                            # Draw angle and status
                            text_pos = (int(mid_shoulder[0]), int(mid_shoulder[1]) - 20)
                            cv2.putText(annotated, f"{pose_status.upper()} ({angle:.1f}°)",
                                      text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                      (255, 255, 255), 4)  # White outline
                            cv2.putText(annotated, f"{pose_status.upper()} ({angle:.1f}°)",
                                      text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                      (0, 255, 0), 2)  # Green text
            
            return annotated, {'confidence': confidence, 'pose_status': pose_status}, pose_status
            
        except Exception as e:
            print(f"Error in pose estimation: {e}")
            import traceback
            traceback.print_exc()
            return frame, {'confidence': 0.0, 'pose_status': 'unknown'}, 'unknown' 