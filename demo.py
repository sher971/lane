import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

class LaneAndObjectDetector:
    def __init__(self):
        # Camera setup
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # GPIO setup for motor control
        GPIO.setmode(GPIO.BCM)
        self.MOTOR_LEFT_PIN1 = 17
        self.MOTOR_LEFT_PIN2 = 18
        self.MOTOR_RIGHT_PIN1 = 22
        self.MOTOR_RIGHT_PIN2 = 23
        
        # Ultrasonic sensor pins
        self.TRIG_PIN = 24
        self.ECHO_PIN = 25

        # Setup GPIO pins
        GPIO.setup(self.MOTOR_LEFT_PIN1, GPIO.OUT)
        GPIO.setup(self.MOTOR_LEFT_PIN2, GPIO.OUT)
        GPIO.setup(self.MOTOR_RIGHT_PIN1, GPIO.OUT)
        GPIO.setup(self.MOTOR_RIGHT_PIN2, GPIO.OUT)
        GPIO.setup(self.TRIG_PIN, GPIO.OUT)
        GPIO.setup(self.ECHO_PIN, GPIO.IN)

    def detect_lanes(self, frame):
        """
        Detect lanes in the road using computer vision techniques
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Region of interest
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([
            [(0, height), 
             (width//2, height*0.6), 
             (width, height)]
        ])
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50, 
            minLineLength=100, 
            maxLineGap=50
        )
        
        # Draw detimport cv2
import numpy as np
import RPi.GPIO as GPIO
import time

class LaneDetection:
    def __init__(self):
        # Camera setup
        self.camera = cv2.VideoCapture(0)  # Use default camera
        
        # Motor GPIO pins (adjust according to your specific motor driver)
        self.MOTOR_LEFT_FORWARD = 17
        self.MOTOR_LEFT_BACKWARD = 27
        self.MOTOR_RIGHT_FORWARD = 22
        self.MOTOR_RIGHT_BACKWARD = 23
        
        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.MOTOR_LEFT_FORWARD, GPIO.OUT)
        GPIO.setup(self.MOTOR_LEFT_BACKWARD, GPIO.OUT)
        GPIO.setup(self.MOTOR_RIGHT_FORWARD, GPIO.OUT)
        GPIO.setup(self.MOTOR_RIGHT_BACKWARD, GPIO.OUT)
    
    def preprocess_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def region_of_interest(self, edges):
        # Define region of interest
        height, width = edges.shape
        mask = np.zeros_like(edges)
        
        # Define polygon for ROI (adjust these points based on your camera setup)
        polygon = np.array([
            [(0, height),
             (width/2, height*0.6),
             (width, height)]
        ], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        return masked_edges
    
    def detect_lanes(self, masked_edges):
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges, 
            1,              # rho
            np.pi/180,      # theta
            threshold=50,   # minimum intersections to detect a line
            minLineLength=100,
            maxLineGap=50
        )
        
        return lines
    
    def calculate_steering(self, lines):
        if lines is None:
            return 0  # No steering if no lines detected
        
        left_lines = []
        right_lines = []
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1 + 1e-5)
                
                if slope < 0:
                    left_lines.append(slope)
                elif slope > 0:
                    right_lines.append(slope)
        
        # Calculate average slopes
        left_slope = np.mean(left_lines) if len(left_lines) > 0 else 0
        right_slope = np.mean(right_lines) if len(right_lines) > 0 else 0
        
        # Simple steering logic
        steering = (left_slope + right_slope) / 2
        return steering
    
    def control_motors(self, steering):
        # Basic motor control based on steering
        if steering < -0.1:
            # Turn left
            GPIO.output(self.MOTOR_LEFT_FORWARD, GPIO.LOW)
            GPIO.output(self.MOTOR_LEFT_BACKWARD, GPIO.HIGH)
            GPIO.output(self.MOTOR_RIGHT_FORWARD, GPIO.HIGH)
            GPIO.output(self.MOTOR_RIGHT_BACKWARD, GPIO.LOW)
        elif steering > 0.1:
            # Turn right
            GPIO.output(self.MOTOR_LEFT_FORWARD, GPIO.HIGH)
            GPIO.output(self.MOTOR_LEFT_BACKWARD, GPIO.LOW)
            GPIO.output(self.MOTOR_RIGHT_FORWARD, GPIO.LOW)
            GPIO.output(self.MOTOR_RIGHT_BACKWARD, GPIO.HIGH)
        else:
            # Move forward
            GPIO.output(self.MOTOR_LEFT_FORWARD, GPIO.HIGH)
            GPIO.output(self.MOTOR_LEFT_BACKWARD, GPIO.LOW)
            GPIO.output(self.MOTOR_RIGHT_FORWARD, GPIO.HIGH)
            GPIO.output(self.MOTOR_RIGHT_BACKWARD, GPIO.LOW)
    
    def run(self):
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Preprocess frame
                edges = self.preprocess_frame(frame)
                
                # Define region of interest
                masked_edges = self.region_of_interest(edges)
                
                # Detect lanes
                lines = self.detect_lanes(masked_edges)
                
                # Calculate steering
                steering = self.calculate_steering(lines)
                
                # Control motors
                self.control_motors(steering)
                
                # Optional: Display frame for debugging
                cv2.imshow('Lane Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Stopped by user")
        
        finally:
            # Cleanup
            self.camera.release()
            cv2.destroyAllWindows()
            GPIO.cleanup()

# Main execution
if __name__ == "__main__":
    lane_detector = LaneDetection()
    lane_detector.run()