#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # Still needed for debug image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np

class HandGestureDetector(Node):
    def __init__(self):
        super().__init__('gesture_detector_node')

        # --- Parameters ---
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('debug_image_topic', '~/debug_image')
        
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value

        # --- Constants ---
        self.base_linear_speed = 0.5  # m/s
        self.base_angular_speed = 1.0 # rad/s
        self.FORWARD_Y_THRESH = 0.4
        self.BACKWARD_Y_THRESH = 0.6
        self.LEFT_X_THRESH = 0.4
        self.RIGHT_X_THRESH = 0.6
        self.VEL_MIN_Y = 0.8
        self.VEL_MAX_Y = 0.2
        
        # --- ROS 2 Setup ---
        self.bridge = CvBridge()
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, debug_image_topic, 10)

        # --- OpenCV Camera Setup ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("!!! Could not open camera. Exiting...")
            rclpy.shutdown()
            return
            
        # Set a reasonable resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.get_logger().info("Camera opened successfully.")

        # --- MediaPipe Setup ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # --- State Variables ---
        self.current_velocity_scale = 0.0
        self._running = True

        # --- ROS 2 Timer ---
        # We replace the subscriber with a timer to process frames
        self.timer = self.create_timer(0.02, self.timer_callback) # 50 FPS

        self.get_logger().info(f"Gesture Detector Node Started (using cv2.VideoCapture).")
        self.get_logger().info(f"Publishing commands to: {cmd_vel_topic}")
        self.get_logger().info("Right hand: Direction | Left hand: Velocity")

    def timer_callback(self):
        if not self._running:
            return
            
        # 1. Read frame from camera
        success, cv_frame = self.cap.read()
        if not success:
            self.get_logger().warn("Failed to grab frame from camera.")
            return

        # 2. Process the frame (same logic as before)
        # Flip, convert to RGB, and process with MediaPipe
        cv_frame = cv2.flip(cv_frame, 1)
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Initialize gesture variables
        lin_x = 0.0
        ang_z = 0.0
        direction_text = "STOP"
        direction_hand_visible = False
        velocity_hand_visible = False
        
        # Create debug image and draw zones
        debug_image = cv_frame.copy()
        h, w, _ = debug_image.shape
        cv2.line(debug_image, (int(self.LEFT_X_THRESH * w), 0), (int(self.LEFT_X_THRESH * w), h), (255, 0, 0), 1)
        cv2.line(debug_image, (int(self.RIGHT_X_THRESH * w), 0), (int(self.RIGHT_X_THRESH * w), h), (255, 0, 0), 1)
        cv2.line(debug_image, (0, int(self.FORWARD_Y_THRESH * h)), (w, int(self.FORWARD_Y_THRESH * h)), (255, 0, 0), 1)
        cv2.line(debug_image, (0, int(self.BACKWARD_Y_THRESH * h)), (w, int(self.BACKWARD_Y_THRESH * h)), (255, 0, 0), 1)
        cv2.line(debug_image, (0, int(self.VEL_MIN_Y * h)), (w, int(self.VEL_MIN_Y * h)), (0, 255, 0), 2)
        cv2.line(debug_image, (0, int(self.VEL_MAX_Y * h)), (w, int(self.VEL_MAX_Y * h)), (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_drawing.draw_landmarks(
                    debug_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                hand_label = handedness.classification[0].label

                # --- Process Right Hand (Direction) ---
                if hand_label == 'Right':
                    direction_hand_visible = True
                    palm_center = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    palm_x = palm_center.x
                    palm_y = palm_center.y
                    
                    if palm_y < self.FORWARD_Y_THRESH:
                        lin_x = self.base_linear_speed
                        direction_text = "FORWARD"
                    elif palm_y > self.BACKWARD_Y_THRESH:
                        lin_x = -self.base_linear_speed
                        direction_text = "BACKWARD"
                    elif palm_x < self.LEFT_X_THRESH:
                        ang_z = self.base_angular_speed
                        direction_text = "LEFT"
                    elif palm_x > self.RIGHT_X_THRESH:
                        ang_z = -self.base_angular_speed
                        direction_text = "RIGHT"
                    else:
                        direction_text = "STOP"
                        
                # --- Process Left Hand (Velocity) ---
                if hand_label == 'Left':
                    velocity_hand_visible = True
                    wrist_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
                    scale = (self.VEL_MIN_Y - wrist_y) / (self.VEL_MIN_Y - self.VEL_MAX_Y)
                    self.current_velocity_scale = max(0.0, min(1.0, scale))

        # --- Safety Checks ---
        if not velocity_hand_visible:
            self.current_velocity_scale = 0.0
        if not direction_hand_visible:
            lin_x = 0.0
            ang_z = 0.0
            direction_text = "STOP"

        # 3. Publish Twist Message
        twist_msg = Twist()
        twist_msg.linear.x = lin_x * self.current_velocity_scale
        twist_msg.angular.z = ang_z * self.current_velocity_scale
        self.cmd_vel_pub.publish(twist_msg)

        # 4. Show local window and publish debug image
        cv2.putText(debug_image, f"Direction: {direction_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(debug_image, f"Velocity Scale: {self.current_velocity_scale*100:.0f}%", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the local pop-up window
        cv2.imshow("Gesture Control", debug_image)
        
        # Also publish to RQt
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f'CvBridge Error on debug publish: {e}')

        # Check for 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("'q' pressed, shutting down.")
            self.destroy_node() # This will stop the spin() in main

    def shutdown_cleanup(self):
        """Called on node destruction to clean up resources."""
        self.get_logger().info("Cleaning up resources...")
        self._running = False
        if self.timer:
            self.timer.cancel()
        
        # Send a final stop command
        self.cmd_vel_pub.publish(Twist()) 
        
        # Release OpenCV resources
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.get_logger().info("Shutdown complete.")

def main(args=None):
    rclpy.init(args=args)
    node = HandGestureDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        # This cleanup runs whether spin() exits normally,
        # from 'q' press (destroy_node), or Ctrl+C
        node.shutdown_cleanup()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

