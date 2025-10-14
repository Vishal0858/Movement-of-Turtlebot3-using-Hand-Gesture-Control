#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import cv2
import mediapipe as mp
import time
from tb3_gesture_control.map_gestures import gesture_to_twist

class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info('GestureNode started, initializing MediaPipe...')

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.x_history = []
        self.y_history = []
        self.movement_threshold = 0.08
        self.last_move_time = time.time()
        self.cooldown = 0.7
        self.direction = "STOP"
        self.last_publish_time = 0.0
        self.stop_after_no_detection = 1.0

        self.center_tolerance = 0.080  # <-- new: region around center where STOP triggers

        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.timer = self.create_timer(0.02, self.loop_callback)
        self._running = True

    def loop_callback(self):
        if not self._running:
            return

        success, frame = self.cap.read()
        if not success:
            self.get_logger().warning("Camera frame not available")
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        detected = False
        gesture = "STOP"  # default

        if results.multi_hand_landmarks:
            detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # wrist coordinates (normalized 0–1)
                x = hand_landmarks.landmark[0].x
                y = hand_landmarks.landmark[0].y

                self.x_history.append(x)
                self.y_history.append(y)
                if len(self.x_history) > 10:
                    self.x_history.pop(0)
                    self.y_history.pop(0)

                # detect motion
                if len(self.x_history) == 10:
                    dx = self.x_history[-1] - self.x_history[0]
                    dy = self.y_history[-1] - self.y_history[0]
                    now = time.time()

                    if now - self.last_move_time > self.cooldown:
                        # Determine direction
                        if abs(dx) > abs(dy):  # horizontal motion
                            if dx > self.movement_threshold:
                                gesture = "RIGHT"
                            elif dx < -self.movement_threshold:
                                gesture = "LEFT"
                        else:  # vertical motion
                            if dy > self.movement_threshold:
                                gesture = "DOWN"
                            elif dy < -self.movement_threshold:
                                gesture = "UP"

                        # If no strong movement, check for center position
                        if gesture == "STOP":
                            if abs(x - 0.5) < self.center_tolerance:
                                gesture = "STOP"
                            # else: keep previous direction (no update)

                        if gesture != "STOP":
                            self.last_move_time = now
                            self.direction = gesture

            # If hand is roughly centered and no motion detected → STOP
            if abs(x - 0.5) < self.center_tolerance and abs(y - 0.5) < self.center_tolerance:
                self.direction = "STOP"

        # if no hand for too long → STOP
        now = time.time()
        if not detected and (now - self.last_move_time) > self.stop_after_no_detection:
            self.direction = "STOP"

        # publish twist message
        if now - self.last_publish_time > 0.1:
            twist = gesture_to_twist(self.direction)
            self.pub.publish(twist)
            self.last_publish_time = now

        # HUD
        cv2.putText(frame, f"Gesture: {self.direction}", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.shutdown()

    def shutdown(self):
        self.pub.publish(Twist())
        self._running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.destroy_timer(self.timer)
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()

if __name__ == '__main__':
    main()
