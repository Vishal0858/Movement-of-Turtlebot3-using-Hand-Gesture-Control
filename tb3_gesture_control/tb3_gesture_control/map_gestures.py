from geometry_msgs.msg import Twist

def gesture_to_twist(gesture: str) -> Twist:
    """
    Map gesture labels to Twist messages.
    Gestures expected: "UP", "DOWN", "LEFT", "RIGHT", "STOP"
    """
    t = Twist()
    # linear.x forward/back, angular.z turning
    max_lin = 0.22  # m/s (safe for turtlebot3 burger)
    max_ang = 0.2   # rad/s
    if gesture == "UP":
        t.linear.x = max_lin
        t.angular.z = 0.0
    elif gesture == "DOWN":
        t.linear.x = -0.12  # slow backward
        t.angular.z = 0.0
    elif gesture == "LEFT":
        t.linear.x = 0.0
        t.angular.z = max_ang
    elif gesture == "RIGHT":
        t.linear.x = 0.0
        t.angular.z = -max_ang
    elif gesture == "STOP":
        t.linear.x = 0.0
        t.angular.z = 0.0
    else:
        # unknown gestures -> stop
        t.linear.x = 0.0
        t.angular.z = 0.0
    return t
