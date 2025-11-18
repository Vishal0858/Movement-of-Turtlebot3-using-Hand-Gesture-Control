Steps to run the package
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------

Step1:- Unzip the package in the source folder of your workspace
------------------------------------------------------------------------------------------

Step2:- Build the package using the command
------------------------------------------------------------------------------------------
        colcon build

    
Step3:- Source the package in each terminal
------------------------------------------------------------------------------------------
        source install/setup.bash
      

Step4:- Export the turtlebot3 model in each terminal
------------------------------------------------------------------------------------------
        export TURTLEBOT3_MODEL=waffle_pi

Step5:- For Hardware test, In One terminal run the turtlebot3 bringup after successfully connecting to turtlebot3 using ssh
------------------------------------------------------------------------------------------
       ros2 launch turtlebot3_bringup robot.launch.py

For Simulaton Test, Run below command in one terminal

        ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

Step6:- In second terminal run the gesture node
------------------------------------------------------------------------------------------
        ros2 run ros2_gesture_control gesture_detector

Note:- Ensure to have same Domain ID of your turtlebot3 and you PC for the connection to be established. 
------------------------------------------------------------------------------------------
