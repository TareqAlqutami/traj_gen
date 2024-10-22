# Trajectory generators for UAVs 
This is a ROS2 package that provide multiple trajectory generators such as minimum snap (or other derivatives) trajectory using waypoints provided.
There are multiple implementations using linear algebra (matrix inverse) and using quadratic programming.

# Dependencies:
- This was tested using ROS2 foxy in Ubuntu 20.04 
- Python version: Python 3.8.10
- Casadi (python) for QP implementation

# How to run:
There are two groups of trajectory generators, each has different way to configure and run as follows.
## **Pre-defined analytical trajectory generator:**
This includes pre-defined trajectories such as Helix, sinusoidal, and trapezoidal trajectories. To use this type:
- Modify `ana_traj_generator_node.py` to select the desired trajectory by specifying the `traj_type` in the class `__init__` and trajectory parameters.
- If the yaw_type has been chosen as `follow` then current position and yaw are required which are updated from topics `/position` and `/att_rpy`. Ensure the data is being published to these topics.
- Run the node
    ```bash
        ros2 run traj_gen ana_traj_generator
    ```
- Subscribe to the published trajectory topics and execute it in your robot
- Load the provided rviz/plotjuggler configurations for visualization

## **Trajectory optimizer and generator:**
This group transforms a list of waypoints and their respective time array to a continuous trajectory that satisfies certain constraints. For instance, a position trajectory that minimizes fourth order (Snap) and ensures all derivatives up to snap are continuous between the waypoints. This is typically achieved using piece-wise polynomial functions where each segment of the path (between two waypoints) is represented by a different polynomial function. Multiple implementation are available that either uses linear algebra and matrix inverse or quadratic/nonlinear programming to find the coefficients of the polynomial functions.
- Specify the waypoints in `waypoints.py`
- Modify the `min_snap_traj_generator_node.py` to import and use your waypoints and specify the `traj_type` and `yaw_type` in the class `__init__`.
- Run the node
    ```bash
        ros2 run traj_gen min_snap_traj_generator
    ```
- Subscribe to the published trajectory topics and execute it in your robot
- Load the provided rviz/plotjuggler configurations for visualization

# Samples
TODO

# Consuming the generated trajectory
The generator trajectory publishes multiple topics that can be subscribed to to visualize and execute in your robot. The trajectory generator publishes the following topics:
- `/target_pose`(PoseStamped): desired pose. This will also include desired yaw if it was specified.
- `/target_twist`(TwistStamped): desired linear velocity. Desired yaw rate is also provided in some implementations.
- `/target_accel`(AccelStamped): desired linear acceleration. Desired yaw angular acceleration is also provided in some implementations.
- `/target_jerk`(Vector3Stamped):  desired linear jerk.
- `/target_snap`(Vector3Stamped):  desired linear snap.
- `/traj_gen/waypoints`(Path): waypoints path for rviz visualization (published once in the beginning)
- `/traj_gen/path`(Path): trajectory path for rviz visualization


# Implementation details
TODO