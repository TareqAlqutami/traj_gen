#!/usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import  Vector3Stamped, PoseStamped, TwistStamped, AccelStamped, Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from std_msgs.msg import Float32

from traj_gen.traj_gen.trajectory_generators import HelixTrajectory, TrapezTrajectory, AttCircleTrajectory, YawTrajectory

class AnaTrajectoryGenerator(Node):
    """
    generate desired trajectory to be executed by the robot
    """
    def __init__(self):
        super().__init__('trajectory_generator')
        
        # setpoints publishers
        self.att_sp_rpy_pub = self.create_publisher(Vector3Stamped,'/target_att_rpy', 10) # att in rpy degrees
        self.pose_sp_pub = self.create_publisher(PoseStamped,'/target_pose', 10)  # pose (xyz and quaternion)     
        self.vel_sp_pub = self.create_publisher(TwistStamped,'/target_twist', 10) # linear and angular velocity
        self.acc_sp_pub = self.create_publisher(AccelStamped,'/target_accel', 10) # linear and angular acceleration
        self.jerk_sp_pub = self.create_publisher(Vector3Stamped,'/target_jerk', 10) # linear jerk
        self.snap_sp_pub = self.create_publisher(Vector3Stamped,'/target_snap', 10) # linear snap
        self.force_sp_pub = self.create_publisher(Float32,'/target_force', 10)
        # self.wrench_sp_pub = self.create_publisher(WrenchStamped,'/target_wrench', 10)

        # state subscribers (used by some traj generators such as yaw follower)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,    depth=1)
        self.att_rpy_sub = self.create_subscription(Vector3Stamped,'/att_rpy', self.rpy_callback, qos_profile)
        self.position_sub = self.create_subscription(Vector3Stamped,'/position', self.position_callback, qos_profile)
        self.cur_position = np.array([0.0,0.0,0.0])
        self.cur_rpy = np.array([0.0,0.0,0.0])
        
        # trajectory visualization in rviz 
        self.waypoints_pub = self.create_publisher(Path, "/traj_gen/waypoints", 10)
        self.full_path_pub = self.create_publisher(Path, "/traj_gen/full_path", 10)
        self.timed_path_pub = self.create_publisher(Path, "/traj_gen/path", 10)
        self.vel_arrows_pub = self.create_publisher(Marker, "/traj_gen/vel_arrows", 10)

        self.xyz_waypoints = []
        self.rpy_waypoints = []
        self.waypoints_t = []
        # for rviz visualization
        self.position_path_msg = Path() # plot trajectory
        self.position_path_msg.header.frame_id = "map"
        self.trail_size = 1000 # maximum history to keep
        self.vel_heads = []
        self.vel_tails = []


        # ================ choose which trajectory to use ========================
        # TODO use rqt_reconfigure to choose trajectory and ensure smooth transitions
        self.traj_type = 'pose_helix'
        # ========================================================================
        
        if self.traj_type == 'pose_helix':
            self.traj_gen = HelixTrajectory(radius=1.0, omega=0.5, dz=0.0005,
                                            position_0=np.array([0.0,0.0,1.3]), yaw_0=0.0,
                                            yaw_mode='follow', z_max=4.0)
        elif self.traj_type == 'trapez':
            self.trapez_gen = TrapezTrajectory(A_xyz=np.array([0.5, 0.5,0.3]), offset_xyz=np.array([0.0,0.0,1.5]),
                                             width_xyz=10*np.ones(3), slope_xyz=np.array([0.5,0.5,0.2]),
                                             A_rpy=np.array([0.0, 0.0,np.pi/8]), offset_rpy=np.array([0.0,0.0,0.0]),
                                             width_rpy=10*np.ones(3), slope_rpy=0.3*np.ones(3))

        elif self.traj_type =='heading':
            yaw_waypoints = np.array([0, np.pi/6, np.pi/3, np.pi/2])
            t_waypoints = np.array([0, 4, 6, 10])
            # yaw_type: 'zeros', 'timed', 'interp', 'follow'
            self.yaw_gen = YawTrajectory(yaw_waypoints=yaw_waypoints,t_waypoints=t_waypoints, yaw_type='interp',current_yaw=0.0)
            
        elif self.traj_type =='att_circle':
            self.att_gen = AttCircleTrajectory(omega=0.5,max_roll=np.deg2rad(10), max_pitch=np.deg2rad(10), max_yaw=np.deg2rad(30),
                                               yaw_0=0.0)

        else:
            raise ValueError(f"Trajectory type ({self.traj_type}) is not supported!")


        update_freq = 50.0  # hz
        self.vis_update = 2.0/update_freq # slower
        self.start_time = self.get_clock().now().nanoseconds
        self.prev_time = 0.0
        self.vis_prev_time = 0.0
        self.update_callback_timer = self.create_timer(1.0/update_freq,self.update_callback)

        # publish waypoints only once (if available)
        waypoints_msgs = Path() # plot trajectory
        waypoints_msgs.header.frame_id = "map"
        for position, att_rpy in zip(self.xyz_waypoints, self.rpy_waypoints):
            q_tmp = Rotation.from_euler(
                'XYZ',[att_rpy[0], att_rpy[1], att_rpy[2]],degrees=True).as_quat()
            q = np.zeros(4)
            q[0],q[1], q[2], q[3] = q_tmp[3], q_tmp[0], q_tmp[1], q_tmp[2]
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]
            pose_msg.pose.orientation.w = q[0]
            pose_msg.pose.orientation.x = q[1]
            pose_msg.pose.orientation.y = q[2]
            pose_msg.pose.orientation.z = q[3]        
            waypoints_msgs.poses.append(pose_msg)
            self.waypoints_pub.publish(waypoints_msgs)



    def update_callback(self) -> None:
        """Callback function for the timer that runs the trajectory update at desired rate"""   
        t = (self.get_clock().now().nanoseconds - self.start_time) /1000_000_000.0 # time since the beginning in seconds
        dt = (t - self.prev_time) # time since last update
        dt_vz = t- self.vis_prev_time
        self.prev_time = t
        
        # holders
        position,vel,att_q, att_rpy, = None, None, None, None
        if self.traj_type =='pose_helix':
             # time to update trajectory
            position, yaw, vel, acc, jerk, snap  = self.traj_gen.eval(t)
            att_rpy = np.array([0.0, 0.0, yaw])
            self.get_logger().info(f"publishing commands: att={att_rpy.round(3)}, position={position.round(3)}", throttle_duration_sec=0.5)
            pose_msg = self.pub_pose_cmd(position, att_rpy)
            rpy_msg = self.pub_rpy_cmd(att_rpy)
            self.pub_linear_twist_cmd(vel)
            self.pub_linear_accel_cmd(acc)
            self.pub_linear_jerk_cmd(jerk)
            self.pub_linear_snap_cmd(snap)
        elif self.traj_type =='trapez':
             # time to update trajectory
            position, att_rpy = self.trapez_gen.eval(t)
            self.get_logger().info(f"publishing commands: att={att_rpy.round(3)}, position={position.round(3)}", throttle_duration_sec=0.5)
            pose_msg = self.pub_pose_cmd(position, att_rpy)
            rpy_msg = self.pub_rpy_cmd(att_rpy)
        elif self.traj_type =='heading':
            yaw, yaw_rate = self.yaw_gen.eval(t,des_pos=[0,0,0], curr_pos=[1,1,0])
            self.get_logger().info(f"publishing commands: yaw={yaw}, yaw_rate={yaw_rate}", throttle_duration_sec=0.5)
            position = self.cur_position
            att_rpy = np.array([0.0, 0.0, yaw])
            pose_msg = self.pub_pose_cmd(position, att_rpy)
            rpy_msg = self.pub_rpy_cmd(att_rpy)
        elif self.traj_type =='att_circle':
            att_rpy, rpy_rate = self.att_gen.eval(t)
            self.get_logger().info(f"publishing commands: rpy={att_rpy}, rpy_rate={rpy_rate}", throttle_duration_sec=0.5)
            position = self.cur_position
            self.pub_pose_cmd(position, att_rpy)
            rpy_msg = self.pub_rpy_cmd(att_rpy)
            rpy_msg = self.pub_full_twist_cmd(np.zeros(3),rpy_rate)
        else:
            raise ValueError(f"Trajectory type ({self.traj_type}) is not supported!")

        # ====== Publish trajectory visualization  ==========
        if position is not None and att_rpy is not None and  dt_vz>self.vis_update:
            self.vis_prev_time = t
            self.pub_traj_path(position,att_rpy)
            # Publish arrow markers for velocity if available
            if vel is not None:
                self.pub_vel_arrows(1,position,vel,dt)

    def rpy_callback(self,msg: Vector3Stamped) -> None:
        self.cur_rpy = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def position_callback(self,msg: Vector3Stamped) -> None:
        self.cur_position = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def pub_rpy_cmd(self,sig:np.array) -> Vector3Stamped:
        # publish trajectory attitude in rpy representation in degrees
        # timestamp = int(self.get_clock().now().nanoseconds / 1000)
        rpy_msg = Vector3Stamped()
        rpy_msg.header.frame_id = 'map'
        rpy_msg.vector.x = np.rad2deg(sig[0])
        rpy_msg.vector.y = np.rad2deg(sig[1])
        rpy_msg.vector.z = np.rad2deg(sig[2])
        self.att_sp_rpy_pub.publish(rpy_msg)
        return rpy_msg
    
    def pub_linear_twist_cmd(self, linear_vel:np.array) ->TwistStamped:
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = 'map'
        vel_msg.twist.linear.x = linear_vel[0]
        vel_msg.twist.linear.y = linear_vel[1]
        vel_msg.twist.linear.z = linear_vel[2]
        self.vel_sp_pub.publish(vel_msg)
        return vel_msg

    def pub_full_twist_cmd(self, linear_vel:np.array, angular_vel:np.array) ->TwistStamped:
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = 'map'
        vel_msg.twist.linear.x = linear_vel[0]
        vel_msg.twist.linear.y = linear_vel[1]
        vel_msg.twist.linear.z = linear_vel[2]
        vel_msg.twist.angular.x = angular_vel[0]
        vel_msg.twist.angular.y = angular_vel[1]
        vel_msg.twist.angular.z = angular_vel[2]

        self.vel_sp_pub.publish(vel_msg)
        return vel_msg
 
    def pub_linear_accel_cmd(self, linear_acc:np.array) ->AccelStamped:
        acc_msg = AccelStamped()
        acc_msg.header.frame_id = 'map'
        acc_msg.accel.linear.x = linear_acc[0]
        acc_msg.accel.linear.y = linear_acc[1]
        acc_msg.accel.linear.z = linear_acc[2]
        self.acc_sp_pub.publish(acc_msg)
        return acc_msg
 
    def pub_full_accel_cmd(self, linear_acc:np.array, angular_acc:np.array) ->AccelStamped:
        acc_msg = AccelStamped()
        acc_msg.header.frame_id = 'map'
        acc_msg.accel.linear.x = linear_acc[0]
        acc_msg.accel.linear.y = linear_acc[1]
        acc_msg.accel.linear.z = linear_acc[2]
        acc_msg.accel.angular.x = angular_acc[0]
        acc_msg.accel.angular.y = angular_acc[1]
        acc_msg.accel.angular.z = angular_acc[2]

        self.acc_sp_pub.publish(acc_msg)
        return acc_msg   

    def pub_linear_jerk_cmd(self, linear_jerk:np.array) ->Vector3Stamped:
        msg = Vector3Stamped()
        msg.header.frame_id = 'map'
        msg.vector.x = linear_jerk[0]
        msg.vector.y = linear_jerk[1]
        msg.vector.z = linear_jerk[2]
        self.jerk_sp_pub.publish(msg)
        return msg
 
    def pub_linear_snap_cmd(self, linear_snap:np.array) ->Vector3Stamped:
        msg = Vector3Stamped()
        msg.header.frame_id = 'map'
        msg.vector.x = linear_snap[0]
        msg.vector.y = linear_snap[1]
        msg.vector.z = linear_snap[2]
        self.snap_sp_pub.publish(msg)
        return msg
     
    def pub_pose_cmd(self,position:np.array,rpy:np.array) -> PoseStamped:
        q_tmp = Rotation.from_euler(
            'XYZ',[rpy[0], rpy[1], rpy[2]]).as_quat()
        q = np.zeros(4)
        q[0] = q_tmp[3]
        q[1] = q_tmp[0]
        q[2] = q_tmp[1]
        q[3] = q_tmp[2]
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.w = q[0]
        msg.pose.orientation.x = q[1]
        msg.pose.orientation.y = q[2]
        msg.pose.orientation.z = q[3]
        self.pose_sp_pub.publish(msg)
        return msg
        
    def pub_force_cmd(self, force:float) ->None:
        msg = Float32()
        msg.data = force
        self.force_sp_pub.publish(msg)    
        return msg

    def pub_traj_path(self, position, att_rpy):
        q_tmp = Rotation.from_euler(
            'XYZ',[att_rpy[0], att_rpy[1], att_rpy[2]]).as_quat()
        q = np.zeros(4)
        q[0],q[1],q[2],q[3]  = q_tmp[3], q_tmp[0], q_tmp[1], q_tmp[2]
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.w = q[0]
        pose_msg.pose.orientation.x = q[1]
        pose_msg.pose.orientation.y = q[2]
        pose_msg.pose.orientation.z = q[3]
        self.position_path_msg.poses.append(pose_msg)
        if len(self.position_path_msg.poses) > self.trail_size:
            del self.position_path_msg.poses[0]  
        self.timed_path_pub.publish(self.position_path_msg)

    def pub_vel_arrows(self,id,position,vel,dt):
        # append position to the tails array and limit array size
        tail_point = Point()
        tail_point.x = position[0]
        tail_point.y = position[1]
        tail_point.z = position[2]
        self.vel_tails.append(tail_point)
        if len(self.vel_tails) > self.trail_size:
            del self.vel_tails[0]  

        # append position to the heads array and limit array size
        head = position+vel*dt
        head_point = Point()
        head_point.x = head[0]
        head_point.y = head[1]
        head_point.z = head[2]
        self.vel_heads.append(head_point)
        if len(self.vel_heads) > self.trail_size:
            del self.vel_heads[0]  
        
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        msg.ns = "arrows"
        msg.id = id
        msg.type = Marker.LINE_LIST
        msg.scale.x = 0.2
        msg.scale.y = 0.2
        msg.scale.z = 0.0
        msg.color.r = 0.8
        msg.color.g = 0.2
        msg.color.b = 0.4
        msg.color.a = 0.9
        for tail,head in zip(self.vel_tails,self.vel_heads):
            msg.points.append(tail)
            msg.points.append(head)
        self.vel_arrows_pub.publish(msg)
  
    def create_points_marker(self, id, points, color=[0.5, 0.5, 0.0, 1.0], scale=[0.4,0.4,0.0]):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        msg.ns = "points"
        msg.id = id
        msg.type = Marker.SPHERE_LIST
        msg.scale.x = scale[0]
        msg.scale.y = scale[1]
        # msg.scale.z = scale[2]
        msg.color.r = color[0]
        msg.color.g = color[1]
        msg.color.b = color[2]
        msg.color.a = color[3]
        for point in points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            msg.points.append(p)
        return msg

        

def main(args=None):
    """Main function to execute"""
    rclpy.init(args=args)

    sig_generator = AnaTrajectoryGenerator()

    rclpy.spin(sig_generator)
    sig_generator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
