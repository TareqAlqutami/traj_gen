# -*- coding: utf-8 -*-
# author: Tareq Alqutami
# email: tareqaziz2010@gmail.com
# license: MIT

from typing import  Tuple
import numpy as np
from numpy import pi
from scipy import signal

class HelixTrajectory:
    """Generate a helix trajectory and yaw waypoints given helix parameters and yaw_mode.
    """
    def __init__(self, radius:float, omega:float, dz:float, position_0: np.array, yaw_0:float, yaw_mode:str='fixed',z_max:float=None):
        """Generate a helix trajectory and yaw waypoints given helix parameters and yaw_mode.

        Args:
            radius (float): radius of the helix in meters
            omega (float): angular velocity in rad/s
            dz (float): displacement in z-axis over time. if set to zero, it becomes a circle. 
            position_0 (np.array): initial position x,y,z
            yaw_0 (float): initial yaw value in radians
            yaw_mode (str, optional): how is yaw waypoint calculated. either fixed value=yaw_0 or calculated to follow the velocity of trajectory. Defaults to 'fixed'.
            z_max (float, optional): maximum z value at which the helix will stop increments in z and continue doing circles. If none, no maximum z value. Defaults to None.
        """
        self.radius = radius
        self.omega = omega # control the angular speed of the circular path
        self.dz = dz # control pitch of the helix, it becomes a circle if set to zero
        self.position_0 = position_0
        self.yaw0 = yaw_0
        self.yaw_mode = yaw_mode
        self.z_max = z_max

        self.position = np.copy(self.position_0)
        self.vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])
        self.jerk = np.array([0.0, 0.0, 0.0])
        self.snap = np.array([0.0, 0.0, 0.0])
        self.prev_position = np.copy(self.position_0)
        self.yaw = self.yaw0
        self.current_yaw  = self.yaw0

    def  eval(self, t)->Tuple[np.array, float, np.array, np.array, np.array, np.array]:
        """Return the trajectory values at time t. This function returns a tuple containing xyz position and its derivatives up to snap and yaw value.

        Args:
            t (float): time in seconds

        Returns:
            tuple[np.array, float, np.array, np.array, np.array, np.array]: xyz-position, yaw, xyz-velocity, xyz-acceleration, xyz-jerk, xyz-snap
        """
        self.position[0] = self.radius * np.cos(self.omega*t) + self.position_0[0]  # x
        self.position[1] = self.radius * np.sin(self.omega*t) + self.position_0[1]   # y
        if self.z_max is not None and self.position[2]<self.z_max:
            # update z if it has not exceed maximum
            self.position[2] +=self.dz #TODO: should this be based on time, instead of increment in each update
        
        self.vel[0] = -self.radius*self.omega * np.sin(self.omega*t)  # dx/dt
        self.vel[1] =  self.radius*self.omega * np.cos(self.omega*t)  # dy/dt
        self.vel[2] = self.dz   # dz/dt

        self.acc[0] = -self.radius*(self.omega**2) * np.cos(self.omega*t)  # ddx/dt
        self.acc[1] = -self.radius*(self.omega**2) * np.sin(self.omega*t)  # ddy/dt
        self.acc[2] = 0   # ddz/dt

        self.jerk[0] =  self.radius*(self.omega**3) * np.sin(self.omega*t)  # dddx/dt
        self.jerk[1] = -self.radius*(self.omega**3) * np.cos(self.omega*t)  # dddy/dt
        self.jerk[2] = 0   # dddz/dt

        self.snap[0] =  self.radius*(self.omega**4) * np.cos(self.omega*t)  # ddddx/dt
        self.snap[1] =  self.radius*(self.omega**4) * np.sin(self.omega*t)  # ddddy/dt
        self.snap[2] = 0   # ddddz/dt
        

        if self.yaw_mode == 'follow':
            if (t > 0):
                # Calculate desired Yaw
                self.yaw = np.arctan2(self.position[1]-self.prev_position[1], self.position[0]-self.prev_position[0])
                    
            # detect when yaw switches from -pi to pi (or vice-versa) and switch manually current_heading 
            if (np.sign(self.yaw) - np.sign(self.current_yaw) and abs(self.yaw-self.current_yaw) >= 2*pi-0.1):
                self.current_yaw = self.current_yaw + np.sign(self.yaw)*2*pi
            
            # Prepare for next iteration
            self.current_yaw = self.yaw
            
        self.prev_position[:] = self.position

        return self.position[:], self.yaw, self.vel[:], self.acc[:], self.jerk[:], self.snap[:]

class TrapezTrajectory:
    """Generate a helix trajectory and yaw waypoints given helix parameters and yaw_mode.
    """
    def __init__(self, A_xyz:np.array, A_rpy:np.array, offset_xyz:np.array, offset_rpy:np.array,width_xyz:float=10, width_rpy:float=10,slope_xyz:float=3, slope_rpy:float=3):
        """Generate a helix trajectory and yaw waypoints given helix parameters and yaw_mode.

        Args:
            A_xyz (np.array): magnitude of the position steps (A+offset <to> -A+offset)
            A_rpy (np.array): magnitude of the orientation steps (A+offset <to> -A+offset)
            offset_xyz (np.array): offsets applied to the xyz signal
            offset_rpy (np.array): offsets applied to the rpy signal
            width_xyz (np.array): width of the transit xyz signal
            width_rpy (np.array): width of the transit rpy signal
            width_xyz (np.array): slope of the transit xyz signal
            slope_rpy (np.array): slope of the transit rpy signal
        """
        self.A_xyz = A_xyz # magnitude of the square (A+offset <to> -A+offset)
        self.A_rpy = A_rpy # magnitude of the square for rpy angles in radians
        self.offset_xyz = offset_xyz # offsets applied to the signal
        self.offset_rpy = offset_rpy # offsets applied to the signal
        self.width_xyz = width_xyz # width of the transit signal
        self.width_rpy = width_rpy # width of the transit signal
        self.slope_xyz = slope_xyz # slope of the transit signal
        self.slope_rpy = slope_rpy # slope of the transit signal

    @staticmethod
    def trapezoid_signal(t:float, width=2., slope=1., amp=1., offs=0):
        """produce trapezoidal signal from sawtooth waveform

        Args:
            t (float): current time
            width (float, optional): . Defaults to 2..
            slope (float, optional): . Defaults to 1..
            amp (float, optional): . Defaults to 1..
            offs (float, optional): . Defaults to 0.

        Returns:
            float: current value
        """
        a = slope*width*signal.sawtooth(2*np.pi*t/width, width=0.5)/4.
        if a>amp/2.:
            a = amp/2.
        if a<-amp/2.:
            a = -amp/2.
        return a + amp/2. + offs

    def  eval(self, t)->Tuple[np.array, np.array]:
        """Return the trajectory values at time t. This function returns a tuple containing xyz position and attitude (rpy) values.

        Args:
            t (float): time in seconds

        Returns:
            tuple[np.array, np.array]: xyz-position, attitude rpy in degrees
        """
        
        roll = TrapezTrajectory.trapezoid_signal(t,  width=self.width_rpy[0], slope=self.slope_rpy[0], amp=self.A_rpy[0],offs=self.offset_rpy[0])
        pitch= TrapezTrajectory.trapezoid_signal(t,  width=self.width_rpy[1], slope=self.slope_rpy[1], amp=self.A_rpy[1],offs=self.offset_rpy[1])
        yaw  =  TrapezTrajectory.trapezoid_signal(t, width=self.width_rpy[2], slope=self.slope_rpy[2], amp=self.A_rpy[2],offs=self.offset_rpy[1])
        x    =  TrapezTrajectory.trapezoid_signal(t, width=self.width_xyz[0], slope=self.slope_xyz[0], amp=self.A_xyz[0],offs=self.offset_xyz[0])
        y    =  TrapezTrajectory.trapezoid_signal(t, width=self.width_xyz[1], slope=self.slope_xyz[1], amp=self.A_xyz[1],offs=self.offset_xyz[1])
        z    =  TrapezTrajectory.trapezoid_signal(t, width=self.width_xyz[2], slope=self.slope_xyz[2], amp=self.A_xyz[2],offs=self.offset_xyz[2])
        position = np.array([x, y, z])
        att = np.array([roll, pitch, yaw])

        return position, att


class AttCircleTrajectory:
    """Generate a sinusoidal signals for roll/pitch/yaw in radians and their first derivative.
    """
    def __init__(self, omega:float, max_roll:float, max_pitch:float,max_yaw:float, yaw_0:float):
        """Generate a sinusoidal signals for roll/pitch/yaw in radians and their first derivative

        Args:
            omega (float): angular velocity in rad/s
            max_roll (float): maximum magnitude of the sinusoidal wave for the roll angle. In radians.
            max_pitch (float): maximum magnitude of the sinusoidal wave for the pitch angle. In radians.
            max_yaw (float): maximum magnitude of the sinusoidal wave for the yaw angle. In radians.
            yaw_0 (float): initial yaw value in radians
        """
        self.omega = omega
        self.yaw0 = yaw_0
        self.max_roll = max_roll
        self.max_pitch = max_pitch
        self.max_yaw = max_yaw

        self.rpy = np.array([0.0, 0.0, 0.0])
        self.rpy_rates = np.array([0.0, 0.0, 0.0])
        self.yaw = self.yaw0
        self.current_yaw  = self.yaw0

    def  eval(self, t)->Tuple[np.array, np.array]:
        """Return the trajectory values at time t. This function returns a tuple containing attitude values in rpy radians and its first derivative.

        Args:
            t (float): time in seconds

        Returns:
            tuple[np.array, np.array]: att(rpy), rpy_rates
        """
        roll  = self.max_roll * np.sin(self.omega*t)
        pitch = self.max_pitch * np.sin(self.omega*t)
        yaw   = self.max_yaw * np.sin(self.omega*t) + self.yaw0
        roll_rate  = self.omega * self.max_roll * np.cos(self.omega*t)
        pitch_rate = self.omega *self.max_pitch * np.cos(self.omega*t)
        yaw_rate   = self.omega *self.max_yaw * np.cos(self.omega*t)
        self.rpy = np.array([roll, pitch, yaw])
        self.rpy_rates = np.array([roll_rate, pitch_rate, yaw_rate])

        # detect when yaw switches from -pi to pi (or vice-versa) and switch manually current_heading 
        if (np.sign(self.yaw) - np.sign(self.current_yaw) and abs(self.yaw-self.current_yaw) >= 2*pi-0.1):
            self.current_yaw = self.current_yaw + np.sign(self.yaw)*2*pi
        # Prepare for next iteration
        self.current_yaw = self.yaw

        return self.rpy[:], self.rpy_rates[:]

class YawTrajectory:

    def __init__(self, yaw_waypoints:np.array, t_waypoints:np.array,  yaw_type:str, current_yaw=0.0):
        """Generate yaw trajectory given yaw waypoints and their respective times.
        Multiple options are available to generate the trajectory. The trajectory returned at time t using the method .eval and 
        it returns the yaw value and yaw_rate.

        Args:
            yaw_waypoints (np.array): yaw waypoints Nx1 
            t_waypoints (np.array):  waypoints time Nx1 
            yaw_type (str): type of trajectory from the following:
                - 'zeros' : yaw is always zero
                - 'timed' : yaw follows the waypoints provided and their respective times. No interpolation between two waypoints.
                - 'interp': yaw follows the waypoints provided and their respective times. Value is linearly interpolated from one waypoint to the next.
                - 'follow': yaw follows drone velocity direction. the value is calculated based on current drone velocity provided (des_pos - curr_pos)*dt
            current_yaw (float, optional): initial yaw value. Defaults to 0.0.
        """

        self.yaw_type = yaw_type
        self.current_yaw = current_yaw #initial heading
        self.t_wps   = np.copy(t_waypoints)
        self.y_wps = np.copy(yaw_waypoints)
        self.t_segment = np.diff(self.t_wps)

        if (self.yaw_type == 'zeros'):
            self.y_wps = np.zeros(len(self.t_wps))

        if not (len(self.t_wps) == len(self.y_wps)):
            raise Exception("Time array and waypoint array not the same size.")

        # Initialize trajectory setpoint
        self.des_yaw_rate = 0.0         # Desired yaw rate
        self.des_yaw_acc = 0.0         # Desired yaw accelration
        self.prev_time = 0.0
        self.first = True
        self.t_idx = 0

    def yaw_follow(self, t:float, dt:float , des_pos:np.array, curr_pos:np.array):
        # yaw follows velocity direction.
        if (t == 0):
            self.des_yaw = 0
        else:
            # Calculate desired Yaw
            self.des_yaw = np.arctan2(des_pos[1]-curr_pos[1], des_pos[0]-curr_pos[0])
       
        # Dirty hack, detect when desEul[2] switches from -pi to pi (or vice-versa) and switch manually current_heading 
        if (np.sign(self.des_yaw) - np.sign(self.current_yaw) and abs(self.des_yaw-self.current_yaw) >= 2*pi-0.1):
            self.current_yaw = self.current_yaw + np.sign(self.des_yaw)*2*pi
        
        # Angle between current vector with the next heading vector
        delta_psi = self.des_yaw - self.current_yaw
        
        # Set Yaw rate
        if dt>0:
            self.des_yaw_rate = delta_psi / dt 

        # Prepare next iteration
        self.current_yaw = self.des_yaw

    def eval(self, t, des_pos:np.array=np.zeros(3), curr_pos:np.array=np.zeros(3)):
        # return the yaw and yaw rate at time t. cur_pos and des_pos are used when type is 'follow' only.
        if self.first:
            self.prev_time = t
        dt = t - self.prev_time
        self.des_yaw = 0.0   # Desired yaw in the world frame 
        self.des_yaw_rate = 0.0         # Desired yaw rate
        self.des_yaw_acc = 0.0         # Desired yaw accelration
 
        # find current time index
        if t == 0: # first waypoint
            self.t_idx = 0
        elif (t >= self.t_wps[-1]): # we reached the last waypoint, keep it
            self.t_idx = -1
        else:
            self.t_idx = np.where(t <= self.t_wps)[0][0] - 1

        # List of possible yaw trajectories
        # ---------------------------
        # Set desired yaw at every t_wps[i]
        if (self.yaw_type == 'timed'):
            self.des_yaw = self.y_wps[self.t_idx]
        # Interpolate yaw between every waypoint, to arrive at desired yaw every t_wps[i]
        elif (self.yaw_type == 'interp'):
            if (t == 0) or (t >= self.t_wps[-1]):
                self.des_yaw = self.y_wps[self.t_idx]
            else:
                scale = (t - self.t_wps[self.t_idx])/self.t_segment[self.t_idx]
                self.des_yaw = (1 - scale)*self.y_wps[self.t_idx] + scale*self.y_wps[self.t_idx + 1]
                
                # Angle between current vector with the next heading vector
                delta_psi = self.des_yaw - self.current_yaw
                # Set Yaw rate
                if dt>0:
                    self.des_yaw_rate = delta_psi / dt 
                # Prepare next iteration
                self.current_yaw = self.des_yaw
    
        # the drone's heading match its desired velocity direction
        elif (self.yaw_type == 'follow'):
            self.yaw_follow(t, dt , des_pos, curr_pos)
    
        return self.des_yaw, self.des_yaw_rate, self.des_yaw_acc

