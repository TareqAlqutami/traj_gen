# -*- coding: utf-8 -*-
# optimizeTrajectory was derived from traj_gen library https://github.com/icsl-Jeon/traj_gen


import time
from typing import List, Tuple
import numpy as np
from numpy import pi
from .linear_type import generate_time_from_vel, min_deriv_poly, min_snap_poly, polynom
from .optimize_type import PolyTrajGen


 
class YawMinAccTrajectory:

    def __init__(self, yaw_waypoints:np.array, t_waypoints:np.array):
        """Generate a minimum acceleration yaw trajectory given time and waypoint values. 
        Args:
            yaw_waypoints (np.array): the waypoint values with Nx1 dimension where N is the number of waypoints.
            t_waypoints (np.array): Corresponding time to arrive at each waypoint, dimension should be Nx1

        Raises:
            Exception: when length of waypoint array and time array are not the same or when time is in increasing order
        """

        self.deriv_order = 2 # acceleration
        self.nb_coeff = self.deriv_order*2
        self.wps   = np.copy(yaw_waypoints)
        self.t_wps   = np.copy(t_waypoints)
        self.t_segment = np.diff(self.t_wps)
        self.t_idx = 0  # index of current segment
        
        # check dimensions and values are good
        if len(self.t_wps) != self.wps.shape[0]:
            raise Exception("Time array and waypoint array not the same size.")
        elif (np.diff(self.t_wps) <= 0).any():
            raise Exception("Time array isn't properly ordered.")  


        self.coeffs = min_deriv_poly(self.wps, self.t_segment, self.deriv_order)

        # initialize the values to zeros
        self.des_yaw = 0.0    # Desired yaw
        self.des_yaw_rate = 0.0    # Desired yaw rate
        self.des_yaw_acc = 0.0    # Desired yaw angular acceleration

        self.prev_time = 0.0
        self.first = True

    def  eval(self, t:float, des_pos:np.array=np.zeros(3), curr_pos:np.array=np.zeros(3))->Tuple[float, float, float]:
        """Return the trajectory values at time t. This function returns a tuple containing yaw, yaw rate, yaw acceleration values

        Args:
            t (float): time in seconds
            des_pos: no used for now, to have the same input as YawTrajectory class
            curr_pos: no used for now, to have the same input as YawTrajectory class

        Returns:
            tuple[float, float, float]: yaw, yaw rate, yaw acceleration
        """
        if self.first:
            self.prev_time = t
        dt = t - self.prev_time
        
        # initialize the values to zeros
        self.des_yaw = 0.0    # Desired yaw
        self.des_yaw_rate = 0.0    # Desired yaw rate
        self.des_yaw_acc = 0.0    # Desired yaw angular acceleration

        # use first waypoint at the beginning
        if t == 0:
            self.t_idx = 0
            self.des_yaw = self.wps[0]
        # Stay hover at the last waypoint position
        elif t >= self.t_wps[-1]:
            self.t_idx = -1
            self.des_yaw = self.wps[-1]
        else:
            # find which time segment we are at
            self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
            # Scaled time (between 0 and duration of segment)
            scale = (t - self.t_wps[self.t_idx])
            
            # ==== Calculate yaw, yaw rate, angular acceleration at time t ====
            # Which coefficients to use
            start = self.nb_coeff * self.t_idx
            end = self.nb_coeff * (self.t_idx + 1)
            
            # desired yaw
            t0 = polynom(self.nb_coeff, 0, scale)
            self.des_yaw = self.coeffs[start:end].dot(t0)
            # desired yaw rate
            t1 = polynom(self.nb_coeff, 1, scale)
            self.des_yaw_rate = self.coeffs[start:end].dot(t1)
            # desired yaw angular acceleration
            t2 = polynom(self.nb_coeff, 2, scale)
            self.des_yaw_acc = self.coeffs[start:end].dot(t2)
            #================

        return self.des_yaw, self.des_yaw_rate, self.des_yaw_acc


class xyzMinDerivTrajectory:

    def __init__(self, xyz_waypoints:np.array, t_waypoints:np.array, traj_type:int):
        """Generate an xyz trajectory given time and waypoint values. 
        The trajectory can use the waypoints as is, interpolate between them, or minimize high order derivatives such as minimum snap. 

        Args:
            xyz_waypoints (np.array): the waypoint values with Nx3 dimension where N is the number of waypoints and 3 represents x,y,z values
            t_waypoints (np.array): Corresponding time to arrive at each waypoint, dimension should be Nx1
            traj_type (int): trajectory type to generate. one of the following options:
                    -1:  generate the waypoints provided and their respective times. No interpolation between two waypoints,
                    0: Interpolate position between every waypoint, to arrive at desired position every t_waypoints[i]. i.e. linear interpolation,    
                    1: minimum velocity trajectory,
                    2: minimum accel trajectory,
                    3: minimum jerk trajectory,
                    4: minimum snap trajectory

        Raises:
            ValueError: when traj_type is not supported
            Exception: when length of waypoint array and time array are not the same or when time is in increasing order
        """

        self.type = traj_type
        self.wps   = np.copy(xyz_waypoints)
        self.t_wps   = np.copy(t_waypoints)
        self.T_segment = np.diff(self.t_wps)
        self.t_idx = 0  # index of current segment
        
        # check dimensions and values are good
        if len(self.t_wps) != self.wps.shape[0]:
            raise Exception("Time array and waypoint array not the same size.")
        elif (np.diff(self.t_wps) <= 0).any():
            raise Exception("Time array isn't properly ordered.")  
        
        
        if self.type>4:
            raise ValueError("Maximum value allowed is 4 (i.e. minimum snap trajectory)")
        
        # minimize derivatives if type>=1
        if self.type >= 1:
            self.deriv_order = self.type       # Looking to minimize which derivative order (eg: Minimum velocity -> first order)
            # Calculate coefficients
            self.coeff_x = min_deriv_poly(self.wps[:,0], self.T_segment, self.deriv_order)
            self.coeff_y = min_deriv_poly(self.wps[:,1], self.T_segment, self.deriv_order)
            self.coeff_z = min_deriv_poly(self.wps[:,2], self.T_segment, self.deriv_order)

        # Initialize trajectory setpoint
        self.des_pos = np.zeros(3)    # Desired position (x, y, z)
        self.des_vel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        self.des_acc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.des_jerk = np.zeros(3)    # Desired jerk (xdotdotdot, ydotdotdot, zdotdotdot)
        self.des_snap = np.zeros(3)    # Desired snap (xdotdotdotdot, ydotdotdotdot, zdotdotdotdot)

        self.prev_time = 0.0
        self.first = True

    def pos_waypoint_min(self, t:float):
        """ The function evaluates  piece-wise polynomial functions at time t given their coefficients. It uses the polynomial times to decide which one to use then
        evaluates position and its derivatives at that given time t.
        """
            
        nb_coeff = self.deriv_order*2

        # Scaled time (between 0 and duration of segment)
        scale = (t - self.t_wps[self.t_idx])
        
        # Which coefficients to use
        start = nb_coeff * self.t_idx
        end = nb_coeff * (self.t_idx + 1)
        
        # Set desired position, velocity and acceleration
        t0 = polynom(nb_coeff, 0, scale)
        self.des_pos = np.array([self.coeff_x[start:end].dot(t0), self.coeff_y[start:end].dot(t0), self.coeff_z[start:end].dot(t0)])

        t1 = polynom(nb_coeff, 1, scale)
        self.des_vel = np.array([self.coeff_x[start:end].dot(t1), self.coeff_y[start:end].dot(t1), self.coeff_z[start:end].dot(t1)])

        t2 = polynom(nb_coeff, 2, scale)
        self.des_acc = np.array([self.coeff_x[start:end].dot(t2), self.coeff_y[start:end].dot(t2), self.coeff_z[start:end].dot(t2)])

        t3 = polynom(nb_coeff, 3, scale)
        self.des_jerk = np.array([self.coeff_x[start:end].dot(t3), self.coeff_y[start:end].dot(t3), self.coeff_z[start:end].dot(t3)])  
        
        t4 = polynom(nb_coeff, 4, scale)
        self.des_snap = np.array([self.coeff_x[start:end].dot(t4), self.coeff_y[start:end].dot(t4), self.coeff_z[start:end].dot(t4)])  
    
    def  eval(self, t:float)->Tuple[np.array, float, np.array, np.array, np.array, np.array]:
        """Return the trajectory values at time t. This function returns a tuple containing xyz position and its derivatives up to snap and yaw value.

        Args:
            t (float): time in seconds

        Returns:
            tuple[np.array, float, np.array, np.array, np.array, np.array]: xyz-position, yaw, xyz-velocity, xyz-acceleration, xyz-jerk, xyz-snap
        """
        if self.first:
            self.prev_time = t
        dt = t - self.prev_time  
        
        # initialize the values to zeros
        self.des_pos = np.zeros(3)    # Desired position (x, y, z)
        self.des_vel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        self.des_acc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.des_jerk = np.zeros(3)    # Desired jerk (xdotdotdot, ydotdotdot, zdotdotdot)
        self.des_snap = np.zeros(3)    # Desired snap (xdotdotdotdot, ydotdotdotdot, zdotdotdotdot)

        # use first waypoint at the beginning
        if t == 0:
            self.t_idx = 0
            self.des_pos = self.wps[0,:]
        # Stay hover at the last waypoint position
        elif (t >= self.t_wps[-1]):
            self.t_idx = -1
            self.des_pos = self.wps[-1,:]
        else:
            # find which time segment we are at
            self.t_idx = np.where(t <= self.t_wps)[0][0] - 1                     
        
            # Set desired positions at every t_wps[i], no interpolation    
            if (self.type == -1):
                self.des_pos = self.wps[self.t_idx,:]     
            # Interpolate position between every waypoint, to arrive at desired position every t_wps[i]. i.e. linear interpolation
            elif (self.type == 0):            
                scale = (t - self.t_wps[self.t_idx])/self.T_segment[self.t_idx]
                self.des_pos = (1 - scale) * self.wps[self.t_idx,:] + scale * self.wps[self.t_idx + 1,:]   

            # Calculate a minimum velocity, acceleration, jerk or snap trajectory
            elif (self.type >= 1 and self.type <= 4):
                self.pos_waypoint_min(t)
            
        
        return self.des_pos[:], self.des_vel[:], self.des_acc[:], self.des_jerk[:], self.des_snap[:]


class xyzMinSnapTrajectory:

    def __init__(self, xyz_waypoints:np.array, t_waypoints:np.array, vel:float=None, calc_times:bool=False, method='lstsq'):
        """Generate an xyz trajectory given time and waypoint values. 
        The trajectory can use the waypoints as is, interpolate between them, or minimize high order derivatives such as minimum snap. 

        Args:
            xyz_waypoints (np.array): the waypoint values with Nx3 dimension where N is the number of waypoints and 3 represents x,y,z values
            t_waypoints (np.array): Corresponding time to arrive at each waypoint, dimension should be Nx1
            vel (float): average velocity 

        Raises:
            Exception: when length of waypoint array and time array are not the same or when time is in increasing order
        """

        self.wps   = np.copy(xyz_waypoints)
        if calc_times and vel is not None:
             self.T_segment = generate_time_from_vel(xyz_waypoints,vel)
             self.t_wps = np.zeros(self.wps.shape[0])
             self.t_wps[1:]   = np.cumsum(self.T_segment)
        else:
            self.t_wps   = np.copy(t_waypoints)
            self.T_segment = np.diff(self.t_wps)
        
        self.t_idx = 0  # index of current segment
        self.method = method

        # check dimensions and values are good
        if len(self.t_wps) != self.wps.shape[0]:
            raise Exception("Time array and waypoint array not the same size.")
        elif (np.diff(self.t_wps) <= 0).any():
            raise Exception("Time array isn't properly ordered.")  
                
        # Calculate coefficients
        print(f"T_segments:{self.T_segment}")
        self.coeffs = min_snap_poly(self.wps, self.T_segment, method=self.method)
        # Initialize trajectory setpoint
        self.des_pos = np.zeros(3)    # Desired position (x, y, z)
        self.des_vel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        self.des_acc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.des_jerk = np.zeros(3)    # Desired jerk (xdotdotdot, ydotdotdot, zdotdotdot)
        self.des_snap = np.zeros(3)    # Desired snap (xdotdotdotdot, ydotdotdotdot, zdotdotdotdot)

        self.prev_time = 0.0
        self.first = True

    def _gen_trajectory(self, t:float):
        """ The function evaluates  piece-wise polynomial functions at time t given their coefficients. It uses the polynomial times to decide which one to use then
        evaluates position and its derivatives at that given time t.
        """
        nb_coeff = 8

        # Scaled time (between 0 and duration of segment)
        scale = (t - self.t_wps[self.t_idx])
        
        # Which coefficients to use
        start = nb_coeff * self.t_idx
        end = nb_coeff * (self.t_idx + 1)
        
        # here we generate the trajectory for each spline from t=0 to t=timeT at a rate of dt (unit: s)
        self.des_pos  = polynom(nb_coeff, derivative=0, t=scale) @ self.coeffs[start:end]
        self.des_vel  = polynom(nb_coeff, derivative=1, t=scale) @ self.coeffs[start:end]
        self.des_acc  = polynom(nb_coeff, derivative=2, t=scale) @ self.coeffs[start:end]
        self.des_jerk = polynom(nb_coeff, derivative=3, t=scale) @ self.coeffs[start:end]
        self.des_snap = polynom(nb_coeff, derivative=4, t=scale) @ self.coeffs[start:end]
                   
    def  eval(self, t:float)->Tuple[np.array, float, np.array, np.array, np.array, np.array]:
        """Return the trajectory values at time t. This function returns a tuple containing xyz position and its derivatives up to snap.

        Args:
            t (float): time in seconds

        Returns:
            tuple[np.array, np.array, np.array, np.array, np.array]: xyz-position, xyz-velocity, xyz-acceleration, xyz-jerk, xyz-snap
        """
        if self.first:
            self.prev_time = t
        dt = t - self.prev_time  
        
        # initialize the values to zeros
        self.des_pos = np.zeros(3)    # Desired position (x, y, z)
        self.des_vel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        self.des_acc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.des_jerk = np.zeros(3)    # Desired jerk (xdotdotdot, ydotdotdot, zdotdotdot)
        self.des_snap = np.zeros(3)    # Desired snap (xdotdotdotdot, ydotdotdotdot, zdotdotdotdot)

        # use first waypoint at the beginning
        if t == 0:
            self.t_idx = 0
            self.des_pos = self.wps[0,:]
        # Stay hover at the last waypoint position
        elif (t >= self.t_wps[-1]):
            self.t_idx = -1
            self.des_pos = self.wps[-1,:]
        else:
            # find which time segment we are at
            self.t_idx = np.where(t <= self.t_wps)[0][0] - 1                     
        
            # Set desired positions at every t_wps[i], no interpolation    
            self._gen_trajectory(t)
            
        
        return self.des_pos[:], self.des_vel[:], self.des_acc[:], self.des_jerk[:], self.des_snap[:]


class optimizeTrajectory:
    def __init__(self, xyz_waypoints:np.array, t_waypoints:np.array, optim_target:str,poly_order:int=7, floating_cubes:np.array=None, t_cubes:np.array=None):
        """Generate an xyz trajectory given time and waypoint values. 
        The trajectory  minimize high order derivatives such as minimum snap using either polynomial equations of poly_order or formulate a QP optimization to minimize derivatives.

        Args:
            xyz_waypoints (np.array): the waypoint values with Nx3 dimension where N is the number of waypoints and 3 represents x,y,z values
            t_waypoints (np.array): Corresponding time to arrive at each waypoint, dimension should be Nx1
            optim_target (str): how trajectory is calculated.
                    'poly-coeff':  optimization target is polynomial coefficients as per Mellinger paper (minimum snap trajectory generation)
                    'end-derivative': optimization target is free end-derivatives of spline segments as per Richter paper (Polynomial trajectory planning for aggressive quadrotor flight in dense indoor environments)
            poly_order (int): order of polynomials when using 'poly-coeff', it should 7 for minimum snap trajectory
            floating_cubes (np.array): an array of 3x2 floating cubes that are used to guide the trajectory through them at certain times. (additional constraints). None means no floating cubes.
            t_cubes (np.array): an array containing times at which the floating cubes are passed through. (additional constraints). None means no floating cubes.
        Raises:
            Exception: when length of waypoint array and time array are not the same or when time is in increasing order
            
        """       
        self.dim = 3
        self.wps   = np.copy(xyz_waypoints)
        self.t_wps   = np.copy(t_waypoints)
        self.optim_target = optim_target #'end-derivative' 'poly-coeff'

        # check dimensions and values are good
        if len(self.t_wps) != self.wps.shape[0]:
            raise Exception("Time array and waypoint array not the same size.")
        elif (np.diff(self.t_wps) <= 0).any():
            raise Exception("Time array isn't properly ordered.")  
        
        max_continuous_deriv = 4
        objWeights = np.array([0, 0, 0, 1])    
        self.traj_gen = PolyTrajGen(self.t_wps, poly_order, self.optim_target, self.dim, max_continuous_deriv)

        Xdot = np.array([0, 0, 0])
        Xddot = np.array([0, 0, 0])

        # add waypoints
        for i in range(self.wps.shape[0]):
            # create pin dictionary
            pin_ = {'t':self.t_wps[i], 'd':0, 'X':self.wps[i]}
            self.traj_gen.addPin(pin_)
        
        # add velocity & acceleration constraints for first waypoint (vel=0, acc=0)
        pin_ = {'t':self.t_wps[0], 'd':1, 'X':Xdot}
        self.traj_gen.addPin(pin_)
        pin_ = {'t':self.t_wps[0], 'd':2, 'X':Xddot,}
        self.traj_gen.addPin(pin_)

        # add velocity & acceleration constraints for last waypoint (vel=0, acc=0)
        pin_ = {'t':self.t_wps[-1], 'd':1, 'X':Xdot}
        self.traj_gen.addPin(pin_)
        pin_ = {'t':self.t_wps[-1], 'd':2, 'X':Xddot,}
        self.traj_gen.addPin(pin_)
        
        # Add passthrough waypoints if provided. These are used to shape the trajectory and ensure it passes through them.
        if floating_cubes is not None and t_cubes is not None and len(floating_cubes)==len(t_cubes):
            for pass_cube, t_cube in zip(floating_cubes, t_cubes):
                # each cube has dimension of 3*2
                pin_ = {'t':t_cube, 'd':0, 'X':pass_cube}
        self.traj_gen.addPin(pin_)

        # solve
        self.traj_gen.setDerivativeObj(objWeights)
        print("solving trajectory optimization")
        time_start = time.time()
        self.traj_gen.solve()
        time_end = time.time()
        print(f"completed optimization in {time_end - time_start}")
    
    
    def  eval(self, t:float)->Tuple[np.array, float, np.array, np.array, np.array, np.array]:
        
        position = np.zeros(3)    # Desired position (x, y, z)
        vel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        acc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        jerk = np.zeros(3)    # Desired jerk (xdotdotdot, ydotdotdot, zdotdotdot)
        snap = np.zeros(3)    # Desired snap (xdotdotdotdot, ydotdotdotdot, zdotdotdotdot)
   
       # use first waypoint at the beginning
        if t == 0:
            t_idx = 0
            position = self.wps[t_idx,:]
        # Stay hover at the last waypoint position
        elif (t >= self.t_wps[-1]):
            t_idx = -1
            position = self.wps[t_idx,:]
        else:            
            position = self.traj_gen.eval(np.array([t]), 0).reshape(3)
            vel = self.traj_gen.eval(np.array([t]), 1).reshape(3)
            acc = self.traj_gen.eval(np.array([t]), 2).reshape(3)
            jerk = self.traj_gen.eval(np.array([t]), 3).reshape(3)
            snap = self.traj_gen.eval(np.array([t]), 4).reshape(3)
        return position, vel, acc, jerk, snap
    