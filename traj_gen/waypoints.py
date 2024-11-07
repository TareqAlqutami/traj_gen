import numpy as np
from numpy import pi

deg2rad = pi/180.0


# ====== format =========
# def func_name():
#   return t, xyz, rpy
# t(np.array): is waypoints times of size Nx1
# xyz(np.ndarray): position waypoints of size Nx3
# rpy(np.ndarray):  orientation waypoints (roll,pitch,yaw) of size Nx3. only yaw is consumed for under-actuated vehicles.

def flip(): 
    # t = np.array([0,4, 6, 8, 12])
    t = np.array([0,8, 10, 15, 20])
    xyz = np.array([
      [0, 0, 2],
      [6.0, 0.0, 6.0],
      [0.0, 0.0, 12.0],
      [-6.0, 0.0, 6.0],
      [0.0, 0.0, 2.0]
      ])
    rpy = np.array([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
      ])
    return t, xyz, rpy

def straight():
    xyz = np.array([
      [0.0, 0.0, 1.0],
      [0.1, 0.0, 1.5],
      [0.2, 0.0, 2.0],
      [0.3, 0.0, 2.5],
      [0.4, 0.0, 3.0],
      [0.5, 0.0, 3.5],
      [0.6, 0.0, 4.0],
      [0.7, 0.0, 4.5],
    ])
    t = np.arange(0,xyz.shape[0]*2,2)
    rpy = np.zeros_like(xyz)
    return t, xyz, rpy

def figure_8():
    xyz = np.array([
        [0.0, 0.0, 2.0],
        [2.0, 2.0, 2.5],
        [4.0, 0.0, 3.0],
        [2.0, -2.0, 2.5],
        [0.0, 0.0, 2.0],
        [-2.0, 2.0,1.5],
        [-4.0, 0.0,1.0],
        [-2.0, -2.0,1.5],
        [0.0, 0.0, 2.0]
    ])
    t = np.arange(0,xyz.shape[0]*2,2)
    rpy = np.zeros_like(xyz)
    return t, xyz, rpy


def long_waypoint():
    xyz = np.array([
      [0.0, 0.0, 0.0],
      [5.0, 2.0, 1.0],
      [8.0, 5.0, 2.0],
      [10.0, 10.0, 3.0],
      [8.0, 15.0, 4.0],
      [5.0, 18.0, 5.0],
      [0.0, 20.0, 6.0],
      [-5.0, 18.0, 7.0],
      [-8.0, 15.0, 8.0],
      [-10.0, 9.0, 9.0],
      [-9.0, 2.0, 10.0],
      [-8.0, -5.0, 11.0],
      [-7.0, -14.0, 11.0],
      [-6.0, -24.0, 10.0],
      [-5.0, -32.0, 9.0],
      [-4.0, -38.0, 8.0],
      [-3.0, -42.0, 7.0],
      [-2.0, -44.0, 6.0],
    ])
    t = np.arange(0,xyz.shape[0],1)
    rpy = np.zeros_like(xyz)
    return t, xyz, rpy

def att_test():
    t = np.array([0, 2, 4, 6, 8, 10])
    xyz = np.array([
      [0.0, 0.0, 1.5],
      [0.0, 0.0, 1.5],
      [0.0, 0.0, 1.5],
      [0.0, 0.0, 1.5],
      [0.0, 0.0, 1.5],
      [0.0, 0.0, 1.5]
      ])
    rpy = np.array([
      [0.00, 0.1, 0],
      [0.00, 0.25, pi/6],
      [0.00, 0.1, pi/6],
      [0.1, 0.00, pi/6],
      [0.2, 0.00, pi/6],
      [0.00, 0.00, pi/6]
      ])
    return t, xyz, rpy
  
def test_position():
    t = np.array([0,2, 4])
    t = np.array([0, 3.74165739, 7.48331478])
    xyz = np.array([
      [0, 0, 0],
      [1.0, 2.0, 3.0],
      [0.0, 0.0, 0.0]
      ])
    rpy = np.array([
      [0, 0, 0],
      [0, 0, pi/6],
      [0, 0, pi/2]
      ])
    return t, xyz, rpy

def raster_scan():
    xyz = np.array([
      [0.0, 1.0, 3.0],
      [0.1, 1.0, 3.0],
      [0.1, 0.5, 3.0],
      [0.1, 0.0, 3.0],

      [0.1, 0.0, 2.8], 
      [0.0, 0.5, 2.8],
      [0.1, 1.0, 2.8],

      [0.1, 1.0, 2.6],
      [0.1, 0.5, 2.6],
      [0.1, 0.0, 2.6],   

      [0.1, 0.0, 2.4], 
      [0.0, 0.5, 2.4],
      [0.1, 1.0, 2.4],

      [0.1, 1.0, 2.2],
      [0.1, 0.5, 2.2],
      [0.1, 0.0, 2.2],   

      [0.1, 0.0, 2.0], 
      [0.0, 0.5, 2.0],
      [0.1, 1.0, 2.0],                
      [0.0, 1.0, 2.0]
      ])

    t = np.arange(0,xyz.shape[0]*2,2)
    rpy = np.zeros_like(xyz)

    return t, xyz, rpy

