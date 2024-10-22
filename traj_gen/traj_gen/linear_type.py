# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
# Functions polynom_slow, min_deriv_poly are derived from Peter Huang's work:
# https://github.com/hbd730/quadcopter-simulation
# author: Peter Huang
# email: hbd730@gmail.com
# license: BSD
# Please feel free to use and modify this, but keep the above information. Thanks!


import numpy as np
from numpy import pi
from numpy.linalg import norm


def polynom(n_coeffs, derivative, t):
    """Calculate polynomial terms of a polynomial with n_coeffs  at the (derivative)th derivative 
    calculate snap terms at t=0 for 7th order polynomial polynom(n_coeffs=8,t=0,derivative=4)
    calculate position terms at t=2 for 7th order polynomial polynom(n_coeffs=8,t=2,derivative=0)    x(t=2) = a0 + a1x + a2x^2 + a3x^3+ a4x^4+ a5x^5 + a6x^6 + a7x^7 >> return >> [1, 2, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7]
    Args:
        n_coeffs (int): number of coefficients, this relates to the order of the polynomial n_coeffs=8 for 7th order polynomial
        derivative (int): derivative order, 0=position, 1=velocity,...
        t (float): time in seconds

    Returns:
        _type_: _description_
    """

    terms = np.zeros(n_coeffs)
    coeffs = np.polyder([1]*n_coeffs,derivative)[::-1]  # use np.polynomial instead?
    pows = t**np.arange(0,n_coeffs-derivative,1)
    terms[derivative:] = coeffs*pows
    return terms

def polynom_slow(n, k, t):
    """ This is a helper function to get the coefficient of coefficient for n-th
        order polynomial with k-th derivative at time t.
        polynom_slow(n, k, t)
        
        Example: n=8 means a 7th-order polynomial  $$x(t) = c_{7}t^7 + c_{6}t^6 + c_{5}t^5 + c_{4}t^4 + c_{3}t^3 + c_{2}t^2 + c_{1}t + c_{0}$$
                - evaluating at t=0 is polynom_slow(n=8, k=0,t=0)  >> array([0., 0., 0., 0., 0., 0., 0., 1.])
                - evaluating at t=1 is polynom_slow(n=8, k=0,t=1)  >> array([1., 1., 1., 1., 1., 1., 1., 1.])
                - evaluating velocity (first derivative)  at t=0 is polynom_slow(n=8, k=1,t=0)  >> array([0., 0., 0., 0., 0., 0., 1., 0.]) 
                  which was the result from equation $$\dot{x}(t) = 7c_{7}t^6 +6 c_{6}t^5 + 5c_{5}t^4 + 4c_{4}t^3 + 3c_{3}t^2 + 2c_{2}t + c_{1}$$
                - evaluating velocity (first derivative) at t=1 is polynom_slow(n=8, k=1,t=1)  >> array([7., 6., 5., 4., 3., 2., 1., 0.])
                - evaluating jerk (third derivative) at t=1 is polynom_slow(8, 3,1)  >> array([210., 120.,  60.,  24.,   6.,   0.,   0.,   0.])
                  which was the result from equation > $$\dddot{x}(t) = 210c_{7}t^4 + 120c_{6}t^3 + 60c_{5}t^2 + 24c_{4}t + 6c_{3}$$
    """
    assert (n > 0 and k >= 0), "order and derivative must be positive."

    cc = np.ones(n)
    D  = np.linspace(n-1, 0, n)

    for i in range(n):
        for j in range(k):
            cc[i] = cc[i] * D[i]
            D[i] = D[i] - 1
            if D[i] == -1:
                D[i] = 0

    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])

    return cc

def generate_time_from_vel(waypoints:np.array, velocity:float ):
    """
    This function computes the time required to travel between each pair of waypoints given the velocity.
    """
    segments = waypoints.shape[0]-1
    times = np.zeros(segments)
    for i in range(segments):
        distance = np.linalg.norm(waypoints[i + 1] - waypoints[i])
        time = distance / velocity
        times[i]  = time
    return times

def test_min_snap_poly_speed(imp=1,k=500):
    # imp=1 or 2 which implementation to test
    # k maximum number of waypoints
    # k is number of waypoints
    import time
    performance=np.zeros((k-2,2))
    print(f"Performing {k-2} tests")
    for n in range(2,k):
        times = np.ones(n-1)
        waypoints = np.random.rand(n,3)*3
        t1 = time.time()
        if imp==1:
            coeff = min_deriv_poly(waypoints[:,0],times,4)
            coeff = min_deriv_poly(waypoints[:,1],times,4)
            coeff = min_deriv_poly(waypoints[:,2],times,4)
        else:
            coeff = min_snap_poly(waypoints,times,4)
        tf = time.time()-t1
        performance[n-2,0] = n
        performance[n-2,1] = tf
    print("performance results:\n")
    print(performance)
    return performance

# ================== First implementation - generic, one dimension only, slower ==========================
def min_deriv_poly(waypoints:np.ndarray, times:np.array, order:int)->np.array:
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [M*N,1] coefficients matrix for the N+1 waypoints (N segments), 
    where M is the number of coefficients per segment and is equal to (order)*2. If one 
    desires to create a minimum velocity, order = 1. Minimum snap would be order = 4. 

    1.The Problem
    Generate a full trajectory across N+1 waypoint is made of N polynomial line segment.
    Each segment is defined as a (2*order-1)-th order polynomial defined as follow:
    Minimum velocity:     Pi = ai_0 + ai1*t
    Minimum acceleration: Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    Minimum jerk:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    Minimum snap:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    Each polynomial has M unknown coefficients, thus we will have M*N unknown to
    solve in total, so we need to come up with M*N constraints.

    2.The constraints
    In general, the constraints is a set of condition which define the initial
    and final state, continuity between each piecewise function. This includes
    specifying continuity in higher derivatives of the trajectory at the
    intermediate waypoints.

    3.Matrix Design
    Since we have M*N unknown coefficients to solve, and if we are given M*N
    equations(constraints), then the problem becomes solving a linear equation.

    A * Coeff = B

    Let's look at B matrix first, B matrix is simple because it is just some constants
    on the right hand side of the equation. There are M*N constraints,
    so B matrix will be [M*N, 1].

    Coeff is the final output matrix consists of M*N elements. 
    Since B matrix is only one column, Coeff matrix must be [M*N, 1].

    A matrix is tricky, we then can think of A matrix as a coeffient-coeffient matrix.
    We are no longer looking at a particular polynomial Pi, but rather P1, P2...PN
    as a whole. Since now our Coeff matrix is [M*N, 1], and B is [M*N, 1], thus
    A matrix must have the form [M*N, M*N].

    A = [A10 A11 ... A1M   A20 A21 ... A2M   ... AN0 AN1 ... ANM
        ...
        ]

    Each element in a row represents the coefficient of coeffient aij under
    a certain constraint, where aij is the jth coeffient of Pi with i = 1...N, j = 0...(M-1).
    """

    n = len(waypoints) - 1
    nb_coeff = order*2

    # initialize A, and B matrix
    A = np.zeros([nb_coeff*n, nb_coeff*n])
    B = np.zeros(nb_coeff*n)

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i] # position for each segment at segment time t=0 is start position
        B[i + n] = waypoints[i+1] # position for each segment at segment time t=T is end position
        # velocities and higher derivatives are zero at t=0 and t=T for all segments
        

    # Example A format (for min snap nb_coeff=8)
    #                 
    #         -------------0-------1-----2-----3------4------5----6-----7-----8------9-----10----11----12----13----14----15----16--------------------------------------------------------------n*8-----
    # seg 1 (pos@t=0)   | a17=0  a16=0 a15=0 a14=0 a13=0  a12=0 a11=0 a10=1                                                               .....     
    # seg 2 (pos@t=0)   |                                                   a27=0  a26=0 a25=0 a24=0 a23=0 a22=0 a21=0 a20=1              .....
    # ...               |                                                                ...................
    # seg n (pos@t=0)   |                                                                                                                 .....    an7=0  an6=0 an5=0 a24=0 an3=0  an2=0 an1=0 an0=1 
    
    # seg 1 (pos@t=T1)  | a17=T1^7 T1^6  T1^5 T1^4  T1^3  T1^2  T1^1 a10=1           
    # seg 2 (pos@t=T2)  |                                                   a27=T2^7 T2^6  T2^5 T2^4  T2^3  T2^2  T2^1 a20=1              .....                  
    # ...               |                                                                ...................
    # seg n (pos@t=Tn)  |                                                                                                                 .....    an7=Tn^7 T2^6 T2^5  T2^4  T2^3  T2^2  T2^1 an0=1
    
    # seg 1 (vel@t=0)   | a17=0  a16=0 a15=0 a14=0 a13=0  a12=0 a11=1 a10=0          
    # seg 1 (acc@t=0)   | a17=0  a16=0 a15=0 a14=0 a13=0  a12=2 a11=0 a10=0          
    # ..... up to order (order=4 means up to jerk@t=0)
    
    # seg n (vel&t=Tn)  |                                                                                                                 .....    an7=7*Tn^6  6*Tn^5    .......         an1=1  an0=0          
    # seg n (acc&t=Tn)  |                                                                                                                 .....    an7=42*Tn^5 30*Tn^4    .......  an2=2 an1=0  an0=0    
    # .....  up to order (order=4 means up to jerk@t=T1)
    
    # seg 1-2 (vel@cont)| a17=0  a16=0 a15=0 a14=0 a13=0  a12=0 a11=1 a10=0  a27=-7*T1^6 -6*T1^5     .........   a21=1 a20=0          (velocity continuity at seg1-seg2)
    # seg 1-2 (acc@cont)| a17=0  a16=0 a15=0 a14=0 a13=0  a12=2 a11=0 a10=0  a27=-42*T1^5 -30*T1^4   ..........  a21=0 a20=0          (acceleration continuity at seg1-seg2)
    # .....
    # seg n-1 (vel@t=Tn)|             ...................                                    ...................                                       (velocity continuity at seg(n-1)-seg n)
    # seg n (vel@t=0)   |             ...................                                    ...................                                       (velocity continuity at seg(n-1)-seg n)
    # .....


    # Constraint 1 - Starting position for every segment (n constraints  - N*M coefficients update)
    for i in range(n):
        A[i][nb_coeff*i:nb_coeff*(i+1)] = polynom(nb_coeff, 0, 0)
        # for min snap nb_coef=8, this becomes
        # A[i][8*i:8*(i+1)] = polynom(8, 0, 0)

    # Constraint 2 - Ending position for every segment (n constraints  - N*M coefficients update)
    for i in range(n):
        A[i+n][nb_coeff*i:nb_coeff*(i+1)] = polynom(nb_coeff, 0, times[i])

    # Constraint 3 - Starting position derivatives (up to order) are null (0.5M-1 constraints (order-1) - M*(order-1) coefficients update)
    for k in range(1, order):
        A[2*n+k-1][:nb_coeff] = polynom(nb_coeff, k, 0)

    # Constraint 4 - Ending position derivatives (up to order) are null    (0.5M-1 constraints (order-1)  - M*(order-1) coefficients update)
    for k in range(1, order):
        A[2*n+(order-1)+k-1][-nb_coeff:] = polynom(nb_coeff, k, times[i])
    
    # Constraint 5 - All derivatives are continuous at each waypoint transition    ((N-1)*(M-1) constraints   - (N-1)*(M-1)*2M coefficients update)??
    for i in range(n-1):
        for k in range(1, nb_coeff-1):
            # it doesn't matter which coefficients are positive and which are negative
            A[2*n+2*(order-1) + i*2*(order-1)+k-1][i*nb_coeff : (i*nb_coeff+nb_coeff*2)] = np.concatenate((polynom(nb_coeff, k, times[i]), -polynom(nb_coeff, k, 0)))
    
    # solve for the coefficients
    # print(f"A({A.shape})=\n{A,round(1)}")
    # print(f"B({B.shape})=\n{B.round(1)}")
    coeff = np.linalg.solve(A, B)
    return coeff

# ================= second implementation, limited to minimum snap, multi-dimensional, faster ============
def min_snap_poly(waypoints:np.ndarray, times:np.array, method='lstsq'):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [M*N,1] coefficients matrix for the N+1 waypoints (N segments), 
    where M is the number of coefficients per segment and is equal to 8.

    For 1 spline, we have 8 unknown coefficients (c7, c6, c5, c4, c3, c2, c1, c0).
    Regarding the constraints, let's denote the number of waypoints by m:
    - m-1 constraints for position at t=0 (start of spline, last waypoint is excluded)
    - m-1 constraints for position at t=T (end of spline, first waypoint is excluded)
    - 1 constraint for velocity at t=0, acceleration at t=0, jerk at t=0 (3 constraints)
    - 1 constraint for velocity at t=T, acceleration at t=T, jerk at t=T (3 constraints)
    - m-2 constraints for continuity of each derivative (1...6) (first and last waypoints are excluded) - (m-2)*6

    Total number of constraints: 2(m-1) + 6 + 6(m-2)
    expected number of unknown coefficients: 8 * m-1 or 8 * number of splines
    
    This was adopted from https://github.com/Mdhvince/UAV-Autonomous-control
    """
    order = 4
    n = len(waypoints) - 1 # number of segments
    dim = len(waypoints[0]) # number of dimensions. for xyz > 3
    nb_coeff = order*2

    # initialize A, and b matrix
    A = np.zeros((nb_coeff * n, nb_coeff * n))
    b = np.zeros((nb_coeff * n,dim)) #

    row_counter = 0
    # ==== Position constraints =====
    # at t=0 - FOR ALL START OF SEGMENTS
    poly = polynom(nb_coeff, derivative=0, t=0)
    for i in range(n):
        wp0 = waypoints[i]
        A[row_counter, i * nb_coeff: nb_coeff * (i + 1)] = poly
        b[row_counter, :] = wp0
        row_counter += 1

    # at t=T - FOR ALL END OF SEGMENTS                                                  
    for i in range(n):
        wpT = waypoints[i + 1]
        timeT = times[i]
        poly = polynom(nb_coeff, derivative=0, t=timeT)
        A[row_counter, i * nb_coeff:nb_coeff * (i + 1)] = poly
        b[row_counter, :] = wpT
        row_counter += 1


    # ==== start and end constraints =====
    # populate the A and b matrices with constraints on the starting and ending splines.
    # - Starting spline constraint: Velocity/Acceleration/Jerk should be 0
    # - Ending spline constraint: Velocity/Acceleration/Jerk should be 0
    # We have 1 constraint for each derivative(3) and for 2 splines. So 3 constraints per splines. In total,
    # we have 6 constraints.
    
    # CONSTRAINTS FOR THE VERY FIRST SEGMENT at t=0
    for k in [1, 2, 3]:
        poly = polynom(nb_coeff, derivative=k, t=0)
        A[row_counter, 0:nb_coeff] = poly
        row_counter += 1

    # CONSTRAINTS FOR THE VERY LAST SEGMENT at t=T
    for k in [1, 2, 3]:
        poly = polynom(nb_coeff, derivative=k, t=times[-1])
        A[row_counter, (n - 1) *nb_coeff:nb_coeff * n] = poly
        row_counter += 1            

    # ==== continuity constraints =====
    # populate the A and b matrices with constraints on intermediate splines to ensure
    # continuity, hence smoothness.
    #  - Constraints up to the sixth derivative at t=0 should be the same at t=T. For example, no change of velocity
    #    between the end of a spline (polyT) and the start of the next spline (poly0)
    # We have one constraint for each derivative(6).

    for s in range(1, n):
        timeT = times[s-1]
        for k in range(1, nb_coeff-1): #[1, 2, 3, 4]:  # , 5, 6]. [1,2,3,4]>> sufficient constraints for min snap but may result in singular matrix if solving using inverse
            poly0 = -1 * polynom(nb_coeff, derivative=k, t=0)
            polyT = polynom(nb_coeff, derivative=k, t=timeT)
            poly = np.hstack((polyT, poly0))  # (end of seg) - (start of seg) must be 0. so no change of vel/acc/...
            A[row_counter, (s - 1) * nb_coeff:nb_coeff * (s + 1)] = poly
            row_counter += 1

    # solve for the coefficients
    # print(f"A({A.shape})=\n{A,round(1)}")
    # print(f"b({b.shape})=\n{b.round(1)}")
    if method == "lstsq":
        coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    else:
        coeffs = np.linalg.solve(A, b)

    return coeffs