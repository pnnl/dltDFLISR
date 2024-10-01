# -*- coding: utf-8 -*-
"""
Created on Dec 12 2019

@author: Moosa Moghimi
Original code available on: https://github.com/Moongar/DSSE/blob/master/WLS.py

@author: modified by Monish Mukherjee (2022)
"""
import numpy as np
import math

import logging
log = logging.getLogger(__name__)

def format_measurement(P_ij:np.ndarray, P_i:np.ndarray, Q_ij:np.ndarray, Q_i:np.ndarray, Vi:np.ndarray, Theta_i:np.ndarray, I_ij:np.ndarray) -> np.ndarray:
    """Format various power system sensor measurements into a standardized array.

    This function takes different types of measurements (real power, reactive power, voltage magnitudes, etc.) and formats them into a unified NumPy array `z` with the appropriate measurement types.

    Args:
        P_ij (np.ndarray): Real power flow measurements between nodes (P_ij).
        P_i (np.ndarray): Real power injections at nodes (P_i).
        Q_ij (np.ndarray): Reactive power flow measurements between nodes (Q_ij).
        Q_i (np.ndarray): Reactive power injections at nodes (Q_i).
        Vi (np.ndarray): Voltage magnitudes at nodes (Vi).
        Theta_i (np.ndarray): Voltage angles at nodes (Theta_i).
        I_ij (np.ndarray): Current flow measurements between nodes (I_ij).

    Returns:
        np.ndarray: A 2D array containing the formatted measurements along with their types and associated information.
    """
    # we add each measurement type one by one
    # types: 1:Pij 2:Pi 3:Qij 4:Qi 5:|Vi| 6:theta Vi 7:|Ireal| 8:|Iimag|
    # active power from pseudo measurements
    
    meas_count = 0 
    
    z =  np.array(()).reshape(0,5)
    ###########################################################################
    ######################### Real Power Measurments ##########################
    ###########################################################################
    if len(P_i) > 0:
        z_P_i = np.array([np.arange(meas_count,meas_count+len(P_i),1, dtype=int), 2 * np.ones(len(P_i)), P_i[:,1], P_i[:,0], np.zeros(len(P_i)), ]).T
        meas_count += len(P_i)
        z = np.concatenate((z, z_P_i))
        
    ###########################################################################
    ######################### Reactive Power Measurments ##########################
    ###########################################################################
    if len(Q_i) > 0:
        z_Q_i = np.array([np.arange(meas_count,meas_count+len(Q_i),1, dtype=int), 4* np.ones(len(Q_i)), Q_i[:,1], Q_i[:,0], np.zeros(len(Q_i))]).T
        meas_count += len(Q_i) 
        z = np.concatenate((z, z_Q_i))
        
    ###########################################################################
    ######################### Real Power Measurments ##########################
    ###########################################################################
    if len(Vi) > 0:
        z_V_i = np.array([np.arange(meas_count,meas_count+len(Vi),1, dtype=int), 5* np.ones(len(Vi)), Vi[:,1], Vi[:,0], np.zeros(len(Vi))]).T
        meas_count += len(Vi) 
        z = np.concatenate((z, z_V_i))          
    return z


def state_estimation(ybus:np.ndarray, z:np.ndarray, zmeas:np.ndarray, err_cov:np.ndarray,
                     iter_max:int, threshold:float, V_guess:np.ndarray, Theta_guess:np.ndarray) -> tuple:
    """Perform state estimation using the Weighted Least Squares (WLS) method.

    This function estimates the voltage magnitudes and angles at each bus in the power distribution network based on the provided measurements and their covariance matrix using the WLS method.

    Args:
        ybus (np.ndarray): The admittance matrix of the power system network.
        z (np.ndarray): Array containing measurement types and associated information.
        zmeas (np.ndarray): Measurement values.
        err_cov (np.ndarray): Error covariance matrix of the measurements.
        iter_max (int): Maximum number of iterations allowed for the WLS method to converge.
        threshold (float): Convergence threshold for the WLS method.
        V_guess (np.ndarray): Initial guess for the voltage magnitudes.
        Theta_guess (np.ndarray): Initial guess for the voltage angles.

    Returns:
        tuple:
            - v_phasor (np.ndarray): Estimated complex voltage phasors for each bus.
            - k (int): Number of iterations taken for the WLS method to converge.
            - f_ind (np.ndarray): Indices of measurement residuals.
            - s_u (float): Sum of the diagonal elements of the gain matrix.
            - e_2 (float): Sum of squares of measurement residuals.
    """
    delta_threshold=np.inf
    
    n = len(ybus)  # number of single phase nodes
    g = np.real(ybus)  # real part of the admittance matrix
    b = np.imag(ybus)  # imaginary art of the admittance matrix
    # x = np.concatenate(
    #     ([-2 * math.pi / 3, -4 * math.pi / 3], np.tile([0, -2 * math.pi / 3, -4 * math.pi / 3], math.floor(n / 3) - 1),
    #      np.ones(n) * (1 + .000001 * np.random.randn(n))))  # our initial guess fot the voltage phasors
    # x = np.concatenate((Theta_guess[3:], V_guess[3:]))
    x = np.concatenate((Theta_guess[3:], V_guess))
    k = 0
    cont = True
    while k < iter_max and cont:
        v = x[n - 3:]  # voltage magnitudes
        ## Fixing the Source voltage at every iteration
        # v = np.concatenate((V_guess[0:3], x[n - 3:]), axis=0)
        # th = np.concatenate(([0], x[0: n - 1]))  # voltage angles. we add a 0 for the reference bus
        ## Fixing the Source angles at every iteration
        th = np.concatenate(([0,-2*math.pi/3,2*math.pi/3], x[0: n - 3]))
        
        # Calculating the measurement functions h(x)
        h = np.zeros(len(z))
        for m in range(0, len(z)):
            if z[m, 1] == 2:  # Pi active load demand at node i
                i = int(z[m, 3]) 
                for jj in range(n):
                    h[m] += v[i] * v[jj] * (g[i, jj] * math.cos(th[i] - th[jj]) + b[i, jj] * math.sin(th[i] - th[jj]))
            elif z[m, 1] == 4:  # Qi reactive load demand at node i
                i = int(z[m, 3]) 
                for jj in range(n):
                    h[m] += v[i] * v[jj] * (g[i, jj] * math.sin(th[i] - th[jj]) - b[i, jj] * math.cos(th[i] - th[jj]))
            elif z[m, 1]  == 5:  # |Vi| voltage phasor magnitude at bus i
                i = int(z[m, 3]) 
                h[m] = v[i]
            elif z[m, 1] == 6:  # Theta Vi voltage phasor phase angle at bus i
                i = int(z[m, 3]) 
                h[m] = th[i]
            elif z[m, 1] == 7 or z[m, 1]  == 8:
                i = ztype[1, m] - 1  # sending node
                jj = ztype[2, m] - 1  # receiving node
                ph = ztype[3, m] - 1  # phase
                a1, b1, c1 = 3 * i + [0, 1, 2]
                a2, b2, c2 = 3 * jj + [0, 1, 2]
                yline = -ybus[np.array([a1, b1, c1])[:, None], np.array([a2, b2, c2])]
                gline = np.real(yline)
                bline = np.imag(yline)
                if ztype[0, m] == 7:  # real part of Iij phasor
                    h[m] = gline[ph, 0] * (v[a1] * math.cos(th[a1]) - v[a2] * math.cos(th[a2])) - \
                           bline[ph, 0] * (v[a1] * math.sin(th[a1]) - v[a2] * math.sin(th[a2])) + \
                           gline[ph, 1] * (v[b1] * math.cos(th[b1]) - v[b2] * math.cos(th[b2])) - \
                           bline[ph, 1] * (v[b1] * math.sin(th[b1]) - v[b2] * math.sin(th[b2])) + \
                           gline[ph, 2] * (v[c1] * math.cos(th[c1]) - v[c2] * math.cos(th[c2])) - \
                           bline[ph, 2] * (v[c1] * math.sin(th[c1]) - v[c2] * math.sin(th[c2]))
                else:  # imaginary part of Iij phasor
                    h[m] = gline[ph, 0] * (v[a1] * math.sin(th[a1]) - v[a2] * math.sin(th[a2])) + \
                           bline[ph, 0] * (v[a1] * math.cos(th[a1]) - v[a2] * math.cos(th[a2])) + \
                           gline[ph, 1] * (v[b1] * math.sin(th[b1]) - v[b2] * math.sin(th[b2])) + \
                           bline[ph, 1] * (v[b1] * math.cos(th[b1]) - v[b2] * math.cos(th[b2])) + \
                           gline[ph, 2] * (v[c1] * math.sin(th[c1]) - v[c2] * math.sin(th[c2])) + \
                           bline[ph, 2] * (v[c1] * math.cos(th[c1]) - v[c2] * math.cos(th[c2]))
            else:
                print("Measurement type not defined!")
       
        # print(h-z)
        # calculating the jacobian of h
        h_jacob = np.zeros([len(z), len(x)])
        for m in range(0, len(z)):
            if z[m, 1] == 2:  # Pi active load demand at node i
                i = int(z[m, 3]) 
                for jj in range(n):
                    if jj != i:
                        if jj > 2:
                            h_jacob[m, jj - 3] = v[i] * v[jj] * (g[i, jj] * math.sin(th[i] - th[jj]) -
                                                                 b[i, jj] * math.cos(th[i] - th[jj]))
                        h_jacob[m, jj + n - 3] = v[i] * (g[i, jj] * math.cos(th[i] - th[jj]) +
                                                             b[i, jj] * math.sin(th[i] - th[jj]))
                if i > 2:
                    h_jacob[m, i - 3] = -v[i] ** 2 * b[i, i]
                    for jj in range(n):
                        h_jacob[m, i - 3] += v[i] * v[jj] * (-g[i, jj] * math.sin(th[i] - th[jj]) +
                                                             b[i, jj] * math.cos(th[i] - th[jj]))
                # if i > 2:
                h_jacob[m, i + n - 3] = v[i] * g[i, i]
                for jj in range(n):
                    h_jacob[m, i + n - 3] += v[jj] * (g[i, jj] * math.cos(th[i] - th[jj]) +
                                                          b[i, jj] * math.sin(th[i] - th[jj]))

            elif z[m, 1] == 4:  # Qi reactive load demand at node i
                i = int(z[m, 3]) 
                for jj in range(n):
                    if jj != i:
                        if jj > 2:
                            h_jacob[m, jj - 3] = v[i] * v[jj] * (-g[i, jj] * math.cos(th[i] - th[jj]) -
                                                                 b[i, jj] * math.sin(th[i] - th[jj]))
                        h_jacob[m, jj + n - 3] = v[i] * (g[i, jj] * math.sin(th[i] - th[jj]) -
                                                             b[i, jj] * math.cos(th[i] - th[jj]))
                if i > 2:
                    h_jacob[m, i - 3] = -v[i] ** 2 * g[i, i]
                    for jj in range(n):
                        h_jacob[m, i - 3] += v[i] * v[jj] * (g[i, jj] * math.cos(th[i] - th[jj]) +
                                                             b[i, jj] * math.sin(th[i] - th[jj]))
                # if i > 2:
                h_jacob[m, i + n - 3] = -v[i] * b[i, i]
                for jj in range(n):
                    h_jacob[m, i + n - 3] += v[jj] * (g[i, jj] * math.sin(th[i] - th[jj]) -
                                                          b[i, jj] * math.cos(th[i] - th[jj]))

            elif z[m, 1] == 5:  # |Vi| voltage phasor magnitude at bus i
                i = int(z[m, 3]) 
                h_jacob[m, i + n - 3] = 1

            elif z[m, 1] == 6:  # Theta Vi voltage phasor phase angle at bus i
                i = int(z[m, 3]) 
                h_jacob[m, i - 3] = 1

            elif z[m, 1] == 7 or z[m, 1] == 8:
                i = ztype[1, m] - 1  # sending node
                jj = ztype[2, m] - 1  # receiving node
                ph = ztype[3, m] - 1  # phase
                a1, b1, c1 = 3 * i + [0, 1, 2]
                a2, b2, c2 = 3 * jj + [0, 1, 2]
                yline = -ybus[np.array([a1, b1, c1])[:, None], np.array([a2, b2, c2])]
                gline = np.real(yline)
                bline = np.imag(yline)
                if ztype[0, m] == 7:  # real part of Iij phasor
                    # derivatives with respect to voltage phase angles
                    if a1 > 0:
                        h_jacob[m, a1-1] = -gline[ph, 0] * v[a1] * math.sin(th[a1]) - bline[ph, 0] * v[a1] * math.cos(th[a1])
                    h_jacob[m, b1-1] = -gline[ph, 1] * v[b1] * math.sin(th[b1]) - bline[ph, 1] * v[b1] * math.cos(th[b1])
                    h_jacob[m, c1-1] = -gline[ph, 2] * v[c1] * math.sin(th[c1]) - bline[ph, 2] * v[c1] * math.cos(th[c1])
                    h_jacob[m, a2-1] = gline[ph, 0] * v[a2] * math.sin(th[a2]) + bline[ph, 0] * v[a2] * math.cos(th[a2])
                    h_jacob[m, b2-1] = gline[ph, 1] * v[b2] * math.sin(th[b2]) + bline[ph, 1] * v[b2] * math.cos(th[b2])
                    h_jacob[m, c2-1] = gline[ph, 2] * v[c2] * math.sin(th[c2]) + bline[ph, 2] * v[c2] * math.cos(th[c2])
                    # derivatives with respect to voltage magnitudes
                    h_jacob[m, a1+n-1] = gline[ph, 0] * math.cos(th[a1]) - bline[ph, 0] * math.sin(th[a1])
                    h_jacob[m, b1+n-1] = gline[ph, 1] * math.cos(th[b1]) - bline[ph, 1] * math.sin(th[b1])
                    h_jacob[m, c1+n-1] = gline[ph, 2] * math.cos(th[c1]) - bline[ph, 2] * math.sin(th[c1])
                    h_jacob[m, a2+n-1] = -gline[ph, 0] * math.cos(th[a2]) + bline[ph, 0] * math.sin(th[a2])
                    h_jacob[m, b2+n-1] = -gline[ph, 1] * math.cos(th[b2]) + bline[ph, 1] * math.sin(th[b2])
                    h_jacob[m, c2+n-1] = -gline[ph, 2] * math.cos(th[c2]) + bline[ph, 2] * math.sin(th[c2])
                else:  # imaginary part of Iij phasor
                    if a1 > 0:
                        h_jacob[m, a1-1] = gline[ph, 0] * v[a1] * math.cos(th[a1]) - bline[ph, 0] * v[a1] * math.sin(th[a1])
                    h_jacob[m, b1-1] = gline[ph, 1] * v[b1] * math.cos(th[b1]) - bline[ph, 1] * v[b1] * math.sin(th[b1])
                    h_jacob[m, c1-1] = gline[ph, 2] * v[c1] * math.cos(th[c1]) - bline[ph, 2] * v[c1] * math.sin(th[c1])
                    h_jacob[m, a2-1] = -gline[ph, 0] * v[a2] * math.cos(th[a2]) + bline[ph, 0] * v[a2] * math.sin(th[a2])
                    h_jacob[m, b2-1] = -gline[ph, 1] * v[b2] * math.cos(th[b2]) + bline[ph, 1] * v[b2] * math.sin(th[b2])
                    h_jacob[m, c2-1] = -gline[ph, 2] * v[c2] * math.cos(th[c2]) + bline[ph, 2] * v[c2] * math.sin(th[c2])
                    # derivatives with respect to voltage magnitudes
                    h_jacob[m, a1+n-1] = gline[ph, 0] * math.sin(th[a1]) + bline[ph, 0] * math.cos(th[a1])
                    h_jacob[m, b1+n-1] = gline[ph, 1] * math.sin(th[b1]) + bline[ph, 1] * math.cos(th[b1])
                    h_jacob[m, c1+n-1] = gline[ph, 2] * math.sin(th[c1]) + bline[ph, 2] * math.cos(th[c1])
                    h_jacob[m, a2+n-1] = -gline[ph, 0] * math.sin(th[a2]) - bline[ph, 0] * math.cos(th[a2])
                    h_jacob[m, b2+n-1] = -gline[ph, 1] * math.sin(th[b2]) - bline[ph, 1] * math.cos(th[b2])
                    h_jacob[m, c2+n-1] = -gline[ph, 2] * math.sin(th[c2]) - bline[ph, 2] * math.cos(th[c2])

            else:
                print("Measurement type not defined!")
        # the right hand side of the equation
        e = (zmeas - h)
        err_cov_inv = np.linalg.pinv(err_cov)
        rhs = h_jacob.transpose() @ err_cov_inv @ e
        # d1 = h_jacob.transpose() @ np.linalg.inv(err_cov) #previusly all inversions were using np.linalg.inv
        # d2 = np.linalg.inv(err_cov) @ (zmeas-h)
        # saving to mat file
        # scipy.io.savemat('C:/Users/Moosa Moghimi/Desktop/testArrays.mat', {'d11': d1, 'd22': d2})
        # print("Array saved")
        # the gain matrix
        gain = h_jacob.transpose() @ err_cov_inv @ h_jacob

        delta_x = np.linalg.solve(gain, rhs)
        # print(np.abs(np.sum(delta_x)))

        x += delta_x
        if (np.abs(np.sum(delta_x)) > delta_threshold) or (np.abs(np.sum(delta_x)) < threshold):
            cont = False
        delta_threshold=np.sum(np.abs(delta_x))
        # if np.max(np.absolute(delta_x)) < threshold:
        #     cont = False
        # k += 1
        
    # print("Iterations to Converge: {}".format(k))
    f_ind = np.linalg.solve(err_cov, e**2)
    
    v_SE =  x[n - 3:]  # voltage magnitudes
    v = v_SE
    # v = np.concatenate((V_guess[0:3], v_SE), axis=0)
    th = np.concatenate(([0,-2*math.pi/3,2*math.pi/3], x[0: n - 3])) # voltage angles. we add a 0 for the reference bus
    v_phasor = v * (np.cos(th) + 1j * np.sin(th))
    
    s_u = np.sum(np.abs(np.diagonal(gain)))
    e_2 = np.sum(e**2)
    return v_phasor, k, f_ind, s_u, e_2
