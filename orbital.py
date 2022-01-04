# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:54:43 2021

@author: Arjan
"""

from math import sin, cos, tan, asin, acos, atan, sqrt, pi, atan2, log
import numpy as np

def R_1(a):
    R = np.array([[1, 0, 0],
                  [0, cos(a), 1*sin(a)],
                  [0, -1*sin(a), cos(a)]])
    return R

def R_2(a):
    R = np.array([[cos(a), 0, 1*sin(a)],
                  [0, 1, 0],
                  [-1*sin(a), 0, cos(a)]])
    return R

def R_3(a):
    R = np.array([[cos(a), 1*sin(a), 0],
                  [-1*sin(a), cos(a), 0],
                  [0, 0, 1]])
    return R

def pos_heliocentric(a, e, i, theta, raan, argPeri):
    # i = i/180*pi
    # raan = raan/180*pi
    # argPeri = argPeri/180*pi
    # theta = theta/180*pi
    
    r = a*(1-e**2)/(1+e*cos(theta))
    r_orbit = np.array([[r*cos(theta)],
                        [r*sin(theta)],
                        [0]])
    
    r_helio = R_3(-1*raan) @ R_1(-1*i) @ R_3(-1*argPeri) @ r_orbit
    
    return r_helio[0][0], r_helio[1][0], r_helio[2][0]

def theta_solve(e, M, error=1.0E-14):
    def e_start(e, M):
        t34 = e**2
        t35 = e*t34
        t33 = cos(M)
        return M + (-0.5*t35 + e + (t34 + 3/2*t33*t35)*t33)*sin(M)

    def eps(e, M, x):
        t1 = cos(x)
        t2 = -1 + e*t1
        t3 = sin(x)
        t4 = e*t3
        t5 = -x + t4 + M
        t6 = t5/(0.5*t5*t4/t2 + t2)
        return t5/((0.5*t3 - 1/6.*t1*t6)*e*t6+t2)
    
    Mnorm = M%(2*pi)
    part = 2*pi if Mnorm > pi else 0
    sign = -1 if Mnorm > pi else 1
    E0 = e_start(e, Mnorm)
    dE = error + 1
    n_iter = 0
    while dE > error:
        E = E0 - eps(e, Mnorm, E0)
        dE = abs(E-E0)
        E0 = E
        n_iter += 1
        if n_iter == 1000:
            print("Doesn't converge :(")
            print(e, M)
            return M
    try:
        return part + acos((cos(E) - e)/(1-e*cos(E)))*sign
    except:
        print(f"Error in theta solve: {E}, {e}")
        return 0

def theta_step(a, theta_old, days=1):    
    mu_sun = 1.327124E11 / (150_000_000**3) * (86_400**2)   # AU^3 / day^2
    meanAnomaly = sqrt(mu_sun/(a**3))
    return (theta_old + days*meanAnomaly)%(2*pi)
