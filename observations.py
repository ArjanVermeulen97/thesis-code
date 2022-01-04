# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:10:39 2021

@author: Arjan
"""

import numpy as np
import pandas as pd
from math import sqrt, exp, tan, log10, pi, acos, asin, sin, cos, tanh
from transformations import angle_calc, ecliptic_to_galactic, R_2, R_3, sun_scale
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import time


### Values vis
vis_aperture = 0.5              # m^2 (0.5m diameter)
vis_pixelAngle = 9.4e-11        # sr, from 2"/pixel
vis_quantumEff = 0.88           # -
vis_transmittance = 0.99        # -
vis_straddleFac = 1             # - (compensated by trailing loss eq)
vis_integrationTime = 24        # s
vis_readNoise = 4               # e-
vis_darkNoise = 0               # e-/s
vis_factor = 8.937E-9           # correction for dumb units
vis_coulomb = 1.6E-19           # J/e-

### Values tir
tir_aperture = 0.5              # m^2 (0.5m diamter)
tir_pixelAngle = 2.12e-10       # sr, from 3"/pixel
tir_quantumEff = 0.55           # -
tir_transmittance = 0.99        # -
tir_straddleFac = 1             # - (compensated by trailing loss eq)
tir_integrationTime = 180       # s
tir_readNoise = 200               # e-
tir_darkNoise = 1000             # e-/s
tir_factor = 7.49E-19           # correction for dumb units
tir_coulomb = 1.6E-19           # J/e-

def blackbody(wavelength, temp, megajansky=False):
    ''' returns blackbody radiation as function of wavelength and temperature'''
    h = 6.626E-34 # Plack constant
    kB = 1.380E-23 # Boltzmann constant
    c = 299792458 # Light speed
    v = c / wavelength
    B = (2*h*v**3)/(c**2) / (np.exp((h*v)/(kB*temp)) - 1)
    if megajansky:
        B = B/(10**-20)
    return B


def vis_mag(t_mag, t_x, t_y, t_z, s_x, s_y, s_z):
    # Returns visual magnitude of target asteroid
    r_x = s_x - t_x     # relative x
    r_y = s_y - t_y     # relative y
    r_z = s_z - t_z     # relative z
    t_abs = sqrt(t_x*t_x + t_y* t_y + t_z*t_z)
    r_abs = sqrt(r_x*r_x + r_y* r_y + r_z*r_z)
    elongation = acos((t_x*r_x + t_y*r_y + t_z*r_z) / (t_abs*r_abs))
    phase = acos((-t_x*r_x - t_y*r_y - t_z*r_z) / (t_abs*r_abs))
    
    if elongation / pi * 180 < 60:
        # Per Stokes at al (2003)
        try:
            V = t_mag + 5*log10(t_abs*r_abs) + 5.03 - 10.373*log10(pi - phase)
        except ValueError:
            # 0 degrees, backlit by sun
            V = 1000
    else:
        # Per Stokes er al (2003)
        phi_1 = exp(-3.33*(tan(phase/2))**0.63)
        phi_2 = exp(-1.87*(tan(phase/2))**1.22)
        V = t_mag + 5*log10(t_abs*r_abs) - 2.5*log10(0.85*phi_1 + 0.15*phi_2)
    
    receivedFlux = 100**((V+26.74)/-5)*1350
    
    return receivedFlux


def tir_mag(t_mag, t_alb, t_x, t_y, t_z, s_x, s_y, s_z):
    stefanBoltzmann = 5.67E-8
    beaming = 1.22
    solarFlux = 1373 / (t_x**2 + t_y**2 + t_z**2)
    R = 664500 / sqrt(t_alb) * 10**(-1*t_mag / 5)
    T_max = ((1 - t_alb)*solarFlux/(beaming*stefanBoltzmann))**0.25
    
    # Next, calculate phase angle
    r_x = s_x - t_x     # relative x
    r_y = s_y - t_y     # relative y
    r_z = s_z - t_z     # relative z
    t_abs = sqrt(t_x*t_x + t_y* t_y + t_z*t_z)
    r_abs = sqrt(r_x*r_x + r_y* r_y + r_z*r_z)
    phase = acos((-t_x*r_x - t_y*r_y - t_z*r_z) / (t_abs*r_abs))
    
    delta_theta = pi/4
    delta_phi = pi/4
    delta_lambda = 299792458 / (10E-6 - 6E-6)
    flux = 0
    for theta in [-3/8*pi+phase, -1/8*pi+phase, 1/8*pi+phase, 3/8*pi+phase]:
        for phi in [-3/8*pi, -1/8*pi, 1/8*pi, 3/8*pi]:
            if theta > 7*pi/16:
                continue # No flux
            T = T_max*cos(theta)**0.25 * cos(phi)**0.25
            flux += delta_theta * delta_phi * delta_lambda *\
                0.5*(blackbody(6E-6, T) + blackbody(10E-6, T)) *\
                    cos(phi) * cos(theta - phase)
    power = flux*0.9*R**2
    receivedFlux = power / ((r_abs * 150_000_000_000)**2)
    return receivedFlux


def build_interpolator(filename):
    dataFile = pd.read_csv(filename, index_col=0, header=0)
    stackedData = dataFile.stack().reset_index().values
    x = []
    y = []
    z = []
    for item in stackedData:
        x.append(float(item[0]))
        y.append(float(item[1]))
        z.append(float(item[2]))
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    return interp


vis_stars = build_interpolator('starbackground.csv')
vis_zodiac = build_interpolator('zodiacgegenschein.csv')
tir_stars = build_interpolator('ir_starbackground.csv')
tir_zodiac = build_interpolator('ir_zodiacgegenschein.csv')


def observation(t_mag, t_alb, t_x, t_y, t_z, s_x, s_y, s_z, mode, verbose=False):
    assert mode == "VIS" or mode == "TIR"
    
    r_x = t_x - s_x     # relative x
    r_y = t_y - s_y     # relative y
    r_z = t_z - s_z     # relative z
    
    try:
        r_abs = sqrt(r_x**2 + r_y**2 + r_z**2)
        s_abs = sqrt(s_x**2 + s_y**2 + s_z**2)
        t_abs = sqrt(t_x**2 + t_y**2 + t_z**2)
        p_abs = sqrt(r_x**2 + r_y**2)
    
        l_z = acos((-r_x*s_x - r_y*s_y) / (p_abs * s_abs))
        b_z = acos((r_x*t_x + r_y*t_y + r_z*t_z) / (r_abs * t_abs))
        l_e = (l_z + pi)%(2*pi)
        b_e = b_z
        l_g, b_g = ecliptic_to_galactic(l_e, b_e)
        l_z = l_z * 180 / pi
        b_z = b_z * 180 / pi
    except:
        print(r_x, r_y, r_z)
        return 0
    
    if mode == "VIS":
        signalBG = (sun_scale(s_abs) * vis_zodiac(l_z, b_z) +\
                    vis_stars(l_g, b_g)) * vis_integrationTime *\
            vis_pixelAngle * vis_factor / vis_coulomb *\
                vis_transmittance * vis_quantumEff * vis_aperture
            
        signalTarget = vis_mag(t_mag, t_x, t_y, t_z, s_x, s_y, s_z) *\
            vis_aperture * vis_quantumEff * vis_transmittance *\
                vis_integrationTime / vis_coulomb
                
        SNR = (signalTarget * vis_straddleFac) /\
            sqrt(vis_readNoise + (vis_darkNoise * vis_integrationTime) +\
                 signalBG + signalTarget)
            
    if mode == "TIR":
        signalBG = (sun_scale(s_abs) * tir_zodiac(l_z, b_z) +\
                    tir_stars(l_g, b_g)) * tir_integrationTime *\
            tir_pixelAngle * tir_factor / tir_coulomb *\
                tir_transmittance * tir_quantumEff * tir_aperture
            
        signalTarget = tir_mag(t_mag, t_alb, t_x, t_y, t_z, s_x, s_y, s_z) *\
            tir_aperture * tir_quantumEff * tir_transmittance *\
                tir_integrationTime / tir_coulomb
                
        SNR = (signalTarget * tir_straddleFac) /\
            sqrt(tir_readNoise + (tir_darkNoise * tir_integrationTime) +\
                 signalBG + signalTarget)
    if verbose:
        print(f"Background signal: {signalBG:.2}")
        print(f"Target signal:     {signalTarget:.2}")
        print(f"Signal-to-noise:   {SNR:.2}")
    return SNR


def detection(SNR):
    return 1 if SNR > 5 else 0


def detection_prob(SNR, binary=True):
    if SNR < 1:
        return False
    if SNR > 5:
        return True
    probability = 0.5 + 0.5*tanh(SNR-3)
    if not binary:
        return probability
    else:
        return int(probability > np.random.random())


if False:
    # Make time estimate
    x_arr = [np.random.random() + 0.5 for i in range(100)]
    y_arr = [np.random.random() + 0.5 for i in range(100)]
    z_arr = [np.random.random() + 0.5 for i in range(100)]
    
    vis_startTime = time.time()
    
    for x in x_arr:
        for y in y_arr:
            for z in z_arr:
                observation(15, 0.1, x, y, z, 1, 0, 0, 'VIS')
    vis_time = time.time() - vis_startTime
    
    tir_startTime = time.time()
    
    for x in x_arr:
        for y in y_arr:
            for z in z_arr:
                observation(15, 0.1, x, y, z, 1, 0, 0, 'TIR')
    tir_time = time.time() - tir_startTime
    
    print(f"VIS time: {str(vis_time/1_000_000)}")
    print(f"TIR time: {str(tir_time/1_000_000)}")
    
    
def image(payload):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize=(12,6))
    im = np.zeros((180, 360))
    for index_x, b_e in enumerate(range(-90, 90)):
        for l_e in range(0, 360):
            l_g, b_g = ecliptic_to_galactic(l_e, b_e)
            l_z = l_e
            b_z = b_e
            
            if payload == 'VIS':
                im[index_x][l_e] = 0*vis_zodiac(l_z, b_z) + vis_stars(l_g, b_g)
            if payload == 'TIR':
                im[index_x][l_e] = 0*tir_zodiac(l_z, b_z) + tir_stars(l_g, b_g)
    if payload == 'VIS':
        plt.imshow(im, extent=[0, 360, -90, 90], vmin=10, vmax=1_000, cmap='bone')
        plt.colorbar()
        plt.xlabel('Ecliptic Longitude [deg]')
        plt.ylabel('Ecliptic Latitude [deg]')
    if payload == 'TIR':
        plt.imshow(im, extent=[0, 360, -90, 90], vmin=0, vmax=35, cmap='hot')
        plt.colorbar()
        plt.xlabel('Ecliptic Longitude [deg]')
        plt.ylabel('Ecliptic Latitude [deg]')