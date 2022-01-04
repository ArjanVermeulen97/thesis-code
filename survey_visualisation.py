# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:59:53 2021

@author: Arjan
"""

import numpy as np
import pandas as pd
import swifter
from scipy import optimize
from math import sqrt, exp, tan, log10, pi, acos, asin, sin, cos, tanh
from observations import observation, detection, detection_prob
from orbital import pos_heliocentric, theta_solve, theta_step
import seaborn as sns
from population_models import init_asteroids_jpl, init_asteroids_granvik
import matplotlib.pyplot as plt
import time
import pickle

def init_asteroids():
    return init_asteroids_granvik(200, True)


def init_satellites(n, a, e, s):
    satellites = {i: {'a': a,
                      'e': e,
                      'i': 0,
                      'long_node': 0,
                      'arg_peri': 0,
                      'anomaly': s*i,
                      'payload': 'VIS',
                      'cadence': 2} for i in range(n)}
    return satellites


def asteroid_positions(day, df_asteroids):
    '''Propagate all orbital elements to certain day and reset observations'''
    df_asteroids['M'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: theta_step(row['a'], row['anomaly'], day), axis=1)
    df_asteroids['theta'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: theta_solve(row['e'], row['M']), axis=1)
    df_asteroids['x'], df_asteroids['y'], df_asteroids['z'] = zip(*df_asteroids.swifter.progress_bar(False).apply(lambda row: pos_heliocentric(row['a'], 
                                                                                                              row['e'], 
                                                                                                              row['i'], 
                                                                                                              row['theta'], 
                                                                                                              row['long_node'], 
                                                                                                              row['arg_peri']),
                                                                                 axis=1
                                                                                 ))
    df_asteroids['detected'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: 1 if row['n_obs'] > 4 and row['detected'] == 0 else row['detected'], axis=1)
    df_asteroids['n_obs'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: 0 if day - row['last_obs'] > 90 else row['n_obs'], axis=1)
    return df_asteroids


def satellite_positions(day, satellites):
    '''Propagate satellite positions'''
    for sat in satellites.values():
        sat['M'] = theta_step(sat['a'], sat['anomaly'], day)
        sat['theta'] = theta_solve(sat['e'], sat['M'])
        sat['x'], sat['y'], sat['z'] = pos_heliocentric(sat['a'],
                                                        sat['e'],
                                                        sat['i'],
                                                        sat['theta'],
                                                        sat['long_node'],
                                                        sat['arg_peri'])
    return satellites


def make_observations(sat, day, df_asteroids):
    if day % sat['cadence']:
        return 0
    df_asteroids['SNR'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: observation(row['H'], 
                                                                     row['albedo'], 
                                                                     row['x'], 
                                                                     row['y'], 
                                                                     row['z'], 
                                                                     sat['x'], 
                                                                     sat['y'], 
                                                                     sat['z'], 
                                                                     sat['payload']
                                                                     ), 
                                             axis=1
                                             )
    df_asteroids['step_obs'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: min(row['step_obs'] + detection_prob(row['SNR']), 1),
                                                  axis=1
                                                  )
    return day

def do_survey(x):
    semi_major, eccentricity, spread = x[0], x[1], x[2]
    n_sats = 5
    df_asteroids = init_asteroids()
    #print(f"Satellites:   {n_sats}")
    #print(f"Semi-Major:   {semi_major}")
    #print(f"Eccentricity: {eccentricity}")
    #print(f"Spread:       {spread}")
    df_asteroids['n_obs'] = 0
    df_asteroids['step_obs'] = 0
    df_asteroids['last_obs'] = 0
    df_asteroids['detected'] = 0
    df_asteroids['diameter'] = df_asteroids.apply(lambda row: 1329000/sqrt(row['albedo'])*10**(-1*row['H'] / 5), axis=1)
    #df_asteroids = df_asteroids.fillna(0.15)
    satellites = init_satellites(n_sats, semi_major, eccentricity, spread)
    completeness = []
    fig, axes = plt.subplots(1,1, figsize=(8,8))
    bg = plt.imread('background.png')
    
    for day in range(0, 731, 2):
        n_detected = df_asteroids[df_asteroids['detected'] > 0]['detected'].count()
        n_undetected = df_asteroids[df_asteroids['detected'] == 0]['detected'].count()
        completeness.append(n_detected / (n_detected + n_undetected))
        df_asteroids = asteroid_positions(day, df_asteroids)
        satellites = satellite_positions(day, satellites)
        if day%satellites[0]['cadence'] == 0:
            for sat in satellites.values():
                make_observations(sat, day, df_asteroids)
        df_asteroids['n_obs'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: row['n_obs'] + row['step_obs'], axis=1)
        df_asteroids['last_obs'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: day if row['step_obs'] > 0 else row['last_obs'], axis=1)

        plt.pause(0.0001)
        if day == 0:
            #pass
            a = input("enter to start...")
        axes.clear()
        axes.set_xlim((-3, 3))
        axes.set_ylim((-3, 3))
        axes.imshow(bg, extent=[-5, 5, -5, 5])
        sns.scatterplot(data=df_asteroids, x='x', y='y', hue='detected', ax=axes, marker='o', legend=False, vmin=0, vmax=3, size='diameter')
        sns.scatterplot(data=pd.DataFrame(satellites).transpose(), x='x', y='y', marker='D', color="red", s=30)
        fig.canvas.draw()
        fig.suptitle(f"Day: {day}, completeness: {completeness[-1]:.2%}, detections: {df_asteroids.loc[df_asteroids['step_obs'] > 0]['step_obs'].count()}")
        df_asteroids['step_obs'] = 0
    #print(f"Completeness: {completeness[-1]:.2%}")
    #print()
    

    return float(1 - completeness[-1])
    
    
if __name__ == '__main__':
    do_survey([0.6, 0.0, 2*pi/5])
    