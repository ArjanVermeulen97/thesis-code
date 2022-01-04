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
import datetime
import pickle

def init_asteroids(n=1000):
    return init_asteroids_granvik(n, random_state=int(datetime.datetime.now().timestamp()))


def init_satellites(n, a, e, s, p):
    satellites = {i: {'a': a[i],
                      'e': e[i],
                      'i': 0,
                      'long_node': 0,
                      'arg_peri': 0,
                      'anomaly': s[i],
                      'payload': p[i],
                      'cadence': 21 if p[i] == 'TIR' else 2} for i in range(n)}
    return satellites


def asteroid_positions(day, df_asteroids):
    '''Propagate all orbital elements to certain day and reset observations'''
    df_asteroids['M'] = df_asteroids.apply(lambda row: theta_step(row['a'], row['anomaly'], day), axis=1)
    df_asteroids['theta'] = df_asteroids.apply(lambda row: theta_solve(row['e'], row['M']), axis=1)
    df_asteroids['x'], df_asteroids['y'], df_asteroids['z'] = zip(*df_asteroids.apply(lambda row: pos_heliocentric(row['a'], 
                                                                                                              row['e'], 
                                                                                                              row['i'], 
                                                                                                              row['theta'], 
                                                                                                              row['long_node'], 
                                                                                                              row['arg_peri']),
                                                                                 axis=1
                                                                                 ))
    df_asteroids['detected'] = df_asteroids.apply(lambda row: day if row['n_obs'] > 4 and row['detected'] == 0 else row['detected'], axis=1)
    df_asteroids['n_obs'] = df_asteroids.apply(lambda row: 0 if day - row['last_obs'] > 90 else row['n_obs'], axis=1)
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
    if day%sat['cadence'] == 0:
        df_asteroids['SNR'] = df_asteroids.apply(lambda row: observation(row['H'], 
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
        df_asteroids['step_obs'] = df_asteroids.apply(lambda row: min(row['step_obs'] + detection_prob(row['SNR']), 2),
                                                      axis=1
                                                      )
    return day

def do_survey(x, n_asteroids=1000):
    assert len(x)%4 == 0
    n_sats = int(len(x)/4)
    semi_major, eccentricity, anomaly, payload = x[0:n_sats], x[n_sats:2*n_sats], x[2*n_sats:3*n_sats], x[3*n_sats:4*n_sats]
    df_asteroids = init_asteroids(n_asteroids)
    #print(f"Satellites:   {n_sats}")
    #print(f"Semi-Major:   {semi_major}")
    #print(f"Eccentricity: {eccentricity}")
    #print(f"Spread:       {spread}")
    df_asteroids['n_obs'] = 0
    df_asteroids['step_obs'] = 0
    df_asteroids['last_obs'] = 0
    df_asteroids['detected'] = 0
    df_asteroids = df_asteroids.fillna(0.15)
    satellites = init_satellites(n_sats, semi_major, eccentricity, anomaly, payload)
    completeness = []
    for day in range(0, 1825, 1):
        n_detected = df_asteroids[df_asteroids['detected'] > 0]['detected'].count()
        n_undetected = df_asteroids[df_asteroids['detected'] == 0]['detected'].count()
        completeness.append(n_detected / (n_detected + n_undetected))
        if day%2 and day%21:
            continue
        df_asteroids = asteroid_positions(day, df_asteroids)
        satellites = satellite_positions(day, satellites)
        for sat in satellites.values():
            make_observations(sat, day, df_asteroids)
        df_asteroids['n_obs'] = df_asteroids.apply(lambda row: row['n_obs'] + row['step_obs'], axis=1)
        df_asteroids['last_obs'] = df_asteroids.apply(lambda row: day if row['step_obs'] > 0 else row['last_obs'], axis=1)
        df_asteroids['step_obs'] = 0
    #print(f"Completeness: {completeness[-1]:.2%}")
    #print()
    return completeness
    
    
if __name__ == '__main__':
    for payload in ['TIR', 'VIS']:
        result = pd.DataFrame(columns=['n', 'a', 'completeness'])
        for n in range(1, 6):
            for a in np.arange(0.2, 2.05, 0.05):
                semi = [a for i in range(n)]
                ecc = [0 for i in range(n)]
                anomaly = [2*pi/n*i for i in range(n)]
                pl = [payload for i in range(n)]
                x = semi + ecc + anomaly + pl
                
                print(f"Running: {payload}, {n} S/C, {a:.2} AU")
                c = do_survey(x, 2500)[-1]
                print(f"{c:.4%}")
                res = {'n': n,
                       'a': a,
                       'completeness': c}
                result = result.append(res, ignore_index=True)
        if payload == 'TIR':
            result.to_csv('tir_2500_granvik_redo.csv')
        if payload == 'VIS':
            result.to_csv('vis_2500_granvik_redo.csv')
    if False:
        tir_one = [0.82, 0, 0, 'TIR']
        vis_one = [0.79, 0, 0, 'VIS']
        hybrid_tt = [0.92, 0.92, 0, 0, 0, pi, 'TIR', 'TIR']
        hybrid_tv = [0.96, 0.96, 0, 0, 0, pi, 'TIR', 'VIS']
        hybrid_vv = [1.0, 1.0, 0, 0, 0, pi, 'VIS', 'VIS']
        tir_five = [1.1, 1.1, 1.1, 1.1, 1.1, 0, 0, 0, 0, 0, 0, 2*pi/5, 2*2*pi/5, 3*2*pi/5, 4*2*pi/5, 'TIR', 'TIR', 'TIR', 'TIR', 'TIR']
        vis_five = [0.97, 0.97, 0.97, 0.97, 0.97, 0, 0, 0, 0, 0, 0, 2*pi/5, 2*2*pi/5, 3*2*pi/5, 4*2*pi/5, 'VIS', 'VIS', 'VIS', 'VIS', 'VIS']
        hybrid_tttt = [1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, pi/2, pi, 3*pi/2, 'TIR', 'TIR', 'TIR', 'TIR']
        hybrid_tttv = [1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, pi/2, pi, 3*pi/2, 'TIR', 'TIR', 'TIR', 'VIS']
        hybrid_ttvv = [1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, pi/2, pi, 3*pi/2, 'TIR', 'TIR', 'VIS', 'VIS']
        hybrid_tvvv = [1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, pi/2, pi, 3*pi/2, 'TIR', 'VIS', 'VIS', 'VIS']
        hybrid_vvvv = [1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, pi/2, pi, 3*pi/2, 'VIS', 'VIS', 'VIS', 'VIS']
        
        result = pd.DataFrame()
        
        options = {'TIR 1': tir_one,
                   'VIS 1': vis_one,
                   'TIR 5': tir_five,
                   'VIS 5': vis_five,
                   'HYB tt': hybrid_tt,
                   'HYB tv': hybrid_tv,
                   'HYB vv': hybrid_vv,
                   'HYB tttt': hybrid_tttt,
                   'HYB tttv': hybrid_tttv,
                   'HYB ttvv': hybrid_ttvv,
                   'HYB tvvv': hybrid_tvvv,
                   'HYB vvvv': hybrid_vvvv}
        
        for option in options:
            print(f"Running {option}...")
            c = do_survey(options[option], 2500)
            print(f"{c[-1]:.4%}")
            print()
            result[option] = c
            if option == 'TIR 1':
                print(result)
        result.to_csv('hybrid_test.csv')
    