# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:36:27 2021

@author: Arjan
"""

import pandas as pd
from math import pi
import warnings

def init_asteroids_jpl(n_asteroids):
    df_asteroids = pd.read_csv('jpl_neo.csv')
    df_asteroids = df_asteroids.sample(n=n_asteroids, random_state=9)
    df_asteroids['long_node'] = df_asteroids['long_node']/180*pi
    df_asteroids['arg_peri'] = df_asteroids['arg_peri']/180*pi
    df_asteroids['anomaly'] = df_asteroids['anomaly']/180*pi
    df_asteroids['i'] = df_asteroids['i']/180*pi
    df_asteroids['n_obs'] = 0
    df_asteroids['step_obs'] = 0
    df_asteroids['last_obs'] = 0
    df_asteroids['detected'] = 0
    df_asteroids = df_asteroids.fillna(0.15)
    return df_asteroids

def init_asteroids_granvik(n_asteroids, stratified=False, random_state=9):
    df_asteroids = pd.read_csv('granvik_neo.csv')
    if stratified:
        bin_labels = [17.5, 18.5, 19.5, 20.5, 21.5, 
                      22.5, 23.5, 24.5]
        df_asteroids['bin'] = pd.cut(df_asteroids['H'],
                                     bins=8,
                                     labels=bin_labels)
        df_asteroids = (df_asteroids
                        .groupby('bin', group_keys=False)
                        .apply(lambda x: x.sample(int(n_asteroids/8), 
                                                  random_state=random_state,
                                                  replace=True)
                               )
                        )
        df_asteroids = df_asteroids.reset_index(drop=True)
    else:
        df_asteroids = df_asteroids.sample(n=n_asteroids, 
                                           random_state=random_state)
        
    df_asteroids['long_node'] = df_asteroids['long_node']/180*pi
    df_asteroids['arg_peri'] = df_asteroids['arg_peri']/180*pi
    df_asteroids['anomaly'] = df_asteroids['anomaly']/180*pi
    df_asteroids['i'] = df_asteroids['i']/180*pi
    df_asteroids['n_obs'] = 0
    df_asteroids['step_obs'] = 0
    df_asteroids['last_obs'] = 0
    df_asteroids['detected'] = 0
    df_asteroids['albedo'] = 0.15
    return df_asteroids

def init_asteroids_neopop(n_asteroids):
    df_asteroids = pd.read_csv("large_impactorPopulation.csv")
    if n_asteroids > 2450:
        warnings.warn("Number of asteroids exceeds available. Using 2450 instead")
    else:
        df_asteroids = df_asteroids.sample(n=n_asteroids, random_state=9)
    df_asteroids['long_node'] = df_asteroids['long_node']/180*pi
    df_asteroids['arg_peri'] = df_asteroids['arg_peri']/180*pi
    df_asteroids['anomaly'] = df_asteroids['anomaly']/180*pi
    df_asteroids['i'] = df_asteroids['i']/180*pi
    df_asteroids['n_obs'] = 0
    df_asteroids['step_obs'] = 0
    df_asteroids['last_obs'] = 0
    df_asteroids['detected'] = 0
    df_asteroids['detect_day'] = 0
    df_asteroids['impacted'] = 0
    return df_asteroids
    
def init_asteroids_granvik_impactors(n_asteroids, stratified=False):
    df_asteroids = pd.read_csv('granvik_impactors.csv')
    df_asteroids = df_asteroids.loc[df_asteroids['t_impact'] > 0]
    if stratified:
        bin_labels = [17.5, 18.5, 19.5, 20.5, 21.5, 
                      22.5, 23.5, 24.5]
        df_asteroids['bin'] = pd.cut(df_asteroids['H'],
                                     bins=8,
                                     labels=bin_labels)
        df_asteroids = (df_asteroids
                        .groupby('bin', group_keys=False)
                        .apply(lambda x: x.sample(int(n_asteroids/8), 
                                                  random_state=9,
                                                  replace=True)
                               )
                        )
        df_asteroids = df_asteroids.reset_index(drop=True)
    else:
        df_asteroids = df_asteroids.sample(n=n_asteroids, 
                                           random_state=9)
    df_asteroids['n_obs'] = 0
    df_asteroids['step_obs'] = 0
    df_asteroids['last_obs'] = 0
    df_asteroids['detected'] = 0
    df_asteroids['albedo'] = 0.15
    df_asteroids['detect_day'] = 0
    df_asteroids['impacted'] = 0
    return df_asteroids
    