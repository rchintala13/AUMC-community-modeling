# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:49:07 2022

@author: rchintal
"""
import csv_data_processing as cdp
import building_3r2c_model_dev as gbm
import itertools
import pickle
import numpy as np
import copy

## Simulation Parameters for Pena Station Buildings
building_number = 47
data_timestep = 5
sysid_toggle = 1 # perform system identification if sysid_toggle = 1
analysis_toggle = 1 # perform pickle file analysis if value = 1

## System Identification Parameters
# Initialize rc params to 1
sysid_params = {
    "training_days":5,
    "validation_days":5,
    "pred_timestep":5,
    "hrzn_hours":1./6,
    "rc_params":{
    "C_r_inv":1.,
    "C_w_inv":1.,
    "R_re_inv":1.,
    "R_ra_inv":1.,
    "R_ea_inv":1.,
    "alpha": 1.,
    }
}

## initial guesses of 3r2c parameters
# sysid runs for all combinations of rc values
T_r = 22. # room air temperature
T_w = 22. # wall temperature
C_r = [ 0.148, 0.333,  0.01] # room air capacitance
C_w = [0.148, 0.5,  0.01] # exterior wall equivalent capacitance
R_re = [10.] # room air to outdoor air thermal reisstance 
R_ra = [10.] # room air to wall thermal resistance
R_ea = [10.] # wall to outdoor air thermal resistance
alpha = [1.0] # solar irradiance proportionality constant

## Sysid 
# import simulation data
EP_data_obj = cdp.csv_data_processing(building_number, data_timestep)
# create sysid data obj
sysid_data_obj = copy.deepcopy(EP_data_obj)
sysid_data_obj.df_resample(sysid_params["pred_timestep"])

# iterate over initial guesses
if sysid_toggle == 1:
    rc_combs = list(itertools.product(C_r,C_w, R_re, R_ra,R_ea,alpha))
    sysid_model_obj_list = []
    for i in np.arange(len(rc_combs)):       
        print('------------------------------------')
        print('Running model %s out of %s.' %(i+1, len(rc_combs)))
        print('------------------------------------')

        #Assign rc_comb values to sysid params
        sysid_params["rc_params"]["C_r_inv"] = 1./rc_combs[i][0]
        sysid_params["rc_params"]["C_w_inv"] = 1./rc_combs[i][1]
        sysid_params["rc_params"]["R_re_inv"] = 1./rc_combs[i][2]
        sysid_params["rc_params"]["R_ra_inv"] = 1./rc_combs[i][3]
        sysid_params["rc_params"]["R_ea_inv"] = 1./rc_combs[i][4]
        sysid_params["rc_params"]["alpha"] = rc_combs[i][5]

        #Create model object for each initial guess
        sysid_model_obj = \
            gbm.building_3r2c_model_dev(sysid_data_obj, sysid_params)
        sysid_model_obj.system_identification(sysid_params, sysid_data_obj)
        sysid_model_obj_list.append(sysid_model_obj)

    # store list of 3r2c models to pickle file
    pickle_file = r'.\pickle_files\\' + 'building_' + str(building_number) + \
        '.pkl'
    with open(pickle_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(sysid_model_obj_list, f)
