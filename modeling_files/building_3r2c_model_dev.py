# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:37:07 2020

@author: rchintal
"""
import EKF_3r2c_formulation as EKF_f
import csv_data_processing as cdp
import numpy as np
import pandas as pd
import copy
from scipy import nanmean
import matplotlib.pyplot as plt
import math

class building_3r2c_model_dev():
    def __init__(
            self,
            data_obj,
            model_params
        ):
        self.room_states = EKF_f.EKF_3r2c_formulation(
            dt_hr = (1./60) * (data_obj.data_timestep),
            T_r = 22.,
            T_w = 22.,
            rc_params = model_params["rc_params"],
            df_data = data_obj.df_data,
        )
        self.mean_sq_error = 0


    def set_EKF_matrices(self, room_states):
        temp_std = 0.1 # Centigrade
        room_states.R = np.diag([temp_std**2])
        room_states.P[0,0] = 2.**2.
        room_states.P[1,1] = 2.**2.
        room_states.P[2,2] = 1000.
        room_states.P[3,3] = 1000.
        room_states.P[4,4] = 1000.
        room_states.P[5,5] = 1000.
        room_states.P[6,6] = 1000.
        room_states.P[7,7] = 0.
        
        return room_states

    def room_states_to_rc_params(self, x):
        rc_params = {}
        rc_params["C_r_inv"] = x[2][0]
        rc_params["R_re_inv"] = x[3][0] 
        rc_params["R_ra_inv"] = x[4][0]
        rc_params["C_w_inv"] = x[5][0]
        rc_params["R_ea_inv"] = x[6][0]
        rc_params["alpha"] = x[7][0]

        return rc_params
    
    def extract_inputs(self, df_data, idx):
        u_idx = df_data[['T_outdoor', 'Q_ghi', 'Q_load', 'Q_hvac']].iloc[idx]

        return u_idx

    def update_temp_and_covariance(self, room_states, z):
        temp_var = copy.deepcopy(room_states)
        temp_var.update(
            z, 
            room_states.HJacobian_at, 
            room_states.hx
        )
        room_states.x[0][0] = temp_var.x[0][0]
        room_states.x[1][0] = temp_var.x[1][0]
        room_states.P = temp_var.P

        return room_states
    
    def states_check(self, x):
        if self.room_states.x[1][0] <16:
            self.room_states.x[1][0] = 16 
        if self.room_states.x[1][0] > 38:
            self.room_states.x[1][0] = 38
        for k in np.arange(2,7):
            if k in [2,5] and x[k][0] < 0.0001:
                x[k][0] = 0.0001
            if k in [2,5] and self.room_states.x[k][0] > 10:
                x[k][0] = 10
            if k in [3,4,6] and self.room_states.x[k][0] < 0.000005 :
                x[k][0] = 0.000005

        return x

    def T_room_prediction_mse(
            self,
            pred_params,
            data_obj
        ):

        # data and prediction parameters
        data_obj.df_resample(pred_params["pred_timestep"]) # resample data if needed
        df_data = data_obj.df_data # dataframe
        dt_hr = (1./60) * pred_params["pred_timestep"] # timestep in hours
        training_days = pred_params["validation_days"] # duration of training data in days
        n_per_day = int(24./dt_hr) # timesteps per day
        n_hrzn = int(pred_params["hrzn_hours"]/dt_hr) # timesteps in horizon

        # initialize room_states for simulation over data
        room_states_sim = EKF_f.EKF_3r2c_formulation(
            dt_hr, 
            T_r = 22., 
            T_w = 22.,
            rc_params = pred_params["rc_params"],
            df_data =  data_obj.df_data)
        
        # store measured and predicted value at end of prediction
        T_measured_list, T_pred_list = [],[]
        
        # Compute predicted output t+hrzn_hrs|t for each timestep
        for i in range(training_days * n_per_day):
            
            # update covariance matrix and wall temperature at start of each day
            if i%n_per_day == 0:
                room_states_sim.P[0,0] = 2**2
                room_states_sim.P[1,1] = 2**2
                room_states_sim.P[2:,2:]  = 0
                room_states_sim.R = np.diag([0.1 **2])
                room_states_sim.x[1][0] = 0.5 * (
                    df_data.T_room.iloc[i]  + 
                    df_data.T_outdoor.iloc[i])
            
            # at each timestep update temperatures and covariance matrix
            T_current = df_data.T_room.iloc[i] # current measured temperature
            room_states_sim = self.update_temp_and_covariance(room_states_sim, T_current)
            
            # Checks to ensure wall temperature doesn't reach unreasonable values
            if room_states_sim.x[1][0] <16.:
                room_states_sim.x[1][0] = 16. 
            if room_states_sim.x[1][0] > 38.:
                room_states_sim.x[1][0] = 38.

            # room states for prediction horizon
            room_states_pred = copy.deepcopy(room_states_sim) 
            

            # iterate over prediction horizon
            for j in range(n_hrzn): 
                pred_ix = i + j  #prediction index
                u_j = self.extract_inputs(df_data, pred_ix) # inputs for prediction index

                # predict one-step-ahead
                room_states_pred.sim_predict(u_j)
                T_pred_j = room_states_pred.x[0][0]

            # append measured and predicted list at end of horizon
            T_measured_list.append(df_data.T_room.iloc[pred_ix + 1])
            T_pred_list.append(T_pred_j)

            #simulate one timestep
            u_sim = self.extract_inputs(df_data, i)
            room_states_sim.predict(u_sim)
            
        error_list = [
            T_measured_list[i] - 
            T_pred_list[i] for i in range(len(T_measured_list))
        ]
        error_sq =  [i ** 2 for i in error_list]
        return nanmean(error_sq)


    def system_identification(
            self, 
            sysid_params,
            sysid_data_obj
        ):

        #training parameters
        dt_hr = (1./60) * sysid_params["pred_timestep"]
        steps_per_day = int(24./dt_hr)
        training_days = sysid_params["training_days"]
        training_idxs = np.arange(
            (training_days + 1) * steps_per_day
        )
        validation_idxs = np.arange(
            training_idxs[-1] + 1,
           (training_days + sysid_params["validation_days"] + 2) * steps_per_day 
        )

        # get data objects
        sysid_data_obj.df_resample(sysid_params["pred_timestep"]) # resample if needed
        training_data_obj = copy.deepcopy(sysid_data_obj)
        training_data_obj.df_data = sysid_data_obj.df_data.iloc[training_idxs, :]
        validation_data_obj = copy.deepcopy(sysid_data_obj)
        validation_data_obj.df_data = sysid_data_obj.df_data.iloc[validation_idxs, :]

        # initialize room_states and covariance matrices with initial guesses
        room_states = copy.deepcopy(self.room_states)
        room_states = self.set_EKF_matrices(room_states)

        #iteration parameters and outputs
        pred_params = copy.deepcopy(sysid_params)
        mean_sq_error_list = []
        mean_sq_error_old = 1000.

        for day in np.arange(training_days):
            print('\n day: %s out of %s training days' % (day+1, training_days))

            for i in range(steps_per_day):
                
                # current simulation index and input/output data
                idx = day * steps_per_day + i
                u_idx = self.extract_inputs(training_data_obj.df_data, idx)
                T_room_current = training_data_obj.df_data.T_room.iloc[idx]
                
                # update covariance matrix and wall temperature at start of each day
                if i == 0: 
                    room_states.P[1,1] = 2.**2
                    room_states.P[1,2:] = 0
                    room_states.P[2:,1] = 0
                    room_states.x[1][0] = 0.5 * (
                        training_data_obj.df_data.T_room.iloc[idx]  + 
                        training_data_obj.df_data.T_outdoor.iloc[idx])

                # update all states every iteration
                room_states.update(
                    T_room_current, 
                    room_states.HJacobian_at, 
                    room_states.hx
                )
                pred_params["rc_params"] = self.room_states_to_rc_params(room_states.x)

                # if update has nan break. Also check reasonableness of states
                if np.isnan(self.room_states.x).any():
                    break
                room_states.x = self.states_check(room_states.x)
                        
                # Update model states if mean squared error is lowered else undo state changes
                mean_sq_error = self.T_room_prediction_mse(
                    pred_params, 
                    validation_data_obj
                )
                if mean_sq_error < mean_sq_error_old:
                    self.room_states = copy.deepcopy(room_states)
                    self.room_states.mean_sq_error = mean_sq_error_old
                    mean_sq_error_old = mean_sq_error
                    mean_sq_error_list.append(mean_sq_error)
                    print('updated mean_sq_error and j',mean_sq_error,i)
                else:
                    room_states = copy.deepcopy(self.room_states)
                    room_states.x[1][0] = 0.5 * (
                        training_data_obj.df_data.T_room.iloc[idx]  + 
                        training_data_obj.df_data.T_outdoor.iloc[idx])
                
                # Prediction step
                room_states.predict(u_idx)
                
            