# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:12:31 2022

@author: rchintal
"""

import csv_data_processing as cdp
import building_3r2c_model_development as gbm
import pickle
import numpy as np
import pandas as pd

class model_analysis_class():
        def __init__(self, pkl_file, data_obj):
            self.pkl_file = pkl_file
            self.data_obj = data_obj
            
        def select_model_index(self, model_list = None, \
                               data_obj = None, n_test_days = 7, \
                               n_test_start_day = 1, n_pred_test_list = [12]):
            if model_list == None: 
                model_list = pickle.load( open( self.pkl_file, "rb" ) )
            if data_obj == None: 
                data_obj = self.data_obj
            # Range of data used for testing
            time_step = data_obj.data_timestep
            id_strt = (1440/time_step) * (n_test_start_day - 1)
            id_end = id_strt + (n_test_days + 1) * (1440/time_step) 
            data_test_pd = data_obj.data_pd[id_strt:id_end]
            # find the model index that provides the least 
            error_list = []
            for model_test in model_list:
                j = 0
                for n_pred in n_pred_test_list:
                    output_list = model_test.T_room_prediction_test(\
                                  model_test.room_states.x, n_sim_days = \
                                  n_test_days, n_prediction_steps = \
                                  n_pred_test_list[j], EP_sim_data_pd = \
                                  data_test_pd)
                    mse_model = output_list[0]
                    if j == 0:
                        error_list.append([mse_model])
                    else:
                        error_list[-1].append(mse_model)
                    j += 1
            # print results
            mse_error_summary = np.array(error_list)
            mse_error_summary = mse_error_summary.transpose()
            column_headers =  ['model_' + str(k) \
                               for k in np.arange(len(model_list))]
            mse_summary_pd = pd.DataFrame(mse_error_summary, \
                                                columns = column_headers)
            print(mse_summary_pd)
            mse_summary_mean_array = mse_summary_pd.mean().values
            min_mse_id = np.argmin(mse_summary_mean_array)
            return(min_mse_id)
            
        def model_performance(self, model_test, data_obj = None, \
                              n_test_days = 7, n_test_start_day = 1, \
                              n_pred_test = 12, x_range = [4,6], \
                              y_range = [16,30]):
            if data_obj == None:
                data_obj = self.data_obj
            # Range of data used for testing
            time_step = data_obj.data_timestep
            id_strt = (1440/time_step) * (n_test_start_day - 1)
            id_end = id_strt + (n_test_days + 1) * (1440/time_step) 
            data_test_pd = data_obj.data_pd[id_strt:id_end]
            ##
            output_list = model_test.T_room_prediction_test(\
                                  model_test.room_states.x, n_sim_days = \
                                  n_test_days, n_prediction_steps = \
                                  n_pred_test, EP_sim_data_pd = \
                                  data_test_pd,plot_fig = True, \
                                      x_range = x_range, y_range = y_range)
            mse_model = output_list[0]
            return mse_model
        

                
        
            