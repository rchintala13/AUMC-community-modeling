# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 02:20:08 2020

@author: rchintal
"""
import pandas as pd
import numpy as np
import copy

class csv_data_processing():
    def __init__(self, building_number, data_timestep):
        data_file = (r'./data_files/' + 'building_' + 
             str(building_number) + '_' + str(data_timestep) + 
             'min.csv')
        self.df_data = pd.read_csv(data_file)
        self.df_cast_columns(
            {
                'Timestamp': {
                'vartype': 'datetime',
                'format':r'%m/%d %H:%M:%S'
                }
            }
        )
        self.df_data = self.df_data.set_index('Timestamp')
        self.data_timestep = data_timestep


    def df_cast_columns(self, dict_column_vartype):
        for key, val in dict_column_vartype.items():
            if key == 'Timestamp':
                # remove leading space from timestamp
                self.df_data['Timestamp'] = (
                    self.df_data['Timestamp'].replace(
                        r"^ +", r"", regex=True
                    )
                )
                # change EP 24:00:00 timemstamp to 1 second before
                self.df_data['Timestamp'] = (
                    self.df_data['Timestamp'].replace(
                        r"24:00:00", r"23:59:59", regex=True
                    )
                )
                # convert string to datetime
                self.df_data[key] = pd.to_datetime(
                    self.df_data[key], 
                    format = val['format'],
                )
                
    def df_resample(
            self,
            new_timestep,
    ):

        # resample data to new timestep
        self.data_timestep = new_timestep
        timestring = str(new_timestep) + 'T'
        self.df_data = self.df_data.resample(timestring).mean()
        
