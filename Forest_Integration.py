#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:42:56 2019

@author: christopherfiaschetti
"""
import pandas as pd
import numpy as np

# Import Forest Data
Forest_Data = pd.read_csv('501Project1/Forest_Data.csv') 

# Drop region values
Forest_Data = Forest_Data.rename(columns={'Region': 'Name'})
Forest_Data = Forest_Data[Forest_Data.Name != 'Northeast Total       ']
Forest_Data = Forest_Data[Forest_Data.Name != 'North Central Total        ']
Forest_Data = Forest_Data[Forest_Data.Name != 'North Central Total        ']
Forest_Data = Forest_Data[Forest_Data.Name != 'North total']
Forest_Data = Forest_Data[Forest_Data.Name != 'Southeast Total        ']
Forest_Data = Forest_Data[Forest_Data.Name != 'South Central Total        ']
Forest_Data = Forest_Data[Forest_Data.Name != 'Pacific Northwest Total        ']
Forest_Data = Forest_Data[Forest_Data.Name != 'South total']
Forest_Data = Forest_Data[Forest_Data.Name != 'Rocky Mountain total:']
Forest_Data = Forest_Data[Forest_Data.Name != 'Pacific Coast total:']
Forest_Data = Forest_Data[Forest_Data.Name != 'Intermountain Total        ']
Forest_Data = Forest_Data[Forest_Data.Name != 'United States:     ']
Forest_Data = Forest_Data[Forest_Data.Name != 'Great Plains Total        ']
Forest_Data = Forest_Data[Forest_Data.Name != 'Pacific Southwest Total        ']


# reindex 
Forest_Data = Forest_Data.reset_index(drop=True)
# make all names matching
for i in range(len(Forest_Data.Name)):
    Forest_Data.Name[i] = Forest_Data.Name[i].strip() 
    
# drop years we aren't using
Forest_Data = Forest_Data.drop(['1997b','1987b','1977b','1963b','1953b','1938b','1920b','1907b','1630c'], axis=1)
dfnew["Forest Acerage(Thousands)"] =  np.nan

for i in range(len(dfnew)): # loop through all rows of big dataframe
    for j in range(len(Forest_Data)): # loop through all rows of forest_data
        for k in range(3): # loop through columns of Forest_data
            p = k+1 
            if (str(dfnew.Year[i]) == Forest_Data.columns[p] and dfnew.Name[i] == Forest_Data.iloc[j,0]):
                dfnew["Forest Acerage(Thousands)"][i] = Forest_Data.iloc[j,p]