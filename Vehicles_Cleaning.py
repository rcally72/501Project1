#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 02:44:26 2019

@author: christopherfiaschetti
"""

# Code to get rid of columns that contain missing values that shouldn't be there

import requests
import os
import csv
from csv import writer
import pandas as pd
import numpy as np

def main():
    dir = os.getcwd()
    os.chdir(dir)
    myData = pd.read_csv('Vehicles.csv' , sep=',', encoding='latin1')
    print(myData[:2])
    #print(myData.columns[myData.isna().any()].tolist()) # get columns with missing values
    myData = myData.drop(columns = "displ", axis=1)# remove columns deemed not needed 
    myData = myData.drop(columns = "drive", axis=1)
    myData = myData.drop(columns = "eng_dscr", axis=1)
    myData = myData.drop(columns = "guzzler", axis=1)
    myData = myData.drop(columns = "startStop", axis=1)
    myData = myData.drop(columns = "mfrCode", axis=1)
    myData = myData.drop(columns = "tCharger", axis=1)
    myData = myData.drop(columns = "sCharger", axis=1)
    myData = myData.drop(columns = "cylinders", axis=1)
    myData = myData.drop(columns = "trany", axis=1)
    myData = myData.drop(columns = "fuelType2", axis=1)
    myData = myData.drop(columns = "trans_dscr", axis=1)
    myData = myData.drop(columns = "atvType", axis=1)
    myData = myData.drop(columns = "c240bDscr", axis=1)
    myData = myData.drop(columns = "c240Dscr", axis=1)
    myData = myData.drop(columns = "createdOn", axis=1)
    myData = myData.drop(columns = "modifiedOn", axis=1)
    myData = myData.drop(columns = "rangeA", axis=1)
    myData = myData.drop(columns = "evMotor", axis=1)
    

    with pd.option_context('display.max_rows', None, 'display.max_columns', None): # print entire row of data
        dataRow = myData[:1] 
        with open('VehiclesC.csv','a') as fd: #get header in the dataset
                writer = csv.writer(fd)
                writer.writerow(dataRow)
              
        for i in range(len(myData)): # loop through all rows
            dataRow = myData.iloc[i] # get the specific row
            with open('VehiclesC.csv','a') as fd: 
                writer = csv.writer(fd)
                writer.writerow(dataRow)#print each row to dataset
                

if __name__ == "__main__":
    # execute only if run as a script
    main()