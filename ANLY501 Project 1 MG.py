#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:11:31 2019

@author: mashagubenko
"""

#501 Project Part 1

#Vehicle Registration Data
#Data source: https://www.fhwa.dot.gov/policyinformation/quickfinddata/qfvehicles.cfm

#MV1 - state motor-vehicle registrations


import pandas as pd
import os

import requests
import json

def ProjData:

    df = VehicleRegYear(1995) #Pulling and cleaning vehicle registration data for 1995
    i = 1996 #start the year counter at 1996
    
    while i < 2015: #looping through the year for which the data are available
        df2 = VehicleRegYear(i) #calling the vehicle reg function
        df = df.append(df2) #appending the original dataframe with the new data
        i += 1 
    
    df = df.dropna() #droping a few empty rows
    df = df.reset_index() #reseting index
    
    NewFileName = os.getcwd() + '/Vehicle_Registration_By_State_By_Year.csv' #creating a path to the new file in the working directory
    df.to_csv(NewFileName) #saving the dataframe to a csv file
    


def VehicleRegYear(year):
    
    filename = str(year) + 'mv1.xls' #making file names fdependent on the year of the data
    
    xls = pd.ExcelFile(filename) #reading the excel file
    
    df = xls.parse(skiprows=11) #skiping the headers

    if df.shape[1] == 16:
       df.columns = ['State', 'A1', 'A2', 'Automobiles', 'B1', 'B2', 'Buses','T1', 'T2', 'Trucks', 'AM1', 'AM2', 'AMT', 'Automobiles Per Capita', 'M1', 'M2']    
    else: 
        df.columns = ['State', 'A1', 'A2', 'Automobiles', 'B1', 'B2', 'Buses','T1', 'T2', 'Trucks', 'AM1', 'AM2', 'AMT', 'Automobiles Per Capita', 'M1', 'M2', 'E']    
           
    label = pd.Index(df['State']).get_loc('Wyoming')
    remrows = len(df) - label - 1
    
    df = df[0:-remrows] #removing the last rows of data which are just the total values for each column and footnotes
    
    df['Motorcicles'] = df['M1'] + df['M2']
    df['Year'] = year    
    
    if df.shape[1] == 18:
        df = df.drop(['A1','A2','B1','B2','T1', 'T2','AM1','AM2','AMT','M1','M2'], axis = 1)
    else:
        df = df.drop(['A1','A2','B1','B2','T1', 'T2','AM1','AM2','AMT','M1','M2', 'E' ], axis = 1)
    
    return(df)
    
########### API CODE  

#Population Data
#Data Source: https://www.census.gov/data/developers/data-sets/popest-popproj/popest.html
#API Key for census.gov
#dc5d8d39d3afc513bfd6e27cad0dde26bc22f3f9


year = 2017 
base = 'http://api.census.gov/data/'
base2 = '/pep/population?'
baseurl = base + str(year) + base2 #setting up the base URL to pull the API data
urlpost = (('GEONAME', 'POP'), 
           {'key': 'dc5d8d39d3afc513bfd6e27cad0dde26bc22f3f9', #setting up each of the necessary parameters
                   'for': 'state:*' } )

response=requests.get(baseurl, urlpost) #using the request function to get the census data
jsontxt = response.json() #organizing the data into a readible text 
#jsontxt = json.loads(response.decode("utf-8"))
print(jsontxt)
 


'''
PopData1 = pd.read_csv('StateLevelPop20102018.csv' , sep=',', encoding='latin1') #read in the dataframe
PopData1 = PopData1.drop(['SUMLEV','REGION','DIVISION','STATE','SEX','ORIGIN','RACE','AGE'], axis = 1)
PopData1.columns = map(str.lower, PopData1.columns)

PopData2 = pd.read_csv('StateLevelPop20002010.csv' , sep=',', encoding='latin1') #read in the dataframe

PopDf = pd.DataFrame()
PopDf['State'] = set(PopData1['Name'])
for i in set(PopData1['Name']):
    for col in PopData1.columns:
        if col != 'Name':
            sum(PopData1[col][PopData1['Name'] == i])
    
'''            
        
if __name__ == '__main__':
    ProjData()
