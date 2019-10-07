#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 11:11:31 2019
@author: mashagubenko
"""

#501 Project Part 1
#This code 1) uses API method to pull population data (2014 - 2018) from census.gov, combines those datasets together, analyzes the new datasets cleanliness and outputs a new csv;
#2) cleans and outputs new csv files for the transit ridership data, vehicle registration data, electric vehicle registration data and population data. 

import pandas as pd
import os

import requests
import json
from urllib import request

#Main function that calls the rest of the functions 
def ProjData():
    
    DataProc() #calling the functino to process the data sets  
    
    CensusData = CensusAPI() #grabing the census population data using API method
    CleanScore(CensusData) #calling the functino to run the cleanliness score on the census data 
    
 #Function to call all of the data processing function except for the API pull    
def DataProc():
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
    
    df2 = EVData() #calling the function to read in and process the EV registration data 
    
    df3 = TransitData() #calling the function to read in and process the transit ridership data 
 
    PopData() #calling the function to merge population data 

#Vehicle Registration Data
#Data source: https://www.fhwa.dot.gov/policyinformation/quickfinddata/qfvehicles.cfm
#MV1 - state motor-vehicle registrations
    
def VehicleRegYear(year):  
    filename = str(year) + 'mv1.xls' #making file names fdependent on the year of the data    
    xls = pd.ExcelFile(filename) #reading the excel file    
    df = xls.parse(skiprows=11) #skiping the headers

    if df.shape[1] == 16: 
        #removing unnecessary columns
       df.columns = ['State', 'A1', 'A2', 'Automobiles', 'B1', 'B2', 'Buses','T1', 'T2', 'Trucks', 'AM1', 'AM2', 'AMT', 'Automobiles Per Capita', 'M1', 'M2']    
    else: 
        #removing unnecessary columns
        df.columns = ['State', 'A1', 'A2', 'Automobiles', 'B1', 'B2', 'Buses','T1', 'T2', 'Trucks', 'AM1', 'AM2', 'AMT', 'Automobiles Per Capita', 'M1', 'M2', 'E']    
           
    label = pd.Index(df['State']).get_loc('Wyoming') #idenitfying the position of last state in the file 
    remrows = len(df) - label - 1 #counting the number of rows to remove 
    df = df[0:-remrows] #removing the last rows of data which are just the total values for each column and footnotes
    
    #Making new data columns
    df['Motorcicles'] = df['M1'] + df['M2'] 
    df['Year'] = year    
    
    if df.shape[1] == 18:
        #removing unnecessary columns
        df = df.drop(['A1','A2','B1','B2','T1', 'T2','AM1','AM2','AMT','M1','M2'], axis = 1)
    else:
        #removing unnecessary columns
        df = df.drop(['A1','A2','B1','B2','T1', 'T2','AM1','AM2','AMT','M1','M2', 'E' ], axis = 1)
    
    return(df) #returning the dataframe

#Electric vehicle registration for the US by year
#Data Source: www.afdc.energy.gov/data

#Function to read in, clean and output a new csv file of the EV registration data 
def EVData():
    xls = pd.ExcelFile('EVRegistrations.xls') #reading the excel file    
    df = xls.parse(skiprows=1) #skiping the headers
    df = df[0:-8] #removing the last 8 rows whihc are the footnotes of the excel file 
    NewFileName = os.getcwd() + '/EV_Registrations_by_Type_US_by_Year.csv' #creating a path to the new file in the working directory
    df.to_csv(NewFileName)#saving it to a csv
    return(df)  #returning the dataframe 
    
#Transit Ridership Data
#Data Source: https://www.transit.dot.gov/ntd/data-product/monthly-module-adjusted-data-release

#Function to read in, clean and output a new csv file of the transit ridership data     
def TransitData():
    xls = pd.ExcelFile('Transit_Adjusted_Database.xls') #reading the excel file    
    df = xls.parse('MASTER') #reading in the data from the Master sheet  
    df = df.drop(['5 digit NTD ID', '4 digit NTD ID'], axis = 1) #droping the unnecessary columns    
    df = df.dropna() #droping missing values 
    NewFileName = os.getcwd() + '/Transit_Ridership_By_City_By_Year.csv' #creating a path to the new file in the working directory
    df.to_csv(NewFileName) #saving the dataframe to a csv file    
    return(df) #returning the dataframe 

#Population Data
#Data Source: https://www.census.gov/data/developers/data-sets/popest-popproj/popest.html
#API Key for census.gov is dc5d8d39d3afc513bfd6e27cad0dde26bc22f3f9
    
#Population data from 2013 until 2019

#Function to access the census.gov api and pull the necessary population data, combine the data into one dataframe and output it 
def CensusAPI():
    year = 2013 #setting up start year, the census.gov api deos not have apis set up for years before 2013
    res = pd.DataFrame(columns=['Year','Location','Population']) #setting up an empty dataframe for for the final results
        
    while year < 2019: #looping through all of the available years 
        #print(year)
        base = 'http://api.census.gov/data/' #base url
        
        if year > 2014: #the census.gov api uses different endpoint urls for different years of data, if the year is over 2014, it is the following 
            base2 = '/pep/population?get=GEONAME,POP&key=dc5d8d39d3afc513bfd6e27cad0dde26bc22f3f9&for=county:*'
        else : #for 2013 and 2014 it is the folloeing 
            base2 = '/pep/natstprc?get=STNAME,POP&DATE_=7&key=dc5d8d39d3afc513bfd6e27cad0dde26bc22f3f9&for=state:*'
        
        baseurl = base + str(year) + base2 #setting up the base URL to pull the API data
        
        if year == 2014 | year == 2013: 
            response=requests.get(baseurl) #using the request function to get the data
            jsontxt = response.json() #outputing it using json
            
            #looping through the json output to append a dataframe with values for each location 
            for i in range(1,len(jsontxt)):
                res = res.append({ #append the data frame with necessary values
                        'Year': year,
                        'Location': jsontxt[i][0],
                        'Population': jsontxt[i][1]}, ignore_index=True) 
            
        else:
            response=request.urlopen(baseurl) #using the request function to get the census data
            html_str = response.read().decode('utf-8') #decoding it using urllib 
            if html_str:
                jsontxt = json.loads(html_str)
                
                #looping through the json output to append a dataframe with values for each location 
                for i in range(1,len(jsontxt)):
                    res = res.append({ #append the data frame with necessary values 
                            'Year': year,
                            'Location': jsontxt[i][0],
                            'Population': jsontxt[i][1]}, ignore_index=True)  
          
        year += 1 
        
    res['Population'] = pd.to_numeric(res['Population']) #turning the population column to numeric
    NewFileName = os.getcwd() + '/Population_by_State_County_by_Year_2014_2018.csv' #creating a path to the new file in the working directory
    res.to_csv(NewFileName) #saving the dataframe down
    return(res)

#Function to produce the cleanliness score for the each of the attributes
def CleanScore(res):
    scores = [] #set up an empty array
    for col in res.columns: #loop through the columns
        if col != 'Location':
            a = sum(res[col]<0/len(res[col])) #count the number of invalid values 
        a = a + sum(res[col].isnull())/len(res[col]) #count percentage of values that are missing for each attribute
        scores.append(a) 
    return(scores) #return scores for each attribute 

#Function to pull, clean and save down population data for both state and city level 
def PopData():
    #State level population data from 2000 to 2010
    
    PopDataS1 = pd.read_csv('StateLevelPop20002010.csv' , sep=',', encoding='latin1') #read in the dataframe
    PopDataS1 = PopDataS1.drop(['REGION','DIVISION','STATE','SEX','ORIGIN','RACE','AGEGRP'], axis = 1) #drop unnecessary  columns
    PopDataS1.columns = map(str.lower, PopDataS1.columns) #make columsn lowercase
    PopDataS1 = PopDataS1[PopDataS1['name'] != 'United States'] #remove the rows showing total values
    
    NewFileName3 = os.getcwd() + '/Population_by_State_by_Year_2000_2010.csv' #creating a path to the new file in the working directory
    PopDataS1.to_csv(NewFileName3) #save the file down
    
    #City level population data from 2000 to 2018
    
    PopDataC1 = pd.read_csv('CityLevelPop20002010.csv' , sep=',', encoding='latin1') #read in the dataframe
    PopDataC1 = PopDataC1.drop(['SUMLEV','STATE','COUNTY','PLACE','COUSUB','CONCIT'], axis = 1) #drop unnecessary columns
    PopDataC1.columns = map(str.lower, PopDataC1.columns) #make columns lower case
    
    PopDataC2 = pd.read_csv('CityLevelPop20102018.csv' , sep=',', encoding='latin1') #read in the dataframe
    PopDataC2 = PopDataC2.drop(['SUMLEV','STATE','COUNTY','PLACE','COUSUB','CONCIT'], axis = 1) #drop unnecessary columns
    PopDataC2.columns = map(str.lower, PopDataC2.columns) #make columns lowercase
      
    PopDataC1 = PopDataC1.append(PopDataC2) #merge the dataframes   
    NewFileName4 = os.getcwd() + '/Population_by_City_by_Year.csv' #creating a path to the new file in the working directory
    PopDataC1.to_csv(NewFileName4)  #save the file down       

if __name__ == '__main__':
    ProjData()
