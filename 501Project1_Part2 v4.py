#obr#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:12:03 2019
@author: 501 Project
"""

# Import libraries
import pandas as pd
import csv
import numpy as np
from scipy import stats

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabasz_score
from sklearn import preprocessing
import pylab as pl
import sklearn.metrics as sm
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from matplotlib import cm

def Main():
    dfnew = MasterDataSet() #create master dataset 
    #StatAnalysis(dfnew) # statistical analysis part 1
    #CorrStat(dfnew) # statistical analysis part 2
    #Clustering(dfnew) # clustering analysis
    association_prep(dfnew) # Preprocess data for association rules
    encoding(dfnew) # Encode variables in preparation for association rules
    associationRULES(dfnew) # Conduct association rules mining
    #Hypo1(dfnew) # Run hypothesis testing 1
    #dfnew = PopB(dfnew) #creating new variable PopBin
    #Hypo2LR(dfnew) # Run hypothesis testing 2
    Hypo2RF(dfnew) # Run hypothesis testing 2 using Random Forest
    #Hypo3(dfnew) # Run hypothesis testing 3
    #Hypo4(dfnew)
    #Hypo5(Hypo5Data(dfnew),'MaxT') # Run hypothesis testing 5
    #Hypo5(Hypo5Data(dfnew),'MinT') # Run hypothesis testing 5
    
    H#ypo5(Hypo5Data(dfnew),'MaxT') # Run hypothesis testing 6
    #Hypo5(Hypo5Data(dfnew),'MinT') # Run hypothesis testing 6
    #Graphs(dfnew)
    

######### DATA PREPROCESSING ######################

# This function creates a master dataset of all our data 
def MasterDataSet():
    
    # Start the master data set by setting a dataframe of all states' 
    # populations by year 
    # Import the population info by State
    popstate = pd.read_csv('Population_by_State_by_Year_2000_2010.csv')
    popstate = popstate.drop(columns=['Unnamed: 0','census2010pop','estimatesbase2000'])
    colpop = popstate.columns
    
    new = pd.DataFrame(columns=['Name', 'Population', 'Year'])
    # Loop through every column and add it as time series information for each state
    for col in colpop[1:len(colpop)]:
        df = popstate[['name',col]]
        df = df.groupby('name',as_index=False).max()
        df = df.rename(columns={col: 'Population', 'name':'Name'})
        df['Year'] = int(col[len(col)-4:len(col)])
        new = pd.concat([new,df],ignore_index=True) #combine the dataframes into one of all years
        
    # Import the population info by State part 2
    popstate = pd.read_csv('nst-est2018-alldata.csv')
    # Loop through every column and add it as time series information for each state
    popstate = popstate[['NAME', 'POPESTIMATE2011',
           'POPESTIMATE2012', 'POPESTIMATE2013', 'POPESTIMATE2014',
           'POPESTIMATE2015', 'POPESTIMATE2016', 'POPESTIMATE2017',
           'POPESTIMATE2018']]
    popstate = popstate.drop(popstate.index[[0,1,2,3,4]])
    colpop = popstate.columns
    
    new2 = pd.DataFrame(columns=['Name', 'Population', 'Year'])
    for col in colpop[1:len(colpop)]:
        df = popstate[['NAME',col]]
        df = df.groupby('NAME',as_index=False).max()
        df = df.rename(columns={col: 'Population', 'NAME':'Name'})
        df['Year'] = int(col[len(col)-4:len(col)])
        new2 = pd.concat([new2,df],ignore_index=True) #combine the dataframes into one of all years
    
    # Combine the two parts of population datasets into one
    new = pd.concat([new,new2],ignore_index=True)    
    
    # List of all states to change abbreviations to full names
    states = {
            'AK': 'Alaska',
            'AL': 'Alabama',
            'AR': 'Arkansas',
            'AS': 'American Samoa',
            'AZ': 'Arizona',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District of Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'GU': 'Guam',
            'HI': 'Hawaii',
            'IA': 'Iowa',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'MA': 'Massachusetts',
            'MD': 'Maryland',
            'ME': 'Maine',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MO': 'Missouri',
            'MP': 'Northern Mariana Islands',
            'MS': 'Mississippi',
            'MT': 'Montana',
            'NA': 'National',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'NE': 'Nebraska',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NV': 'Nevada',
            'NY': 'New York',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'PR': 'Puerto Rico',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VA': 'Virginia',
            'VI': 'Virgin Islands',
            'VT': 'Vermont',
            'WA': 'Washington',
            'WI': 'Wisconsin',
            'WV': 'West Virginia',
            'WY': 'Wyoming'
    }
    
    
    # Import the transit data
    # Annual
    # Years 2002 - 2017 (most data for 2017)
    # Location as Lat/Long and City Name
    # Add to the master data set each piece of info from the transit data set by state by year 
    transit = pd.read_csv('TransitwithLatLong.csv')
    stat = transit['HQ State'].apply(lambda x: states[x]) #create a list of all states
    transit['HQ State'] = stat #add a column of all states
    #Total trips by state by year
    a = transit.groupby(['HQ State', 'FY End Year'], as_index = False)['Unlinked Passenger Trips FY'].sum()
    a = a.rename(columns={'FY End Year': 'Year', 'HQ State':'Name', 'Unlinked Passenger Trips FY': 'Passenger Trips'})
    #Largest service area by state by year
    b = transit.groupby(['HQ State', 'FY End Year'], as_index = False)['Service Area SQ Miles'].max()
    b = b.rename(columns={'FY End Year': 'Year', 'HQ State':'Name', 'Service Area SQ Miles': 'Largest Transit Service Area'})
    #Total trips by state by year
    c = transit.groupby(['HQ State', 'FY End Year'], as_index = False)['Operating Expenses FY'].sum()
    c = c.rename(columns={'FY End Year': 'Year', 'HQ State':'Name'})
    
    # Add transit info to the master dataframe by state and year
    dfnew = pd.merge(new, a, how='left', on=['Name', 'Year'])
    dfnew = pd.merge(dfnew, b, how='left', on=['Name', 'Year'])
    dfnew = pd.merge(dfnew, c, how='left', on=['Name', 'Year'])
    
    # Import the vehicle registration data
    # Annual by State
    # Location as state
    # Years 1995 - 2014
    # Add the data to the master dataset by state by year 
    vehreg = pd.read_csv('Vehicle_Registration_By_State_By_Year.csv')
    vehreg = vehreg.drop(columns=['Unnamed: 0', 'index'])
    vehreg = vehreg.rename(columns = {'State': 'Name', 'Motorcicles': 'Motorcycles'})
    # Conform state names
    vehreg.Name = vehreg.Name.str.replace(r'[\(\)\d]+', '')
    vehreg.Name = vehreg.Name.str.replace('/','')
    vehreg = vehreg.replace('Dist. of Col.', 'District of Columbia')
    
    # Clean the state names information for state names neding in ' ' or '  '
    for k in vehreg.Name:
        if k[len(k)-1] == ' ':
            if k[len(k)-2] == ' ':
                vehreg = vehreg.replace(k, k[0:len(k)-2])
            else:
                vehreg = vehreg.replace(k, k[0:len(k)-1])
                
    # Add to the master dataframe by state and year
    dfnew = pd.merge(dfnew, vehreg, how='left', on=['Name', 'Year'])
    
    # Import the data set on electric vehicle registrations 
    # Annual 
    # Aggregated for the entire country by vehicle type & year so location is US
    # Years: 1999-2017
    evreg = pd.read_csv('EV_Registrations_by_Type_US_by_Year.csv')
    evreg = evreg.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
    evreg = evreg.replace('U',0) #replace empty values with 0s 
    colev = evreg.columns
    
    # Add the ev registration data to the master dataset
    # Total number of ev registrations in the US is added to each state for each type of ev
    dfnew['Hybrid'] = 0 
    dfnew['PlugHybrid'] = 0 
    dfnew['Electric'] = 0 
    # Add a column of total values by state by year 
    for col in colev:
        dfnew['Hybrid'].mask(dfnew['Year']== int(col), evreg[col][0], inplace=True)
        dfnew['PlugHybrid'].mask(dfnew['Year']== int(col), evreg[col][1], inplace=True)
        dfnew['Electric'].mask(dfnew['Year']== int(col), evreg[col][2], inplace=True)
    
    # Estimate state level values by dividing the country totals by population 
    dfnew['Hybrid'] = pd.to_numeric(dfnew['Hybrid'])
    dfnew['PlugHybrid'] = pd.to_numeric(dfnew['PlugHybrid'])
    dfnew['Electric'] = pd.to_numeric(dfnew['Electric'])
    dfnew['Hybrid'] = dfnew['Hybrid']/dfnew['Population']
    dfnew['PlugHybrid'] = dfnew['PlugHybrid']/dfnew['Population']
    dfnew['Electric'] = dfnew['Electric']/dfnew['Population']
    
    # Import the data on airport location
    # Add number of airports by type by state to the master dataset
    airloc = pd.read_csv('AirportLocation.csv') 
    # Conform state names
    airloc['Location'] = airloc['Location'].str.lower()
    airloc['Location'] = airloc['Location'].str.title()
    airloc = airloc.replace('District Of Columbia', 'District of Columbia')
    airloc = airloc[airloc['type'] != 'closed'] #drop the rows where airports have been closed
    airloc = airloc.rename(columns={'Location': 'Name'})
    
    # Split into dataframes for each airport type
    a1 = airloc[airloc['type'] == 'balloonport']
    a2 = airloc[airloc['type'] == 'heliport']
    a3 = airloc[airloc['type'] == 'large_airport']
    a4 = airloc[airloc['type'] == 'medium_airport']
    a5 = airloc[airloc['type'] == 'seaplane_base']
    a6 = airloc[airloc['type'] == 'small_airport']
    
    # Aggregate the totals of each type of airprot by state by year 
    # And merge with the master data frame
    a11 = a1.groupby(['Name'], as_index = False).size().to_frame('Balloonport')
    dfnew = pd.merge(dfnew, a11, how='left', on=['Name'])
    a21 = a2.groupby(['Name'], as_index = False).size().to_frame('Heliport')
    dfnew = pd.merge(dfnew, a21, how='left', on=['Name'])
    a31 = a3.groupby(['Name'], as_index = False).size().to_frame('Large Airport')
    dfnew = pd.merge(dfnew, a31, how='left', on=['Name'])
    a41 = a4.groupby(['Name'], as_index = False).size().to_frame('Medium Airport')
    dfnew = pd.merge(dfnew, a41, how='left', on=['Name'])
    a51 = a5.groupby(['Name'], as_index = False).size().to_frame('Seaplane Base')
    dfnew = pd.merge(dfnew, a51, how='left', on=['Name'])
    a61 = a6.groupby(['Name'], as_index = False).size().to_frame('Small Airport')
    dfnew = pd.merge(dfnew, a61, how='left', on=['Name'])
    
    # Import the urban land data set
    urbland = pd.read_csv('Urban_Land(2000-2010).csv') 
    urbland = urbland.drop(urbland.index[[0,9]]) # Drop totals
    
    # Set up lists of states belonging to each region
    NE = ['Connecticut', 'Delaware', 'Maine', 'Massachusetts', 
          'Maryland', 'New Hampshire', 'New Jersey', 'New York', 'Pennsylvania', 
          'Rhode Island', 'Vermont']
    
    SE = ['Alabama', 'Florida', 'Georgia', 'Kentucky', 
          'Mississippi', 'North Carolina', 'South Carolina', 'Tennessee', 
          'Virginia', 'West Virginia']
    
    PS = ['California', 'Arizona', 'Hawaii', 'Utah', 'Nevada']
    
    NC = ['Iowa', 'Kansas', 'Minnesota', 'Missouri', 
          'Nebraska', 'North Dakota', 'South Dakota']
    
    SC = ['Arkansas', 'Louisiana', 'Oklahoma', 'Texas']
    
    RM = ['Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico']
    
    GP = ['Montana', 'North Dakota', 'South Dakota', 'Nebraska', 'Oklahoma']
    
    PN = ['Montana', 'California', 'Wyoming', 'Oregon', 'Washington']
    
    # Set up columns corresponding to each of the region in the master dataset
    reg = ['NE','SE','PS','NC','SC','RM','GP','PN']
    for lst in reg:
        dfnew[lst] = 0
    
    # Loop through the lists of regions and identify states that belong to 
    # each region by assigning 1 to that column    
    reglist = [NE,SE,PS,NC,SC,RM,GP,PN]       
    i = 0 
    while i < len(reglist):  
        #print(reg[i])
        for j in range(0,len(dfnew['Name'])):
            if dfnew['Name'][j] in reglist[i]:
                dfnew[reg[i]][j] = 1
        i += 1
     
    # Pull the data from the urban data set on growth % of urban land by each region
    # Add the total % increase to each state belonging to the region in the master dataset    
    dfnew['Percent increase in urban land 2000-2010'] = 0
    for i in range(0, len(reg)-1):
        dfnew['Percent increase in urban land 2000-2010'].mask(dfnew[reg[i]] == 1, urbland['Percent increase in urban land 2000-2010'][i+1] , inplace=True)
        
    # Import the data set on alterantive fuel stations
    # Type of fuel by station information
    # Count the number of alterantive fuel facilities by state and add it to the master dataset
    altfuel = pd.read_csv('AlternativeFuelStationLocation.csv')
    # Conform state names
    altfuel['Location'] = altfuel['Location'].str.lower()
    altfuel['Location'] = altfuel['Location'].str.title()
    altfuel = altfuel.replace('District Of Columbia', 'District of Columbia')
    altfuel = altfuel.rename(columns={'Location': 'Name'})
    alt1 = altfuel.groupby(['Name'], as_index = False).size().to_frame('Number of Alt Fuel Facilities')
    dfnew = pd.merge(dfnew, alt1, how='left', on=['Name'])
    
    # Import the facility pollution data set
    # Count the number of factories by state and add it to the master data set
    facpol = pd.read_csv('FacilityPollution.csv')
    # Conform state names
    facpol['Location'] = facpol['Location'].str.lower()
    facpol['Location'] = facpol['Location'].str.title()
    facpol = facpol.replace('District Of Columbia', 'District of Columbia')
    facpol = facpol.rename(columns={'Location': 'Name'})
    fp1 = facpol.groupby(['Name'], as_index = False).size().to_frame('Number of Factories')
    dfnew = pd.merge(dfnew, fp1, how='left', on=['Name'])
    
    
    # Read in temperature data that contains full list of lat-longs
    temps_old = pd.read_csv("temps_old.csv")
    
    # Save full list of coordinates
    geos_old = temps_old[['lat','long']]
    geos_old = geos_old.drop_duplicates()
    states = []
    
    ## This section pairs the full list of states with the full lat-long list
        
    # Save full state list back into Python variable
    with open('states.csv', 'r') as f:
      reader = csv.reader(f)
      statelist = list(reader)
    
    # Convert list to array to enable its orientation to be flipped from row to column
    statelist = np.array(statelist)
    # Flip array to column
    statelist = statelist.reshape(-1, 1)
    # Remove null value that ensued from flip
    statelist = statelist[statelist != '']
    
    # Add column of corresponding state names into coordinates dataframe
    geos_old["Name"] = statelist
    
    ## This section pairs the lat-long list we are actually using with its corresponding states list
    
    # Read in temperature data that we are actually using
    temps = pd.read_csv("temps2.csv")
    
    # Save list of states corresponding to coordinates we are actually using
    with open('statelist_final.csv', 'r') as f:
      reader = csv.reader(f)
      statelist_final = list(reader)
    
    # Convert list to array to enable its orientation to be flipped from row to column
    statelist_final = np.array(statelist_final)
    # Flip array to column
    statelist_final = statelist_final.reshape(-1, 1)
    # Remove null value that ensued from flip
    statelist_final = statelist_final[statelist_final != '']
    
    # Save coordinates of points we are actually using
    geos = temps[['lat','long']]
    geos = geos.drop_duplicates()
    
    # Merge full lat-long-state record with lat-long pairs we are using for analysis,
    # leaving only the lat-long-state sets we are actually using
    geoo = pd.merge(geos, geos_old, how='left', on=['lat','long'])
    geoo = geoo.dropna()
    
    # Add column of state names to coordinates we are using for analysis
    geoo['Name'] = statelist_final
    
    # Add state name column to temperatures dataframe
    temperatures = temps.merge(geoo, how = 'inner', on = ['lat', 'long'])
    
    # Define subset dataframe that contains only the numeric temperature data columns
    temp_sub = temperatures[['jan','apr','jul','oct']]
    # Calculate min, max, and mean for each year for each location. Don't include zero values
    # in consideration.
    temperatures['max'] = temp_sub[temp_sub!=0].max(axis=1)
    temperatures['min'] = temp_sub[temp_sub!=0].min(axis=1)
    temperatures['mean'] = temp_sub[temp_sub!=0].mean(axis=1)
    
    # Define lists of the unique years and state names in the dataset
    unique_states = temperatures['Name'].unique()
    unique_years = temperatures['year'].unique()
    
    # Columns that we will eventually add to master dataset
    columns = ['Name','Year','MinT','MaxT','MeanT']
    # Dataframe that we will merge with master dataset
    summary = pd.DataFrame(columns = columns)
    
    # For each state...
    for state in range(0,len(unique_states)):
        
        # Save dataframe that contains only the rows from this state
        matrix = temperatures[temperatures['Name'] == unique_states[state]]
        
        # For each year...
        for year in unique_years:
            
            # Define row that will store this row-year pair's data
            row = pd.DataFrame(columns = columns, index=[0])
            #Label row with state name and year
            row['Name'] = unique_states[state]
            row['Year'] = year
            
            # Define subset of this state's matrix that only contains rows from this year
            new = matrix[matrix['year']==year]
            
            # Calculate mean, min, and max
            row['MeanT'] = new['mean'].mean()
            row['MinT'] = new['min'].mean()
            row['MaxT'] = new['max'].mean()
        
            # Append row containing this state-year pair's state name, year, mean, min, and max
            summary = summary.append(row)
    
    # Merge with the master dataset by state and year
    dfnew = pd.merge(dfnew, summary, how='left', on=['Name', 'Year'])
    
    # Import Forest Data
    Forest_Data = pd.read_csv('Forest_Data.csv') 
    
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
    dfnew["Forest Acreage(Thousands)"] =  np.nan
    
    for i in range(len(dfnew)): # loop through all rows of big dataframe
        for j in range(len(Forest_Data)): # loop through all rows of forest_data
            for k in range(3): # loop through columns of Forest_data
                p = k+1 
                if (str(dfnew.Year[i]) == Forest_Data.columns[p] and dfnew.Name[i] == Forest_Data.iloc[j,0]):
                    dfnew["Forest Acreage(Thousands)"][i] = Forest_Data.iloc[j,p]
    
    
    # Convert ints and floats improperly saved as strings to numeric variables
    for i in range(0,len(dfnew)):
        dfnew['Percent increase in urban land 2000-2010'][i] = float(dfnew['Percent increase in urban land 2000-2010'][i])

        if type(dfnew['Forest Acreage(Thousands)'][i]) == str:
            dfnew['Forest Acreage(Thousands)'][i] = dfnew['Forest Acreage(Thousands)'][i].replace(',','')
            dfnew['Forest Acreage(Thousands)'][i] = int(dfnew['Forest Acreage(Thousands)'][i])

    
    
    return(dfnew)


########### STATISTICAL ANALYSIS #######################

# Part 1 - population - mean, median,mode,stdev
def StatAnalysis(dfnew):                
    pop = np.array(dfnew.Population)
    
    print(dfnew.columns)
    print('Population')
    print(np.mean(pop))
    print(np.median(pop)) 
    print(stats.mode(pop, axis=None))#
    print(np.std(pop))   
    plt.subplots(1, 1) 
    plt.hist(pop)
    plt.title('Population')
    
    # Part 1 - Automobiles - mean, median,mode,stdev
    auto = np.array(dfnew.Automobiles)
    auto = auto[~np.isnan(auto)]
    print('automobiles')
    print(np.mean(auto))
    print(np.median(auto)) 
    print(stats.mode(auto, axis=None))#
    print(np.std(auto))  
    plt.subplots(1, 1) 
    plt.hist(auto)
    plt.title('Autmobiles')
    
    # Part 1 - Trucks - mean, median,mode,stdev
    Truck = np.array(dfnew.Trucks)
    Truck = Truck[~np.isnan(Truck)]
    
    print('Trucks')
    print(np.mean(Truck))
    print(np.median(Truck)) 
    print(stats.mode(Truck, axis=None))#
    print(np.std(Truck)) 
    plt.subplots(1, 1) 
    plt.hist(Truck)
    plt.title('Trucks')
    
    # Part 1 - Buses - mean, median,mode,stdev
    Buses = np.array(dfnew.Buses)
    Buses = Buses[~np.isnan(Buses)]
    
    print('Buses')
    print(np.mean(Buses))
    print(np.median(Buses)) 
    print(stats.mode(Buses, axis=None))#
    print(np.std(Buses)) 
    plt.subplots(1, 1) 
    plt.hist(Buses)
    plt.title('Buses') 
    
    # Part 1 - Trucks - mean, median,mode,stdev
    Moto = np.array(dfnew.Motorcycles)
    Moto = Moto[~np.isnan(Moto)]
    print('Motorcycles')
    print(np.mean(Moto))
    print(np.median(Moto)) 
    print(stats.mode(Moto, axis=None))#
    print(np.std(Moto)) 
    plt.subplots(1, 1) 
    plt.hist(Moto)
    plt.title('Motorcycles') 
    
    # Part 1 - Large Airports - mean, median,mode,stdev
    LAir = np.array(dfnew['Large Airport'])
    LAir = LAir[~np.isnan(LAir)]
    print('Large Airports')
    print(np.mean(LAir))
    print(np.median(LAir)) 
    print(stats.mode(LAir, axis=None))#
    print(np.std(LAir)) 
    plt.subplots(1, 1) 
    plt.hist(LAir)
    plt.title('Large Airports') 
    
    # Part 1 - Small Airports - mean, median,mode,stdev
    SAir = np.array(dfnew['Small Airport'])
    SAir = SAir[~np.isnan(SAir)]
    print('Small Airports')
    print(np.mean(SAir))
    print(np.median(SAir)) 
    print(stats.mode(SAir, axis=None))#
    print(np.std(SAir)) 
    plt.subplots(1, 1) 
    plt.hist(SAir)
    plt.title('Small Airports') 
    
    # Part 1 - Medium Airports - mean, median,mode,stdev
    MAir = np.array(dfnew['Medium Airport'])
    MAir = MAir[~np.isnan(MAir)]
    print('Medium Airports')
    print(np.mean(MAir))
    print(np.median(MAir)) 
    print(stats.mode(MAir, axis=None))#
    print(np.std(MAir))  
    plt.subplots(1, 1) 
    plt.hist(MAir)
    plt.title('Medium Airports') 
    
    # Part 1 - Temperatures- mean, median,mode,stdev
    MT = np.array(dfnew['MeanT'])
    MT = MT[~np.isnan(MT)]
    print('Temps')
    print(np.mean(MT))
    print(np.median(MT)) 
    print(stats.mode(MT, axis=None))#
    print(np.std(MT))  
    plt.subplots(1, 1) 
    plt.hist(MT)
    plt.title('Mean Temps') 
    
    # Part 1 - Number of Factories  - mean, median,mode,stdev
    NFacts = np.array(dfnew['Number of Factories'])
    NFacts = NFacts[~np.isnan(NFacts)]
    print('Num Factories')
    print(np.mean(NFacts))
    print(np.median(NFacts)) 
    print(stats.mode(NFacts, axis=None))#
    print(np.std(NFacts))  
    plt.subplots(1, 1) 
    plt.hist(NFacts)
    plt.title('Factories') 

# Part 2 - Correlation and scatterplots

def CorrStat(dfnew):
    # Check population and factories
    data = dfnew[['Population', 'Number of Factories']] #get desired data
    PandF = pd.DataFrame(data)
    PandF = PandF.dropna()
    PandF.Population = pd.to_numeric(PandF.Population)
    print(PandF.corr(method ='pearson'))
    plt.style.use('ggplot')
    plt.subplots(1, 1) 
    plt.scatter(PandF.Population, PandF['Number of Factories'])
    plt.title('Population vs Factories') 
    plt.show()
    
    # Check population and Temperature
    data = dfnew[['MeanT', 'Population']]#get desired data
    PandT = pd.DataFrame(data)
    PandT = PandT.dropna()
    PandT.Population = pd.to_numeric(PandT.Population)
    print(PandT.corr(method ='pearson'))
    plt.style.use('ggplot')
    plt.subplots(1, 1) 
    plt.scatter(PandT.Population, PandT.MeanT)
    plt.title('Population vs Temp') 
    plt.show()
    
    # Check Factories and Temperature
    data = dfnew[['MeanT', 'Number of Factories']]#get desired data
    TandF = pd.DataFrame(data)
    TandF = TandF.dropna()
    print(TandF.corr(method ='pearson'))
    plt.style.use('ggplot')
    plt.subplots(1, 1) 
    plt.scatter(TandF['Number of Factories'], TandF.MeanT)
    plt.title('Factories vs Temp') 
    plt.show()
    
    #additional correlation checks
    data = dfnew[['Population', 'Automobiles']]#get desired data
    TandF = pd.DataFrame(data)
    TandF = TandF.dropna()
    TandF.Population = pd.to_numeric(TandF.Population)
    print(TandF.corr(method ='pearson'))
    
    data = dfnew[['Population', 'Trucks']]#get desired data
    TandF = pd.DataFrame(data)
    TandF = TandF.dropna()
    TandF.Population = pd.to_numeric(TandF.Population)
    print(TandF.corr(method ='pearson'))
    
    data = dfnew[['Population', 'Motorcycles']]#get desired data
    TandF = pd.DataFrame(data)
    TandF = TandF.dropna()
    TandF.Population = pd.to_numeric(TandF.Population)
    print(TandF.corr(method ='pearson'))
    
    data = dfnew[['Population', 'Buses']]#get desired data
    TandF = pd.DataFrame(data)
    TandF = TandF.dropna()
    TandF.Population = pd.to_numeric(TandF.Population)
    print(TandF.corr(method ='pearson'))
    
    data = dfnew[['Population', 'Large Airport']]#get desired data
    TandF = pd.DataFrame(data)
    TandF = TandF.dropna()
    TandF.Population = pd.to_numeric(TandF.Population)
    print(TandF.corr(method ='pearson'))
    
    data = dfnew[['Population', 'Small Airport']]#get desired data
    TandF = pd.DataFrame(data)
    TandF = TandF.dropna()
    TandF.Population = pd.to_numeric(TandF.Population)
    print(TandF.corr(method ='pearson'))
    
    data = dfnew[['Population', 'Medium Airport']]#get desired data
    TandF = pd.DataFrame(data)
    TandF = TandF.dropna()
    TandF.Population = pd.to_numeric(TandF.Population)
    print(TandF.corr(method ='pearson'))
    
    data = dfnew[['Number of Factories', 'Trucks']]#get desired data
    TandF = pd.DataFrame(data)
    TandF = TandF.dropna()
    print(TandF.corr(method ='pearson'))
    
    # Check Alternative Fueling stations and Temperature
    data = dfnew[['MeanT', 'Number of Alt Fuel Facilities']]
    TandAFF = pd.DataFrame(data)
    TandAFF = TandAFF.dropna()
    print(TandAFF.corr(method ='pearson'))
    plt.style.use('ggplot')
    plt.subplots(1, 1) 
    plt.scatter(TandAFF['Number of Alt Fuel Facilities'], TandAFF.MeanT)
    plt.title('Alternative Fueling Facilities vs Temp') 
    plt.show()

########### CLUSTERING ANALYSIS #######################

# Picking 3 scenarios to cluster (different attributes) using all 3 clustering techniques.
def Clustering(dfnew):    
    Scenario1(dfnew) #For Scenario 1, lets consider the following attributes: Name, Population, MinT, and MaxT.
    Scenario2(dfnew) #For Scenario 2, lets consider the following attributes: Regions and % increase in urban land.
    Scenario3(dfnew) #For Scenario 3, lets consider the following attributes: Name, Temperature, Automobiles Per Capita
    Scenario4(dfnew)  #For Scenario 4, lets consider the following attributes: Name, Number of Factories, MinT, MaxT.
    Dbscan(dfnew) #Overall Dataset

def Scenario1 (dfnew) :
    print("For Scenario 1, lets consider the following attributes: Name, Population, MinT, and MaxT.")
    print()   
    print("Results for Hierarchical Clustering:")
    print()
    dfnew.fillna(method ='ffill', inplace = True) 
    dfnew.Name = dfnew.Name.astype("category").cat.codes
    X=dfnew.iloc[:,[0,1,31,32]].values
    labelencoder_X= LabelEncoder ()
    X[:,1]=labelencoder_X.fit_transform(X[:,1])
   
    Y= pd.DataFrame(X)
    #Creating a dendogram to find the optimal number of clusters for the columns mentioned above.
    print("Dendrogram: Using linkage type 'Ward':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'ward')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Ward:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Complete':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'complete')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Complete:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Average':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'average')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Average :')
    plt.show()
    plt.clf()
    # Fitting H.Clustering to the dataset
    
    #TYPE 1= Affinity: Euclidean and Linkage Type: Ward
    hc= AgglomerativeClustering(n_clusters=2, affinity= 'euclidean', linkage = 'ward')
    y_hc= hc.fit_predict(X)
    accuracy1 = sm.accuracy_score(y_hc, hc.labels_)
    print("The Accuracy Score for Type 1 is : " + str(accuracy1))
    
    #TYPE 2= Affinity: Euclidean and Linkage Type: Complete
    hc2= AgglomerativeClustering(n_clusters=2, affinity= 'euclidean', linkage = 'complete')
    y_hc2= hc2.fit_predict(X)
    accuracy2 = sm.accuracy_score(y_hc2, hc2.labels_)
    print("The Accuracy Score for Type 2 is : " + str(accuracy2))
    
    #TYPE 3= Affinity: Manhattan and Linkage Type: Average
    hc3= AgglomerativeClustering(n_clusters=2, affinity= 'manhattan', linkage = 'average')
    y_hc3= hc3.fit_predict(X)
    accuracy3 = sm.accuracy_score(y_hc3, hc3.labels_)
    print("The Accuracy Score for Type 3 is : " + str(accuracy3))
    
    ####Lets visualize the clusters######
    plt.scatter(X[y_hc== 0,0], X[y_hc== 0,1], s=5, c= 'blue', label= 'Cluster 1')
    plt.scatter(X[y_hc== 1,0], X[y_hc== 1,1], s=5, c= 'red', label= 'Cluster 2')
    plt.title('Graphing The Cluster for Scenario 1')
    plt.legend()
    plt.show()

    print("Results for KMEANS Clustering:")
    print()
    # First remove missing data
    nullCount = dfnew.isnull().values.ravel().sum()
    #print("\nNull Count:\n");
    pprint(nullCount)
    dfnew.dropna()
    #pprint(len(dfnew.index))
    #Conversion to numerical 
    dfnew['Name'] = pd.Categorical(dfnew['Name'])
    dfnew['Name'] = dfnew['Name'].cat.codes
    dfnew.shape[1]
    dfnew=pd.concat([dfnew['Name'], dfnew['Population'], dfnew['MinT'], dfnew['MaxT']], 
                 axis=1, keys=['Name', 'Population', 'MinT', 'MaxT'])
    x = dfnew.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    #pprint(normalizedDataFrame[:10])
    # Create clusters (TEST 1)
    k = 500
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    k2 = 100
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    
    # PCA
    print("PCA projection:")
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    
    # This shows how good the PCA performs on this dataset
    print(pca2D.explained_variance_)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title('PCA Graph for Scenario 1')
    plt.show()
    
    # Clear plot
    plt.clf()
    
def Scenario2 (dfnew):
    print("For Scenario 2, lets consider the following attributes: Regions and % increase in urban land.")
    print()   
    print("Results for Hierarchical Clustering:")
    print()
    dfnew.fillna(method ='ffill', inplace = True) 
    X=dfnew.iloc[:,[20,21,22,23,24,25,26,27]].values
    labelencoder_X= LabelEncoder ()
    X[:,1]=labelencoder_X.fit_transform(X[:,1])
       
    Y= pd.DataFrame(X)
    #Creating a dendogram to find the optimal number of clusters for the columns mentioned above.
    print("Dendrogram: Using linkage type 'Ward':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'ward')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Ward:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Complete':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'complete')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Complete:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Average':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'average')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Average :')
    plt.show()
    plt.clf()
    # Fitting H.Clustering to the dataset
    
    #TYPE 1= Affinity: Euclidean and Linkage Type: Ward
    hc= AgglomerativeClustering(n_clusters=5, affinity= 'euclidean', linkage = 'ward')
    y_hc= hc.fit_predict(X)
    accuracy1 = sm.accuracy_score(y_hc, hc.labels_)
    print("The Accuracy Score for Type 1 is : " + str(accuracy1))
    
    #TYPE 2= Affinity: Euclidean and Linkage Type: Complete
    hc2= AgglomerativeClustering(n_clusters=5, affinity= 'euclidean', linkage = 'complete')
    y_hc2= hc2.fit_predict(X)
    accuracy2 = sm.accuracy_score(y_hc2, hc2.labels_)
    print("The Accuracy Score for Type 2 is : " + str(accuracy2))
    
    #TYPE 3= Affinity: Manhattan and Linkage Type: Average
    hc3= AgglomerativeClustering(n_clusters=5, affinity= 'manhattan', linkage = 'average')
    y_hc3= hc3.fit_predict(X)
    accuracy3 = sm.accuracy_score(y_hc3, hc3.labels_)
    print("The Accuracy Score for Type 3 is : " + str(accuracy3))
    
    ####Lets visualize the clusters######
    plt.scatter(X[y_hc== 0,0], X[y_hc== 0,1], s=5, c= 'blue', label= 'Cluster 1')
    plt.scatter(X[y_hc== 1,0], X[y_hc== 1,1], s=5, c= 'red', label= 'Cluster 2')
    plt.scatter(X[y_hc== 2,0], X[y_hc== 2,1], s=5, c= 'green', label= 'Cluster 3')
    plt.title('Graph for Scenario 2')
    plt.legend()
    plt.show()
    
    print("Results for KMEANS Clustering:")
    print()
    # First remove missing data
    nullCount = dfnew.isnull().values.ravel().sum()
    #print("\nNull Count:\n");
    pprint(nullCount)
    dfnew.dropna()
    #pprint(len(dfnew.index))
    #Conversion to numerical 
    dfnew=pd.concat([dfnew['NE'], dfnew['SE'], dfnew['PS'],dfnew['NC'],dfnew['SC'],dfnew['RM'],dfnew['GP'],dfnew['PN'], dfnew['Percent increase in urban land 2000-2010']],
                 axis=1, keys=['NE', 'SE', 'PS', 'NC','SC','RM','GP','PN','Percent increase in urban land 2000-2010'])
    x = dfnew.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    #pprint(normalizedDataFrame[:10])
    # Create clusters (TEST 1)
    k = 5
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    
    # Create clusters (TEST 1)
    k2 = 10
    kmeans = KMeans(n_clusters=k2)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    # Checking to see how it fits on different dimensions
    #print(pd.crosstab(cluster_labels, dfnew['Population']))
    #print(pd.crosstab(cluster_labels, dfnew['MinT']))
    #print(pd.crosstab(cluster_labels, dfnew['MaxT']))
    
    # PCA
    print("PCA projection:")
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    
    # This shows how good the PCA performs on this dataset
    print(pca2D.explained_variance_)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("PCA 2-Dimentional Plot for Scenario 2")
    
    # Write to file
    #plt.savefig("pca2D.png")
    plt.show()
    # Clear plot
    plt.clf()
    
def Scenario3 (dfnew):
    print("For Scenario 3, lets consider the following attributes: Name, MinT, MaxT, and Automobile Per Capita.")
    print()   
    print("Results for Hierarchical Clustering:")
    print()
    dfnew.Name = dfnew.Name.astype("category").cat.codes
    dfnew.fillna(method ='ffill', inplace = True) 
    X=dfnew.iloc[:,[0,9,31,32]].values
    labelencoder_X= LabelEncoder ()
    X[:,1]=labelencoder_X.fit_transform(X[:,1])
       
    Y= pd.DataFrame(X)
    #Creating a dendogram to find the optimal number of clusters for the columns mentioned above.
    print("Dendrogram: Using linkage type 'Ward':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'ward')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Ward:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Complete':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'complete')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Complete :')
    plt.show()
    plt.clf()
    
    print("Dendrogram: Using linkage type 'Average':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'average')) #Let's use the ward method first.
    plt.title('Dendrogram using linkage type Average :')
    plt.show()
    plt.clf()
    # Fitting H.Clustering to the dataset
    
    #TYPE 1= Affinity: Euclidean and Linkage Type: Ward
    hc= AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage = 'ward')
    y_hc= hc.fit_predict(X)
    accuracy1 = sm.accuracy_score(y_hc, hc.labels_)
    print("The Accuracy Score for Type 1 is : " + str(accuracy1))
    
    #TYPE 2= Affinity: Euclidean and Linkage Type: Complete
    hc2= AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage = 'complete')
    y_hc2= hc2.fit_predict(X)
    accuracy2 = sm.accuracy_score(y_hc2, hc2.labels_)
    print("The Accuracy Score for Type 2 is : " + str(accuracy2))
    
    #TYPE 3= Affinity: Manhattan and Linkage Type: Average
    hc3= AgglomerativeClustering(n_clusters=3, affinity= 'manhattan', linkage = 'average')
    y_hc3= hc3.fit_predict(X)
    accuracy3 = sm.accuracy_score(y_hc3, hc3.labels_)
    print("The Accuracy Score for Type 3 is : " + str(accuracy3))
    
    ####Lets visualize the clusters######
    plt.scatter(X[y_hc== 0,0], X[y_hc== 0,1], s=5, c= 'blue', label= 'Cluster 1')
    plt.scatter(X[y_hc== 1,0], X[y_hc== 1,1], s=5, c= 'red', label= 'Cluster 2')
    plt.scatter(X[y_hc== 2,0], X[y_hc== 2,1], s=5, c= 'green', label= 'Cluster 3')
    plt.title('Graph for Scenario 3')
    plt.legend()
    plt.show()
    
    print("Results for KMEANS Clustering:")
    print()
    # First remove missing data
    nullCount = dfnew.isnull().values.ravel().sum()
    #print("\nNull Count:\n");
    pprint(nullCount)
    dfnew.dropna()
    #pprint(len(dfnew.index))
    #Conversion to numerical 
    dfnew=pd.concat([dfnew['Name'], dfnew['Automobiles Per Capita'], dfnew['MinT'],dfnew['MaxT']],
                 axis=1, keys=['Name', 'Automobiles Per Capita', 'MinT', 'MaxT'])
    x = dfnew.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    #pprint(normalizedDataFrame[:10])
    # Create clusters (TEST 1)
    k = 35
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    
    # Create clusters (TEST 1)
    k2 = 250
    kmeans = KMeans(n_clusters=k2)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    # Checking to see how it fits on different dimensions
    #print(pd.crosstab(cluster_labels, dfnew['Population']))
    #print(pd.crosstab(cluster_labels, dfnew['MinT']))
    #print(pd.crosstab(cluster_labels, dfnew['MaxT']))
    
    # PCA
    print("PCA projection:")
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    
    # This shows how good the PCA performs on this dataset
    print(pca2D.explained_variance_)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("PCA 2-Dimentional Plot for Scenario 3")
    
    # Write to file
    #plt.savefig("pca2D.png")
    plt.show()
    # Clear plot
    plt.clf()
    
def Scenario4 (dfnew):
    print("For Scenario 4, lets consider the following attributes: Name,Number of Factories, MinT, MaxT,")
    print()   
    print("Results for Hierarchical Clustering:")
    print()
    dfnew.fillna(method ='ffill', inplace = True) 
    X=dfnew.iloc[:,[0,30,31,32]].values
    labelencoder_X= LabelEncoder ()
    X[:,1]=labelencoder_X.fit_transform(X[:,1])
       
    Y= pd.DataFrame(X)
    #Creating a dendogram to find the optimal number of clusters for the columns mentioned above.
    print("Dendrogram: Using linkage type 'Ward':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'ward')) #Let's use the ward method first.
    plt.title('Dendrogram: using linkage: Ward')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Complete':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'complete')) #Let's use the ward method first.
    plt.title('Dendrogram: using linkage: Complete')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Average':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'average')) #Let's use the ward method first.
    plt.title('Dendrogram: using linkage: Average')
    plt.show()
    plt.clf()
    # Fitting H.Clustering to the dataset
    
    #TYPE 1= Affinity: Euclidean and Linkage Type: Ward
    hc= AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage = 'ward')
    y_hc= hc.fit_predict(X)
    accuracy1 = sm.accuracy_score(y_hc, hc.labels_)
    print("The Accuracy Score for Type 1 is : " + str(accuracy1))
    
    #TYPE 2= Affinity: Euclidean and Linkage Type: Complete
    hc2= AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage = 'complete')
    y_hc2= hc2.fit_predict(X)
    accuracy2 = sm.accuracy_score(y_hc2, hc2.labels_)
    print("The Accuracy Score for Type 2 is : " + str(accuracy2))
    
    #TYPE 3= Affinity: Manhattan and Linkage Type: Average
    hc3= AgglomerativeClustering(n_clusters=3, affinity= 'manhattan', linkage = 'average')
    y_hc3= hc3.fit_predict(X)
    accuracy3 = sm.accuracy_score(y_hc3, hc3.labels_)
    print("The Accuracy Score for Type 3 is : " + str(accuracy3))
    
    ####Lets visualize the clusters######
    plt.scatter(X[y_hc== 0,0], X[y_hc== 0,1], s=5, c= 'blue', label= 'Cluster 1')
    plt.scatter(X[y_hc== 1,0], X[y_hc== 1,1], s=5, c= 'red', label= 'Cluster 2')
    plt.scatter(X[y_hc== 2,0], X[y_hc== 2,1], s=5, c= 'green', label= 'Cluster 3')

    plt.title('Graph for Scenario 4')
    plt.legend()
    plt.show()
    
    print("Results for KMEANS Clustering:")
    print()
    # First remove missing data
    nullCount = dfnew.isnull().values.ravel().sum()
    #print("\nNull Count:\n");
    pprint(nullCount)
    dfnew.dropna()
    #pprint(len(dfnew.index))
    #Conversion to numerical 
    dfnew=pd.concat([dfnew['Name'], dfnew['Number of Factories'], dfnew['MinT'],dfnew['MaxT']],
                 axis=1, keys=['Name', 'Number of Factories', 'MinT', 'MaxT'])
    x = dfnew.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    #pprint(normalizedDataFrame[:10])
    # Create clusters (TEST 1)
    k = 80
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    
    # Create clusters (TEST 1)
    k2 = 500
    kmeans = KMeans(n_clusters=k2)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    # Checking to see how it fits on different dimensions
    #print(pd.crosstab(cluster_labels, dfnew['Population']))
    #print(pd.crosstab(cluster_labels, dfnew['MinT']))
    #print(pd.crosstab(cluster_labels, dfnew['MaxT']))
    
    # PCA
    print("PCA projection:")
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    
    # This shows how good the PCA performs on this dataset
    print(pca2D.explained_variance_)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("PCA 2-Dimentioanl Plot for Scenario 4")
    
    # Write to file
    #plt.savefig("pca2D.png")
    plt.show()
    # Clear plot
    plt.clf()

def Dbscan(dfnew):
    X, label = make_moons(n_samples=200, noise= 0.15 , random_state=30)
    print (X[:3,])
    model= DBSCAN(eps=0.25, min_samples=12).fit(X)
    print(model)
    fig,ax= plt.subplots(figsize=(10,8))
    sctr= ax.scatter(X[:,0],X[:,1], c= model.labels_, s=140, alpha=0.8)
    plt.title('DBSCAN for the dataset')
    fig.show()



######### Association Rules
def association_prep(dfnew):
    
    import statistics 
    import math 
    #pip install mlxtend
    #from mlxtend.frequent_patterns import association_rules, apriori                            
    from apyori import apriori


    # Approximate null temperature values for DC 2015-2019 using linear approximation
    # from the two previous years

    # Set state name constant
    state = 'District of Columbia'
    # For each year...
    for year in range(2015,2019):
        # Save row of interest
        record = dfnew[(dfnew['Name'] == state) & (dfnew['Year'] == year)].index.values[0]
        # Obtain row from previous year
        record1 = dfnew[(dfnew['Name'] == state) & (dfnew['Year'] == year-1)].index.values[0]
        # Obtain row from two years prior
        record2 = dfnew[(dfnew['Name'] == state) & (dfnew['Year'] == year-2)].index.values[0]
        # Calculate approximations using two previous years' numbers
        dfnew['MeanT'][record] = dfnew['MeanT'][record1] + (dfnew['MeanT'][record1] - dfnew['MeanT'][record2])
        dfnew['MaxT'][record] = statistics.mean([dfnew['MaxT'][record1], dfnew['MaxT'][record2]])
        dfnew['MinT'][record] = statistics.mean([dfnew['MinT'][record1], dfnew['MinT'][record2]])


    # Replace null values in airport type columns with zeros, as nulls likely are indicative of a value of 0
    dfnew[['Balloonport','Heliport','Large Airport','Medium Airport','Seaplane Base','Small Airport']] = dfnew[['Balloonport','Heliport','Large Airport','Medium Airport','Seaplane Base','Small Airport']].replace(np.nan,0)

    # Define empty columns that will become population change from previous and 
    # population percentage change from previous year
    dfnew['Pop_Increase'] = ''
    dfnew['Pop_Percent_Increase'] = ''
    dfnew['MaxTInc'] = ''


    # Calculate new columns' values for years for which we have previous year's data (i.e. all except 2000)

    # For each row...
    for row in range(0,len(dfnew)):
        # Save off state and year
        state = dfnew['Name'][row]
        year = dfnew['Year'][row]
        # Find row that is same state but in previous year
        plus = dfnew[(dfnew['Name'] == state) & (dfnew['Year'] == year-1)]
        # Calculate difference between this year's pop and previous year's pop
        popplus = dfnew['Population'][row] - plus['Population']
        #Calculate difference between this year's max temp and previous year's max temp
        maxTplus = dfnew['MaxT'][row] - plus['MaxT']
        # Calculate percentage change between this year's pop and previous year's pop
        popplus_percent = popplus/plus['Population']
    
        # Avoid erroring by skipping if previous year's values did not exist (null or 1999)
        if plus.empty == False:
            # Add population statistics to this row
            dfnew['Pop_Increase'][row] = popplus.values[0]
            dfnew['Pop_Percent_Increase'][row] = popplus_percent.values[0]
            # Avoid erroring by skipping if previous year's values didn't exist
        if maxTplus.empty == False:
            # Otherwise, add to temp change column
            dfnew['MaxTInc'][row] = maxTplus.values[0]
        
            
    # Approximate new columns' values for year 2000 through linear approximation
    # Assume that 1999->2000 change was similar to 2000->2001 change.
    for row in dfnew[dfnew['Pop_Increase']==''].index:
        state = dfnew['Name'][row]
        year = dfnew['Year'][row]
        plus = dfnew[(dfnew['Name'] == state) & (dfnew['Year'] == year+1)]
        popplus = plus['Population'] - dfnew['Population'][row]
        popplus_percent = popplus/dfnew['Population'][row]
        dfnew['Pop_Increase'][row] = popplus.values[0]
        dfnew['Pop_Percent_Increase'][row] = popplus_percent.values[0]
        
    return(dfnew)
    
    
## Binning - bin each of 8 variables using k=4. Inspect with histogram and then
## choose appropriate binwidths. General goal is equi-depth, while also trying
## to capture stark separations in value concentrations and negative vs. positive
# in the case of population change.
def encoding(dfnew): 
    
    # Create bins for total count of medium+large airports
    dfnew['Airports'] = dfnew['Large Airport'] + dfnew['Medium Airport']
    #dfnew['Airports'].hist()
    dfnew['AeroBins'] = pd.cut(dfnew['Airports'],[-1,10,22,40,65])
    dfnew['AeroBins'] = dfnew['AeroBins'].astype('category')
    dfnew['AeroBins'] = dfnew['AeroBins'].cat.codes    


    # Create bins for population percentage increase
    #dfnew['Pop_Percent_Increase'].hist()
    dfnew['PopChangeBins'] = pd.cut(dfnew['Pop_Percent_Increase'],[-.07,0,.01,.02,.05])
    dfnew['PopChangeBins'] = dfnew['PopChangeBins'].astype('category')
    dfnew['PopChangeBins'] = dfnew['PopChangeBins'].cat.codes

    # Create bins for automobile count
    #dfnew['Automobiles'].hist()
    dfnew['AutoBins'] = pd.cut(dfnew['Automobiles'],[-.01,1e6,3e6,1e7,2.2e7])
    dfnew['AutoBins'] = dfnew['AutoBins'].astype('category')
    dfnew['AutoBins'] = dfnew['AutoBins'].cat.codes  
    
    # Create bins for total population
    #dfnew['Population'].hist()
    dfnew['PopBins'] = pd.cut(dfnew['Population'],[-1,1e6,5e6,1e7,4e7])
    dfnew['PopBins'] = dfnew['PopBins'].astype('category')
    dfnew['PopBins'] = dfnew['PopBins'].cat.codes 

    # Create bins for mean temperature
    #dfnew['MeanT'].hist()
    dfnew['TBins'] = pd.cut(dfnew['MeanT'],[-5,5,10,18,25])
    dfnew['TBins'] = dfnew['TBins'].astype('category')
    dfnew['TBins'] = dfnew['TBins'].cat.codes

    # Create bins for truck counts
    #dfnew['Trucks'].hist()
    dfnew['TruckBins'] = pd.cut(dfnew['Trucks'],[0,6e4,1e6,2e6,15e6])
    dfnew['TruckBins'] = dfnew['TruckBins'].astype('category')
    dfnew['TruckBins'] = dfnew['TruckBins'].cat.codes

    # Create bins for bus counts
    #dfnew['Buses'].hist()
    dfnew['BusBins'] = pd.cut(dfnew['Buses'],[-1,10000,20000,40000,120000])
    dfnew['BusBins'] = dfnew['BusBins'].astype('category')
    dfnew['BusBins'] = dfnew['BusBins'].cat.codes

    # Create bins for urban land increase percentages
    #dfnew['Percent increase in urban land 2000-2010'].hist()
    dfnew['UrbanBins'] = pd.cut(dfnew['Percent increase in urban land 2000-2010'],[-1,7,15,20,30])
    dfnew['UrbanBins'] = dfnew['UrbanBins'].astype('category')
    dfnew['UrbanBins'] = dfnew['UrbanBins'].cat.codes

    return(dfnew)

def associationRULES(dfnew):
    
    # Import library
    from collections import OrderedDict
    
    # Choose columns that will be included in association rules mining
    assoc = dfnew[['Name','Year','AeroBins','PopChangeBins','AutoBins','PopBins','TBins','TruckBins','BusBins','UrbanBins']]
    # Remove rows with null values
    assoc = assoc.loc[(assoc != -1).all(axis=1)]

    # Encode different variables' bins with different numbers so that each variable
    # bin is a unique number between 0 and 31. Smallest bin number within each span corresponds
    # to smallest values, largest bin numbers correspond to largest values.
    assoc['AeroBins'] += 0
    assoc['PopChangeBins'] += 4
    assoc['AutoBins'] += 8
    assoc['PopBins'] += 12
    assoc['TBins'] += 16
    assoc['TruckBins'] += 20
    assoc['BusBins'] += 24
    assoc['UrbanBins'] += 28

    # Run Apriori algorithm on selected binned subset of dataset, using specified minimum supports and confidences.
    frq_itemset = list(apriori(assoc.values, min_support = .1, min_confidence = .1))

    # Define dataframe that will hold Apriori algorithm's cleaned output
    rules = pd.DataFrame(columns=('Items','Antecedent','Consequent','Support','Confidence'))

    # Define empty lists that will be appended to and then placed in Apriori output dataframe
    Support =[]
    Confidence = []
    Lift = []
    Items = []
    Antecedent = []
    Consequent=[]

    # For each frequent itemset...
    for RelationRecord in frq_itemset:
        # For each subset of each frequent itemset...
        for ordered_stat in RelationRecord.ordered_statistics:
        
            # Add support of this itemset subset to list of all supports
            Support.append(RelationRecord.support)
            # Save the items in this itemset subset to list of all itemsets
            Items.append(RelationRecord.items)
            # Save the antecedent item(s) of the rule to list of all antecedents
            Antecedent.append(ordered_stat.items_base)
            # Save the consequent item(s) of the rule to list of all consequents
            Consequent.append(ordered_stat.items_add)
            # Save confidence to list of all confidences
            Confidence.append(ordered_stat.confidence)

    # Save appended lists to dataframe
    rules['Items'] = list(map(set, Items))                                   
    rules['Antecedent'] = list(map(set, Antecedent))
    rules['Consequent'] = list(map(set, Consequent))
    rules['Support'] = Support
    rules['Confidence'] = Confidence


    # Calculate frequent itemsets using min_support of 0.1, 0.2, and 0.3
    rules_sup_10 = rules[rules['Support'] >= .1]
    rules_sup_20 = rules[rules['Support'] >= .2]
    rules_sup_30 = rules[rules['Support'] >= .3]

    # Obtain lists of unique frequent itemsets for each min_support.
    # Select column containing sets of items, convert sets to lists, and save lists in
    # an ordered dict of frozensets, as saviing it to a dict removes all redundant values
    # (e.g. situations where n=4 and one combination has 0 as antecedent and 4,8,12 as consequent,
    # while the other has 4 as antecedent and 0, 8, 12 as consequent. I am interested in those
    # as separate entities, but I am performing these computations only to analyze unique frequent itemsets.)
    Items_10 = rules_sup_10['Items']
    Items_10 = Items_10.tolist()
    Items_10 = [set(i) for i in OrderedDict.fromkeys(frozenset(item) for item in Items_10)]
    #Items_10 = pd.DataFrame(Items_10)

    Items_20 = rules_sup_20['Items']
    Items_20 = Items_20.tolist()
    Items_20 = [set(i) for i in OrderedDict.fromkeys(frozenset(item) for item in Items_20)]
    #Items_20 = pd.DataFrame(Items_20)

    Items_30 = rules_sup_30['Items']
    Items_30 = Items_30.tolist()
    Items_30 = [set(i) for i in OrderedDict.fromkeys(frozenset(item) for item in Items_30)]
    #Items_30 = pd.DataFrame(Items_30)



########### HYPOTHESIS TESTING #######################

# Hypothesis 1             
#Did the number of vehicle registrations (automobiles, buses, trucks 
# and motorcycles) impact the minimum and maximum temperatures?
    
# Function that runs Hypothesis Test 1
def Hypo1(dfnew):
    # Linear Regression of Max & Min Temps by Car Registrations & Population 
    print('Linear Regression results for Max Temperatures\n')
    Hypo1LR(dfnew,'MaxT')
    print('Linear Regression results for Min Temperatures\n')
    Hypo1LR(dfnew,'MinT')
    print('Granger Causality Analysis Results\n')
    Hypo1GC(dfnew)

#Linear Regression             
def Hypo1LR(dfnew, var):
    import statsmodels.api as sm
    d1 = dfnew[['Automobiles','Population','Buses','Trucks','Motorcycles', 'MaxT', 'MinT', 'Year','Name']] 
    for col in ['Automobiles','Buses','Population','Trucks','Motorcycles', 'MaxT', 'MinT','Year']:
        d1[col] = pd.to_numeric(d1[col])
    d1.isnull().any()
    d1 = d1.dropna()
    d1.Name = d1.Name.astype("category").cat.codes
    # Linear Regression of Temp by Car Registrations & Population
    X = d1[['Automobiles','Buses','Trucks','Motorcycles', 'Population']]
    Y = d1[var]
    X = sm.add_constant(X) #add a constant to the lm
    model = sm.OLS(Y, X).fit() #fit the model 
    predictions = model.predict(X) #predict values 
    print_model = model.summary() 
    print(print_model) #print model results 

#Granger Causality Analysis 
def Hypo1GC(dfnew):
    from statsmodels.tsa.stattools import grangercausalitytests
    d1 = dfnew[['Automobiles','Population','Buses','Trucks','Motorcycles', 'MaxT', 'MinT', 'Year','Name']] 
    for col in ['Automobiles','Buses','Population','Trucks','Motorcycles', 'MaxT', 'MinT','Year']:
        d1[col] = pd.to_numeric(d1[col])
    d1.isnull().any()
    d1 = d1.dropna()
    d1.Name = d1.Name.astype("category").cat.codes
    # Print Granger Causality Result for Max T and each of the dependent variables
    print('Granger Causality for Maximum Temperatures\n')
    for col in ['Automobiles','Buses','Trucks','Motorcycles', 'Population']:
        print('\nGranger Causality for ' + col)
        grangercausalitytests(d1[['MaxT', col]], maxlag=2)
    
    # Print Granger Causality Result for Min T and each of the dependent variables
    print('\nGranger Causality for Minimum Temperatures\n')
    for col in ['Automobiles','Buses','Trucks','Motorcycles', 'Population']:
        print('\nGranger Causality for ' + col)
        grangercausalitytests(d1[['MinT', col]], maxlag=2)

# Hypothesis 2
# Are population bin, car registrations, number of airports and 
# region the most important attributes to explain the changes in max temperature? 

# Function that adds a variable that bins data by population 
def PopB(dfnew):    
    dfnew['PopBin'] = 0
    # Top 25% of pop 
    dfnew['PopBin'].mask(dfnew.Population >= 6.770010e+06, 1, inplace=True)
    # 50% - 75% of pop
    dfnew['PopBin'].mask(dfnew.Population < 6.770010e+06, 2, inplace=True)
    # 50% to 25% of pop
    dfnew['PopBin'].mask(dfnew.Population < 4.146101e+06, 3, inplace=True)
    # Bottom 25% of pop
    dfnew['PopBin'].mask(dfnew.Population < 1.682930e+06, 4, inplace=True)
    return(dfnew)

# Function that runs Hypothesis Test 2
def Hypo2LR(dfnew):
    import statsmodels.api as sm
    dfnew = dfnew.fillna(0)
    X = dfnew[['PopBin','Automobiles','Buses','Trucks','Motorcycles', 'Hybrid',
           'PlugHybrid', 'Electric', 'Balloonport', 'Heliport', 'Large Airport',
           'Medium Airport', 'Seaplane Base', 'Small Airport', 'NE', 'SE', 'PS',
           'NC', 'SC', 'RM', 'GP', 'PN',
           'Number of Alt Fuel Facilities', 'Number of Factories', 'Largest Transit Service Area', 'Operating Expenses FY']]
    Y = dfnew['MaxT']
    Y = pd.to_numeric(Y)
    X = sm.add_constant(X) #add a constant to the lm
    model = sm.OLS(Y, X).fit() #fit the model 
    predictions = model.predict(X) #predict values 
    print_model = model.summary() 
    print(print_model)    



########## Use Random Forest to evaluate relevance of each feature in predicting temperature increases
def Hypo2RF(dfnew):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    
    # Save dataframe that is solely Olympic years, but remove Puerto Rico because we don't have temp data
    rf = dfnew[(dfnew['Year'].isin([2000,2004,2008,2012,2016])) & (dfnew['Name'] != 'Puerto Rico')]
    # Define empty column that will store change in max temperature over the previous 4 years
    rf['MaxTInc'] = ''

    # For each row...
    for row in rf.index:
        # Save off state and year
        state = rf['Name'][row]
        year = rf['Year'][row]
        # We will not have a temperature change number for 2000 (because we don't have data from 1996 in the dataframe), so don't calculate for that year
        if year != 2000:
            # Find row that is same state but 4 years prior
            plus = rf[(rf['Name'] == state) & (rf['Year'] == year-4)]
            #Calculate difference between this year's max temp and max temp in this state 4 years prior
            maxTplus = rf['MaxT'][row] - plus['MaxT']
            # Otherwise, add to temp change column
            rf['MaxTInc'][row] = maxTplus.values[0]

    # Remove year-2000 columns because it has served its purpose of helping generate values for year-2004 rows
    rf = rf[rf['Year'] != 2000]
    # Sort temperature change values to allow for bin edges to be defined
    values = sorted(rf['MaxTInc'])
    # Calculate equi-depth bin edges
    edges = [values[0]-.01,values[50]+.001,values[101]+.001,values[152]+.001,values[203]+.01]
    # Assign category variable, the class that we will test for, based on bins
    rf['Class'] = pd.cut(rf['MaxTInc'], bins=edges)
    # Make class a category variable
    rf['Class'] = rf['Class'].astype('category')
    # Encode category variable as numbers
    rf['Class'] = rf['Class'].cat.codes    

    ## Remove null values from vehicular rows by assuming linear approximation between 2008 and 2012
    ## continues for 2016 (i.e. change from 2012 to 2016 is approximately the same as change from 2008 to 2012)
    # List columns that we want to approximate
    vehicles = ['Automobiles','Buses','Trucks','Motorcycles']
    # For each row
    for i in rf.index:
        # For each column
        for vehicle in vehicles:
            # If value is null...
            if np.isnan(rf[vehicle][i]) == True:
                # Obtain row containing 2012 data for that state
                goodrow1 = rf[(rf['Year'] == 2012) & (rf['Name'] == rf['Name'][i])]
                goodrow2 = rf[(rf['Year'] == 2008) & (rf['Name'] == rf['Name'][i])]
                # Save 2012 + (2012-2008) data in this row (which is a year-2016 row because 
                # those are the only rows containing null values for these columns)
                rf[vehicle][i] = goodrow1[vehicle].values[0] + goodrow1[vehicle].values[0] - goodrow2[vehicle].values[0]


    # Save variables that I want to test
    rf_input = rf[['Year','Automobiles','Buses','Trucks','Motorcycles','Heliport','Airports','NE','SE','PS','NC','SC','RM','GP','PN','Percent increase in urban land 2000-2010','Number of Alt Fuel Facilities','Pop_Increase','Pop_Percent_Increase','Class']]

    # Save data to be classified to array
    valueArray = rf_input.values
    # Save array that removes the temperature increase classifier
    X = valueArray[:, 0:19]
    # Normalize values in this array
    preprocessing.normalize(X)
    # save temperature increase classes to separate array
    Y = valueArray[:, 19]


    ## First, I'm going to evaluate the applicability of random forest to this dataset
    # generally by running creating a test set of 20% of the full dataset and running 2-fold validation.
    
    # Identify proportion of the dataset - 20% - that will be used for testing
    test_size = 0.20
    # Random state
    seed = 7
    # Separate dataset into training and test portions
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Define model
    model = RandomForestClassifier(n_estimators=20)
    # Define number of folds
    kfold = KFold(n_splits=5, random_state=seed, shuffle=False)
    # Perform 2-fold cross-validation on the training set.
    results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    # Print data on classifier's performance for training data
    print("RF's performance vis-a-vis training data: mean = ",str(results.mean()), ", Std Dev = ", str(results.std()))
   
    # Use classifier to predict classes for test set
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validate)

    # Print performance on test set
    print("\nRF's performance vis-a-vis test data:\nAccuracy: ",accuracy_score(Y_validate, predictions))
    print("Confusion Matrix:\n",confusion_matrix(Y_validate, predictions))
    print("Classification Report:\n",classification_report(Y_validate, predictions))  


    ## I'm also going to use Random Forest to help evaluate our hypothesis regarding the relative importance
    ## of each variable for predicting temperature rise.

    # Run Random Forest on full dataset
    model.fit(X, Y, n_features=20)  
    # Calculate relative importance of each feature
    final = model.feature_importances_
    # Print importance coefficients alongside column names
    printout = pd.DataFrame(rf_input.columns[:-1], final)
    printout



# Hypothesis 3 
#Can we predict which states will have higher max temp 
# using the attributes from above? (see which predictive model is best - SVM, NB, DT)

# Function that runs Hypothesis Test 3
def Hypo3(dfnew):
    X_train,Y_train,X_validate,Y_validate = Hypo3Data(dfnew)
    # SVM
    SVM(X_train,Y_train,X_validate,Y_validate) # call function       
    # DecisionTree
    DT(X_train,Y_train,X_validate,Y_validate) # call function       
    # Naive-Bayes    
    NB(dfnew)    
    # KNN
    KNN(X_train,Y_train,X_validate,Y_validate)

# Function that preporcesses data for Hypothesis 3
def Hypo3Data(dfnew):
    Hypo3Data501 = dfnew[['MaxT', 'PopBin','Population','Automobiles','Buses','Trucks','Motorcycles','Hybrid', 'PlugHybrid', 'Electric',
                          'Balloonport', 'Heliport', 'Large Airport', 'Medium Airport', 'Small Airport','Seaplane Base','NE','SE','PS','NC','SC','RM','GP','PN']] #get required columns
    Hypo3Data501 = pd.DataFrame(Hypo3Data501) # turn data into dataframe
    Hypo3Data501 = Hypo3Data501.dropna() #drop rows with na's
    #get data so the type can be handled 
    Hypo3Data501['Population'] = Hypo3Data501['Population'].astype(str).astype(int) 
    Hypo3Data501['MaxT'] = Hypo3Data501['MaxT'].astype(int)
    Hypo3Data501['Automobiles'] = round(Hypo3Data501['Automobiles'])
    Hypo3Data501['Automobiles'] = Hypo3Data501['Automobiles'].astype(int)
    Hypo3Data501['Buses'] = round(Hypo3Data501['Buses'])
    Hypo3Data501['Buses'] = Hypo3Data501['Buses'].astype(int)
    Hypo3Data501['Trucks'] = round(Hypo3Data501['Trucks'])
    Hypo3Data501['Trucks'] = Hypo3Data501['Trucks'].astype(int)
    Hypo3Data501['Motorcycles'] = round(Hypo3Data501['Motorcycles'])
    Hypo3Data501['Motorcycles'] = Hypo3Data501['Motorcycles'].astype(int)
    Hypo3Data501['Hybrid'] = round(Hypo3Data501['Hybrid'].astype(int))
    Hypo3Data501['Hybrid'] = Hypo3Data501['Hybrid'].astype(int)
    Hypo3Data501['PlugHybrid'] = round(Hypo3Data501['PlugHybrid'].astype(int))
    Hypo3Data501['PlugHybrid'] = Hypo3Data501['PlugHybrid'].astype(int)
    Hypo3Data501['Electric'] = round(Hypo3Data501['Electric'].astype(int))
    Hypo3Data501['Electric'] = Hypo3Data501['Electric'].astype(int)
    Hypo3Data501['Balloonport'] = round(Hypo3Data501['Balloonport'])
    Hypo3Data501['Balloonport'] = Hypo3Data501['Balloonport'].astype(int)
    Hypo3Data501['Heliport'] = round(Hypo3Data501['Heliport'])
    Hypo3Data501['Heliport'] = Hypo3Data501['Heliport'].astype(int)
    Hypo3Data501['Large Airport'] = round(Hypo3Data501['Large Airport'])
    Hypo3Data501['Large Airport'] = Hypo3Data501['Large Airport'].astype(int)
    Hypo3Data501['Medium Airport'] = round(Hypo3Data501['Medium Airport'])
    Hypo3Data501['Medium Airport'] = Hypo3Data501['Medium Airport'].astype(int)
    Hypo3Data501['Small Airport'] = round(Hypo3Data501['Small Airport'])
    Hypo3Data501['Small Airport'] = Hypo3Data501['Small Airport'].astype(int)
    Hypo3Data501['Seaplane Base'] = round(Hypo3Data501['Seaplane Base'])
    Hypo3Data501['Seaplane Base'] = Hypo3Data501['Seaplane Base'].astype(int)
    
    valueArray = Hypo3Data501.values
    X1 = valueArray[:, 0:23] #get full dataset
    Y1 = valueArray[:, 0]
    
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X1, Y1, test_size=test_size, random_state=seed) 
  
    return(X_train,Y_train,X_validate,Y_validate)

# SVM Function
def SVM(X_train,Y_train,X_validate,Y_validate): # this function runs SVM on the specified data and gets accuracy
    clf = SVC(gamma='scale', decision_function_shape='ovo')
    svcACC = clf.fit(X_train, Y_train)
    print("SVM Training Accuracy: ", SVC.score(svcACC, X_train, Y_train, sample_weight=None))

    svcACC = clf.fit(X_validate, Y_validate)
    print("SVM Test Accuracy: ",SVC.score(svcACC, X_validate, Y_validate, sample_weight=None))

# DT Function
def DT(X_train,Y_train,X_validate,Y_validate): # this function runs decision tree on the specified data and gets accuracy. Based off of cheatsheet code
    num_folds = 10
    seed = 7
    scoring = 'accuracy'    
    models = []
    models.append(('CART', DecisionTreeRegressor()))

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
 
    dt = DecisionTreeClassifier()
    dt.fit(X_train, Y_train)
    predictions = dt.predict(X_validate)   
    print()
    print("DT Accuracy Score", accuracy_score(Y_validate, predictions))
    print("DT Confiusion Matrix: \n",confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))
    #fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_validate, predictions)
    #plt.plot(fpr, tpr, color='b')
    print(Y_validate[0:10])
    print(predictions[0:10])

# NB Function    
def NB(dfnew):
    dfnew = dfnew.fillna(0)
    X = dfnew[['PopBin','Automobiles','Buses','Trucks','Motorcycles', 'Hybrid',
           'PlugHybrid', 'Electric', 'Balloonport', 'Heliport', 'Large Airport',
           'Medium Airport', 'Seaplane Base', 'Small Airport', 'NE', 'SE', 'PS',
           'NC', 'SC', 'RM', 'GP', 'PN',
           'Number of Alt Fuel Facilities', 'Number of Factories', 'Largest Transit Service Area', 'Operating Expenses FY']]
    Y = dfnew['MaxT']
    Y = pd.to_numeric(Y)
    Y = Y.astype('int')
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed) 
    num_folds = 10 #defining number of folds for the Kfold method    
    scoring = 'accuracy' #defining the desired score    
    nb = MultinomialNB()
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
    cv_results = cross_val_score(nb, X_train, Y_train, cv=kfold, scoring=scoring)           
    nb.fit(X_train,Y_train)
    predictions = nb.predict(X_validate)
    print(accuracy_score(Y_validate, predictions))   

# KNN Function    
def KNN(X_train,Y_train,X_validate,Y_validate):
    num_folds = 10
    seed = 7
    scoring = 'accuracy'
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validate)

    print()
    print("KNN Accuracy Score", accuracy_score(Y_validate, predictions))
    print("KNN Confiusion Matrix: \n", confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))
    print(Y_validate[0:10])
    print(predictions[0:10])
    
# Hypothesis 4
# Does Temperature in North States increase generally the same as 
# it does for South States over the time frame of our data?

def Hypo4(dfnew):
    import sklearn
    from sklearn.preprocessing import normalize
    import seaborn
    import plotly.graph_objects as go
    
    Hypo4Data501 = dfnew[['Name', 'Year','MeanT']] #get required columns
    Hypo4Data501 = pd.DataFrame(Hypo4Data501) # turn data into dataframe
    Hypo4Data501 = Hypo4Data501.dropna() #drop rows with na's
    Hypo4Data501.MeanTChange = np.nan #make new column for differences in temp from year to year 
    StatesNames = pd.DataFrame(Hypo4Data501.Name.unique())
    StatesNames.columns = ['Name']
    StatesNames['TChange'] = np.nan
    
    # loop through all states for each year. calculate the difference in the state's mean temp
    # from the year before. Do this for all states and all years. Also keep track of the means of all the states total
    for i in range(len(StatesNames.Name)):
        dataF = Hypo4Data501.loc[Hypo4Data501.Name == StatesNames.Name[i]]
        dataF = dataF.reset_index(drop=True)
        lst = [0] * len(dataF)
        lst = np.array(lst)
        for j in range(len(dataF)):
            if j != 0:
                diff = dataF.MeanT[j]-dataF.MeanT[j-1]
                Hypo4Data501.loc[(Hypo4Data501['Name']==dataF.Name[j]) & (Hypo4Data501.Year==dataF.Year[j]), 'MeanTChange'] = diff
                lst[j] = diff
        mdiffA = np.array(lst)
        mdiff = np.mean(mdiffA)
        StatesNames.TChange[i] = mdiff #means of all the states mean differences per year
        
    #form North group
    NorthStates = StatesNames[StatesNames['Name'].isin(['Alaska', 'Connecticut','Idaho','Illinois','Indiana','Iowa','Maine','Massachusetts','Michigan','Minnesota',
                              'Montana','Nebraska','New Hampshire','New Jersey','New York','North Dakota','Ohio','Oregon','Pennsylvania','Rhode Island'
                              'South Dakota','Vermont','Washington','Wisconsin','Wyoming'])]
    SouthStates = StatesNames[StatesNames['Name'].isin(['Alabama','Arizona','Arkansas','California','Colorado','Delaware','District of Columbia','Florida',
                              'Georgia','Hawaii','Kansas','Kentucky','Louisiana','Mississippi','Missouri','Nevada','New Mexico','North Carolina','Oklahoma',
                              'South Carolina','Tennessee','Texas','Utah','Virginia','West Virginia', 'Maryland'])]
    
    NorthStates = pd.DataFrame(NorthStates)
    SouthStates = pd.DataFrame(SouthStates)
    print(NorthStates.mean(axis = 0) ) #mean of north
    print(SouthStates.mean(axis = 0) ) #mean of south
    print(stats.ttest_ind(NorthStates['TChange'],SouthStates['TChange']))
    Hypo4Data501N = Hypo4Data501[Hypo4Data501['Name'].isin(['Alaska', 'Connecticut','Idaho','Illinois','Indiana','Iowa','Maine','Massachusetts','Michigan','Minnesota',
                              'Montana','Nebraska','New Hampshire','New Jersey','New York','North Dakota','Ohio','Oregon','Pennsylvania','Rhode Island'
                              'South Dakota','Vermont','Washington','Wisconsin','Wyoming'])]
    Hypo4Data501S = Hypo4Data501[Hypo4Data501['Name'].isin(['Alabama','Arizona','Arkansas','California','Colorado','Delaware','District of Columbia','Florida',
                              'Georgia','Hawaii','Kansas','Kentucky','Louisiana','Mississippi','Missouri','Nevada','New Mexico','North Carolina','Oklahoma',
                              'South Carolina','Tennessee','Texas','Utah','Virginia','West Virginia', 'Maryland'])]
    
    #set up time series boxplots for each region     
    ts = pd.Series(Hypo4Data501N['MeanTChange'], index=Hypo4Data501N.Year)
    fig, ax = plt.subplots(figsize=(12,5))
    seaborn.boxplot(x = ts.index, y = Hypo4Data501N['MeanTChange'], ax=ax).set_title("North States Mean Temperature Changes")
    
    ts = pd.Series(Hypo4Data501S['MeanTChange'], index=Hypo4Data501S.Year)
    fig, ax = plt.subplots(figsize=(12,5))
    seaborn.boxplot(x = ts.index, y = Hypo4Data501S['MeanTChange'], ax=ax).set_title("South States Mean Temperature Changes")
    
    # plot Temp changes overall on US map
    codes = ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME',
             'MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
             'TX','UT','VT','VA','WA','WV','WI','WY']
    StatesNames['Abr'] = codes
    
    fig = go.Figure(data=go.Choropleth(
        locations=StatesNames['Abr'], # get locations correct
        z = StatesNames['TChange'].astype(float), # color data by temp change 
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        colorbar_title = "Temp Change",
    ))
    
    fig.update_layout(
        title_text = '2000-2018 USA Temperature Change in Celsius',
        geo_scope='usa', # limite map scope to USA
    )
    
    fig.show()
    fig.write_image("fig1.png") #export as png to folder
    
    
    # test regions 
    NorthEast = StatesNames[StatesNames['Name'].isin(['Connecticut','Maine','Massachusetts','New Hampshire','New Jersey','New York',
                            'Pennsylvania','Rhode Island','Vermont'])]
    MidWest = StatesNames[StatesNames['Name'].isin(['Illinois','Indiana','Iowa','Michigan','Minnesota','Kansas',
                             'Nebraska','North Dakota','Ohio','South Dakota','Wisconsin','Missouri'])]
    South = StatesNames[StatesNames['Name'].isin(['Alabama','Arkansas','Florida', 'Georgia','Kentucky','Louisiana','Mississippi','District of Columbia',
                            'North Carolina','Oklahoma', 'South Carolina','Tennessee','Texas','Virginia','West Virginia','Maryland','Delaware'])]
    West = StatesNames[StatesNames['Name'].isin(['Arizona','California','Colorado','Wyoming','Alaska','Oregon','Montana','Washington',
                              'Hawaii','Nevada','New Mexico','Utah','Virginia',])]
    NorthEast = pd.DataFrame(NorthEast)
    MidWest = pd.DataFrame(MidWest)
    South = pd.DataFrame(South)
    West = pd.DataFrame(West)
    print(NorthEast['TChange'].mean(axis = 0) )
    print(MidWest['TChange'].mean(axis = 0) )
    print(South['TChange'].mean(axis = 0) )
    print(West['TChange'].mean(axis = 0) )
    NorthEast = np.array(NorthEast['TChange']) #get array of data
    MidWest = np.array(MidWest['TChange']) #get array of data
    South = np.array(South['TChange']) #get array of data
    West =  np.array(West['TChange']) #get array of data
    NorthEast=np.reshape(NorthEast,(-1, 1)) # reshape array
    MidWest=np.reshape(MidWest,(-1, 1))# reshape array
    West=np.reshape(West,(-1, 1))# reshape array
    South=np.reshape(South,(-1, 1))# reshape array
    NorthEast=sklearn.preprocessing.normalize(NorthEast) # normalize
    MidWest=sklearn.preprocessing.normalize(MidWest)# normalize
    West=sklearn.preprocessing.normalize(West)# normalize
    South=sklearn.preprocessing.normalize(South)# normalize
    print(stats.f_oneway(NorthEast, MidWest, South, West))
    
    
    Hypo4Data501NE = Hypo4Data501[Hypo4Data501['Name'].isin(['Connecticut','Maine','Massachusetts','New Hampshire','New Jersey','New York',
                            'Pennsylvania','Rhode Island','Vermont'])]
    Hypo4Data501MW = Hypo4Data501[Hypo4Data501['Name'].isin(['Illinois','Indiana','Iowa','Michigan','Minnesota','Kansas',
                             'Nebraska','North Dakota','Ohio','South Dakota','Wisconsin','Missouri'])]
    Hypo4Data501SS = Hypo4Data501[Hypo4Data501['Name'].isin(['Alabama','Arkansas','Florida', 'Georgia','Kentucky','Louisiana','Mississippi','District of Columbia',
                            'North Carolina','Oklahoma', 'South Carolina','Tennessee','Texas','Virginia','West Virginia','Maryland','Delaware'])]
    Hypo4Data501W = Hypo4Data501[Hypo4Data501['Name'].isin(['Arizona','California','Colorado','Wyoming','Alaska','Oregon','Montana','Washington',
                              'Hawaii','Nevada','New Mexico','Utah','Virginia',])]
        
    #set up time series boxplots for each region     
    ts = pd.Series(Hypo4Data501NE['MeanTChange'], index=Hypo4Data501NE.Year)
    fig, ax = plt.subplots(figsize=(12,5))
    seaborn.boxplot(x = ts.index, y = Hypo4Data501NE['MeanTChange'], ax=ax).set_title("Northeast Mean Temperature Changes")
    
    ts = pd.Series(Hypo4Data501MW['MeanTChange'], index=Hypo4Data501MW.Year)
    fig, ax = plt.subplots(figsize=(12,5))
    seaborn.boxplot(x = ts.index, y = Hypo4Data501MW['MeanTChange'], ax=ax).set_title("Midwest Mean Temperature Changes")
    
    ts = pd.Series(Hypo4Data501SS['MeanTChange'], index=Hypo4Data501SS.Year)
    fig, ax = plt.subplots(figsize=(12,5))
    seaborn.boxplot(x = ts.index, y = Hypo4Data501SS['MeanTChange'], ax=ax).set_title("Southern Mean Temperature Changes")
    
    ts = pd.Series(Hypo4Data501W['MeanTChange'], index=Hypo4Data501W.Year)
    fig, ax = plt.subplots(figsize=(12,5))
    seaborn.boxplot(x = ts.index, y = Hypo4Data501W['MeanTChange'], ax=ax).set_title("West Mean Temperature Changes")

# Hypothesis 5
# Did states with significant increase of urban land 
# between 2000 and 2010 have higher maximum temperatures?     
    
# This function creates a dataset that computes differences in each attribute between 2010 and 2000 by state
# These data are used for both Hypo 5 & 6
def Hypo5Data(dfnew):
    df1 = dfnew[dfnew.Year == 2000]
    df2 = dfnew[dfnew.Year == 2010]
    df3 = pd.merge(df1,df2[['Name','Population', 'Passenger Trips','Operating Expenses FY',
            'MinT', 'MaxT', 'MeanT']], on = 'Name', how='left')
    
    cols = ['Population', 'Passenger Trips','Operating Expenses FY',
            'MinT', 'MaxT', 'MeanT']
    for col in cols:
        name1 = str(col)+'_x'
        name2 = str(col)+'_y'
        df3[col] = df3[name2] - df3[name1]
        df3 = df3.drop(columns = [name1,name2])
        df3[col] = pd.to_numeric(df3[col])
    
    df3 = df3[['MaxT','MinT', 'MeanT', 'Percent increase in urban land 2000-2010', 'Population', 'Name', 'Automobiles', 'Buses', 'Trucks', 'Motorcycles', 'Large Airport', 'Medium Airport', 'Small Airport' ]]
    df3['Percent increase in urban land 2000-2010'] = pd.to_numeric(df3['Percent increase in urban land 2000-2010'])
    df3 = df3.dropna(subset=['Percent increase in urban land 2000-2010'])
    df3 = df3.fillna(0)
    return(df3)
    
# Function that runs hypothesis testing 5
def Hypo5(df3, var):
    import statsmodels.api as sm
    X = df3[['Percent increase in urban land 2000-2010','Population','Automobiles', 'Buses', 'Trucks', 'Motorcycles', 'Large Airport', 'Medium Airport', 'Small Airport']]
    Y = df3[var]
    Y = pd.to_numeric(Y)
    X = sm.add_constant(X) #add a constant to the lm
    model = sm.OLS(Y, X).fit() #fit the model
    predictions = model.predict(X) #predict values 
    print_model = model.summary()
    print(print_model) #print model results 

# Hypothesis 6
#Which attributes drove most change in max and min temperatures between 2000 and 2010?  
# Function that runs hypothesis testing 6
def Hypo6(df3,var):
    from scipy import stats
    # Create a new variable that separates the dataframe into two groups
    df3['IncreaseU'] = np.where(df3['Percent increase in urban land 2000-2010'] > 13, 1, 0)
    a = df3[df3.IncreaseU == 1]
    b = df3[df3.IncreaseU == 0]
    a1 = a[var]
    a1 =pd.to_numeric(a1)
    b1 = b[var] 
    b1 =pd.to_numeric(b1) 
    
    # Run t-test for change in maximum temperature of both groups
    tval, pval = stats.ttest_ind(a1,b1)
    print('P value is ', pval,'\n T statistic is ', tval)
    

########## GRAPHS

def Graphs(dfnew):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Histogram of urban land icnrease values distribution 
    df3 = Hypo5Data(dfnew)
    plt.hist(df3['Percent increase in urban land 2000-2010'])
    plt.title("Percent Increase in Urban Land (2000-2010)", fontsize=12)


    states = dfnew['Name'].unique()
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(states), replace=False)

    # Draw Plot -- Min T by state over time
    plt.figure(figsize=(20,12), dpi= 80)
    for i, y in enumerate(states):
        if i >= 0:        
            plt.plot('Year', 'MinT', data=dfnew.loc[dfnew.Name==y, :], color=mycolors[i], label=y)
            #plt.text(dfnew.loc[dfnew.Name==y, :].shape[0], dfnew.loc[dfnew.Name==y, 'Population'][-1:].values[0], y, fontsize=12, color=mycolors[i])
            plt.gca().set(ylabel='$Minimum Temperature (Celsius)$', xlabel='$Year$')
            plt.yticks(fontsize=12, alpha=.7)
            plt.title("Time Series Plot of State Min Temperatures", fontsize=20)
            plt.legend(loc=2, ncol=4)
    
    # Draw Plot -- Max T by state over time
    plt.figure(figsize=(20,12), dpi= 80)
    for i, y in enumerate(states):
        if i >= 0:        
            plt.plot('Year', 'MaxT', data=dfnew.loc[dfnew.Name==y, :], color=mycolors[i], label=y)
            #plt.text(dfnew.loc[dfnew.Name==y, :].shape[0], dfnew.loc[dfnew.Name==y, 'Population'][-1:].values[0], y, fontsize=12, color=mycolors[i])
            plt.gca().set(ylabel='$Maximum Temperature (Celsius)$', xlabel='$Year$')
            plt.yticks(fontsize=12, alpha=.7)
            plt.title("Time Series Plot of State Max Temperatures", fontsize=20)
            plt.legend(loc=2, ncol=4)
    
    # Draw Plot -- Car Reg per capita by state over time
    plt.figure(figsize=(20,12), dpi= 80)
    for i, y in enumerate(states):
        if i >= 0:        
            plt.plot('Year', 'Automobiles Per Capita', data=dfnew.loc[dfnew.Name==y, :], color=mycolors[i], label=y)
            #plt.text(dfnew.loc[dfnew.Name==y, :].shape[0], dfnew.loc[dfnew.Name==y, 'Population'][-1:].values[0], y, fontsize=12, color=mycolors[i])
            plt.gca().set(ylabel='$Automobiles Per Capita$', xlabel='$Year$')
            plt.yticks(fontsize=12, alpha=.7)
            plt.title("Time Series Plot of State Per Capita Vehicle Registrations", fontsize=20)
            plt.legend(loc=2, ncol=4)
    
    # Boxplots of Min and Max Tem Ranges Over Time
    fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
    sns.boxplot(x='Year', y='MaxT', data=dfnew, ax=axes[0]) # Max T range over time
    sns.boxplot(x='Year', y='MinT', data=dfnew, ax=axes[1]) # Min T range over time
    axes[0].set_title('Maximum Temperatures Over Time', fontsize=12); 
    axes[1].set_title('Minimum Temperatures Over Time', fontsize=12)
       
    # Dataset for 3D graphs
    d1 = dfnew[['Automobiles','Population','Buses','Trucks','Motorcycles', 'MaxT', 'MinT', 'Year','Name']] 
    for col in ['Automobiles','Buses','Population','Trucks','Motorcycles', 'MaxT', 'MinT','Year']:
        d1[col] = pd.to_numeric(d1[col])
    d1.isnull().any()
    d1 = d1.dropna()
    d1.Name = d1.Name.astype("category").cat.codes
    
    # 3D Plot of Max Temp by Car Reg by Year
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.set_facecolor('white')
    ax1.scatter3D(d1.Year, d1.MaxT,d1['Automobiles'],  c=d1.Name, cmap='hsv')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Max Temp (Celsius)')
    ax1.set_zlabel('Vehicle Regs')
    
    # 3D Plot of Min Temp by Car Reg by Year
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_facecolor('white')
    ax.scatter3D(d1.Year,d1.MinT, d1['Automobiles'],c=d1.Name, cmap='hsv')
    ax.set_xlabel('Year')
    ax.set_ylabel('Min Temp (Celsius)')
    ax.set_zlabel('Vehicle Regs')
    
    # 3D Plot of Max Temp and Min Temp by Year 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_facecolor('white')
    ax.scatter3D(d1.MaxT,d1.MinT, d1['Population'],c=d1.Name, cmap='hsv')
    ax.set_xlabel('Min Temp (Celsius)')
    ax.set_ylabel('Max Temp (Celsius)')
    ax.set_zlabel('Population')




if __name__ == '__main__':
    Main()