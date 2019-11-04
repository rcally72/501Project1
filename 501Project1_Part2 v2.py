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
    Clustering(dfnew)
    Hypo1(dfnew) # Run hypothesis testing 1
    dfnew = PopB(dfnew)
    Hypo2LR(dfnew) # Run hypothesis testing 2
    Hypo3(dfnew) # Run hypothesis testing 3
    
    Hypo5(Hypo5Data(dfnew),'MaxT') # Run hypothesis testing 5
    Hypo5(Hypo5Data(dfnew),'MinT') # Run hypothesis testing 5
    
    Hypo5(Hypo5Data(dfnew),'MaxT') # Run hypothesis testing 6
    Hypo5(Hypo5Data(dfnew),'MinT') # Run hypothesis testing 6


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
    
    return(dfnew)


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
    d1 = dfnew[['Automobiles','Population','Buses','Trucks','Motorcycles', 'MaxT', 'MinT', 'RangeT', 'Year','Name']] 
    for col in ['Automobiles','Buses','Population','Trucks','Motorcycles', 'MaxT', 'MinT','RangeT','Year']:
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
    d1 = dfnew[['Automobiles','Population','Buses','Trucks','Motorcycles', 'MaxT', 'MinT', 'RangeT', 'Year','Name']] 
    for col in ['Automobiles','Buses','Population','Trucks','Motorcycles', 'MaxT', 'MinT','RangeT','Year']:
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

# Hypothesis 3 
#Can we predict which states will have higher max temp 
# using the attributes from above? (see which predictive model is best - SVM, NB, DT)

# Function that runs Hypothesis Test 3
def Hypo3(dfnew):
    # SVM
    SVM(Hypo3Data(dfnew)) # call function       
    # DecisionTree
    DT(Hypo3Data(dfnew)) # call function       
    # Naive-Bayes    
    NB(dfnew)    
    # KNN
    KNN(Hypo3Data(dfnew))

# SVM Funcrion
def SVM(X_train,Y_train,X_validate,Y_validate): # this function runs SVM on the specified data and gets accuracy
    clf = SVC(gamma='scale', decision_function_shape='ovo')
    svcACC = clf.fit(X_train, Y_train)
    print("SVM Training Accuracy: ", SVC.score(svcACC, X_train, Y_train, sample_weight=None))

    svcACC = clf.fit(X_validate, Y_validate)
    print("SVM Test Accuracy: ",SVC.score(svcACC, X_validate, Y_validate, sample_weight=None))

# DT Function
def DT(X_train,Y_train,X_validate,Y_validate): # this function runs decision tree on the specified data and gets accuracy. Based off of cheatsheet code
    num_folds = 3
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
def NB(X_train, X_validate, Y_train, Y_validate):
    num_folds = 10 #defining number of folds for the Kfold method 
    seed = 7
    scoring = 'accuracy' #defining the desired score    
    nb = MultinomialNB()
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
    cv_results = cross_val_score(nb, X_train, Y_train, cv=kfold, scoring=scoring)           
    nb.fit(X_train,Y_train)
    predictions = nb.predict(X_validate)
    print(accuracy_score(Y_validate, predictions))   

# KNN Function    
def KNN(X_train,Y_train,X_validate,Y_validate):
    num_folds = 3
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

# Function that preporcesses data for Hypothesis 3
def Hypo3Data(dfnew):
    Hypo3Data501 = dfnew[['MaxT', 'PopBin','Automobiles','Buses','Trucks','Motorcicles','Hybrid', 'PlugHybrid', 'Electric',
                          'Ballonport', 'Heliport', 'Large Airport', 'Medium Airport', 'Small Airport','Seaplane Base','NE','SE','PS','NC','SC','RM','GP','PN']] #get required columns
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
    Hypo3Data501['Motorcicles'] = round(Hypo3Data501['Motorcicles'])
    Hypo3Data501['Motorcicles'] = Hypo3Data501['Motorcicles'].astype(int)
    Hypo3Data501['Hybrid'] = round(Hypo3Data501['Hybrid'])
    Hypo3Data501['Hybrid'] = Hypo3Data501['Hybrid'].astype(int)
    Hypo3Data501['PlugHybrid'] = round(Hypo3Data501['PlugHybrid'])
    Hypo3Data501['PlugHybrid'] = Hypo3Data501['PlugHybrid'].astype(int)
    Hypo3Data501['Electric'] = round(Hypo3Data501['Electric'])
    Hypo3Data501['Electric'] = Hypo3Data501['Electric'].astype(int)
    Hypo3Data501['Ballonport'] = round(Hypo3Data501['Ballonport'])
    Hypo3Data501['Ballonport'] = Hypo3Data501['Ballonport'].astype(int)
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
    X1 = valueArray[:, 0:6] #get full dataset
    Y1 = valueArray[:, 0]
    
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X1, Y1, test_size=test_size, random_state=seed) 
  
    return(X_train, X_validate, Y_train, Y_validate)
    

# Hypothesis 4
# Does Temperature in North States increase generally the same as 
# it does for South States over the time frame of our data?












# Hypothesis 5
# Did states with significant increase of urban land 
# between 2000 and 2010 have higher maximum temperatures?     
    
# This functino creates a dataset that computes differences in each attribute between 2010 and 2000 by state
# These data are used for both Hypo 5 & 6
def Hypo5Data(dfnew):
    df1 = dfnew[dfnew.Year == 2000]
    df2 = dfnew[dfnew.Year == 2010]
    df3 = pd.merge(df1,df2[['Name','Population', 'Passenger Trips','Operating Expenses FY',
            'MinT', 'MaxT', 'MeanT', 'RangeT']], on = 'Name', how='left')
    
    cols = ['Population', 'Passenger Trips','Operating Expenses FY',
            'MinT', 'MaxT', 'MeanT', 'RangeT']
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
    d1 = dfnew[['Automobiles','Population','Buses','Trucks','Motorcycles', 'MaxT', 'MinT', 'RangeT', 'Year','Name']] 
    for col in ['Automobiles','Buses','Population','Trucks','Motorcycles', 'MaxT', 'MinT','RangeT','Year']:
        d1[col] = pd.to_numeric(d1[col])
    d1.isnull().any()
    d1 = d1.dropna()
    d1.Name = d1.Name.astype("category").cat.codes
    
    # 3D Plot of Max Temp by Car Reg by Year
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(d1.Year, d1.MaxT,d1['Automobiles'],  c=d1.Name, cmap='hsv')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Max Temp (Celsius)')
    ax1.set_zlabel('Vehicle Regs')
    
    # 3D Plot of Min Temp by Car Reg by Year
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(d1.Year,d1.MinT, d1['Automobiles'],c=d1.Name, cmap='hsv')
    ax.set_xlabel('Year')
    ax.set_ylabel('Min Temp (Celsius)')
    ax.set_zlabel('Vehicle Regs')
    
    # 3D Plot of Max Temp and Min Temp by Year 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(d1.MaxT,d1.MinT, d1['Population'],c=d1.Name, cmap='hsv')
    ax.set_xlabel('Min Temp (Celsius)')
    ax.set_ylabel('Max Temp (Celsius)')
    ax.set_zlabel('Population')



if __name__ == '__main__':
    Main()