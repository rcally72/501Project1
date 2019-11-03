#obr#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:12:03 2019

@author: mashagubenko
"""

####### Masha's Code



# Import libraries
import pandas as pd

# Import the population info by State
# Annual
# Location as State

# Start the master data set by setting a datafrae of all states' population by year 
popstate = pd.read_csv('Population_by_State_by_Year_2000_2010.csv')
popstate = popstate.drop(columns=['Unnamed: 0','census2010pop','estimatesbase2000'])
colpop = popstate.columns

new = pd.DataFrame(columns=['Name', 'Population', 'Year'])
for col in colpop[1:len(colpop)]:
    df = popstate[['name',col]]
    df = df.groupby('name',as_index=False).max()
    df = df.rename(columns={col: 'Population', 'name':'Name'})
    df['Year'] = int(col[len(col)-4:len(col)])
    new = pd.concat([new,df],ignore_index=True)

popstate = pd.read_csv('nst-est2018-alldata.csv')
popstate = popstate[['NAME', 'POPESTIMATE2010', 'POPESTIMATE2011',
       'POPESTIMATE2012', 'POPESTIMATE2013', 'POPESTIMATE2014',
       'POPESTIMATE2015', 'POPESTIMATE2016', 'POPESTIMATE2017',
       'POPESTIMATE2018']]
popstate = popstate.drop(popstate.index[[0,1,2,3,4]])
colpop = popstate.columns

new2 = pd.DataFrame(columns=['Name', 'Population', 'Year'])
for col in colpop[1:len(colpop)]:
    df = popstate[['NAME',col]]
    df = df.groupby('NAME',as_index=False).sum()
    df = df.rename(columns={col: 'Population', 'NAME':'Name'})
    df['Year'] = int(col[len(col)-4:len(col)])
    new2 = pd.concat([new2,df],ignore_index=True)

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
stat = transit['HQ State'].apply(lambda x: states[x])
transit['HQ State'] = stat
#Total trips by state by year
a = transit.groupby(['HQ State', 'FY End Year'], as_index = False)['Unlinked Passenger Trips FY'].sum()
a = a.rename(columns={'FY End Year': 'Year', 'HQ State':'Name', 'Unlinked Passenger Trips FY': 'Passenger Trips'})
#Largest service area by state by year
b = transit.groupby(['HQ State', 'FY End Year'], as_index = False)['Service Area SQ Miles'].max()
b = b.rename(columns={'FY End Year': 'Year', 'HQ State':'Name', 'Service Area SQ Miles': 'Largest Transit Service Area'})
#Total trips by state by year
c = transit.groupby(['HQ State', 'FY End Year'], as_index = False)['Operating Expenses FY'].sum()
c = c.rename(columns={'FY End Year': 'Year', 'HQ State':'Name'})

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
vehreg.Name = vehreg.Name.str.replace(r'[\(\)\d]+', '')
vehreg.Name = vehreg.Name.str.replace('/','')
vehreg = vehreg.replace('Dist. of Col.', 'District of Columbia')

for k in vehreg.Name:
    if k[len(k)-1] == ' ':
        if k[len(k)-2] == ' ':
            vehreg = vehreg.replace(k, k[0:len(k)-2])
        else:
            vehreg = vehreg.replace(k, k[0:len(k)-1])
        #k = k[0:len(k)-1]

dfnew = pd.merge(dfnew, vehreg, how='left', on=['Name', 'Year'])

# Import the data set on electric vehicle registrations 
# Annual 
# Aggregated for the entire country by vehicle type & year so location is US
# Years: 1999-2017
evreg = pd.read_csv('EV_Registrations_by_Type_US_by_Year.csv')
evreg = evreg.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
evreg = evreg.replace('U',0)
colev = evreg.columns

# Add the ev registration data to the master dataset
# Total number of ev registrations in the US is added to each state for each type of ev
# TO DO: to decide how we want to approximate this - by CPI? by Pop? 
dfnew['Hybrid'] = 0 
dfnew['PlugHybrid'] = 0 
dfnew['Electric'] = 0 
for col in colev:
    dfnew['Hybrid'].mask(dfnew['Year']== int(col), evreg[col][0], inplace=True)
    dfnew['PlugHybrid'].mask(dfnew['Year']== int(col), evreg[col][1], inplace=True)
    dfnew['Electric'].mask(dfnew['Year']== int(col), evreg[col][2], inplace=True)


# Import the data on airport location
# Add number of airports by type by state to the master dataset
airloc = pd.read_csv('AirportLocation.csv') #11696
airloc['Location'] = airloc['Location'].str.lower()
airloc['Location'] = airloc['Location'].str.title()
airloc = airloc.replace('District Of Columbia', 'District of Columbia')
airloc = airloc[airloc['type'] != 'closed']
airloc = airloc.rename(columns={'Location': 'Name'})

a1 = airloc[airloc['type'] == 'balloonport']
a2 = airloc[airloc['type'] == 'heliport']
a3 = airloc[airloc['type'] == 'large_airport']
a4 = airloc[airloc['type'] == 'medium_airport']
a5 = airloc[airloc['type'] == 'seaplane_base']
a6 = airloc[airloc['type'] == 'small_airport']

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

# Set up columns corresponding to each of the region
reg = ['NE','SE','PS','NC','SC','RM','GP','PN']
for lst in reg:
    dfnew[lst] = 0

# Loop through the lists of regions and identify states that belong to 
# each region by assigning 1 to that column    
reglist = [NE,SE,PS,NC,SC,RM,GP,PN]       
i = 0 
while i < len(reglist):  
    print(reg[i])
    for j in range(0,len(dfnew['Name'])):
        if dfnew['Name'][j] in reglist[i]:
            dfnew[reg[i]][j] = 1
    i += 1
 
# Pull the data from the urban data set on growth % of urban land by eahc region
# Add the total % increase to each state belonging to the region in the master dataset    
# TO DO: to decide how we want to split this up by state - by pop? by pop growth? by # of large cities?
dfnew['Percent increase in urban land 2000-2010'] = 0
for i in range(0, len(reg)-1):
    dfnew['Percent increase in urban land 2000-2010'].mask(dfnew[reg[i]] == 1, urbland['Percent increase in urban land 2000-2010'][i+1] , inplace=True)
    

# Import the data set on alterantive fuel stations
# Type of fuel by station information
# COunt the number of alterantive fuel facilities by state and add it to the master dataset
altfuel = pd.read_csv('AlternativeFuelStationLocation.csv')
altfuel['Location'] = altfuel['Location'].str.lower()
altfuel['Location'] = altfuel['Location'].str.title()
altfuel = altfuel.replace('District Of Columbia', 'District of Columbia')
altfuel = altfuel.rename(columns={'Location': 'Name'})
alt1 = altfuel.groupby(['Name'], as_index = False).size().to_frame('Number of Alt Fuel Facilities')
dfnew = pd.merge(dfnew, alt1, how='left', on=['Name'])

# Import the facility pollution data set
# Count the number of factories by state and add it to the master data set
facpol = pd.read_csv('FacilityPollution.csv')
facpol['Location'] = facpol['Location'].str.lower()
facpol['Location'] = facpol['Location'].str.title()
facpol = facpol.replace('District Of Columbia', 'District of Columbia')
facpol = facpol.rename(columns={'Location': 'Name'})
fp1 = facpol.groupby(['Name'], as_index = False).size().to_frame('Number of Factories')
dfnew = pd.merge(dfnew, fp1, how='left', on=['Name'])

###### End Masha's Code


#data1 = pd.read_csv('temps.csv')
#data2 = pd.read_csv('temps5.csv')

#data3 = pd.concat([data1,data2])
#data3.to_csv('temps2.csv')



######### Ryan's Code

import requests
import json
import pandas as pd
import csv
import numpy as np

# Read in temperature data that contains full list of lat-longs
temps_old = pd.read_csv("temps_old.csv")

# Save full list of coordinates
geos_old = temps_old[['lat','long']]
geos_old = geos_old.drop_duplicates()
states = []

#7240 - Alaska
#9313 - Florida
#9697 - Midway Atoll

# Warning: this takes approximately 97 minutes to run. To save you time, I saved
# the output of this function to a csv that I then read in to a dataframe so that you can
# skip the API and just use the output.
# Move through each row of temperature dataset


#for i in geos.index.values:  
    
    # Store as a string the coordinates at which the temperature readings were obtained 
#    coords = str(geos['lat'][i]) + "," + str(geos['long'][i])

    # Endpoint URL
#    BaseURL="https://maps.googleapis.com/maps/api/geocode/json"
    # Define variables necessary to append to base URL in correct order for reverse geocoding
#    URLPost = {'latlng': coords,
#              'result_type':'administrative_area_level_1',
#              'key': 'AIzaSyBRDtYg7tsh4zFQ2wfw4Ic9gJtK2xS1Qqs'}
    # Send GET request to Google Geocoding API               
#    response = requests.get(BaseURL, URLPost)
    # Run json() function on results from website to provide readable output
#    jsontxt = response.json()
#    state = jsontxt['results'][0]['address_components'][0]['long_name']
    # Add to list of states
#    states.append(state)
#    print(len(states))

# Write state column to csv
#f = open("states.csv", "w")
#for item in states:
#    f.write(item)
#    f.write(",")
#f.close()
    
#f = open("statelist_final.csv", "w")
#for item in file:
#    f.write(item)
#    f.write(",")
#f.close()


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

dfnew = pd.merge(dfnew, summary, how='left', on=['Name', 'Year'])

######## End Ryan's Code



######Chris's Code

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
                
for i in range(0,len(dfnew)):
    if type(dfnew['Forest Acreage(Thousands)'][i]) == str:
        dfnew['Forest Acreage(Thousands)'][i] = dfnew['Forest Acreage(Thousands)'][i].replace(',','')
        dfnew['Forest Acreage(Thousands)'][i] = int(dfnew['Forest Acreage(Thousands)'][i])
######## End Chris's Code
