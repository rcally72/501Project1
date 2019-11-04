#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:37:48 2019

@author: christopherfiaschetti
"""


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
dfnew["Forest Acerage(Thousands)"] =  np.nan

for i in range(len(dfnew)): # loop through all rows of big dataframe
    for j in range(len(Forest_Data)): # loop through all rows of forest_data
        for k in range(3): # loop through columns of Forest_data
            p = k+1 
            if (str(dfnew.Year[i]) == Forest_Data.columns[p] and dfnew.Name[i] == Forest_Data.iloc[j,0]):
                dfnew["Forest Acerage(Thousands)"][i] = Forest_Data.iloc[j,p]

# Analysis
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# Part 1 - population - mean, median,mode,stdev
                
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
Moto = np.array(dfnew.Motorcicles)
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

# Correlation and scatterplots

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

data = dfnew[['Population', 'Motorcicles']]#get desired data
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





# Check Alternative Fieling stations and Temperature
data = dfnew[['MeanT', 'Number of Alt Fuel Facilities']]
TandAFF = pd.DataFrame(data)
TandAFF = TandAFF.dropna()
print(TandAFF.corr(method ='pearson'))
plt.style.use('ggplot')
plt.subplots(1, 1) 
plt.scatter(TandAFF['Number of Alt Fuel Facilities'], TandAFF.MeanT)
plt.title('Alternative Fueling Facilities vs Temp') 
plt.show()

# Hypo 4 - Are temp increases generally the same for southern vs northern states?
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