#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:06:05 2019

@author: ryanpatrickcallahan
"""
# Libraries:


def main(x):
    
    # Libraries
    import pandas as pd
    import urllib.request
    import tarfile
    
    # Define URL of interest
    #url2 = 'ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qfe.tar.gz'
    # Pull relevant .gz file from URL
    #urllib.request.urlretrieve(url2, 'temps.gz')
    # Open the .gz file using Python's .tar file decompressor
    #tar = tarfile.open("temps.gz")
    # Extract folder from .tar
    #tar.extractall()
    # Read the .dat file full of the temperature data into a dataframe
    datum = pd.read_csv('ghcnm.v4.0.1.20191004/ghcnm.tavg.v4.0.1.20191004.qfe.dat',header=None, delim_whitespace=True,error_bad_lines=False,engine='python')
    # Read the .inv file full of the collection locations into a dataframe
    locations = pd.read_csv('ghcnm.v4.0.1.20191004/ghcnm.tavg.v4.0.1.20191004.qfe.inv', header=None, delim_whitespace=True, error_bad_lines=False)
    # Add column names to locations dataframe
    locations.columns = ['id','lat','long','elevation','name']


    # Only retain columns containing temperature data
    datum = datum[[0,1,3,5,7,9,11,13,15,17,19,21,23]]
    # Rename these columns' headers to month names
    datum.columns = ['code','jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    # Parse out station identifier from the identifier-date column to enable column merger
    datum['id'] = datum['code'].astype(str).str[0:11]
    # Add station information to each temperature entry by mergin temperature and location datasets
    datum = pd.merge(datum, locations, on='id')
    # Parse out year from identifier-date column
    datum['year'] = datum['code'].astype(str).str[11:15]
    # Remove identifier-date column, as both relevant elements have been parsed into separate columns
    datum = datum.drop(['code'], axis=1)
    
    
    # Make list of month column names for purposes of iteration
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    # Define variable that will store indices of rows with bad data for removal
    kills = []
    # Define counter of cells with data that was invalid and could not be approximated
    killcount = []
    
    # Carve off subset of dataset for testing purposes
    datum = datum[0:1000]
    
    # Define variable that will save size of dataset before cleaning
    full = len(datum)
    
    
    numberize(datum,months)
    print(killcount)
    firstlast(datum,months,kills,killcount)
    print(killcount)
    clean(datum,months,kills,killcount)
    print(killcount)
    
    # Remove all designated rows
    datum = datum.drop(kills,axis=0)
    
    cleanliness(datum,kills,killcount,full)
    normalize(datum,months)



## Define function that converts to type "int" numerals incorrectly stored as type "string"
def numberize(datum, months):
    # Move through each row of the dataset
    for i in range(0,len(datum)):
        # Move through each of the 12 month columns for this row
        for month in months:
            if type(datum.loc[i,month]) is str:
                # Check whether this "string" is merely an improperly stored number
                if datum.loc[i,month].lstrip('-').isdigit():
                   # If string is in fact a numeral, convert to type int
                   datum.loc[i,month] = int(datum.loc[i,month])




# Because my smoothing technique requires looking at the previous and following years' data,
# first and final rows in dataset must be handled separately
def firstlast(datum, months, kills, killcount):
    
    # Move through first and last row of the dataset
    for i in 0,len(datum)-1:
        # Define counter that will store how many of the row's values are converted into a valid form.
        # After each row is processed, this counter will be checked, and if all cells either already
        # were in a valid form or were able to be converted into a valid form, the row is retained.
        # Otherwise, the row is dropped.
        counter = 0
        # Move through each of the 12 month columns for this row
        for month in months:
            # If cell is null, add nothing to the counter (which will ultimately result in the row being removed)
            if datum.loc[i,month] == None or type(datum.loc[i,month]) is str:
                counter += 0
            # If cell is already stored as a number, check whether number is in the range of valid temps
            if type(datum.loc[i,month]) is int or type(datum.loc[i,month]) is float:
                # If temp is within valid range (which I have arbitrarily decided is -40C < T < 40C), add 1 to counter
                if datum.loc[i,month] < 4000 and datum.loc[i,month] > -4000:
                    counter += 1
                # If temp is not a valid temperature, reset counter to 0, which is the equivalent
                # of removing the row
                else:
                    counter += 0
        
        # Add number of invalid cells to counter
        killcount.append(12-counter)
        # If not all 12 rows were able to be properly converted into numbers, add that row's index to the list of those designated for removal        
        if counter != 12:
            kills.append(i)



## Clean all rows of data other than first and last, which were already cleaned
def clean(datum, months, kills, killcount):
    
    #Libraries
    import statistics
    
    # Move through each row of the dataset
    for i in range(1,len(datum)-1):
        # Define counter that will store how many of the row's values are converted into a valid form.
        # After each row is processed, this counter will be checked, and if all cells either already
        # were in a valid form or were able to be converted into a valid form, the row is retained.
        # Otherwise, the row is dropped.
        counter = 0
        # Move through each of the 12 month columns for this row
        for month in months:
            # If cell is null...
            if datum.loc[i,month] == None:
                # Check whether values from preceding and following years for this month are integers or floats, and could therefore potentially offer a valid approximation
                if datum.loc[i,'name'] == datum.loc[i-1,'name'] and (type(datum.loc[i+1,month]) is int or type(datum.loc[i+1,month]) is float) and (type(datum.loc[i-1,month]) is int or type(datum.loc[i-1,month]) is float):
                    # Check whether values from preceding and following years are in the valid range
                    if datum.loc[i+1,month] < 4000 and datum.loc[i+1,month] > -4000 and datum.loc[i-1,month] < 4000 and datum.loc[i-1,month] > -4000:
                        # If both are within the valid range, save them in an array, take the average, and store that average as an approximation for this year
                        border = [datum.loc[i-1,month],datum.loc[i+1,month]]
                        datum.loc[i,month] = statistics.mean(border)
                        # Add 1 to counter, as this cell was properly approximated
                        counter += 1
            # If value is already in int or float form...   
            elif type(datum.loc[i,month]) is int or type(datum.loc[i,month]) is float:
                # Check whether number is outside valid range
                if datum.loc[i,month] > 4000 or datum.loc[i,month] < -4000:
                    # If number was outside valid range, check variable types of preceding and following years for potential for approximation
                    if datum.loc[i,'name'] == datum.loc[i-1,'name'] and (type(datum.loc[i+1,month]) is int or type(datum.loc[i+1,month]) is float) and (type(datum.loc[i-1,month]) is int or type(datum.loc[i-1,month]) is float):
                        # If preceding and following years were numbers, check whether they are both within valid range
                        if datum.loc[i+1,month] < 4000 and datum.loc[i+1,month] > -4000 and datum.loc[i-1,month] < 4000 and datum.loc[i-1,month] > -4000:
                            # If both are within the valid range, save them in an array, take the average, and store that average as an approximation for this year
                            border = [datum.loc[i-1,month],datum.loc[i+1,month]]
                            datum.loc[i,month] = statistics.mean(border)
                            # Add 1 to counter, as this cell was properly approximated
                            counter +=1
                # If number was in the valid range, leave it unchanged and add 1 to the counter
                else:
                    counter +=1
            # If cell is currently stored in string form...
            elif type(datum.loc[i,month]) is str:
                # If string was not converted to a numeral by numberize(), it's not a valid number.
                # Therefore, check whether preceding and following years can provide valid approximation
                if datum.loc[i,'name'] == datum.loc[i-1,'name'] and (type(datum.loc[i+1,month]) is int or type(datum.loc[i+1,month]) is float) and (type(datum.loc[i-1,month]) is int or type(datum.loc[i-1,month]) is float):
                    if datum.loc[i+1,month] < 4000 and datum.loc[i+1,month] > -4000:
                        # If both preceding and following years are of valid variable type and within the valid range, save them in an array, take the average, and store that average as an approximation for this year
                        border = [datum.loc[i-1,month],datum.loc[i+1,month]]
                        datum.loc[i,month] = statistics.mean(border)
                        # Add 1 to counter
                        counter += 1
        
        # Add number of invalid cells to counter
        killcount.append(12-counter)
        # If not all 12 rows were able to be properly converted into numbers, add that row's index to the list of those designated for removal        
        if counter != 12:
            kills.append(i)

        

        
# Quantify how clean the dataset was       
def cleanliness(datum,kills,killcount,full):
    # Count how many rows were removed due to invalid data
    badrows = len(kills)
    # Report percentage of datapoints that were bad and percentage of rows that were removed.
    print("Of ", full, "rows, ", badrows, "rows, or ", (badrows/full)*100 , " percent, were eliminated due to containing at least one bad datapoint.\n")
    print(sum(killcount), "datapoints, ", (sum(killcount)/(12*full))*100 ,"percent of the dataset, were invalid and could not be approximated.")


## Temperature values are stored at T*10^2. Therefore, divide all values by 100.     
def normalize(datum,months):
    # Divide all temperature values by 100, as they are stored as T*10^2
    for month in months:
        # Normalize temp values
        datum[month] = datum[month]/100  



if __name__ == "__main__":
    import sys
    # execute only if run as a script
    main(sys.argv)