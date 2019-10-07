# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def AlternativeFuel(): 
    Alternativefuel= pd.read_csv('AlternativeFuelStations.csv' , sep=',', encoding='latin1')
    print(Alternativefuel.dtypes)
    Alternativefuel=Alternativefuel.rename(columns={"State":"Location"})
    missingvalues=Alternativefuel.isnull().sum()
    missingval= "The number of missing values in each column:\n" + str(missingvalues)
    print(missingval)
    Alternativefuel=Alternativefuel.dropna()
    Alternativefuel.to_csv('AlternativeFuelStationLocation.csv')
   
    
def AirportUSA():
    AirportLocation= pd.read_csv('AiportDataSet.csv' , sep=',', encoding='latin1')
    print(AirportLocation.dtypes)
    missingvalues=AirportLocation.isnull().sum()
    Aiportmissingval= "The number of missing values in each column:\n" + str(missingvalues)
    print(Aiportmissingval)
    AirportLocation=AirportLocation.dropna()
    AirportLocation=AirportLocation.rename(columns={"State":"Location"})
    AirportLocation=AirportLocation.rename(columns={"municipality":"City"})
    AirportLocation.to_csv('AirportLocation.csv')


def NuclearPlant():
    NuclearLocation= pd.read_csv('NuclearPlants.csv' , sep=',', encoding='latin1')
    print(NuclearLocation.dtypes)
    missingvalues=NuclearLocation.isnull().sum()
    Nuclearmissingval= "The number of missing values in each column:\n" + str(missingvalues)
    print(Nuclearmissingval)
    NuclearLocation=NuclearLocation.dropna()
    NuclearLocation=NuclearLocation.drop(["p90_1200", "p00_1200","p10_1200","p90u_1200","p00u_1200","p10u_1200","p90r_1200","p00r_1200","p10r_1200","p90_30"], axis = 1)
    NuclearLocation=NuclearLocation.rename(columns={"Country":"City"})
    NuclearLocation.to_csv('NuclearLocation.csv')


def AirPollution():
    FacilityPollution= pd.read_csv('FacilityAirPollution.csv' , sep=',', encoding='latin1')
    print(FacilityPollution.dtypes)
    missingvalues=FacilityPollution.isnull().sum()
    Facilitymissingval= "The number of missing values in each column:\n" + str(missingvalues)
    print(Facilitymissingval)
    FacilityPollution=FacilityPollution.dropna()
    FacilityPollution=FacilityPollution.drop(["Parent_Companies_2014_GHG", "Parent_Companies_2014_TRI","TRI_Air_Emissions_14_in_lbs","TRI_Air_Emissions_13_in_lbs","TRI_Air_Emissions_12_in_lbs","TRI_Air_Emissions_11_in_lbs","TRI_Air_Emissions_10_in_lbs"], axis = 1)
    FacilityPollution=FacilityPollution.rename(columns={"State":"Location"})
    FacilityPollution.to_csv('FacilityPollution.csv')
    
    
if __name__=="__main__":
    AlternativeFuel ()
    AirportUSA ()
    NuclearPlant ()
    AirPollution ()