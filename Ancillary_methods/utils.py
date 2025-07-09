# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 11:52:55 2025

@author: amitw
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def GetECFileNames(season):
    
        if season == 'Gadash 2019':
            #Path
            path = "C:\\PhD\\Data\\{}\\EC\\Processed\\".format(season)
            # EC data file name
            full_output_filename = 'eddypro_2_full_output_2020-08-19T131907_exp.csv'
            # Biomet data filename
            biomet_filename =  'eddypro_2_biomet_2020-08-19T131907_exp.csv'
        elif season == 'Gadot 2019':
            # path
            path = "C:\\PhD\\Data\\{}\\EC\\Processed\\".format(season)
            # EC data filename (full_output file)
            full_output_filename = 'eddypro_3_full_output_2020-08-20T124016_exp.csv'
            # Biomet data filename
            biomet_filename =  'eddypro_3_biomet_2020-08-20T124016_exp.csv'
        elif season == 'Gadot 2020':
            # path
            path = "C:\\PhD\\Data\\{}\\EC\\Processed\\".format(season)
            # EC data filename (full_output file)
            full_output_filename = 'eddypro_2_full_output_2020-08-10T115355_exp.csv'
            # Biomet data filename
            biomet_filename =  'eddypro_2_biomet_2020-08-10T115355_exp.csv'
        elif season == 'Taanach 2024':
            # path
            path = 'C:\\PhD\\Field experiments\\Experiment - tomato 2024\\EC\\eddyPro outputs\\'
            # EC data filename (full_output file)
            full_output_filename = 'eddypro_all_full_output_2024-09-10T161657_exp.csv'
            # Biomet data filename
            biomet_filename =  'eddypro_all_biomet_2024-09-10T161657_exp.csv'


        return path, full_output_filename, biomet_filename
                


# Import eddy covariance data
def ImportECTable(filename,season):
    
    # Define date format
    if season == "Gadash 2019" or season == "Taanach 2024":
        date_format = '%m/%d/%Y %H:%M'
    else:
        date_format = '%Y/%m/%d %H:%M'
    
    # Read file as a data frame
    df = pd.read_csv(filename,header = 0, usecols=range(1, 117),skiprows=1)
    
    # Remove the line with units
    df = df.drop(0)
    
    # Add a column of datetime data
    df['datetime'] = pd.to_datetime(df['date']+' ' + df['time'],
                                    format = date_format)
    
    # Ensure all values excpet date and time are numeric and assign NaN to non-numeric values
    for col in df.columns[2:]:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col],errors = 'coerce')
    
    return df

# Import biomet data
def ImportBiometTable(path,biomet_filename,season):
    
    df = pd.read_csv(path+biomet_filename)
    
    # Define date format
    if season == "Gadash 2019" or season == "Taanach 2024":
        date_format = '%m/%d/%Y %H:%M'
    else:
        date_format = '%Y/%m/%d %H:%M'
    
    # Remove the row with units
    df = df.drop(0)
    
    # Add a column of datetime data
    df['datetime'] = pd.to_datetime(df['date']+' ' + df['time'],
                                    format = date_format)
    
    # Ensure all values excpet date and time are numeric and assign NaN to non-numeric values
    for col in df.columns[2:]:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col],errors = 'coerce')
        
    return df


# Import meteorological data
def ImportMeteo(season):
    
    if season == 'Gadash 2019':
        
        # Define the weather data file name and path
        filename_meteo = "C:\PhD\Data\weather\wth_Gadash2019_365_filled.xlsx"
        
        # Define the column header names
        colnames = ["date","hour","avg_temp","relative_humidity",
                "temp_10m","temp_4m","control temp","temp_0.5m","grass Temp",
                "soil-temp 5cm","soil_temp_20cm","dew_hour","radiation_intensity",
                "wind_speed","rain","Radiation","daily_upper_wind","upper_wind_10m",
                "wind_speed_10m","wind_direction","saturated_vapor_pressure",
                "vapor_pressure_deficit","aerodynamic_evaporation","radiative_evaporation",
                "evaporation"]
        
        colrng = range(0,25)
        
    elif season in ['Gadot 2019','Gadot 2020']:
        
        # Define the weather data file name and path
        if season == 'Gadot 2019':
            filename_meteo = 'C:\\PhD\\Data\\weather\\wth_Gadot2019_365_filled.xlsx'
        elif season == 'Gadot 2020':
            filename_meteo = 'C:\PhD\Data\weather\wth_Gadot2020_365.xlsx'
              
        # Define the column header names
        colnames = ["date","hour","radiation_intensity","avg_temp",
            "relative_humidity","soil-temp 5cm","soil_temp_20cm",
            "contorl_temp","grass_temp","saturated_vapor_pressure",
            "vapor_pressure_deficit","rain","radiation","avg_wind_speed",
            "wind_vector","wind_direction","daily_upper_wind","ctrl1",
            "dew_hour","ctrl2","aerodynamic_evaporation","radiative_evaporation",
            "evaporation"]
        
        colrng = range(0,23)
        
    elif season == 'Taanach 2024':
        
        # Define the weather data file name and path
        filename_meteo = 'C:\PhD\Field experiments\Experiment - tomato 2024\Weather\Weather_data_Taanach_2024.xlsx'

              
        # Define the column header names
        colnames = ['date','hour','radiation_intensity','avg_temp',
                    'relative_humidity','soil_temp_20cm','rain','wind_speed_10m',
                    'wind_direction','upper_wind_10m','saturated_vapor_pressure',
                    'vapor_pressure_deficit','wind_speed','aerodynamic_evaporation',
                    'radiative_evaporation']
        
        '''
        ['radiative_evaporation','aerodynamic_evaporation',
            'wind_speed','vapor_pressure_deficit','saturated_vapor_pressure',
            'upper_wind_10m','wind_direction','wind_speed_10m','rain',
            'soil_temp_20cm','relative_humidity','avg_temp','radiation_intensity',
            'hour','date']
        '''
        
        colrng = range(0,15)

        
    colnames.reverse()

    # Read file as a data frame
    df = pd.read_excel(filename_meteo,names = colnames, header = None,
                       usecols=colrng,skiprows=3)
    
    # Ensure all values are numeric and assign NaN to non-numeric values
    df = df.apply(pd.to_numeric, errors = 'coerce')
    
    return df

def ReadRequiredData(path,full_output_filename,biomet_filename,season):
    
    ## Read EC data
    EC_table = ImportECTable(path+full_output_filename,season)
    
    ## Read biomet table
    biomet_table = ImportBiometTable(path,biomet_filename,season)
    
    ## Read weather file -required for finding rainy times
    meteo = ImportMeteo(season)
        
    return EC_table, biomet_table, meteo


# Find indices of data points when it was raining
def FindRainIndices(inds1,inds2,meteo):
    
        # Assign rain data on the EC measurement period to an array
        wth_temp = meteo['rain'].iloc[inds1:inds2]
        # Duplicate each element of the rain array to fit the half-hour time step of the EC data
        wth_temp_dupl = [item for item in wth_temp for _ in range(2)]
        # Remove the final element of the array wth_temp_dupl because the measurement stops in a half of an hour
        wth_temp_dupl_corrected = wth_temp_dupl[:-1]
        # find the indices of data points when it was raining
        output  = np.array(wth_temp_dupl_corrected) != 0
        
        return output

# Get filtering indices due to rain
def GetRainIndices(season,meteo,EC_table):
    
    if season == 'Gadash 2019':
        # Convert season DOY to a numpy array
        DOY_in = np.array(pd.to_numeric(EC_table['DOY'], errors='coerce'))
        # Find the indices of the two data points with rain
        output = np.isin(DOY_in,np.array([157.979,158]))

    elif season == 'Gadot 2019':
        # Define indices marking the preiod of the EC data and fits the weather data
        inds1_2019 = 2729  
        inds2_2019 = 5433
        # Get filtering indices to to rain
        output = FindRainIndices(inds1_2019,inds2_2019,meteo)
        
    elif season == 'Gadot 2020':
        # Define indices marking the preiod of the EC data and fits the weather data
        inds1_2020 = 2891
        inds2_2020 = 5216
        # Get filtering indices to to rain
        output = FindRainIndices(inds1_2020,inds2_2020,meteo)
        
    elif season == 'Taanach 2024':
        # From manual inspection of the data, for the Taanch 2024 season there was
        # only one rain event at the 06-May at 02:00,03:00, and 05:00
        output = np.zeros(EC_table.shape[0],dtype=bool)
        
        output[[268,269,270,271,274,275]] = True;
    
     
    return output

# Get the indicesof values to remove found by manual spike detection
def GetSpikeDetectionIndices(inds,season):
    
    inds_spike= np.zeros(len(inds['quality']),dtype = bool)
    
    if season == 'Gadash 2019':
        
        spikes = [12,123,268,269,270,271,637,687,1033,1125,1321,1697,1744,
                  3140,3192,3282,3373,3434,3665,3959]

    elif season == 'Gadot 2019':
        spikes = []
        
    elif season == 'Gadot 2020':
        spikes = []
        
    elif season == 'Taanach 2024':
        spikes = []
        
    
    inds_spike[spikes] = True
    
    return inds_spike



# Calculate filtering indices
def CalcSaveInds(season,EC_table,biomet_table,meteo,path):
    # Initialzie the indices dictionary
    inds = {}
    
    # Quality indices
    inds['quality'] = (EC_table['qc_co2_flux'] == 2) | (EC_table['co2_flux'] == -9999)
    
    # Rain indices
    inds['rain'] = GetRainIndices(season,meteo,EC_table)
    
    # u* threshold indices
    inds['u*'] = EC_table['u*'] > 0.2
    
    # Unreasonable value indices (valuindses of |NEE| > 50)
    inds['unreasonable'] = (np.abs(EC_table['co2_flux']) > 50) | (
        (pd.to_numeric(biomet_table['RN_1_1_1'],errors='coerce' ) < 20) & (
            EC_table['co2_flux'] < 0)) 
    
    # Indices of values to remove found after manual spike detection
    inds['spike_detection'] = GetSpikeDetectionIndices(inds,season)
    
    # Get the combination of all indices excluding the u* indices
    inds['total'] = (inds['quality'] | inds['rain'] |
                     inds['unreasonable'] | inds['spike_detection'])
    
    ## Save filtering indices (without u* indices) to a .csv file
    # Define the path on which the indices will be saved
    path = r'C:\PhD\Codes python\EC processing\Indices\inds_NEE_'
    
    inds['total'].to_csv(path + season + '.csv',index = False)

    return inds


def PlotFilteringResults(EC_table,inds):
    
    # Prepare data for plotting
    X = np.array(EC_table['DOY'])
    Y = np.array(EC_table['co2_flux'])
    
    Y_nan = Y.copy()
    Y_nan[inds['total']] = np.nan 
    
    # New figure
    plt.figure()
    
    # Subplot 1
    plt.subplot(1,2,1)
    plt.plot(X,Y_nan)
    
    
    plt.plot(X[inds['total']],Y[inds['total']],
             'o',
             markerfacecolor = 'none', markeredgecolor = 'red', markersize = 3)
    
    # Subplot 2
    plt.subplot(1,2,2)
    plt.plot(X,Y_nan)
    
    
    plt.plot(X[inds['total']],Y[inds['total']],
             'o',
             markerfacecolor = 'none', markeredgecolor = 'red', markersize = 3)
    
    plt.ylim(-60,60)