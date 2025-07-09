# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:44:38 2025

@author: amitw

A class that writes the filtered eddy-covariance data to a text (.txt) file in
a format suitable for the REddyProc R pakcage

"""

import pandas as pd
import numpy as np
import os

class WriteFilteredData:
    
    def __init__(self, inds = None,season = None,meteo = None,
                 biomet_table = None, EC_table = None):
        self.inds = inds
        self.season = season
        self.meteo = meteo
        self.biomet_table = biomet_table
        self.EC_table = EC_table
        # Get the indices of the wheather data points that corresponds tothe EC data
        self.inds1, self.inds2 = self.GetECPeriod()
        # In this function, the meteorological data which is originally in hourly
        # time steps is converted to half hour time steps by duplicating each
        # hourly input
        self.GetMeteo()
        
        #Create the R input file
        self.CreateRFileInput()
        
        # If necessary complete the input data table to an extent of 3 month
        # where all the data on additional days is -9999
        if (self.df.shape[0]-1) < (3*30*48):  # 3*30*48 is the amount of elememnts for a 3 months period 
            self.CompleteDataTo3Month()
        else:
            self.df_extended = self.df
            
        # self.SaveRInputTable()
        
        
    def GetECPeriod(self):
        '''
        This function defines the indices indicating the eddy covariance measurement period 
        inside the preiod of the meteotological data
        '''
        
        if self.season == 'Gadash 2019':        
            # Define indices for the beginning and end of the eddy covariance
            # meaurement period
            inds1 = 2920
            inds2 = 4929
        elif self.season == 'Gadot 2019':
            inds1 = 2730 # Note that in the EC data it starts frpm 17:30 while in the weather file the data is hourly, hence it starts from 17:00
            inds2 = 5433 # Ends at 13-Aug-1029, 20:00
        elif self.season == 'Gadot 2020':
            inds1 = 2892
            inds2 = 5216 
        elif self.season == 'Taanach 2024':
            inds1 = 2892
            inds2 = 4450
            
        return inds1, inds2
    
    def GetMeteo(self):
        # Define names for running hte for loop
        meteo_str = ['radiation_intensity','avg_temp','vapor_pressure_deficit']
        var_names = ['Rg','Ta','VPD']
        
        # Definethe dictionery object containing duplicated meteo data
        duplicated = {}
        
        # Duplcate the data for each of the required meteorological vriables
        for i in range(len(meteo_str)):
            temp_meteo =  self.meteo[meteo_str[i]].iloc[self.inds1-1:self.inds2]
            
            duplicated[var_names[i]] = [x for x in temp_meteo for _ in range(2)]
         
        # Assign duplicted values to dedicated arrays
        self.Rg = duplicated['Rg'][:-1]
        self.Ta = duplicated['Ta'][:-1]
        self.VPD = duplicated['VPD'][:-1]
        

    
    def ConvertHourToFraction(self,EC_table):
        '''
        This function comvert the hour to a fraction of 24 hours as used in the 
        R input files
        '''
        hours = EC_table['datetime'].dt.hour
        
        minute_fraction = EC_table['datetime'].dt.minute/60
        
        time_fraction = hours + minute_fraction
        
        return time_fraction
    
    
    def ApplyIndicesOnNEE(self,EC_table):
        
        # Create a copy of the EC table
        temp = EC_table.copy()
        
        temp.loc[self.inds['total'], 'co2_flux'] = -9999
        
        return temp['co2_flux']
    
    def GetSoilTemp(self):
    
        mean_soil_temp = self.biomet_table[['TS_1_1_1','TS_1_1_3','TS_1_1_5','TS_1_1_7']].mean(axis=1)
    
        output = mean_soil_temp.where(mean_soil_temp == -9999,mean_soil_temp-273.15)
        
        output = [i+273.15 if i<-100 else i for i in output]
        
        output = [round(x,2) for x in output]
        
        return output
    
    def CreateRFileInput(self):
        
        # Save the dataframe with eddy covariance as a new varible
        EC_table = self.EC_table
        
        self.df = pd.DataFrame({
        'Year': EC_table['datetime'].dt.year,
        'DoY': round(np.floor(EC_table['DOY'])),
        'Hour': round(self.ConvertHourToFraction(EC_table),2),
        'NEE': round(self.ApplyIndicesOnNEE(EC_table),2),
        'LE': round(EC_table['LE'],2),
        'H': round(EC_table['H'],2),
        'Rg': np.round(self.Rg,2),
        'Tair': np.round(self.Ta,2),
        'Tsoil': self.GetSoilTemp(),
        'rH': round(self.EC_table['RH'],2),
        'VPD': np.round(self.VPD,2),
        'Ustar': round(self.EC_table['u*'],2),   
        
        })
        
        # Assign units
        units = ['-','-','-','umolm-2s-1','Wm-2','Wm-2','Wm-2',
            'degC','degC','%','hPa','ms-1']
        
        # Convert unit strings list to a DataFrame
        row_df = pd.DataFrame([units], columns=self.df.columns)
        
        # Add the units at the second row of the dataframe
        self.df = pd.concat([row_df,self.df.iloc[0:]],ignore_index=True)
        
         

    def CompleteDataTo3Month(self):
        
        df_extended = self.df.copy()
        
        # Calculate number of elemnts to add
        n_elements2add = 3*30*48 - self.df.shape[0] + 10 # I add extra 10 days to be on the safe side
        
        for i in range(n_elements2add):
            if df_extended['Hour'].iloc[-1] != 23.5:
                hour_temp = df_extended['Hour'].iloc[-1] + 0.5
                DOY_temp = df_extended['DoY'].iloc[-1]
                if df_extended['DoY'].iloc[-1] == 365:
                    year_temp = df_extended['Year'].iloc[-1] + 1
                else:
                    year_temp = df_extended['Year'].iloc[-1]
                    
            else:
                hour_temp = 0
                DOY_temp = df_extended['DoY'].iloc[-1] + 1

            # Create a new dataframe row to concatanate at the end of the eisting dataframe
            new_row_df = pd.DataFrame([[year_temp,DOY_temp,hour_temp]+[-9999]*(df_extended.shape[1]-3)],
                                      columns=self.df.columns)
            
            # Add the units at the second row of the dataframe
            df_extended = pd.concat([df_extended, new_row_df],ignore_index=True)
        
        self.df_extended = df_extended
        
        
        
    def WriteRInputTable(self):
        
        # Define the path of the R imput files
        path_io = 'C:\PhD\Codes python\EC processing\R input output files'
        
        # Modify season name to include '_'
        season = self.season.replace(' ','_')
        
        # Define input file name
        input_filename = F'{season}_NEE_gap_filling_input.txt'
        
        # Save data (after initial filtering) to a text file in format
        # suitable fo th R code
        self.df_extended.to_csv(os.path.join(path_io,input_filename), sep='\t', index=False)
        
        # Save the season name in a CSV file to make it available for the R code
        fn = r'C:\PhD\Codes python\EC processing\R input output files\season_name.txt'

        with open(fn,'w') as f:
            f.write(self.season)
    
            
    
        
        
    
        
        
