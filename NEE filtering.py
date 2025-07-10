# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:48:44 2025

@author: amitw

NEE filtering (CO2 flux from eddy covariance data)

This code is for filtering of NEE data obtained by eddy covariance system.

The output of this code is an indices array for data points that should
be filtered out. The indices are saved in C:\PhD\Codes\EC analysis\
 
  The filtering crititions are:
* Quality check
* rain events
* u* threshold

"""

import subprocess
import pandas as pd

from Ancillary_methods.WriteFilteredData import WriteFilteredData
from Ancillary_methods.utils import GetECFileNames,  ReadRequiredData, CalcSaveIndsNEE



if __name__ == "__main__":
    
    # Define season
    season = 'Taanach 2024'
    
    ## Define the path and filename of the data
    path, full_output_filename, biomet_filename = GetECFileNames(season)
    
    ## Read required data - eddy covariance, biomet, and meteorlogical data
    EC_table, biomet_table, meteo = ReadRequiredData(path,
                                     full_output_filename,biomet_filename,season)
    
    ## Calculate filtering indices
    inds = CalcSaveIndsNEE(season,EC_table,biomet_table,meteo,path)
    
    ## Create an input for the REddyProc R script
    WW = WriteFilteredData(inds,season,meteo,biomet_table,EC_tableflux_type = 'NEE')
    # Write the NEE gap filling input file
    WW.WriteRInputTable()
    
    ## Run the R script
    rscript_exe = "C:/Program Files/R/R-4.3.3/bin/Rscript.exe"
    
    r_script_path = r"C:/PhD/Codes R/Reddyproc/NEE_u_star_gap_filling_python.R"
    
    result = subprocess.run([rscript_exe, r_script_path], capture_output=True,text=True)
    print(result.stdout)

    # Read Reddyproc output
    # Read the file, specifying tab separator and handling NA values
    output_filename = rf"C:\PhD\Codes python\EC processing\R input output files\{season.replace(' ','_')}_filled_NEE.txt"
    
    df = pd.read_csv(output_filename, sep='\t', na_values=["NA"])

    
    








