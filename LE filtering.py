## lE filtering

# This code is for filtering of lE data obtained by eddy covariance system.
# This code plots the energy balance scatter plot before and after filtering.
# The output of this code is an indices array for data ppoints that should
# be filtered out.The indice are saved in C:\PhD\Codes\EC analysis\

# The filtering crititions are:
# *Quality check
# *Energy balance
# *rain events

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess

from Ancillary_methods.WriteFilteredData import WriteFilteredData
from Ancillary_methods.utils import GetECFileNames,  ReadRequiredData
from Ancillary_methods.utils import  GetEBComponents, CalcSaveIndsLE, PlotEnergyBalance 
    

if __name__ == "__main__":
    
    # Define season
    season = 'Gadot 2020'
    
    ## Define the path and filename of the data
    path, full_output_filename, biomet_filename = GetECFileNames(season)
    
    ## Read required data - eddy covariance, biomet, and meteorlogical data
    EC_table, biomet_table, meteo = ReadRequiredData(path,
                                     full_output_filename,biomet_filename,season)
    
    ## Get energy balance components
    LE_raw, LE_quality, H, Rn, G, available_energy, consumed_energy = (
        GetEBComponents(EC_table, biomet_table, season)
        )
  
    ## Calculate filtering indices and save them as a .csv file
    inds = CalcSaveIndsLE(
        season, LE_raw, LE_quality, meteo,
        available_energy, consumed_energy,EC_table
        )
    
    ## Plot energy balance
    if False:
        # Visualize the Energy balance results
        PlotEnergyBalance(inds,available_energy, consumed_energy,inds['total'])
    
    ## Create an input for the REddyProc R script
    WW = WriteFilteredData(inds,season,meteo,biomet_table,EC_table,flux_type = 'LE')
    # Write the NEE gap filling input file
    WW.WriteRInputTable()
    
    ## Run the R script
    rscript_exe = "C:/Program Files/R/R-4.3.3/bin/Rscript.exe"
    
    r_script_path = r"C:/PhD/Codes R/Reddyproc/lE_gap_filling.R"
    
    result = subprocess.run([rscript_exe, r_script_path], capture_output=True,text=True)
    print(result)
    print(result.stdout)

    # Read Reddyproc output
    # Read the file, specifying tab separator and handling NA values
    output_filename = rf"C:\PhD\Codes python\EC processing\R input output files\{season.replace(' ','_')}_filled_LE.txt"
    
    df = pd.read_csv(output_filename, sep='\t', na_values=["NA"])







    
