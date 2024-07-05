import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.interpolate import interp1d

def generate_winddata():
    t = np.arange(0,61,1)
    mean_wind_speed = 12
    std_dev_wind_speed = 4
    wind_speed = mean_wind_speed + std_dev_wind_speed * np.random.randn(len(t))
    wind_speed = np.maximum(wind_speed,3)
    
    gust_factor = 1.5
    gust_timeing = np.arange(20,31)
    wind_speed[gust_timeing] = wind_speed[gust_timeing] * gust_factor
    wind_speed = pd.Series(wind_speed).rolling(window=3).mean().fillna(wind_speed[0])
    
    cut_in_speed = 4
    cut_out_speed = 25
    rated_speed = 12
    rated_power = 1890
    
    power_output = np.zeros(len(t))
    for i, ws in enumerate(wind_speed):
        if ws < rated_speed:
            power_output[i] = rated_power * (ws - cut_in_speed) / (rated_speed - cut_in_speed)
        else:
            power_output[i] = rated_power
    power_output = power_output/10
  
    
    origData = power_output
    origiTime = t
    newTime = np.linspace(0,60,601)
    
    interpolation_function = interp1d(origiTime, origData, kind='nearest', fill_value='extrapolate')
    newData = interpolation_function(newTime)
    
    # newData = np.interp(newTime, origiTime, origData,'nearest')
    
    # wind_data_table = {
    #     'Time':newTime,
    #     'PowerOutput':newData
    # }
    #print(wind_data_table)
    return {
        'Time':newTime,
        'PowerOutput':newData
    }
    
for i in range(1000):
    winddata = generate_winddata()
    filename = f'./windData_1000/wind_power_{i+1}.mat'
    structured_data = np.zeros((len(winddata['Time']),),dtype=[('Time', 'f8'), ('PowerOutput', 'f8')])
    structured_data['Time'] = winddata['Time']
    structured_data['PowerOutput'] = winddata['PowerOutput']
    savemat(filename, {'wind_data_table': structured_data})
    # savemat(filename,{'wind_data_table': winddata})



            
    
    

