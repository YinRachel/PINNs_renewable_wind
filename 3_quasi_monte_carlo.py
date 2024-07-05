import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.interpolate import interp1d
from scipy.stats.qmc import Sobol  # 导入 Sobol 序列生成器
from scipy import stats

def generate_winddata(sampler, idx, sample_size):
    t = np.arange(0, 61, 1)
    mean_wind_speed = 12
    std_dev_wind_speed = 4
    
    # 使用 Sobol 序列生成标准正态分布
    normal_samples = sampler.random(n=sample_size)  # 生成 2^6 = 64 个样本
    z_scores = stats.norm.ppf(normal_samples[:, idx])  # 转换为正态分布
    
    wind_speed = mean_wind_speed + std_dev_wind_speed * z_scores[:len(t)]
    wind_speed = np.maximum(wind_speed, 3)
    
    gust_factor = 1.5
    gust_timing = np.arange(20, 31)
    wind_speed[gust_timing] = wind_speed[gust_timing] * gust_factor
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
    power_output = power_output / 10
    
    origData = power_output
    origiTime = t
    newTime = np.linspace(0, 60, 601)
    
    interpolation_function = interp1d(origiTime, origData, kind='nearest', fill_value='extrapolate')
    newData = interpolation_function(newTime)
    
    return {
        'Time': newTime,
        'PowerOutput': newData
    }
sample_size = 64
# 初始化 Sobol 序列生成器
sobol_sampler = Sobol(d=2)  # 仅使用 2 维

for i in range(300):
    winddata = generate_winddata(sobol_sampler, 0, sample_size)  # 第二个参数 0 用于选择 Sobol 维度
    filename = f'./windData_quasi_300/wind_power_{i+1}.mat'
    structured_data = np.zeros((len(winddata['Time']),), dtype=[('Time', 'f8'), ('PowerOutput', 'f8')])
    structured_data['Time'] = winddata['Time']
    structured_data['PowerOutput'] = winddata['PowerOutput']
    savemat(filename, {'wind_data_table': structured_data})
