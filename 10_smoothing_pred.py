import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

df = pd.read_excel('./4_pred_results/both_5.xlsx',engine='openpyxl')

u_real = df['u_real']
u_pred = df['u_pred_modified']
# method1
# 移动平均函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 应用移动平均
window_size = 15
u_pred_smoothed = moving_average(u_pred, window_size)

# 绘图
plt.plot(u_real, label='real')
plt.plot(np.arange(window_size-1, len(u_pred)), u_pred_smoothed, label='pred', color='red')

plt.legend()
plt.xlabel('time(0.1s)')
plt.ylabel('voltage angles(rad)')
plt.show()

u_real_trimmed = u_real[(window_size - 1):]

l2_error = np.sqrt(np.sum((u_pred_smoothed - u_real_trimmed)**2))
print("L2 Error:", l2_error)

rmse = np.sqrt(mean_squared_error(u_real_trimmed, u_pred_smoothed))
print("rmse",rmse)
# u_pred_smoothed_df = pd.DataFrame(u_pred_smoothed,columns=['u_pred_smoothed'])
# u_pred_smoothed_df.to_excel('4_pred_results/smoothed_5.xlsx')

# method2
# data_series = pd.Series(u_pred)
# u_pred_smoothed = data_series.ewm(alpha=0.3).mean()

# # 绘图
# plt.plot(u_pred, label='u_pred_modified')
# plt.plot(u_pred_smoothed, label='u_pred_smoothed', color='red')
# plt.legend()
# plt.show()

# method3
# u_pred_smoothed = savgol_filter(u_pred, window_length=11, polyorder=2)

# # 绘图
# plt.plot(u_pred, label='u_pred_modified')
# plt.plot(u_pred_smoothed, label='u_pred_smoothed', color='red')
# plt.legend()
# plt.show()