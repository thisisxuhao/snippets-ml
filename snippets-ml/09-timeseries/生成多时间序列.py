"""
生成时间序列并绘图
"""
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

random.seed(1123)

# 设置时间序列的起始时间和时间间隔
start_time = '2025-01-01 00:00:00'
time_interval = '1min'  # 每分钟记录一次数据

n = 2880
# 生成时间索引，这里以一天为例，即n个时间点
time_index = pd.date_range(start=start_time, periods=n, freq=time_interval)
time_index_datetime_list = time_index.to_pydatetime().tolist()
print(time_index_datetime_list[:3])
# 定义六个传感器的周期和起始相位
periods = [120, 240, 360, 480, 600, 720]  # 分别对应2小时、4小时、6小时、8小时、10小时、12小时的周期
phases = [0, 30, 60, 90, 120, 150]  # 起始相位，单位为分钟

# 生成六个时间序列数据
np.random.seed(0)  # 设置随机种子，保证结果可复现
sensor_values = []
for period, phase in zip(periods, phases):
    # 生成具有周期性和随机波动的数据
    values = 50 + \
             random.randint(0, 10) * np.sin(2 * np.pi * (np.arange(n) - phase) / period / random.randint(5, 15)) +  \
             random.randint(0, 10) * np.sin(2 * np.pi * (np.arange(n) - phase) / period) + \
             random.randint(0, 10) * np.sin(2 * np.pi * (np.arange(n) - phase) / period / random.randint(1, 3)) + \
             random.randint(0, 10) * np.sin(2 * np.pi * (np.arange(n) - phase) / period / random.randint(3, 5)) + \
             random.randint(1, 9) * np.sin(2 * np.pi * (np.arange(n) - phase) / period / random.randint(5, 15)) \
             + np.random.normal(loc=0, scale=1, size=n)

    sensor_values.append(values)

# 构造六个时间序列
sensor_series = [pd.Series(data=values, index=time_index, name=f'Sensor{i+1}') for i, values in enumerate(sensor_values)]

# 绘图
plt.figure(figsize=(12, 8))
for series in sensor_series:
    plt.plot(series, label=series.name)

plt.title('Sensor Time Series with Different Periods and Phases')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
