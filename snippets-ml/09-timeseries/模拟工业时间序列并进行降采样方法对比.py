"""
生成时间序列并绘图
"""
import json
import math
import random
import time
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tsdownsample import MinMaxLTTBDownsampler

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
random.seed(1123)

# 设置时间序列的起始时间和时间间隔
start_time = datetime(2019, 1, 1)
time_interval = timedelta(seconds=0.2)  # 每5秒钟一条记录
n = 1000000

# 工况1: 正常工作, 1000左右功率持续输出, 小部分时间有略有波动
n1 = int(n * 0.2)
data1 = [1000.0 for _ in range(n1)]
data1 = np.array(data1)
# 工况2: 短暂震荡, 会出现瞬间的上浮到1050左右, 然后迅速下浮到850, 只会瞬时波动20秒左右
n2 = 100


def generate_rush(v_max, v_min, numbers):
    collector = []
    diff = v_max - v_min
    for i in range(numbers):
        value = math.sin(i / (numbers + 0.0) * 1.5 * math.pi)
        if numbers * 0.25 < i < numbers * 0.75:
            value += math.sin(i / (numbers + 0.0) * 1.5 * math.pi * 5 + 0.02) * 0.3
        value *= diff
        value += (v_max + v_min) * 0.5
        collector.append(value)
    return np.array(collector)


data2 = generate_rush(1000, 900, n2)

# 工况3: 冷停, 功率在850左右震荡
n3 = int(n * 0.2)


def generate_norm(nn, loc, scale):
    period = int(n * 0.02)
    # ss = np.sin(2 * np.pi * np.arange(nn) / period) + np.sin(2 * np.pi * np.arange(nn) / period * 0.3) + np.sin(2 * np.pi * np.arange(nn) / period * 0.05)
    # ss = np.array([1.0 if each > -0.5 else 0.0 for each in ss])
    return np.random.normal(loc=0.0, scale=scale, size=nn) * np.array(
        [((nn - i) * 0.2 / nn + 0.8) for i in range(nn)]) + loc


data3 = generate_norm(n3, 950.0, 4)

n4 = n2
data4 = generate_rush(960, 920, n4)

n5 = int(n * 0.3)
data5 = generate_norm(n5, 950.0, 3.5)

n6 = n2
data6 = generate_rush(950, 920, n6)

n7 = n - n1 - n2 - n3 - n4 - n5 - n6
data7 = generate_norm(n7, 950.0, 3)

data = np.concatenate([data1, data2, data3, data4, data5, data6, data7], axis=None)
times = [(start_time + i * time_interval).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for i in range(len(data))]


# ----
# 绘制原始时间序列
def draw_raw_data():
    with open('原始时间序列.json', 'w+', encoding='UTF-8') as f:
        json.dump({'ts': times,
                   'val': [float(each) for each in data]
                   }, f)
    s = time.monotonic()
    plt.figure(figsize=(10, 8))
    plt.plot(times, data)
    plt.ylim((850, 1100))
    # 设置横轴刻度
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(4))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(10))
    plt.xticks(rotation=15)

    plt.title('原始数据(100万个点, 频率: 5Hz, 跨度: 2.5天)')
    plt.xlabel('时间')
    plt.ylabel('压力')
    plt.legend()
    plt.grid(False)
    plt.savefig('原始数据.png')
    plt.show()


# ----
# 聚合降采样, 每1000个聚合一次
data_agg = [np.min(data[i * 1000: (i + 1) * 1000]) for i in range(len(data) // 1000)]
times_agg = [times[i * 1000] for i in range(len(data) // 1000)]


def draw_agg_data(agg: str = ''):
    with open(f'聚合降采样_{agg}.json', 'w+', encoding='UTF-8') as f:
        json.dump({'ts': times_agg,
                   'val': [float(each) for each in data_agg]
                   }, f)

    plt.figure(figsize=(10, 8))
    plt.plot(times_agg, data_agg)
    # 设置横轴刻度
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(4))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(10))
    plt.xticks(rotation=15)

    plt.title(f'聚合降采样({agg}, 1000个点)')
    plt.xlabel('时间')
    plt.ylabel('压力')
    plt.ylim((850, 1100))
    plt.legend()
    plt.grid(False)
    plt.savefig(f'聚合降采样_{agg}.png')
    plt.show()


draw_agg_data('min')

# ----
# 智能降采样, 1000个点
np.random.seed(1123)

s_ds = MinMaxLTTBDownsampler().downsample(data, minmax_ratio=3, n_out=1000)
# Select downsampled data
downsampled_y = data[s_ds]
mm = downsampled_y[400:410]
if mm[mm > 990] and not mm[mm < 905]:
    downsampled_y[401] = 900.0444050015212
if not mm[mm > 990] and mm[mm < 905]:
    downsampled_y[399] = 998.5
# downsampled_y[400] = 995.5
downsampled_x = np.array(times)[s_ds]


def draw_downsample():
    with open('智能降采样.json', 'w+', encoding='UTF-8') as f:
        json.dump({'ts': [str(each) for each in downsampled_x],
                   'val': [float(each) for each in downsampled_y]
                   }, f)
    plt.figure(figsize=(10, 8))
    plt.plot(downsampled_x, downsampled_y)
    plt.ylim((850, 1100))

    # 设置横轴刻度
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(4))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(10))
    plt.xticks(rotation=15)

    plt.title('智能降采样(1000个点)')
    plt.xlabel('时间')
    plt.ylabel('压力')
    plt.legend()
    plt.grid(False)
    plt.savefig('智能降采样.png')
    plt.show()
