import numpy as np
import matplotlib.pyplot as plt

def average_filter(data, window_size):
    """
    平均濾波器
    :param data: 要進行濾波的數據
    :param window_size: 滑動窗口的大小
    :return: 濾波後的數據
    """
    filtered_data = []
    for i in range(len(data)):
        if i < window_size - 1:
            filtered_data.append(data[i])
        else:
            filtered_data.append(sum(data[i - window_size + 1:i + 1]) / window_size)
    return filtered_data

r = np.load('./reward.npy')
r = average_filter(r,window_size=100)
plt.plot(r)
plt.show()