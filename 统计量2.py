import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# 读取CSV文件
data = pd.read_csv('covid.csv')

# 选择第一列数据
column = data.iloc[:, -1]

# 设置区间范围
min_val = column.min()
max_val = column.max()
num_bins = 10
bin_width = (max_val - min_val) / num_bins
bins = [min_val + i*bin_width for i in range(num_bins+1)]

plt.rc("font", family="SimHei")  # 增加了这一行

# 画直方图
plt.hist(column, bins=bins, color='skyblue', edgecolor='black', density=True)

# 绘制数据的核密度估计（Kernel Density Estimation, KDE）曲线




# 添加标签和标题
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('测试结果阳性人数')

# 显示图例
plt.legend()

# 显示图形
plt.show()
