import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# 读取CSV文件
data = pd.read_csv('covid.csv')

# 选择第一列数据
column = data.iloc[:, 11]

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
kde = stats.gaussian_kde(column)
x = np.linspace(min_val, max_val, 1000)
plt.plot(x, kde(x), color='red', label='数据的密度估计')

# 计算并绘制正态分布的概率密度函数（PDF）
mean = np.mean(column)
std_dev = np.std(column)
normal_pdf = stats.norm.pdf(x, mean, std_dev)
plt.plot(x, normal_pdf, color='blue', label='正态分布')

# 添加标签和标题
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('担心感染 covid19 加权平均值')

# 显示图例
plt.legend()

# 显示图形
plt.show()

# 计算数据的均值和标准差
mean = np.mean(column)
std_dev = np.std(column)

# 设定置信水平和自由度
confidence_level = 0.95
alpha = 1 - confidence_level
df = len(column) - 1  # 自由度

# 使用 t 分布计算均值置信区间
t_value = stats.t.ppf((1 - alpha / 2), df)
margin_of_error = t_value * std_dev / np.sqrt(len(column))
confidence_interval = (mean - margin_of_error, mean + margin_of_error)

print(f"置信水平为 {confidence_level} 的置信区间为: {confidence_interval}")

# 使用卡方分布计算方差置信区间
chi2_lower = stats.chi2.ppf(alpha / 2, df)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)
var_lower = (df * std_dev**2) / chi2_upper
var_upper = (df * std_dev**2) / chi2_lower

variance_interval = (var_lower, var_upper)

print(f"置信水平为 {confidence_level} 的方差置信区间为: {variance_interval}")
