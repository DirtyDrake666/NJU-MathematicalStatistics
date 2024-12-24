import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

# 读取CSV文件
data = pd.read_csv('covid.csv')

# 获取目标列（假设为第11列）
column = data.iloc[:, 11]

# 计算偏度
skewness = stats.skew(column)
print(f"数据的偏度值: {skewness}")

# 计算峰度
kurtosis = stats.kurtosis(column)
print(f"数据的峰度值: {kurtosis}")

# 偏度显著性检验（D’Agostino’s K-squared检验，结合偏度和峰度）
stat, p_value = stats.normaltest(column)
print(f"偏度显著性检验的统计量: {stat}, p值: {p_value}")

# 生成QQ图
plt.rc("font", family="SimHei")  # 增加这一行设置字体
plt.rc('axes', unicode_minus=False)
stats.probplot(column, dist="norm", plot=plt)
plt.title('QQ图')
plt.show()

# 根据偏度、峰度和p值来判断正态性
print("\n正态性分析结果：")
if p_value < 0.05:
    print("拒绝原假设，数据显著偏离正态分布。")
else:
    print("不能拒绝原假设，数据接近正态分布。")

# 对偏度和峰度进行解读
if skewness > 0:
    print("数据右偏（正偏）。")
elif skewness < 0:
    print("数据左偏（负偏）。")
else:
    print("数据没有明显偏斜，接近正态分布。")

if kurtosis > 3:
    print("数据具有较高的峰度，分布较尖锐。")
elif kurtosis < 3:
    print("数据具有较低的峰度，分布较平坦。")
else:
    print("数据的峰度接近正态分布。")

# -------------------------- 经验分布函数 (ECDF) 与 正态分布 CDF 比较 --------------------------

# 计算经验分布函数（ECDF）
sorted_column = np.sort(column)  # 对数据排序
ecdf = np.arange(1, len(sorted_column) + 1) / len(sorted_column)  # 经验分布函数

# 计算正态分布的CDF（假设数据符合正态分布）
mean = np.mean(column)
std = np.std(column)
normal_cdf = stats.norm.cdf(sorted_column, loc=mean, scale=std)

# 绘制ECDF与正态分布CDF的比较
plt.figure(figsize=(10, 6))
plt.plot(sorted_column, ecdf, label='经验分布函数 (ECDF)', color='blue')
plt.plot(sorted_column, normal_cdf, label='正态分布 CDF', color='red', linestyle='--')
plt.title('经验分布函数 (ECDF) 与 正态分布 CDF 比较')
plt.xlabel('数据值')
plt.ylabel('累积概率')
plt.legend()
plt.grid(True)
plt.show()
