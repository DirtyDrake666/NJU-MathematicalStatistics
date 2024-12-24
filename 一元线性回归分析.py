import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import statsmodels.api as sm

# 1. 加载数据
df = pd.read_csv('covid.csv')

# 2. 检查数据（确保你知道数据的结构）
print(df.head())

# 3. 计算Pearson相关系数
corr, p_value = pearsonr(df['hh_cmnty_cli'], df['tested_positive'])

# 5. 打印相关系数和p值
print(f'Pearson correlation: {corr:.2f}')
print(f'p-value: {p_value:.3e}')

# 6. 一元线性回归分析（使用statsmodels）
# 选择特征和目标变量
X = df[['hh_cmnty_cli']]  # 特征变量
y = df['tested_positive']  # 目标变量

# 添加常数项（截距）到特征矩阵
X = sm.add_constant(X)

# 创建OLS模型并拟合
model = sm.OLS(y, X)
results = model.fit()

# 打印回归结果
print(results.summary())

# 7. 绘制回归线
plt.figure(figsize=(10, 6))
plt.scatter(df['hh_cmnty_cli'], df['tested_positive'], alpha=0.6, label='Data points')
plt.plot(df['hh_cmnty_cli'], results.fittedvalues, color='red', label='Regression Line')
plt.title('Linear Regression: hh_cmnty_cli vs tested_positive')
plt.xlabel('hh_cmnty_cli')
plt.ylabel('tested_positive')
plt.legend()
plt.grid(True)

# 显示Pearson相关系数和p值
plt.figtext(0.15, 0.75, f'Pearson correlation: {corr:.2f}', fontsize=12, color='blue')
plt.figtext(0.15, 0.70, f'p-value: {p_value:.3e}', fontsize=12, color='blue')

# 显示图像
plt.show()
