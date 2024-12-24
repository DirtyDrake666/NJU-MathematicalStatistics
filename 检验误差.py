import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence
import scipy.stats as stats

# 1. 加载数据
df = pd.read_csv('covid.csv')

# 2. 选择特征和目标变量
X = df[['hh_cmnty_cli']]  # 特征变量
y = df['tested_positive']  # 目标变量

# 3. 添加常数项（截距）到特征矩阵
X = sm.add_constant(X)

# 4. 创建OLS模型并拟合
model = sm.OLS(y, X)
results = model.fit()

# 获取拟合值和残差
fitted_values = results.fittedvalues
residuals = results.resid

# 获取标准化残差和Cook's 距离
influence = OLSInfluence(results)
cooks_distance = influence.cooks_distance[0]
standardized_residuals = influence.resid_studentized_internal

# 绘图：拟合值对残差图
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.scatter(fitted_values, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Fitted Values vs Residuals')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# 绘图：Normal Q-Q 图
plt.subplot(2, 2, 2)
# 使用 scipy.stats.norm 生成标准正态分布的 Q-Q 图
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q Plot')

# 绘图：Scale-Location 图（标准化残差与拟合值的平方根）
plt.subplot(2, 2, 3)
plt.scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Scale-Location Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Sqrt(|Standardized Residuals|)')

# 绘图：Cook's 距离图
plt.subplot(2, 2, 4)
plt.stem(np.arange(len(cooks_distance)), cooks_distance, markerfmt='o')  # 删除 use_line_collection 参数
plt.title("Cook's Distance")
plt.xlabel('Index')
plt.ylabel("Cook's Distance")

plt.tight_layout()
plt.show()

# 获取回归系数的置信区间（默认为95%置信区间）
conf_interval = results.conf_int(alpha=0.05)

# 打印回归系数及其置信区间
print("回归系数及其置信区间：")
for i, col in enumerate(conf_interval.index):
    print(f"{col}: {results.params[col]:.4f}  [{conf_interval.iloc[i, 0]:.4f}, {conf_interval.iloc[i, 1]:.4f}]")

