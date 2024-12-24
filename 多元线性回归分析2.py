import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv('covid.csv')

# 选择所有列，除去最后一列作为自变量，最后一列作为因变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 删除缺失值
X = X.dropna()
y = y[X.index]

# 添加常数项（截距）
X = sm.add_constant(X)

# 使用逐步回归方法选择最优特征（假设你已经运行了这部分代码）
selected_features = ['const', 'cli', 'ili', 'wnohh_cmnty_cli', 'wbelief_masking_effective',
                     'wbelief_distancing_effective', 'wcovid_vaccinated_friends', 'wlarge_event_indoors',
                     'wothers_masked_public', 'wshop_indoors', 'wrestaurant_indoors', 'wworried_catch_covid',
                     'hh_cmnty_cli', 'nohh_cmnty_cli', 'wearing_mask_7d', 'public_transit', 'worried_finances']

# 最终选择的特征
X_selected = X[selected_features]

# 拟合OLS回归模型
model = sm.OLS(y, X_selected).fit()

# 获取回归模型的残差和拟合值
residuals = model.resid
fitted_values = model.fittedvalues

# 计算标准化残差
influence = model.get_influence()
standardized_residuals = influence.resid_studentized_internal

# 步骤1：识别标准化残差绝对值大于2的异常点
outliers = np.where(np.abs(standardized_residuals) > 2)[0]

# 步骤2：去除异常点
X_selected_no_outliers = X_selected.drop(outliers)
y_no_outliers = y.drop(outliers)

# 拟合新的OLS回归模型，去除异常点
model_no_outliers = sm.OLS(y_no_outliers, X_selected_no_outliers).fit()

# 获取新的回归模型的残差和拟合值
residuals_no_outliers = model_no_outliers.resid
fitted_values_no_outliers = model_no_outliers.fittedvalues

# 计算新的标准化残差
influence_no_outliers = model_no_outliers.get_influence()
standardized_residuals_no_outliers = influence_no_outliers.resid_studentized_internal

# 步骤3：绘制新的标准化残差散点图
plt.figure(figsize=(12, 6))

# 残差散点图（不带异常点）
plt.subplot(1, 2, 1)
plt.scatter(fitted_values_no_outliers, residuals_no_outliers, color='blue', edgecolors='k', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted (No Outliers)')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')

# 标准化残差散点图（不带异常点）
plt.subplot(1, 2, 2)
plt.scatter(fitted_values_no_outliers, standardized_residuals_no_outliers, color='blue', edgecolors='k', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Standardized Residuals vs Fitted (No Outliers)')
plt.xlabel('Fitted values')
plt.ylabel('Standardized Residuals')

plt.tight_layout()
plt.show()

# 输出新的回归模型总结
print(model_no_outliers.summary())
