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

# 获取影响分析对象
influence = model.get_influence()

# 计算 Cook's Distance
cooks_d = influence.cooks_distance[0]

# 计算 Leverage（帽子值）
leverage = influence.hat_matrix_diag

# 获取标准化残差
standardized_residuals = influence.resid_studentized_internal

# 设置 Cook's Distance 和 Leverage 的阈值
cook_threshold = 4 / len(X_selected)  # 常用的阈值 4/n
leverage_threshold = 2 * (len(selected_features) / len(X_selected))  # 常用的 Leverage 阈值

# 找到强影响点
high_influence_points = np.where(cooks_d > cook_threshold)[0]
high_leverage_points = np.where(leverage > leverage_threshold)[0]

# 输出强影响点信息
print(f"Strong influence points based on Cook's Distance: {high_influence_points}")
print(f"High leverage points based on Leverage: {high_leverage_points}")

