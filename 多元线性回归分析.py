import pandas as pd
import statsmodels.api as sm
# 读取数据
data = pd.read_csv('covid.csv')

# 选择因变量和自变量
X = data.iloc[:, :-1]
y = data['tested_positive']

# 添加常数项
X = sm.add_constant(X)
# 拟合OLS回归模型
model = sm.OLS(y, X).fit()

# 查看回归结果
print(model.summary())


# 逐步回归函数：基于AIC进行特征选择
def stepwise_selection(X, y, threshold_in=0.01, threshold_out=0.05, verbose=True):
    """
    基于AIC的逐步回归算法，选择最优特征
    参数：
    X: 特征矩阵
    y: 目标变量
    threshold_in: 进入模型的p-value阈值
    threshold_out: 离开模型的p-value阈值
    verbose: 是否打印每一步的变化

    返回：
    最终选择的特征列
    """
    initial_list = X.columns.tolist()  # 初始所有特征
    best_aic = float('inf')  # 初始化最小AIC为正无穷
    while (len(initial_list) > 0):
        changed = False
        # 回归分析
        model = sm.OLS(y, X[initial_list]).fit()
        # 获取模型的AIC
        current_aic = model.aic

        # 如果当前AIC较小，更新最优AIC
        if current_aic < best_aic:
            best_aic = current_aic
            selected_features = initial_list.copy()  # 记录当前最优特征集
            print(f"Best AIC updated: {best_aic} with features {selected_features}")

        # 获取每个特征的p值
        pvalues = model.pvalues[1:]  # 排除常数项的p值
        max_p_value = pvalues.max()  # 最大的p值

        if max_p_value > threshold_out:
            # 如果最大p值超过阈值，去掉该特征
            changed = True
            excluded_feature = pvalues.idxmax()
            initial_list.remove(excluded_feature)
            if verbose:
                print(f"Dropped {excluded_feature} with p-value {max_p_value}")

        if not changed:
            break  # 如果没有特征被剔除，退出循环

    # 返回最优特征
    return selected_features


# 执行逐步回归，选择最优特征
selected_features = stepwise_selection(X, y)

# 显示最终选择的特征
print("Selected features:", selected_features)

# 使用最终选择的特征拟合回归模型
X_selected = X[selected_features]

# 再次添加常数项（截距）
X_selected = sm.add_constant(X_selected)

# 拟合OLS回归模型
model = sm.OLS(y, X_selected).fit()

# 输出回归结果
print(model.summary())
