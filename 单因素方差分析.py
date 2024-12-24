import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 加载数据
df = pd.read_csv('covid.csv')

# 将 'wearing_mask_7d' 按四分位数分组
df['mask_group'] = pd.qcut(df['wearing_mask_7d'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 查看分组后的数据分布
print(df['mask_group'].value_counts())

# 创建单因素方差分析模型
model = ols('tested_positive ~ C(mask_group)', data=df).fit()

# 计算ANOVA表
anova_table = anova_lm(model)

# 输出ANOVA表
print(anova_table)

# 执行F检验
f_statistic = anova_table['F'][0]
p_value = anova_table['PR(>F)'][0]

# 输出F值和p值
print("F值:", f_statistic)
print("p值:", p_value)
result = stats.f.ppf(0.05, 3, 1990)
print(result)
# 根据p值判断是否拒绝原假设
alpha = 0.05
if p_value < alpha:
    print("拒绝原假设，'wearing_mask_7d' 对 'tested_positive' 存在显著影响")
else:
    print("接受原假设，'wearing_mask_7d' 对 'tested_positive' 不存在显著影响")
