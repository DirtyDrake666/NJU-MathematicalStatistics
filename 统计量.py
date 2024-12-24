import pandas as pd

# 读取CSV文件
data = pd.read_csv('covid.csv')

# 生成描述性统计信息
summary = data.describe()

# 提取最小值、最大值、均值和标准差
min_val = summary.loc['min']
max_val = summary.loc['max']
mean_val = summary.loc['mean']
std_val = summary.loc['std']

# 合并成一个表格
stats_df = pd.concat([min_val, max_val, mean_val, std_val], axis=1)
stats_df.columns = ['Min', 'Max', 'Mean', 'Std']

# 保存为表格
stats_df.to_csv('stats_summary.csv')

print(stats_df)