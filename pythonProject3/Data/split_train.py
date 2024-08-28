import pandas as pd
from sklearn.model_selection import train_test_split

# 读取new_train.tsv
df = pd.read_csv('data/new_train.tsv', sep='\t')

# 使用train_test_split进行划分
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# 保存划分后的训练集和验证集
train_df.to_csv('data/new_train.tsv', index=False, header=True, sep='\t', encoding='utf-8')
val_df.to_csv('data/new_val.tsv', index=False, header=True, sep='\t', encoding='utf-8')

print("数据集划分完成！")
