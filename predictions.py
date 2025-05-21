# 单个文件进行预测(注意RF3中没有使用数据标准化处理，其实准确性挺高的)
import pandas as pd
import joblib
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
# 加载模型
model = load('res/RF3.joblib')

# 原始文件路径
original_file_path = 'pre_sample/DIR-880_ARM_binary_CSV_Pre.csv'
# 读取 CSV 文件
data = pd.read_csv(original_file_path)

features = ['src', 'fname', 'funcaddr', 'keyimportapiinfo', 'netkeywords']
X = data.drop(columns=features) #删除标签列

# 检查数据类型并转换为数值类型
for column in X.columns:   # 此时是包含data中的所有列
    if X[column].dtype == 'object':  # 检查当前列的数据类型是否为object，代指字符串或类别数据。
        X[column] = pd.to_numeric(X[column], errors='coerce')  # 尝试将该列转换为数值类型。

# 处理缺失值，填充为0或者其他策略
X.fillna(0, inplace=True)

# 数据标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# 使用缩放后的数据和选择的特征进行预测
predictions = model.predict(X)

# 打印预测结果
print(predictions)

name_list = []

for i in predictions:
    name_list.append(i)
data["label"] = name_list

# 定义文件路径
file_path = original_file_path.replace('.csv', '_result.csv')
data.to_csv(file_path, index=False, encoding = 'utf-8')

# 读取CSV文件
data = pd.read_csv(file_path, encoding='utf-8')

# 找到最后一列值为1的行
condition1 = data['label'] == 1
rows_with_one1 = data[condition1]

# 找到最后一列值为0的行
condition0 = data['label'] == 0
rows_with_one0 = data[condition0]

# 获取这些行的第一列的名称
column_names1 = rows_with_one1.iloc[:, 0]
column_names0 = rows_with_one0.iloc[:, 0]

# 获取这些行的第一列和第二列的值
# 注意：这里使用.values来获取numpy数组形式的数据，或者使用.tolist()转换为列表
values_column1_and_2_for_ones = rows_with_one1.iloc[:, [0, 1]].values  # 或者使用.tolist()
values_column1_and_2_for_zeros = rows_with_one0.iloc[:, [0, 1]].values  # 或者使用.tolist()

# 如果您想查看这些值
print("Values for rows with label 1 (first two columns):")
print(values_column1_and_2_for_ones)

print("Values for rows with label 0 (first two columns):")
print(values_column1_and_2_for_zeros)

# 打印结果
# print("column_names1:", column_names1)
# print("column_names0:", column_names0)
# 将列名转换为集合，去除重复项
unique_column_names1 = set(column_names1)
unique_column_names0 = set(column_names0)

# 打印去重后的列名
print("------------正负样本函数对应的程序-------------")
print("unique_column_names1:", unique_column_names1)
print("unique_column_names0:", unique_column_names0)

# 发现有一样的时候，删除便签为0的
# 找出unique_column_names1中包含在unique_column_names0中的元素
common_elements = set(unique_column_names0).intersection(set(unique_column_names1))

# 删除unique_column_names0中的共同元素
unique_column_names0_modified = [elem for elem in unique_column_names0 if elem not in common_elements]

# 输出结果
print("------------在正样本中出现的程序，在负样本中删除-------------")
print("Pos_samples_list:", unique_column_names1)
print("Pos_samples_len:", len(unique_column_names1))
print("Neg_sample_list (modified):", unique_column_names0_modified)
print("Neg_sample_list_len:", len(unique_column_names0_modified))

# 将输出结果写到文件中d
# 获取文件名和目录路径
directory, filename = original_file_path.rsplit('/', 1)
filename_without_extension, _ = filename.split('.')

# 定义正样本和负样本的文件路径
pos_sample_file_path = f"{directory}/{filename_without_extension}_Pos_samples.txt"
neg_sample_file_path = f"{directory}/{filename_without_extension}_Neg_sample.txt"
pos_sample_file_path_json = f"{directory}/{filename_without_extension}_Pos_samples.json"

# 将输出结果写到文件中
with open(pos_sample_file_path, 'w', encoding='utf-8') as file1:
    file1.write('\n'.join(unique_column_names1))

with open(neg_sample_file_path, 'w', encoding='utf-8') as file2:
    file2.write('\n'.join(unique_column_names0_modified))

with open(pos_sample_file_path_json, 'w', encoding='utf-8') as file2:
    file2.write('\n'.join([' '.join(row) for row in values_column1_and_2_for_ones]))

print("两个列表的内容已经写入到文件中。")

