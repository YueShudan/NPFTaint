import pandas as pd
import joblib
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

model = load('res/RF3.joblib')

original_file_path = 'pre_sample/DIR-880_ARM_binary_CSV_Pre.csv'

data = pd.read_csv(original_file_path)

features = ['src', 'fname', 'funcaddr', 'keyimportapiinfo', 'netkeywords']
X = data.drop(columns=features) 

for column in X.columns:   
    if X[column].dtype == 'object':  
        X[column] = pd.to_numeric(X[column], errors='coerce') 

X.fillna(0, inplace=True)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

predictions = model.predict(X)

print(predictions)

name_list = []

for i in predictions:
    name_list.append(i)
data["label"] = name_list

file_path = original_file_path.replace('.csv', '_result.csv')
data.to_csv(file_path, index=False, encoding = 'utf-8')

data = pd.read_csv(file_path, encoding='utf-8')

condition1 = data['label'] == 1
rows_with_one1 = data[condition1]

condition0 = data['label'] == 0
rows_with_one0 = data[condition0]

column_names1 = rows_with_one1.iloc[:, 0]
column_names0 = rows_with_one0.iloc[:, 0]

values_column1_and_2_for_ones = rows_with_one1.iloc[:, [0, 1]].values 
values_column1_and_2_for_zeros = rows_with_one0.iloc[:, [0, 1]].values  


print("Values for rows with label 1 (first two columns):")
print(values_column1_and_2_for_ones)

print("Values for rows with label 0 (first two columns):")
print(values_column1_and_2_for_zeros)


# print("column_names1:", column_names1)
# print("column_names0:", column_names0)

unique_column_names1 = set(column_names1)
unique_column_names0 = set(column_names0)

print("unique_column_names1:", unique_column_names1)
print("unique_column_names0:", unique_column_names0)

common_elements = set(unique_column_names0).intersection(set(unique_column_names1))

unique_column_names0_modified = [elem for elem in unique_column_names0 if elem not in common_elements]

print("Pos_samples_list:", unique_column_names1)
print("Pos_samples_len:", len(unique_column_names1))
print("Neg_sample_list (modified):", unique_column_names0_modified)
print("Neg_sample_list_len:", len(unique_column_names0_modified))

directory, filename = original_file_path.rsplit('/', 1)
filename_without_extension, _ = filename.split('.')

pos_sample_file_path = f"{directory}/{filename_without_extension}_Pos_samples.txt"
neg_sample_file_path = f"{directory}/{filename_without_extension}_Neg_sample.txt"
pos_sample_file_path_json = f"{directory}/{filename_without_extension}_Pos_samples.json"

with open(pos_sample_file_path, 'w', encoding='utf-8') as file1:
    file1.write('\n'.join(unique_column_names1))

with open(neg_sample_file_path, 'w', encoding='utf-8') as file2:
    file2.write('\n'.join(unique_column_names0_modified))

with open(pos_sample_file_path_json, 'w', encoding='utf-8') as file2:
    file2.write('\n'.join([' '.join(row) for row in values_column1_and_2_for_ones]))

print("The contents of the two lists have been written into the file.")

