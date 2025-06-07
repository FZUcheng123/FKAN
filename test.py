import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efficient_kan import KAN
from efficient_kan.mlp import MLP
from sklearn.metrics import accuracy_score, classification_report
import sympy
from sklearn.metrics import mean_squared_error, r2_score
# Check device availability
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
input_data = pd.read_csv('./sampleture/MAIZEZHENSHI2024-08-15.csv', header=None, skiprows=1)    #tezhengyouxuan
num_columns = input_data.shape[1]  # Get number of columns
print(num_columns)
target_data = pd.read_csv('./sampleture/trageMAIZEZHENSHI2024-08-15.csv', header=None, skiprows=1)
# x = pd.read_csv('./data/x.csv', header=None)



# Convert data to tensors
input_tensor = torch.Tensor(np.float32(np.array(input_data))).to(device)
target_tensor = torch.Tensor(np.float32(np.array(target_data))).to(device)
# x = np.float32(np.array(x))







# Load KAN model
model_kan = KAN([num_columns, 64, 1])
model_kan.to(device)
model_kan.load_state_dict(torch.load('./model1/best_kan.pth'))
# print(model_kan)
output_kan = model_kan(input_tensor)
output_kan_np = output_kan.detach().cpu().numpy()
# # 打印模型结构
# print("KAN 模型结构:")
# print(model_kan)

# Prepare for plotting
y_target = target_tensor.cpu().numpy().flatten()  # Ground truth values matching x
y_kan = output_kan_np.flatten()  # KAN model output matching x

# Convert outputs to binary values based on a threshold of 0.5
y_kan_binary = (y_kan >= 0.5).astype(int)

# Ensure the 15th column is extracted and converted to numpy
column_15 = input_data.iloc[:, 14].to_numpy()  # Note: Python is zero-indexed, so column 15 is index 14.
column_42 = input_data.iloc[:, 41].to_numpy()  # Note: Python is zero-indexed, so column 15 is index 14.
# Convert binary outputs to pandas DataFrame
y_kan_binary_df = pd.DataFrame(y_kan_binary, columns=["KAN_Binary_Output"])

# y_mlp_binary_df.loc[column_15 < 2, "KAN_Binary_Output"] = 0
# y_kan_binary_df.loc[column_15 < 2, "KAN_Binary_Output"] = 0
# y_mlp_binary_df.loc[column_42 > 12.5, "KAN_Binary_Output"] = 0
# y_kan_binary_df.loc[column_42 > 12.5, "KAN_Binary_Output"] = 0




# Save DataFrames to CSV files in the same folder
# y_mlp_binary_df.to_csv('./data/结果/mlp_202405.csv', index=False)
# y_kan_binary_df.to_csv('./sampleture/kan_202405.csv', index=False)

# print("MLP and KAN binary outputs have been saved to './data/' folder as CSV files.")

# Print binary outputs along with true labels
print("True Labels:", y_target)

print("KAN Binary Output:", y_kan_binary)

# Calculate accuracy
accuracy_kan = accuracy_score(y_target, y_kan_binary_df)
mse_kan = mean_squared_error(y_target, y_kan_binary_df)
f1_kan = f1_score(y_target, y_kan_binary_df)
# PA生产者精度
PA_kan = recall_score(y_target, y_kan_binary_df)
#UA用户精度
UA_kan =  precision_score(y_target, y_kan_binary_df)

print(f"KAN Accuracy: {accuracy_kan:.4f}")
print(f"mse_kan: {mse_kan:.4f}")
print(f"f1_kan: {f1_kan:.4f}")
print(f"UA_kan: {UA_kan:.4f}")
print(f"PA_kan: {PA_kan:.4f}")




