import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from efficient_kan import KAN
from efficient_kan import pytrans
from MSERegLoss import MSERegLoss
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
input = pd.read_csv('./samplenumber/maize2002019-08-15.csv', header=None, skiprows=1)    #tezhengyouxuan
num_columns = input.shape[1]
print(num_columns)
target = pd.read_csv('./samplenumber/targemaize2002019-08-15.csv', header=None, skiprows=1)
input_arr = np.float32(np.array(input))
target_arr = np.float32(np.array(target))
# # 检查输入数据中的 NaN 和 Infinity
# if np.isnan(input_arr).any() or np.isinf(input_arr).any():
#     print("输入数据中包含 NaN 或 Infinity。")
# else:
#     print("输入数据中没有 NaN 或 Infinity。")
#
# # 检查目标数据中的 NaN 和 Infinity
# if np.isnan(target_arr).any() or np.isinf(target_arr).any():
#     print("目标数据中包含 NaN 或 Infinity。")
# else:
#     print("目标数据中没有 NaN 或 Infinity。")
# # 查找输入数据中 NaN 的位置
# nan_indices_input = np.where(np.isnan(input_arr))
# print("输入数据中 NaN 的位置:", nan_indices_input)
#
# # 查找输入数据中 Infinity 的位置
# inf_indices_input = np.where(np.isinf(input_arr))
# print("输入数据中 Infinity 的位置:", inf_indices_input)
#
# # 查找目标数据中 NaN 的位置
# nan_indices_target = np.where(np.isnan(target_arr))
# print("目标数据中 NaN 的位置:", nan_indices_target)
#
# # 查找目标数据中 Infinity 的位置
# inf_indices_target = np.where(np.isinf(target_arr))
# print("目标数据中 Infinity 的位置:", inf_indices_target)

# Split data into train and validation sets
train_input, val_input, train_target, val_target = train_test_split(input_arr, target_arr, test_size=0.2, random_state=42)
train_input_tensor = torch.tensor(train_input)
train_target_tensor = torch.tensor(train_target)
val_input_tensor = torch.tensor(val_input)
val_target_tensor = torch.tensor(val_target)

# Create datasets and dataloaders
train_dataset = TensorDataset(train_input_tensor, train_target_tensor)
val_dataset = TensorDataset(val_input_tensor, val_target_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define the model
model = KAN([num_columns, 64, 1])  # Ensure the model architecture is correct for regression
model.to(device)
print(model)
# Define optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define the custom loss function
# loss_func = MSERegLoss(alpha=0.01)  # This can be replaced with nn.MSELoss() if needed

# 将损失函数替换为 HuberLoss  HuberLoss中的delta参数控制了损失函数对离群值的敏感度。
loss_func = nn.HuberLoss(delta=1)  # delta 可以调整灵敏度



# Lists to store loss values for plotting
train_loss_all = []
val_loss_all = []
losses = []

# Early stopping parameters
patience = 50
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop
for epoch in range(50):
    # Training phase
    train_loss = 0
    train_num = 0
    model.train()
    with tqdm(train_loader) as pbar:
        for i, (input, target) in enumerate(pbar):
            input = input.view(-1, num_columns).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * input.size(0)
            train_num += input.size(0)
            pbar.set_postfix(loss=train_loss / train_num, lr=optimizer.param_groups[0]['lr'])
    train_loss_all.append(train_loss / train_num)

    # Validation phase
    model.eval()
    val_loss = 0
    val_num = 0
    with torch.no_grad():
        for input, target in val_loader:
            input = input.view(-1, num_columns).to(device)
            target = target.to(device)
            output = model(input)
            val_loss += loss_func(output, target).item() * input.size(0)
            val_num += input.size(0)
    val_loss /= val_num
    val_loss_all.append(val_loss)

    # Update learning rate scheduler
    scheduler.step()

    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}")

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./model1/best_kan.pth")  # Save best model
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# # # Plotting loss curves
# fig, ax = plt.subplots()
# ax.plot(range(1, epoch + 2), train_loss_all, label='Train Loss')
# ax.plot(range(1, epoch + 2), val_loss_all, label='Val Loss')
# ax.set_title('Loss Curves')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.legend()
# plt.show()


# # 2. 将训练损失和验证损失导出为 CSV 文件
# csv_file = 'FeatureKANloss_curves.csv'
#
# # 写入 CSV 文件
# with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     # 写入表头
#     writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
#     # 写入每一轮的损失数据
#     for i in range(epoch):
#         writer.writerow([i + 1, train_loss_all[i], val_loss_all[i]])
#
# print(f"训练和验证损失已导出到文件：{csv_file}")







# Final model evaluation on the validation set
model.eval()
val_predictions = []
val_targets = []

# Get predictions and calculate metrics
with torch.no_grad():
    for input, target in val_loader:
        input = input.view(-1, num_columns).to(device)
        target = target.to(device)
        output = model(input)
        val_predictions.extend(output.cpu().numpy())
        val_targets.extend(target.cpu().numpy())

# Convert to numpy arrays for evaluation
val_predictions = np.array(val_predictions).flatten()
val_targets = np.array(val_targets).flatten()

# Mean Squared Error (MSE) and R2 score
mse = mean_squared_error(val_targets, val_predictions)
r2 = r2_score(val_targets, val_predictions)

# Calculate accuracy: convert predictions to binary (0 or 1)
binary_predictions = (val_predictions >= 0.5).astype(int)
binary_targets = (val_targets >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(binary_predictions == binary_targets) * 100  # Convert to percentage

print(f"Validation MSE: {mse:.4f}")
print(f"Validation R2 Score: {r2:.4f}")
print(f"Validation Accuracy: {accuracy:.2f}%")


# Calculate accuracy
accuracy_LSTM = accuracy_score(binary_targets, binary_predictions)
mse_LSTM = mean_squared_error(binary_targets, binary_predictions)
f1_LSTM = f1_score(binary_targets, binary_predictions)
# PA生产者精度
PA_LSTM = recall_score(binary_targets, binary_predictions)
#UA用户精度
UA_LSTM =  precision_score(binary_targets, binary_predictions)


# Print accuracy
print(f"FKAN Accuracy: {accuracy_LSTM:.4f}")
print(f"mse_FKAN: {mse_LSTM:.4f}")
print(f"f1_FKAN: {f1_LSTM:.4f}")
print(f"PA_FKAN: {PA_LSTM:.4f}")
print(f"UA_FKAN: {UA_LSTM:.4f}")











# Optionally, you could also plot predicted vs true values
# plt.scatter(val_targets, val_predictions, alpha=0.5)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('True vs Predicted Values')
# plt.show()

#
# lib = ['x']
# symbolic_output = model.auto_symbolic(lib=lib)
# print(symbolic_output)









# import torch
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso          #减少参数量
# from sklearn.linear_model import ElasticNet     #减少参数量
# from sklearn.svm import SVR
# from sklearn.linear_model import OrthogonalMatchingPursuit
# from sklearn.linear_model import Lars
# from sklearn.linear_model import BayesianRidge
#
#
# model.eval()
# with torch.no_grad():
#     predictions = model(torch.tensor(input_arr).to(device)).cpu().numpy()
#     targets = target_arr  # 假设 target_arr 已经是 numpy 格式
#
# from sklearn.linear_model import Ridge
# from sympy import symbols, Add
#
# # 定义 74 个符号变量
#
# # 使用 Ridge 回归拟合模型
# ridge_model = Ridge(alpha=1.0)  # 调整 alpha 以控制正则化力度
# # ridge_model = Lasso(alpha=0.1)  # 调整 alpha 控制正则化强度
# # ridge_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
# # ridge_model = OrthogonalMatchingPursuit()
# # ridge_model = BayesianRidge()
# # ridge_model = LinearRegression()
# X_symbols = symbols(f'x0:{input_arr.shape[1]}')
#
# # ridge_model =ElasticNet()
#
# ridge_model.fit(input_arr, predictions)
#
# # 获取回归系数和截距
# coefficients = ridge_model.coef_.flatten()
# intercept = ridge_model.intercept_.item()
#
# # 构建符号表达式
# expression = Add(*[coeff * X_symbols[i] for i, coeff in enumerate(coefficients)]) + intercept
# print("拟合的符号表达式为：", expression)
# from sympy import simplify
#
# simplified_expression = simplify(expression)
# print("简化后的符号表达式为：", simplified_expression)

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import ElasticNet
# from sympy import symbols, Add, simplify
# from sympy import Float
# # 创建多项式特征
# degree = 2  # 假设需要二阶非线性
# poly = PolynomialFeatures(degree=degree, include_bias=False)
# input_poly = poly.fit_transform(input_arr)
#
# # 获取多项式特征名
# try:
#     # 新版本 Scikit-learn
#     poly_features = poly.get_feature_names_out([f'x{i}' for i in range(input_arr.shape[1])])
# except AttributeError:
#     # 旧版本 Scikit-learn
#     poly_features = poly.get_feature_names([f'x{i}' for i in range(input_arr.shape[1])])
#
# # 拟合模型
# ridge_model = ElasticNet()
# ridge_model.fit(input_poly, predictions)
#
# # 构建符号表达式
# coefficients = ridge_model.coef_.flatten()
# intercept = ridge_model.intercept_.item()
# X_symbols = symbols(f'x0:{input_arr.shape[1]}')
#
# from sympy import Float
#
# # 将系数和多项式符号表达式结合
# # 修正符号表达式生成逻辑
# expression = Add(*[Float(coeff) * symbols(poly_features[i].replace(" ", "_")) for i, coeff in enumerate(coefficients)]) + Float(intercept)
# simplified_expression = simplify(expression)
#
#
# print("拟合的符号表达式为：", expression)
# print("简化后的符号表达式为：", simplified_expression)
#
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import ElasticNet
# from sympy import symbols, Add, simplify, Float
# import numpy as np
#
# # 创建多项式特征
# degree = 2  # 假设需要二阶非线性
# poly = PolynomialFeatures(degree=degree, include_bias=False)
# input_poly = poly.fit_transform(input_arr)
#
# # 获取多项式特征名
# try:
#     # 新版本 Scikit-learn
#     poly_features = poly.get_feature_names_out([f'x{i}' for i in range(input_arr.shape[1])])
# except AttributeError:
#     # 旧版本 Scikit-learn
#     poly_features = poly.get_feature_names([f'x{i}' for i in range(input_arr.shape[1])])
#
# # 拟合模型
# ridge_model = ElasticNet(max_iter=10000)
# ridge_model.fit(input_poly, predictions)
#
# # 获取回归系数和截距
# coefficients = np.abs(ridge_model.coef_)  # 取绝对值表示特征重要性
# intercept = ridge_model.intercept_.item()
#
# # 筛选变量
# sorted_indices = np.argsort(coefficients)[::-1]  # 按重要性降序排序
# print(sorted_indices)
# cumulative_importance = np.cumsum(coefficients[sorted_indices]) / np.sum(coefficients)
# selected_indices = sorted_indices[cumulative_importance <= 1]  # 累积贡献达到95%的特征索引
#
# # 构建符号表达式
# selected_coefficients = ridge_model.coef_[selected_indices]
# selected_features = [poly_features[i] for i in selected_indices]
#
# expression = Add(
#     *[Float(coeff) * symbols(feature.replace(" ", "_")) for coeff, feature in zip(selected_coefficients, selected_features)]
# ) + Float(intercept)
#
# simplified_expression = simplify(expression)
#
# # 打印结果
# print("拟合的符号表达式为：", expression)
# print("简化后的符号表达式为：", simplified_expression)




# import shap  # 导入 SHAP 库
# # 计算 SHAP 特征重要性
# print("\nCalculating SHAP Values for Feature Importance...")
# train_input_tensor = torch.tensor(input_arr, dtype=torch.float32)
# model = KAN([num_columns, 64, 1]).to(device)
# model.load_state_dict(torch.load("./model1/best_kan.pth"))
# model.eval()
#
#
# def model_predict(input_data):
#     input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
#     with torch.no_grad():
#         return model(input_tensor).cpu().numpy()
#
#
# # 创建 SHAP explainer
# explainer = shap.Explainer(model_predict, input_arr)
# shap_values = explainer(input_arr)
#
#
# shap_values_array = shap_values.values  # SHAP values for all features
# feature_names = [f"Feature {i + 1}" for i in range(num_columns)]
# shap_df = pd.DataFrame(shap_values_array, columns=feature_names)
# output_csv_file = "./data/2023河南伪样本的均值/shap_values.csv"
# shap_df.to_csv(output_csv_file, index=False)
# # 可视化 SHAP 值
# shap.summary_plot(shap_values.values, input_arr, plot_type="bar",
#                   feature_names=[f"Feature {i + 1}" for i in range(num_columns)])
#
# shap.summary_plot(shap_values.values, input_arr, plot_type="bar",
#                   feature_names=feature_names, max_display=num_columns)
