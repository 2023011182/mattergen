import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from gnn_regression import CEGNet

# 1. 配置路径（递归查找子文件夹）
graph_dir = "transformed_data/"  # 根目录
model_path = "models/best_regression_model.pt"  # 训练好的GNN模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 递归查找所有子文件夹中的.pt文件
pt_files = []
for root, dirs, files in os.walk(graph_dir):
    for file in files:
        if file.endswith(".pt"):
            pt_files.append(os.path.join(root, file))

if not pt_files:
    print(f"未在目录 {graph_dir} 及其子文件夹中找到.pt文件，请检查目录！")
    exit()

# 加载第一个有效的图来检查数据有效性
valid_data = None
for pt_path in pt_files:
    try:
        data = torch.load(pt_path)
        if data.x is not None:
            valid_data = data
            break
    except Exception as e:
        print(f"检查文件 {pt_path} 时出错：{e}，已跳过")
        continue

if valid_data is None:
    print(f"目录 {graph_dir} 及其子文件夹中未找到有效图数据（x为None）！")
    exit()

# 硬编码维度以匹配checkpoint
node_in_dim = 5
edge_in_dim = 1

# 加载所有有效数据并调整特征维度
dataset = []
struct_names = []
pt_file_paths = []
for pt_path in pt_files:
    try:
        data = torch.load(pt_path)
    except Exception as e:
        print(f"加载文件 {pt_path} 失败：{e}，已跳过")
        continue
        
    if data.x is not None:
        # 调整节点特征维度
        if data.x.size(1) > 5:
            data.x = data.x[:, :5]
        elif data.x.size(1) < 5:
            padding = torch.zeros(data.x.size(0), 5 - data.x.size(1))
            data.x = torch.cat([data.x, padding], dim=1)
        # 调整边特征维度
        if data.edge_attr is not None:
            if data.edge_attr.size(1) > 1:
                data.edge_attr = data.edge_attr[:, :1]
            elif data.edge_attr.size(1) < 1:
                padding = torch.zeros(data.edge_attr.size(0), 1 - data.edge_attr.size(1))
                data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)
        dataset.append(data)
        struct_names.append(os.path.splitext(os.path.basename(pt_path))[0])
        pt_file_paths.append(pt_path)

if not dataset:
    print(f"目录 {graph_dir} 及其子文件夹中未找到有效图数据（x为None）！")
    exit()

loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 3. 加载GNN模型
model = CEGNet(
    node_in_features=node_in_dim,
    edge_in_features=edge_in_dim,
    hidden_dim=128,
    output_dim=1,
    dropout_rate=0.3
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 4. 批量预测（修复：确保输出为标量）
all_preds = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        # 将预测结果从二维张量转为一维（去除多余维度）
        preds = preds.squeeze(dim=1)  # 关键修复：删除维度为1的维度
        all_preds.extend(preds.cpu().numpy())

# 5. 输出预测结果并保存到CSV
results = pd.DataFrame({
    "structure": struct_names,
    "pt_file_path": pt_file_paths,
    "predicted_property": all_preds
})

# 打印部分结果（修复：确保取到标量值）
print(f"从目录 {graph_dir} 及其子文件夹加载了 {len(dataset)} 个有效图数据，预测结果：")
for i in range(min(5, len(results))):
    # 确保获取标量值（处理可能的数组情况）
    pred_value = results['predicted_property'][i].item() if isinstance(results['predicted_property'][i], np.ndarray) else results['predicted_property'][i]
    print(f"{results['structure'][i]}: {pred_value:.4f}")
if len(results) > 5:
    print(f"... 共 {len(results)} 条结果")

# 保存结果到CSV
results.to_csv("regression_prediction.csv", index=False)
print(f"预测结果已保存至 regression_prediction.csv，包含{len(results)}条记录")