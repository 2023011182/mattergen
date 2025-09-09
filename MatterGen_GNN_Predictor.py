import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from GNN import LiteCEGNet

# 1. 配置路径
graph_dir = "transformed_data/mattergen_graphs/"  # 转换后的图数据目录
model_path = "models/best_regression_model.pt"  # 训练好的GNN模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载图数据集（直接加载.pt文件，忽略y标签）
pt_files = [f for f in os.listdir(graph_dir) if f.endswith(".pt")]
if not pt_files:
    print("未找到.pt文件，请检查目录！")
    exit()

# 加载第一个有效的图来检查数据有效性（不用于维度）
valid_data = None
for f in pt_files:
    data = torch.load(os.path.join(graph_dir, f))
    if data.x is not None:
        valid_data = data
        break
if valid_data is None:
    print("未找到有效图数据（x为None）！")
    exit()

# 硬编码维度以匹配checkpoint（训练时使用node_in_features=5, edge_in_features=1）
node_in_dim = 5  # 节点特征维度，匹配checkpoint
edge_in_dim = 1  # 边特征维度，假设为1

# 加载所有有效数据（忽略y，只检查x，并调整x维度到5）
dataset = []
struct_names = []
for f in pt_files:
    data = torch.load(os.path.join(graph_dir, f))
    if data.x is not None:
        # 调整节点特征维度到5
        if data.x.size(1) > 5:
            data.x = data.x[:, :5]  # 截断
        elif data.x.size(1) < 5:
            padding = torch.zeros(data.x.size(0), 5 - data.x.size(1))
            data.x = torch.cat([data.x, padding], dim=1)  # 填充零
        # 调整边特征维度到1（如果需要）
        if data.edge_attr is not None:
            if data.edge_attr.size(1) > 1:
                data.edge_attr = data.edge_attr[:, :1]  # 截断
            elif data.edge_attr.size(1) < 1:
                padding = torch.zeros(data.edge_attr.size(0), 1 - data.edge_attr.size(1))
                data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)  # 填充零
        dataset.append(data)
        struct_names.append(f.replace(".pt", ""))
if not dataset:
    print("未找到有效图数据（x为None）！")
    exit()

loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 3. 加载GNN模型（需与训练时的结构一致）
model = LiteCEGNet(
    node_in_features=node_in_dim,
    edge_in_features=edge_in_dim,
    hidden_dim=128,  # 需与训练时一致
    output_dim=1,    # 回归任务（如预测体弹模量）
    dropout_rate=0.3
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # 切换到评估模式

# 4. 批量预测
all_preds = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)  # 模型输出：[batch_size]
        all_preds.extend(preds.cpu().numpy())

# 5. 输出预测结果
results = dict(zip(struct_names, all_preds))
print("预测结果（结构名: 性能值）：")
for name, pred in results.items():
    print(f"{name}: {pred:.4f}")

# 保存结果到CSV
import pandas as pd
pd.DataFrame({"structure": struct_names, "predicted_property": all_preds}).to_csv(
    "mattergen_predictions.csv", index=False
)