import os
import torch
import numpy as np
import json
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from gnn_classification import ClassificationCEGNet

# 1. 配置路径
graph_dir = "transformed_data/mattergen_graphs/"
model_path = "models/best_classification_model.pt"
history_path = "results/classification_history.json"  # 加载训练时的阈值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练时使用的分类阈值
try:
    with open(history_path, "r") as f:
        history = json.load(f)
    CLASS_THRESHOLD = history["threshold"]
    print(f"使用训练时的分类阈值: {CLASS_THRESHOLD:.4f}\n")  # 单独打印阈值，方便查看
except Exception as e:
    print(f"警告：无法加载训练历史，使用默认概率阈值。错误：{e}")
    CLASS_THRESHOLD = 0.5
    print(f"使用默认分类阈值: {CLASS_THRESHOLD:.4f}\n")  # 单独打印阈值

# 2. 加载图数据集
pt_files = [f for f in os.listdir(graph_dir) if f.endswith(".pt")]
if not pt_files:
    print("未找到.pt文件，请检查目录！")
    exit()

valid_data = None
for f in pt_files:
    data = torch.load(os.path.join(graph_dir, f))
    if data.x is not None:
        valid_data = data
        break
if valid_data is None:
    print("未找到有效图数据（x为None）！")
    exit()

# 节点/边特征维度
node_in_dim = 5
edge_in_dim = 1

dataset = []
struct_names = []
for f in pt_files:
    data = torch.load(os.path.join(graph_dir, f))
    if data.x is not None:
        # 节点特征处理
        if data.x.size(1) > 5:
            data.x = data.x[:, :5]
        elif data.x.size(1) < 5:
            padding = torch.zeros(data.x.size(0), 5 - data.x.size(1))
            data.x = torch.cat([data.x, padding], dim=1)
        # 边特征处理
        if data.edge_attr is not None:
            if data.edge_attr.size(1) > 1:
                data.edge_attr = data.edge_attr[:, :1]
            elif data.edge_attr.size(1) < 1:
                padding = torch.zeros(data.edge_attr.size(0), 1 - data.edge_attr.size(1))
                data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)
        else:
            data.edge_attr = torch.zeros(data.edge_index.size(1), 1)
        dataset.append(data)
        struct_names.append(f.replace(".pt", ""))
if not dataset:
    print("未找到有效图数据（x为None）！")
    exit()

loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 3. 加载分类模型
model = ClassificationCEGNet(
    node_in_features=node_in_dim,
    edge_in_features=edge_in_dim,
    hidden_dim=64,
    dropout_rate=0.3
).to(device)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()

# 4. 批量预测
all_pred_probs = []
all_pred_classes = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        
        pred_logits = output.get('pred_cls', None)
        if pred_logits is None:
            raise KeyError(f"模型输出字典中未找到'pred_cls'键，可用键：{output.keys()}")
        
        pred_probs = torch.sigmoid(pred_logits).cpu().numpy().flatten()
        pred_classes = (pred_probs >= CLASS_THRESHOLD).astype(int)  # 这里也用统一的阈值，保持一致
        all_pred_probs.extend(pred_probs)
        all_pred_classes.extend(pred_classes)

# 5. 输出结果（重点修改部分：显示阈值对比）
results = {
    "structure": struct_names,
    "pred_prob": all_pred_probs,
    "pred_class": all_pred_classes
}

# 打印带阈值对比的结果
print("分类结果（结构名: 预测概率, 与阈值对比, 预测类别）：")
print("-" * 80)  # 分隔线，增强可读性
for name, prob, cls in zip(struct_names, all_pred_probs, all_pred_classes):
    # 增加对比说明：明确显示概率与阈值的大小关系
    comparison = "≥ 阈值" if prob >= CLASS_THRESHOLD else "< 阈值"
    print(f"{name}: 概率={prob:.4f}，{comparison}（阈值={CLASS_THRESHOLD:.4f}），预测类别={cls}")
print("-" * 80)

# 保存结果
import pandas as pd
pd.DataFrame(results).to_csv("mattergen_classifications.csv", index=False)