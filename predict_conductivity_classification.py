import os
import torch
import numpy as np
import json
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from gnn_classification import ClassificationCEGNet

# 1. 配置路径 - 确保指向之前保存pt文件的目录
# 根据之前的代码，pt文件保存在transformed_data下的各个子目录中
base_graph_dir = "transformed_data/"  # 根目录
model_path = "models/best_classification_model.pt"
history_path = "results/classification_history.json"  # 加载训练时的阈值
output_csv = "classification_prediction.csv"  # 输出结果文件
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练时使用的分类阈值
try:
    with open(history_path, "r") as f:
        history = json.load(f)
    CLASS_THRESHOLD = history["threshold"]
    print(f"使用训练时的分类阈值: {CLASS_THRESHOLD:.4f}\n")
except Exception as e:
    print(f"警告：无法加载训练历史，使用默认概率阈值。错误：{e}")
    CLASS_THRESHOLD = 0.5
    print(f"使用默认分类阈值: {CLASS_THRESHOLD:.4f}\n")

# 2. 加载图数据集 - 搜索所有子目录中的pt文件
def find_all_pt_files(root_dir):
    """查找根目录下所有子目录中的.pt文件"""
    pt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pt"):
                pt_files.append(os.path.join(dirpath, filename))
    return pt_files

# 获取所有pt文件的完整路径
all_pt_files = find_all_pt_files(base_graph_dir)
if not all_pt_files:
    print(f"未在 {base_graph_dir} 及其子目录中找到.pt文件，请检查目录！")
    exit()

print(f"找到 {len(all_pt_files)} 个.pt文件，准备进行预测...")

# 检查是否有有效数据
valid_data = None
for file_path in all_pt_files:
    try:
        data = torch.load(file_path)
        if data.x is not None:
            valid_data = data
            break
    except Exception as e:
        print(f"检查文件 {file_path} 时出错: {e}")
        continue

if valid_data is None:
    print("未找到有效图数据（x为None）！")
    exit()

# 节点/边特征维度
node_in_dim = 5
edge_in_dim = 1

dataset = []
file_paths = []  # 存储文件完整路径
struct_names = []  # 存储文件名

for file_path in all_pt_files:
    try:
        data = torch.load(file_path)
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
            file_paths.append(file_path)  # 保存完整路径
            struct_names.append(os.path.basename(file_path).replace(".pt", ""))  # 保存文件名
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        continue

if not dataset:
    print("未找到有效图数据（x为None）！")
    exit()

print(f"成功加载 {len(dataset)} 个有效图数据")
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
        pred_classes = (pred_probs >= CLASS_THRESHOLD).astype(int)
        all_pred_probs.extend(pred_probs)
        all_pred_classes.extend(pred_classes)

# 5. 输出结果（包含文件路径）
results = {
    "file_path": file_paths,          # 新增：pt文件的完整路径
    "structure_name": struct_names,   # 结构文件名（不含路径）
    "pred_prob": all_pred_probs,      # 预测概率
    "pred_class": all_pred_classes    # 预测类别
}

# 打印带阈值对比的结果
print("\n分类结果（路径: 预测概率, 预测类别）：")
print("-" * 120)
for path, name, prob, cls in zip(file_paths, struct_names, all_pred_probs, all_pred_classes):
    comparison = "≥ 阈值" if prob >= CLASS_THRESHOLD else "< 阈值"
    print(f"{path}: 概率={prob:.4f}，{comparison}（阈值={CLASS_THRESHOLD:.4f}），预测类别={cls}")
print("-" * 120)

# 保存结果到CSV，包含文件路径信息
import pandas as pd
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n结果已保存到 {output_csv}，包含每个pt文件的完整路径")