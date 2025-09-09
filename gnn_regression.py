import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, cast

from torch_geometric.data import Data, Dataset as GeoDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 全局配置 ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

GRAPHS_DIR    = "processed_data/graphs"  # 确保该目录存在.pt文件
MODELS_DIR    = "models"
RESULTS_DIR   = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 训练与模型规模
BATCH_SIZE    = 16
NUM_EPOCHS    = 240
LEARNING_RATE = 1e-3
HIDDEN_DIM    = 80
DROPOUT_RATE  = 0.2
LR_PATIENCE   = 10
WEIGHT_DECAY  = 1e-4

# 特征精简相关
NODE_BOTTLENECK_DIM = 8
L1_GATE_COEF        = 5e-4
USE_Y_STANDARDIZE   = True  # 是否标准化标签y

# 回归任务配置
OPTIMIZE_FOR          = "r2"  # 'r2' | 'neg_mse'
EARLY_STOP_PATIENCE   = 20    # 早停耐心（>0启用）
EARLY_STOP_MIN_DELTA  = 1e-4

# ------------- 工具函数 -------------
def _np_json_default(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _get_val_score(val_metrics: Dict[str, Dict[str, float]], val_loss: float) -> float:
    if OPTIMIZE_FOR == "r2":
        try:
            return float(val_metrics["regression"]["r2"])
        except Exception:
            return -float(val_loss)
    elif OPTIMIZE_FOR == "neg_mse":
        try:
            return -float(val_metrics["regression"]["mse"])
        except Exception:
            return -float(val_loss)
    return -float(val_loss)

# ------------- 1. 数据集（仅回归） -------------
class GraphDataset(GeoDataset):
    def __init__(self, root_dir: str, y_scaler: Optional[StandardScaler] = None, is_train: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.file_list: List[str] = []
        self.compositions: List[str] = []
        self.y_scaler = y_scaler
        self.is_train = is_train  # 是否为训练集（用于拟合scaler）
        
        if os.path.isdir(root_dir):
            all_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pt")])
            skipped = 0
            ys = []  # 收集所有y用于标准化
            for f in all_files:
                full = os.path.join(root_dir, f)
                try:
                    data_obj: Data = torch.load(full, map_location="cpu")
                except Exception as e:
                    print(f"[警告] 无法读取 {f}: {e}，已跳过。")
                    skipped += 1
                    continue

                has_x = (
                    hasattr(data_obj, "x")
                    and isinstance(data_obj.x, torch.Tensor)
                    and data_obj.x.dim() == 2
                    and data_obj.x.size(0) > 0
                    and data_obj.x.size(1) > 0
                )
                has_y = (
                    hasattr(data_obj, "y")
                    and isinstance(data_obj.y, torch.Tensor)
                    and data_obj.y.numel() > 0
                )

                if has_x and has_y:
                    self.file_list.append(f)
                    comp = os.path.splitext(f)[0].split('_')[0] if '_' in f else f[:-3]
                    self.compositions.append(comp)
                    ys.append(data_obj.y.item())  # 收集y值
                else:
                    skipped += 1
            if skipped:
                print(f"[数据集] 已过滤 {skipped} 个无效文件，保留 {len(self.file_list)} 个。")
            
            # 拟合标准化器（仅训练集）
            if self.is_train and USE_Y_STANDARDIZE and ys:
                self.y_scaler = StandardScaler()
                self.y_scaler.fit(np.array(ys).reshape(-1, 1))
        else:
            print(f"[警告] 目录不存在: {root_dir}")
    
    def len(self) -> int:
        return len(self.file_list)
    
    def get(self, idx: int) -> Data:
        pt_fname = self.file_list[idx]
        full_path = os.path.join(self.root_dir, pt_fname)
        data_obj: Data = torch.load(full_path)

        # 校验数据有效性
        if not hasattr(data_obj, 'y') or not isinstance(data_obj.y, torch.Tensor) or data_obj.y.numel() == 0:
            raise RuntimeError(f"[错误] 文件 {pt_fname} 缺少有效的 y 标签。")
        if not hasattr(data_obj, 'x') or not isinstance(data_obj.x, torch.Tensor) or data_obj.x.dim() != 2:
            raise RuntimeError(f"[错误] 文件 {pt_fname} 的节点特征无效。")
            
        # 标准化标签（如果需要）
        if USE_Y_STANDARDIZE and self.y_scaler is not None:
            y_np = data_obj.y.cpu().numpy().reshape(1, -1)
            y_scaled = self.y_scaler.transform(y_np).flatten()
            data_obj.y = torch.tensor(y_scaled, dtype=torch.float)
        
        data_obj.material_id = os.path.splitext(pt_fname)[0]
        return data_obj
    
    def get_y_scaler(self):
        return self.y_scaler

class IndexSubset(GeoDataset):
    def __init__(self, base: GraphDataset, indices: List[int]):
        super().__init__()
        self.base: GraphDataset = base
        self._sub_indices: List[int] = list(indices)

    def len(self) -> int:
        return len(self._sub_indices)

    def get(self, idx: int) -> Data:
        base_idx = self._sub_indices[idx]
        return self.base.get(base_idx)

# ------------- 2. 消息传递层 -------------
class EnhancedCEGMessagePassing(MessagePassing):
    def __init__(self, node_in_channels: int, edge_in_channels: int, out_channels: int, heads: int = 1):
        super().__init__(aggr='add')
        self.sender_node_lin = nn.Linear(node_in_channels, out_channels)
        self.edge_lin = nn.Linear(edge_in_channels, out_channels) if edge_in_channels > 0 else None
        
        msg_input_dim = out_channels + (out_channels if self.edge_lin else 0)
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, out_channels),
            nn.ReLU()
        )

        self.update_x_lin = nn.Linear(node_in_channels, out_channels)
        self.update_mlp = nn.Sequential(
            nn.Linear(out_channels + out_channels, out_channels),
            nn.ReLU()
        )
        self.gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        node_part = self.sender_node_lin(x_j)
        if self.edge_lin is not None and edge_attr is not None and edge_attr.size(1) > 0:
            edge_part = self.edge_lin(edge_attr)
            cat = torch.cat([node_part, edge_part], dim=-1)
        else:
            cat = node_part
        return self.message_mlp(cat)

    def update(self, aggr_msg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        old_x = self.update_x_lin(x)
        cat = torch.cat([old_x, aggr_msg], dim=-1)
        gate_value = self.gate(cat)
        updated = self.update_mlp(cat)
        return gate_value * updated + (1 - gate_value) * old_x

# ------------- 3. 回归模型 -------------
class RegressionCEGNet(nn.Module):
    def __init__(self, node_in_features: int, edge_in_features: int, hidden_dim: int,
                 output_dim: int = 1, dropout_rate: float = 0.2,
                 bottleneck_dim: int = NODE_BOTTLENECK_DIM, l1_gate_coef: float = L1_GATE_COEF):
        super().__init__()
        self.pool = global_mean_pool
        self.dropout = nn.Dropout(dropout_rate)
        self.l1_gate_coef = l1_gate_coef

        # 特征门控
        self.feature_gates = nn.Parameter(torch.zeros(node_in_features))

        # 瓶颈投影
        self.in_lin = nn.Linear(node_in_features, bottleneck_dim)

        # 两层消息传递
        self.conv1 = EnhancedCEGMessagePassing(bottleneck_dim, edge_in_features, hidden_dim, heads=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = EnhancedCEGMessagePassing(hidden_dim, edge_in_features, hidden_dim, heads=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 回归预测头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.regressor = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 门控 + 瓶颈
        gates = torch.sigmoid(self.feature_gates)
        x = x * gates
        x = F.relu(self.in_lin(x))

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        # 图池化
        graph_emb = self.pool(x, batch)
        graph_emb = self.fc(graph_emb)

        # 回归输出
        pred_reg = self.regressor(graph_emb).squeeze(-1)

        return {"pred_reg": pred_reg, "gates": gates}

    def get_loss(self, pred: Dict[str, torch.Tensor], data: Data) -> torch.Tensor:
        # 回归损失
        reg_loss = F.mse_loss(pred["pred_reg"], data.y.view(-1))
        # L1正则化门控
        gate_loss = self.l1_gate_coef * torch.norm(self.feature_gates, p=1)
        return reg_loss + gate_loss

# ------------- 4. 训练与评估函数 -------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = model.get_loss(pred, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs  # 按样本数加权
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, y_scaler=None):
    model.eval()
    all_preds = []
    all_ys = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = model.get_loss(pred, batch)
            total_loss += loss.item() * batch.num_graphs
            
            # 保存预测和真实值（如需反标准化）
            pred_np = pred["pred_reg"].cpu().numpy()
            y_np = batch.y.cpu().numpy()
            all_preds.extend(pred_np)
            all_ys.extend(y_np)
    
    # 反标准化（如果训练时标准化了标签）
    if USE_Y_STANDARDIZE and y_scaler is not None:
        all_preds = y_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
        all_ys = y_scaler.inverse_transform(np.array(all_ys).reshape(-1, 1)).flatten()
    
    # 计算指标
    mse = np.mean((np.array(all_preds) - np.array(all_ys)) **2)
    mae = mean_absolute_error(all_ys, all_preds)
    r2 = r2_score(all_ys, all_preds)
    
    return {
        "loss": total_loss / len(loader.dataset),
        "regression": {"mse": mse, "mae": mae, "r2": r2}
    }

# ------------- 5. 主函数入口 -------------
def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载数据集并划分
    full_dataset = GraphDataset(GRAPHS_DIR, is_train=True)
    if len(full_dataset) == 0:
        print(f"错误：在 {GRAPHS_DIR} 中未找到有效数据，请检查路径。")
        return
    y_scaler = full_dataset.get_y_scaler()

    # 划分训练集和验证集（8:2）
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
    train_dataset = IndexSubset(full_dataset, train_idx)
    val_dataset = IndexSubset(full_dataset, val_idx)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型（需根据数据特征维度设置）
    # 从第一个样本获取特征维度
    sample_data = full_dataset.get(0)
    node_in_dim = sample_data.x.size(1)
    edge_in_dim = sample_data.edge_attr.size(1) if sample_data.edge_attr is not None else 0
    print(f"节点特征维度: {node_in_dim}, 边特征维度: {edge_in_dim}")

    model = RegressionCEGNet(
        node_in_features=node_in_dim,
        edge_in_features=edge_in_dim,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    # 3. 优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max' if OPTIMIZE_FOR == 'r2' else 'min', 
                                 patience=LR_PATIENCE, factor=0.5)

    # 4. 训练循环
    best_val_score = -float('inf')
    early_stop_counter = 0
    history = defaultdict(list)

    for epoch in range(NUM_EPOCHS):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        # 评估
        train_metrics = evaluate(model, train_loader, device, y_scaler)
        val_metrics = evaluate(model, val_loader, device, y_scaler)
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_r2"].append(val_metrics["regression"]["r2"])
        
        # 打印日志
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  训练损失: {train_loss:.4f} | 训练R²: {train_metrics['regression']['r2']:.4f}")
        print(f"  验证损失: {val_metrics['loss']:.4f} | 验证R²: {val_metrics['regression']['r2']:.4f}")
        
        # 学习率调度
        val_score = _get_val_score(val_metrics, val_metrics["loss"])
        scheduler.step(val_score)
        
        # 早停逻辑
        if val_score > best_val_score + EARLY_STOP_MIN_DELTA:
            best_val_score = val_score
            early_stop_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_regression_model.pt"))
            print(f"  保存最佳模型（验证分数: {best_val_score:.4f}）")
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE and EARLY_STOP_PATIENCE > 0:
                print(f"  早停触发（{early_stop_counter}轮无提升）")
                break

    # 5. 保存训练历史
    with open(os.path.join(RESULTS_DIR, "regression_history.json"), "w") as f:
        json.dump(history, f, default=_np_json_default, indent=2)
    print(f"训练完成，历史记录保存至 {RESULTS_DIR}/regression_history.json")

if __name__ == "__main__":
    main()