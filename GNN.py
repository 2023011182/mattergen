import os
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.optimizer import Optimizer 
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, cast

from torch_geometric.data import Data, Dataset as GeoDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
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

GRAPHS_DIR    = "processed_data/graphs"  # 直接读取这里的 .pt
MODELS_DIR    = "models"                 # 保存/加载模型文件
RESULTS_DIR   = "results"                # 结果保存目录
BATCH_SIZE    = 8                        # 建议减小，避免CUDA OOM
NUM_EPOCHS    = 100                      # 增加训练轮次
LEARNING_RATE = 5e-4                     # 降低学习率提高稳定性
HIDDEN_DIM    = 128
DROPOUT_RATE  = 0.3
LR_PATIENCE   = 10                       # 学习率调整耐心值
# 分类阈值，log10(σ) >= CLASSIFICATION_THRESHOLD 判为高导电率
CLASSIFICATION_THRESHOLD = -4.0          # 根据论文设定的阈值
# 任务类型: "regression", "classification", "both"
TASK_TYPE     = "both"

# ------------- 工具：numpy -> JSON 序列化 -------------
def _np_json_default(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

# ------------- 1. 增强的数据集 -------------
class GraphDataset(GeoDataset):
    """
    增强的GraphDataset: 
    1. 初始化时过滤掉缺少标签或空特征的图
    2. 支持分类任务的标签转换
    3. 记录材料化学组成用于后续聚类
    """
    def __init__(self, root_dir: str, task: str = "both", threshold: float = CLASSIFICATION_THRESHOLD):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.threshold = threshold
        self.file_list: List[str] = []
        self.compositions: List[str] = []  # 用于存储化学组成
        
        if os.path.isdir(root_dir):
            all_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pt")])
            skipped = 0
            for f in all_files:
                full = os.path.join(root_dir, f)
                try:
                    data_obj: Data = torch.load(full, map_location="cpu")
                except Exception as e:
                    print(f"[警告] 无法读取 {f}: {e}，已跳过。")
                    skipped += 1
                    continue

                # 节点特征检查
                has_x = (
                    hasattr(data_obj, "x")
                    and isinstance(data_obj.x, torch.Tensor)
                    and data_obj.x.dim() == 2
                    and data_obj.x.size(0) > 0
                    and data_obj.x.size(1) > 0
                )
                # 标签检查
                has_y = (
                    hasattr(data_obj, "y")
                    and isinstance(data_obj.y, torch.Tensor)
                    and data_obj.y.numel() > 0
                )

                if has_x and has_y:
                    self.file_list.append(f)
                    comp = os.path.splitext(f)[0].split('_')[0] if '_' in f else f[:-3]
                    self.compositions.append(comp)
                else:
                    skipped += 1
            if skipped:
                print(f"[数据集] 已过滤 {skipped} 个缺少标签或为空特征的图文件。保留 {len(self.file_list)} 个。")
        else:
            print(f"[警告] 目录不存在: {root_dir}")
    
    def len(self) -> int:
        return len(self.file_list)
    
    def get(self, idx: int) -> Data:
        pt_fname = self.file_list[idx]
        full_path = os.path.join(self.root_dir, pt_fname)
        data_obj: Data = torch.load(full_path)  # 直接加载 Data 对象
        
        # 防御性检查（不返回空图）
        if (not hasattr(data_obj, 'y') 
            or not isinstance(getattr(data_obj, 'y'), torch.Tensor) 
            or cast(torch.Tensor, data_obj.y).numel() == 0):
            raise RuntimeError(f"[错误] 文件 {pt_fname} 缺少有效的 y 标签，请从数据集中移除。")
        if (not hasattr(data_obj, 'x')
            or not isinstance(data_obj.x, torch.Tensor)
            or data_obj.x.dim() != 2
            or data_obj.x.size(1) == 0
            or data_obj.x.size(0) == 0):
            raise RuntimeError(f"[错误] 文件 {pt_fname} 的节点特征为空，请从数据集中移除。")
            
        # 添加分类标签
        if self.task in ["classification", "both"]:
            y_t = cast(torch.Tensor, data_obj.y).view(-1)
            y_val = float(y_t[0].item())
            data_obj.y_cls = torch.tensor(1.0 if y_val >= self.threshold else 0.0, dtype=torch.float)
            
        # 添加材料ID (来自文件名)
        data_obj.material_id = os.path.splitext(pt_fname)[0]
        
        return data_obj
    
    def get_composition(self, idx: int) -> str:
        """获取指定索引的材料化学组成"""
        return self.compositions[idx]
    
    def get_all_compositions(self) -> List[str]:
        """获取所有材料的化学组成"""
        return self.compositions

# ------------- 1.1 自定义索引子集（避免 Subset 类型告警） -------------
class IndexSubset(GeoDataset):
    def __init__(self, base: GraphDataset, indices: List[int]):
        super().__init__()
        self.base: GraphDataset = base
        self._sub_indices: List[int] = list(indices)

    def len(self) -> int:
        return len(self._sub_indices)

    def get(self, idx: int) -> Data:
        # 这里的 idx 是子集中的局部索引，需要映射到 base 的全局索引
        base_idx = self._sub_indices[idx]
        return self.base.get(base_idx)

# ------------- 2. 注意力机制 -------------
class AttentionLayer(nn.Module):
    """
    自注意力层: 用于聚焦重要特征
    """
    def __init__(self, in_features: int, dropout_rate: float = 0.1):
        super().__init__()
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(in_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = self.dropout(F.softmax(scores, dim=-1))
        
        # 应用注意力权重
        return torch.matmul(attention, v)

# ------------- 3. 增强的消息传递 -------------
class EnhancedCEGMessagePassing(MessagePassing):
    """
    增强的消息传递模块:
    1. 添加门控更新机制
    2. 多头注意力
    """
    def __init__(self, node_in_channels: int, edge_in_channels: int, out_channels: int, heads: int = 1):
        super().__init__(aggr='add')  # "add"聚合所有邻居消息
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels
        self.heads = heads

        # 线性层: 发送方节点特征
        self.sender_node_lin = nn.Linear(node_in_channels, out_channels)

        # 若 edge_in_channels>0，则线性处理边特征
        if edge_in_channels > 0:
            self.edge_lin = nn.Linear(edge_in_channels, out_channels)
        else:
            self.edge_lin = None
        
        # 消息组装 MLP：sender_node + (edge)
        msg_input_dim = out_channels + (out_channels if self.edge_lin else 0)
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, out_channels),
            nn.ReLU()
        )

        # 注意力层
        if heads > 1:
            self.attention = nn.ModuleList([
                AttentionLayer(out_channels) for _ in range(heads)
            ])
            self.attention_combine = nn.Linear(out_channels * heads, out_channels)
        else:
            self.attention = None

        # update MLP：结合旧节点特征(投影后)与聚合消息
        self.update_x_lin = nn.Linear(node_in_channels, out_channels)
        self.update_mlp = nn.Sequential(
            nn.Linear(out_channels + out_channels, out_channels),
            nn.ReLU()
        )

        # 门控更新机制
        self.gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        # 发送方节点特征
        node_part = self.sender_node_lin(x_j)
        if self.edge_lin is not None and edge_attr is not None and edge_attr.size(1)>0:
            edge_part = self.edge_lin(edge_attr)
            cat = torch.cat([node_part, edge_part], dim=-1)
        else:
            cat = node_part
        return self.message_mlp(cat)

    def update(self, aggr_msg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        old_x = self.update_x_lin(x)
        
        # 多头注意力处理
        if self.attention is not None:
            attention_outputs = []
            for attn_layer in self.attention:
                attention_outputs.append(attn_layer(aggr_msg))
            if len(attention_outputs) > 1:
                aggr_msg = self.attention_combine(torch.cat(attention_outputs, dim=-1))
            else:
                aggr_msg = attention_outputs[0]
        
        cat = torch.cat([old_x, aggr_msg], dim=-1)
        
        # 使用门控机制决定更新比例
        gate_value = self.gate(cat)
        updated = self.update_mlp(cat)
        
        return gate_value * updated + (1 - gate_value) * old_x

# ------------- 4. 增强的GNN模型 -------------
class EnhancedCEGNet(nn.Module):
    """
    增强版CEGNet:
    1. 多种池化策略
    2. 残差连接
    3. 批归一化
    4. 分类与回归双任务
    5. MC Dropout不确定性量化
    """
    def __init__(self, node_in_features: int, edge_in_features: int, hidden_dim: int, 
                 output_dim: int = 1, dropout_rate: float = 0.3, task_type: str = "both", heads: int = 2):
        super().__init__()
        self.node_in_features = node_in_features
        self.edge_in_features = edge_in_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task_type = task_type
        
        # 图卷积层
        self.conv1 = EnhancedCEGMessagePassing(node_in_features, edge_in_features, hidden_dim, heads=heads)
        self.conv2 = EnhancedCEGMessagePassing(hidden_dim, edge_in_features, hidden_dim, heads=heads)
        self.conv3 = EnhancedCEGMessagePassing(hidden_dim, edge_in_features, hidden_dim, heads=heads)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 多种池化策略
        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool
        self.pool_sum = global_add_pool
        
        # 全局注意力层
        self.global_attention = AttentionLayer(hidden_dim)
        
        # 最终预测层
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 回归输出
        self.regressor = nn.Linear(hidden_dim // 2, output_dim)
        
        # 分类输出
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # MC Dropout配置
        self.do_mc_dropout = False
        self.training_dropout_rate = dropout_rate

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 第一层
        x1 = self.conv1(x, edge_index, edge_attr)
        if batch is not None and batch.numel() > 0:  # 确保有批次数据
            x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1) if self.training or self.do_mc_dropout else x1
        
        # 第二层 (带残差连接)
        x2 = self.conv2(x1, edge_index, edge_attr)
        if batch is not None and batch.numel() > 0:  # 确保有批次数据
            x2 = self.bn2(x2)
        x2 = F.relu(x2) + x1  # 残差连接
        x2 = self.dropout(x2) if self.training or self.do_mc_dropout else x2
        
        # 第三层
        x3 = self.conv3(x2, edge_index, edge_attr)
        if batch is not None and batch.numel() > 0:  # 确保有批次数据
            x3 = self.bn3(x3)
        x3 = F.relu(x3) + x2  # 残差连接
        
        # 多池化策略
        x_mean = self.pool_mean(x3, batch)
        x_max = self.pool_max(x3, batch)
        x_sum = self.pool_sum(x3, batch)
        
        # 组合多种池化结果
        x_combined = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(x_combined))
        x = self.dropout(x) if self.training or self.do_mc_dropout else x
        x = F.relu(self.fc2(x))
        x = self.dropout(x) if self.training or self.do_mc_dropout else x
        
        # 任务分支
        outputs: Dict[str, torch.Tensor] = {}
        if self.task_type in ["regression", "both"]:
            outputs["regression"] = self.regressor(x).squeeze(-1)
        
        if self.task_type in ["classification", "both"] and hasattr(data, 'y_cls'):
            outputs["classification"] = torch.sigmoid(self.classifier(x)).squeeze(-1)
            
        return outputs

    # MC Dropout不确定性估计相关方法
    def enable_mc_dropout(self) -> None:
        """启用MC Dropout模式"""
        self.do_mc_dropout = True
        
    def disable_mc_dropout(self) -> None:
        """禁用MC Dropout模式"""
        self.do_mc_dropout = False

# ------------- 5. 训练与评估增强函数 -------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Optimizer, 
                   criterion_dict: Dict[str, nn.Module], device: torch.device, task_type: str = "both") -> Tuple[float, Dict[str, float]]:
    """
    增强的训练函数: 支持多任务学习
    """
    model.train()
    total_loss = 0.0
    task_losses: Dict[str, float] = {"regression": 0.0, "classification": 0.0}
    total_graphs = 0
    
    for batch_data in loader:
        # 检查y是否有效
        if (not hasattr(batch_data, 'x')) or batch_data.x is None or batch_data.x.size(0) == 0:
            continue
        if (not hasattr(batch_data, 'y')) or (not isinstance(batch_data.y, torch.Tensor)) or batch_data.y.numel() == 0:
            continue

        # 为满足 Pylance 的签名，将 device 转为 str
        batch_data = batch_data.to(str(device))

        optimizer.zero_grad()
        outputs = model(batch_data)
        
        # 用 Tensor 初始化，避免 float.backward 告警
        loss: torch.Tensor = torch.tensor(0.0, device=device)
        # 回归损失
        if task_type in ["regression", "both"] and "regression" in outputs:
            reg_loss = criterion_dict["regression"](outputs["regression"], cast(torch.Tensor, batch_data.y))
            loss = loss + reg_loss
            task_losses["regression"] += float(reg_loss.detach().item()) * batch_data.num_graphs
            
        # 分类损失
        if task_type in ["classification", "both"] and "classification" in outputs and hasattr(batch_data, 'y_cls'):
            cls_loss = criterion_dict["classification"](outputs["classification"], cast(torch.Tensor, batch_data.y_cls))
            loss = loss + cls_loss
            task_losses["classification"] += float(cls_loss.detach().item()) * batch_data.num_graphs
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.detach().item()) * batch_data.num_graphs
        total_graphs += batch_data.num_graphs

    # 计算平均损失
    if total_graphs > 0:
        return (
            total_loss / total_graphs,
            {k: (v / total_graphs) for k, v in task_losses.items() if v > 0.0}
        )
    return 0.0, {k: 0.0 for k in task_losses.keys()}

def evaluate_model(model: nn.Module, loader: DataLoader, criterion_dict: Dict[str, nn.Module], 
                  device: torch.device, task_type: str = "both") -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    """
    增强的评估函数: 支持多任务评估
    """
    model.eval()
    total_loss = 0.0
    task_losses: Dict[str, float] = {"regression": 0.0, "classification": 0.0}
    total_graphs = 0
    
    # 用于各项指标计算
    reg_preds, reg_targets = [], []
    cls_preds, cls_targets = [], []
    
    with torch.no_grad():
        for batch_data in loader:
            # 检查y是否有效
            if (not hasattr(batch_data, 'x')) or batch_data.x is None or batch_data.x.size(0) == 0:
                continue
            if (not hasattr(batch_data, 'y')) or (not isinstance(batch_data.y, torch.Tensor)) or batch_data.y.numel() == 0:
                continue

            batch_data = batch_data.to(str(device))
            outputs = model(batch_data)
            
            loss: torch.Tensor = torch.tensor(0.0, device=device)
            # 回归损失与预测
            if task_type in ["regression", "both"] and "regression" in outputs:
                reg_loss = criterion_dict["regression"](outputs["regression"], cast(torch.Tensor, batch_data.y))
                loss = loss + reg_loss
                task_losses["regression"] += float(reg_loss.detach().item()) * batch_data.num_graphs
                
                reg_preds.append(outputs["regression"].cpu().numpy())
                reg_targets.append(cast(torch.Tensor, batch_data.y).cpu().numpy())
                
            # 分类损失与预测
            if task_type in ["classification", "both"] and "classification" in outputs and hasattr(batch_data, 'y_cls'):
                cls_loss = criterion_dict["classification"](outputs["classification"], cast(torch.Tensor, batch_data.y_cls))
                loss = loss + cls_loss
                task_losses["classification"] += float(cls_loss.detach().item()) * batch_data.num_graphs
                
                cls_preds.append(outputs["classification"].cpu().numpy())
                cls_targets.append(cast(torch.Tensor, batch_data.y_cls).cpu().numpy())

            total_loss += float(loss.detach().item()) * batch_data.num_graphs
            total_graphs += batch_data.num_graphs
    
    if total_graphs == 0:
        return float('inf'), {}, {}, {}
        
    # 计算平均损失
    avg_loss = total_loss / total_graphs
    avg_task_losses = {k: float(v / total_graphs) for k, v in task_losses.items() if v > 0.0}
    
    # 合并所有批次的预测结果
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    
    # 回归指标
    if task_type in ["regression", "both"] and reg_preds:
        reg_preds_np = np.concatenate(reg_preds).flatten()
        reg_targets_np = np.concatenate(reg_targets).flatten()
        results["regression"] = (reg_preds_np, reg_targets_np)
        
        metrics["regression"] = {
            "mae": float(mean_absolute_error(reg_targets_np, reg_preds_np)),
            "r2": float(r2_score(reg_targets_np, reg_preds_np)),
            "mse": float(np.mean((reg_preds_np - reg_targets_np) ** 2))
        }
    
    # 分类指标
    if task_type in ["classification", "both"] and cls_preds:
        cls_preds_np = np.concatenate(cls_preds).flatten()
        cls_targets_np = np.concatenate(cls_targets).flatten()
        # 二值化预测概率
        cls_pred_binary = (cls_preds_np >= 0.5).astype(float)
        results["classification"] = (cls_preds_np, cls_targets_np)
        
        metrics["classification"] = {
            "accuracy": float(accuracy_score(cls_targets_np, cls_pred_binary)),
            "precision": float(precision_score(cls_targets_np, cls_pred_binary, zero_division=0)),
            "recall": float(recall_score(cls_targets_np, cls_pred_binary, zero_division=0)),
            "f1": float(f1_score(cls_targets_np, cls_pred_binary, zero_division=0)),
            "mcc": float(matthews_corrcoef(cls_targets_np, cls_pred_binary))
        }
    
    return float(avg_loss), metrics, results, avg_task_losses

def mc_dropout_predict(model: nn.Module, data: Data, device: torch.device, n_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用MC Dropout进行不确定性估计预测
    """
    model.eval()
    model.enable_mc_dropout()
    
    predictions = []
    data = data.to(str(device))
    
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(data)
            if "regression" in outputs:
                predictions.append(outputs["regression"].cpu().numpy())
    
    model.disable_mc_dropout()
    
    # 计算平均值和标准差
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred

# ------------- 6. 化学空间聚类 -------------
def cluster_materials(dataset: GraphDataset, n_clusters: int = 5) -> np.ndarray:
    """
    根据材料化学组成对材料进行聚类:
    简单实现，实际可采用论文中的ELMD
    """
    try:
        # 从文件名提取化学信息并简单向量化
        compositions = dataset.get_all_compositions()
        
        # 简单特征提取 - 元素计数
        element_counts = defaultdict(list)
        all_elements = set()
        
        for comp in compositions:
            # 分析化学式 (简化版)
            elements = {}
            import re
            matches = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', comp)
            for el, count in matches:
                count = float(count) if count else 1.0
                elements[el] = elements.get(el, 0) + count
                all_elements.add(el)
        
            # 将元素添加到计数字典
            for el in all_elements:
                element_counts[el].append(elements.get(el, 0))
        
        # 创建特征矩阵
        all_elements = sorted(list(all_elements))
        X = np.zeros((len(compositions), len(all_elements)))
        
        for i, el in enumerate(all_elements):
            X[:, i] = element_counts[el]
        
        # 标准化
        X = StandardScaler().fit_transform(X)
        
        # PCA降维
        pca = PCA(n_components=min(10, X.shape[1]))
        X_pca = pca.fit_transform(X)
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=1.0, min_samples=3).fit(X_pca)
        
        # 处理噪声点 (-1标签)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        n_clusters_found = len(unique_labels)
        print(f"[聚类] 发现 {n_clusters_found} 个聚类，包含噪声点的比例: {(labels == -1).sum() / len(labels):.2%}")
        
        # 如果聚类太少，使用KMeans强制分组
        if n_clusters_found < 3:
            from sklearn.cluster import KMeans
            clustering = KMeans(n_clusters=n_clusters).fit(X_pca)
            labels = clustering.labels_
            print(f"[聚类] 使用KMeans重新分组为 {n_clusters} 个聚类")
        
        return labels
    except Exception as e:
        print(f"[聚类错误] {e}")
        # 失败时返回随机聚类
        return np.random.randint(0, n_clusters, size=len(dataset))

# ------------- 7. LOCO交叉验证 -------------
def loco_cross_validation(dataset: GraphDataset, model_class: type, device: torch.device, n_clusters: int = 5, **model_kwargs) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    实现Leave-One-Cluster-Out交叉验证
    """
    # 聚类分组
    cluster_labels = cluster_materials(dataset, n_clusters)
    unique_clusters = np.unique(cluster_labels)
    
    # 收集结果
    all_metrics: List[Dict[str, Any]] = []
    
    for test_cluster in unique_clusters:
        print(f"\n[LOCO-CV] 测试聚类 {test_cluster}")
        
        # 根据聚类分割数据
        train_indices = [i for i, label in enumerate(cluster_labels) if label != test_cluster]
        test_indices = [i for i, label in enumerate(cluster_labels) if label == test_cluster]
        
        if len(test_indices) == 0:
            continue
            
        print(f"[LOCO-CV] 训练样本: {len(train_indices)}, 测试样本: {len(test_indices)}")
        
        # 使用 IndexSubset
        train_dataset = IndexSubset(dataset, train_indices)
        test_dataset = IndexSubset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # 确定模型输入维度 (安全获取)
        try:
            sample_batch = next(iter(train_loader))
            node_in_dim = sample_batch.x.size(1) if sample_batch.x.size(0) else 0
            edge_in_dim = sample_batch.edge_attr.size(1) if hasattr(sample_batch, 'edge_attr') and sample_batch.edge_attr is not None and sample_batch.edge_attr.size(0) else 0
        except (StopIteration, AttributeError) as e:
            print(f"[LOCO-CV] 无法获取样本批次: {e}")
            continue
        
        if node_in_dim == 0:
            print("[LOCO-CV] 节点特征维度为0，无法训练")
            continue
        
        if edge_in_dim == 0:
            edge_in_dim = 1
        
        # 创建模型
        model = model_class(
            node_in_features=node_in_dim,
            edge_in_features=edge_in_dim,
            **model_kwargs
        ).to(device)
        
        # 创建优化器和损失函数
        optimizer: Optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE)
        
        criterion_dict = {
            "regression": nn.MSELoss(),
            "classification": nn.BCELoss()
        }
        
        # 训练
        best_val_loss = float('inf')
        best_metrics: Optional[Dict[str, Dict[str, float]]] = None
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            # 训练一个周期
            train_loss, task_losses = train_one_epoch(
                model, train_loader, optimizer, criterion_dict, device, task_type=TASK_TYPE
            )
            
            # 验证
            val_loss, val_metrics, _, val_task_losses = evaluate_model(
                model, test_loader, criterion_dict, device, task_type=TASK_TYPE
            )
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 打印进度
            if epoch % 10 == 0:
                print(f"[LOCO-CV] Epoch {epoch}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if val_metrics.get("regression"):
                    print(f"  回归 - MAE: {val_metrics['regression']['mae']:.4f}, R²: {val_metrics['regression']['r2']:.4f}")
                
                if val_metrics.get("classification"):
                    cls_metrics = val_metrics["classification"]
                    print(f"  分类 - Acc: {cls_metrics['accuracy']:.4f}, MCC: {cls_metrics['mcc']:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= 20:
                print(f"[LOCO-CV] 早停于 epoch {epoch}")
                break
        
        # 收集结果（强转基础类型以便 JSON 序列化）
        all_metrics.append({
            "cluster": int(test_cluster),
            "test_size": int(len(test_indices)),
            "metrics": best_metrics or {}
        })
    
    # 计算总体指标
    overall_metrics: Dict[str, Dict[str, float]] = {}
    
    # 回归指标
    if any(m.get("metrics") and "regression" in m["metrics"] for m in all_metrics):
        mae_values = [float(m["metrics"]["regression"]["mae"]) for m in all_metrics if m.get("metrics") and "regression" in m["metrics"]]
        r2_values = [float(m["metrics"]["regression"]["r2"]) for m in all_metrics if m.get("metrics") and "regression" in m["metrics"]]
        
        overall_metrics["regression"] = {
            "mae": float(np.mean(mae_values)) if mae_values else float('nan'),
            "mae_std": float(np.std(mae_values)) if mae_values else float('nan'),
            "r2": float(np.mean(r2_values)) if r2_values else float('nan'),
            "r2_std": float(np.std(r2_values)) if r2_values else float('nan')
        }
    
    # 分类指标
    if any(m.get("metrics") and "classification" in m["metrics"] for m in all_metrics):
        acc_values = [float(m["metrics"]["classification"]["accuracy"]) for m in all_metrics if m.get("metrics") and "classification" in m["metrics"]]
        mcc_values = [float(m["metrics"]["classification"]["mcc"]) for m in all_metrics if m.get("metrics") and "classification" in m["metrics"]]
        f1_values  = [float(m["metrics"]["classification"]["f1"]) for m in all_metrics if m.get("metrics") and "classification" in m["metrics"]]
        
        overall_metrics["classification"] = {
            "accuracy": float(np.mean(acc_values)) if acc_values else float('nan'),
            "accuracy_std": float(np.std(acc_values)) if acc_values else float('nan'),
            "mcc": float(np.mean(mcc_values)) if mcc_values else float('nan'),
            "mcc_std": float(np.std(mcc_values)) if mcc_values else float('nan'),
            "f1": float(np.mean(f1_values)) if f1_values else float('nan'),
            "f1_std": float(np.std(f1_values)) if f1_values else float('nan')
        }
    
    print("\n[LOCO-CV 总结] ====================")
    if "regression" in overall_metrics:
        print(f"回归 - 平均MAE: {overall_metrics['regression']['mae']:.4f} ± {overall_metrics['regression']['mae_std']:.4f}")
        print(f"回归 - 平均R²: {overall_metrics['regression']['r2']:.4f} ± {overall_metrics['regression']['r2_std']:.4f}")
    
    if "classification" in overall_metrics:
        print(f"分类 - 平均准确率: {overall_metrics['classification']['accuracy']:.4f} ± {overall_metrics['classification']['accuracy_std']:.4f}")
        print(f"分类 - 平均MCC: {overall_metrics['classification']['mcc']:.4f} ± {overall_metrics['classification']['mcc_std']:.4f}")
        print(f"分类 - 平均F1: {overall_metrics['classification']['f1']:.4f} ± {overall_metrics['classification']['f1_std']:.4f}")
    
    return overall_metrics, all_metrics

# ------------- 8. 结果可视化 -------------
def plot_results(results: Dict[str, Tuple[np.ndarray, np.ndarray]], model_name: str, save_dir: Optional[str] = None) -> None:
    """结果可视化"""
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 回归结果可视化
    if "regression" in results:
        preds, targets = results["regression"]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, preds, alpha=0.6)
        
        # 添加对角线
        min_val = min(min(targets), min(preds))
        max_val = max(max(targets), max(preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'{model_name} - Regression Performance')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{model_name}_regression.png"), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 误差分布
        errors = preds - targets
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title(f'{model_name} - Error Distribution')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{model_name}_error_dist.png"), dpi=300)
            plt.close()
        else:
            plt.show()
    
    # 分类结果可视化
    if "classification" in results:
        probs, targets = results["classification"]
        preds = (probs >= 0.5).astype(int)
        
        # 混淆矩阵
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} - Confusion Matrix')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # ROC曲线
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc="lower right")
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curve.png"), dpi=300)
            plt.close()
        else:
            plt.show()

# ------------- 9. 主函数 -------------
def main() -> None:
    # 检查目录
    if not os.path.exists(GRAPHS_DIR):
        print(f"[错误] 找不到 {GRAPHS_DIR}")
        return
    
    # 创建数据集（初始化即完成过滤）
    dataset = GraphDataset(GRAPHS_DIR, task=TASK_TYPE, threshold=CLASSIFICATION_THRESHOLD)
    if dataset.len() == 0:
        print(f"[错误] {GRAPHS_DIR} 下无可用的 .pt 文件（可能均缺少标签或特征为空）")
        return

    # 已过滤后的有效样本数
    valid_count = dataset.len()
    print(f"[数据检查] 有效样本数: {valid_count}/{valid_count}")
    
    # 创建结果目录
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用 {device}")

    # 常规训练与评估
    print("\n[训练] 开始标准训练流程...")
    
    # 数据集切分
    idxs = list(range(dataset.len()))
    train_idxs, test_idxs = train_test_split(idxs, test_size=0.2, random_state=SEED)
    train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.2, random_state=SEED)

    # 使用 IndexSubset 避免类型告警
    train_ds = IndexSubset(dataset, train_idxs)
    val_ds = IndexSubset(dataset, val_idxs)
    test_ds = IndexSubset(dataset, test_idxs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    print(f"[数据] 训练集: {len(train_ds)}, 验证集: {len(val_ds)}, 测试集: {len(test_ds)}")

    # 确定节点/边特征维度（安全获取）
    try:
        sample_batch = next(iter(train_loader))
    except StopIteration:
        print("[错误] 训练集为空，无法训练。")
        return

    node_in_dim = sample_batch.x.size(1) if sample_batch.x.size(0) else 0
    edge_in_dim = sample_batch.edge_attr.size(1) if hasattr(sample_batch, 'edge_attr') and sample_batch.edge_attr is not None and sample_batch.edge_attr.size(0) else 0
    
    if node_in_dim == 0:
        print("[错误] 节点特征维度为0，无法训练")
        return
    
    if edge_in_dim == 0:
        edge_in_dim = 1
        
    print(f"[模型] 节点特征: {node_in_dim}维, 边特征: {edge_in_dim}维")

    # 创建增强GNN模型
    model = EnhancedCEGNet(
        node_in_features=node_in_dim, 
        edge_in_features=edge_in_dim,
        hidden_dim=HIDDEN_DIM, 
        output_dim=1, 
        dropout_rate=DROPOUT_RATE,
        task_type=TASK_TYPE,
        heads=2
    ).to(device)
    
    # 损失函数
    criterion_dict = {
        "regression": nn.MSELoss(),
        "classification": nn.BCELoss()
    }

    # 优化器与学习率调度器
    optimizer: Optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE)

    # 模型保存路径
    best_path = os.path.join(MODELS_DIR, "best_enhanced_cegnet_model.pt")

    # 训练
    best_val_loss = float('inf')
    best_epoch = -1
    early_stop_counter = 0
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "lr": []
    }

    print("\n[训练] 开始训练...")
    for ep in range(NUM_EPOCHS):
        # 训练一个周期
        tr_loss, task_losses = train_one_epoch(model, train_loader, optimizer, criterion_dict, device, task_type=TASK_TYPE)
        
        # 验证
        val_loss, val_metrics, _, val_task_losses = evaluate_model(model, val_loader, criterion_dict, device, task_type=TASK_TYPE)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # 记录训练历史
        training_history["train_loss"].append(tr_loss)
        training_history["val_loss"].append(val_loss)
        training_history["lr"].append(current_lr)
        
        # 打印进度
        log_msg = f"[Epoch {ep+1}/{NUM_EPOCHS}] Train:{tr_loss:.4f}, Val:{val_loss:.4f}, LR:{current_lr:.6f}"
        
        # 任务特定进度
        if TASK_TYPE in ["regression", "both"] and "regression" in val_metrics:
            reg_metrics = val_metrics["regression"]
            log_msg += f" | Reg - MAE:{reg_metrics['mae']:.4f}, R²:{reg_metrics['r2']:.4f}"
        
        if TASK_TYPE in ["classification", "both"] and "classification" in val_metrics:
            cls_metrics = val_metrics["classification"]
            log_msg += f" | Cls - Acc:{cls_metrics['accuracy']:.4f}, MCC:{cls_metrics['mcc']:.4f}"
        
        print(log_msg)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # 早停
        if early_stop_counter >= 20:
            print(f"[训练] 早停于 epoch {ep+1}")
            break

    print(f"\n[训练] 完成. 最佳验证损失:{best_val_loss:.4f} (Epoch {best_epoch+1})")

    # 测试最佳模型
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, test_metrics, test_results, _ = evaluate_model(model, test_loader, criterion_dict, device, task_type=TASK_TYPE)
        
        print("\n[测试] 最佳模型性能:")
        if "regression" in test_metrics:
            reg_metrics = test_metrics["regression"]
            print(f"  回归 - MSE: {reg_metrics['mse']:.4f}, MAE: {reg_metrics['mae']:.4f}, R²: {reg_metrics['r2']:.4f}")
        
        if "classification" in test_metrics:
            cls_metrics = test_metrics["classification"]
            print(f"  分类 - Accuracy: {cls_metrics['accuracy']:.4f}, Precision: {cls_metrics['precision']:.4f}")
            print(f"         Recall: {cls_metrics['recall']:.4f}, F1: {cls_metrics['f1']:.4f}, MCC: {cls_metrics['mcc']:.4f}")
        
        # 绘制结果
        plot_results(test_results, "EnhancedCEGNet", save_dir=RESULTS_DIR)
        
        # 不确定性估计示例 (仅对测试集的前几个样本)
        print("\n[不确定性] 使用MC Dropout进行不确定性估计...")
        n_samples = min(5, len(test_ds))
        for i in range(n_samples):
            sample_data = test_ds.get(i)  # 显式 get，返回 Data
            true_value = float(cast(torch.Tensor, sample_data.y).view(-1)[0].item())
            
            mean_pred, std_pred = mc_dropout_predict(model, sample_data, device, n_samples=30)
            print(f"样本 {i+1} - 真实值: {true_value:.4f}, 预测值: {float(mean_pred[0]):.4f} ± {float(std_pred[0]):.4f}")
    else:
        print("[警告] 未找到最佳模型文件，跳过测试。")
    
    # LOCO交叉验证
    print("\n[LOCO-CV] 开始Leave-One-Cluster-Out交叉验证...")
    loco_metrics, cluster_metrics = loco_cross_validation(
        dataset=dataset,
        model_class=EnhancedCEGNet,
        device=device,
        n_clusters=5,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
        dropout_rate=DROPOUT_RATE,
        task_type=TASK_TYPE,
        heads=2
    )
    
    # 保存LOCO-CV结果
    loco_results = {
        "overall": loco_metrics,
        "per_cluster": cluster_metrics
    }
    with open(os.path.join(RESULTS_DIR, "loco_cv_results.json"), 'w') as f:
        json.dump(loco_results, f, indent=2, default=_np_json_default)

if __name__ == "__main__":
    main()