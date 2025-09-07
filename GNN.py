import os
import json
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
from torch_geometric.nn import MessagePassing, global_mean_pool  # 仅保留mean池化

from sklearn.model_selection import train_test_split
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

GRAPHS_DIR    = "processed_data/graphs"
MODELS_DIR    = "models"
RESULTS_DIR   = "results"

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
USE_Y_STANDARDIZE   = True

# 多任务与优化目标
CLASSIFICATION_THRESHOLD = -4.0
TASK_TYPE     = "both"
LOSS_W_REG    = 1.0
LOSS_W_CLS    = 0.2

# 验证集优化指标：'r2' | 'neg_mse'
OPTIMIZE_FOR          = "r2"
EARLY_STOP_PATIENCE   = 0     # =0 禁用早停
EARLY_STOP_MIN_DELTA  = 1e-4   # 指标提升最小幅度（R²）

# ------------- 工具：numpy -> JSON 序列化 -------------
def _np_json_default(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _get_val_score(val_metrics: Dict[str, Dict[str, float]], val_loss: float) -> float:
    # 优先使用回归R²；没有时用 -val_loss 兜底
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

# ------------- 1. 数据集（含过滤） -------------
class GraphDataset(GeoDataset):
    def __init__(self, root_dir: str, task: str = "both", threshold: float = CLASSIFICATION_THRESHOLD):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.threshold = threshold
        self.file_list: List[str] = []
        self.compositions: List[str] = []
        
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
        data_obj: Data = torch.load(full_path)

        if (not hasattr(data_obj, 'y') 
            or not isinstance(getattr(data_obj, 'y'), torch.Tensor) 
            or cast(torch.Tensor, data_obj.y).numel() == 0):
            raise RuntimeError(f"[错误] 文件 {pt_fname} 缺少有效的 y 标签。")
        if (not hasattr(data_obj, 'x')
            or not isinstance(data_obj.x, torch.Tensor)
            or data_obj.x.dim() != 2
            or data_obj.x.size(1) == 0
            or data_obj.x.size(0) == 0):
            raise RuntimeError(f"[错误] 文件 {pt_fname} 的节点特征为空。")
            
        if self.task in ["classification", "both"]:
            y_t = cast(torch.Tensor, data_obj.y).view(-1)
            y_val = float(y_t[0].item())
            data_obj.y_cls = torch.tensor(1.0 if y_val >= self.threshold else 0.0, dtype=torch.float)
        data_obj.material_id = os.path.splitext(pt_fname)[0]
        return data_obj
    
    def get_all_compositions(self) -> List[str]:
        return self.compositions

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

# ------------- 2. 精简版消息传递层 -------------
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

# ------------- 3. 轻量模型（特征门控 + 瓶颈 + 2层消息传递 + mean池化） -------------
class LiteCEGNet(nn.Module):
    def __init__(self, node_in_features: int, edge_in_features: int, hidden_dim: int,
                 output_dim: int = 1, dropout_rate: float = 0.2, task_type: str = "both",
                 bottleneck_dim: int = NODE_BOTTLENECK_DIM, l1_gate_coef: float = L1_GATE_COEF):
        super().__init__()
        self.task_type = task_type
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

        # 预测头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.regressor = nn.Linear(hidden_dim // 2, output_dim)
        self.classifier = nn.Linear(hidden_dim // 2, 1)

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

        g = self.pool(x, batch)
        g = self.fc(g)

        outputs: Dict[str, torch.Tensor] = {}
        if self.task_type in ["regression", "both"]:
            outputs["regression"] = self.regressor(g).squeeze(-1)
        if self.task_type in ["classification", "both"] and hasattr(data, 'y_cls'):
            outputs["classification"] = torch.sigmoid(self.classifier(g)).squeeze(-1)

        return outputs

    def regularization_loss(self) -> torch.Tensor:
        return self.l1_gate_coef * torch.sum(torch.sigmoid(self.feature_gates))

# ------------- 4. 训练/评估（支持y标准化 + 指标加权） -------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Optimizer, 
                    criterion_dict: Dict[str, nn.Module], device: torch.device, task_type: str = "both",
                    y_norm: Optional[Tuple[float, float]] = None) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    task_losses: Dict[str, float] = {"regression": 0.0, "classification": 0.0}
    total_graphs = 0
    
    for batch_data in loader:
        if (not hasattr(batch_data, 'x')) or batch_data.x is None or batch_data.x.size(0) == 0:
            continue
        if (not hasattr(batch_data, 'y')) or (not isinstance(batch_data.y, torch.Tensor)) or batch_data.y.numel() == 0:
            continue

        batch_data = batch_data.to(str(device))
        optimizer.zero_grad()
        outputs = model(batch_data)
        
        loss: torch.Tensor = torch.tensor(0.0, device=device)

        # 回归损失（标准化）
        if task_type in ["regression", "both"] and "regression" in outputs:
            target_y = cast(torch.Tensor, batch_data.y)
            if y_norm is not None:
                mu, std = y_norm
                std = std if std > 0 else 1.0
                target_y = (target_y - mu) / std
            reg_loss = criterion_dict["regression"](outputs["regression"], target_y)
            loss = loss + LOSS_W_REG * reg_loss
            task_losses["regression"] += float(reg_loss.detach().item()) * batch_data.num_graphs

        # 分类损失
        if task_type in ["classification", "both"] and "classification" in outputs and hasattr(batch_data, 'y_cls'):
            cls_loss = criterion_dict["classification"](outputs["classification"], cast(torch.Tensor, batch_data.y_cls))
            loss = loss + LOSS_W_CLS * cls_loss
            task_losses["classification"] += float(cls_loss.detach().item()) * batch_data.num_graphs

        # 特征门控正则
        if hasattr(model, "regularization_loss"):
            loss = loss + cast(LiteCEGNet, model).regularization_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.detach().item()) * batch_data.num_graphs
        total_graphs += batch_data.num_graphs

    if total_graphs > 0:
        return (
            total_loss / total_graphs,
            {k: (v / total_graphs) for k, v in task_losses.items() if v > 0.0}
        )
    return 0.0, {k: 0.0 for k in task_losses.keys()}

def evaluate_model(model: nn.Module, loader: DataLoader, criterion_dict: Dict[str, nn.Module], 
                   device: torch.device, task_type: str = "both", y_norm: Optional[Tuple[float, float]] = None
                   ) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    task_losses: Dict[str, float] = {"regression": 0.0, "classification": 0.0}
    total_graphs = 0
    
    reg_preds, reg_targets = [], []
    cls_preds, cls_targets = [], []
    
    with torch.no_grad():
        for batch_data in loader:
            if (not hasattr(batch_data, 'x')) or batch_data.x is None or batch_data.x.size(0) == 0:
                continue
            if (not hasattr(batch_data, 'y')) or (not isinstance(batch_data.y, torch.Tensor)) or batch_data.y.numel() == 0:
                continue

            batch_data = batch_data.to(str(device))
            outputs = model(batch_data)
            
            loss: torch.Tensor = torch.tensor(0.0, device=device)
            # 回归
            if task_type in ["regression", "both"] and "regression" in outputs:
                target_y = cast(torch.Tensor, batch_data.y)
                if y_norm is not None:
                    mu, std = y_norm
                    std = std if std > 0 else 1.0
                    target_y = (target_y - mu) / std
                reg_loss = criterion_dict["regression"](outputs["regression"], target_y)
                loss = loss + LOSS_W_REG * reg_loss
                task_losses["regression"] += float(reg_loss.detach().item()) * batch_data.num_graphs

                # 收集预测（反标准化）
                preds_np = outputs["regression"].cpu().numpy()
                if y_norm is not None:
                    mu, std = y_norm
                    preds_np = preds_np * (std if std > 0 else 1.0) + mu
                reg_preds.append(preds_np)
                reg_targets.append(cast(torch.Tensor, batch_data.y).cpu().numpy())
                
            # 分类
            if task_type in ["classification", "both"] and "classification" in outputs and hasattr(batch_data, 'y_cls'):
                cls_loss = criterion_dict["classification"](outputs["classification"], cast(torch.Tensor, batch_data.y_cls))
                loss = loss + LOSS_W_CLS * cls_loss
                task_losses["classification"] += float(cls_loss.detach().item()) * batch_data.num_graphs

                cls_preds.append(outputs["classification"].cpu().numpy())
                cls_targets.append(cast(torch.Tensor, batch_data.y_cls).cpu().numpy())

            total_loss += float(loss.detach().item()) * batch_data.num_graphs
            total_graphs += batch_data.num_graphs
    
    if total_graphs == 0:
        return float('inf'), {}, {}, {}
        
    avg_loss = total_loss / total_graphs
    avg_task_losses = {k: float(v / total_graphs) for k, v in task_losses.items() if v > 0.0}
    
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    
    if task_type in ["regression", "both"] and reg_preds:
        reg_preds_np = np.concatenate(reg_preds).flatten()
        reg_targets_np = np.concatenate(reg_targets).flatten()
        results["regression"] = (reg_preds_np, reg_targets_np)
        metrics["regression"] = {
            "mae": float(mean_absolute_error(reg_targets_np, reg_preds_np)),
            "r2": float(r2_score(reg_targets_np, reg_preds_np)),
            "mse": float(np.mean((reg_preds_np - reg_targets_np) ** 2))
        }
    
    if task_type in ["classification", "both"] and cls_preds:
        cls_preds_np = np.concatenate(cls_preds).flatten()
        cls_targets_np = np.concatenate(cls_targets).flatten()
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

def mc_dropout_predict(model: nn.Module, data: Data, device: torch.device, n_samples: int = 30, y_norm: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    data = data.to(str(device))
    with torch.no_grad():
        outputs = model(data)
    preds = outputs["regression"].cpu().numpy()
    if y_norm is not None:
        mu, std = y_norm
        preds = preds * (std if std > 0 else 1.0) + mu
    return preds, np.zeros_like(preds)

# ------------- 5. 化学空间聚类 -------------
def cluster_materials(dataset: GraphDataset, n_clusters: int = 5) -> np.ndarray:
    try:
        compositions = dataset.get_all_compositions()
        element_counts = defaultdict(list)
        all_elements = set()
        
        for comp in compositions:
            elements = {}
            import re
            matches = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', comp)
            for el, count in matches:
                count = float(count) if count else 1.0
                elements[el] = elements.get(el, 0) + count
                all_elements.add(el)
            for el in all_elements:
                element_counts[el].append(elements.get(el, 0))
        
        all_elements = sorted(list(all_elements))
        X = np.zeros((len(compositions), len(all_elements)))
        for i, el in enumerate(all_elements):
            X[:, i] = element_counts[el]
        
        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(10, X.shape[1]))
        X_pca = pca.fit_transform(X)
        
        clustering = DBSCAN(eps=1.0, min_samples=3).fit(X_pca)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        n_clusters_found = len(unique_labels)
        print(f"[聚类] 发现 {n_clusters_found} 个聚类，噪声比例: {(labels == -1).sum() / len(labels):.2%}")
        
        if n_clusters_found < 3:
            from sklearn.cluster import KMeans
            clustering = KMeans(n_clusters=n_clusters).fit(X_pca)
            labels = clustering.labels_
            print(f"[聚类] 使用KMeans重新分为 {n_clusters} 类")
        
        return labels
    except Exception as e:
        print(f"[聚类错误] {e}")
        return np.random.randint(0, n_clusters, size=len(dataset))

# ------------- 6. LOCO-CV（以R²为早停目标） -------------
def loco_cross_validation(dataset: GraphDataset, model_class: type, device: torch.device, n_clusters: int = 5, **model_kwargs) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    cluster_labels = cluster_materials(dataset, n_clusters)
    unique_clusters = np.unique(cluster_labels)
    all_metrics: List[Dict[str, Any]] = []
    
    for test_cluster in unique_clusters:
        print(f"\n[LOCO-CV] 测试聚类 {test_cluster}")
        train_indices = [i for i, label in enumerate(cluster_labels) if label != test_cluster]
        test_indices = [i for i, label in enumerate(cluster_labels) if label == test_cluster]
        if len(test_indices) == 0:
            continue
        print(f"[LOCO-CV] 训练样本: {len(train_indices)}, 测试样本: {len(test_indices)}")
        
        train_dataset = IndexSubset(dataset, train_indices)
        test_dataset = IndexSubset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # 训练集y标准化参数
        y_vals = []
        for i in range(len(train_dataset)):
            d = train_dataset.get(i)
            y_vals.append(float(cast(torch.Tensor, d.y).view(-1)[0].item()))
        if len(y_vals) == 0:
            print("[LOCO-CV] 训练集y为空")
            continue
        y_mu = float(np.mean(y_vals))
        y_std = float(np.std(y_vals)) if USE_Y_STANDARDIZE else 1.0
        y_norm = (y_mu, y_std if y_std > 1e-12 else 1.0)

        # 输入维度
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
        
        model = model_class(
            node_in_features=node_in_dim,
            edge_in_features=edge_in_dim,
            **model_kwargs
        ).to(device)
        
        optimizer: Optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # 学习率调度器改为“max”，监控R²
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=LR_PATIENCE)
        criterion_dict = {
            "regression": nn.MSELoss(),
            "classification": nn.BCELoss()
        }
        
        best_val_score = -float('inf')
        best_metrics: Optional[Dict[str, Dict[str, float]]] = None
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            train_one_epoch(
                model, train_loader, optimizer, criterion_dict, device, task_type=TASK_TYPE, y_norm=y_norm if USE_Y_STANDARDIZE else None
            )
            val_loss, val_metrics, _, _ = evaluate_model(
                model, test_loader, criterion_dict, device, task_type=TASK_TYPE, y_norm=y_norm if USE_Y_STANDARDIZE else None
            )

            val_score = _get_val_score(val_metrics, val_loss)
            scheduler.step(val_score)

            if epoch % 10 == 0:
                msg = f"[LOCO-CV] Epoch {epoch}/{NUM_EPOCHS}, ValScore(R²优先) {val_score:.4f}"
                if val_metrics.get("regression"):
                    msg += f" | Reg MAE {val_metrics['regression']['mae']:.4f}, R² {val_metrics['regression']['r2']:.4f}"
                if val_metrics.get("classification"):
                    cm = val_metrics["classification"]
                    msg += f" | Cls Acc {cm['accuracy']:.4f}, MCC {cm['mcc']:.4f}"
                print(msg)

            # 以R²为准保存最佳
            if val_score > best_val_score + EARLY_STOP_MIN_DELTA:
                best_val_score = val_score
                best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1

            if EARLY_STOP_PATIENCE and EARLY_STOP_PATIENCE > 0 and patience_counter >= EARLY_STOP_PATIENCE:
                print(f"[LOCO-CV] 早停于 epoch {epoch}（按R²，耐心={EARLY_STOP_PATIENCE}）")
                break
        
        all_metrics.append({
            "cluster": int(test_cluster),
            "test_size": int(len(test_indices)),
            "metrics": best_metrics or {}
        })
    
    overall_metrics: Dict[str, Dict[str, float]] = {}
    if any(m.get("metrics") and "regression" in m["metrics"] for m in all_metrics):
        mae_values = [float(m["metrics"]["regression"]["mae"]) for m in all_metrics if m.get("metrics") and "regression" in m["metrics"]]
        r2_values = [float(m["metrics"]["regression"]["r2"]) for m in all_metrics if m.get("metrics") and "regression" in m["metrics"]]
        overall_metrics["regression"] = {
            "mae": float(np.mean(mae_values)) if mae_values else float('nan'),
            "mae_std": float(np.std(mae_values)) if mae_values else float('nan'),
            "r2": float(np.mean(r2_values)) if r2_values else float('nan'),
            "r2_std": float(np.std(r2_values)) if r2_values else float('nan')
        }
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
    return overall_metrics, all_metrics

# ------------- 7. 可视化 -------------
def plot_results(results: Dict[str, Tuple[np.ndarray, np.ndarray]], model_name: str, save_dir: Optional[str] = None) -> None:
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if "regression" in results:
        preds, targets = results["regression"]
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, preds, alpha=0.6)
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
    if "classification" in results:
        probs, targets = results["classification"]
        preds = (probs >= 0.5).astype(int)
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

# ------------- 8. 主函数（以R²为“最佳”标准） -------------
def main() -> None:
    if not os.path.exists(GRAPHS_DIR):
        print(f"[错误] 找不到 {GRAPHS_DIR}")
        return
    
    dataset = GraphDataset(GRAPHS_DIR, task=TASK_TYPE, threshold=CLASSIFICATION_THRESHOLD)
    if dataset.len() == 0:
        print(f"[错误] {GRAPHS_DIR} 下无可用 .pt")
        return

    valid_count = dataset.len()
    print(f"[数据检查] 有效样本数: {valid_count}/{valid_count}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用 {device}")

    print("\n[训练] 开始标准训练流程...")
    
    idxs = list(range(dataset.len()))
    train_idxs, test_idxs = train_test_split(idxs, test_size=0.2, random_state=SEED)
    train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.2, random_state=SEED)

    train_ds = IndexSubset(dataset, train_idxs)
    val_ds = IndexSubset(dataset, val_idxs)
    test_ds = IndexSubset(dataset, test_idxs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    print(f"[数据] 训练集: {len(train_ds)}, 验证集: {len(val_ds)}, 测试集: {len(test_ds)}")

    # 训练集y标准化
    y_vals = []
    for i in range(len(train_ds)):
        d = train_ds.get(i)
        y_vals.append(float(cast(torch.Tensor, d.y).view(-1)[0].item()))
    y_mu = float(np.mean(y_vals))
    y_std = float(np.std(y_vals)) if USE_Y_STANDARDIZE else 1.0
    y_norm = (y_mu, y_std if y_std > 1e-12 else 1.0)
    print(f"[回归标准化] mean={y_mu:.4f}, std={y_norm[1]:.4f}")

    # 输入维度
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

    # 轻量模型
    model = LiteCEGNet(
        node_in_features=node_in_dim, 
        edge_in_features=edge_in_dim,
        hidden_dim=HIDDEN_DIM, 
        output_dim=1, 
        dropout_rate=DROPOUT_RATE,
        task_type=TASK_TYPE,
        bottleneck_dim=NODE_BOTTLENECK_DIM,
        l1_gate_coef=L1_GATE_COEF
    ).to(device)
    
    criterion_dict = {
        "regression": nn.MSELoss(),     # 用MSE与R²一致
        "classification": nn.BCELoss()
    }

    optimizer: Optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 学习率调度器改为“max”，监控R²
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=LR_PATIENCE)

    best_path = os.path.join(MODELS_DIR, "best_lite_cegnet.pt")

    best_val_score = -float('inf')  # 以R²为准
    best_epoch = -1
    patience_counter = 0

    print("\n[训练] 开始训练...")
    for ep in range(NUM_EPOCHS):
        tr_loss, _ = train_one_epoch(
            model, train_loader, optimizer, criterion_dict, device, task_type=TASK_TYPE,
            y_norm=y_norm if USE_Y_STANDARDIZE else None
        )
        val_loss, val_metrics, _, _ = evaluate_model(
            model, val_loader, criterion_dict, device, task_type=TASK_TYPE,
            y_norm=y_norm if USE_Y_STANDARDIZE else None
        )

        val_score = _get_val_score(val_metrics, val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_score)
        
        log_msg = f"[Epoch {ep+1}/{NUM_EPOCHS}] TrainLoss:{tr_loss:.4f}, ValScore(R²优先):{val_score:.4f}, LR:{current_lr:.6f}"
        if TASK_TYPE in ["regression", "both"] and "regression" in val_metrics:
            reg_metrics = val_metrics["regression"]
            log_msg += f" | Reg - MAE:{reg_metrics['mae']:.4f}, R²:{reg_metrics['r2']:.4f}"
        if TASK_TYPE in ["classification", "both"] and "classification" in val_metrics:
            cls_metrics = val_metrics["classification"]
            log_msg += f" | Cls - Acc:{cls_metrics['accuracy']:.4f}, MCC:{cls_metrics['mcc']:.4f}"
        print(log_msg)
        
        # 以R²为准保存最佳
        if val_score > best_val_score + EARLY_STOP_MIN_DELTA:
            best_val_score = val_score
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # 可配置早停（按R²）
        if EARLY_STOP_PATIENCE and EARLY_STOP_PATIENCE > 0 and patience_counter >= EARLY_STOP_PATIENCE:
            print(f"[训练] 早停于 epoch {ep+1}（按R²，耐心={EARLY_STOP_PATIENCE}）")
            break

    print(f"\n[训练] 完成. 最佳验证R²:{best_val_score:.4f} (Epoch {best_epoch+1})")
    print("[特征门控] sigmoid(gates):", torch.sigmoid(model.feature_gates).detach().cpu().numpy())

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, test_metrics, test_results, _ = evaluate_model(
            model, test_loader, criterion_dict, device, task_type=TASK_TYPE,
            y_norm=y_norm if USE_Y_STANDARDIZE else None
        )
        print("\n[测试] 最佳模型性能:")
        if "regression" in test_metrics:
            reg_metrics = test_metrics["regression"]
            print(f"  回归 - MSE: {reg_metrics['mse']:.4f}, MAE: {reg_metrics['mae']:.4f}, R²: {reg_metrics['r2']:.4f}")
        if "classification" in test_metrics:
            cls_metrics = test_metrics["classification"]
            print(f"  分类 - Accuracy: {cls_metrics['accuracy']:.4f}, Precision: {cls_metrics['precision']:.4f}")
            print(f"         Recall: {cls_metrics['recall']:.4f}, F1: {cls_metrics['f1']:.4f}, MCC: {cls_metrics['mcc']:.4f}")
        plot_results(test_results, "LiteCEGNet", save_dir=RESULTS_DIR)

        print("\n[不确定性] 简单预测（含反标准化）...")
        n_samples = min(5, len(test_ds))
        for i in range(n_samples):
            sample_data = test_ds.get(i)
            true_value = float(cast(torch.Tensor, sample_data.y).view(-1)[0].item())
            mean_pred, _ = mc_dropout_predict(model, sample_data, device, n_samples=1, y_norm=y_norm if USE_Y_STANDARDIZE else None)
            print(f"样本 {i+1} - 真实值: {true_value:.4f}, 预测值: {float(mean_pred[0]):.4f}")
    else:
        print("[警告] 未找到最佳模型文件，跳过测试。")
    
    print("\n[LOCO-CV] 开始Leave-One-Cluster-Out交叉验证...")
    loco_metrics, cluster_metrics = loco_cross_validation(
        dataset=dataset,
        model_class=LiteCEGNet,
        device=device,
        n_clusters=5,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
        dropout_rate=DROPOUT_RATE,
        task_type=TASK_TYPE,
        bottleneck_dim=NODE_BOTTLENECK_DIM,
        l1_gate_coef=L1_GATE_COEF
    )
    loco_results = {
        "overall": loco_metrics,
        "per_cluster": cluster_metrics
    }
    with open(os.path.join(RESULTS_DIR, "loco_cv_results.json"), 'w') as f:
        json.dump(loco_results, f, indent=2, default=_np_json_default)

if __name__ == "__main__":
    main()