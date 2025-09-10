import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, cast

from torch_geometric.data import Data, Dataset as GeoDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
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

# 训练与模型规模 - 针对小数据集调整
BATCH_SIZE    = 8  # 减小批次大小
NUM_EPOCHS    = 300  # 适当增加训练轮次
LEARNING_RATE = 5e-4  # 减小学习率
HIDDEN_DIM    = 64  # 减小模型规模防止过拟合
DROPOUT_RATE  = 0.3  # 增加dropout
WEIGHT_DECAY  = 5e-4  # 增加权重衰减

# 特征精简相关
NODE_BOTTLENECK_DIM = 8
L1_GATE_COEF        = 8e-4  # 增加L1正则化强度

# 分类任务配置 - 核心修改：固定阈值为-6，禁用自动计算
USE_MEDIAN_THRESHOLD = False  # 无需使用中位数
CLASSIFICATION_THRESHOLD = -6  # 固定分类阈值为-6（关键修改）
OPTIMIZE_FOR          = "f1"  # 'f1' | 'accuracy' | 'mcc'
EARLY_STOP_PATIENCE   = 30    # 增加早停耐心
EARLY_STOP_MIN_DELTA  = 1e-4

# 交叉验证配置
K_FOLDS = 5  # 5折交叉验证

# ------------- 工具函数 -------------
def _np_json_default(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _get_val_score(val_metrics: Dict[str, Dict[str, float]], val_loss: float) -> float:
    if OPTIMIZE_FOR == "f1":
        try:
            return float(val_metrics["classification"]["f1"])
        except Exception:
            return -float(val_loss)
    elif OPTIMIZE_FOR == "accuracy":
        try:
            return float(val_metrics["classification"]["accuracy"])
        except Exception:
            return -float(val_loss)
    elif OPTIMIZE_FOR == "mcc":
        try:
            return float(val_metrics["classification"]["mcc"])
        except Exception:
            return -float(val_loss)
    return -float(val_loss)

# ------------- 1. 数据集（平衡分类） -------------
class GraphDataset(GeoDataset):
    def __init__(self, root_dir: str, threshold: Optional[float] = None, 
                 use_median_threshold: bool = USE_MEDIAN_THRESHOLD, is_train: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.use_median_threshold = use_median_threshold
        # 核心修改：强制使用传入的固定阈值（此处为-6），不自动计算
        self.threshold = threshold if threshold is not None else CLASSIFICATION_THRESHOLD
        if self.threshold is None:
            raise ValueError("分类阈值必须指定（当前配置为-6）")
        
        self.file_list: List[str] = []
        self.compositions: List[str] = []
        self.all_ys = []  # 仅用于统计类别分布，不用于计算阈值
        
        if os.path.isdir(root_dir):
            all_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pt")])
            skipped = 0

            # 遍历文件：筛选有效数据 + 收集y值用于类别统计
            for f in all_files:
                full = os.path.join(root_dir, f)
                try:
                    data_obj: Data = torch.load(full, map_location="cpu")
                except Exception as e:
                    print(f"[警告] 无法读取 {f}: {e}，已跳过。")
                    skipped += 1
                    continue

                # 校验数据有效性（必须有节点特征x和标签y）
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
                    self.all_ys.append(data_obj.y.item())  # 收集y值用于统计类别
                    # 提取材料成分（用于后续追踪）
                    comp = os.path.splitext(f)[0].split('_')[0] if '_' in f else f[:-3]
                    self.compositions.append(comp)
                else:
                    skipped += 1
            
            # 打印固定阈值信息（确认使用-6）
            print(f"[数据集] 使用固定分类阈值: {self.threshold:.4f}")
            
            # 统计类别分布（基于固定阈值-6）
            if self.all_ys:
                class0 = sum(1 for y in self.all_ys if y < self.threshold)  # y < -6 → 类别0
                class1 = len(self.all_ys) - class0  # y ≥ -6 → 类别1
                print(f"[数据集] 类别分布: 类别0={class0} ({class0/len(self.all_ys):.1%}), 类别1={class1} ({class1/len(self.all_ys):.1%})")
            
            if skipped:
                print(f"[数据集] 已过滤 {skipped} 个无效文件，保留 {len(self.file_list)} 个。")
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
            
        # 核心修改：基于固定阈值-6生成分类标签
        y_val = data_obj.y.item()
        data_obj.y_cls = torch.tensor(1.0 if y_val >= self.threshold else 0.0, dtype=torch.float)  # y≥-6→1，否则→0
        data_obj.material_id = os.path.splitext(pt_fname)[0]
        return data_obj
    
    def get_threshold(self) -> float:
        if self.threshold is None:
            raise ValueError("分类阈值未设置")
        return self.threshold

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

# ------------- 3. 分类模型 -------------
class ClassificationCEGNet(nn.Module):
    def __init__(self, node_in_features: int, edge_in_features: int, hidden_dim: int,
                 dropout_rate: float = 0.2, bottleneck_dim: int = NODE_BOTTLENECK_DIM,
                 l1_gate_coef: float = L1_GATE_COEF):
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

        # 分类预测头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Linear(hidden_dim // 2, 1)  # 二分类输出

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

        # 分类输出（sigmoid在损失函数中处理）
        pred_cls = self.classifier(graph_emb).squeeze(-1)

        return {"pred_cls": pred_cls, "gates": gates}

    def get_loss(self, pred: Dict[str, torch.Tensor], data: Data) -> torch.Tensor:
        # 计算类别权重，处理类别不平衡（基于固定阈值-6的分布）
        num_neg = torch.sum(data.y_cls == 0)
        num_pos = torch.sum(data.y_cls == 1)
        
        # 避免除零错误，在目标设备上创建张量
        pos_weight = torch.where(
            num_pos > 0,
            num_neg / num_pos,
            torch.tensor(1.0, device=data.y_cls.device)
        )
        
        # 分类损失（带权重的二元交叉熵）
        cls_loss = F.binary_cross_entropy_with_logits(pred["pred_cls"], data.y_cls.view(-1), pos_weight=pos_weight)
        # L1正则化门控
        gate_loss = self.l1_gate_coef * torch.norm(self.feature_gates, p=1)
        return cls_loss + gate_loss

# ------------- 4. 训练与评估函数 -------------
def train_epoch(model, loader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = model.get_loss(pred, batch)
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 对于CyclicLR等需要按步更新的调度器
        if scheduler is not None and isinstance(scheduler, CyclicLR):
            scheduler.step()
        
        total_loss += loss.item() * batch.num_graphs  # 按样本数加权
        # 记录预测概率和标签
        all_preds.extend(torch.sigmoid(pred["pred_cls"]).cpu().detach().numpy())
        all_labels.extend(batch.y_cls.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(loader.dataset)
    preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]  # 预测概率阈值仍为0.5（与分类标签阈值-6无关）
    metrics = {
        "accuracy": accuracy_score(all_labels, preds_binary),
        "precision": precision_score(all_labels, preds_binary, labels=[0, 1], zero_division=0),
        "recall": recall_score(all_labels, preds_binary, labels=[0, 1], zero_division=0),
        "f1": f1_score(all_labels, preds_binary, labels=[0, 1], zero_division=0),
        "mcc": matthews_corrcoef(all_labels, preds_binary) if len(np.unique(all_labels)) > 1 else 0.0
    }
    
    return avg_loss, {"classification": metrics}

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = model.get_loss(pred, batch)
            total_loss += loss.item() * batch.num_graphs
            
            # 保存预测和真实标签
            all_preds.extend(torch.sigmoid(pred["pred_cls"]).cpu().numpy())
            all_labels.extend(batch.y_cls.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(loader.dataset)
    preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]  # 预测概率阈值为0.5（固定）
    metrics = {
        "accuracy": accuracy_score(all_labels, preds_binary),
        "precision": precision_score(all_labels, preds_binary, labels=[0, 1], zero_division=0),
        "recall": recall_score(all_labels, preds_binary, labels=[0, 1], zero_division=0),
        "f1": f1_score(all_labels, preds_binary, labels=[0, 1], zero_division=0),
        "mcc": matthews_corrcoef(all_labels, preds_binary) if len(np.unique(all_labels)) > 1 else 0.0
    }
    
    return avg_loss, {"classification": metrics}

# 训练单个模型的函数
def train_single_model(train_dataset, val_dataset, node_in_dim, edge_in_dim, device, fold=None):
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = ClassificationCEGNet(
        node_in_features=node_in_dim,
        edge_in_features=edge_in_dim,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    # 优化器和学习率调度器 - 使用AdamW和CyclicLR
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CyclicLR(
        optimizer,
        base_lr=LEARNING_RATE / 10,
        max_lr=LEARNING_RATE,
        step_size_up=20,
        mode='triangular2',
        cycle_momentum=False
    )

    # 训练循环
    best_val_score = -float('inf')
    best_val_f1 = 0.0
    early_stop_counter = 0
    history = defaultdict(list)

    for epoch in range(NUM_EPOCHS):
        # 训练
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, scheduler)
        # 评估
        val_loss, val_metrics = evaluate(model, val_loader, device)
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["classification"]["f1"])
        history["val_f1"].append(val_metrics["classification"]["f1"])
        history["train_accuracy"].append(train_metrics["classification"]["accuracy"])
        history["val_accuracy"].append(val_metrics["classification"]["accuracy"])
        
        # 打印日志
        if (epoch + 1) % 10 == 0:
            print(f"{'Fold ' + str(fold) + ' ' if fold is not None else ''}Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"  训练损失: {train_loss:.4f} | 训练F1: {train_metrics['classification']['f1']:.4f}")
            print(f"  验证损失: {val_loss:.4f} | 验证F1: {val_metrics['classification']['f1']:.4f}")
        
        # 学习率调度（如果是ReduceLROnPlateau）
        val_score = _get_val_score(val_metrics, val_loss)
        
        # 早停逻辑
        if val_score > best_val_score + EARLY_STOP_MIN_DELTA:
            best_val_score = val_score
            best_val_f1 = val_metrics["classification"]["f1"]
            early_stop_counter = 0
            # 保存当前折的最佳模型
            if fold is not None:
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"best_classification_model_fold_{fold}.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_classification_model.pt"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE and EARLY_STOP_PATIENCE > 0:
                if (epoch + 1) % 10 == 0:
                    print(f"  早停触发（{early_stop_counter}轮无提升）")
                break

    return best_val_f1, model

# ------------- 5. 主函数入口 -------------
def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载数据集 - 传入固定阈值-6（与全局配置一致）
    full_dataset = GraphDataset(
        GRAPHS_DIR, 
        use_median_threshold=USE_MEDIAN_THRESHOLD,
        threshold=CLASSIFICATION_THRESHOLD  # 此处为-6
    )
    
    if len(full_dataset) == 0:
        print(f"错误：在 {GRAPHS_DIR} 中未找到有效数据，请检查路径。")
        return
    
    # 检查类别分布是否合理（基于固定阈值-6）
    all_labels = [full_dataset.get(i).y_cls.item() for i in range(len(full_dataset))]
    class_counts = np.bincount(all_labels)
    print(f"总数据集类别分布: 类别0={class_counts[0]}, 类别1={class_counts[1]}")
    
    if len(class_counts) < 2:
        print("错误：数据集中只有一个类别，无法进行分类训练！")
        return

    # 获取特征维度
    sample_data = full_dataset.get(0)
    node_in_dim = sample_data.x.size(1)
    edge_in_dim = sample_data.edge_attr.size(1) if (sample_data.edge_attr is not None and sample_data.edge_attr.numel() > 0) else 0
    print(f"节点特征维度: {node_in_dim}, 边特征维度: {edge_in_dim}")

    # 使用交叉验证训练多个模型并选择最佳
    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_f1_scores = []
    best_f1 = 0.0
    best_model = None

    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(full_dataset)), all_labels)):
        print(f"\n----- 折 {fold + 1}/{K_FOLDS} -----")
        train_dataset = IndexSubset(full_dataset, train_ids)
        val_dataset = IndexSubset(full_dataset, val_ids)
        
        # 训练当前折的模型
        fold_val_f1, model = train_single_model(
            train_dataset, val_dataset, node_in_dim, edge_in_dim, device, fold
        )
        
        fold_f1_scores.append(fold_val_f1)
        print(f"折 {fold + 1} 最佳验证F1: {fold_val_f1:.4f}")
        
        # 跟踪所有折中表现最好的模型
        if fold_val_f1 > best_f1:
            best_f1 = fold_val_f1
            best_model = model

    # 保存最佳模型
    torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, "best_classification_model.pt"))
    print(f"\n交叉验证平均F1分数: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    print(f"最佳模型F1分数: {best_f1:.4f}")

    # 保存训练历史和固定阈值-6
    history = {"cv_f1_scores": fold_f1_scores, "mean_cv_f1": np.mean(fold_f1_scores), 
               "best_f1": best_f1, "threshold": full_dataset.get_threshold()}  # threshold会保存为-6
    with open(os.path.join(RESULTS_DIR, "classification_history.json"), "w") as f:
        json.dump(history, f, default=_np_json_default, indent=2)
    print(f"训练完成，历史记录保存至 {RESULTS_DIR}/classification_history.json")
    print(f"使用的分类阈值为: {full_dataset.get_threshold():.4f}（固定值）")
    
    # 输出最佳模型的F1分数
    print(f"最佳模型的F1分数: {best_f1:.4f}")

if __name__ == "__main__":
    main()