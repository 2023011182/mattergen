import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import warnings

# 禁用警告输出
warnings.filterwarnings("ignore")

from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset as GeoDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------- 全局配置 ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

GRAPHS_DIR    = "processed_data/graphs"  # 图数据存储目录
MODELS_DIR    = "models"                 # 模型保存目录
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 5e-4  # 降低学习率以提高稳定性
HIDDEN_DIM    = 256   # 增加隐藏层维度提升模型容量
OUTPUT_DIM    = 1     # 回归任务
DROPOUT_RATE  = 0.3
PATIENCE      = 10    # 早停耐心值

# ------------- 1. 数据集处理 -------------
class GraphDataset(GeoDataset):
    """
    从指定目录加载图数据，支持自定义标签字段，确保数据格式统一
    """
    def __init__(self, root_dir, label_fields=['y', 'conductivity', 'label']):
        super().__init__()
        self.root_dir = root_dir
        self.label_fields = label_fields  # 支持的标签字段列表
        self.file_list = []
        self.valid_files = []  # 有效文件列表
        self.label_scaler = None  # 标签标准化器
        
        if os.path.isdir(root_dir):
            all_files = sorted(os.listdir(root_dir))
            for f in all_files:
                if f.endswith(".pt"):
                    self.file_list.append(f)
            
            # 验证并清理文件
            self._validate_and_clean_files()
            # 收集所有标签用于标准化
            self._fit_label_scaler()
    
    def _validate_and_clean_files(self):
        """验证文件内容，统一标签为'y'，清理无效文件"""
        invalid_count = 0
        for f in self.file_list:
            try:
                full_path = os.path.join(self.root_dir, f)
                data_obj = torch.load(full_path, map_location="cpu")
                
                # 检查必要属性
                if not hasattr(data_obj, 'x') or not isinstance(data_obj.x, torch.Tensor) or data_obj.x.numel() == 0:
                    raise ValueError("缺少有效的节点特征 'x' 或特征为空")
                if not hasattr(data_obj, 'edge_index') or not isinstance(data_obj.edge_index, torch.Tensor) or data_obj.edge_index.numel() == 0:
                    raise ValueError("缺少有效的边索引 'edge_index' 或边索引为空")
                
                # 查找并统一标签字段
                found_label = False
                for label_field in self.label_fields:
                    if hasattr(data_obj, label_field) and isinstance(getattr(data_obj, label_field), torch.Tensor) and getattr(data_obj, label_field).numel() > 0:
                        # 将找到的标签重命名为'y'
                        setattr(data_obj, 'y', getattr(data_obj, label_field).float())
                        # 删除原标签字段（如果不是'y'）
                        if label_field != 'y':
                            delattr(data_obj, label_field)
                        # 确保没有其他冲突属性
                        if hasattr(data_obj, 'conductivity') and 'conductivity' not in self.label_fields:
                            del data_obj.conductivity
                        torch.save(data_obj, full_path)
                        self.valid_files.append(f)
                        found_label = True
                        break
                
                if not found_label:
                    # 显示文件所有属性帮助调试
                    attrs = [attr for attr in dir(data_obj) if not attr.startswith('_') and not callable(getattr(data_obj, attr))]
                    raise ValueError(f"未找到有效标签字段 {self.label_fields}，文件包含属性: {attrs}")
                    
            except Exception:
                invalid_count += 1
        
        # 只输出关键信息，不输出警告
        print(f"[数据集] 共找到 {len(self.file_list)} 个文件，有效文件 {len(self.valid_files)} 个，无效文件 {invalid_count} 个")
    
    def _fit_label_scaler(self):
        """拟合标签标准化器"""
        if len(self.valid_files) == 0:
            return
            
        labels = []
        for idx in range(min(100, len(self.valid_files))):  # 采样拟合，加快速度
            data = self.get(idx)
            labels.append(data.y.cpu().numpy().reshape(-1))
        
        labels = np.concatenate(labels)
        self.label_scaler = StandardScaler()
        self.label_scaler.fit(labels.reshape(-1, 1))
    
    def len(self):
        return len(self.valid_files)
    
    def get(self, idx):
        pt_fname = self.valid_files[idx]
        full_path = os.path.join(self.root_dir, pt_fname)
        data_obj = torch.load(full_path)
        
        # 确保数据格式正确
        if not hasattr(data_obj, 'y'):
            raise ValueError(f"文件 {pt_fname} 缺少标签字段 'y'")
        
        # 标准化标签（可选）
        if self.label_scaler is not None:
            y_np = data_obj.y.cpu().numpy().reshape(-1, 1)
            y_scaled = self.label_scaler.transform(y_np)
            data_obj.y = torch.tensor(y_scaled, dtype=torch.float32)
        
        return data_obj

# ------------- 2. 消息传递层 -------------
class CEGMessagePassing(MessagePassing):
    """
    自定义消息传递层，整合节点特征和边特征
    """
    def __init__(self, node_in_channels, edge_in_channels, out_channels):
        super().__init__(aggr='add')  # 使用加法聚合
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels

        # 节点特征处理
        self.sender_node_lin = nn.Linear(node_in_channels, out_channels)

        # 边特征处理（如果存在）
        if edge_in_channels > 0:
            self.edge_lin = nn.Linear(edge_in_channels, out_channels)
        else:
            self.edge_lin = None
        
        # 消息处理MLP
        msg_input_dim = out_channels + (out_channels if self.edge_lin else 0)
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, out_channels),
            nn.BatchNorm1d(out_channels),  # 增加批归一化
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )

        # 更新函数MLP
        self.update_x_lin = nn.Linear(node_in_channels, out_channels)
        self.update_mlp = nn.Sequential(
            nn.Linear(out_channels + out_channels, out_channels),
            nn.BatchNorm1d(out_channels),  # 增加批归一化
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j是源节点特征
        node_part = self.sender_node_lin(x_j)
        if self.edge_lin is not None and edge_attr is not None and edge_attr.size(1) > 0:
            edge_part = self.edge_lin(edge_attr)
            cat = torch.cat([node_part, edge_part], dim=-1)
        else:
            cat = node_part
        return self.message_mlp(cat)

    def update(self, aggr_msg, x):
        # 结合聚合消息和当前节点特征
        old_x = self.update_x_lin(x)
        cat = torch.cat([old_x, aggr_msg], dim=-1)
        return self.update_mlp(cat)

# ------------- 3. 模型定义 -------------
class CEGNet(nn.Module):
    """
    基于自定义消息传递的图神经网络
    """
    def __init__(self, node_in_features, edge_in_features, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.node_in_features = node_in_features
        self.edge_in_features = edge_in_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 图卷积层
        self.conv1 = CEGMessagePassing(node_in_features, edge_in_features, hidden_dim)
        self.conv2 = CEGMessagePassing(hidden_dim, edge_in_features, hidden_dim)
        self.conv3 = CEGMessagePassing(hidden_dim, edge_in_features, hidden_dim)

        # 全局池化
        self.pool = global_mean_pool

        # 输出层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc3 = nn.Linear(hidden_dim//4, output_dim)  # 增加一层全连接

        # 正则化
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim//2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim//4)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 图卷积层
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)

        # 全局池化
        x = self.pool(x, batch)

        # 输出层
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        out = self.fc3(x)  # shape=[batch_size,1]
        return out.squeeze(-1)  # -> [batch_size]

# ------------- 4. 训练与评估函数 -------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练单个epoch"""
    model.train()
    total_loss = 0
    total_graphs = 0
    preds_all = []
    tgts_all = []
    
    for batch_idx, batch_data in enumerate(loader):
        try:
            if batch_data.x.size(0) == 0:
                continue
                
            batch_data = batch_data.to(device)
            
            # 确保没有冲突属性
            if hasattr(batch_data, 'conductivity'):
                del batch_data.conductivity

            optimizer.zero_grad()
            out = model(batch_data)
            y = batch_data.y.view(-1)  # 调整标签形状
            
            loss = criterion(out, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch_data.num_graphs
            total_graphs += batch_data.num_graphs
            
            # 收集预测和目标值用于计算R²
            preds_all.append(out.detach().cpu().numpy().flatten())
            tgts_all.append(y.cpu().numpy().flatten())
            
        except Exception:
            continue

    # 计算训练集R²
    if total_graphs > 0 and len(preds_all) > 0:
        preds_all = np.concatenate(preds_all, axis=0)
        tgts_all = np.concatenate(tgts_all, axis=0)
        r2 = r2_score(tgts_all, preds_all)
    else:
        r2 = float('nan')
        
    return total_loss / total_graphs if total_graphs > 0 else 0, r2

def evaluate_model(model, loader, criterion, device, dataset=None):
    """评估模型性能，返回损失、预测值、目标值和R²"""
    model.eval()
    total_loss = 0
    total_graphs = 0
    preds_all = []
    tgts_all = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            try:
                if batch_data.x.size(0) == 0:
                    continue
                    
                batch_data = batch_data.to(device)
                
                # 确保没有冲突属性
                if hasattr(batch_data, 'conductivity'):
                    del batch_data.conductivity

                out = model(batch_data)
                y = batch_data.y.view(-1)
                loss = criterion(out, y)

                total_loss += loss.item() * batch_data.num_graphs
                total_graphs += batch_data.num_graphs

                # 保存预测和目标值（如果有标准化器，需要反归一化）
                if dataset and dataset.label_scaler:
                    preds_np = dataset.label_scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).flatten()
                    tgts_np = dataset.label_scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1)).flatten()
                else:
                    preds_np = out.cpu().numpy().flatten()
                    tgts_np = y.cpu().numpy().flatten()
                    
                preds_all.append(preds_np)
                tgts_all.append(tgts_np)
                
            except Exception:
                continue
    
    if total_graphs == 0:
        return float('inf'), None, None, float('nan')
        
    avg_loss = total_loss / total_graphs
    preds_all = np.concatenate(preds_all, axis=0)
    tgts_all = np.concatenate(tgts_all, axis=0)
    
    # 计算R²
    try:
        r2 = r2_score(tgts_all, preds_all)
    except:
        r2 = float('nan')
    
    return avg_loss, preds_all, tgts_all, r2

# ------------- 5. 主函数 -------------
def main():
    # 创建模型保存目录
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 加载数据集
    if not os.path.exists(GRAPHS_DIR):
        print(f"[错误] 找不到图数据目录: {GRAPHS_DIR}")
        return
        
    # 可以根据实际数据中的标签字段调整label_fields
    dataset = GraphDataset(GRAPHS_DIR, label_fields=['y', 'conductivity', 'label'])
    if dataset.len() == 0:
        print(f"[错误] 没有有效的图数据文件，无法训练")
        return

    # 数据集划分
    idxs = list(range(dataset.len()))
    train_idxs, test_idxs = train_test_split(idxs, test_size=0.2, random_state=SEED)
    train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.2, random_state=SEED)

    train_ds = Subset(dataset, train_idxs)
    val_ds = Subset(dataset, val_idxs)
    test_ds = Subset(dataset, test_idxs)

    # 创建数据加载器
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 确定特征维度
    try:
        sample_data = dataset.get(0)
        node_in_dim = sample_data.x.size(1) if sample_data.x.numel() > 0 else 0
        edge_in_dim = sample_data.edge_attr.size(1) if (hasattr(sample_data, 'edge_attr') and 
                                                       sample_data.edge_attr.numel() > 0) else 0
    except Exception as e:
        print(f"[错误] 无法确定特征维度: {str(e)}")
        return

    if node_in_dim == 0:
        print("[错误] 节点特征维度为0，无法训练")
        return
    if edge_in_dim == 0:
        edge_in_dim = 1  # 边特征维度为0时设置默认值

    print(f"[模型配置] 节点特征维度: {node_in_dim}, 边特征维度: {edge_in_dim}, 隐藏层维度: {HIDDEN_DIM}")

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[训练设备] 使用 {'GPU' if device.type == 'cuda' else 'CPU'} 进行训练")

    # 初始化模型、优化器和损失函数
    model = CEGNet(node_in_dim, edge_in_dim, HIDDEN_DIM, OUTPUT_DIM, DROPOUT_RATE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # 增加权重衰减
    criterion = nn.MSELoss()

    # 模型保存路径
    best_path = os.path.join(MODELS_DIR, "best_cegnet_model.pt")

    # 训练过程
    best_val_r2 = -float('inf')  # 改为用R²判断最佳模型
    best_epoch = -1
    no_improve_epochs = 0  # 早停计数器

    print("\n开始训练...")
    print(f"[表头] Epoch | 训练MSE | 训练R² | 验证MSE | 验证R²")
    for ep in range(NUM_EPOCHS):
        # 训练并获取训练R²
        tr_loss, tr_r2 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # 验证并获取验证R²
        val_loss, _, _, val_r2 = evaluate_model(model, val_loader, criterion, device, dataset)

        # 打印训练信息，包含R²
        print(f"[Epoch {ep+1}/{NUM_EPOCHS}] {tr_loss:.4f} | {tr_r2:.4f} | {val_loss:.4f} | {val_r2:.4f}")

        # 保存最佳模型（基于验证R²）
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
            no_improve_epochs = 0  # 重置早停计数器
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print(f"[早停] 在Epoch {ep+1} 触发早停，最佳验证R²在Epoch {best_epoch+1}")
                break
    
    print(f"\n训练完成. 最佳验证R²: {best_val_r2:.4f} (Epoch {best_epoch+1})")

    # 测试最佳模型
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, preds, tgts, test_r2 = evaluate_model(model, test_loader, criterion, device, dataset)
        print(f"\n[测试结果] MSE: {test_loss:.4f}")
        
        if preds is not None and tgts is not None and len(preds) > 0 and len(tgts) > 0:
            mae = mean_absolute_error(tgts, preds)
            print(f"[测试结果] MAE: {mae:.4f}, R²: {test_r2:.4f}")
        else:
            print("[测试结果] 没有有效的预测结果")
    else:
        print("[错误] 未找到最佳模型文件，无法进行测试")

if __name__ == "__main__":
    main()