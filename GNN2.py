import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data, Dataset as GeoDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- 全局配置 ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

GRAPHS_DIR    = "processed_data/graphs"  # 直接读取这里的 .pt
MODELS_DIR    = "models"                 # 保存/加载模型文件
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM    = 128
OUTPUT_DIM    = 1  # 回归
DROPOUT_RATE  = 0.3

# ------------- 1. Dataset：遍历 .pt 文件 -------------
class GraphDataset(GeoDataset):
    """
    直接从 `processed_data/graphs/` 里的 .pt 文件加载图数据。
    """
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.file_list = []
        if os.path.isdir(root_dir):
            all_files = sorted(os.listdir(root_dir))
            for f in all_files:
                if f.endswith(".pt"):
                    self.file_list.append(f)
        else:
            print(f"[警告] 目录不存在: {root_dir}")
    
    def len(self):
        return len(self.file_list)
    
    def get(self, idx):
        pt_fname = self.file_list[idx]
        full_path = os.path.join(self.root_dir, pt_fname)
        data_obj = torch.load(full_path)  # 直接加载 Data 对象
        return data_obj

# ------------- 2. CEGMessagePassing -------------
class CEGMessagePassing(MessagePassing):
    """
    与之前相同的三步流程: message, aggregate, update
    sender_node_lin/edge_lin -> message_mlp -> update_x_lin -> update_mlp
    """
    def __init__(self, node_in_channels, edge_in_channels, out_channels):
        super().__init__(aggr='add')  # "add"聚合所有邻居消息
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels

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

        # update MLP：结合旧节点特征(投影后)与聚合消息
        self.update_x_lin = nn.Linear(node_in_channels, out_channels)
        self.update_mlp = nn.Sequential(
            nn.Linear(out_channels + out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # 发送方节点特征
        node_part = self.sender_node_lin(x_j)
        if self.edge_lin is not None and edge_attr is not None and edge_attr.size(1)>0:
            edge_part = self.edge_lin(edge_attr)
            cat = torch.cat([node_part, edge_part], dim=-1)
        else:
            cat = node_part
        return self.message_mlp(cat)

    def update(self, aggr_msg, x):
        old_x = self.update_x_lin(x)
        cat = torch.cat([old_x, aggr_msg], dim=-1)
        return self.update_mlp(cat)

# ------------- 3. CEGNet -------------
class CEGNet(nn.Module):
    """
    三层CEGMessagePassing -> global_mean_pool -> MLP
    与之前保持一致
    """
    def __init__(self, node_in_features, edge_in_features, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.node_in_features = node_in_features
        self.edge_in_features = edge_in_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.conv1 = CEGMessagePassing(node_in_features, edge_in_features, hidden_dim)
        self.conv2 = CEGMessagePassing(hidden_dim, edge_in_features, hidden_dim)
        self.conv3 = CEGMessagePassing(hidden_dim, edge_in_features, hidden_dim)

        self.pool = global_mean_pool

        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)  # shape=[batch_size,1]
        return out.squeeze(-1) # -> [batch_size]

# ------------- 4. 训练与评估 -------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss=0
    total_graphs=0
    for batch_data in loader:
        if batch_data.x.size(0)==0:
            continue
        batch_data = batch_data.to(device)

        optimizer.zero_grad()
        out = model(batch_data)
        y   = batch_data.y.view(-1, 1)  # [batch_size,1], 以兼容 MSE
        loss= criterion(out.view(-1,1), y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss+=loss.item()*batch_data.num_graphs
        total_graphs+=batch_data.num_graphs

    return total_loss/total_graphs if total_graphs>0 else 0

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss=0
    total_graphs=0
    preds_all=[]
    tgts_all =[]
    with torch.no_grad():
        for batch_data in loader:
            if batch_data.x.size(0)==0:
                continue
            batch_data=batch_data.to(device)
            out=model(batch_data)
            y=batch_data.y.view(-1,1)
            loss=criterion(out.view(-1,1), y)

            total_loss+=loss.item()*batch_data.num_graphs
            total_graphs+=batch_data.num_graphs

            preds_all.append(out.cpu().numpy())
            tgts_all.append(batch_data.y.cpu().numpy())
    
    if total_graphs==0:
        return float('inf'), None, None
    preds_all=np.concatenate(preds_all, axis=0).flatten()
    tgts_all =np.concatenate(tgts_all, axis=0).flatten()
    return (total_loss/total_graphs), preds_all, tgts_all

# ------------- 5. main -------------
def main():
    if not os.path.exists(GRAPHS_DIR):
        print(f"[错误] 找不到 {GRAPHS_DIR}")
        return
    dataset=GraphDataset(GRAPHS_DIR)
    if dataset.len()==0:
        print(f"[错误] {GRAPHS_DIR} 下无 .pt 文件")
        return

    # 切分
    idxs=list(range(dataset.len()))
    train_idxs, test_idxs= train_test_split(idxs, test_size=0.2, random_state=SEED)
    train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.2, random_state=SEED)

    from torch.utils.data import DataLoader as BaseDataLoader
    train_ds=Subset(dataset, train_idxs)
    val_ds  =Subset(dataset, val_idxs)
    test_ds =Subset(dataset, test_idxs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # 确定节点/边特征维度
    sample_batch= next(iter(train_loader))
    node_in_dim = sample_batch.x.size(1) if sample_batch.x.size(0) else 0
    edge_in_dim = sample_batch.edge_attr.size(1) if sample_batch.edge_attr.size(0) else 0
    if node_in_dim==0:
        print("[错误] 节点特征维度=0,无法训练.")
        return
    if edge_in_dim==0:
        edge_in_dim=1

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=CEGNet(node_in_dim, edge_in_dim, HIDDEN_DIM, OUTPUT_DIM, DROPOUT_RATE).to(device)
    optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion=nn.MSELoss()

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    best_path= os.path.join(MODELS_DIR, "best_cegnet_model.pt")

    best_val=float('inf')
    best_epoch=-1

    for ep in range(NUM_EPOCHS):
        tr_loss= train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _= evaluate_model(model, val_loader, criterion, device)

        if val_loss<best_val:
            best_val=val_loss
            best_epoch=ep
            torch.save(model.state_dict(), best_path)
        
        print(f"[Epoch {ep+1}/{NUM_EPOCHS}] trainMSE={tr_loss:.4f}, valMSE={val_loss:.4f}")
    
    print(f"\n训练完成. 最佳验证集MSE={best_val:.4f} (Epoch {best_epoch+1})")

    # 测试
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, preds, tgts = evaluate_model(model, test_loader, criterion, device)
        print(f"[Test] MSE={test_loss:.4f}")
        if preds is not None:
            mae=mean_absolute_error(tgts,preds)
            try:
                r2=r2_score(tgts,preds)
            except:
                r2=float('nan')
            print(f"[Test] MAE={mae:.4f}, R²={r2:.4f}")
    else:
        print("[警告] 未找到best_cegnet_model.pt，跳过测试.")

if __name__=="__main__":
    main()