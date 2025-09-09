import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# --------------------------
# 1. 解决 Qt Wayland 插件问题（Linux 必加，已验证有效）
# --------------------------
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 2. 数据路径（与你的 MatterGen 目录完全一致）
GRAPHS_DIR = "processed_data/graphs"
RESULTS_DIR = "results"

def load_conductivity_values(graphs_dir: str) -> list[float]:
    """Load valid conductivity values (y-values) from .pt files"""
    if not os.path.isdir(graphs_dir):
        raise ValueError(f"Directory not found: {graphs_dir}")
    
    pt_files = [f for f in os.listdir(graphs_dir) if f.endswith(".pt")]
    if not pt_files:
        raise ValueError(f"No .pt files found in {graphs_dir}")
    
    conductivity_values = []
    skipped_files = []
    
    for f in pt_files:
        full_path = os.path.join(graphs_dir, f)
        try:
            # 加载数据（仅用CPU，无需GPU）
            data_obj: Data = torch.load(full_path, map_location="cpu")
            # 验证电导率值有效性
            if hasattr(data_obj, "y") and isinstance(data_obj.y, torch.Tensor) and data_obj.y.numel() > 0:
                conductivity_values.append(data_obj.y.item())
            else:
                skipped_files.append(f)
        except Exception as e:
            print(f"Warning: Failed to read {f}: {str(e)[:50]}...")  # 截断长错误信息
            skipped_files.append(f)
    
    if skipped_files:
        print(f"Skipped {len(skipped_files)} invalid file(s) (missing y or corrupted)")
    return conductivity_values

def plot_conductivity_scatter(conductivity_values: list[float], save_path: str):
    """Plot scatter plot with English labels (no font dependency)"""
    # 图形配置（适合学术展示）
    plt.figure(figsize=(10, 6))  # 宽10英寸，高6英寸，比例协调
    sample_indices = range(len(conductivity_values))
    
    # 绘制散点图（优化视觉效果）
    scatter = plt.scatter(
        sample_indices,
        conductivity_values,
        alpha=0.6,        # 透明度：解决点重叠问题
        s=30,             # 标记大小：平衡清晰度与美观度
        c="navy",         # 标记颜色：深蓝色，学术图表常用
        edgecolors="white",
        linewidth=0.5     # 白色边框：增强点的轮廓感
    )
    
    # 英文标签（无字体问题，直接显示）
    plt.title("Conductivity Value Distribution in Dataset", fontsize=15, pad=20)
    plt.xlabel("Sample Index", fontsize=12, labelpad=10)
    plt.ylabel("Conductivity Value (log-transformed)", fontsize=12, labelpad=10)  # 注明对数转换，避免误解
    
    # 网格线（辅助读数，不干扰数据）
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # 自动调整布局：避免标签截断
    plt.tight_layout()
    
    # 保存图表（300 DPI，学术印刷级分辨率）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")

def main():
    try:
        # 加载数据
        print("Loading conductivity data...")
        conductivity_values = load_conductivity_values(GRAPHS_DIR)
        
        if not conductivity_values:
            print("Error: No valid conductivity data found")
            return
        
        # 确认数据加载结果
        print(f"Successfully loaded {len(conductivity_values)} conductivity values")
        
        # 生成并保存图表
        plot_conductivity_scatter(
            conductivity_values,
            save_path=os.path.join(RESULTS_DIR, "conductivity_scatter_final.png")
        )
        
        # 输出统计信息（学术报告常用）
        print("\n=== Conductivity Statistics ===")
        print(f"Mean:        {np.mean(conductivity_values):.4f}")
        print(f"Median:      {np.median(conductivity_values):.4f}")
        print(f"Minimum:     {np.min(conductivity_values):.4f}")
        print(f"Maximum:     {np.max(conductivity_values):.4f}")
        print(f"Standard Dev: {np.std(conductivity_values):.4f}")
        
    except Exception as e:
        print(f"Execution error: {e}")

if __name__ == "__main__":
    import torch  # 延迟导入：减少启动时间
    main()