import os
import torch
import zipfile
from tempfile import TemporaryDirectory
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN

# 1. 配置路径 - 修正路径，去掉可能多余的"mattergen/"前缀
# 根据错误信息，当前工作目录已经是mattergen，所以路径应该从generated/开始
zip_file_paths = [
    "generated/LiPS_ehull_0p0/generated_crystals_cif.zip",
    "generated/LiPS_ehull_0p05_gs_7p0/generated_crystals_cif.zip",
    "generated/LiPS_ehull_0p05_gs_7p5/generated_crystals_cif.zip",
    "generated/LiS_ehull_0p1_gs_7p5/generated_crystals_cif.zip",
    "generated/LiS_ehull_0p02_gs_7p5/generated_crystals_cif.zip",
    "generated/LiS_ehull_0p05_gs_7p5/generated_crystals_cif.zip"
]

base_save_dir = "transformed_data/"  # 转换后的图数据根保存目录
os.makedirs(base_save_dir, exist_ok=True)
print(f"根保存目录已创建或存在: {os.path.abspath(base_save_dir)}")

# 2. 特征提取函数（需与GNN训练时一致）
def get_node_features(atom):
    """提取节点特征（原子序数、电负性、原子半径）"""
    electronegativity = atom.X  # 电负性
    atomic_radius = atom.atomic_radius if atom.atomic_radius is not None else 0.0  # 原子半径，处理 None
    return torch.tensor([
        atom.Z,                  # 原子序数
        electronegativity,       # 电负性
        atomic_radius            # 原子半径
    ], dtype=torch.float)

def get_edge_features(structure, edge_index):
    """提取边特征（键长）"""
    u, v = edge_index
    distances = []
    for i, j in zip(u, v):
        dist = structure.get_distance(i, j)  # 原子i和j的距离
        distances.append(dist)
    return torch.tensor(distances, dtype=torch.float).unsqueeze(1)  # 形状：[num_edges, 1]

# 3. 循环处理每个zip文件
for zip_path in zip_file_paths:
    # 获取绝对路径用于调试
    abs_zip_path = os.path.abspath(zip_path)
    
    # 检查文件是否存在
    if not os.path.exists(abs_zip_path):
        print(f"\n警告：文件不存在 - {abs_zip_path}")
        continue
    
    # 为每个zip文件创建独立的子目录，避免文件冲突
    # 从zip路径中提取目录名作为子目录
    zip_dir_name = os.path.basename(os.path.dirname(zip_path))
    graph_save_dir = os.path.join(base_save_dir, zip_dir_name)
    os.makedirs(graph_save_dir, exist_ok=True)
    print(f"\n处理文件: {abs_zip_path}")
    print(f"输出目录: {os.path.abspath(graph_save_dir)}")
    
    try:
        # 从ZIP文件读取并转换结构为图数据
        with zipfile.ZipFile(abs_zip_path, 'r') as zip_ref:
            # 获取ZIP中的所有CIF文件
            cif_files = [f for f in zip_ref.namelist() if f.endswith('.cif')]
            print(f"发现{len(cif_files)}个CIF文件，开始转换...")
            
            # 使用临时目录解压文件
            with TemporaryDirectory() as temp_dir:
                # 解压所有CIF文件到临时目录
                for idx, cif_file in enumerate(cif_files):
                    zip_ref.extract(cif_file, temp_dir)
                    
                    # 构建临时文件路径
                    temp_file_path = os.path.join(temp_dir, cif_file)
                    
                    try:
                        # 加载结构
                        struct = Structure.from_file(temp_file_path)
                        
                        # 构建结构图（用CrystalNN找近邻原子作为边）
                        struct_graph = StructureGraph.from_local_env_strategy(struct, CrystalNN())
                        edges = struct_graph.graph.edges()  # 边索引 (u, v)
                        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()  # 形状：[2, num_edges]
                        
                        # 提取节点特征（修正为每个站点）
                        node_feats = torch.stack([get_node_features(site.specie) for site in struct.sites])  # 形状：[num_atoms, node_dim]
                        
                        # 提取边特征
                        edge_feats = get_edge_features(struct, edge_index)  # 形状：[num_edges, edge_dim]
                        
                        # 构建PyG Data对象
                        graph_data = Data(
                            x=node_feats,
                            edge_index=edge_index,
                            edge_attr=edge_feats,
                            batch=None  # 批量处理时由DataLoader自动添加
                        )
                        
                        # 构建保存路径
                        base_name = os.path.basename(cif_file)
                        save_path = os.path.join(graph_save_dir, f"{base_name.replace('.cif', '.pt')}")
                        torch.save(graph_data, save_path)
                        
                        # 每处理10个文件打印一次进度
                        if (idx + 1) % 10 == 0:
                            print(f"已处理 {idx + 1}/{len(cif_files)} 个文件")
                            
                    except Exception as e:
                        print(f"处理文件 {cif_file} 时出错: {str(e)}")
                        continue
        print(f"完成处理 {abs_zip_path}，共处理 {len(cif_files)} 个文件")
    except Exception as e:
        print(f"处理ZIP文件 {abs_zip_path} 时出错: {str(e)}")
        continue

print("\n所有ZIP文件处理完毕")
    