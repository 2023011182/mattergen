#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LithiumVision 晶体结构三维可视化工具
专注于锂离子超导体候选材料的高质量三维结构可视化。
支持交互式展示和静态图像导出。
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import zipfile
import tempfile
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import colormaps

try:
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifParser, CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import JmolNN, CrystalNN, BrunnerNN_real
    from pymatgen.vis.structure_vtk import StructureVis
except ImportError:
    logging.error("请安装pymatgen: pip install pymatgen")
    sys.exit(1)

# 尝试导入交互式可视化模块
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("未安装plotly，无法生成交互式可视化。建议安装: pip install plotly")

try:
    import nglview
    import ase
    from ase.io import write as ase_write
    from ase.io import read as ase_read
    NGLVIEW_AVAILABLE = True
except ImportError:
    NGLVIEW_AVAILABLE = False
    logging.warning("未安装nglview或ase，无法生成NGLView交互式可视化。建议安装: pip install nglview ase")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('visualize_structures')

# 为常见元素定义精美的颜色映射
ELEMENT_COLORS = {
    'H': '#FFFFFF', 'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5', 
    'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050', 
    'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Si': '#F0C8A0', 
    'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F', 'K': '#8F40D4', 
    'Ca': '#3DFF00', 'Ti': '#BFC2C7', 'Cr': '#8A99C7', 'Mn': '#9C7AC7', 
    'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033', 
    'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F', 'As': '#BD80E3', 
    'Se': '#FFA100', 'Br': '#A62929', 'Rb': '#702EB0', 'Sr': '#00FF00', 
    'Y': '#94FFFF', 'Zr': '#94E0E0', 'Nb': '#73C2C9', 'Mo': '#54B5B5', 
    'Tc': '#3B9E9E', 'Ru': '#248F8F', 'Rh': '#0A7D8C', 'Pd': '#006985', 
    'Ag': '#C0C0C0', 'Cd': '#FFD98F', 'In': '#A67573', 'Sn': '#668080', 
    'Sb': '#9E63B5', 'Te': '#D47A00', 'I': '#940094', 'Cs': '#57178F', 
    'Ba': '#00C900', 'La': '#70D4FF', 'Ce': '#FFFFC7', 'Pr': '#D9FFC7', 
    'Nd': '#C7FFC7', 'Pm': '#A3FFC7', 'Sm': '#8FFFC7', 'Eu': '#61FFC7', 
    'Gd': '#45FFC7', 'Tb': '#30FFC7', 'Dy': '#1FFFC7', 'Ho': '#00FF9C', 
    'Er': '#00E675', 'Tm': '#00D452', 'Yb': '#00BF38', 'Lu': '#00AB24', 
    'Hf': '#4DC2FF', 'Ta': '#4DA6FF', 'W': '#2194D6', 'Re': '#267DAB', 
    'Os': '#266696', 'Ir': '#175487', 'Pt': '#D0D0E0', 'Au': '#FFD123', 
    'Hg': '#B8B8D0', 'Tl': '#A6544D', 'Pb': '#575961', 'Bi': '#9E4FB5'
}

# 为其他元素定义默认颜色
DEFAULT_ELEMENT_COLOR = '#CCCCCC'

# 元素的相对大小 (原子半径的相对比例)
ELEMENT_RADII = {
    'H': 0.32, 'Li': 1.45, 'Be': 1.05, 'B': 0.85, 'C': 0.7, 'N': 0.65, 'O': 0.6, 'F': 0.5,
    'Na': 1.8, 'Mg': 1.5, 'Al': 1.25, 'Si': 1.1, 'P': 1.0, 'S': 1.0, 'Cl': 1.0, 'K': 2.2,
    'Ca': 1.8, 'Ti': 1.4, 'Cr': 1.4, 'Mn': 1.4, 'Fe': 1.4, 'Co': 1.35, 'Ni': 1.35, 'Cu': 1.35,
    'Zn': 1.35, 'Ga': 1.3, 'Ge': 1.25, 'As': 1.15, 'Se': 1.15, 'Br': 1.15, 'Rb': 2.35,
    'Sr': 2.0, 'Y': 1.8, 'Zr': 1.55, 'Nb': 1.45, 'Mo': 1.45, 'Tc': 1.35, 'Ru': 1.3,
    'Rh': 1.35, 'Pd': 1.4, 'Ag': 1.6, 'Cd': 1.55, 'In': 1.55, 'Sn': 1.45, 'Sb': 1.45,
    'Te': 1.4, 'I': 1.4, 'Cs': 2.6, 'Ba': 2.15, 'La': 1.95, 'Ce': 1.85
}

DEFAULT_RADIUS = 1.2 

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成锂离子超导体候选材料的高质量三维可视化')
    parser.add_argument('--candidates', type=str, required=True, help='候选材料CSV文件路径')
    parser.add_argument('--top', type=int, default=10, help='要可视化的顶级结构数量')
    parser.add_argument('--output_dir', type=str, default='results/visualizations', help='输出图像的目录')
    parser.add_argument('--format', type=str, default='html', 
                        choices=['png', 'jpg', 'pdf', 'svg', 'html', 'interactive'], 
                        help='输出图像格式，html/interactive格式为交互式可视化')
    parser.add_argument('--bonding', type=str, default='crystal', 
                        choices=['jmol', 'crystal', 'brunner', 'none'],
                        help='键合分析方法')
    parser.add_argument('--supercell', type=str, default='auto', help='超胞大小，格式为NxMxL，或使用"auto"自动确定')
    parser.add_argument('--dpi', type=int, default=300, help='图像DPI')
    parser.add_argument('--plot_size', type=int, default=800, help='图像大小（像素）')
    parser.add_argument('--view', type=str, default='3d', choices=['3d', 'ball_stick', 'polyhedral', 'space_filling'],
                        help='可视化视图类型')
    parser.add_argument('--title_prefix', type=str, default='LithiumVision候选结构',
                        help='图像标题前缀')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    return parser.parse_args()

def get_cif_structure(project_root, row, temp_dir=None):
    """从CIF文件或ZIP文件中读取结构"""
    cif_file = row['cif_file']
    source_file = row.get('source_file', '')
    
    # 首先尝试直接使用完整路径
    try:
        if os.path.exists(cif_file):
            return Structure.from_file(cif_file)
    except Exception as e:
        logger.warning(f"无法使用完整路径读取结构 {cif_file}: {e}")
    
    # 尝试在项目根目录中查找
    try:
        full_path = os.path.join(project_root, cif_file)
        if os.path.exists(full_path):
            return Structure.from_file(full_path)
    except Exception as e:
        logger.warning(f"无法从项目根目录读取结构 {full_path}: {e}")
    
    # 尝试从data/all_cifs目录读取
    try:
        filename = os.path.basename(cif_file)
        source_dir = os.path.basename(os.path.dirname(cif_file))
        all_cifs_path = os.path.join(project_root, "data", "all_cifs", source_dir, filename)
        if os.path.exists(all_cifs_path):
            return Structure.from_file(all_cifs_path)
    except Exception as e:
        logger.warning(f"无法从all_cifs目录读取结构 {all_cifs_path}: {e}")
    
    # 尝试从原始路径的生成目录读取
    try:
        old_path = project_root / "data" / "generated" / source_file / "generated_crystals_cif" / os.path.basename(cif_file)
        if old_path.exists():
            return Structure.from_file(old_path)
    except Exception as e:
        logger.warning(f"无法从生成目录读取结构 {old_path}: {e}")
    
    # 尝试从ZIP文件中读取
    try:
        zip_path = project_root / "data" / "generated" / source_file / "generated_crystals_cif.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                if temp_dir is None:
                    temp_dir = tempfile.mkdtemp()
                
                # 查找匹配的CIF文件
                cif_files = [f for f in zip_ref.namelist() if f.endswith('.cif')]
                cif_file_base = os.path.basename(cif_file)
                cif_file_in_zip = None
                
                for cf in cif_files:
                    if os.path.basename(cf) == cif_file_base:
                        cif_file_in_zip = cf
                        break
                
                if cif_file_in_zip:
                    extracted_path = Path(temp_dir) / os.path.basename(cif_file_in_zip)
                    with zip_ref.open(cif_file_in_zip) as source, open(extracted_path, 'wb') as target:
                        target.write(source.read())
                    
                    try:
                        return Structure.from_file(extracted_path)
                    except Exception as e:
                        logger.warning(f"无法从ZIP提取的文件读取结构: {e}")
                else:
                    logger.warning(f"在ZIP文件中未找到CIF文件: {cif_file}")
    except Exception as e:
        logger.warning(f"无法从ZIP文件读取结构: {e}")
    
    # 在所有可能的位置搜索文件
    try:
        for root, dirs, files in os.walk(os.path.join(project_root, "data")):
            if os.path.basename(cif_file) in files:
                full_path = os.path.join(root, os.path.basename(cif_file))
                return Structure.from_file(full_path)
    except Exception as e:
        logger.warning(f"搜索文件时出错: {e}")
    
    logger.error(f"找不到CIF文件: {cif_file}")
    return None

def make_supercell(structure, supercell_str):
    """根据指定的大小创建超胞，自动扩展小晶胞"""
    try:
        # 检查是否需要自动选择超胞大小
        if supercell_str.lower() == "auto":
            # 获取晶胞信息
            a, b, c = structure.lattice.abc
            num_atoms = len(structure)
            volume = structure.lattice.volume
            
            # 判断晶胞是否过小
            is_small_cell = (volume < 100) or (num_atoms < 8) or (min(a, b, c) < 3.0)
            
            if is_small_cell:
                # 根据体积和尺寸判断扩展倍数
                dims = [1, 1, 1]
                
                # 对于特别小的晶胞，按方向扩展
                if a < 4.0:
                    dims[0] = 2
                if b < 4.0:
                    dims[1] = 2
                if c < 4.0:
                    dims[2] = 2
                
                # 如果总体积还是太小，均匀扩展
                if volume < 50 and dims == [1, 1, 1]:
                    dims = [2, 2, 2]
                
                logger.info(f"晶胞较小 (体积={volume:.2f}Å³, 原子数={num_atoms})，自动扩展为 {dims[0]}x{dims[1]}x{dims[2]} 超胞")
                return structure.make_supercell(dims)
            else:
                logger.info(f"晶胞大小适中 (体积={volume:.2f}Å³, 原子数={num_atoms})，使用原始晶胞")
                return structure
        
        # 使用指定的超胞大小
        if supercell_str == "1x1x1":
            return structure  # 不需要创建超胞
        
        dims = [int(x) for x in supercell_str.split('x')]
        if len(dims) != 3:
            logger.warning(f"超胞格式不正确: {supercell_str}，使用默认1x1x1")
            return structure
        
        return structure.make_supercell(dims)
    except Exception as e:
        logger.warning(f"创建超胞时出错: {e}，使用原始结构")
        return structure

def get_bond_analyzer(method):
    """获取键合分析器"""
    if method == 'jmol':
        return JmolNN()
    elif method == 'crystal':
        return CrystalNN()
    elif method == 'brunner':
        return BrunnerNN_real()
    else:
        return None

def create_structure_info(structure):
    """创建结构信息文本"""
    try:
        spg_analyzer = SpacegroupAnalyzer(structure)
        symmetry_data = spg_analyzer.get_symmetry_dataset()
        spg_symbol = symmetry_data["international"]
        spg_number = symmetry_data["number"]
        formula = structure.composition.reduced_formula
        
        # 获取晶格参数
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        volume = structure.lattice.volume
        
        info = f"化学式: {formula}\n"
        info += f"空间群: {spg_symbol} (#{spg_number})\n"
        info += f"晶格参数: a={a:.4f}, b={b:.4f}, c={c:.4f}\n"
        info += f"晶格角度: α={alpha:.2f}°, β={beta:.2f}°, γ={gamma:.2f}°\n"
        info += f"晶胞体积: {volume:.4f} Å³\n"
        info += f"原子数: {len(structure)}\n"
        
        return info
    except Exception as e:
        logger.error(f"生成结构信息时出错: {e}")
        return "无法获取结构信息"

def plot_plotly(structure, title, output_file, bonding_method='crystal', width=800, height=800):
    """使用Plotly创建交互式3D结构可视化"""
    if not PLOTLY_AVAILABLE:
        logger.error("未安装plotly库，无法创建交互式可视化")
        return False
    
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # 提取原子位置并转换为笛卡尔坐标
    sites = structure.sites
    coords = np.array([site.coords for site in sites])
    
    # 原子信息
    atom_types = [site.specie.symbol for site in sites]
    unique_atoms = set(atom_types)
    
    # 获取原子类型与索引的映射
    atom_indices = {atom: [i for i, a in enumerate(atom_types) if a == atom] for atom in unique_atoms}
    
    # 添加晶格框架线
    lattice = structure.lattice
    origin = np.array([0, 0, 0])
    a, b, c = lattice.matrix[0], lattice.matrix[1], lattice.matrix[2]
    
    # 晶格点
    points = np.array([
        origin, origin + a, origin + b, origin + c,
        origin + a + b, origin + a + c, origin + b + c,
        origin + a + b + c
    ])
    
    # 连接边
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), 
        (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)
    ]
    
    # 绘制晶格框架
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[points[i][0], points[j][0]],
            y=[points[i][1], points[j][1]],
            z=[points[i][2], points[j][2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # 键合分析
    if bonding_method != 'none':
        bond_analyzer = get_bond_analyzer(bonding_method)
        if bond_analyzer:
            try:
                graph = StructureGraph.with_local_env_strategy(structure, bond_analyzer)
                
                # 绘制键
                bond_x, bond_y, bond_z = [], [], []
                for i, j, d in graph.graph.edges(data=True):
                    start = coords[i]
                    end = coords[j]
                    # 添加连接线，中间有一个None来断开线段
                    bond_x.extend([start[0], end[0], None])
                    bond_y.extend([start[1], end[1], None])
                    bond_z.extend([start[2], end[2], None])
                
                # 添加键
                fig.add_trace(go.Scatter3d(
                    x=bond_x, y=bond_y, z=bond_z,
                    mode='lines',
                    line=dict(color='grey', width=2.5),
                    opacity=0.7,
                    showlegend=False,
                    hoverinfo='none'
                ))
            except Exception as e:
                logger.warning(f"键合分析失败: {e}")
    
    # 绘制原子
    for atom in unique_atoms:
        indices = atom_indices[atom]
        color = ELEMENT_COLORS.get(atom, DEFAULT_ELEMENT_COLOR)
        radius = ELEMENT_RADII.get(atom, DEFAULT_RADIUS) * 0.5  # 缩放半径
        
        atom_coords = coords[indices]
        
        fig.add_trace(go.Scatter3d(
            x=atom_coords[:, 0],
            y=atom_coords[:, 1],
            z=atom_coords[:, 2],
            mode='markers',
            marker=dict(
                size=radius*20,  # 放大原子大小以便更好地可视化
                color=color,
                symbol='circle',
                line=dict(color='black', width=0.5),
                opacity=0.9
            ),
            name=atom,
            text=[f"{atom} @ ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})" for coord in atom_coords],
            hoverinfo='text'
        ))
    
    # 设置布局和视图
    info_text = create_structure_info(structure)
    
    fig.update_layout(
        title=dict(
            text=f"{title}<br><span style='font-size:12px'>{info_text.replace(chr(10), '<br>')}</span>",
            x=0.5,
            y=0.95,
            font=dict(size=16)
        ),
        width=width,
        height=height,
        scene=dict(
            xaxis=dict(title='X (Å)', showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(title='Y (Å)', showgrid=True, gridwidth=1, gridcolor='lightgray'),
            zaxis=dict(title='Z (Å)', showgrid=True, gridwidth=1, gridcolor='lightgray'),
            aspectmode='data'  # 保持真实比例
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.7)"
        ),
        margin=dict(l=10, r=10, t=120, b=10),  # 调整边距以容纳标题
        hovermode='closest'
    )
    
    try:
        fig.write_html(output_file)
        logger.info(f"已保存交互式可视化HTML: {output_file}")
        return True
    except Exception as e:
        logger.error(f"保存HTML可视化时出错: {e}")
        return False

def export_nglview(structure, title, output_file, bonding_method='crystal'):
    """使用NGLView创建高级交互式分子可视化"""
    if not NGLVIEW_AVAILABLE:
        logger.error("未安装nglview和ase库，无法创建NGLView可视化")
        return False
    
    try:
        # 创建临时CIF文件
        temp_cif = tempfile.NamedTemporaryFile(suffix='.cif', delete=False)
        cif_writer = CifWriter(structure)
        cif_writer.write_file(temp_cif.name)
        temp_cif.close()
        
        # 使用ASE读取
        ase_atoms = ase_read(temp_cif.name, format='cif')
        
        # 创建NGLView视图
        view = nglview.show_ase(ase_atoms)
        
        # 为不同的元素设置不同的表示方式
        element_set = set(atom.symbol for atom in ase_atoms)
        
        # 先添加基本的球棍模型
        view.add_representation('ball+stick')
        
        # 为每个元素添加特定的空间填充表示
        for element in element_set:
            color = ELEMENT_COLORS.get(element, DEFAULT_ELEMENT_COLOR)
            radius = ELEMENT_RADII.get(element, DEFAULT_RADIUS)
            
            # 为元素添加空间填充表示，透明度设为0.5以便同时看到内部键合
            view.add_representation('spacefill', selection=f'#{element}', 
                                    color=color, radius=radius*0.5, opacity=0.5)
        
        # 设置视图属性
        view.camera = 'orthographic'  # 正交投影，方便查看晶体结构
        view.center()  # 居中显示结构
        
        # 保存为HTML
        view._display_image = False  # 禁用静态图像
        view._init_gui = True  # 启用GUI控件
        
        # 添加标题和信息
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
                .container {{ width: 100%; }}
                .header {{ padding: 10px; background-color: #f0f0f0; text-align: center; }}
                .info {{ white-space: pre-wrap; font-family: monospace; padding: 10px; 
                        background-color: #f9f9f9; margin-bottom: 10px; }}
                .viewer {{ height: 600px; width: 100%; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{title}</h2>
                </div>
                <div class="info">
                    {create_structure_info(structure)}
                </div>
                <div class="viewer">
                    {view._repr_html_()}
                </div>
            </div>
            <script>
                // 添加自动旋转功能
                document.addEventListener('DOMContentLoaded', function() {{
                    setTimeout(function() {{
                        var stage = Object.values(window)[0].__ngl_stage__;
                        if (stage) {{
                            stage.autoView();
                            stage.setSpin(true); // 开启自动旋转
                        }}
                    }}, 1000);
                }});
            </script>
        </body>
        </html>
        """
        
        # 保存HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 清理临时文件
        try:
            os.unlink(temp_cif.name)
        except:
            pass
            
        logger.info(f"已保存NGLView交互式可视化: {output_file}")
        return True
    except Exception as e:
        logger.error(f"创建NGLView可视化时出错: {e}")
        return False 

def plot_matplotlib(structure, title, output_file, bonding_method='crystal', dpi=300, figsize=(10, 10)):
    """使用matplotlib创建静态结构图像"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取原子位置并转换为笛卡尔坐标
    sites = structure.sites
    coords = np.array([site.coords for site in sites])
    
    # 绘制单元格
    lattice = structure.lattice
    origin = np.array([0, 0, 0])
    a, b, c = lattice.matrix[0], lattice.matrix[1], lattice.matrix[2]
    
    # 单元格边界点
    points = np.array([
        origin, origin + a, origin + b, origin + c,
        origin + a + b, origin + a + c, origin + b + c,
        origin + a + b + c
    ])
    
    # 绘制单元格边缘
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), 
        (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)
    ]
    
    for i, j in edges:
        ax.plot3D([points[i][0], points[j][0]], 
                  [points[i][1], points[j][1]], 
                  [points[i][2], points[j][2]], 'k-', alpha=0.5)
    
    # 绘制原子
    atom_types = [site.specie.symbol for site in sites]
    unique_atoms = set(atom_types)
    
    # 获取原子类型与索引的映射
    atom_indices = {atom: [i for i, a in enumerate(atom_types) if a == atom] for atom in unique_atoms}
    
    # 如果需要键合分析
    if bonding_method != 'none':
        bond_analyzer = get_bond_analyzer(bonding_method)
        if bond_analyzer:
            try:
                graph = StructureGraph.with_local_env_strategy(structure, bond_analyzer)
                
                # 绘制键
                for i, j, d in graph.graph.edges(data=True):
                    start = coords[i]
                    end = coords[j]
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                            'k-', alpha=0.6, linewidth=1)
            except Exception as e:
                logger.warning(f"键合分析失败: {e}")
    
    # 绘制原子
    for atom in unique_atoms:
        indices = atom_indices[atom]
        color = ELEMENT_COLORS.get(atom, DEFAULT_ELEMENT_COLOR)
        radius = ELEMENT_RADII.get(atom, DEFAULT_RADIUS) * 0.4  # 缩放半径以适应图
        
        atom_coords = coords[indices]
        ax.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2], 
                  c=color, s=radius*200, label=atom, alpha=0.8, edgecolors='k', linewidths=0.5)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # 设置轴标签
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # 设置标题
    ax.set_title(title)
    
    # 调整视角
    ax.view_init(elev=20, azim=30)  # 俯仰角20度，方位角30度
    
    # 添加结构信息
    info_text = create_structure_info(structure)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    fig.text(0.05, 0.05, info_text, fontsize=8, verticalalignment='bottom', 
             bbox=props, family='monospace')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"已保存静态图像: {output_file}")
    return True

def main():
    """主函数"""
    args = parse_arguments()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 获取项目根目录
    project_root = Path.cwd()
    
    # 加载候选材料数据 - 修改以支持ZIP文件
    candidates_df = None
    if args.candidates.endswith('.zip'):
        # 处理ZIP文件：提取CIF文件并创建DataFrame
        zip_path = Path(args.candidates)
        if not zip_path.exists():
            logger.error(f"ZIP文件不存在: {args.candidates}")
            return 1
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                cif_files = [f for f in zip_ref.namelist() if f.endswith('.cif')]
                if not cif_files:
                    logger.error(f"ZIP文件中未找到CIF文件: {args.candidates}")
                    return 1
                
                # 提取CIF文件到临时目录
                extracted_paths = []
                for cif_file in cif_files:
                    extracted_path = Path(temp_dir) / Path(cif_file).name
                    with zip_ref.open(cif_file) as source, open(extracted_path, 'wb') as target:
                        target.write(source.read())
                    extracted_paths.append(str(extracted_path))
                
                # 创建DataFrame
                data = {
                    'cif_file': extracted_paths,
                    'formula': [Path(p).stem for p in extracted_paths],  # 使用文件名作为公式占位符
                    'source_file': [zip_path.name] * len(extracted_paths)  # 使用ZIP文件名作为来源
                }
                candidates_df = pd.DataFrame(data)
                logger.info(f"从ZIP文件加载了{len(candidates_df)}个CIF文件")
        except Exception as e:
            logger.error(f"处理ZIP文件时出错: {e}")
            return 1
    else:
        # 原有逻辑：加载CSV文件
        try:
            candidates_df = pd.read_csv(args.candidates)
            logger.info(f"已加载{len(candidates_df)}个候选材料")
        except Exception as e:
            logger.error(f"无法加载候选材料CSV文件: {e}")
            return 1
    
    if candidates_df is None or candidates_df.empty:
        logger.error("未找到有效的候选材料数据")
        return 1
    
    # 限制处理的结构数量
    if len(candidates_df) > args.top:
        candidates_df = candidates_df.head(args.top)
        logger.info(f"处理前{args.top}个候选材料")
    
    # 处理每个结构
    processed_count = 0
    
    for idx, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="可视化结构"):
        try:
            structure = get_cif_structure(project_root, row, temp_dir)
            if structure is None:
                logger.warning(f"跳过无法加载的结构: {row['cif_file']}")
                continue
            
            # 创建超胞
            structure = make_supercell(structure, args.supercell)
            
            # 文件名 - 提取基本名称，不包含路径
            cif_filename = os.path.basename(row['cif_file'])
            formula = row.get('formula', 'unknown')
            source = row.get('source_file', 'unknown')
            
            # 创建简洁的文件名
            base_name = f"{idx+1:02d}_{cif_filename.replace('.cif', '')}"
            
            # 标题
            title = f"{args.title_prefix} #{idx+1}: {formula} ({source})"
            
            # 根据格式选择可视化方法
            if args.format == 'html':
                output_file = output_dir / f"{base_name}.html"
                plot_plotly(structure, title, output_file, args.bonding, args.plot_size, args.plot_size)
            elif args.format == 'interactive':
                output_file = output_dir / f"{base_name}_nglview.html"
                export_nglview(structure, title, output_file, args.bonding)
            else:
                output_file = output_dir / f"{base_name}.{args.format}"
                plot_matplotlib(structure, title, output_file, args.bonding, args.dpi, (10, 10))
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"处理结构时出错 {row.get('cif_file', 'unknown')}: {e}")
    
    # 清理临时目录
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
    
    logger.info(f"已完成 {processed_count}/{len(candidates_df)} 个结构的可视化")
    logger.info(f"可视化结果保存在: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())