#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载模块 - 加载由data_processess.py处理的数据

这个模块提供了从data_processess.py生成的处理后数据中加载图数据的功能，
并将其转换为GNN.py可用的格式。
"""

import os
import torch
import logging
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ProcessedGraphDataset(Dataset):
    """
    加载data_processess.py处理后的图数据集
    """
    def __init__(self, root_dir, split='train', transform=None, pre_transform=None):
        """
        初始化数据集
        
        参数:
            root_dir (str): 处理后数据的根目录
            split (str): 使用的数据集划分 ('train', 'val', 'test')
            transform: 数据转换
            pre_transform: 预处理转换
        """
        self.root_dir = root_dir
        self.split = split
        self.file_list = []
        
        # 加载对应的文件列表
        split_file = os.path.join(root_dir, f"{split}_files.txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_list = [line.strip() for line in f if line.strip()]
            logger.info(f"加载了 {len(self.file_list)} 个{split}数据文件")
        else:
            logger.warning(f"找不到{split}划分文件: {split_file}")
        
        # 加载特征统计信息用于归一化
        self.node_stats = None
        self.edge_stats = None
        node_stats_path = os.path.join(root_dir, "node_feature_stats.pt")
        edge_stats_path = os.path.join(root_dir, "edge_feature_stats.pt")
        
        if os.path.exists(node_stats_path):
            self.node_stats = torch.load(node_stats_path)
            logger.info("加载了节点特征统计信息")
        
        if os.path.exists(edge_stats_path):
            self.edge_stats = torch.load(edge_stats_path)
            logger.info("加载了边特征统计信息")
            
        super(ProcessedGraphDataset, self).__init__(root_dir, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return self.file_list
    
    def len(self):
        return len(self.file_list)
    
    def get(self, idx):
        """
        获取指定索引的数据
        
        参数:
            idx (int): 数据索引
            
        返回:
            Data: PyTorch Geometric数据对象
        """
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        try:
            data = torch.load(file_path)
            
            # 确保data是Data对象
            if not isinstance(data, Data):
                logger.warning(f"文件 {file_path} 不包含有效的Data对象")
                # 创建一个空的Data对象
                return Data(x=torch.tensor([], dtype=torch.float))
            
            # 应用归一化
            if self.node_stats is not None and hasattr(data, 'x') and data.x is not None:
                data.x = (data.x - self.node_stats['mean']) / self.node_stats['std']
            
            if self.edge_stats is not None and hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = (data.edge_attr - self.edge_stats['mean']) / self.edge_stats['std']
            
            # 将log_conductivity作为y值用于训练
            if hasattr(data, 'log_conductivity'):
                data.y = data.log_conductivity
            elif hasattr(data, 'conductivity'):
                # 如果没有log_conductivity但有conductivity，手动计算log值
                data.y = torch.log(torch.clamp(data.conductivity, min=1e-10))
            
            return data
            
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {str(e)}")
            # 返回一个空的Data对象
            return Data(x=torch.tensor([], dtype=torch.float))
    
    def get_stats(self):
        """
        获取数据集统计信息
        
        返回:
            dict: 包含数据集统计信息的字典
        """
        return {
            'node_stats': self.node_stats,
            'edge_stats': self.edge_stats,
            'num_samples': len(self.file_list)
        }

def create_data_loaders(processed_dir, batch_size=32, num_workers=0):
    """
    为训练、验证和测试创建数据加载器
    
    参数:
        processed_dir (str): 处理后数据的根目录
        batch_size (int): 批处理大小
        num_workers (int): 数据加载的工作线程数
        
    返回:
        tuple: (train_loader, val_loader, test_loader)
    """
    try:
        logger.info(f"开始创建数据加载器，数据目录: {processed_dir}")
        
        # 检查目录是否存在
        if not os.path.exists(processed_dir):
            logger.error(f"数据目录不存在: {processed_dir}")
            raise FileNotFoundError(f"数据目录不存在: {processed_dir}")
            
        # 检查必要的文件是否存在
        required_files = ['train_files.txt', 'val_files.txt', 'test_files.txt']
        for file in required_files:
            file_path = os.path.join(processed_dir, file)
            if not os.path.exists(file_path):
                logger.error(f"必要的文件不存在: {file_path}")
                raise FileNotFoundError(f"必要的文件不存在: {file_path}")
        
        # 在Windows系统上默认使用0个工作线程以避免潜在问题
        if os.name == 'nt' and num_workers > 0:
            logger.info(f"检测到Windows系统，默认将num_workers设为0以避免潜在问题")
            num_workers = 0
        
        logger.info(f"创建训练数据集...")
        train_dataset = ProcessedGraphDataset(processed_dir, split='train')
        logger.info(f"创建验证数据集...")
        val_dataset = ProcessedGraphDataset(processed_dir, split='val')
        logger.info(f"创建测试数据集...")
        test_dataset = ProcessedGraphDataset(processed_dir, split='test')
        
        # 获取数据集大小
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(test_dataset)
        
        logger.info(f"数据集大小: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")
        
        # 创建数据加载器，设置pin_memory=True以加速GPU训练
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.error(f"创建数据加载器时出错: {str(e)}")
        # 返回空的数据加载器，而不是None
        empty_dataset = ProcessedGraphDataset(processed_dir, split='train')
        empty_loader = DataLoader(empty_dataset, batch_size=1)
        return empty_loader, empty_loader, empty_loader

def get_feature_dimensions(processed_dir):
    """
    从处理后的数据中获取特征维度
    
    参数:
        processed_dir (str): 处理后数据的根目录
        
    返回:
        tuple: (node_dim, edge_dim) 节点和边特征的维度
    """
    # 加载第一个训练样本以确定特征维度
    train_dataset = ProcessedGraphDataset(processed_dir, split='train')
    
    if len(train_dataset) == 0:
        logger.warning("没有找到训练数据，无法确定特征维度")
        return 0, 0
    
    # 获取第一个样本
    sample = train_dataset[0]
    
    node_dim = sample.x.size(1) if hasattr(sample, 'x') and sample.x is not None and sample.x.dim() > 1 else 0
    edge_dim = sample.edge_attr.size(1) if hasattr(sample, 'edge_attr') and sample.edge_attr is not None and sample.edge_attr.dim() > 1 else 0
    
    logger.info(f"特征维度: 节点特征={node_dim}, 边特征={edge_dim}")
    
    return node_dim, edge_dim 