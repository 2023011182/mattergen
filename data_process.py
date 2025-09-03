#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preprocessing Script for Li-ion Solid State Electrolyte Conductivity Prediction

This script processes CIF crystal structure files and corresponding conductivity data
from the LithiumVision repository to prepare input for neural network training.
It handles the extraction of structural features from CIF files, converts them to
graph representations suitable for GNNs, and aligns them with conductivity values.

Based on the development strategy and existing code in the LithiumVision repository.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pymatgen.core.structure import Structure
import logging
import zipfile
import warnings
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

class CrystalDataProcessor:
    """
    Processor for crystal structures to extract features for conductivity prediction.
    Handles CIF files and converts them to graph representations for GNN models.
    """
    
    def __init__(self, 
                 cif_dir=None, 
                 conductivity_file=None,
                 output_dir="processed_data",
                 cutoff_radius=8.0):
        """
        Initialize the processor with paths and parameters.
        
        Args:
            cif_dir (str): Directory containing CIF files or zip archive
            conductivity_file (str): Path to CSV file with conductivity values
            output_dir (str): Directory to save processed data
            cutoff_radius (float): Cutoff radius for graph construction (Angstroms)
        """
        self.cif_dir = cif_dir
        self.conductivity_file = conductivity_file
        self.output_dir = output_dir
        self.cutoff_radius = cutoff_radius
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Maps for element properties
        self._setup_element_properties()
        
    def _setup_element_properties(self):
        """Setup mappings for element properties used for node features."""
        # Atomic radii in Angstroms (reference: https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements)
        self.atomic_radii = {
            1: 0.25, 2: 0.31, 3: 1.45, 4: 1.05, 5: 0.85, 6: 0.70, 7: 0.65, 8: 0.60, 
            9: 0.50, 10: 0.38, 11: 1.80, 12: 1.50, 13: 1.25, 14: 1.10, 15: 1.00, 
            16: 1.00, 17: 1.00, 18: 0.71, 19: 2.20, 20: 1.80, 21: 1.60, 22: 1.40, 
            23: 1.35, 24: 1.40, 25: 1.40, 26: 1.40, 27: 1.35, 28: 1.35, 29: 1.35, 
            30: 1.35, 31: 1.30, 32: 1.25, 33: 1.15, 34: 1.15, 35: 1.15, 36: 0.88, 
            37: 2.35, 38: 2.00, 39: 1.85, 40: 1.55, 41: 1.45, 42: 1.45, 43: 1.35, 
            44: 1.30, 45: 1.35, 46: 1.40, 47: 1.60, 48: 1.55, 49: 1.55, 50: 1.45, 
            51: 1.45, 52: 1.40, 53: 1.40, 54: 1.08, 55: 2.60, 56: 2.15, 57: 1.95, 
            58: 1.85, 59: 1.85, 60: 1.85, 61: 1.85, 62: 1.85, 63: 1.85, 64: 1.80, 
            65: 1.75, 66: 1.75, 67: 1.75, 68: 1.75, 69: 1.75, 70: 1.75, 71: 1.75, 
            72: 1.55, 73: 1.45, 74: 1.35, 75: 1.35, 76: 1.30, 77: 1.35, 78: 1.35, 
            79: 1.35, 80: 1.50, 81: 1.90, 82: 1.80, 83: 1.60, 84: 1.90, 85: 1.50, 
            86: 1.20, 87: 2.60, 88: 2.15, 89: 1.10, 90: 1.80, 91: 1.80, 92: 1.75, 
            93: 1.75, 94: 1.75, 95: 1.75
        }
        
        # Electronegativity (Pauling scale)
        self.electronegativity = {
            1: 2.20, 2: 0.00, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 
            9: 3.98, 10: 0.00, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 
            16: 2.58, 17: 3.16, 18: 0.00, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 
            23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 
            30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 0.00, 
            37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90, 
            44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96, 
            51: 2.05, 52: 2.10, 53: 2.66, 54: 0.00, 55: 0.79, 56: 0.89, 57: 1.10, 
            58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13, 62: 1.17, 63: 1.20, 64: 1.20, 
            65: 1.22, 66: 1.23, 67: 1.24, 68: 1.24, 69: 1.25, 70: 1.10, 71: 1.27, 
            72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28, 
            79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20, 
            86: 0.00, 87: 0.70, 88: 0.90, 89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38, 
            93: 1.36, 94: 1.28, 95: 1.30
        }
        
        # Number of valence electrons
        self.valence_electrons = {
            1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
            11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
            19: 1, 20: 2, 21: 2, 22: 2, 23: 2, 24: 1, 25: 2, 26: 2, 27: 2, 28: 2, 29: 1, 30: 2,
            31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8,
            37: 1, 38: 2, 39: 2, 40: 2, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 0, 47: 1, 48: 2,
            49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8,
            55: 1, 56: 2
            # For simplicity, we're only listing up to Ba (Z=56), add more as needed
        }
        
    def load_conductivity_data(self):
        """
        Load conductivity data from CSV file.
        
        Returns:
            pandas.DataFrame: DataFrame with material IDs and conductivity values
        """
        if not self.conductivity_file or not os.path.exists(self.conductivity_file):
            logger.warning("Conductivity file not found: %s", self.conductivity_file)
            return pd.DataFrame(columns=['material_id', 'conductivity'])
        
        try:
            df = pd.read_csv(self.conductivity_file)
            logger.info(f"Loaded conductivity data for {len(df)} materials")
            
            # Basic validation and processing
            if 'material_id' not in df.columns:
                # Try to identify the ID column
                id_columns = [col for col in df.columns if 'id' in col.lower()]
                if id_columns:
                    df.rename(columns={id_columns[0]: 'material_id'}, inplace=True)
                    logger.info(f"Renamed column '{id_columns[0]}' to 'material_id'")
                else:
                    # If no ID column found, create one from the index
                    df['material_id'] = df.index.astype(str)
                    logger.warning("No material ID column found, created from index")
            
            # Identify conductivity column
            cond_columns = [col for col in df.columns if any(x in col.lower() for x in ['conductivity', 'cond', 'ionic'])]
            if cond_columns:
                if 'conductivity' not in df.columns:
                    df.rename(columns={cond_columns[0]: 'conductivity'}, inplace=True)
                    logger.info(f"Renamed column '{cond_columns[0]}' to 'conductivity'")
            else:
                logger.error("No conductivity column found in the data")
                return pd.DataFrame(columns=['material_id', 'conductivity'])
            
            # Apply log transformation for training if needed (similar to what GNN.py does)
            if 'log_conductivity' not in df.columns:
                # Ensure conductivity values are positive for log transformation
                positive_mask = df['conductivity'] > 0
                if not all(positive_mask):
                    logger.warning(f"Found {(~positive_mask).sum()} non-positive conductivity values")
                
                # Create log-transformed values for positive conductivities
                df['log_conductivity'] = np.nan
                df.loc[positive_mask, 'log_conductivity'] = np.log(df.loc[positive_mask, 'conductivity'])
                logger.info(f"Created log-transformed conductivity for {positive_mask.sum()} entries")
            
            return df[['material_id', 'conductivity', 'log_conductivity']].dropna()
            
        except Exception as e:
            logger.error(f"Error loading conductivity data: {str(e)}")
            return pd.DataFrame(columns=['material_id', 'conductivity'])
    
    def _extract_cif_files(self, target_dir):
        """Extract CIF files from zip archives if needed."""
        # Check if the input is a zip file
        if self.cif_dir and self.cif_dir.endswith('.zip') and os.path.exists(self.cif_dir):
            logger.info(f"Extracting CIF files from {self.cif_dir}")
            try:
                with zipfile.ZipFile(self.cif_dir, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                return glob.glob(os.path.join(target_dir, "**/*.cif"), recursive=True)
            except Exception as e:
                logger.error(f"Error extracting zip file: {str(e)}")
                return []
        
        # If the input is a directory, find all CIF files
        elif self.cif_dir and os.path.isdir(self.cif_dir):
            return glob.glob(os.path.join(self.cif_dir, "**/*.cif"), recursive=True)
            
        # Check for zip files in the current directory
        elif any(os.path.exists(zip_name) for zip_name in ["randomized_cifs.zip", "processed_graphs.zip"]):
            target = "randomized_cifs.zip" if os.path.exists("randomized_cifs.zip") else "processed_graphs.zip"
            logger.info(f"Using existing zip file: {target}")
            self.cif_dir = target
            return self._extract_cif_files(target_dir)
            
        else:
            logger.error("No CIF files or zip archives found")
            return []
    
    def _get_element_features(self, atomic_number):
        """Get combined element features for node representation."""
        features = []
        
        # Add atomic number (one-hot or as value)
        features.append(float(atomic_number))
        
        # Add atomic radius (normalized)
        radius = self.atomic_radii.get(atomic_number, 0.0)
        features.append(radius)
        
        # Add electronegativity
        electroneg = self.electronegativity.get(atomic_number, 0.0)
        features.append(electroneg)
        
        # Add valence electrons
        valence = self.valence_electrons.get(atomic_number, 0)
        features.append(float(valence))
        
        # Li-specific flag (important for Li-ion conductivity)
        is_li = 1.0 if atomic_number == 3 else 0.0
        features.append(is_li)
        
        return features
        
    def cif_to_graph(self, cif_file):
        """
        Convert a CIF file to a PyTorch Geometric Data object.
        
        Args:
            cif_file (str): Path to CIF file
            
        Returns:
            torch_geometric.data.Data: Graph representation of the crystal structure
            or None if processing fails
        """
        try:
            # Parse CIF file using pymatgen
            structure = Structure.from_file(cif_file)
            
            # Get unique material ID from filename
            material_id = os.path.splitext(os.path.basename(cif_file))[0]
            
            # Extract node features (atomic properties)
            node_features = []
            node_positions = []
            
            for site in structure.sites:
                # Get the dominant species (element) at this site
                species = site.species
                dominant_element = max(species.items(), key=lambda x: x[1])[0]
                atomic_number = dominant_element.Z
                
                # Get combined element features
                node_feature = self._get_element_features(atomic_number)
                node_features.append(node_feature)
                
                # Store fractional coordinates
                node_positions.append(site.frac_coords)
            
            # Convert to torch tensors
            node_features = torch.tensor(node_features, dtype=torch.float)
            node_positions = torch.tensor(node_positions, dtype=torch.float)
            
            # Create edges based on distance cutoff
            edge_index = []
            edge_attr = []
            
            # Get lattice matrix for distance calculations
            lattice_matrix = torch.tensor(structure.lattice.matrix, dtype=torch.float)
            
            # Create edges based on distance cutoff (periodic boundary conditions)
            num_nodes = len(node_positions)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # Avoid self-loops
                        # Calculate distance considering periodic boundary conditions
                        # This is a simplified version - pymatgen has more sophisticated methods
                        pos_i = node_positions[i]
                        pos_j = node_positions[j]
                        
                        # Calculate distance vector with periodic boundary conditions
                        # Convert fractional to cartesian for accurate distances
                        cart_i = torch.matmul(pos_i, lattice_matrix)
                        cart_j = torch.matmul(pos_j, lattice_matrix)
                        
                        # Use minimum image convention for periodic boundary
                        dist = structure.get_distance(i, j)
                        
                        if dist <= self.cutoff_radius:
                            edge_index.append([i, j])
                            
                            # Edge feature: distance between atoms
                            edge_attr.append([dist])
            
            # If no edges were created, the cutoff might be too small
            if not edge_index:
                logger.warning(f"No edges created for {cif_file}. Try increasing cutoff_radius.")
                # Optionally, return None or a graph with no edges if that's handled by the model
                # For now, return None, as a disconnected graph might not be meaningfully processed by GCN
                return None 
                
            # Convert to torch tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Add global (graph-level) features
            # These could include lattice parameters, space group, etc.
            lattice_params = torch.tensor([
                structure.lattice.a,
                structure.lattice.b,
                structure.lattice.c,
                structure.lattice.alpha,
                structure.lattice.beta,
                structure.lattice.gamma,
                structure.volume
            ], dtype=torch.float)
            
            # Calculate composition-based features
            composition = structure.composition
            element_counts = {}
            total_atoms = composition.num_atoms
            
            for element in composition:
                element_counts[element.symbol] = composition[element] / total_atoms
            
            # Calculate specific features relevant for Li-ion conductivity
            li_fraction = element_counts.get('Li', 0.0)
            
            # Create final Data object
            data = Data(
                x=node_features,  # Node features
                edge_index=edge_index,  # Edge connectivity
                edge_attr=edge_attr,  # Edge features (distances)
                pos=node_positions,  # Fractional coordinates (optional)
                lattice=lattice_params,  # Lattice parameters (optional global feature)
                li_fraction=torch.tensor([li_fraction], dtype=torch.float),  # Li content (optional global feature)
                material_id=material_id  # Material identifier
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing {cif_file}: {str(e)}")
            return None
    
    def process_all_cifs(self, temp_dir="tmp_cifs"):
        """
        Process all CIF files and convert them to graph data objects.
        
        Args:
            temp_dir (str): Temporary directory for extracted CIF files
            
        Returns:
            dict: Mapping of material IDs to graph data objects
        """
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Extract CIF files if needed
        cif_files = self._extract_cif_files(temp_dir)
        
        if not cif_files:
            logger.error("No CIF files found to process.")
            return {}
            
        logger.info(f"Found {len(cif_files)} CIF files to process")
        
        # Load conductivity data
        conductivity_df = self.load_conductivity_data()
        
        # Process each CIF file
        graph_data = {}
        success_count = 0
        error_count = 0
        
        for cif_file in tqdm(cif_files, desc="Processing CIF files"):
            try:
                # Get material ID from filename
                material_id = os.path.splitext(os.path.basename(cif_file))[0]
                
                # Convert CIF to graph
                graph = self.cif_to_graph(cif_file)

                if graph is not None:
                    # Add conductivity data if available
                    if not conductivity_df.empty:
                        match = conductivity_df[conductivity_df['material_id'] == material_id]
                        if not match.empty:
                            # Add both raw and log-transformed conductivity
                            graph.conductivity = torch.tensor([match['conductivity'].values[0]], dtype=torch.float)
                            graph.log_conductivity = torch.tensor([match['log_conductivity'].values[0]], dtype=torch.float)
                            # 关键：加上y属性
                            graph.y = torch.tensor([match['log_conductivity'].values[0]], dtype=torch.float)
                    
                    # Store graph with material ID as key
                    graph_data[material_id] = graph
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {cif_file}: {str(e)}")
                error_count += 1
        
        logger.info(f"Successfully processed {success_count} CIF files")
        logger.info(f"Failed to process {error_count} CIF files")
        
        return graph_data
    
    def save_processed_data(self, graph_data):
        """
        Save processed graph data to disk.
        
        Args:
            graph_data (dict): Dictionary mapping material IDs to graph data objects
            
        Returns:
            tuple: Lists of saved files with and without conductivity data
        """
        if not graph_data:
            logger.error("No data to save")
            return [], []
            
        # Create subdirectories
        graph_dir = os.path.join(self.output_dir, "graphs")
        os.makedirs(graph_dir, exist_ok=True)
        
        with_cond = []
        without_cond = []
        
        # Save each graph object
        for material_id, graph in tqdm(graph_data.items(), desc="Saving processed data"):
            output_path = os.path.join(graph_dir, f"{material_id}.pt")
            
            # Check if conductivity data is available
            has_conductivity = hasattr(graph, 'conductivity') and hasattr(graph, 'log_conductivity')
            
            try:
                torch.save(graph, output_path)
                if has_conductivity:
                    with_cond.append(os.path.join("graphs", f"{material_id}.pt")) # Save relative path
                else:
                    without_cond.append(os.path.join("graphs", f"{material_id}.pt")) # Save relative path
            except Exception as e:
                logger.error(f"Error saving {material_id}: {str(e)}")
        
        logger.info(f"Saved {len(with_cond)} graphs with conductivity data")
        logger.info(f"Saved {len(without_cond)} graphs without conductivity data")
        
        # Create index files with relative paths from output_dir
        with open(os.path.join(self.output_dir, "with_conductivity.txt"), 'w') as f:
            f.write('\n'.join(with_cond))
            
        with open(os.path.join(self.output_dir, "without_conductivity.txt"), 'w') as f:
            f.write('\n'.join(without_cond))
            
        return with_cond, without_cond
    
    def create_dataset_splits(self, with_conductivity_relative_paths, test_ratio=0.1, val_ratio=0.1, seed=42):
        """
        Create train/val/test splits from the processed data.
        
        Args:
            with_conductivity_relative_paths (list): List of relative file paths with conductivity data
            test_ratio (float): Ratio for test set
            val_ratio (float): Ratio for validation set
            seed (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary with train/val/test file lists (relative paths)
        """
        if not with_conductivity_relative_paths:
            logger.error("No files with conductivity data available for dataset creation")
            return {"train": [], "val": [], "test": []}
            
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Shuffle the relative paths
        file_paths = np.array(with_conductivity_relative_paths)
        indices = np.random.permutation(len(file_paths))
        
        # Calculate split sizes
        total_size = len(file_paths)
        test_size = max(1, int(total_size * test_ratio))
        val_size = max(1, int(total_size * val_ratio))
        train_size = total_size - test_size - val_size
        
        # Ensure splits aren't negative and sum correctly
        if train_size < 0:
            logger.warning("Adjusting split sizes: train_size became negative. Setting minimal splits.")
            test_size = min(test_size, total_size // 3) # Cap test size
            val_size = min(val_size, (total_size - test_size) // 2) # Cap val size
            train_size = total_size - test_size - val_size # Recalculate train size
            train_size = max(0, train_size) # Ensure train size is non-negative
            logger.info(f"Adjusted splits: {train_size} train, {val_size} val, {test_size} test")


        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_files = file_paths[train_indices].tolist()
        val_files = file_paths[val_indices].tolist()
        test_files = file_paths[test_indices].tolist()
        
        logger.info(f"Created dataset splits: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
        
        # Save splits to files
        splits = {"train": train_files, "val": val_files, "test": test_files}
        
        for split_name, files in splits.items():
            with open(os.path.join(self.output_dir, f"{split_name}_files.txt"), 'w') as f:
                # Write one file path per line
                f.write('\n'.join(files))
        
        # Save split sizes to a summary file
        with open(os.path.join(self.output_dir, "dataset_summary.txt"), 'w') as f:
            f.write(f"Total files with conductivity: {len(with_conductivity_relative_paths)}\n")
            f.write(f"Training set: {len(train_files)} files\n")
            f.write(f"Validation set: {len(val_files)} files\n")
            f.write(f"Test set: {len(test_files)} files\n")
            f.write(f"\nRandom seed: {seed}\n")
            f.write(f"Test ratio: {test_ratio}\n")
            f.write(f"Validation ratio: {val_ratio}\n")
        
        return splits

    def run(self):
        """
        Execute the full preprocessing pipeline.
        
        Returns:
            dict: Summary of the preprocessing results
        """
        logger.info("Starting preprocessing pipeline")
        
        # Process CIF files
        graph_data = self.process_all_cifs()
        
        if not graph_data:
            logger.error("No graph data was generated.")
            return {"status": "error", "message": "No graph data generated"}
        
        # Save processed data
        # Save returns relative paths now
        with_cond_relative, without_cond_relative = self.save_processed_data(graph_data)
        
        # Create dataset splits using relative paths
        splits = self.create_dataset_splits(with_cond_relative)
        
        # Calculate and save feature statistics
        if with_cond_relative:
            # Need to load graphs again from relative paths or pass graph_data dict
            # Passing graph_data dict is easier here
            self._calculate_feature_statistics(graph_data)
        
        # Summarize results
        summary = {
            "status": "success",
            "total_processed": len(graph_data),
            "with_conductivity": len(with_cond_relative),
            "without_conductivity": len(without_cond_relative),
            "train_size": len(splits["train"]),
            "val_size": len(splits["val"]),
            "test_size": len(splits["test"]),
            "output_dir": self.output_dir
        }
        
        
        # Save summary as JSON
        import json
        with open(os.path.join(self.output_dir, "preprocessing_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Preprocessing completed successfully")
        logger.info(f"Results saved to {self.output_dir}")
        
        return summary
    
    def _calculate_feature_statistics(self, graph_data):
        """
        Calculate statistics for node and edge features for normalization.
        
        Args:
            graph_data (dict): Dictionary of graph data objects
        """
        if not graph_data:
            return
        
        logger.info("Calculating feature statistics for normalization")
        
        # Collect all node and edge features from graphs with conductivity
        all_node_features = []
        all_edge_features = []
        
        # Only use data with conductivity for statistics, consistent with training set origin
        graphs_with_conductivity = [g for g in graph_data.values() if hasattr(g, 'log_conductivity')]
        
        for graph in graphs_with_conductivity:
            if hasattr(graph, 'x') and graph.x is not None:
                all_node_features.append(graph.x)
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                all_edge_features.append(graph.edge_attr)
        
        if all_node_features:
            # Concatenate all node features
            all_node_features = torch.cat(all_node_features, dim=0)
            
            # Calculate mean and std for each feature dimension
            node_mean = torch.mean(all_node_features, dim=0)
            node_std = torch.std(all_node_features, dim=0)
            
            # Replace zero std with 1 to avoid division by zero
            node_std[node_std == 0] = 1.0
            
            # Save node feature statistics
            torch.save({
                'mean': node_mean,
                'std': node_std
            }, os.path.join(self.output_dir, "node_feature_stats.pt"))
            
            logger.info(f"Node feature mean: {node_mean}")
            logger.info(f"Node feature std: {node_std}")
        else:
             logger.warning("No node features found in graphs with conductivity. Skipping node stats.")
        
        if all_edge_features:
            # Concatenate all edge features
            all_edge_features = torch.cat(all_edge_features, dim=0)
            
            # Calculate mean and std for each feature dimension
            edge_mean = torch.mean(all_edge_features, dim=0)
            edge_std = torch.std(all_edge_features, dim=0)
            
            # Replace zero std with 1 to avoid division by zero
            edge_std[edge_std == 0] = 1.0
            
            # Save edge feature statistics
            torch.save({
                'mean': edge_mean,
                'std': edge_std
            }, os.path.join(self.output_dir, "edge_feature_stats.pt"))
            
            logger.info(f"Edge feature mean: {edge_mean}")
            logger.info(f"Edge feature std: {edge_std}")
        else:
             logger.warning("No edge features found in graphs with conductivity. Skipping edge stats.")


def main():
    """Main function for running the preprocessing pipeline."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess CIF files for Li-ion conductivity prediction")
    parser.add_argument("--cif_dir", type=str, default="randomized_cifs.zip",
                        help="Directory containing CIF files or ZIP archive")
    parser.add_argument("--conductivity_file", type=str, default="extracted_conductivity.csv",
                        help="Path to CSV file with conductivity data")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                        help="Directory to save processed data")
    parser.add_argument("--cutoff_radius", type=float, default=8.0,
                        help="Cutoff radius for graph construction (Angstroms)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio of data for test set")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio of data for validation set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset splitting")
    
    args = parser.parse_args()
    
    # Create processor and run pipeline
    processor = CrystalDataProcessor(
        cif_dir=args.cif_dir,
        conductivity_file=args.conductivity_file,
        output_dir=args.output_dir,
        cutoff_radius=args.cutoff_radius
    )
    
    results = processor.run()
    
    # Print summary
    if results["status"] == "success":
        print("\n" + "="*50)
        print("Preprocessing completed successfully!")
        print(f"Total CIF files processed: {results['total_processed']}")
        print(f"Files with conductivity data: {results['with_conductivity']}")
        print(f"Files without conductivity data: {results['without_conductivity']}")
        print("\nDataset splits:")
        print(f"  Training set: {results['train_size']} files")
        print(f"  Validation set: {results['val_size']} files")
        print(f"  Test set: {results['test_size']} files")
        print("\nProcessed data is available in:")
        print(f"  {results['output_dir']}")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print(f"Preprocessing failed: {results['message']}")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()

