# -*- coding: utf-8 -*-
"""
CEGNet Model Implementation for Conductivity Prediction from CIF Files

This script implements a revised CEGNet model based on the initial version (X0CBS)
and incorporates feedback and suggestions from the code review report (VgjIt).
The goal is to predict material conductivity from Crystallographic Information
Files (CIF) with improved data handling, model architecture, robustness, and
best practices.

Changes incorporated based on code review:
1.  Robust target variable normalization (MinMax or StandardScaler) with inverse transform.
2.  Improved handling of missing/None atom features (mean/median imputation or indicator).
3.  Consideration for encoding categorical features (Group, Row) - implemented simple handling for now.
4.  Revised `CEGMessagePassing` update step to integrate old node features (`x_i`).
5.  Improved robustness for problematic CIFs (skip processing errors).
6.  Unified path management for `conductivity.csv`.
7.  Mechanism to save and load model configuration (input features, scaler params)
    for robust prediction.
8.  General code quality improvements.

Dependencies:
- Python 3.7+
- torch >= 1.8.0
- torch_geometric >= 1.7.0
- pymatgen >= 2022.0.1
- numpy >= 1.19.0
- scikit-learn >= 0.24.0
- tqdm >= 4.50.0
- pandas >= 1.0.0 (for cleaner CSV reading)

Install dependencies using:
pip install torch torch_geometric pymatgen numpy scikit-learn tqdm pandas validate-xml

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Added for activations and potential layer use
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing # Explicitly import MessagePassing
# from torch_geometric.utils import to_dense_adj # Removed, not used by MessagePassing base
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
# Use more robust methods for structure from CIF if available/needed
# from pymatgen.core.structure import Structure
# from pymatgen.analysis.local_env import CrystalNN # For more advanced graph definition potentially
import numpy as np
import random
import pandas as pd # Use pandas for CSV reading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Added MinMaxScaler
from tqdm import tqdm
import warnings
import json # To save/load config


# Suppress potential warnings from pymatgen parsing for non-standard CIFs
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress potential warnings about graph construction from pymatgen older versions or specific cases
warnings.filterwarnings("ignore", message="No structure found in", category=UserWarning)
warnings.filterwarnings("ignore", message="Could not guess element for", category=UserWarning)
warnings.filterwarnings("ignore", message="Some atoms are not bonded to any other atom", category=UserWarning)


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Configuration ---
# Define constants and hyperparameters
DATA_DIR = 'data/cif_files/'  # Directory containing CIF files and conductivity data
# CONDUCTIVITY_FILE path is handled relative to raw data directory now
PROCESSED_DIR_NAME = 'processed' # Subdirectory name for processed data
MODELS_DIR = 'models' # Directory to save models and configs
CUTOFF = 5.0  # Angstroms, cutoff distance for creating graph edges
BATCH_SIZE = 32
NUM_EPOCHS = 200 # Increased epochs as is typical
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3 # Slightly increased dropout
HIDDEN_DIM = 128
OUTPUT_DIM = 1  # Predicting a single conductivity value

# Preprocessing options
TARGET_SCALER_TYPE = 'MinMaxScaler' # Options: 'StandardScaler', 'MinMaxScaler', None
ATOM_FEATURE_MISSING_STRATEGY = 'mean_imputation' # Options: 'zero_fill', 'mean_imputation', 'median_imputation', 'indicator'
# Note: 'indicator' would require adjusting model input size and feature extraction logic

# --- Data Loading and Preprocessing (Task 7sFpg Integration & Review Feedback) ---

def load_conductivity_data(conductivity_file_path):
    """
    Loads conductivity data from a CSV file using pandas.

    Args:
        conductivity_file_path (str): Path to the CSV file. Expected format:
                                 filename,conductivity
                                 material_abc.cif,1.234

    Returns:
        dict: A dictionary mapping CIF filenames to conductivity values, or None if file not found or parse error.
    """
    if not os.path.exists(conductivity_file_path):
        print(f"Error: Conductivity file not found at {conductivity_file_path}")
        return None
    try:
        # Read CSV, assuming first row is header
        df = pd.read_csv(conductivity_file_path)
        # Basic validation
        if 'filename' not in df.columns or 'conductivity' not in df.columns:
             print(f"Error: CSV file {conductivity_file_path} must contain 'filename' and 'conductivity' columns.")
             return None

        # Convert to dictionary, handling potential errors
        conductivity_map = {}
        # Use .items() to iterate row-wise if performance is an issue, or list comprehension
        # A simple dict comprehension from the DataFrame is efficient for typical sizes
        try:
            # Filter out rows with missing values in essential columns before conversion
            df.dropna(subset=['filename', 'conductivity'], inplace=True)
            df['filename'] = df['filename'].astype(str).str.strip() # Ensure filename is string and strip whitespace
            df['conductivity'] = pd.to_numeric(df['conductivity'], errors='coerce') # Coerce errors to NaN
            df.dropna(subset=['conductivity'], inplace=True) # Drop rows where conductivity couldn't be converted

            conductivity_map = pd.Series(df.conductivity.values, index=df.filename).to_dict()

        except Exception as e:
             print(f"Error processing conductivity data in {conductivity_file_path}: {e}")
             return None

    except pd.errors.EmptyDataError:
        print(f"Warning: Conductivity file {conductivity_file_path} is empty.")
        return {} # Return empty dict for empty file
    except Exception as e:
        print(f"Error loading conductivity file {conductivity_file_path}: {e}")
        return None
    return conductivity_map

# Global scalers to be fitted on the training data
target_scaler = None
# Atom feature scalers will be handled within the dataset processing or separately

def get_atom_features(structure, feature_strategy=ATOM_FEATURE_MISSING_STRATEGY):
    """
    Extracts atom features from a pymatgen Structure object.
    Handles missing values and potentially categorical features based on strategy.
    Features included: atomic number, group, row, block, electronegativity,
                       atomic radius, ionization energy, electron affinity.

    Args:
        structure (pymatgen.core.Structure): The material structure.
        feature_strategy (str): How to handle missing/None features.

    Returns:
        torch.Tensor: Tensor of shape [num_atoms, num_features].
        list: List of indices corresponding to categorical features (e.g., Group, Row, Block).
    """
    node_features = []
    categorical_feature_indices = []
    base_features = ['Z', 'group', 'row', 'block', 'X', 'atomic_radius', 'ionization_energy', 'electron_affinity'] # Added 'block'
    categorical_feature_names = ['group', 'row', 'block']
    numerical_feature_names = [f for f in base_features if f not in categorical_feature_names]


    for site in structure:
        atom = site.specie
        features = []
        for feature_name in base_features:
            try:
                 feature_value = getattr(atom, feature_name, None)

                 if feature_name in categorical_feature_names:
                      # For categorical, store the value or a placeholder (-1 for None)
                      # Imputation/encoding applied later if needed
                      features.append(feature_value if feature_value is not None else -1)
                 else: # Numerical features
                      # For numerical, store as float, convert None to np.nan for robust handling
                      features.append(float(feature_value) if feature_value is not None else np.nan)

            except Exception as e:
                # print(f"Warning: Could not get feature {feature_name} for atom {atom}: {e}. Using None/NaN.")
                features.append(np.nan if feature_name in numerical_feature_names else -1) # Use NaN for failing numerical, -1 for failing categorical


        node_features.append(features)

    if not node_features:
        return torch.empty(0, len(base_features), dtype=torch.float), []

    # Convert list of lists to numpy array
    # Ensure all numerical columns are float type, categorical can be object/int
    node_features_np = np.array(node_features, dtype=object) # Allow mixed types
    # Convert numerical columns to float, handling NaNs
    for i, col_name in enumerate(base_features):
        if col_name in numerical_feature_names:
             try:
                 node_features_np[:, i] = node_features_np[:, i].astype(np.float64)
             except ValueError:
                 # This might happen if a supposed numerical feature has invalid data (e.g., string)
                 print(f"Warning: Could not convert numerical feature '{col_name}' to float. Setting to NaN.")
                 node_features_np[:, i] = np.nan # Ensure it becomes NaN


    # Identify indices for categorical features *in the final feature array*
    categorical_feature_indices = [base_features.index(name) for name in categorical_feature_names if name in base_features]
    numerical_feature_indices = [i for i in range(len(base_features)) if i not in categorical_feature_indices]


    # --- Imputation (Applied based on *external* stats during processing, but define logic here) ---
    # This function extracts features *per structure*. Imputation/scaling should be fitted on a collection.
    # The robust way: In CIFDataset.process(), gather all features, fit imputer/scaler, then apply.
    # For consistency during prediction or when called standalone, we *would* need fitted stats/objects.
    # As implemented in CIFDataset.process(), the stats are calculated on the fly *during* processing.
    # Here, we just return the raw array with NaNs. Imputation/scaling happens in CIFDataset.process phase 2.

    return torch.tensor(node_features_np, dtype=torch.float), categorical_feature_indices


def create_graph_from_structure(structure, cutoff):
    """
    Creates a graph representation from a pymatgen Structure object based on distance.
    Nodes are atoms, edges connect atoms within the specified cutoff distance.
    Includes basic edge validity check (e.g., self-loops).
    Edge attribute is distance.

    Args:
        structure (pymatgen.core.Structure): The material structure.
        cutoff (float): Maximum distance (in Angstroms) for edge creation.

    Returns:
        tuple: A tuple containing:
            - edge_index (torch.Tensor): Tensor of shape [2, num_edges] representing graph connectivity.
            - edge_attr (torch.Tensor): Tensor of shape [num_edges, num_edge_features] (currently distance).
            - num_edges (int): Number of edges created. Returns 0 if no edges are formed. Edge_attr dim is 1 if edges exist.
    """
    edge_index = []
    edge_attr = []
    num_atoms = len(structure)

    if num_atoms == 0:
         print("Warning: Cannot create graph for structure with 0 atoms.")
         return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 1, dtype=torch.float), 0 # Return edge_attr dim 1 expected


    # Get all neighbors within the cutoff distance
    # Suppress warnings if structure is problematic (e.g., isolated atoms etc.)
    try:
        # neighbors = structure.get_all_neighbors(cutoff) # Can be slow for large structures
        # Use alternative if available and faster, e.g., neighbor list builder
        # For simplicity, stick to get_all_neighbors for now.
        # Filter warnings as defined at the top.
        all_neighbors = structure.get_all_neighbors(cutoff)
    except Exception as e:
        print(f"Warning: Could not get neighbors for structure: {e}. Skipping graph creation.")
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 1, dtype=torch.float), 0 # Return edge_attr dim 1 expected


    for i, neighbors in enumerate(all_neighbors):
        for neighbor_data in neighbors:
            j = neighbor_data[3]  # Index of the neighbor atom
            distance = neighbor_data[1] # Distance

            # Avoid self-loops and duplicate edges (for undirected graph, add one direction is enough for PyG)
            # BUT for message passing symmetry, explicitly adding i->j and j->i is clearer
            # Let's stick to adding both directions (i,j) and (j,i) to edge_index, attr
            # Ensure i < j to avoid adding the pair twice if only one direction was desired implicitly.
            # However, for standard MP layers, explicit bidirectional edges are standard.

            if i != j:
                edge_index.append([i, j])
                edge_attr.append([distance])
                # No need for j->i if the GNN layer handles it with `flow="source_to_target"` and symmetry.
                # But the original CEGNet concept might imply directed messages. Let's add both for robustness against layer implementations.
                # Adding both directions:
                edge_index.append([j, i]) # Add reverse edge
                edge_attr.append([distance])


    # Convert to tensors AFTER adding both directions if needed
    # Use .t().contiguous() for edge_index [2, num_edges] format
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty(2, 0, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty(0, 1, dtype=torch.float) # Ensure edge_attr has 1 column even if empty

    num_edges = len(edge_index) if edge_index else 0

    if num_edges == 0 and num_atoms > 0:
         print(f"\nWarning: No edges created for structure with {num_atoms} atoms within cutoff {cutoff}.")

    return edge_index_t, edge_attr_t, num_edges


class CIFDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset for CIF files, integrating preprocessing
    and robust handling of conductivity data and processing errors.
    """
    def __init__(self, root, cutoff=CUTOFF, transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory where the dataset files are stored.
                        Expected structure:
                        root/raw/ -> contains .cif and conductivity.csv
                        root/processed/ -> preprocessed Data objects are saved here
            cutoff (float): Distance cutoff for graph edge creation.
        """
        self.root = root
        self.raw_data_dir = os.path.join(root, 'raw')
        self.processed_data_dir = os.path.join(root, PROCESSED_DIR_NAME)
        self.conductivity_file_path = os.path.join(self.raw_data_dir, 'conductivity.csv')
        self.cutoff = cutoff

        # Global scalers/imputers fitted during process()
        # Need to store these or their parameters to apply consistently
        self.target_scaler = None
        self.atom_imputation_stats = {} # e.g., {'feature_name': mean_value}
        self.atom_scaler = None # If applying atom feature scaling later

        # --- Data Loading and Preprocessing ---
        # Load conductivity data immediately
        self.conductivity_map = load_conductivity_data(self.conductivity_file_path)
        if self.conductivity_map is None:
             # If conductivity data couldn't be loaded, the dataset is effectively empty for training purposes
             self._raw_file_names = []
             print("Failed to load conductivity data. Dataset will be empty.")
        else:
            # Filter raw files based on available conductivity data AND actual file existence
            if not os.path.exists(self.raw_data_dir):
                 print(f"Raw data directory not found: {self.raw_data_dir}. Dataset will be empty.")
                 self._raw_file_names = []
            else:
                 self._raw_file_names = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.cif') and f in self.conductivity_map]


        # Check if processed data exists and is complete before calling super
        processed_list_path = os.path.join(self.processed_dir, 'processed_file_names.txt')
        processed_files_exist = os.path.exists(processed_list_path)
        expected_processed_count = len(self._raw_file_names) # Files with CIF and conductivity
        current_processed_count = 0
        if processed_files_exist:
             try:
                  with open(processed_list_path, 'r') as f:
                       current_processed_files_list = [line.strip() for line in f if line.strip()]
                       current_processed_count = len(current_processed_files_list)
                  # Basic check: do the actual .pt files exist for these names?
                  valid_processed_count = sum(os.path.exists(os.path.join(self.processed_dir, fname)) for fname in current_processed_files_list)
                  if valid_processed_count != current_processed_count:
                       print(f"Warning: processed_file_names.txt lists {current_processed_count} files, but only {valid_processed_count} .pt files exist. Re-processing.")
                       processed_files_exist = False # Force re-process
                  elif current_processed_count != expected_processed_count:
                       print(f"Processed data found ({current_processed_count}), but input count changed ({expected_processed_count}). Re-processing.")
                       processed_files_exist = False # Force re-process
                  else:
                        # Check if any processing options changed (e.g. cutoff, strategy, scaler type)
                        # This would require saving preprocessing config alongside processed files
                        # Simple check for now: if the list matches expected.
                        print("Processed data seems complete and matches raw files.")
                        # Load scaler parameters if processed data exists
                        self.target_scaler, _ = load_scaler_params(root) # Load from root level (models dir in main_train)


             except Exception as e:
                  print(f"Error checking processed data status: {e}. Re-processing.")
                  processed_files_exist = False # Force re-process

        if processed_files_exist and current_processed_count > 0:
             # If processed files exist and seem valid/complete, bypass the process() method
             print(f"Found {current_processed_count} processed files. Loading existing data.")
             # The self.processed_file_names property will load the list from the file automatically after super() __init__

            #  Need to ensure self.processed_file_names gets populated even if process is skipped.
            #  The super().__init__() calls process() only if needed. We need the list regardless.
            #  Let's load the list here if we are skipping process.
             if os.path.exists(processed_list_path):
                 with open(processed_list_path, 'r') as f:
                      self.__processed_file_names = [line.strip() for line in f if line.strip()] # Use private name to avoid property reload loop


        # Initialize super class - this might call process() if not skipped
        super(CIFDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw') # Ensure raw_dir is based on the dataset root

    @property
    def processed_dir(self):
        return os.path.join(self.root, PROCESSED_DIR_NAME) # Ensure processed_dir is based on the dataset root


    @property
    def raw_file_names(self):
        """List of raw CIF files that have corresponding conductivity data and exist."""
        # This list is filtered and populated in __init__
        return self._raw_file_names

    @property
    def processed_file_names(self):
        """List of relative paths of processed files corresponding to successful CIF processing."""
        # This file lists the .pt filenames
        processed_list_path = os.path.join(self.processed_dir, 'processed_file_names.txt')
        if hasattr(self, '__processed_file_names'):
             # If loaded in __init__ when skipping process()
             return self.__processed_file_names
        elif os.path.exists(processed_list_path):
            # Load the list upon first access if process() was called or needed
            with open(processed_list_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
             return [] # Return empty if list file doesn't exist

    def process(self):
        """Processes raw CIF files into PyTorch Geometric Data objects."""
        print("\n--- Starting Data Processing ---")
        processed_files = [] # List to store names of successfully processed files
        processed_list_path = os.path.join(self.processed_dir, 'processed_file_names.txt')

        # Clean up processed directory if re-processing (optional but good practice)
        # print(f"Clearing processed directory: {self.processed_dir}")
        # if os.path.exists(self.processed_dir):
        #     import shutil
        #     shutil.rmtree(self.processed_dir)
        # os.makedirs(self.processed_dir)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if not self.conductivity_map: # Check if map is empty after loading
             print("No conductivity data loaded or available for raw files. Processing skipped.")
             # Save an empty list file to indicate attempted processing resulted in no data
             with open(processed_list_path, 'w') as f:
                  pass # Write nothing
             self.__processed_file_names = [] # Update internal list
             return

        # Store all extracted features and targets temporarily to fit scalers/imputers
        all_node_features_list = [] # List of numpy arrays, one per structure
        all_targets_list = [] # List of scalar target values

        print(f"Found {len(self.raw_file_names)} raw CIF files with conductivity data.")
        print("First pass: Extracting features and targets for transformation fitting...")
        temp_data_map = {} # Map raw_file_name to extracted data before transformation


        for raw_file in tqdm(self.raw_file_names, desc="Step 1/2: Initial Data Extraction"):
            cif_filepath = os.path.join(self.raw_dir, raw_file)
            # filename_base = os.path.splitext(raw_file)[0] # Not needed here
            conductivity = self.conductivity_map.get(raw_file)

            try:
                # Robustly get structure
                structure = None
                try:
                     parser = CifParser(cif_filepath)
                     structures = parser.get_structures(primitive=False)
                     if structures:
                          structure = structures[0] # Get the first structure found
                except Exception as struct_e:
                     # print(f"\nWarning: Could not get structure from {raw_file}: {struct_e}. Skipping.")
                     continue # Skip to the next file


                if structure is None or len(structure) == 0:
                     # print(f"\nWarning: Structure from {raw_file} is empty or invalid. Skipping.")
                     continue

                # Extract raw node features (with NaNs for numerical None, -1 for categorical None)
                x_raw_tensor, categorical_indices = get_atom_features(structure)
                x_raw_np = x_raw_tensor.numpy() # Convert back to numpy for easier manipulation in next steps

                # Ensure node feature dimension is consistent even for empty structures (should be 0 nodes with N dim)
                if x_raw_np.shape[1] == 0 and x_raw_np.shape[0] > 0:
                     # This shouldn't really happen if base_features is not empty, but safety check
                     print(f"Warning: Extracted 0 features for atoms in {raw_file}.")
                     continue


                edge_index, edge_attr, num_edges = create_graph_from_structure(structure, self.cutoff)


                y_raw_tensor = torch.tensor([conductivity], dtype=torch.float) # Conductivity is guaranteed here

                # Store raw features and target for transformation fitting
                if x_raw_np.shape[0] > 0: # Only consider structures with at least one atom
                     all_node_features_list.append(x_raw_np)
                     # Store targets for scaler fitting
                     all_targets_list.append(y_raw_tensor.item())

                     # Store intermediate results mapped by raw_file for the second pass
                     temp_data_map[raw_file] = {
                          'x_raw_np': x_raw_np,
                          'edge_index': edge_index,
                          'edge_attr': edge_attr,
                          'y_raw_tensor': y_raw_tensor
                     }


            except Exception as e:
                print(f"\nError processing CIF file {raw_file}: {e}. Skipping.")
                # Ensure the file is NOT added to processed_files list
                continue # Skip to the next file

        # Check if any valid data was extracted
        if not all_targets_list:
             print("No valid data extracted after first pass. Cannot fit transformations or process data.")
             with open(processed_list_path, 'w') as f: pass # Save empty list
             self.__processed_file_names = [] # Update internal list
             return

        # --- Fit Transformations ---
        # Fit Target Scaler
        self.target_scaler = None # Reset or initialize
        all_targets_np = np.array(all_targets_list).reshape(-1, 1)
        if TARGET_SCALER_TYPE == 'StandardScaler':
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(all_targets_np)
            print("Fitted StandardScaler for targets.")
        elif TARGET_SCALER_TYPE == 'MinMaxScaler':
             self.target_scaler = MinMaxScaler()
             self.target_scaler.fit(all_targets_np)
             print("Fitted MinMaxScaler for targets.")
        else:
            print("Target scaling is disabled.")

        # Fit Atom Feature Imputer (and potential Scaler)
        # Combine all extracted raw node features into a single numpy array/pandas DataFrame
        # Need to handle potential empty arrays in all_node_features_list
        combined_features_list = [arr for arr in all_node_features_list if arr.size > 0]
        if combined_features_list:
             # Assuming all arrays in the list have the same number of columns (checked in get_atom_features)
             combined_features_np = np.vstack(combined_features_list)

             # Use pandas DataFrame to handle mixed types and NaNs for imputation
             base_features = ['Z', 'group', 'row', 'block', 'X', 'atomic_radius', 'ionization_energy', 'electron_affinity']
             categorical_feature_names = ['group', 'row', 'block']
             numerical_feature_names = [f for f in base_features if f not in categorical_feature_names]

             combined_features_df = pd.DataFrame(combined_features_np, columns=base_features)

             self.atom_imputation_stats = {}
             if ATOM_FEATURE_MISSING_STRATEGY in ['mean_imputation', 'median_imputation']:
                 for col_name in numerical_feature_names:
                      # Check for and fill NaNs in numerical columns
                      if combined_features_df[col_name].isnull().any():
                           imputation_value = combined_features_df[col_name].mean() if ATOM_FEATURE_MISSING_STRATEGY == 'mean_imputation' else combined_features_df[col_name].median()
                           self.atom_imputation_stats[col_name] = imputation_value
                 print(f"Calculated imputation stats for numerical atom features based on '{ATOM_FEATURE_MISSING_STRATEGY}'.")
                 # Note: Imputation itself is applied in the second pass below.

             elif ATOM_FEATURE_MISSING_STRATEGY == 'zero_fill':
                  # In this case, NaNs for numerical features are replaced by 0.0
                  print("Atom feature missing strategy is 'zero_fill'. NaNs will be replaced by 0.0.")
                  # No stats to save for zero fill, the logic is direct.

             # TODO: Fit Atom Feature Scaler here if needed (e.g., StandardScaler on numerical features)
             # self.atom_scaler = StandardScaler().fit(combined_features_df[numerical_feature_names].dropna()) # Example


        else:
             print("No node features collected from valid structures. Cannot fit atom feature transformations.")


        # Save transformation parameters (target scaler, atom imputation stats)
        # These should be saved alongside the model in the models directory
        save_location_for_params = os.path.join(self.root, '..', MODELS_DIR) # Assume models_dir is relative to data_root
                                                                            # This needs adjustment if MODELS_DIR is absolute or different relative structure
        # Let's save to the root level of the dataset for simplicity, assuming model loading knows where to look
        # Save scaler params to self.root as this is where the dataset lives during processing.
        # The main_train function will then move or copy them to the MODELS_DIR.
        # A cleaner approach: pass the models_dir path to the Dataset process method.
        # For now, save to self.root and rely on main_train to handle saving/loading from correct place.
        # Let's adjust main_train to pass models_dir path.

        # Let's pass models_dir to the Dataset's process method.

    # Need to remove the process method from here and call it externally from main_train, passing models_dir.
    # Or allow process to accept models_dir path. Let's pass models_dir.

    # Refactor: process method now accepts models_dir path
    def process(self, models_dir):
        """Processes raw CIF files into PyTorch Geometric Data objects and saves transformation params."""
        print("\n--- Starting Data Processing ---")
        processed_files = [] # List to store names of successfully processed files
        processed_list_path = os.path.join(self.processed_dir, 'processed_file_names.txt')

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)


        if not self.conductivity_map: # Check if map is empty after loading
             print("No conductivity data loaded or available for raw files. Processing skipped.")
             with open(processed_list_path, 'w') as f: pass # Save empty list
             self.__processed_file_names = [] # Update internal list
             return

        # Store all extracted features and targets temporarily to fit scalers/imputers
        all_node_features_list = [] # List of numpy arrays, one per structure
        all_targets_list = [] # List of scalar target values

        print(f"Found {len(self.raw_file_names)} raw CIF files with conductivity data.")
        print("First pass: Extracting features and targets for transformation fitting...")
        temp_data_map = {} # Map raw_file_name to extracted data before transformation

        base_features = ['Z', 'group', 'row', 'block', 'X', 'atomic_radius', 'ionization_energy', 'electron_affinity']
        categorical_feature_names = ['group', 'row', 'block']
        numerical_feature_names = [f for f in base_features if f not in categorical_feature_names]


        for raw_file in tqdm(self.raw_file_names, desc="Step 1/2: Initial Data Extraction"):
            cif_filepath = os.path.join(self.raw_dir, raw_file)
            conductivity = self.conductivity_map.get(raw_file)

            try:
                # Extract raw node features (with NaNs for numerical None, -1 for categorical None)
                # Re-call get_atom_features to ensure potentially refined logic runs
                parser = CifParser(cif_filepath)
                structures = parser.get_structures(primitive=False)
                if not structures:
                     # print(f"\nWarning: Could not get structure from {raw_file}. Skipping.")
                     continue
                structure = structures[0] # Get the first structure found

                if len(structure) == 0:
                     # print(f"\nWarning: Structure from {raw_file} is empty. Skipping.")
                     continue

                x_raw_tensor, categorical_indices = get_atom_features(structure)
                x_raw_np = x_raw_tensor.numpy()


                # Store raw features and target for transformation fitting
                if x_raw_np.shape[0] > 0: # Only consider structures with at least one atom
                     all_node_features_list.append(x_raw_np)
                     all_targets_list.append(conductivity) # store conductivity as scalar


                     # Store intermediate results for the second pass
                     # Do not create graph here yet, wait until second pass after deciding what to do with 0-edge graphs
                     # Let's create graph here and store it, simpler.
                     edge_index, edge_attr, num_edges = create_graph_from_structure(structure, self.cutoff)

                     temp_data_map[raw_file] = {
                          'x_raw_np': x_raw_np,
                          'edge_index': edge_index,
                          'edge_attr': edge_attr,
                          'y_raw_tensor': torch.tensor([conductivity], dtype=torch.float)
                     }


            except Exception as e:
                print(f"\nError during initial processing of {raw_file}: {e}. Skipping.")
                continue # Skip to the next file


        # Check if any valid data was extracted
        if not all_targets_list:
             print("No valid data extracted after first pass. Cannot fit transformations or process data.")
             with open(processed_list_path, 'w') as f: pass # Save empty list
             self.__processed_file_names = [] # Update internal list
             return

        # --- Fit Transformations ---
        # Fit Target Scaler
        self.target_scaler = None
        all_targets_np = np.array(all_targets_list).reshape(-1, 1)
        if TARGET_SCALER_TYPE == 'StandardScaler':
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(all_targets_np)
            print("Fitted StandardScaler for targets.")
        elif TARGET_SCALER_TYPE == 'MinMaxScaler':
             self.target_scaler = MinMaxScaler()
             self.target_scaler.fit(all_targets_np)
             print("Fitted MinMaxScaler for targets.")
        else:
            print("Target scaling is disabled.")

        # Fit Atom Feature Imputer (and potential Scaler)
        # Combine all extracted raw node features into a single numpy array/pandas DataFrame
        combined_features_list = [arr for arr in all_node_features_list if arr.size > 0] # Filter out empty arrays if any
        if combined_features_list:
             combined_features_np = np.vstack(combined_features_list)
             combined_features_df = pd.DataFrame(combined_features_np, columns=base_features)

             self.atom_imputation_stats = {}
             if ATOM_FEATURE_MISSING_STRATEGY in ['mean_imputation', 'median_imputation']:
                 for col_name in numerical_feature_names:
                      if combined_features_df[col_name].isnull().any():
                           imputation_value = combined_features_df[col_name].mean() if ATOM_FEATURE_MISSING_STRATEGY == 'mean_imputation' else combined_features_df[col_name].median()
                           self.atom_imputation_stats[col_name] = float(imputation_value) # Store as float
                 print(f"Calculated imputation stats for numerical atom features.")

             elif ATOM_FEATURE_MISSING_STRATEGY == 'zero_fill':
                  print("Atom feature missing strategy is 'zero_fill'.")
                  # No stats to save, the logic is direct replacement of NaN with 0.0

             # TODO: Implement and fit Atom Feature Scaler here if needed

        else:
             print("No node features collected from valid structures. Cannot fit atom feature transformations.")
             # Still proceed to save processed data if targets were collected (e.g. 0-atom materials with known conductivity, though unlikely)
             # But if no nodes or targets, the dataset is truly empty.

        # Save transformation parameters (target scaler, atom imputation stats)
        # Save scaler params to the models_dir
        save_scaler_params(models_dir, self.target_scaler, TARGET_SCALER_TYPE)
        # Save atom imputation stats
        atom_imputation_stats_path = os.path.join(models_dir, 'atom_imputation_stats.json')
        with open(atom_imputation_stats_path, 'w') as f:
             json.dump(self.atom_imputation_stats, f, indent=4)
        print(f"Saved atom imputation stats to {atom_imputation_stats_path}")

        # --- Second pass: Apply Transformations and Save Data Objects ---
        print("Second pass: Applying transformations and saving Data objects...")
        successful_files_for_data_object = []

        for raw_file, results in tqdm(temp_data_map.items(), desc="Step 2/2: Applying transformations and saving"):
            filename_base = os.path.splitext(raw_file)[0]
            processed_filename = f'{filename_base}.pt'
            processed_filepath = os.path.join(self.processed_dir, processed_filename)

            x_raw_np = results['x_raw_np']
            edge_index = results['edge_index']
            edge_attr = results['edge_attr']
            y_raw_tensor = results['y_raw_tensor']


            # 1. Apply Imputation to numerical atom features
            x_processed_np = np.copy(x_raw_np) # Work on a copy
            col_names = ['Z', 'group', 'row', 'block', 'X', 'atomic_radius', 'ionization_energy', 'electron_affinity'] # Need feature names again

            for i, col_name in enumerate(col_names):
                 if col_name in numerical_feature_names:
                      # Replace NaNs based on strategy and calculated stats
                      if ATOM_FEATURE_MISSING_STRATEGY == 'zero_fill':
                           imputation_value = 0.0
                      elif ATOM_FEATURE_MISSING_STRATEGY in ['mean_imputation', 'median_imputation']:
                           imputation_value = self.atom_imputation_stats.get(col_name, 0.0) # Use calculated stats, fallback to 0 if no stats or feature missing

                      # Apply imputation to numerical columns with NaNs
                      x_processed_np[:, i] = np.nan_to_num(x_processed_np[:, i], nan=imputation_value)

                 # Categorical features are left as is for now (-1 for None)
                 # If One-Hot or Embedding was needed, it would be applied here.


            # TODO: Apply Atom Feature Scaling here if implemented

            x_processed_tensor = torch.tensor(x_processed_np, dtype=torch.float)


            # 2. Scale Target Value
            y_scaled_tensor = y_raw_tensor
            if self.target_scaler is not None:
                 try:
                      y_scaled_np = self.target_scaler.transform(y_raw_tensor.numpy().reshape(-1, 1))
                      y_scaled_tensor = torch.tensor(y_scaled_np, dtype=torch.float)
                 except Exception as e:
                      print(f"\nError scaling target for {raw_file}: {e}. Skipping.")
                      continue # Skip saving this sample if target scaling fails


            # Create PyTorch Geometric Data object
            # Ensure edge_attr has 1 column even if edge_index is empty
            if edge_attr is None or edge_attr.size(1) == 0:
                # If create_graph returned an empty edge_attr or shape issue, create a dummy one if edges exist
                 if edge_index is not None and edge_index.size(1) > 0:
                      # Edges exist, but no attribute? Use a placeholder maybe? Or just an empty tensor with correct dimension?
                      # Let's ensure create_graph always returns edge_attr with 1 column for distance if edges exist.
                      # If edges_exist but edge_attr has 0 size, re-check create_graph.
                      # Assuming create_graph returns torch.empty(0, 1) if no edges, and [E, 1] if edges exist.
                       pass # Use the edge_attr returned by create_graph


            data = Data(x=x_processed_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_scaled_tensor)

            # Save the processed data object
            try:
                 torch.save(data, processed_filepath)
                 successful_files_for_data_object.append(processed_filename)
            except Exception as e:
                 print(f"\nError saving processed data for {raw_file} to {processed_filepath}: {e}. Skipping.")
                 continue


        # Save the list of successfully processed files for quick loading later
        with open(processed_list_path, 'w') as f:
             for fname in successful_files_for_data_object:
                  f.write(f"{fname}\n")

        self.__processed_file_names = successful_files_for_data_object # Update internal list
        print(f"Successfully processed and saved {len(self.__processed_file_names)} valid data objects.")

        # After processing successfully, update the list used by the property
        # self.processed_file_names = successful_files_for_data_object # This would call the setter if one exists. Direct access needed?
        # The property will read the file on next access, which is fine.

    def len(self):
        """Returns the number of processed samples."""
        return len(self.processed_file_names) # Uses the property, reads from file if not cached

    def get(self, idx):
        """Loads and returns a single processed Data object by index."""
        processed_filename = self.processed_file_names[idx]
        data = torch.load(os.path.join(self.processed_dir, processed_filename))
        return data

    def get_node_feature_dims(self):
         """Helper to get input feature dimensions after processing."""
         if len(self.processed_file_names) > 0:
              try:
                   example_data = self.get(0) # Load first processed sample
                   node_dim = example_data.x.size(1) if example_data.x is not None and example_data.x.numel() > 0 else 0
                   edge_dim = example_data.edge_attr.size(1) if example_data.edge_attr is not None and example_data.edge_attr.numel() > 0 else 0
                   if edge_dim == 0 and example_data.edge_index is not None and example_data.edge_index.size(1) > 0:
                        # Case where edges exist but edge_attr was somehow saved empty
                        print("Warning: Detected edges but 0-dimensional edge_attr in processed data. Assuming edge_in_features = 1.")
                        edge_dim = 1 # Assume 1 for distance if edges exist

                   return node_dim, edge_dim
              except Exception as e:
                   print(f"Error loading first processed sample to determine feature dimensions: {e}")
                   return 0, 0 # Fallback if first sample fails to load/has issues
         else:
              print("Dataset is empty. Cannot determine feature dimensions.")
              return 0, 0

    def get_transformers(self):
         """Returns the fitted target scaler and atom imputation stats/scaler."""
         # Note: Atom imputation stats are loaded during the second processing pass or when load_atom_imputation_stats is called.
         # For prediction, load_atom_imputation_stats needs to be called separately using the models_dir.
         # This dataset object is mainly for training data loading. Prediction uses separate functions.
         # This method is perhaps not needed within the Dataset itself.
         # Return None or raise error if called during training or prediction workflow isn't through dataset object.
         # Let's rely on loading functions in the prediction script.
         pass


def load_atom_imputation_stats(models_dir):
     """Loads atom feature imputation statistics."""
     stats_path = os.path.join(models_dir, 'atom_imputation_stats.json')
     if not os.path.exists(stats_path):
          print(f"Error: Atom imputation stats file not found at {stats_path}")
          return {} # Return empty dict
     try:
          with open(stats_path, 'r') as f:
               stats = json.load(f)
          print(f"Loaded atom imputation stats from {stats_path}")
          return stats
     except Exception as e:
          print(f"Error loading atom imputation stats from {stats_path}: {e}")
          return {}


def apply_atom_feature_transformation(x_raw_np, base_features, numerical_feature_names, imputation_stats, missing_strategy='zero_fill'):
     """
     Applies imputation (and potentially scaling if implemented) to raw atom features numpy array.
     This logic should be consistent with CIFDataset.process() second pass.
     """
     x_processed_np = np.copy(x_raw_np)

     for i, col_name in enumerate(base_features):
          if col_name in numerical_feature_names:
               # Apply imputation based on strategy and provided stats
               imputation_value = 0.0 # Default for zero_fill or if stats missing
               if missing_strategy == 'zero_fill':
                    imputation_value = 0.0
               elif missing_strategy in ['mean_imputation', 'median_imputation']:
                    # Get from loaded stats; use 0.0 fallback if stat not found
                    imputation_value = imputation_stats.get(col_name, 0.0)

               # Apply imputation to numerical columns with NaNs
               x_processed_np[:, i] = np.nan_to_num(x_processed_np[:, i], nan=imputation_value)

          # Categorical features left as is (-1 for None)

     # TODO: Apply Atom Feature Scaling here if implemented and corresponding scaler is loaded

     return torch.tensor(x_processed_np, dtype=torch.float)


# --- CEGNet Model Definition (Task fjM37 Implementation & Review Feedback) ---

from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential, Linear, ReLU, Dropout

class CEGMessagePassing(MessagePassing):
    """
    CEGNet Message Passing Layer with improved Update Function.
    Combines sender node features and edge features to create messages.
    Aggregates messages and updates receiver node features using both old features and aggregated message.
    """
    def __init__(self, node_in_channels, edge_in_channels, out_channels):
        # aggr='add' for simple summation
        super(CEGMessagePassing, self).__init__(aggr='add')
        self.node_in_channels = node_in_channels # Store for update fn dimension check
        self.edge_in_channels = edge_in_channels # 存储边特征维度
        self.out_channels = out_channels # Output dimension of this layer

        # Check edge_in_channels. If 0, this layer might be incompatible or need bypass/adjustment.
        if edge_in_channels <= 0:
             print(f"Warning: CEGMessagePassing initialized with edge_in_channels={edge_in_channels}. Edge features will not be used.")
             # Adjust message MLP input if edge_in_channels is 0
             message_mlp_input_dim = out_channels # Only processed_x_j
             # Initialize self.edge_lin but don't use it in message
             self.edge_lin = None # Signal that edge features are not used
        else:
            message_mlp_input_dim = out_channels + out_channels # processed_x_j + processed_edge_attr
            self.edge_lin = Linear(edge_in_channels, out_channels)


        # MLP for processing sender node features and edge features combined for MESSAGE generation
        # Process sender node features individually first
        self.sender_node_lin = Linear(node_in_channels, out_channels)


        # MLP for the message function (takes concatenated features)
        self.message_mlp = Sequential(
            Linear(message_mlp_input_dim, out_channels), # Input is processed_x_j + processed_edge_attr (or just processed_x_j if edge_in_channels=0)
            ReLU()
            # Add dropout? Usually in the main graph layers.
        )

        # MLP for the final UPDATE function (takes old node feature and aggregated message)
        # Input to update MLP is concatenation of original node feature (transformed) and aggregated message
        # The original node feature dimension is node_in_channels. Aggregated message dim is out_channels.
        # Transform original node feature x_i first
        self.update_x_lin = Linear(node_in_channels, out_channels) # Project original x_i to match agg_msg dim

        self.update_mlp = Sequential(
             # Updated input dimension: transformed original x_i (out_channels) + aggregated message (out_channels)
             Linear(out_channels + out_channels, out_channels),
             ReLU()
             # Final activation or dropout applied by the main CEGNet module typically
        )


    def forward(self, x, edge_index, edge_attr):
        # x: Initial node features [N, node_in_channels]
        # edge_index: [2, E]
        # edge_attr: [E, edge_in_channels]
        # batch: [N] (implicitly handled by PyG and propagate)

        # If edge_attr is None or empty despite edge_in_channels > 0, handle gracefully
        # This can happen for graphs with nodes but no edges.
        if edge_attr is None or edge_attr.numel() == 0:
             if self.edge_in_channels > 0:
                  # If edge features are expected but not provided for this graph, create dummy zero edge_attr
                  # This requires knowing the number of edges in this specific graph
                  num_edges_in_batch = edge_index.size(1) if edge_index is not None else 0
                  if num_edges_in_batch > 0:
                       # Create zero tensor with expected dimensions (num_edges, self.edge_in_channels)
                       edge_attr = torch.zeros(num_edges_in_batch, self.edge_in_channels, device=x.device)
                       print(f"Warning: Missing edge_attr for batch with {num_edges_in_batch} edges. Using zero edge_attr.")
                  else:
                       # No edges, no edge_attr needed.
                       edge_attr = None # Set explicit None if no edges


        # Need to ensure propagate handles edge_attr=None if edge_in_channels > 0.
        # The message function accesses edge_attr. If edge_in_channels > 0 but edge_attr is None/empty, propagate will fail.
        # Need to pass edge_attr to propagate only if it's not None AND self.edge_in_channels > 0.
        # Or handle the edge_lin call inside message() based on self.edge_lin being None.

        if self.edge_in_channels > 0 and (edge_attr is None or edge_attr.numel() == 0):
             # This case should ideally be handled by creating zero edge_attr above if edges exist.
             # If edges exist but edge_attr is None/Empty Tensor, it's an issue upstream or in DataLoader batching.
             # Let's add a check to ensure message is only called with non-empty edge_attr if req'd.
             # The propagate call itself needs x, edge_index. Passing edge_attr is optional.

             # If edge_attr is None or empty, and edge_in_channels > 0, propagate will likely fail
             # Let's re-ensure edge_attr is a zero tensor of correct size if edges exist
             num_edges_in_batch = edge_index.size(1) if edge_index is not None else 0
             if num_edges_in_batch > 0 and (edge_attr is None or edge_attr.numel() == 0):
                 edge_attr = torch.zeros(num_edges_in_batch, self.edge_in_channels, device=x.device)
                 # print(f"Debug: Created zero edge_attr {edge_attr.shape} for {num_edges_in_batch} edges.")
             elif num_edges_in_batch == 0:
                  # No edges, edge_attr should be empty or None. The message function won't be called via propagate.
                  pass # OK


        # Start propagation.
        # Pass edge_attr ONLY if self.edge_in_channels > 0, otherwise propagate will fail if message expects it.
        if self.edge_in_channels > 0:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        else:
             # If edge features are not used, propagate without passing edge_attr.
             # The message function signature must not expect edge_attr in this case, or handle None.
             # As written, message() expects edge_attr, so edge_in_channels > 0 is implicitly required.
             # Need to adjust message signature or logic if edge_in_channels=0 is a valid scenario.
             # Given current message() signature `message(self, x_i, x_j, edge_attr)`, edge_attr must always be passed by propagate.
             # If edge_in_channels=0, edge_attr will have shape [E, 0].
             # The message MLP input dim handles this if edge_in_channels=0 was set correctly in __init__.
             # So, pass edge_attr always, ensuring it's correctly shaped [E, edge_in_channels].
             # The zero-filling above ensures edge_attr is not None if edge_in_channels > 0 and edges exist.
             return self.propagate(edge_index, x=x, edge_attr=edge_attr) # Confident edge_attr is correctly shaped



    def message(self, x_i, x_j, edge_attr):
        # x_i: Features of receiving nodes [E, node_in_channels]
        # x_j: Features of sending nodes [E, node_in_channels]
        # edge_attr: Edge features [E, edge_in_channels]

        # Process sender node features
        processed_x_j = self.sender_node_lin(x_j)

        # Process edge features ONLY IF edge_in_channels > 0
        if self.edge_lin is not None: # Check if edge_lin was initialized (i.e. edge_in_channels > 0)
             processed_edge_attr = self.edge_lin(edge_attr)
             # Combine processed sender node features and edge features
             message = torch.cat([processed_x_j, processed_edge_attr], dim=-1)
        else:
             # If no edge features, message is just processed sender node features
             if edge_attr is not None and edge_attr.size(1) > 0:
                  # This case suggests edge_in_channels was 0 but edge_attr was somehow non-empty?
                  # This shouldn't happen with the current logic, but safety check.
                  # If it happens, using only processed_x_j is the fallback based on edge_in_channels=0.
                  print("Warning: edge_in_channels is 0 but got non-empty edge_attr in message(). Ignoring edge_attr.")

             message = processed_x_j # Use only processed node features


        message = self.message_mlp(message) # Apply MLP and ReLU

        return message # message for edge (j -> i)

    # aggregate method is 'add' by default in __init__

    def update(self, agg_msg, x):
        # agg_msg: aggregated messages for each node [N, out_channels] (Output of propagation including aggregation)
        # x: Original node features passed to propagate method [N, node_in_channels]

        # Process original node features to match dimension of aggregated message
        processed_x = self.update_x_lin(x)
        processed_x = F.relu(processed_x) # Apply activation

        # Combine original node features (processed) and aggregated message
        updated_node_features = torch.cat([processed_x, agg_msg], dim=-1)

        # Apply final update MLP and activation
        updated_node_features = self.update_mlp(updated_node_features)

        return updated_node_features # Updated node features [N, out_channels]



class CEGNet(nn.Module):
    """
    Core CEGNet Architecture for Conductivity Prediction.
    Consists of CEGMessagePassing layers and global pooling followed by MLPs.
    Incorporates Dropout after activations.
    """
    def __init__(self, node_in_features, edge_in_features, hidden_dim, output_dim, dropout_rate):
        super(CEGNet, self).__init__()
        self.node_in_features = node_in_features
        self.edge_in_features = edge_in_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # If node_in_features is 0, this model won't work.
        if node_in_features <= 0:
             raise ValueError(f"node_in_features must be > 0, but got {node_in_features}")
        # If edge_in_features is 0, the CEGMessagePassing layer can still be initialized,
        # but edge features won't be used in messages. This is handled in CEGMessagePassing.

        # Initial linear layer to potentially handle categorical features or project general features
        # Not strictly needed if first CEGMessagePassing layer handles this in sender_node_lin.
        # Let's rely on CEGMessagePassing's internal linear layers.

        # CEG Message Passing Layers
        # Ensure input/output dimensions flow correctly between layers
        # First layer takes original features (node_in_features, edge_in_features) and outputs hidden_dim
        self.conv1 = CEGMessagePassing(node_in_features, edge_in_features, hidden_dim)
        # Subsequent layers take hidden_dim from the previous layer and edge_in_features again
        self.conv2 = CEGMessagePassing(hidden_dim, edge_in_features, hidden_dim)
        self.conv3 = CEGMessagePassing(hidden_dim, edge_in_features, hidden_dim)


        # Global pooling layer
        self.pool = global_mean_pool

        # Readout network (MLP)
        # Input to FC1 is hidden_dim from the pooling layer
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, output_dim)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Handle case of data with no atoms or edges gracefully
        if x.size(0) == 0:
            # Return a default value (e.g., 0 conductivity) if the graph is empty
            # print("Warning: Graph with 0 nodes encountered during forward pass. Returning 0 conductivity.")
            return torch.zeros(data.num_graphs if data.batch is not None else 1, self.output_dim, device=x.device) # Ensure batch dimension is handled

        # Ensure edge_attr is not None if required by MessagePassing layers
        # The CEGMessagePassing forward handles the case where edge_in_channels > 0 but edge_attr is logically empty.
        # But PyG DataLoader might batch empty edge_attrs as None if no edges in batch.
        # Need to ensure edge_attr is a tensor, even if empty, if any Layer expects it.
        # CEGMessagePassing's message function explicitly uses `edge_attr`.

        # Handle case where a batch might have ZERO edges, causing edge_index/edge_attr to be empty tensors.
        # PyG's DataLoader _might_ return edge_attr as None if batch size > 1 and one graph has no edges? Needs check.
        # Safest: ensure edge_attr is always a tensor with correct dimension [E, edge_in_features].
        # If the batch has 0 edges, edge_index is [2, 0]. edge_attr should be [0, edge_in_features].

        # No need for explicit check here; if create_graph handles 0 edges correctly (empty tensor [0, dim]),
        # and DataLoader concatenates empty tensors correctly, this is fine.
        # The CEGMessagePassing forward contains the check for `edge_attr is None or edge_attr.numel() == 0`.

        # Pass through GNN layers
        # Apply ReLU and Dropout *after* each convolution layer
        # The ReLU is part of CEGMessagePassing's update_mlp, so it's already applied within the layer.
        # Apply Dropout AFTER the layer module call.
        x = self.conv1(x, edge_index, edge_attr)
        x = self.dropout(x) 

        x = self.conv2(x, edge_index, edge_attr)
        x = self.dropout(x) 

        x = self.conv3(x, edge_index, edge_attr)
        # No dropout after the last GNN layer before pooling, typically.
        # x = self.dropout(x) # Optional dropout here

        # Global pooling
        graph_representation = self.pool(x, batch)

        # Readout network
        out = self.fc1(graph_representation)
        out = self.relu(out)
        out = self.dropout(out) # Dropout before final output layer
        out = self.fc2(out)     # Final linear layer (no activation for regression output)

        return out


# --- Utility Functions ---

def check_data_paths(data_root):
    """
    检查数据路径是否有效
    
    参数:
        data_root (str): 数据根目录
        
    返回:
        bool: 如果路径有效返回True，否则返回False
    """
    import os
    import logging
    logger = logging.getLogger("GNN")
    
    # 检查数据根目录是否存在
    if not os.path.exists(data_root):
        logger.error(f"数据根目录不存在: {data_root}")
        return False
    
    # 检查必要的文件是否存在
    required_files = [
        'train_files.txt', 
        'val_files.txt', 
        'test_files.txt',
        'node_feature_stats.pt',
        'edge_feature_stats.pt'
    ]
    
    for file in required_files:
        file_path = os.path.join(data_root, file)
        if not os.path.exists(file_path):
            logger.error(f"必要的文件不存在: {file_path}")
            return False
    
    # 检查graphs目录是否存在
    graphs_dir = os.path.join(data_root, 'graphs')
    if not os.path.exists(graphs_dir):
        logger.error(f"graphs目录不存在: {graphs_dir}")
        return False
    
    # 读取train_files.txt，检查第一个训练文件是否存在
    try:
        with open(os.path.join(data_root, 'train_files.txt'), 'r') as f:
            first_file = f.readline().strip()
            if first_file:
                first_file_path = os.path.join(data_root, first_file)
                if not os.path.exists(first_file_path):
                    logger.error(f"第一个训练文件不存在: {first_file_path}")
                    return False
    except Exception as e:
        logger.error(f"读取train_files.txt时出错: {str(e)}")
        return False
    
    logger.info(f"数据路径检查通过: {data_root}")
    return True

# --- Training Script ---

def train_model(model, train_loader, criterion, optimizer, device):
    """训练CEGNet模型一个epoch"""
    model.train()
    total_loss = 0
    num_graphs = 0
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(train_loader, desc="训练中")
    for data in progress_bar:
        # 处理可能的空批次或空图
        if data.x is None or data.x.size(0) == 0:
            continue # 跳过空图
            
        try:
            # 确保数据在正确的设备上
            data = data.to(device)
            
            # 清除梯度
            optimizer.zero_grad(set_to_none=True)  # 更高效的清除梯度方式
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            target = data.y.view(-1, OUTPUT_DIM)  # 确保目标形状匹配输出
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新权重
            optimizer.step()
            
            # 累计损失和处理的图数量
            batch_loss = loss.item() * data.num_graphs
            total_loss += batch_loss
            num_graphs += data.num_graphs
            
            # 更新进度条
            progress_bar.set_postfix(loss=f"{batch_loss/data.num_graphs:.4f}")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\nCUDA内存不足，跳过当前批次: {str(e)}")
                # 清理缓存以释放GPU内存
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    if num_graphs == 0:
        print("警告: 在训练epoch中没有处理有效的图。")
        return 0.0  # 返回0损失表示问题

    return total_loss / num_graphs

def evaluate_model(model, loader, criterion, device):
    """评估CEGNet模型"""
    model.eval()
    total_loss = 0
    num_graphs = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="评估中")
        for data in progress_bar:
            # 处理可能的空批次或空图
            if data.x is None or data.x.size(0) == 0:
                continue # 跳过空图
                
            try:
                # 确保数据在正确的设备上
                data = data.to(device)
                
                # 前向传播
                output = model(data)
                
                # 计算损失
                target = data.y.view(-1, OUTPUT_DIM)
                loss = criterion(output, target)
                
                # 累计损失和处理的图数量
                batch_loss = loss.item() * data.num_graphs
                total_loss += batch_loss
                num_graphs += data.num_graphs
                
                # 存储输出和目标用于计算其他指标
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())
                
                # 更新进度条
                progress_bar.set_postfix(loss=f"{batch_loss/data.num_graphs:.4f}")
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\nCUDA内存不足，跳过当前批次: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    if num_graphs == 0:
        # 如果验证/测试集为空或所有图都有问题
        print("警告: 在评估过程中没有处理有效的图。")
        return float('inf') if loader is not None else 0.0 # 如果验证集为空则返回无穷大损失，如果测试集为空则返回0

    return total_loss / num_graphs

def save_config(model_dir, config):
     """Saves model configuration to a JSON file."""
     if not os.path.exists(model_dir):
          os.makedirs(model_dir)
     config_path = os.path.join(model_dir, 'model_config.json')
     with open(config_path, 'w') as f:
          json.dump(config, f, indent=4)
     # print(f"Saved model configuration to {config_path}") # too verbose in loop


def load_config(model_dir):
     """Loads model configuration from a JSON file."""
     config_path = os.path.join(model_dir, 'model_config.json')
     if not os.path.exists(config_path):
          print(f"Error: Config file not found at {config_path}")
          return None
     try:
          with open(config_path, 'r') as f:
               config = json.load(f)
          # print(f"Loaded model configuration from {config_path}") # too verbose
          return config
     except Exception as e:
          print(f"Error loading config file {config_path}: {e}")
          return None


# Function to save scalers parameters
def save_scaler_params(model_dir, scaler, scaler_type):
    """Saves scaler parameters."""
    if scaler is None:
        # print("No scaler to save (target scaling is None).") # too verbose
        return

    params = {'type': scaler_type}
    try:
         if scaler_type == 'StandardScaler':
             params['mean'] = scaler.mean_.tolist()
             params['var'] = scaler.var_.tolist()
         elif scaler_type == 'MinMaxScaler':
             params['min'] = scaler.min_.tolist()
             params['scale'] = scaler.scale_.tolist()
         else:
              print(f"Warning: Unknown scaler type '{scaler_type}'. Parameters not saved.")
              return
    except Exception as e:
         print(f"Error extracting scaler parameters for type {scaler_type}: {e}")
         return # Don't save if parameter extraction fails


    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    scaler_params_path = os.path.join(model_dir, 'scaler_params.json')
    with open(scaler_params_path, 'w') as f:
        json.dump(params, f, indent=4)
    # print(f"Saved scaler parameters to {scaler_params_path}") # too verbose

def load_scaler_params(model_dir):
    """Loads scaler parameters and re-creates the scaler object."""
    scaler_params_path = os.path.join(model_dir, 'scaler_params.json')
    if not os.path.exists(scaler_params_path):
        # print(f"Info: Scaler params file not found at {scaler_params_path}. Assuming no target scaling was used.")
        return None, None # No scaler params found

    try:
        with open(scaler_params_path, 'r') as f:
            params = json.load(f)

        scaler_type = params.get('type')
        scaler = None
        if scaler_type == 'StandardScaler':
            scaler = StandardScaler()
            # Need to set attributes manually. This assumes the structure remains compatible.
            # Check if mean/var keys exist before accessing.
            if 'mean' in params and 'var' in params:
                scaler.mean_ = np.array(params['mean'])
                scaler.var_ = np.array(params['var'])
                scaler.scale_ = np.sqrt(scaler.var_) # StandardScaler also has scale_
                # print(f"Loaded StandardScaler parameters and re-created scaler.") # too verbose
            else:
                 print(f"Error loading StandardScaler params from {scaler_params_path}: missing mean or var keys.")
                 return None, None # Return None if invalid params
        elif scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
             # Check if min/scale keys exist
            if 'min' in params and 'scale' in params:
                scaler.min_ = np.array(params['min'])
                scaler.scale_ = np.array(params['scale'])
                # Reconstruct other attributes if needed for robust usage (though min_ and scale_ are usually sufficient for transform/inverse)
                scaler.data_min_ = scaler.min_
                scaler.data_range_ = scaler.scale_
                scaler.data_max_ = scaler.data_min_ + scaler.data_range_
                # print(f"Loaded MinMaxScaler parameters and re-created scaler.") # too verbose
            else:
                 print(f"Error loading MinMaxScaler params from {scaler_params_path}: missing min or scale keys.")
                 return None, None # Return None if invalid params
        else:
             print(f"Warning: Unknown scaler type '{scaler_type}' in params file {scaler_params_path}.")
             return None, None # Return None for unknown type

        return scaler, scaler_type
    except Exception as e:
         print(f"Error loading scaler parameters from {scaler_params_path}: {e}")
         return None, None # Return None on error


def inverse_transform_target(scaled_output, scaler, scaler_type):
     """Applies inverse transformation to scaled target/predictions."""
     if scaler is None:
          # print("No scaler available, returning scaled output directly.") # too verbose
          # Ensure the output is item() if it's a tensor/array to return a float scalar
          return scaled_output.item() if isinstance(scaled_output, torch.Tensor) or isinstance(scaled_output, np.ndarray) else scaled_output # No scaling was applied

     try:
          # Ensure input is numpy array with shape (-1, 1)
          # If already numpy array, ensure shape correct. If tensor, convert.
          if isinstance(scaled_output, torch.Tensor):
               scaled_output_np = scaled_output.cpu().numpy().reshape(-1, 1)
          elif isinstance(scaled_output, np.ndarray):
               scaled_output_np = scaled_output.reshape(-1, 1)
          elif isinstance(scaled_output, (int, float)):
               scaled_output_np = np.array([[scaled_output]]) # Single scalar becomes [[scalar]]
          else:
               print(f"Warning: Unexpected input type for inverse_transform_target: {type(scaled_output)}. Returning input.")
               return scaled_output # Unexpected type

          original_output_np = scaler.inverse_transform(scaled_output_np)
          return original_output_np.flatten() # Return flattened array (will be size 1 for single prediction)

     except Exception as e:
          print(f"Error during inverse transformation with {scaler_type}: {e}")
          # Attempt basic inverse for common cases if sklearn object fails? No, rely on sklearn.
          return scaled_output # Return scaled output if inverse fails


def main_train(data_root="processed_data", models_dir=MODELS_DIR, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """
    主函数用于加载数据、准备数据集/加载器、训练和评估模型。
    处理数据转换、保存/加载转换和模型生命周期。
    
    Args:
        data_root (str): 处理好的数据的根目录
        models_dir (str): 保存模型的目录
        batch_size (int): 批处理大小
        epochs (int): 训练的轮数
        lr (float): 学习率
    """
    print(f"\n--- 开始训练过程 ---")
    
    # 创建日志记录器
    import logging
    logger = logging.getLogger("GNN")
    
    # 创建模型目录
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"创建模型目录: {models_dir}")
    
    # 检查数据路径
    logger.info(f"检查数据路径: {data_root}")
    if not check_data_paths(data_root):
        logger.error(f"数据路径检查失败，无法继续训练")
        return
    
    # 导入数据加载模块
    try:
        # 优先尝试从当前目录导入
        try:
            from data_loader import create_data_loaders, get_feature_dimensions
            logger.info("从当前目录成功导入数据加载模块")
        except ImportError:
            from LithiumVision.data_loader import create_data_loaders, get_feature_dimensions
            logger.info("从LithiumVision目录成功导入数据加载模块")
    except ImportError:
        logger.warning("无法导入数据加载模块，尝试添加路径")
        try:
            import sys
            sys.path.append('.')
            sys.path.append('./LithiumVision')
            from data_loader import create_data_loaders, get_feature_dimensions
            logger.info("添加路径后成功导入数据加载模块")
        except ImportError:
            logger.error("无法导入数据加载模块，请确保data_loader.py在正确的位置")
            return
    
    # 从处理好的数据中获取特征维度
    node_in_features, edge_in_features = get_feature_dimensions(data_root)
    
    if node_in_features == 0:
        print("错误：确定的节点输入特征维度为0。无法继续。请检查特征提取逻辑。")
        return
    
    # 如果边特征维度为0但节点维度大于0，设置为1
    if edge_in_features == 0 and node_in_features > 0:
        print("警告：尽管节点存在，确定的边输入特征维度为0。假设边特征维度为1以进行模型初始化。")
        edge_in_features = 1
    
    print(f"确定的特征维度: 节点特征={node_in_features}, 边特征={edge_in_features}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(data_root, batch_size=batch_size)
    
    # 如果没有训练数据，终止
    if len(train_loader) == 0:
        print("没有有效的训练数据。请检查处理后的数据目录。")
        print(f"预期的处理后数据目录: {data_root}")
        return
    
    # 设置设备和显示GPU信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # 转换为GB
        print(f"\n使用GPU: {gpu_name} (内存: {gpu_memory:.2f} GB)")
        
        # 设置使用确定性算法以提高再现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 尝试自动调整内存使用
        torch.cuda.empty_cache()
    else:
        print("\n未检测到GPU，使用CPU进行训练（速度可能较慢）")
    
    print(f"使用设备: {device}")
    
    try:
        # 初始化模型
        model = CEGNet(
            node_in_features=node_in_features,
            edge_in_features=edge_in_features,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            dropout_rate=DROPOUT_RATE
        ).to(device)
        
        print(f"模型已创建并移至{device}设备")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"初始化模型时出错: {e}")
        return
    
    criterion = nn.MSELoss()  # 均方误差用于回归
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 保存初始配置
    model_config = {
        'node_in_features': node_in_features,
        'edge_in_features': edge_in_features,
        'hidden_dim': HIDDEN_DIM,
        'output_dim': OUTPUT_DIM,
        'dropout_rate': DROPOUT_RATE,
        'cutoff': CUTOFF,
        'target_scaler_type': TARGET_SCALER_TYPE,
        'atom_feature_missing_strategy': ATOM_FEATURE_MISSING_STRATEGY,
        'base_features': ['Z', 'group', 'row', 'block', 'X', 'atomic_radius', 'ionization_energy', 'electron_affinity'],
        'numerical_feature_names': [f for f in ['Z', 'group', 'row', 'block', 'X', 'atomic_radius', 'ionization_energy', 'electron_affinity'] if f not in ['group', 'row', 'block']]
    }
    save_config(models_dir, model_config)
    
    # 训练循环
    print("\n开始训练循环...")
    best_val_loss = float('inf')
    best_epoch = -1
    model_save_path = os.path.join(models_dir, 'best_regression_model.pt')
    
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        # 根据验证损失保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"在验证损失为 {best_val_loss:.4f} 时保存最佳模型到 {model_save_path}")
    
    print(f"\n训练完成。最佳验证损失: {best_val_loss:.4f}，在轮次 {best_epoch+1}")
    
    # 在测试集上进行最终评估
    print("\n在测试集上评估...")
    if os.path.exists(model_save_path):
        loaded_config = load_config(models_dir)
        if loaded_config is None:
            print("加载模型配置用于测试评估时出错。跳过。")
            return
        
        try:
            best_model = CEGNet(
                node_in_features=loaded_config['node_in_features'],
                edge_in_features=loaded_config['edge_in_features'],
                hidden_dim=loaded_config['hidden_dim'],
                output_dim=loaded_config['output_dim'],
                dropout_rate=0.0  # 评估时不使用dropout
            ).to(device)
            
            best_model.load_state_dict(torch.load(model_save_path, map_location=device))
            test_loss = evaluate_model(best_model, test_loader, criterion, device)
            print(f"测试损失 (MSE): {test_loss:.4f}")
        except Exception as e:
            print(f"加载最佳模型或在测试集上评估时出错: {e}")
            return
        
        # 计算其他指标，如MAE、R²
        from sklearn.metrics import mean_absolute_error, r2_score
        all_preds = []
        all_targets = []
        
        best_model.eval()
        with torch.no_grad():
            for data in test_loader:
                if data.x.size(0) == 0:
                    continue
                data = data.to(device)
                output = best_model(data)
                all_preds.append(output.cpu().numpy())
                all_targets.append(data.y.cpu().numpy())
        
        if not all_preds or not all_targets:
            print("测试评估过程中没有收集到有效的预测/目标。跳过指标计算。")
            return
        
        # 连接批次的结果
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        
        # 计算指标
        mae = mean_absolute_error(all_targets, all_preds)
        try:
            r2 = r2_score(all_targets, all_preds)
        except ValueError as e:
            if "constant" in str(e):
                print("警告：无法计算R²，因为测试集目标是常数。")
                r2 = float('nan')
            else:
                raise e
        
        print(f"测试MAE (log尺度): {mae:.4f}")
        print(f"测试R² (log尺度): {r2:.4f}")
    else:
        print(f"在 {model_save_path} 未找到最佳模型。跳过最终评估。")
        print("训练完成但没有保存最佳模型（例如，由于处理错误）。")


# --- Prediction/Inference Script ---

def predict_conductivity_from_cif(cif_filepath, model_dir=MODELS_DIR):
    """
    Predicts conductivity for a single CIF file using a trained model and its configuration.

    Args:
        cif_filepath (str): Path to the input CIF file.
        model_dir (str): Directory containing the saved model state_dict ('best_regression_model.pt'),
                         model configuration ('model_config.json'),
                         scaler params ('scaler_params.json'), and atom imputation stats ('atom_imputation_stats.json').

    Returns:
        float: Predicted conductivity on the original scale, or None if processing/prediction fails.
    """
    print(f"\n--- Starting Prediction Process for {os.path.basename(cif_filepath)} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(model_dir, 'best_regression_model.pt')

    # Load model configuration
    config = load_config(model_dir)
    if config is None:
        print("Failed to load model config. Cannot predict.")
        return None

    # Load scaler parameters for target inverse transformation
    scaler, scaler_type = load_scaler_params(model_dir)
    if scaler is None and config.get('target_scaler_type') is not None:
         print("Warning: Config indicates target scaling was used, but scaler params could not be loaded. Predictions will be on scaled data.")
         # Proceed but note the issue - predictions will be scaled values.

    # Load atom imputation stats
    atom_imputation_stats = load_atom_imputation_stats(model_dir) # Will return {} and print error if not found
    missing_strategy = config.get('atom_feature_missing_strategy', 'zero_fill') # Use saved strategy, default to zero_fill


    # Initialize the model architecture using loaded config
    try:
        model = CEGNet(
            node_in_features=config.get('node_in_features', 0), # Use .get with default 0 for robustness
            edge_in_features=config.get('edge_in_features', 1), # Default to 1 as distance is expected
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=config.get('output_dim', 1),
            dropout_rate=0.0 # Always use 0.0 dropout for inference
        ).to(device)
        # Basic check for valid input dimensions from config
        if model.node_in_features <= 0:
             print(f"Error: Invalid node_in_features from config: {model.node_in_features}. Cannot initialize model.")
             return None
        # Edge in features can be 0 theoretically, but layer expects >0. Check in CEGMessagePassing.


    except Exception as e:
        print(f"Error initializing model from config: {e}")
        return None


    # Load trained weights
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        print("Successfully loaded model weights.")
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        return None

    # Process the input CIF file into a Data object using the same logic as training data preprocessing
    try:
        parser = CifParser(cif_filepath)
        structures = parser.get_structures(primitive=False)
        if not structures:
             print(f"Error: Could not parse or find structure in CIF file {cif_filepath}.")
             return None
        structure = structures[0] # Get the first structure

        if len(structure) == 0:
             print(f"Error: Structure from CIF file {cif_filepath} is empty.")
             return None

    except Exception as e:
        print(f"Error parsing CIF file {cif_filepath}: {e}")
        return None

    try:
        # Extract raw atom features (with NaNs for numerical None, -1 for categorical None)
        x_raw_tensor, categorical_indices = get_atom_features(structure)
        x_raw_np = x_raw_tensor.numpy()


        # Ensure feature dimensions match what the model was trained on
        expected_node_features_from_config = config.get('node_in_features', -1)
        if x_raw_np.shape[1] != expected_node_features_from_config:
             print(f"Error: Extracted node feature dimension ({x_raw_np.shape[1]}) does not match trained model dimension ({expected_node_features_from_config}).")
             print("This could be due to changes in feature extraction logic or the base_features list.")
             return None # Dimension mismatch, cannot proceed


        # Apply Imputation (and potential Scaling) using loaded stats/scalers
        # Need feature names and numerical feature names from config to apply transforms consistently
        base_features = config.get('base_features', ['Z', 'group', 'row', 'block', 'X', 'atomic_radius', 'ionization_energy', 'electron_affinity']) # Use saved names
        numerical_feature_names = config.get('numerical_feature_names', [f for f in base_features if f not in ['group', 'row', 'block']]) # Use saved names

        x_processed_tensor = apply_atom_feature_transformation(
            x_raw_np,
            base_features,
            numerical_feature_names,
            atom_imputation_stats, # Pass loaded stats
            missing_strategy # Pass loaded strategy
        )
        # TODO: Apply Atom Feature Scaling here if implemented and scaler is loaded

        # Create graph using SAVE cutoff from config
        cutoff_val = config.get('cutoff', CUTOFF)
        edge_index, edge_attr, num_edges = create_graph_from_structure(structure, cutoff_val)

        # Ensure edge_attr dimension matches trained model if edges exist
        expected_edge_features_from_config = config.get('edge_in_features', 1)
        if edge_index.size(1) > 0 and edge_attr.size(1) != expected_edge_features_from_config:
             print(f"Error: Created edge feature dimension ({edge_attr.size(1)}) does not match trained model dimension ({expected_edge_features_from_config}) although edges exist.")
             # If config says edge_in_features=0, but creation yields edge_attr[:1], this check is tricky.
             # The CEGMessagePassing layer attempts to handle edge_in_channels=0 by not using edge_lin.
             # If created edge_attr has dim 1 but model expects 0, or vice versa?
             # Let's check only if edge_in_features from config > 0.
             if expected_edge_features_from_config > 0 and edge_attr.size(1) != expected_edge_features_from_config:
                 print(f"Error: Created edge feature dimension ({edge_attr.size(1)}) does not match trained model dimension ({expected_edge_features_from_config}). Cannot predict.")
                 return None # Dimension mismatch


        # Create a dummy target 'y' and batch index for single graph prediction
        y = torch.tensor([0.0], dtype=torch.float) # Placeholder target
        # PyG DataLoader adds batch index when batching. For a single graph, create batch index manually.
        batch_index = torch.zeros(x_processed_tensor.size(0), dtype=torch.long)


        # Create PyTorch Geometric Data object
        data = Data(x=x_processed_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch_index)

        data = data.to(device)

        # Make prediction
        with torch.no_grad():
            output_scaled = model(data) # Output is on scaled target scale

        # Inverse transform the prediction using the loaded scaler
        # Pass the loaded scaler and type, not the ones potentially from the dataset object
        predicted_conductivity = inverse_transform_target(output_scaled.item(), scaler, scaler_type)

        # The inverse_transform_target returns a numpy array of shape (1,) and flatten() flattens it.
        # Get the scalar value from the array.
        return predicted_conductivity.item() if predicted_conductivity is not None and isinstance(predicted_conductivity, np.ndarray) else predicted_conductivity

    except Exception as e:
        print(f"Error during prediction processing for {os.path.basename(cif_filepath)}: {e}")
        return None


# --- Example Usage ---

if __name__ == "__main__":
    # 设置OpenMP环境变量以避免冲突
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # 设置详细日志
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("GNN")
    logger.info("启动GNN模型训练脚本")
    
    # 添加命令行参数支持
    import argparse
    import sys
    
    # 检查Python版本和依赖库
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    # 确定当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(current_dir, "processed_data")
    default_models_dir = os.path.join(current_dir, "models")
    logger.info(f"当前目录: {current_dir}")
    logger.info(f"默认数据目录: {default_data_dir}")
    logger.info(f"默认模型目录: {default_models_dir}")
    
    parser = argparse.ArgumentParser(description="CEGNet模型训练和评估")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, 
                        help="处理好的数据的根目录")
    parser.add_argument("--models_dir", type=str, default=default_models_dir,
                        help="保存模型的目录")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="批处理大小")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="学习率")
    parser.add_argument("--predict", type=str, default=None,
                        help="预测单个CIF文件的电导率")
    parser.add_argument("--preprocess", action="store_true",
                        help="运行预处理步骤")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="是否使用GPU加速训练（默认启用）")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用CPU训练（覆盖--gpu选项）")
    
    args = parser.parse_args()
    
    print("\n======= CEGNet示例工作流 =======\n")
    
    # 如果需要预处理数据
    if args.preprocess:
        print("\n--- 运行数据预处理 ---")
        try:
            # 尝试导入数据处理模块
            try:
                # 先尝试从当前目录导入
                from data_process import CrystalDataProcessor
                print("从当前目录成功导入数据处理模块")
            except ImportError:
                try:
                    from LithiumVision.data_processess import CrystalDataProcessor
                    print("从LithiumVision目录成功导入数据处理模块")
                except ImportError:
                    import sys
                    sys.path.append('.')
                    sys.path.append('./LithiumVision')
                    from data_process import CrystalDataProcessor
                    print("添加路径后成功导入数据处理模块")
            
            # 创建处理器并运行
            processor = CrystalDataProcessor(
                cif_dir=os.path.join(current_dir, "randomized_cifs.zip"),  # 使用当前目录下的zip文件
                conductivity_file=os.path.join(current_dir, "extracted_conductivity.csv"),  # 使用当前目录下的电导率文件
                output_dir=args.data_dir,  # 输出到指定目录
                cutoff_radius=CUTOFF  # 使用相同的截断半径
            )
            
            results = processor.run()
            print("\n--- 数据预处理完成 ---")
            
            if results["status"] == "success":
                print(f"成功处理了 {results['total_processed']} 个CIF文件")
                print(f"有电导率数据的文件: {results['with_conductivity']}")
                print(f"没有电导率数据的文件: {results['without_conductivity']}")
            else:
                print(f"预处理失败: {results.get('message', '未知错误')}")
        
        except Exception as e:
            print(f"预处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果提供了预测选项
    elif args.predict:
        print(f"\n--- 对文件 {args.predict} 进行预测 ---")
        predicted_value = predict_conductivity_from_cif(
            cif_filepath=args.predict,
            model_dir=args.models_dir
        )
        
        if predicted_value is not None:
            print(f"预测的电导率: {predicted_value:.4e}")
        else:
            print("预测失败。")
    
    # 否则进行训练
    else:
        print("\n--- 运行训练示例 ---")
        
        # 处理GPU/CPU选项
        if args.cpu:
            # 如果指定了--cpu参数，强制使用CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("已设置强制使用CPU进行训练")
        elif args.gpu and not torch.cuda.is_available():
            print("警告: 已指定使用GPU但未检测到可用的CUDA设备，将使用CPU训练")
        
        try:
            main_train(
                data_root=args.data_dir,
                models_dir=args.models_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.learning_rate
            )
            print("\n--- 训练示例完成 ---")
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n======= CEGNet示例工作流完成 =======\n")
