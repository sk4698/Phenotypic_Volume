import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import anndata
import scanpy as sc 
from itertools import combinations
import seaborn as sns
import statsmodels.stats.multitest as multi

def sample_cells_with_replacement(gene_cell_df, cell_name, sample_size=1000):
    current_cell_type_df = gene_cell_df[cell_name]
    num_columns = current_cell_type_df.shape[1]
    if num_columns < sample_size:
        print(f"{cell_name} has fewer than {sample_size} cells.")
        return current_cell_type_df
    sampled_indices = np.random.choice(range(num_columns), size=sample_size, replace=True)
    sampled_columns_df = current_cell_type_df.iloc[:, sampled_indices]
    return sampled_columns_df.values

def compute_empirical_covariance(imputed_expression):
    covariance_matrix = np.cov(imputed_expression, rowvar=False)
    return covariance_matrix

def log_volume_of_nonzero_singular_values(covariance_matrix):
    U, singular_values, V = np.linalg.svd(covariance_matrix, full_matrices=False)
    nonzero_singular_values = singular_values[singular_values > 10**(-30)]
    log_volume = np.sum(np.log(nonzero_singular_values))
    total_genes = covariance_matrix.shape[0]
    normalized_log_volume = log_volume / total_genes
    return normalized_log_volume

def phenotypic_volume(adata, layer = None, subset = [], num_iterations = 20):

    if not layer:
        adata.layers["counts"] = adata.X.copy()
        adata.obs['original_total_counts'] = adata.obs['nCount_RNA']
        sc.pp.normalize_total(adata, exclude_highly_expressed=True)
        sc.pp.log1p(adata)
        adata.layers["log_counts"] = adata.X.copy()
        df = adata.to_df(layer='log_counts').transpose()
    else:
        df = adata.to_df(layer = 'log_counts').transpose()
    
    if not subset:
        subset = list(pd.unique(df.columns))

    volumes = {}
    sample_size = 1000
    largest_cluster = float("-inf")
    for cell in subset:
        largest_cluster = max(df[cell].shape[1], largest_cluster)
    for cell in subset:
        current_cell_type_df = df[cell]
        num_columns = current_cell_type_df.shape[1]
        sample_size = min(sample_size, num_columns)
    num_iterations = 10*largest_cluster//sample_size
    for cell in subset:
        current_cell_type_df = df[cell]
        num_columns = current_cell_type_df.shape[1]
        sample_size = min(sample_size, num_columns)
    print(num_iterations)
    for cell in subset:
        print(cell)
        cell_type_volumes = []
        for _ in range(num_iterations): #largest cluster/group of cells (n), every cell should have good probability 20*1000/n should be about 5-10ish
            print(_)
            X = sample_cells_with_replacement(df, cell, sample_size)
            Y = compute_empirical_covariance(X)
            Z = log_volume_of_nonzero_singular_values(Y)
            cell_type_volumes.append(Z)
        volumes[cell] = cell_type_volumes

    return pd.DataFrame(volumes) 

# example usage
if __name__ == "__main__":
    subset = 'tissue'
    adata = sc.read('tcells_decipher_trajectories_091123 (1).h5ad')
    adata.obs_names = adata.obs[subset]
    PV = phenotypic_volume(adata, 'log_counts')
    PV.to_csv(f't-cells-{subset}.csv')



