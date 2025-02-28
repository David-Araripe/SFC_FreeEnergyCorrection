import numpy as np
from typing import List, Tuple, Dict

def calculate_errors(pairs: List[Tuple], G_dict: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Calculate errors for all calculation schemes"""
    error_results = {}
    
    for scheme, G in G_dict.items():
        num_nodes = G.shape[0]
        path_dep = np.zeros(num_nodes)  # Max error per node
        path_indep = np.zeros(num_nodes)  # RMS error per node
        
        # Precompute all pair errors for this scheme
        pair_errors = []
        for i, j, adj_ddg, error, orig_ddg in pairs:  # Unpack according to get_mapped_pairs output
            calc_ddg = G[j] - G[i]  # Calculate the predicted ΔG
            abs_error = abs(orig_ddg - calc_ddg)  # Calculate the absolute error
            pair_errors.append((i, j, abs_error))
        
        # Calculate node errors
        for node in range(num_nodes):
            errors = []
            for i, j, error in pair_errors:
                if i == node or j == node:
                    errors.append(error)
            
            if errors:
                path_dep[node] = np.max(errors)  # Maximum error for the node
                path_indep[node] = np.sqrt(np.mean(np.square(errors)))  # RMS error for the node
            else:
                path_dep[node] = 0.0
                path_indep[node] = 0.0
        
        error_results[scheme] = (path_dep, path_indep)
    
    return error_results