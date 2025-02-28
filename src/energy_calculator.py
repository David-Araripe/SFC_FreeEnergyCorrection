import numpy as np
from typing import List, Tuple, Dict
import logging
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_energies(
    pairs: List[Tuple],
    num_mols: int,
    col_types: int,
    ref_energy: float,
    max_iter: int = 50000,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """Perform state-function based free energy corrections"""
    logging.debug("Starting energy calculations...")
    results = {}
    schemes = []

    # Initialize schemes
    G_sfc = np.zeros(num_mols)
    results['sfc'] = G_sfc
    schemes.append(('sfc', 2))

    if col_types == 4:  # Only if we have error data
        G_wsfc = np.zeros(num_mols)
        results['wsfc'] = G_wsfc
        schemes.append(('wsfc', 3))

    # Define the loss function for optimization
    def loss_function(G):
        loss = 0.0
        for i, j, adj_ddg, error, _ in pairs:
            diff = (G[j] - G[i]) - adj_ddg
            if weight_idx == 3:  # Use exponential decay for weight
                w = np.exp(-error)
            else:  # Use constant weight
                w = 1.0
            loss += w * diff**2
        return loss

    # Optimize each scheme
    for scheme, weight_idx in schemes:
        G = results[scheme]
        if verbose:
            print(f"\n=== Optimizing {scheme.upper()} scheme ===")

        # Use minimize from scipy to optimize G
        result = minimize(loss_function, G, method='L-BFGS-B', options={'maxiter': max_iter, 'disp': verbose})
        G[:] = result.x  # Update G with the optimized values

        if verbose:
            print(f"{scheme.upper()} Optimization completed with loss: {result.fun}")

    # Align reference energy
    ref_shift = results['sfc'][0] - ref_energy
    results['sfc'] -= ref_shift
    if 'wsfc' in results:
        results['wsfc'] -= results['wsfc'][0] - ref_energy

    logging.info("Energy calculations completed.")
    return results
