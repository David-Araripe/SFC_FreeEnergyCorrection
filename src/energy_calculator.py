import numpy as np
import numpy.linalg as la  # Import numpy.linalg
from typing import List, Tuple, Dict, Optional, Any
import logging
import time # Import time module
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a custom exception for rank deficiency
class RankDeficiencyError(ValueError):
    pass

def _validate_linear_system(
    A: np.ndarray,
    N: int,
    W: Optional[np.ndarray] = None,
    col_types: Optional[int] = None,
    threshold: float = 1e3,
    ref_idx: Optional[int] = None
) -> None:
    """
    Validates the linear system Ax = b before attempting to solve.
    Checks for rank deficiency (indicating potential disconnected graph) and 
    high condition numbers (indicating potential numerical instability).

    Args:
        A: The design matrix (P x N).
        N: The number of variables (molecules).
        W: Optional weight vector or matrix (for condition number checks).
        col_types: Column types (needed for weighted condition number check).
        threshold: Condition number warning threshold.
        ref_idx: Reference index used for augmenting matrix methods (for condition check).

    Raises:
        RankDeficiencyError: If rank(A) < N - 1, indicating the system is fundamentally
                             underdetermined (likely disconnected graph).
    """
    logging.debug("Validating linear system (Rank and Condition Numbers)...")
    P = A.shape[0]

    # --- 1. Rank Check (Critical for Solvability) ---
    # Check rank of the *original* matrix A. Rank < N-1 implies disconnected components.
    try:
        rank_A = np.linalg.matrix_rank(A)
    except Exception as e:
        logging.error(f"Failed to compute rank(A): {e}")
        # Treat failure to compute rank as a potential issue
        raise ValueError(f"Could not determine rank of matrix A: {e}") from e

    logging.info(f"Rank check: rank(A) = {rank_A}, N = {N}")
    if rank_A < N - 1:
        error_msg = (
            f"Matrix A is rank deficient (rank={rank_A} < N-1={N-1}). "
            f"This often indicates disconnected components in the molecular graph. "
            f"The system is underdetermined and cannot be solved uniquely."
        )
        logging.error(error_msg)
        raise RankDeficiencyError(error_msg)
    elif rank_A == N - 1:
        logging.info("Matrix A has rank N-1, indicating a connected graph (expected for relative energies).")
    else: # rank_A == N (or potentially > N if P >= N, but still full column rank)
         logging.info(f"Matrix A has rank {rank_A} (>= N), system is likely overdetermined or well-determined.")


    # --- 2. Condition Number Check (Numerical Stability - Optional Warning) ---
    # Perform condition number check on potentially augmented matrix, as this reflects
    # the system the *augmented matrix solvers* actually solve.
    A_check = A.copy()
    w_check = None

    # Augment structurally if ref_idx is provided for the check
    if ref_idx is not None and 0 <= ref_idx < N:
        logging.debug(f"Augmenting structurally for condition check with ref_idx={ref_idx}")
        e_ref = np.zeros((1, N), dtype=A.dtype)
        e_ref[0, ref_idx] = 1.0
        A_check = np.vstack([A, e_ref])
        if W is not None:
            if W.ndim == 2 and W.shape[0] == W.shape[1] == P: w_orig = np.diag(W)
            elif W.ndim == 1 and W.shape[0] == P: w_orig = W
            else: raise ValueError("Invalid W shape for augmentation check")
            w_check = np.concatenate([w_orig, [1.0]])
    elif W is not None:
        if W.ndim == 2 and W.shape[0] == W.shape[1] == P: w_check = np.diag(W)
        elif W.ndim == 1 and W.shape[0] == P: w_check = W
        else: raise ValueError("Invalid W shape")

    # Check cond(A_check) or cond(A)
    cond_matrix_name = 'A_aug' if ref_idx is not None else 'A'
    try:
        cond_A_val = np.linalg.cond(A_check)
    except Exception as e:
        logging.error(f"Failed to compute cond({cond_matrix_name}): {e}")
        cond_A_val = np.inf

    logging.info(f"cond({cond_matrix_name}) [SFC check] = {cond_A_val:.2e}")
    if cond_A_val > threshold:
        logging.warning(f"cond({cond_matrix_name}) = {cond_A_val:.2e} exceeds threshold {threshold:.1e}; "
                        "direct lstsq may be unstable.")

    # Check cond(Aw_check) or cond(Aw)
    if w_check is not None and col_types == 4:
        cond_weighted_matrix_name = 'Aw_aug' if ref_idx is not None else 'Aw'
        w_sqrt = np.sqrt(w_check)
        Aw_check_weighted = A_check * w_sqrt[:, None]
        try:
            cond_Aw_val = np.linalg.cond(Aw_check_weighted)
        except Exception as e:
            logging.error(f"Failed to compute cond({cond_weighted_matrix_name}): {e}")
            cond_Aw_val = np.inf

        logging.info(f"cond({cond_weighted_matrix_name}) [WSFC check] = {cond_Aw_val:.2e}")
        if cond_Aw_val > threshold:
            logging.warning(f"cond({cond_weighted_matrix_name}) = {cond_Aw_val:.2e} exceeds threshold {threshold:.1e}; "
                            "weighted lstsq may be unstable.")

    logging.debug("Linear system validation complete.")

def _solve_matrix_equations(
    A: np.ndarray,
    b: np.ndarray,
    W: Optional[np.ndarray],
    col_types: int,
    N: int,
    ref_idx: int = 0,
    ref_energy: float = 0.0,
    ref_weight: float = 1e6
) -> Dict[str, np.ndarray]:
    """
    Solves the augmented linear systems using (weighted) lstsq.
    Assumes rank check was performed beforehand.
    Returns ALIGNED energy vectors.
    """
    matrix_results: Dict[str, np.ndarray] = {}
    P = A.shape[0]

    # --- Augment System --- 
    logging.debug(f"Augmenting system for matrix solve: ref_idx={ref_idx}, ref_energy={ref_energy:.4f}, ref_weight={ref_weight:.1e}")
    e_ref = np.zeros((1, N), dtype=A.dtype); e_ref[0, ref_idx] = 1.0
    A_aug = np.vstack([A, e_ref]); b_aug = np.concatenate([b, [ref_energy]])
    w_wsfc_aug = None
    if col_types == 4 and W is not None:
        if W.ndim == 2 and W.shape[0] == W.shape[1] == P: w_orig = np.diag(W)
        elif W.ndim == 1 and W.shape[0] == P: w_orig = W
        else: logging.warning("WSFC: Invalid W shape"); col_types = 3
        if col_types == 4: w_wsfc_aug = np.concatenate([w_orig, [ref_weight]])

    # --- Solve Augmented SFC --- 
    logging.debug("Calculating SFC using direct lstsq on augmented system...")
    # Rank check removed - done in _validate_linear_system
    G_sfc, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    matrix_results['sfc_matrix'] = G_sfc
    logging.debug("SFC matrix calculation complete.")

    # --- Solve Augmented WSFC --- 
    if col_types == 4 and w_wsfc_aug is not None:
        logging.debug("Calculating WSFC using weighted lstsq on augmented system...")
        w_sqrt_aug = np.sqrt(w_wsfc_aug)
        Aw_aug = A_aug * w_sqrt_aug[:, None]
        bw_aug = b_aug * w_sqrt_aug
        # Rank check removed - done in _validate_linear_system
        G_wsfc, *_ = np.linalg.lstsq(Aw_aug, bw_aug, rcond=None)
        matrix_results['wsfc_matrix'] = G_wsfc
        logging.debug("WSFC matrix calculation complete.")

    return matrix_results

def _optimize_loss_function(
    pairs: List[Tuple],
    num_mols: int,
    weights_wsfc: Optional[np.ndarray],
    col_types: int,
    max_iter: int,
    tol: float,
    verbose: bool,
    N: int
) -> Dict[str, np.ndarray]:
    """Solves the systems using optimization (L-BFGS-B). Returns G vectors and corresponding weights."""
    opt_results = {}
    opt_schemes = []
    num_pairs = len(pairs)

    # Initial guess for SFC optimizer
    G_sfc_opt = np.zeros(N)
    opt_results['sfc'] = G_sfc_opt
    # Store SFC weights (all ones)
    opt_results['sfc_weights'] = np.ones(num_pairs)
    opt_schemes.append(('sfc', 1.0)) # Pass constant weight 1.0 for SFC

    if col_types == 4 and weights_wsfc is not None:
        # Initial guess for WSFC optimizer
        G_wsfc_opt = np.zeros(N)
        opt_results['wsfc'] = G_wsfc_opt
        # Store WSFC weights (pre-calculated exponential decay)
        opt_results['wsfc_weights'] = weights_wsfc.copy()
        opt_schemes.append(('wsfc', None)) # Pass None, weight will be calculated inside loss

    # Define the loss function for optimization
    def loss_function(G, weight_flag):
        loss = 0.0
        current_weights = np.zeros(num_pairs) # Temp store weights for this call
        for k, (i, j, adj_ddg, error, _) in enumerate(pairs):
            if i >= N or j >= N:
                raise IndexError(f"Internal error: Molecule index out of bounds ({i}, {j}) for num_mols={N} during loss calculation")
            diff = (G[j] - G[i]) - adj_ddg
            if weight_flag is None:  # WSFC: Use exponential decay weight
                if weights_wsfc is None:
                     raise ValueError("weights_wsfc cannot be None when weight_flag is None (WSFC mode)")
                if k >= len(weights_wsfc):
                     raise IndexError(f"Internal error: Pair index {k} out of bounds for weights_wsfc (length {len(weights_wsfc)})")
                w = weights_wsfc[k]
            else:  # SFC: Use constant weight
                w = weight_flag
            loss += w * diff**2
            current_weights[k] = w # Store the weight used
        return loss
        # Note: We aren't returning current_weights from here, the weights are fixed per scheme

    # Optimize each scheme
    for scheme, weight_param in opt_schemes:
        G_opt = opt_results[scheme] # Get the initial guess array
        if verbose:
            print(f"=== Optimizing {scheme.upper()} scheme (L-BFGS-B) ===")
        logging.debug(f"Optimizing {scheme.upper()} using L-BFGS-B...")

        result = minimize(loss_function, G_opt, args=(weight_param,), method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': tol, 'disp': verbose})

        if not result.success:
             logging.warning(f"Optimization for {scheme.upper()} did not converge: {result.message}")

        opt_results[scheme][:] = result.x

        if verbose:
            print(f"{scheme.upper()} Optimization completed with loss: {result.fun}")
        logging.debug(f"{scheme.upper()} optimization complete. Final loss: {result.fun}")

    # Return results including G vectors and the fixed weights used for each scheme
    return opt_results

def _irls_optimize_loss_function(
    pairs: List[Tuple],
    num_mols: int,
    col_types: int,
    max_iter: int,
    tol: float,
    N: int,
    irls_max_iter: int = 10,
    irls_tol: float = 1e-4,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    IRLS optimization for G, dynamically calculating weights based on residuals.
    Returns a dictionary containing the final energy vector ('irls') 
    and the final weight vector ('irls_weights').
    """
    # === Initialize ===
    G = np.zeros(N)
    num_pairs = len(pairs)
    W = np.ones(num_pairs)
    eps = 1e-6
    residuals = np.zeros(num_pairs)

    logging.debug("Starting IRLS optimization...")
    if verbose:
        print("=== Starting IRLS Optimization ===")

    # IRLS main loop
    for irls_iter in range(irls_max_iter):
        G_old = G.copy()
        W_old = W.copy()
        def loss_function_irls(G_vec):
            loss = 0.0
            for k, (i, j, adj_ddg, _, _) in enumerate(pairs):
                 if i >= N or j >= N:
                     raise IndexError(f"IRLS G-opt: Index out of bounds ({i},{j}) for N={N}")
                 diff = (G_vec[j] - G_vec[i]) - adj_ddg
                 loss += W[k] * diff ** 2
            return loss
        logging.debug(f"IRLS iter {irls_iter}: Optimizing G...")
        result = minimize(loss_function_irls, G, method='L-BFGS-B', options={'maxiter': max_iter, 'gtol': tol, 'disp': verbose})
        if not result.success:
            logging.warning(f"IRLS iter {irls_iter}: Optimization for G did not converge: {result.message}")
        G = result.x
        logging.debug(f"IRLS iter {irls_iter}: Updating weights W...")
        for k, (i, j, adj_ddg, _, _) in enumerate(pairs):
            if i >= N or j >= N:
                 raise IndexError(f"IRLS W-update: Index out of bounds ({i},{j}) for N={N}")
            residuals[k] = (G[j] - G[i]) - adj_ddg
        W = 1.0 / (np.abs(residuals) + eps)
        G_change = np.linalg.norm(G - G_old) / (np.linalg.norm(G_old) + eps)
        W_change = np.linalg.norm(W - W_old) / (np.linalg.norm(W_old) + eps)
        logging.debug(f"IRLS iter {irls_iter}: G_change={G_change:.3e}, W_change={W_change:.3e}, mean_W={W.mean():.3f}")
        if verbose:
            print(f"IRLS iter {irls_iter}: G_rel_change={G_change:.3e}, W_rel_change={W_change:.3e}, mean_weight={W.mean():.3f}")
        if G_change < irls_tol and W_change < irls_tol:
            logging.info(f"IRLS converged at iteration {irls_iter}")
            if verbose:
                print(f"IRLS converged at iteration {irls_iter}")
            break
    else:
        logging.warning(f"IRLS did not converge within {irls_max_iter} iterations.")
        if verbose:
            print(f"IRLS Warning: Did not converge within {irls_max_iter} iterations.")

    # Return results including the final G and the final W
    return {
        "irls": G,
        "irls_weights": W
        # "irls_residuals": residuals # Could also return if needed elsewhere
    }

def _irls_solve_matrix_equations(
    A: np.ndarray,
    b: np.ndarray,
    N: int,            
    P: int,            
    ref_idx: int = 0,
    ref_energy: float = 0.0,
    ref_weight: float = 1e6,
    irls_max_iter: int = 10,
    irls_tol: float = 1e-4,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Solves A G ≈ b via IRLS using augmented matrix operations.
    Assumes rank check was performed beforehand.
    Returns ALIGNED G and augmented weights.
    """
    # --- Augment System --- 
    logging.debug(f"Augmenting system for IRLS matrix solve: ref_idx={ref_idx}, ref_energy={ref_energy:.4f}, ref_weight={ref_weight:.1e}")
    e_ref = np.zeros((1, N), dtype=A.dtype); e_ref[0, ref_idx] = 1.0
    A_aug = np.vstack([A, e_ref]); b_aug = np.concatenate([b, [ref_energy]])
    
    # --- Initialize --- 
    G = np.zeros(N, dtype=float)
    w = np.ones(P + 1, dtype=float); w[-1] = ref_weight
    eps = 1e-6

    logging.debug("Starting IRLS (augmented matrix method) optimization...")
    if verbose:
        print("=== Starting IRLS (Augmented Matrix Method) Optimization ===")

    for it in range(irls_max_iter):
        G_old = G.copy(); w_old = w.copy()

        # --- Solve augmented weighted system --- 
        logging.debug(f"IRLS-Matrix iter {it}: solving augmented weighted system...")
        w_sqrt = np.sqrt(w)
        Aw_aug = A_aug * w_sqrt[:, None]; bw_aug = b_aug * w_sqrt

        try:
            # Rank check removed - done in _validate_linear_system
            G, _, _, _ = np.linalg.lstsq(Aw_aug, bw_aug, rcond=None)
        except la.LinAlgError as e:
            logging.error(f"IRLS-Matrix iter {it}: LinAlgError: {e}")
            G = G_old; w = w_old 
            logging.warning("IRLS-Matrix: Reverting to G,w from previous iteration due to LinAlgError.")
            break

        # --- Update weights --- 
        residuals_unweighted = A_aug @ G - b_aug
        w[:P] = 1.0 / (np.abs(residuals_unweighted[:P]) + eps)

        # --- Check convergence --- 
        G_change = np.linalg.norm(G - G_old) / (np.linalg.norm(G_old) + eps)
        w_change = np.linalg.norm(w - w_old) / (np.linalg.norm(w_old) + eps)

        logging.debug(f"IRLS-Matrix iter {it}: G_change={G_change:.3e}, w_change={w_change:.3e}, mean_w={w[:P].mean():.3f}")
        if verbose:
            print(f"IRLS-Matrix iter {it}: G_rel_change={G_change:.3e}, w_rel_change={w_change:.3e}, mean_weight(orig)={w[:P].mean():.3f}")

        if G_change < irls_tol and w_change < irls_tol:
            logging.info(f"IRLS-Matrix converged at iteration {it}")
            if verbose: print(f"=== IRLS-Matrix converged at iteration {it} ===")
            break
    else:
        logging.warning(f"IRLS-Matrix did not converge within {irls_max_iter} iterations.")
        if verbose: print(f"=== IRLS-Matrix Warning: Did not converge in {irls_max_iter} iters ===")

    return {
        "irls_matrix": G, 
        "irls_matrix_weights": w
    }

def calculate_energies(
    pairs: List[Tuple],
    num_mols: int,
    col_types: int,
    ref_energy: float,
    max_iter: int = 50000,
    tol: float = 1e-8,
    verbose: bool = False,
    mode: str = 'both',
    w_opt_mode: str = 'std_info_weighted',
    irls_max_iter: int = 10,
    irls_tol: float = 1e-4,
    ref_weight: float = 1e6
) -> Dict[str, Any]:
    """
    Perform state-function based free energy corrections.
    Matrix modes (_solve_matrix_equations, _irls_solve_matrix_equations) use an augmented
    system approach to enforce the reference energy constraint directly, returning aligned results.
    Optimization modes (_optimize_loss_function, _irls_optimize_loss_function) solve the 
    original system and require a separate alignment step afterwards.

    Args:
        pairs: List of tuples (mol_i, mol_j, adj_ddg, error, lambda).
        num_mols: Total number of unique molecules (N).
        col_types: Number of columns in the input data (3 or 4).
        ref_energy: Reference energy for the first molecule (index 0).
        mode: Calculation engine ('matrix', 'optimize', 'both'). Defaults to 'both'.
        w_opt_mode: Weighting scheme: 
                    'std_info_weighted' (uses fixed weights: 1 for SFC, exp(-error) for WSFC), 
                    'irls' (uses Iteratively Reweighted Least Squares).
                    Defaults to 'std_info_weighted'.
        max_iter: Max iterations for optimizer G-step (L-BFGS-B in std/IRLS opt mode).
        tol: Convergence threshold for optimizer G-step (L-BFGS-B in std/IRLS opt mode).
        verbose: If True, print optimization progress details.
        irls_max_iter: Max iterations for outer IRLS loop (used if w_opt_mode='irls').
        irls_tol: Convergence threshold for outer IRLS loop (used if w_opt_mode='irls').
        ref_weight: Weight applied to the reference constraint in augmented matrix methods.

    Returns:
        Dictionary containing calculated energy vectors, potentially weight vectors,
        and execution times for each calculated scheme (e.g., 'sfc_matrix_time').
        Keys depend on the mode and w_opt_mode used.
        Examples:
        - mode='matrix', w_opt_mode='std': 'sfc_matrix', 'wsfc_matrix'
        - mode='matrix', w_opt_mode='irls': 'irls_matrix', 'irls_matrix_weights'
        - mode='optimize', w_opt_mode='std': 'sfc', 'wsfc', 'sfc_weights', 'wsfc_weights'
        - mode='optimize', w_opt_mode='irls': 'irls', 'irls_weights'
    Raises:
        RankDeficiencyError: If the system matrix A is rank deficient.
    """
    logging.debug(f"Starting energy calculations (engine mode: {mode}, weight mode: {w_opt_mode})...")
    results = {}
    P = len(pairs); N = num_mols
    if P == 0: logging.warning("Empty pairs list"); return {}
    if N == 0: logging.warning("num_mols is zero"); return {}

    # --- Construct Matrices --- 
    A = np.zeros((P, N)); b = np.zeros(P); errors = np.ones(P)
    W_matrix = None; weights_wsfc = None
    for k, pair_data in enumerate(pairs):
        if col_types == 4 and len(pair_data) >= 4: i, j, adj_ddg, error = pair_data[:4]; errors[k] = error
        elif col_types == 3 and len(pair_data) >= 3: i, j, adj_ddg = pair_data[:3]
        else: raise ValueError(f"Pair data {k} len {len(pair_data)} mismatch {col_types}")
        if not (0 <= i < N and 0 <= j < N): raise IndexError(f"Indices {i},{j} out of bounds {N}")
        A[k, i] = -1; A[k, j] = 1; b[k] = adj_ddg
    if col_types == 4:
        # Normalize errors first
        error_norm = errors / np.sum(errors)
        # Calculate weights as inverse square of normalized errors
        weights_wsfc = 1.0 / (error_norm ** 2)
        W_matrix = np.diag(weights_wsfc)
    logging.debug(f"Constructed A ({P}x{N}), b ({P}), W ({W_matrix.shape if W_matrix is not None else 'None'})")

    ref_idx = 0 # Assuming DataLoader maps ref mol to index 0

    try:
        # --- Validate System (Rank and Condition Numbers) --- 
        _validate_linear_system(A, N, W_matrix if col_types == 4 else None, col_types, ref_idx=ref_idx)
        # Note: Passing ref_idx for cond number check relevance to augmented methods.
        # Rank check is on original A.

        # --- Run selected engine(s) with selected weighting scheme --- 
        if mode in ['matrix', 'both']:
            if w_opt_mode == 'std_info_weighted':
                 logging.info("Running Standard Matrix mode...")
                 start_time = time.time()
                 matrix_results = _solve_matrix_equations(A, b, W_matrix, col_types, N, ref_idx, ref_energy, ref_weight)
                 end_time = time.time()
                 # Add times for schemes calculated by this function
                 if 'sfc_matrix' in matrix_results: results['sfc_matrix_time'] = end_time - start_time
                 if 'wsfc_matrix' in matrix_results: results['wsfc_matrix_time'] = end_time - start_time # Using same time for now
                 results.update(matrix_results)
            elif w_opt_mode == 'irls':
                 logging.info("Running IRLS Matrix mode...")
                 start_time = time.time()
                 irls_matrix_results = _irls_solve_matrix_equations(A, b, N, P, ref_idx, ref_energy, ref_weight, irls_max_iter, irls_tol, verbose)
                 end_time = time.time()
                 # Add time for the scheme calculated
                 results['irls_matrix_time'] = end_time - start_time
                 results.update(irls_matrix_results)
            else: logging.error(f"Unknown w_opt_mode '{w_opt_mode}' for matrix mode.")

        # Execute Optimization Engine calculation(s)
        if mode in ['optimize', 'both']:
            if w_opt_mode == 'std_info_weighted':
                 logging.info("Running Standard Optimization mode...")
                 start_time = time.time()
                 opt_results = _optimize_loss_function(pairs, N, weights_wsfc, col_types, max_iter, tol, verbose, N)
                 end_time = time.time()
                 # Add times for schemes calculated by this function
                 if 'sfc' in opt_results: results['sfc_time'] = end_time - start_time
                 if 'wsfc' in opt_results: results['wsfc_time'] = end_time - start_time # Using same time for now
                 results.update(opt_results)
            elif w_opt_mode == 'irls':
                 logging.info("Running IRLS Optimization mode...")
                 start_time = time.time()
                 irls_results = _irls_optimize_loss_function(pairs, N, col_types, max_iter, tol, N, irls_max_iter, irls_tol, verbose)
                 end_time = time.time()
                 # Add time for the scheme calculated
                 results['irls_time'] = end_time - start_time
                 results.update(irls_results)
            else: logging.error(f"Unknown w_opt_mode '{w_opt_mode}' for optimize mode.")

        # --- Align reference energy (ONLY for Optimization results) --- 
        optimization_keys_to_align = [k for k in results.keys() if k in ['sfc', 'wsfc', 'irls']]
        if N > 0 and ref_energy is not None and optimization_keys_to_align:
            logging.debug(f"Aligning Optimization results to reference energy: {ref_energy} for molecule index {ref_idx}")
            for scheme in optimization_keys_to_align:
                if scheme in results:
                    G_vector = results[scheme]
                    if isinstance(G_vector, np.ndarray) and G_vector.size == N:
                        if G_vector.size > 0:
                            shift = G_vector[ref_idx] - ref_energy
                            results[scheme] = G_vector - shift
                            logging.debug(f"Aligned {scheme.upper()}: Shift applied = {-shift:.4f}")
                        else: logging.warning(f"Cannot align empty vector for scheme '{scheme}'.")
                    else: logging.warning(f"Result for scheme '{scheme}' is not a valid vector for alignment.")
        elif optimization_keys_to_align and (N <= 0 or ref_energy is None):
            logging.warning("Cannot align opt results: N<=0 or ref_energy is None.")

    except RankDeficiencyError as e:
        logging.error(f"Calculation aborted due to rank deficiency: {e}")
        # Optionally, re-raise or return specific error indicator if needed upstream
        return {} # Return empty results as planned
    except Exception as e:
        logging.exception(f"An unexpected error occurred during energy calculation: {e}")
        # Catch any other unexpected errors
        return {} # Return empty on other errors too

    logging.info(f"Energy calculations completed (engine mode: {mode}, weight mode: {w_opt_mode}).")
    return results
