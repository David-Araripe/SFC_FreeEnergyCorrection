import numpy as np
import pandas as pd  # Make sure to add this import at the top of your file

from .data_loader import DataLoader
from .energy_calculator import calculate_energies
from .error_estimator import calculate_errors


def run_and_process_results(
    dl: DataLoader,
    ref_energy: float,
    mode: str = "optimize",
    w_opt_mode: str = "std_info_weighted",
    max_iter: int = 50000,
    tol: float = 1e-8,
    irls_max_iter: int = 10,
    irls_tol: float = 1e-4,
    ref_weight: float = 1e6,
    include_pairs: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Runs the full energy calculation and processing pipeline.

    This function is designed for programmatic use (e.g., in a Jupyter notebook),
    separating the core logic from CLI parsing and file I/O.

    Args:
        dl (DataLoader): An initialized DataLoader object containing the graph data.
        ref_energy (float): The absolute energy of the reference molecule.
        include_pairs (bool): If True, calculates and includes pairwise results.
        verbose (bool): If True, includes detailed weight vectors in the output.
        (other args): Parameters for the calculation engine.

    Returns:
        dict: A nested dictionary where keys are calculation schemes (e.g., 'sfc').
              Each value is another dictionary containing:
              - 'node_results': A pandas DataFrame with node-specific results.
              - 'pair_results': A pandas DataFrame with pairwise results (or None).
              - 'execution_time': The execution time for the scheme.
              - 'weights': The weight vector if verbose=True (or None).
    """
    # 1. Perform core calculations
    results = calculate_energies(
        pairs=dl.get_mapped_pairs(),
        num_mols=len(dl.id_map),
        col_types=dl.col_types,
        ref_energy=ref_energy,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        mode=mode,
        w_opt_mode=w_opt_mode,
        irls_max_iter=irls_max_iter,
        irls_tol=irls_tol,
        ref_weight=ref_weight,
    )

    # 2. Calculate errors
    energy_vector_keys = [
        k
        for k, v in results.items()
        if isinstance(v, np.ndarray)
        and v.size == len(dl.id_map)
        and not k.endswith("_weights")
        and not k.endswith("_time")
    ]
    results_for_error_calc = {k: results[k] for k in energy_vector_keys}
    error_results = calculate_errors(dl.get_mapped_pairs(), results_for_error_calc)

    # 3. Structure the output
    processed_output = {}
    schemes_to_report = energy_vector_keys

    for scheme in schemes_to_report:
        # --- Process Node Data ---
        node_data_list = []
        for idx in sorted(dl.id_map.values()):
            lig = dl.rev_map[idx]
            G_vector = results.get(scheme)
            err_vector = error_results.get(scheme)

            row = {"Ligand": lig}
            if G_vector is not None and err_vector is not None and idx < len(G_vector):
                path_dep_err, path_indep_err = err_vector
                row["Energy"] = G_vector[idx]
                row["Path_Dep_Error"] = path_dep_err[idx] if idx < len(path_dep_err) else np.nan
                row["Path_Indep_Error"] = path_indep_err[idx] if idx < len(path_indep_err) else np.nan
            else:
                row["Energy"] = np.nan
                row["Path_Dep_Error"] = np.nan
                row["Path_Indep_Error"] = np.nan
            node_data_list.append(row)

        node_df = pd.DataFrame(node_data_list)

        # --- Process Pair Data (Optional) ---
        pair_df = None
        if include_pairs:
            pair_data_list = []
            G = results.get(scheme)
            if G is not None:
                for i, j, _, _, orig_ddg in dl.get_mapped_pairs():
                    lig1, lig2 = dl.rev_map[i], dl.rev_map[j]
                    calc_ddg = G[j] - G[i]
                    calc_error = abs(orig_ddg - calc_ddg)
                    pair_data_list.append(
                        {
                            "Pair": f"{lig1}-{lig2}",
                            "Original_ddG": orig_ddg,
                            "Calculated_ddG": calc_ddg,
                            "Error": calc_error,
                        }
                    )
            pair_df = pd.DataFrame(pair_data_list)

        # --- Assemble final dictionary for the scheme ---
        processed_output[scheme] = {
            "node_results": node_df,
            "pair_results": pair_df,
            "execution_time": results.get(f"{scheme}_time"),
            "weights": results.get(f"{scheme}_weights") if verbose else None,
        }

    return processed_output
