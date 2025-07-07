import argparse
import numpy as np
import os
import logging
import csv
from .data_loader import DataLoader
from .energy_calculator import calculate_energies
from .error_estimator import calculate_errors
from datetime import datetime

def print_pair_results(dl: DataLoader, results: dict, error_results: dict, args):
    """Print pairwise comparison results for all schemes to console"""
    print("\nPairwise Results (Console Output):")
    
    # Filter keys to only include actual energy vectors, excluding times and weights
    num_mols = len(dl.id_map)
    filtered_schemes = []
    for k, v in results.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size == num_mols and not k.endswith('_time') and not k.endswith('_weights'):
            filtered_schemes.append(k)
    
    if not filtered_schemes:
        logging.warning("Initial scheme filter in print_pair_results yielded no schemes. Trying a broader filter for ndarrays.")
        for k, v in results.items():
            if isinstance(v, np.ndarray) and not k.endswith('_time') and not k.endswith('_weights'):
                # Check if it's already added to avoid duplicates if conditions overlap
                if k not in filtered_schemes:
                    filtered_schemes.append(k)
        
        if not filtered_schemes:
            logging.warning("print_pair_results: No valid energy vector schemes found in results to print after broader filter.")
            print("  (No valid energy vector schemes found to display pairwise results)")
            return

    header = "Pair\tOriginal"
    for scheme in filtered_schemes:
        header += f"\t{scheme.upper()}\t{scheme.upper()}_Error"
    print(header)

    for i, j, _, error, orig_ddg in dl.get_mapped_pairs():
        lig1 = dl.rev_map[i]
        lig2 = dl.rev_map[j]
        
        row = f"{lig1}-{lig2}\t{orig_ddg:.4f}"
        
        for scheme in filtered_schemes:
            G = results[scheme] # G should now always be a numpy array due to filtering
            if G is not None and i < len(G) and j < len(G):
                calc_ddg = G[j] - G[i]
                calc_error = abs(orig_ddg - calc_ddg)  # Calculate error
                row += f"\t{calc_ddg:.4f}\t{calc_error:.4f}"
            else:
                row += "\tN/A\tN/A" # Handle cases where scheme might not be calculated or G is unexpectedly None/wrong size
        
        print(row)

def main():
    """Main entry point for the free energy calculation tool"""
    parser = argparse.ArgumentParser(description="Enhanced Free Energy Calculator")
    parser.add_argument('-f', '--file', required=True, help='Input file path')
    parser.add_argument('-r', '--ref', required=True, help='Reference molecule ID')
    parser.add_argument('-e', '--energy', type=float, required=True, help='Reference absolute energy')
    parser.add_argument('-p', '--pair', choices=['yes', 'no'], default='no',
                       help='Output pairwise results to console and file (default: no)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show optimization details and write verbose weights CSV')
    parser.add_argument('--max_iter', type=int, default=50000,
                       help='Max iterations for optimizer G-step (default: 50000)')
    parser.add_argument('--tol', type=float, default=1e-8,
                       help='Convergence threshold for optimizer G-step (default: 1e-8)')
    parser.add_argument('--no_header', action='store_true',
                       help='Input file has no header')
    # Mode selects calculation engine(s)
    parser.add_argument('--mode', choices=['matrix', 'optimize', 'both'], default='optimize',
                       help="Calculation engine: 'matrix', 'optimize', or 'both' (default: optimize)")
    # Weight optimization mode selects weighting scheme
    parser.add_argument('--w_opt_mode', choices=['std_info_weighted', 'irls'], default='std_info_weighted',
                       help="Weighting scheme: 'std_info_weighted' (SFC/WSFC) or 'irls' (Iteratively Reweighted Least Squares) (default: std_info_weighted)")
    # IRLS specific arguments (only used if w_opt_mode='irls')
    parser.add_argument('--irls_max_iter', type=int, default=10,
                       help='Max iterations for outer IRLS loop (default: 10)')
    parser.add_argument('--irls_tol', type=float, default=1e-4,
                       help='Convergence threshold for outer IRLS loop (default: 1e-4)')
    # Add ref_weight argument
    parser.add_argument('--ref_weight', type=float, default=1e6, 
                       help='Weight for the reference energy constraint in augmented matrix methods (default: 1e6)')

    args = parser.parse_args()

    try:
        # Initialize data loader
        dl = DataLoader(filename=args.file, ref_mol=args.ref, skip_header=not args.no_header)
        # dl.print_mapping()
        # dl.validate_data()

        # Perform calculations using the specified mode and parameters
        results = calculate_energies(
            pairs=dl.get_mapped_pairs(),
            num_mols=len(dl.id_map),
            col_types=dl.col_types,
            ref_energy=args.energy,
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=args.verbose,
            mode=args.mode,
            w_opt_mode=args.w_opt_mode,
            irls_max_iter=args.irls_max_iter,
            irls_tol=args.irls_tol,
            ref_weight=args.ref_weight
        )

        # Calculate errors for all schemes that produced *energy vector* results
        energy_vector_keys = [k for k, v in results.items() if isinstance(v, np.ndarray) and v.size == len(dl.id_map) and not k.endswith('_weights') and not k.endswith('_time')]
        results_for_error_calc = {k: results[k] for k in energy_vector_keys}
        error_results = calculate_errors(dl.get_mapped_pairs(), results_for_error_calc)
        
        # Get timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # --- Output Results to Files and Console ---
        base_filename, _ = os.path.splitext(args.file)
        schemes_to_report = energy_vector_keys

        # --- Console Output (Node) ---
        print("\nNode Results (Console Output):")
        node_header_console = "Ligand"
        for scheme in schemes_to_report:
            node_header_console += f"\t{scheme.upper()}"
            node_header_console += f"\t{scheme.upper()}_Path_Dep_Error\t{scheme.upper()}_Path_Indep_Error"
        print(node_header_console)

        node_data_rows = []
        for idx in sorted(dl.id_map.values()):
            lig = dl.rev_map[idx]
            row_values = { 'ligand': lig }
            for scheme in schemes_to_report:
                 G_vector = results.get(scheme)
                 err_vector = error_results.get(scheme)
                 if G_vector is not None and err_vector is not None and idx < len(G_vector):
                     row_values[f'{scheme}_energy'] = G_vector[idx]
                     path_dep_err, path_indep_err = err_vector
                     if idx < len(path_dep_err) and idx < len(path_indep_err):
                         row_values[f'{scheme}_path_dep'] = path_dep_err[idx]
                         row_values[f'{scheme}_path_indep'] = path_indep_err[idx]
                     else:
                         row_values[f'{scheme}_path_dep'] = np.nan
                         row_values[f'{scheme}_path_indep'] = np.nan
                 else:
                     row_values[f'{scheme}_energy'] = np.nan
                     row_values[f'{scheme}_path_dep'] = np.nan
                     row_values[f'{scheme}_path_indep'] = np.nan
            node_data_rows.append(row_values)

        for row_data in node_data_rows:
            row_str = row_data['ligand']
            for scheme in schemes_to_report:
                row_str += f"\t{row_data.get(f'{scheme}_energy', np.nan):.4f}"
                row_str += f"\t{row_data.get(f'{scheme}_path_dep', np.nan):.4f}"
                row_str += f"\t{row_data.get(f'{scheme}_path_indep', np.nan):.4f}"
            print(row_str)

        # --- Console Output (Pair, optional) ---
        if args.pair == 'yes':
            print_pair_results(dl, results, error_results, args)

        # --- Write Results to Files --- 
        for scheme in schemes_to_report:
            # Node File
            node_filename = f"{base_filename}_{scheme}_node.txt"
            node_header_file = "Ligand\tEnergy\tPath_Dep_Error\tPath_Indep_Error"
            with open(node_filename, 'w') as f_node:
                f_node.write(node_header_file + "\n")
                for row_data in node_data_rows:
                     if f'{scheme}_energy' in row_data:
                         energy_val = row_data[f'{scheme}_energy']
                         path_dep_val = row_data[f'{scheme}_path_dep']
                         path_indep_val = row_data[f'{scheme}_path_indep']
                         if not np.isnan(energy_val):
                             f_node.write(f"{row_data['ligand']}\t{energy_val:.4f}\t{path_dep_val:.4f}\t{path_indep_val:.4f}\n")
            print(f"Node results for {scheme.upper()} saved to: {node_filename}")

            # Pair File (if requested)
            if args.pair == 'yes':
                pair_filename = f"{base_filename}_{scheme}_pair.txt"
                pair_header = "Pair\tOriginal_ddG\tCalculated_ddG\tError"
                with open(pair_filename, 'w') as f_pair:
                    f_pair.write(pair_header + "\n")
                    for i, j, _, error, orig_ddg in dl.get_mapped_pairs():
                        lig1 = dl.rev_map[i]
                        lig2 = dl.rev_map[j]
                        G = results.get(scheme)
                        # print(f"G: {G}")
                        if G is not None and i < len(G) and j < len(G):
                             calc_ddg = G[j] - G[i]
                             calc_error = abs(orig_ddg - calc_ddg)
                             f_pair.write(f"{lig1}-{lig2}\t{orig_ddg:.4f}\t{calc_ddg:.4f}\t{calc_error:.4f}\n")
                print(f"Pairwise results for {scheme.upper()} saved to: {pair_filename}")

            # Time File - Fetch specific time from results
            time_key = f"{scheme}_time"
            scheme_exec_time = results.get(time_key)
            time_filename = f"{base_filename}_{scheme}_time.txt"
            with open(time_filename, 'w') as f_time:
                f_time.write(f"Timestamp: {timestamp}\n")
                f_time.write(f"Input file: {args.file}\n")
                f_time.write(f"Calculation scheme: {scheme.upper()}\n")
                if scheme_exec_time is not None:
                    f_time.write(f"Execution time: {scheme_exec_time:.4f} seconds\n")
                else:
                    f_time.write(f"Execution time: N/A\n") # Handle case where time might be missing
            print(f"Timing information for {scheme.upper()} saved to: {time_filename}")

        # --- Write Verbose Weights CSV (if verbose) --- 
        if args.verbose:
            print("\nWriting verbose weight files...")
            all_pairs_data = dl.get_mapped_pairs()
            num_orig_pairs = len(all_pairs_data)
            
            for key, value in results.items():
                if key.endswith('_weights') and isinstance(value, np.ndarray):
                    scheme_name = key.replace('_weights', '')
                    weights_vector = value
                    
                    # Determine if weights are augmented (P+1) or original (P)
                    is_augmented = len(weights_vector) == num_orig_pairs + 1
                    num_pairs_to_write = num_orig_pairs if not is_augmented else num_orig_pairs + 1

                    # Check consistency if not augmented
                    if not is_augmented and len(weights_vector) != num_orig_pairs:
                        logging.warning(f"Skipping verbose weights for '{scheme_name}': Mismatch between weights ({len(weights_vector)}) and pairs ({num_orig_pairs})")
                        continue
                        
                    csv_filename = f"{base_filename}_{scheme_name}_verbose_weights.csv"
                    try:
                        with open(csv_filename, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["Pair_Index", "Mol_i", "Mol_j", "Weight"])
                            
                            # Write weights for original pairs
                            for k, (i, j, _, _, _) in enumerate(all_pairs_data):
                                if k < len(weights_vector):
                                    weight = weights_vector[k]
                                    writer.writerow([k, i, j, f"{weight:.6f}"])
                                else: # Should not happen if checks above are correct
                                    logging.error(f"Index out of bounds writing weights for {scheme_name}, pair {k}")
                                    break
                                    
                            # If weights were augmented, write the reference constraint weight
                            if is_augmented:
                                writer.writerow(["Ref_Constraint", -1, dl.id_map[args.ref], f"{weights_vector[-1]:.6f}"]) 
                                
                        print(f"Verbose weights for {scheme_name.upper()} saved to: {csv_filename}")
                    except IOError as e:
                        logging.error(f"Failed to write verbose weights file {csv_filename}: {e}")
                        print(f"Error: Could not write verbose weights file {csv_filename}")
                    except KeyError:
                        logging.error(f"Failed to find reference molecule ID '{args.ref}' in id_map for verbose weight output.")
                    except IndexError as e:
                         logging.error(f"Index error while writing verbose weights for {scheme_name}: {e}")
                         print(f"Error: Index mismatch while writing verbose weights for {scheme_name}")
            
    except Exception as e:
        logging.error(f"Runtime Error: {str(e)}", exc_info=True) 
        logging.error(f"Runtime Error: {str(e)}")

if __name__ == "__main__":
    main() 