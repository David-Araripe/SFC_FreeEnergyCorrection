import argparse
import numpy as np
from src.data_loader import DataLoader
from src.energy_calculator import calculate_energies
from src.error_estimator import calculate_errors
import time
from datetime import datetime
import os
import pathlib

def print_pair_results(dl: DataLoader, results: dict, error_results: dict, args):
    """Print pairwise comparison results for all schemes"""
    print("\nPairwise Results:")
    
    # Dynamically build header based on available schemes
    schemes = list(results.keys())
    header = "Pair\tOriginal"
    for scheme in schemes:
        header += f"\t{scheme.upper()}\t{scheme.upper()}_Error"
    print(header)

    for i, j, _, error, orig_ddg in dl.get_mapped_pairs():
        lig1 = dl.rev_map[i]
        lig2 = dl.rev_map[j]
        
        # Build row with original value
        row = f"{lig1}-{lig2}\t{orig_ddg:.4f}"
        
        # Add calculated values and errors for each scheme
        for scheme in schemes:
            G = results[scheme]
            calc_ddg = G[j] - G[i]
            calc_error = abs(orig_ddg - calc_ddg)  # Calculate error
            row += f"\t{calc_ddg:.4f}\t{calc_error:.4f}"
        
        print(row)

def save_pair_results(dl: DataLoader, results: dict, error_results: dict, output_file: str):
    """Save pairwise comparison results for all schemes to a file"""
    schemes = list(results.keys())
    
    with open(output_file, 'w') as f:
        # Write header
        header = "Pair\tOriginal"
        for scheme in schemes:
            header += f"\t{scheme.upper()}\t{scheme.upper()}_Error"
        f.write(header + "\n")

        # Write data rows
        for i, j, _, error, orig_ddg in dl.get_mapped_pairs():
            lig1 = dl.rev_map[i]
            lig2 = dl.rev_map[j]
            
            row = f"{lig1}-{lig2}\t{orig_ddg:.4f}"
            
            for scheme in schemes:
                G = results[scheme]
                calc_ddg = G[j] - G[i]
                calc_error = abs(orig_ddg - calc_ddg)
                row += f"\t{calc_ddg:.4f}\t{calc_error:.4f}"
            
            f.write(row + "\n")

def save_node_results(dl: DataLoader, results: dict, error_results: dict, output_file: str):
    """Save node results for all schemes to a file"""
    schemes = list(results.keys())
    
    with open(output_file, 'w') as f:
        # Write header
        header = "Ligand"
        for scheme in schemes:
            header += f"\t{scheme.upper()}"
            header += f"\t{scheme.upper()}_Path_Independent_Error\t{scheme.upper()}_Path_Dependent_Error"
        f.write(header + "\n")

        # Write data rows
        for idx in sorted(dl.id_map.values()):
            lig = dl.rev_map[idx]
            row = f"{lig}"
            
            for scheme in schemes:
                row += f"\t{results[scheme][idx]:.4f}"
                path_dep, path_indep = error_results[scheme]
                row += f"\t{path_dep[idx]:.4f}\t{path_indep[idx]:.4f}"
            
            f.write(row + "\n")

def main():
    """Main entry point for the free energy calculation tool"""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Enhanced Free Energy Calculator")
    parser.add_argument('-f', '--file', required=True, help='Input file path')
    parser.add_argument('-r', '--ref', required=True, help='Reference molecule ID')
    parser.add_argument('-e', '--energy', type=float, required=True, help='Reference absolute energy')
    parser.add_argument('-p', '--pair', choices=['yes', 'no'], default='no',
                       help='Output pairwise results')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show optimization details')
    parser.add_argument('--max_iter', type=int, default=50000,
                       help='Max iterations (default: 50000)')
    parser.add_argument('--no_header', action='store_true',
                       help='Input file has no header')
    args = parser.parse_args()

    try:
        # Create output directory if it doesn't exist
        output_dir = "sfc_output_compounds"
        os.makedirs(output_dir, exist_ok=True)

        # Get input file name without extension for output file prefix
        input_file_name = pathlib.Path(args.file).stem
        
        # Initialize data loader
        dl = DataLoader(args.file, args.ref, not args.no_header)
        dl.print_mapping()
        dl.validate_data()

        # Perform calculations
        results = calculate_energies(
            pairs=dl.get_mapped_pairs(),
            num_mols=len(dl.id_map),
            col_types=dl.col_types,
            ref_energy=args.energy,
            max_iter=args.max_iter,
            verbose=args.verbose
        )

        # Calculate errors for all schemes
        error_results = calculate_errors(dl.get_mapped_pairs(), results)

        # Save node results
        node_output_file = os.path.join(output_dir, f"{input_file_name}_sfc_node.txt")
        save_node_results(dl, results, error_results, node_output_file)
        print(f"\nNode results saved to: {node_output_file}")

        # Save pair results if requested
        if args.pair == 'yes':
            pair_output_file = os.path.join(output_dir, f"{input_file_name}_sfc_pair.txt")
            save_pair_results(dl, results, error_results, pair_output_file)
            print(f"Pair results saved to: {pair_output_file}")

        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save execution time to file in sfc_output directory
        time_output_file = os.path.join(output_dir, f"{input_file_name}_sfc_time.txt")
        with open(time_output_file, 'w') as time_file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time_file.write(f"Input file: {args.file}\n")
            time_file.write(f"Timestamp: {timestamp}\n")
            time_file.write(f"Execution time: {execution_time:.2f} seconds\n")
            time_file.write(f"\nCalculation parameters:\n")
            time_file.write(f"Reference molecule: {args.ref}\n")
            time_file.write(f"Reference energy: {args.energy}\n")
            time_file.write(f"Max iterations: {args.max_iter}\n")
            time_file.write(f"Number of molecules: {len(dl.id_map)}\n")
            time_file.write(f"Number of pairs: {len(dl.get_mapped_pairs())}\n")
        print(f"Execution time information saved to: {time_output_file}")
            
    except Exception as e:
        print(f"Runtime Error: {str(e)}")

if __name__ == "__main__":
    main() 