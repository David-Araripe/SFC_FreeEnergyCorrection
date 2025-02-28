# State-Function Based Free Energy Correction (SFC)

### Overview
The State-Function Based Free Energy Correction (SFC) algorithm is designed to improve the accuracy of free energy calculations in molecular systems. It utilizes state function properties to correct systematic errors in pairwise free energy differences, ensuring thermodynamic consistency across the entire network of molecular transformations.

### Features
- Naturally satisfies thermodynamic cycle consistency
- Fast computation suitable for large molecular networks, applicable to large-scale lead compound optimization based on FEP-RBFE calculations
- Supports using estimated errors of calculated ddG values as weights for optimization

### Installation
1. Clone this repository:
```bash
git clone <repository_url>
cd sfc
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Basic Usage
Run the SFC algorithm on your input file:
```bash
python main.py -f example/n20_c0.10_u0.50_edges.csv -r 1 -e 0.0 -p yes --no_header
```

Parameters:
- `-f, --file`: Input file containing pairwise energy data
- `-r, --ref`: Reference molecule (default: first molecule in data)
- `-e, --ref_ene`: Reference energy value (default: 0.00)
- `-p, --print`: Print option (yes: print detailed results, no: only molecule energies)
- `--no_header`: Input file has no header row

#### Input File Format
The input file should be a CSV file with the following columns:
```
Molecule1 Molecule2 ddG [Error]
```
- `Molecule1, Molecule2`: Molecule pair identifiers
- `ddG`: Free energy difference between the molecules
- `Error`: (Optional) Uncertainty in the free energy difference

### Output Files
The program generates several output files in the `sfc_output` directory:
- `*_sfc_pair.txt`: Corrected pairwise energies
- `*_sfc_node.txt`: Corrected molecular energies
- `*_sfc_time.txt`: Execution time and statistics
