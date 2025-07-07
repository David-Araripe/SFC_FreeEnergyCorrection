[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

⚠️ Note ⚠️ that this is a fork of the original repo. For the original repo please visit https://github.com/ZheLi-Lab/State-Function-based-free-energy-correction-SFC-

# State-Function Based Free Energy Correction (SFC)

### Overview
The State-Function Based Free Energy Correction (SFC) algorithm is designed to improve the accuracy of relative binding free energy calculations. It utilizes state function properties to correct  errors in pairwise free energy differences without requiring cycle identification, ensuring thermodynamic consistency across the entire network of molecular transformations.

### Features
- Naturally satisfies thermodynamic cycle consistency.
- Fast computation suitable for large molecular networks, applicable to large-scale lead compound optimization based on FEP-RBFE calculations.
- Supports using estimated errors of calculated ddG values as weights for optimization.

### Installation

```bash
python -m pip install git+https://github.com/David-Araripe/SFC_FreeEnergyCorrection.git
```

### Usage

#### Basic Usage (based on least squares optimization)
Run the SFC algorithm on your input file:
```bash
sfc -f example/n20_c0.10_u0.50_edges.csv -r 1 -e 0.0 -p yes --no_header
```

Parameters:
- `-f, --file`: Input file containing pairwise energy data
- `-r, --ref`: Reference molecule (default: first molecule in data)
- `-e, --energy`: Reference energy value 
- `-p, --pair`: if yes: print detailed results of the pair ddG, no: only molecule energies
- `--no_header`: Input file has no header row

#### Matrix-Based Usage (matrix mode)
To use the SFC algorithm with the matrix-based solver (instead of the optimizer), specify the `--mode matrix` option:

```bash
sfc -f example/n20_c0.10_u0.50_edges.csv -r 1 -e 0.0 --mode matrix --no_header
```

Parameters:
- `--mode matrix`: Use the matrix method for free energy correction (recommended for ultra-large-scale datasets)
- Other parameters are the same as in the basic usage


#### Input File Format
The input file should be a CSV file with the following columns:
```
Molecule1 Molecule2 ddG [Error]
```
- `Molecule1, Molecule2`: Molecule pair identifiers
- `ddG`: Free energy difference between the molecules
- `Error`: (Optional) Uncertainty in the free energy difference

### Output Files
The program generates several output files in the same directory as the input file:
- `*_pair.txt`: Corrected pairwise ΔΔG
- `*_node.txt`: Corrected ΔG
- `*_time.txt`: Execution time and statistics

### Reference
Liu, R.; Lai, Y.; Yao, Y.; Huang, W.; Zhong, Y.; Luo, H.-B.; Li, Z. State Function-Based Correction: A Simple and Efficient Free-Energy Correction Algorithm for Large-Scale Relative Binding Free-Energy Calculations. J. Phys. Chem. Lett. https://doi.org/10.1021/acs.jpclett.5c01119.
