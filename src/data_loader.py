import numpy as np
from collections import defaultdict, deque, OrderedDict
from typing import List, Tuple, Dict, Set
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """Enhanced data loading class with validation and mapping"""
    def __init__(self, filename: str, ref_mol: str = None, skip_header: bool = True):
        logging.debug(f"Initializing DataLoader with file: {filename}, ref_mol: {ref_mol}, skip_header: {skip_header}")
        self.raw_pairs = []
        self.id_map = OrderedDict()
        self.rev_map = {}
        self.col_types = 3
        self._load_data(filename, ref_mol, skip_header)

    def _load_data(self, filename, ref_mol, skip_header):
        """Load and validate input data file"""
        all_ids = set()
        
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Process header line
                if line_num == 0:  # Always check the first line for column types
                    cols = line.split()
                    self.col_types = len(cols)
                    if self.col_types not in (3, 4):
                        raise ValueError("Input file must contain 3 or 4 columns")
                    if skip_header:
                        continue  # Skip the header if specified

                parts = line.split()
                if len(parts) < 3 or len(parts) > 4:
                    raise ValueError(f"Line {line_num+1} column mismatch")

                lig1, lig2 = parts[0].strip(), parts[1].strip()
                orig_ddg = float(parts[2])
                error = float(parts[3]) if self.col_types == 4 else 1.0  # Default error if not provided

                self.raw_pairs.append((lig1, lig2, orig_ddg, error))
                all_ids.update([lig1, lig2])

        # Handle reference molecule
        all_ids = sorted(all_ids)
        if ref_mol:
            if ref_mol not in all_ids:
                raise ValueError(f"Reference molecule '{ref_mol}' not found")
            all_ids.remove(ref_mol)
            all_ids = [ref_mol] + sorted(all_ids)

        # Build mappings
        self.id_map = {uid: idx for idx, uid in enumerate(all_ids)}
        self.rev_map = {v: k for k, v in self.id_map.items()}

    def get_mapped_pairs(self) -> List[Tuple]:
        """Generate normalized pair data with indices"""
        logging.debug("Getting mapped pairs...")
        mapped_pairs = []
        for lig1, lig2, orig_ddg, error in self.raw_pairs:
            i = self.id_map[lig1]
            j = self.id_map[lig2]
            mapped_pairs.append((i, j, orig_ddg, error, orig_ddg))
        return mapped_pairs

    def print_mapping(self):
        """Display molecule ID mapping"""
        print("\n=== Molecule ID Mapping ===")
        for name, idx in self.id_map.items():
            print(f"{name} -> {idx}")

    def validate_data(self):
        """Perform data integrity checks"""
        logging.debug("Validating data...")
        print("\n=== Data Integrity Validation ===")
        seen_pairs = set()
        for lig1, lig2, *_ in self.raw_pairs:
            pair = tuple(sorted([lig1, lig2]))
            if pair in seen_pairs:
                print(f"Warning: Duplicate pair {lig1}-{lig2}")
            seen_pairs.add(pair)
        logging.info(f"Total molecules: {len(self.id_map)}")
        logging.info(f"Valid pairs: {len(self.get_mapped_pairs())}")
