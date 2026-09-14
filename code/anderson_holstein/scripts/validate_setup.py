#!/usr/bin/env python3
"""Lightweight release validation without a TDVP propagation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import calc_rho  # noqa: E402
import construct  # noqa: E402
import params  # noqa: E402


def main() -> None:
    params.validate_parameters()
    construct.validate_term_structure()
    calc_rho.validate_calc_rho()
    number_of_terms = len(construct.build_product_terms())
    if number_of_terms != 330:
        raise AssertionError(
            f"expected 330 product-MPO terms, got {number_of_terms}"
        )
    print("MPS-DQME release validation passed")
    print(f"total MPS sites: {params.nsite_total}")
    print(f"dense RDO dimension: {params.ndvr_dense} x {params.ndvr_dense}")
    print(f"product-MPO terms: {number_of_terms}")
    print(f"maximum MPS rank: {params.nrmax}")
    print(f"data directory: {params.DATA_DIR}")


if __name__ == "__main__":
    main()
