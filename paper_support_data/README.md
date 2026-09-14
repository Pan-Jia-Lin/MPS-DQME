# Data supporting the MPS-DQME manuscript

This directory contains the numerical data supplied for the figures in the
accompanying manuscript. The data are grouped by figure number so that the
published plots can be traced to the archived calculations.

## Directory map

| Directory | Manuscript figure and archived content |
| --- | --- |
| `Fig2_data/` | Anharmonic bosonic impurity: HEOM/MPS-DQME population comparison and bond-dimension convergence of `rho_00` and `rho_11` |
| `Fig3_data/` | Single-impurity Anderson model: four many-body-state populations and convergence of the doubly occupied population |
| `Fig4_data/` | Single-impurity Anderson model: bond-dimension convergence of the transient particle current and its long-time inset |
| `Fig5_data/` | Single-impurity Anderson model: steady-state current-voltage data and differential conductance; each `vol=.../` directory is one bias point |
| `Fig6_data/` | Anderson-Holstein model: transient-current convergence with respect to local vibrational Hilbert-space dimension `nb` |
| `Fig7_Fig8_data/` | Anderson-Holstein model: bond-dimension convergence of local vibrational populations (Fig. 7) and electronic populations (Fig. 8) |
| `Fig9_data/` | Anderson-Holstein model: bond-dimension convergence of the transient particle current |
| `Fig10_Fig11_data/` | Anderson-Holstein model: short/intermediate-time (Fig. 10) and long-time (Fig. 11) currents for bosonic bath couplings `lambda = 0, 0.01, 0.02, 0.05, 0.1` |

The directory map follows the figure captions in the manuscript dated
September 14, 2026.

## Common file conventions

| Pattern | Meaning |
| --- | --- |
| `L_up_curr*.dat` | Left-lead, spin-up current |
| `L_down_curr*.dat` | Left-lead, spin-down current |
| `R_up_curr*.dat` | Right-lead, spin-up current |
| `R_down_curr*.dat` | Right-lead, spin-down current |
| `diag_RDO.dat` | Diagonal populations of the reduced density operator |
| `dynamic_of_population.dat` | Time-dependent populations |
| `etal*.dat`, `expn*.dat` | Coefficients and exponents of the bath-correlation decomposition |
| `benchmark_*.npy` | NumPy binary array containing independent benchmark results |

Files with `_part1`, `_part2`, and similar suffixes are consecutive archived
segments of a calculation. Preserve their numerical ordering when combining
them. Current directions and basis ordering follow the conventions documented
in the corresponding implementation under `../code/`.

## Parameter-encoded directory names

- `cutoff=<value>/` records the cutoff/rank used for that calculation.
- `vol=<value>/` records the applied voltage used for that calculation.
- `lambda=<value>_current_data/` records the electron-phonon coupling used for
  the current dataset.

These names are retained from the original calculations to preserve the link
between the archived outputs and the executed parameter sets.

## Recommended citation and reuse

When using these data, cite the accompanying article and the frozen software
release. Do not infer physical convergence solely from the presence of a data
file; consult the manuscript and code READMEs for the convergence protocol.
