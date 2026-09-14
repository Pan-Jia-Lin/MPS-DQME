# MPS-DQME for the Anderson-Holstein Impurity Model

Version 1.0 (research release)

This repository implements the dissipaton-embedded quantum master equation
(DQME) as a single matrix-product-state (MPS) evolution for a spinful,
single-orbital Anderson-Holstein impurity coupled to two fermionic reservoirs
and one bosonic environment. Time propagation uses a one-site
projector-splitting TDVP integrator.

The release is the cleaned version of the implementation benchmarked against
the independent Lang-Firsov-frame MPS-HEOM code `Full_V1.39`. It contains the
corrected fermionic parity mapping for lead currents and the corrected
asymmetric bosonic dissipaton ladder.

## Physical model

The production DQME is written before the Lang-Firsov transformation:

```text
H_sys = sum_sigma epsilon_sigma n_sigma
        + U n_up n_down
        + omega a^dagger a
        + lambda omega (a^dagger + a)(n_up + n_down).
```

The supplied benchmark parameters are

```text
epsilon_up = epsilon_down = -4.9
U = 10.2
omega = 1.0
lambda = sqrt(0.1)
left voltage = 0
right voltage = 1.39
dt = 0.025
maximum MPS rank = 240
```

The independent HEOM benchmark is formulated after the Lang-Firsov
transformation with `epsilon=-5` and `U=10`. Because that HEOM bosonic bath
couples to `Q`, the corresponding pre-transformation DQME bath operator is

```text
A_bath = Q + 2 lambda (n_up + n_down).
```

Using `Q` alone would define a different physical bath coupling and is not the
strict counterpart of the supplied HEOM benchmark.

## Corrected dissipaton conventions

### Fermionic first-tier parity

The physical first-tier RDOs used in the current are

```text
rho_m_physical = P @ rho_m_raw
rho_n_physical = rho_n_raw @ P
P = (-1)^(n_up+n_down).
```

This `m-left / n-right` parity dressing was selected by a direct Liouvillian
current test. With the bosonic bath disabled, all four DQME lead/spin currents
agree with the HEOM reference to approximately `1e-5`.

### Bosonic ladder

For `eta = eta_R + i eta_I` and `s = sqrt(abs(eta))`, the implemented
coefficients are

| Ladder action | Left action | Right action |
| --- | --- | --- |
| raising `b^dagger` | `eta_I/s - i eta_R/s` | `eta_I/s + i eta_R/s` |
| lowering `b` | `-i s` | `+i s` |

The lowering direction is therefore a pure commutator. This preserves the sign
of negative real Prony weights and places the imaginary anticommutator
contribution in the raising direction.

## Repository layout

```text
code/anderson_holstein/
├── README.md
├── RELEASE_NOTES.md
├── VERSION
├── requirements.txt
├── data/                  Prony coefficients and exponents
├── examples/run_rank240/  Empty recommended working directory
├── scripts/validate_setup.py
└── src/                   MPS-DQME implementation
```

Principal modules:

| Module | Purpose |
| --- | --- |
| `src/params.py` | Model, bath, site layout, cutoffs and propagation parameters |
| `src/construct.py` | Product-term Liouvillian and MPO construction |
| `src/init_rho.py` | Empty-dot/vacuum initial MPS |
| `src/calc_rho.py` | RDO reconstruction, observables and resolved currents |
| `src/ksltt.py` | Projector-splitting TDVP propagation |
| `src/prop.py` | Propagation loop and diagnostic output |
| `src/main.py` | Executable entry point |

The legacy correlation modules are retained for compatibility but the validated
release workflow has `run_correlation=False`.

## Requirements

- Python 3.8 or newer
- NumPy
- SciPy
- TensorNetwork only for the optional legacy correlation driver

Install the core dependencies with

```bash
python -m pip install -r requirements.txt
```

Ordinary propagation does not require TensorNetwork. Install it only if the
legacy correlation path is enabled.

## Validate the installation

From the release root, run

```bash
python scripts/validate_setup.py
```

This performs lightweight checks without assembling the compressed MPO or
starting a TDVP evolution:

- parameter and mirror-site consistency;
- finite Liouvillian coefficients and local operator dimensions;
- all 330 product-MPO terms of the `Q+2 lambda N` build;
- empty-dot/vacuum RDO, unit trace and zero initial current.

## Run

Outputs are written to the current working directory. Keeping them outside the
source tree is recommended:

```bash
cd examples/run_rank240
python ../../src/main.py > run.log 2>&1
```

For an unattended Linux run:

```bash
cd examples/run_rank240
nohup python ../../src/main.py > run.log 2>&1 &
echo $! > run.pid
```

The supplied `nsteps=6000` corresponds to a final time of 150 and can be very
expensive at rank 240. Use a short run for installation testing, for example by
temporarily setting `nsteps=4` in `src/params.py`.

To use Prony files stored elsewhere, set

```bash
export MPS_DQME_DATA_DIR=/absolute/path/to/prony/files
```

The directory must contain the four filenames specified in `src/params.py`.

## Main outputs

All complex-valued tables contain `time`, `real`, and `imaginary` columns.

| File | Quantity |
| --- | --- |
| `test_trace.dat` | RDO trace |
| `rdoout.dat` | Full dense system RDO in the basis `|n_up,n_down,n_phonon>` |
| `n1_ave.dat`, `n2_ave.dat` | Electron occupations `n_up`, `n_down` |
| `phonon_number_ave.dat` | System phonon occupation |
| `L_up_curr.dat`, etc. | Lead- and spin-resolved particle currents |
| `continuity.dat` | True centered-difference continuity diagnostic |
| `rhoall.dat` | Final MPS checkpoint when `Iswrite_rho=True` |

The centered continuity file covers interior points only: times `dt` through
`(nsteps-1)dt`. Its residual is

```text
[N(t+dt)-N(t-dt)]/(2 dt) - sum_leads,spins I(t).
```

## Current and occupation conventions

The DQME files use the particle-current direction implemented by the displayed
DQME current operator. The historical HEOM benchmark uses the opposite sign;
compare them using

```text
I_HEOM = -I_DQME.
```

The DQME `n1_ave` and `n2_ave` are electron occupations. In the historical HEOM
benchmark these filenames contain hole occupations, so convert with

```text
n_electron_HEOM = 1 - n_hole_HEOM / trace_HEOM.
```

For quantitative comparisons, normalize DQME observables by the instantaneous
DQME trace and HEOM observables by the HEOM trace.

## Validation record

The rank-240 reference run was propagated to `t=11`. After trace normalization
and current-sign alignment, the real-part RMSE against the independent HEOM
benchmark was

| Channel | RMSE over `0.025 <= t <= 11` |
| --- | ---: |
| left/up | `2.78e-4` |
| left/down | `2.29e-4` |
| right/up | `1.29e-4` |
| right/down | `1.13e-4` |

The mean four-channel RMSE for `t>=8` was `1.26e-4`. Electron-occupation RMSEs
were `7.75e-5` (up) and `3.12e-4` (down). The centered continuity RMS was about
`7e-4` after the initial transient.

At `t=11`, the unnormalized trace was

```text
0.9986141663 + 0.0030380414 i.
```

The full RDO still showed a Hermiticity defect of approximately `9.8e-3` at
that time. The transport observables are well benchmarked, but rank, local
bosonic cutoffs and time step remain numerical convergence parameters. Full-RDO
eigenvalues, coherences or entropies require separate convergence tests.

## Changing parameters safely

Edit the clearly marked sections in `src/params.py`, then rerun
`scripts/validate_setup.py`.

- `nrtt` and `nrmax`: MPS rank caps;
- `dt` and `nsteps`: time grid;
- `nboson`, `nb2`: system-phonon and bosonic-dissipaton cutoffs;
- `e1`, `e2`, `Uu`, `omega`, `lamb`: system parameters;
- `voltage`: lead/spin/orbital voltage table;
- `Mf`, `Mb` and the four input files: Prony decomposition.

Do not set `Mf=0` or `Mb=0` in this mixed-bath layout. The site maps assume
that both sectors exist. Likewise, zero bosonic weights require special care
because the scaled representation contains `eta/sqrt(abs(eta))`. A dedicated
pure-fermion release should remove the bosonic sites rather than retain them as
zero-coupling shells.

## Reproducibility and publication

Record the release version, source checksum, Prony files, `params.py`, Python
package versions and all convergence parameters for every published data set.

No legal license or author/citation metadata is imposed by this technical
release because those choices belong to the code owners. Before public posting,
add the intended `LICENSE` and citation information for the associated paper.

## Code provenance

This implementation combines MPS-DQME-specific scripts with retained or
adapted MPS/TDVP utilities from the HEOM+MPS implementation associated with
Qiang Shi and co-workers. The original author and reference headers are
preserved at the beginning of the relevant files. See the parent repository's
`CODE_PROVENANCE.md` for the exact file list and full attribution.
