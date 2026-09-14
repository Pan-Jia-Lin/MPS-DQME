# Release notes

## Version 1.0 — 2026-09-14

First cleaned research release of the split-system Anderson-Holstein MPS-DQME.

Included validated corrections:

- fermionic first-tier parity mapping `rho_m=P@rho_m_raw` and
  `rho_n=rho_n_raw@P`;
- lead-, spin- and orbital-resolved current grouping from explicit mode
  metadata;
- pre-Lang-Firsov bath operator `Q+2 lambda N` matching the independent
  Lang-Firsov-frame HEOM benchmark;
- asymmetric bosonic raising coefficients and pure-commutator lowering ladder;
- explicit split-system RDO reconstruction in the
  `|n_up,n_down,n_phonon>` basis;
- true centered-difference continuity diagnostic;
- lazy loading of the optional TensorNetwork dependency, so the validated
  propagation path requires only NumPy and SciPy;
- clean `src/`, `data/`, `scripts/` and `examples/` release layout.

Core Liouvillian and TDVP results were benchmarked at rank 240 through `t=11`.
See `README.md` for numerical accuracy and remaining convergence limitations.
