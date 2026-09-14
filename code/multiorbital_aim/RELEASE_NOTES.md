# Release notes — v1.0.1

- Restored the original complete six-pole fermionic spectral decomposition.
- Set `Mf=6`; the actively read `expn1.dat` and `etal1.dat` each contain exactly
  six entries.

## v1.0.0

- Promoted the validated multi-orbital AIM implementation from development
  version v0.1.
- Removed unreachable Anderson-Holstein/bosonic construction and diagnostic
  code inherited from the mixed-bath ancestor.
- Preserved the TDVP/MPS numerical core.
- Included arbitrary orbital count, Hermitian hopping, `U`, `Uprime`, `J1`,
  spin-flip `J2`, and pair-hopping `J3`.
- Included orbital-aware mode metadata, mirrored ket/bra indexing, generalized
  fermion parity, resolved currents, and dense-Hamiltonian validation.
- Shipped the already benchmarked two-orbital/two-pole stress-test input.
