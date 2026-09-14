# MPS-DQME for the Multi-Orbital Anderson Impurity Model

Release: **v1.0.1** (2026-09-14)

This package implements the non-Markovian fermionic DQME as one MPS equation.
Every system spin-orbital and every fermionic dissipaton is an individual MPS
site. No Lindblad approximation, system phonon, bosonic bath, or zero-coupling
bosonic placeholder is present.

## Model

The supported Hamiltonian contains a Hermitian one-particle matrix, intra- and
inter-orbital density interactions, Hund density correction, spin flip, and
pair hopping. The precise formula and ordered-pair convention are frozen in
`CONVENTIONS.md`.

Chain order:

```text
system ket -- dissipaton ket(m) -- dissipaton bra(n, reversed)
           -- system bra(reversed)
```

Ket fermions use left-anchored Jordan-Wigner strings; bra fermions use
right-anchored strings. First-tier currents use
`rho_m_phys=P@rho_m_raw` and `rho_n_phys=rho_n_raw@P`.

## Quick start

Install the packages in `requirements.txt`, edit the clearly marked model and
propagation section of `params.py`, then run:

```bash
python validate_hsys.py
python calc_rho.py
python construct.py
python main.py
```

The shipped input is the two-orbital algebra-stress benchmark verified against
MPS-HEOM: complex inter-orbital hopping, nonzero `Uprime/J1/J2/J3`, the original
complete six-pole Prony decomposition, empty initial state, and rank 200. It is a code-validation input rather
than a claim of physical convergence.

## Outputs

- `output.dat`: diagonal RDO populations;
- `rdoout.dat`: complete dense system RDO;
- `n_m{orbital}_s{spin}.dat`: electron occupations;
- `I_a{lead}_m{orbital}_s{spin}.dat`: resolved currents;
- `test_trace.dat`: complex trace;
- `continuity.dat`: online diagnostic (use centered differences offline for
  quantitative validation);
- `L_total_curr.dat`, `R_total_curr.dat`, `total_curr.dat`: summed currents.

The legacy HEOM comparison uses the opposite current-direction convention:
`I_HEOM = -I_DQME`.

## Validation status

- one-orbital SIAM regression passed;
- two-orbital dense/product Hamiltonian equality: maximum error 0;
- vacuum RDO and zero-current checks passed;
- the earlier two-pole same-Prony two-orbital MPS-HEOM benchmark passed through the available
  interval: all-population RMSE `7.9e-5` for `t>=4`, all-current RMSE `1.7e-5`
  for `t>=4`, and final common RDO relative error `2.4e-4`; the restored
  six-pole input requires its own convergence run;
- full publication calculations still require rank, timestep, pole-number, and
  long-time convergence tests.

## Reproducibility

Keep the exact executed source and pole files with every dataset. Do not edit
`params.py` after a run and then archive it as though it generated that run.

## Code provenance

This implementation combines MPS-DQME-specific scripts with retained or
adapted MPS/TDVP utilities from the HEOM+MPS implementation associated with
Qiang Shi and co-workers. The original author and reference headers are
preserved at the beginning of the relevant files. See the parent repository's
`CODE_PROVENANCE.md` for the exact file list and full attribution.
